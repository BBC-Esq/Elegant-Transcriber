from __future__ import annotations

import contextlib
import gc
import io
import threading
import uuid
from pathlib import Path
from typing import Optional

import torch
from PySide6.QtCore import QObject, QMutex, QMutexLocker, QThread, Signal


class _NullWriter:
    def write(self, *a, **kw):
        pass

    def flush(self, *a, **kw):
        pass

from core.logging_config import get_logger
from core.exceptions import ModelLoadError
from core.models.metadata import ModelMetadata
from download_model import find_local_model, get_local_model_path, MODELS

logger = get_logger(__name__)


TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


_NETWORK_ERROR_TERMS = [
    "connection", "network", "resolve", "urlerror", "timeout",
    "unreachable", "dns", "socket", "offline",
]


def _is_network_error(exception: Exception) -> bool:
    msg = str(exception).lower()
    return any(term in msg for term in _NETWORK_ERROR_TERMS)


def _resolve_torch_dtype(device: str, precision: str):
    if device == "cpu" and precision in ("float16", "bfloat16"):
        return torch.float32
    return TORCH_DTYPE_MAP.get(precision, torch.float32)


def _get_remote_nemo_size(model_id: str) -> int:
    filename = MODELS.get(model_id)
    if not filename:
        return 0
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.repo_info(model_id, repo_type="model", files_metadata=True)
        for sibling in info.siblings:
            if sibling.rfilename == filename:
                return int(sibling.size or 0)
    except Exception as e:
        logger.debug(f"Could not get remote size for {model_id}: {e}")
    return 0


def _load_parakeet_model(local_path: str, device: str, torch_dtype) -> object:
    # Targeted import: pulling in the full nemo.collections.asr namespace
    # transitively loads ~19 unrelated ASR model classes (CTC, hybrid, K2,
    # diarization, SSL, multitalker, etc.) and adds ~16s to startup.
    # EncDecRNNTBPEModel is the concrete class for Parakeet TDT v2 and v3.
    logger.info("Importing NeMo Parakeet model class (EncDecRNNTBPEModel)...")
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel

    stderr_capture = io.StringIO()

    logger.info(f"Loading Parakeet from local path: {local_path}")
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(stderr_capture):
        model = EncDecRNNTBPEModel.restore_from(
            restore_path=local_path, map_location="cpu"
        )

    captured = stderr_capture.getvalue()
    if captured:
        logger.info(f"Model load stderr: {captured[:500]}")

    model = model.to(device)
    if device == "cuda" and torch_dtype != torch.float32:
        model = model.to(dtype=torch_dtype)
    model.eval()

    _noop = lambda *a, **kw: None
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'unfreeze'):
        model.encoder.unfreeze = _noop
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'unfreeze'):
        model.decoder.unfreeze = _noop
    if hasattr(model, 'joint') and hasattr(model.joint, 'unfreeze'):
        model.joint.unfreeze = _noop

    logger.info(f"Parakeet model loaded successfully on {device}")
    return model


def _load_canary_model(local_path: str, device: str, torch_dtype) -> object:
    logger.info("Importing NeMo SALM for Canary-Qwen...")
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        from nemo.collections.speechlm2.models import SALM

    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3Model
        if not hasattr(Qwen3ForCausalLM, '_patched_get_input_embeddings'):
            Qwen3ForCausalLM._patched_get_input_embeddings = True
            Qwen3ForCausalLM.get_input_embeddings = lambda self: getattr(self.model, 'embed_tokens', None)
            Qwen3Model.get_input_embeddings = lambda self: getattr(self, 'embed_tokens', None)
            logger.info("Applied Qwen3 get_input_embeddings monkey-patch")
    except (ImportError, AttributeError):
        pass

    stderr_capture = io.StringIO()
    logger.info(f"Loading SALM from: {local_path}")
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(stderr_capture):
        model = SALM.from_pretrained(local_path)

    captured = stderr_capture.getvalue()
    if captured:
        logger.info(f"Model load stderr: {captured[:500]}")

    model = model.to(device)
    if device == "cuda" and torch_dtype != torch.float32:
        model = model.to(dtype=torch_dtype)
    model.eval()

    logger.info(f"Canary-Qwen model loaded successfully on {device}")
    return model


def _download_model_sync(model_id: str) -> str:
    from huggingface_hub import hf_hub_download, snapshot_download

    filename = MODELS.get(model_id)
    local_path = get_local_model_path(model_id)
    if local_path is None:
        raise ModelLoadError(f"Unknown model_id '{model_id}'")

    if filename:
        local_dir = local_path.parent
        local_dir.mkdir(parents=True, exist_ok=True)
        hf_hub_download(repo_id=model_id, filename=filename, local_dir=str(local_dir))
        if not local_path.is_file():
            raise ModelLoadError(f"Download completed but file not found at {local_path}")
        return str(local_path)

    local_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=model_id, local_dir=str(local_path))
    if not (local_path / "config.json").is_file():
        raise ModelLoadError(f"Download completed but config.json not found at {local_path}")
    return str(local_path)


class _ModelLoaderThread(QThread):
    model_loaded = Signal(object, str, str, str, str)
    error_occurred = Signal(str, str)
    download_started = Signal(str, object, str)
    download_progress = Signal(object, object, str)
    download_finished = Signal(str, str)
    download_cancelled = Signal(str)
    loading_started = Signal(str, str)

    def __init__(
        self,
        model_name: str,
        precision: str,
        device: str,
        model_version: str,
        cancel_event: threading.Event,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.precision = precision
        self.device = device
        self.model_version = model_version
        self.cancel_event = cancel_event

    @property
    def signals(self):
        return self

    def _cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def run(self) -> None:
        try:
            model_id = ModelMetadata.get_model_id(self.model_name)
            if not model_id:
                self.signals.error_occurred.emit(
                    f"Unknown model '{self.model_name}'", self.model_version
                )
                return

            model_type = ModelMetadata.get_model_type(self.model_name)

            local_path = find_local_model(model_id)

            if local_path is None:
                try:
                    total_bytes = _get_remote_nemo_size(model_id) if MODELS.get(model_id) else 0
                except Exception:
                    total_bytes = 0

                self.signals.download_started.emit(
                    self.model_name, total_bytes, self.model_version
                )
                try:
                    local_path = self._download(model_id)
                except InterruptedError:
                    self.signals.download_cancelled.emit(self.model_version)
                    return
                except Exception as e:
                    if _is_network_error(e):
                        self.signals.error_occurred.emit(
                            f"Download failed for '{self.model_name}': "
                            f"Network connection lost. Please check your internet "
                            f"connection and try again.",
                            self.model_version,
                        )
                    else:
                        self.signals.error_occurred.emit(
                            f"Download failed for '{self.model_name}': {e}",
                            self.model_version,
                        )
                    return

                self.signals.download_finished.emit(self.model_name, self.model_version)

            if self._cancelled():
                self.signals.download_cancelled.emit(self.model_version)
                return

            self.signals.loading_started.emit(self.model_name, self.model_version)

            torch_dtype = _resolve_torch_dtype(self.device, self.precision)
            if model_type == "canary":
                model = self._load_canary(local_path, self.device, torch_dtype)
            else:
                model = self._load_parakeet(local_path, self.device, torch_dtype)

            if self._cancelled():
                self.signals.download_cancelled.emit(self.model_version)
                return

            self.signals.model_loaded.emit(
                model, self.model_name, self.precision, self.device, self.model_version
            )

        except ModelLoadError as e:
            logger.error(f"Model load error: {e}")
            self.signals.error_occurred.emit(str(e), self.model_version)
        except Exception as e:
            if self._cancelled():
                self.signals.download_cancelled.emit(self.model_version)
            else:
                logger.exception("Unexpected error loading model")
                self.signals.error_occurred.emit(
                    f"Unexpected error: {e}", self.model_version
                )

    def _download(self, model_id: str) -> str:
        from huggingface_hub import hf_hub_download, snapshot_download
        from tqdm.auto import tqdm

        filename = MODELS[model_id]
        local_path = get_local_model_path(model_id)

        total_bytes = 0
        if filename:
            try:
                total_bytes = _get_remote_nemo_size(model_id)
            except Exception:
                pass

        progress_signal = self.signals.download_progress
        version = self.model_version
        cancel_event = self.cancel_event

        class _ProgressTqdm(tqdm):
            def __init__(self, *args, **kwargs):
                kwargs.setdefault("file", _NullWriter())
                super().__init__(*args, **kwargs)

            def update(self, n=1):
                if cancel_event.is_set():
                    raise InterruptedError("Download cancelled")
                super().update(n)
                total = self.total or total_bytes or 0
                progress_signal.emit(int(self.n), int(total), version)

        if filename:
            local_dir = local_path.parent
            local_dir.mkdir(parents=True, exist_ok=True)
            try:
                hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    local_dir=str(local_dir),
                    tqdm_class=_ProgressTqdm,
                )
            except InterruptedError:
                raise
            except TypeError:
                hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    local_dir=str(local_dir),
                )
                if total_bytes:
                    progress_signal.emit(total_bytes, total_bytes, version)

            if not local_path.is_file():
                raise ModelLoadError(
                    f"Download completed but file not found at {local_path}"
                )
            return str(local_path)

        local_path.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=str(local_path),
                tqdm_class=_ProgressTqdm,
            )
        except InterruptedError:
            raise
        except TypeError:
            snapshot_download(
                repo_id=model_id,
                local_dir=str(local_path),
            )

        if not (local_path / "config.json").is_file():
            raise ModelLoadError(
                f"Download completed but config.json not found at {local_path}"
            )
        return str(local_path)

    def _load_parakeet(self, local_path: str, device: str, torch_dtype) -> object:
        return _load_parakeet_model(local_path, device, torch_dtype)

    def _load_canary(self, local_path: str, device: str, torch_dtype) -> object:
        return _load_canary_model(local_path, device, torch_dtype)


def _unload_model(model) -> None:
    try:
        del model
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    gc.collect()


class ModelManager(QObject):
    model_loaded = Signal(str, str, str)
    model_error = Signal(str)
    download_started = Signal(str, object)
    download_progress = Signal(object, object)
    download_finished = Signal(str)
    download_cancelled = Signal()
    loading_started = Signal(str)

    def __init__(self):
        super().__init__()
        self._model = None
        self._model_version: Optional[str] = None
        self._pending_version: Optional[str] = None
        self._model_mutex = QMutex()
        self._current_settings: dict = {}
        self._cancel_event: Optional[threading.Event] = None
        self._active_thread: Optional[_ModelLoaderThread] = None

    def load_model(self, model_name: str, precision: str, device: str) -> None:
        logger.info(f"Requesting model load: {model_name}, {precision}, {device}")

        if self._cancel_event:
            self._cancel_event.set()

        if self._active_thread and self._active_thread.isRunning():
            self._active_thread.wait(100)

        new_version = str(uuid.uuid4())
        self._pending_version = new_version
        self._cancel_event = threading.Event()

        thread = _ModelLoaderThread(
            model_name, precision, device, new_version, self._cancel_event
        )
        thread.model_loaded.connect(self._on_model_loaded)
        thread.error_occurred.connect(self._on_model_error)
        thread.download_started.connect(self._on_download_started)
        thread.download_progress.connect(self._on_download_progress)
        thread.download_finished.connect(self._on_download_finished)
        thread.download_cancelled.connect(self._on_download_cancelled)
        thread.loading_started.connect(self._on_loading_started)
        self._active_thread = thread
        thread.start()

    def cancel_loading(self) -> None:
        if self._cancel_event:
            self._cancel_event.set()

    def get_model(self) -> tuple[Optional[object], Optional[str]]:
        with QMutexLocker(self._model_mutex):
            return self._model, self._model_version

    def get_or_load_model(self, model_key: str, device: str, precision: str) -> Optional[object]:
        model_name = model_key.split(" - ")[0] if " - " in model_key else model_key
        model_type = ModelMetadata.get_model_type(model_name)
        model_id = ModelMetadata.get_model_id(model_name)
        if not model_id:
            logger.error(f"Unknown model '{model_name}'")
            return None

        config_key = (model_name, device, precision)

        with QMutexLocker(self._model_mutex):
            if self._model is not None and self._current_settings.get("_config_key") == config_key:
                return self._model

        try:
            local_path = find_local_model(model_id)
            if local_path is None:
                logger.info(f"Model not cached; downloading {model_id} synchronously...")
                local_path = _download_model_sync(model_id)

            torch_dtype = _resolve_torch_dtype(device, precision)
            if model_type == "canary":
                new_model = _load_canary_model(local_path, device, torch_dtype)
            else:
                new_model = _load_parakeet_model(local_path, device, torch_dtype)

            with QMutexLocker(self._model_mutex):
                if self._model is not None and self._current_settings.get("_config_key") != config_key:
                    _unload_model(self._model)
                self._model = new_model
                self._model_version = str(uuid.uuid4())
                self._current_settings = {
                    "model_name": model_name,
                    "precision": precision,
                    "device_type": device,
                    "_config_key": config_key,
                }

            self.model_loaded.emit(model_name, precision, device)
            return new_model

        except Exception as e:
            logger.error(f"get_or_load_model failed: {e}", exc_info=True)
            self.model_error.emit(str(e))
            return None

    def _on_download_started(self, model_name: str, total_bytes: int, version: str) -> None:
        if version == self._pending_version:
            self.download_started.emit(model_name, total_bytes)

    def _on_download_progress(self, downloaded: int, total: int, version: str) -> None:
        if version == self._pending_version:
            self.download_progress.emit(downloaded, total)

    def _on_download_finished(self, model_name: str, version: str) -> None:
        if version == self._pending_version:
            self.download_finished.emit(model_name)

    def _on_download_cancelled(self, version: str) -> None:
        if version == self._pending_version:
            self.download_cancelled.emit()

    def _on_loading_started(self, model_name: str, version: str) -> None:
        if version == self._pending_version:
            self.loading_started.emit(model_name)

    def _on_model_loaded(self, model, name: str, precision: str, device: str, version: str) -> None:
        if version != self._pending_version:
            logger.info(f"Ignoring stale model load (version {version})")
            _unload_model(model)
            return

        with QMutexLocker(self._model_mutex):
            if self._model is not None:
                _unload_model(self._model)
            self._model = model
            self._model_version = version

        self._current_settings = {
            "model_name": name,
            "precision": precision,
            "device_type": device,
            "_config_key": (name, device, precision),
        }
        logger.info(f"Model loaded successfully: {name}")
        self.model_loaded.emit(name, precision, device)

    def _on_model_error(self, error: str, version: str) -> None:
        if version == self._pending_version:
            logger.error(f"Model error: {error}")
            self.model_error.emit(error)

    def cleanup(self) -> None:
        if self._cancel_event:
            self._cancel_event.set()

        if self._active_thread and self._active_thread.isRunning():
            self._active_thread.wait(5000)

        with QMutexLocker(self._model_mutex):
            if self._model is not None:
                _unload_model(self._model)
                self._model = None
                self._model_version = None

        logger.debug("ModelManager cleanup complete")
