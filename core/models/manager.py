import gc
import io
import contextlib
import logging
from typing import Optional, Dict, Any
from PySide6.QtCore import QObject, Signal, QMutex

import torch

from config.constants import PARAKEET_MODELS
from download_model import find_local_model

logger = logging.getLogger(__name__)

TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

for _name in ["nemo", "nemo.collections", "nemo.utils", "nemo.core",
              "lightning", "lightning_fabric", "pytorch_lightning",
              "nemo_logger", "nv_one_logger", "wandb", "numba"]:
    logging.getLogger(_name).setLevel(logging.ERROR)


class ModelManager(QObject):
    model_loaded = Signal(str, str)
    model_error = Signal(str)

    def __init__(self):
        super().__init__()
        self._current_model = None
        self._current_config = None
        self._last_error = None
        self._model_mutex = QMutex()

    def get_or_load_model(self, model_key: str, device: str,
                         precision: str) -> Optional[Any]:
        config = {
            'model_key': model_key,
            'device': device,
            'precision': precision,
        }

        self._model_mutex.lock()
        try:
            if self._current_model is None or self._current_config != config:
                self._release_current_model()
                self._current_model = self._load_model(config)
                self._current_config = config
                self.model_loaded.emit(model_key, device)

            return self._current_model
        except Exception as e:
            logger.error(f"Model loading failed: {e}", exc_info=True)
            self._last_error = str(e)
            self.model_error.emit(str(e))
            return None
        finally:
            self._model_mutex.unlock()

    def _load_model(self, config: Dict[str, Any]) -> Any:
        model_info = PARAKEET_MODELS[config['model_key']]
        model_id = model_info['model_id']
        device = config['device']
        precision = config['precision']

        logger.info(f"Loading model: {model_id} on {device} with {precision}")

        if device == "cpu" and precision in ("float16", "bfloat16"):
            torch_dtype = torch.float32
        else:
            torch_dtype = TORCH_DTYPE_MAP.get(precision, torch.float32)

        logger.info("Importing NeMo ASR...")
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            import nemo.collections.asr as nemo_asr

        local_path = find_local_model(model_id)
        stderr_capture = io.StringIO()

        if local_path:
            logger.info(f"Loading model from local path: {local_path}")
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(stderr_capture):
                model = nemo_asr.models.ASRModel.restore_from(
                    restore_path=local_path, map_location="cpu"
                )
        else:
            logger.info(f"Loading pretrained model from Hub: {model_id}")
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(stderr_capture):
                model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=model_id, map_location="cpu"
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

        logger.info(f"Model loaded successfully on {device}")
        return model

    def _release_current_model(self) -> None:
        if self._current_model is not None:
            del self._current_model
            self._current_model = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def cleanup(self) -> None:
        self._model_mutex.lock()
        try:
            self._release_current_model()
        finally:
            self._model_mutex.unlock()
