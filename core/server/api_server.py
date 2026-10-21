from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.server_settings import TranscriptionSettings
from core.models.metadata import ModelMetadata
from core.transcription.parakeet_core import (
    CANARY_OVERLAP_SECONDS,
    transcribe_canary,
    transcribe_text,
    transcribe_with_timestamps,
)

logger = logging.getLogger(__name__)

SR = 16000
CANARY_MAX_CHUNK_LENGTH = 40

# Valid segment_length range, matching the GUI/config validator. Client-supplied
# values arrive over HTTP without the config-layer clamp, so the server must
# enforce these bounds itself. A segment_length <= CANARY_OVERLAP_SECONDS makes
# Canary's chunking step zero or negative, which get_chunks rejects.
MIN_SEGMENT_LENGTH = 10
MAX_SEGMENT_LENGTH = 100

# Matches the GUI spinbox and the config validator. An unclamped negative
# value reaches group_words_into_segments as max_duration and degenerates
# the output into one segment per word.
MIN_SEGMENT_DURATION = 1
MAX_SEGMENT_DURATION = 90


class AppState:
    model_manager: Any = None
    default_settings: Optional[TranscriptionSettings] = None
    transcription_active: bool = False
    queue: Optional[asyncio.Queue] = None
    worker_task: Optional[asyncio.Task] = None
    cancel_event: Event = Event()


_state = AppState()


def set_app_state(*, model_manager, default_settings: TranscriptionSettings) -> None:
    _state.model_manager = model_manager
    _state.default_settings = default_settings
    _state.cancel_event = Event()
    _state.transcription_active = False


@dataclass
class WorkItem:
    audio: np.ndarray
    settings: TranscriptionSettings
    model_info: Dict[str, Any]
    future: asyncio.Future


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int = SR) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    ratio = target_sr / orig_sr
    n_samples = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, n_samples)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    if audio.size == 0:
        return audio.flatten().astype(np.float32)
    orig_dtype = audio.dtype
    if audio.ndim > 1:
        if audio.shape[0] <= audio.shape[-1]:
            audio = audio.mean(axis=0)
        else:
            audio = audio.mean(axis=-1)
    audio = audio.flatten().astype(np.float32)
    if audio.size == 0:
        return audio

    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        return audio / float(max(abs(int(info.min)), int(info.max) + 1))

    # Scale by the true peak rather than an assumed int16 range, but only
    # for signals that actually exceed full scale: quiet float audio that
    # is already in [-1, 1] must not be amplified.
    max_val = max(abs(float(audio.max())), abs(float(audio.min())))
    if max_val > 1.0:
        audio = audio / max_val
    return audio


def _load_audio_from_file(path: str) -> np.ndarray:
    import av
    container = av.open(path)
    try:
        resampler = av.AudioResampler(format='s16', layout='mono', rate=SR)
        frames = []
        for frame in container.decode(audio=0):
            resampled = resampler.resample(frame)
            for r in resampled:
                frames.append(r.to_ndarray())
        for r in resampler.resample(None):
            frames.append(r.to_ndarray())
    finally:
        container.close()
    if not frames:
        raise ValueError("No audio frames decoded - file may be empty or corrupted")
    return np.concatenate(frames, axis=1).flatten().astype(np.float32) / 32768.0


def _detect_format(filename: Optional[str], audio_format: str) -> str:
    if audio_format and audio_format != "auto":
        return audio_format

    if filename:
        ext = Path(filename).suffix.lower()
        if ext == ".npy":
            return "numpy"
        if ext == ".pt":
            return "tensor"
        audio_exts = {
            ".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac",
            ".wma", ".webm", ".mp4", ".mkv", ".avi", ".asf", ".amr",
        }
        if ext in audio_exts:
            return "file"

    return "file"


def normalize_audio(
    data: bytes,
    filename: Optional[str] = None,
    audio_format: str = "auto",
    sample_rate: int = SR,
    dtype: str = "float32",
) -> np.ndarray:
    fmt = _detect_format(filename, audio_format)

    if fmt == "numpy":
        buf = io.BytesIO(data)
        audio = np.load(buf, allow_pickle=False)
        audio = _to_mono_float32(audio)
        return _resample(audio, sample_rate, SR)

    if fmt == "tensor":
        buf = io.BytesIO(data)
        tensor = torch.load(buf, map_location="cpu", weights_only=True)
        audio = tensor.numpy()
        audio = _to_mono_float32(audio)
        return _resample(audio, sample_rate, SR)

    if fmt == "pcm":
        np_dtype = {
            "float32": np.float32, "float64": np.float64,
            "int16": np.int16, "int32": np.int32,
        }.get(dtype, np.float32)
        audio = np.frombuffer(data, dtype=np_dtype)
        audio = _to_mono_float32(audio)
        return _resample(audio, sample_rate, SR)

    suffix = Path(filename).suffix if filename else ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(data)
        tmp.close()
        return _load_audio_from_file(tmp.name)
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass


class ServerTranscriptionWorker:

    def __init__(self, settings: TranscriptionSettings, model_info: Dict[str, Any],
                 cancel_event: Event):
        self.settings = settings
        self.model_info = model_info
        self.model_type = model_info.get("model_type", "parakeet")
        self.cancel_event = cancel_event

    def transcribe(self, model, audio: np.ndarray) -> Dict[str, Any]:
        use_timestamps = (
            self.settings.word_timestamps and self.model_type != "canary"
        )

        if self.model_type == "canary":
            text = self._transcribe_canary(model, audio)
            return {"text": text, "segments": []}
        elif use_timestamps:
            segments = self._transcribe_with_timestamps(model, audio)
            text = " ".join(s[2] for s in segments)
            return {
                "text": text,
                "segments": [
                    {"start": round(s, 3), "end": round(e, 3), "text": t}
                    for s, e, t in segments
                ],
            }
        else:
            text = self._transcribe_text(model, audio)
            return {"text": text, "segments": []}

    def _canary_segment_length(self) -> int:
        seg_len = min(self.settings.segment_length, CANARY_MAX_CHUNK_LENGTH)
        # Defensive floor: keep the chunk step (segment_samples -
        # overlap_samples) positive even if settings were built without
        # _build_settings' clamp.
        return max(seg_len, CANARY_OVERLAP_SECONDS + 1)

    def _transcribe_text(self, model, audio: np.ndarray) -> str:
        return transcribe_text(
            model, audio, self.settings.segment_length,
            cancel_check=self.cancel_event.is_set,
        )

    def _transcribe_canary(self, model, audio: np.ndarray) -> str:
        return transcribe_canary(
            model, audio, self._canary_segment_length(),
            cancel_check=self.cancel_event.is_set,
        )

    def _transcribe_with_timestamps(
        self, model, audio: np.ndarray
    ) -> List[Tuple[float, float, str]]:
        return transcribe_with_timestamps(
            model, audio, self.settings.segment_length, self.settings.segment_duration,
            cancel_check=self.cancel_event.is_set,
        )


async def _queue_worker(queue: asyncio.Queue, cancel_event: Event):
    while True:
        item: WorkItem = await queue.get()
        _state.transcription_active = True
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _do_transcription, item, cancel_event)
            if not item.future.done():
                item.future.set_result(result)
        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            if not item.future.done():
                item.future.set_exception(e)
        finally:
            _state.transcription_active = False
            queue.task_done()


def _do_transcription(item: WorkItem, cancel_event: Event) -> Dict[str, Any]:
    if cancel_event.is_set():
        raise RuntimeError("Server shutting down")

    start_time = time.perf_counter()

    model_key = item.settings.model_key
    model_info = item.model_info

    model = _state.model_manager.get_or_load_model(
        model_key, item.settings.device, model_info["precision"],
        cancel_event=cancel_event,
    )
    if model is None:
        raise RuntimeError("Failed to load model")

    worker = ServerTranscriptionWorker(
        item.settings, model_info, cancel_event,
    )
    result = worker.transcribe(model, item.audio)

    if cancel_event.is_set():
        raise RuntimeError("Server shutting down")

    elapsed = time.perf_counter() - start_time
    result["processing_time_seconds"] = round(elapsed, 3)
    result["model_used"] = model_key
    result["audio_duration_seconds"] = round(len(item.audio) / SR, 3)
    return result


def _resolve_model_key(
    model_name: Optional[str],
    precision: Optional[str],
    device: Optional[str],
    defaults: TranscriptionSettings,
) -> Tuple[str, Dict[str, Any]]:
    registry = ModelMetadata.get_all_models_with_precisions()
    default_info = registry.get(defaults.model_key, {})
    target_name = model_name or default_info.get("name", "")
    target_prec = precision if precision else default_info.get("precision", "")

    key = f"{target_name} - {target_prec}"
    if key in registry:
        return key, registry[key]

    for k, v in registry.items():
        if v["name"] == target_name:
            return k, v

    raise ValueError(
        f"Unknown model: '{target_name}' with precision '{target_prec}'. "
        f"Available: {list(registry.keys())}"
    )


def _build_settings(
    model_name: Optional[str],
    precision: Optional[str],
    device: Optional[str],
    output_format: Optional[str],
    word_timestamps: Optional[bool],
    segment_length: Optional[int],
    segment_duration: Optional[int],
) -> Tuple[TranscriptionSettings, Dict[str, Any]]:
    defaults = _state.default_settings

    model_key, model_info = _resolve_model_key(model_name, precision, device, defaults)

    seg_len = segment_length or defaults.segment_length
    seg_len = max(MIN_SEGMENT_LENGTH, min(int(seg_len), MAX_SEGMENT_LENGTH))

    seg_dur = segment_duration or defaults.segment_duration
    seg_dur = max(MIN_SEGMENT_DURATION, min(int(seg_dur), MAX_SEGMENT_DURATION))

    settings = TranscriptionSettings(
        model_key=model_key,
        device=device or defaults.device,
        segment_length=seg_len,
        segment_duration=seg_dur,
        output_format=output_format or defaults.output_format,
        word_timestamps=word_timestamps if word_timestamps is not None else defaults.word_timestamps,
        recursive=False,
        selected_extensions=[],
    )
    return settings, model_info


class RawTranscribeRequest(BaseModel):
    audio_data: str
    audio_format: str = "numpy"
    sample_rate: int = 16000
    dtype: str = "float32"
    model: Optional[str] = None
    precision: Optional[str] = None
    device: Optional[str] = None
    output_format: Optional[str] = None
    word_timestamps: Optional[bool] = None
    segment_length: Optional[int] = None
    segment_duration: Optional[int] = None


def create_app() -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        queue = asyncio.Queue()
        cancel_event = _state.cancel_event
        _state.queue = queue
        worker_task = asyncio.create_task(_queue_worker(queue, cancel_event))
        _state.worker_task = worker_task
        logger.info("Transcription queue worker started")
        yield
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
        while not queue.empty():
            try:
                item = queue.get_nowait()
                if not item.future.done():
                    item.future.set_exception(
                        RuntimeError("Server shutting down")
                    )
            except asyncio.QueueEmpty:
                break
        logger.info("Transcription server shut down")

    app = FastAPI(
        title="Elegant Transcriber API",
        description=(
            "Transcribe audio using NVIDIA NeMo models (Parakeet v2/v3, Canary-Qwen). "
            "Accepts audio files, numpy arrays, PyTorch tensors, raw PCM, "
            "and base64-encoded data."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/models")
    async def models():
        registry = ModelMetadata.get_all_models_with_precisions()
        result = {}
        for key, info in registry.items():
            result[key] = {
                "name": info["name"],
                "model_id": info["model_id"],
                "precision": info["precision"],
                "model_type": info["model_type"],
                "avg_vram_usage": info["avg_vram_usage"],
                "default_segment_length": info["default_segment_length"],
                "supports_timestamps": info["model_type"] != "canary",
            }
        return result

    @app.get("/status")
    async def status():
        return {
            "server_running": True,
            "queue_depth": _state.queue.qsize() if _state.queue else 0,
            "transcription_active": _state.transcription_active,
        }

    @app.post("/transcribe")
    async def transcribe(
        audio: UploadFile = File(...),
        audio_format: Optional[str] = Form("auto"),
        sample_rate: Optional[int] = Form(SR),
        dtype: Optional[str] = Form("float32"),
        model: Optional[str] = Form(None),
        precision: Optional[str] = Form(None),
        device: Optional[str] = Form(None),
        output_format: Optional[str] = Form(None),
        word_timestamps: Optional[bool] = Form(None),
        segment_length: Optional[int] = Form(None),
        segment_duration: Optional[int] = Form(None),
    ):
        try:
            data = await audio.read()
            if not data:
                raise HTTPException(status_code=400, detail="Empty audio data")

            audio_array = normalize_audio(
                data,
                filename=audio.filename,
                audio_format=audio_format,
                sample_rate=sample_rate,
                dtype=dtype,
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process audio: {e}",
            )

        try:
            settings, model_info = _build_settings(
                model, precision, device, output_format,
                word_timestamps, segment_length, segment_duration,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        loop = asyncio.get_event_loop()
        future = loop.create_future()

        item = WorkItem(
            audio=audio_array,
            settings=settings,
            model_info=model_info,
            future=future,
        )
        await _state.queue.put(item)

        try:
            result = await future
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

        return result

    @app.post("/transcribe/raw")
    async def transcribe_raw(request: RawTranscribeRequest):
        try:
            data = base64.b64decode(request.audio_data)
            if not data:
                raise HTTPException(status_code=400, detail="Empty audio data")

            audio_array = normalize_audio(
                data,
                filename=None,
                audio_format=request.audio_format,
                sample_rate=request.sample_rate,
                dtype=request.dtype,
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to decode/process audio: {e}",
            )

        try:
            settings, model_info = _build_settings(
                request.model, request.precision, request.device,
                request.output_format, request.word_timestamps,
                request.segment_length, request.segment_duration,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        loop = asyncio.get_event_loop()
        future = loop.create_future()

        item = WorkItem(
            audio=audio_array,
            settings=settings,
            model_info=model_info,
            future=future,
        )
        await _state.queue.put(item)

        try:
            result = await future
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

        return result

    return app
