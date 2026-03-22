"""FastAPI server for audio transcription with flexible input formats."""

import asyncio
import base64
import io
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.constants import ALL_MODELS, CANARY_MAX_CHUNK_LENGTH
from config.settings import TranscriptionSettings

logger = logging.getLogger(__name__)

SR = 16000


# ---------------------------------------------------------------------------
# App state — injected by ServerManager before uvicorn starts
# ---------------------------------------------------------------------------

class AppState:
    model_manager: Any = None
    default_settings: Optional[TranscriptionSettings] = None
    transcription_active: bool = False
    queue: Optional[asyncio.Queue] = None
    worker_task: Optional[asyncio.Task] = None
    cancel_event: Event = Event()


_state = AppState()


def set_app_state(*, model_manager, default_settings: TranscriptionSettings):
    _state.model_manager = model_manager
    _state.default_settings = default_settings
    _state.cancel_event.clear()
    _state.transcription_active = False


# ---------------------------------------------------------------------------
# Work item for the queue
# ---------------------------------------------------------------------------

@dataclass
class WorkItem:
    audio: np.ndarray
    settings: TranscriptionSettings
    model_info: Dict[str, Any]
    future: asyncio.Future


# ---------------------------------------------------------------------------
# Audio normalisation helpers
# ---------------------------------------------------------------------------

def _resample(audio: np.ndarray, orig_sr: int, target_sr: int = SR) -> np.ndarray:
    """Resample audio to target sample rate using linear interpolation."""
    if orig_sr == target_sr:
        return audio
    ratio = target_sr / orig_sr
    n_samples = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, n_samples)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    """Ensure audio is mono float32 in [-1, 1]."""
    if audio.ndim > 1:
        if audio.shape[0] <= audio.shape[-1]:
            audio = audio.mean(axis=0)
        else:
            audio = audio.mean(axis=-1)
    audio = audio.flatten().astype(np.float32)
    if audio.max() > 1.0 or audio.min() < -1.0:
        max_val = max(abs(audio.max()), abs(audio.min()))
        if max_val > 0:
            if np.issubdtype(audio.dtype, np.integer) or max_val > 10:
                audio = audio / 32768.0
    return audio


def _load_audio_from_file(path: str) -> np.ndarray:
    """Load audio file using PyAV (same as batch_processor.load_audio)."""
    import av
    container = av.open(path)
    resampler = av.AudioResampler(format='s16', layout='mono', rate=SR)
    frames = []
    for frame in container.decode(audio=0):
        resampled = resampler.resample(frame)
        for r in resampled:
            frames.append(r.to_ndarray())
    container.close()
    return np.concatenate(frames, axis=1).flatten().astype(np.float32) / 32768.0


def _detect_format(filename: Optional[str], audio_format: str) -> str:
    """Return normalised format string."""
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
    """Convert any supported input to float32 mono numpy array at 16 kHz."""
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

    # Default: audio file — write to temp, load with PyAV
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


# ---------------------------------------------------------------------------
# Server-side transcription worker (plain class, not QThread)
# ---------------------------------------------------------------------------

class ServerTranscriptionWorker:
    """Mirrors BatchProcessor transcription methods without Qt dependencies."""

    def __init__(self, settings: TranscriptionSettings, model_info: Dict[str, Any],
                 cancel_event: Event):
        self.settings = settings
        self.model_info = model_info
        self.model_type = model_info.get("model_type", "parakeet")
        self.cancel_event = cancel_event

    def transcribe(self, model, audio: np.ndarray) -> Dict[str, Any]:
        """Run transcription and return result dict."""
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

    # -- chunking --

    def _get_chunks(self, audio: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        seg_len = self.settings.segment_length
        if self.model_type == "canary":
            seg_len = min(seg_len, CANARY_MAX_CHUNK_LENGTH)
        segment_samples = seg_len * SR
        audio_length = len(audio)
        if audio_length > segment_samples:
            chunks = []
            for i in range((audio_length + segment_samples - 1) // segment_samples):
                start = i * segment_samples
                end = min(start + segment_samples, audio_length)
                chunks.append((audio[start:end], start / SR))
            return chunks
        return [(audio, 0.0)]

    # -- Parakeet text-only --

    def _transcribe_text(self, model, audio: np.ndarray) -> str:
        chunks = self._get_chunks(audio)
        all_texts = []
        for chunk_audio, _ in chunks:
            if self.cancel_event.is_set():
                break
            with torch.inference_mode():
                output = model.transcribe(
                    [chunk_audio], batch_size=1, timestamps=False, verbose=False,
                )
            all_texts.append(self._extract_text(output))
        return " ".join(t for t in all_texts if t)

    # -- Canary --

    def _transcribe_canary(self, model, audio: np.ndarray) -> str:
        import soundfile as sf

        chunks = self._get_chunks(audio)
        all_texts = []
        temp_files = []

        try:
            for chunk_audio, _ in chunks:
                if self.cancel_event.is_set():
                    break
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_files.append(tmp.name)
                sf.write(tmp.name, chunk_audio, SR)
                tmp.close()

                chunk_duration = len(chunk_audio) / SR
                max_tokens = max(128, int(chunk_duration * 10))

                with torch.inference_mode():
                    answer_ids = model.generate(
                        prompts=[
                            [{"role": "user",
                              "content": f"Transcribe the following: {model.audio_locator_tag}",
                              "audio": [tmp.name]}]
                        ],
                        max_new_tokens=max_tokens,
                    )
                txt = model.tokenizer.ids_to_text(answer_ids[0].cpu()).strip()
                all_texts.append(txt)
        finally:
            for tf_path in temp_files:
                try:
                    os.remove(tf_path)
                except OSError:
                    pass

        return " ".join(t for t in all_texts if t)

    # -- Parakeet with timestamps --

    def _transcribe_with_timestamps(
        self, model, audio: np.ndarray
    ) -> List[Tuple[float, float, str]]:
        chunks = self._get_chunks(audio)
        all_segments = []
        for chunk_audio, time_offset in chunks:
            if self.cancel_event.is_set():
                break
            with torch.inference_mode():
                hypotheses = model.transcribe(
                    [chunk_audio], batch_size=1, timestamps=True,
                    return_hypotheses=True, verbose=False,
                )
            chunk_segs = self._extract_timestamp_segments(hypotheses, time_offset)
            if chunk_segs:
                all_segments.extend(chunk_segs)
            else:
                txt = self._extract_text(hypotheses)
                if txt:
                    chunk_end = time_offset + len(chunk_audio) / SR
                    all_segments.append((time_offset, chunk_end, txt))
        return all_segments

    # -- extraction helpers --

    @staticmethod
    def _extract_text(output) -> str:
        if not output:
            return ""
        if isinstance(output, (list, tuple)):
            first = output[0]
            if isinstance(first, str):
                return first.strip()
            if hasattr(first, "text"):
                return first.text.strip()
            return str(first).strip()
        return str(output).strip()

    def _extract_timestamp_segments(
        self, output, time_offset: float
    ) -> List[Tuple[float, float, str]]:
        segments = []
        if not output or not isinstance(output, (list, tuple)):
            return segments
        hyp = output[0]
        if not hasattr(hyp, "timestamp") or not hyp.timestamp:
            return segments
        ts = hyp.timestamp
        if not isinstance(ts, dict):
            return segments
        word_ts = ts.get("word", []) or ts.get("segment", [])
        for entry in word_ts:
            start = entry.get("start", 0) + time_offset
            end = entry.get("end", start) + time_offset
            text = (
                entry.get("word", "") or entry.get("char", "")
                or entry.get("segment", "")
            )
            if text.strip():
                segments.append((start, end, text.strip()))
        if segments:
            segments = self._group_words_into_segments(
                segments, max_duration=float(self.settings.segment_duration)
            )
        return segments

    @staticmethod
    def _group_words_into_segments(
        word_segments: List[Tuple[float, float, str]],
        max_duration: float = 10.0,
    ) -> List[Tuple[float, float, str]]:
        if not word_segments:
            return []
        grouped = []
        current_words = []
        current_start = word_segments[0][0]
        for start, end, word in word_segments:
            if current_words and (end - current_start) > max_duration:
                grouped.append((
                    current_start,
                    current_words[-1][1],
                    " ".join(w[2] for w in current_words),
                ))
                current_words = []
                current_start = start
            current_words.append((start, end, word))
        if current_words:
            grouped.append((
                current_start,
                current_words[-1][1],
                " ".join(w[2] for w in current_words),
            ))
        return grouped


# ---------------------------------------------------------------------------
# Queue worker
# ---------------------------------------------------------------------------

async def _queue_worker():
    """Process transcription requests one at a time."""
    while True:
        item: WorkItem = await _state.queue.get()
        _state.transcription_active = True
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _do_transcription, item)
            if not item.future.done():
                item.future.set_result(result)
        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            if not item.future.done():
                item.future.set_exception(e)
        finally:
            _state.transcription_active = False
            _state.queue.task_done()


def _do_transcription(item: WorkItem) -> Dict[str, Any]:
    """Blocking transcription — runs in thread pool."""
    start_time = time.perf_counter()

    model_key = item.settings.model_key
    model_info = item.model_info

    model = _state.model_manager.get_or_load_model(
        model_key, item.settings.device, model_info["precision"],
    )
    if model is None:
        raise RuntimeError("Failed to load model")

    worker = ServerTranscriptionWorker(
        item.settings, model_info, _state.cancel_event,
    )
    result = worker.transcribe(model, item.audio)

    elapsed = time.perf_counter() - start_time
    result["processing_time_seconds"] = round(elapsed, 3)
    result["model_used"] = model_key
    result["audio_duration_seconds"] = round(len(item.audio) / SR, 3)
    return result


# ---------------------------------------------------------------------------
# Resolve model key from user-supplied name + precision
# ---------------------------------------------------------------------------

def _resolve_model_key(
    model_name: Optional[str],
    precision: Optional[str],
    device: Optional[str],
    defaults: TranscriptionSettings,
) -> Tuple[str, Dict[str, Any]]:
    """Return (model_key, model_info) from user params or defaults."""
    target_name = model_name or ALL_MODELS[defaults.model_key]["name"]
    target_prec = precision if precision else ALL_MODELS[defaults.model_key]["precision"]

    key = f"{target_name} - {target_prec}"
    if key in ALL_MODELS:
        return key, ALL_MODELS[key]

    # Try to find a matching model by name alone
    for k, v in ALL_MODELS.items():
        if v["name"] == target_name:
            return k, v

    raise ValueError(
        f"Unknown model: '{target_name}' with precision '{target_prec}'. "
        f"Available: {list(ALL_MODELS.keys())}"
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
    """Merge user-supplied params with defaults, return (settings, model_info)."""
    defaults = _state.default_settings

    model_key, model_info = _resolve_model_key(model_name, precision, device, defaults)

    settings = TranscriptionSettings(
        model_key=model_key,
        device=device or defaults.device,
        segment_length=segment_length or defaults.segment_length,
        segment_duration=segment_duration or defaults.segment_duration,
        output_format=output_format or defaults.output_format,
        word_timestamps=word_timestamps if word_timestamps is not None else defaults.word_timestamps,
        recursive=False,
        selected_extensions=[],
    )
    return settings, model_info


# ---------------------------------------------------------------------------
# Pydantic model for JSON endpoint
# ---------------------------------------------------------------------------

class RawTranscribeRequest(BaseModel):
    audio_data: str  # base64-encoded
    audio_format: str = "numpy"  # numpy, tensor, pcm, wav, mp3, etc.
    sample_rate: int = 16000
    dtype: str = "float32"
    model: Optional[str] = None
    precision: Optional[str] = None
    device: Optional[str] = None
    output_format: Optional[str] = None
    word_timestamps: Optional[bool] = None
    segment_length: Optional[int] = None
    segment_duration: Optional[int] = None


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        _state.queue = asyncio.Queue()
        _state.cancel_event.clear()
        _state.worker_task = asyncio.create_task(_queue_worker())
        logger.info("Transcription queue worker started")
        yield
        # Shutdown
        if _state.worker_task:
            _state.worker_task.cancel()
            try:
                await _state.worker_task
            except asyncio.CancelledError:
                pass
        if _state.queue:
            while not _state.queue.empty():
                try:
                    item = _state.queue.get_nowait()
                    if not item.future.done():
                        item.future.set_exception(
                            RuntimeError("Server shutting down")
                        )
                except asyncio.QueueEmpty:
                    break
        logger.info("Transcription server shut down")

    app = FastAPI(
        title="Elegant Audio Transcriber API",
        description=(
            "Transcribe audio using NVIDIA NeMo models. Accepts audio files, "
            "numpy arrays, PyTorch tensors, raw PCM, and base64-encoded data."
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

    # -- routes --

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/models")
    async def models():
        result = {}
        for key, info in ALL_MODELS.items():
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
        """Transcribe audio from a multipart file upload.

        Accepts standard audio files (.mp3, .wav, .flac, etc.),
        numpy arrays (.npy), PyTorch tensors (.pt), or raw PCM data.
        """
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
        """Transcribe audio from base64-encoded data in a JSON body.

        Supports numpy arrays (np.save → base64), PyTorch tensors
        (torch.save → base64), raw PCM bytes, or full audio file bytes.
        """
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
