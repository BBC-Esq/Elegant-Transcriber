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

from config.constants import ALL_MODELS, CANARY_MAX_CHUNK_LENGTH, CHUNK_OVERLAP_SECONDS
from core.transcription.stitching import stitch_texts, stitch_timestamp_segments
from config.settings import TranscriptionSettings

logger = logging.getLogger(__name__)

SR = 16000



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
    import av
    container = av.open(path)
    try:
        resampler = av.AudioResampler(format='s16', layout='mono', rate=SR)
        frames = []
        for frame in container.decode(audio=0):
            resampled = resampler.resample(frame)
            for r in resampled:
                frames.append(r.to_ndarray())
    finally:
        container.close()
    if not frames:
        raise ValueError("No audio frames decoded — file may be empty or corrupted")
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

    def _get_chunks(self, audio: np.ndarray) -> List[Tuple[np.ndarray, float, bool]]:
        seg_len = self.settings.segment_length
        if self.model_type == "canary":
            seg_len = min(seg_len, CANARY_MAX_CHUNK_LENGTH)
        segment_samples = seg_len * SR
        overlap_samples = CHUNK_OVERLAP_SECONDS * SR
        audio_length = len(audio)

        if audio_length <= segment_samples:
            return [(audio, 0.0, False)]

        chunks = []
        start = 0
        while start < audio_length:
            end = min(start + segment_samples, audio_length)
            is_not_first = start > 0
            chunks.append((audio[start:end], start / SR, is_not_first))

            next_start = start + segment_samples - overlap_samples
            if next_start >= audio_length:
                break
            if audio_length - next_start < overlap_samples * 2:
                chunks.append((audio[next_start:audio_length], next_start / SR, True))
                break
            start = next_start

        return chunks

    def _transcribe_text(self, model, audio: np.ndarray) -> str:
        chunks = self._get_chunks(audio)
        all_texts = []
        had_overlap = []
        for chunk_audio, _, has_overlap in chunks:
            if self.cancel_event.is_set():
                break
            with torch.inference_mode():
                output = model.transcribe(
                    [chunk_audio], batch_size=1, timestamps=False, verbose=False,
                )
            all_texts.append(self._extract_text(output))
            had_overlap.append(has_overlap)
        return stitch_texts(all_texts, had_overlap, self.model_type)

    def _transcribe_canary(self, model, audio: np.ndarray) -> str:
        import soundfile as sf

        chunks = self._get_chunks(audio)
        all_texts = []
        had_overlap = []
        temp_files = []

        try:
            for chunk_audio, _, has_overlap in chunks:
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
                had_overlap.append(has_overlap)
        finally:
            for tf_path in temp_files:
                try:
                    os.remove(tf_path)
                except OSError:
                    pass

        return stitch_texts(all_texts, had_overlap, self.model_type)

    def _transcribe_with_timestamps(
        self, model, audio: np.ndarray
    ) -> List[Tuple[float, float, str]]:
        chunks = self._get_chunks(audio)
        all_chunk_words = []
        had_overlap = []
        for chunk_audio, time_offset, has_overlap in chunks:
            if self.cancel_event.is_set():
                break
            with torch.inference_mode():
                hypotheses = model.transcribe(
                    [chunk_audio], batch_size=1, timestamps=True,
                    return_hypotheses=True, verbose=False,
                )
            word_segs = self._extract_word_timestamps(hypotheses, time_offset)
            if not word_segs:
                txt = self._extract_text(hypotheses)
                if txt:
                    chunk_end = time_offset + len(chunk_audio) / SR
                    word_segs = [(time_offset, chunk_end, txt)]
            all_chunk_words.append(word_segs or [])
            had_overlap.append(has_overlap)

        stitched_words = stitch_timestamp_segments(all_chunk_words, had_overlap)
        if stitched_words:
            return self._group_words_into_segments(
                stitched_words, max_duration=float(self.settings.segment_duration)
            )
        return []

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

    def _extract_word_timestamps(
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


async def _queue_worker():
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


def _resolve_model_key(
    model_name: Optional[str],
    precision: Optional[str],
    device: Optional[str],
    defaults: TranscriptionSettings,
) -> Tuple[str, Dict[str, Any]]:
    target_name = model_name or ALL_MODELS[defaults.model_key]["name"]
    target_prec = precision if precision else ALL_MODELS[defaults.model_key]["precision"]

    key = f"{target_name} - {target_prec}"
    if key in ALL_MODELS:
        return key, ALL_MODELS[key]

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
        _state.queue = asyncio.Queue()
        _state.cancel_event.clear()
        _state.worker_task = asyncio.create_task(_queue_worker())
        logger.info("Transcription queue worker started")
        yield
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
