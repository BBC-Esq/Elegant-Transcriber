import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from threading import Event

from PySide6.QtCore import QThread, Signal, QElapsedTimer
import numpy as np
import torch
import av

from config.settings import TranscriptionSettings

logger = logging.getLogger(__name__)

SR = 16000


def load_audio(audio_path: str, target_sr: int = SR) -> np.ndarray:
    container = av.open(audio_path)
    resampler = av.AudioResampler(format='s16', layout='mono', rate=target_sr)
    frames = []
    for frame in container.decode(audio=0):
        resampled = resampler.resample(frame)
        for r in resampled:
            frames.append(r.to_ndarray())
    container.close()
    return np.concatenate(frames, axis=1).flatten().astype(np.float32) / 32768.0


def _fmt_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_vtt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def write_segments(segments: List[Tuple[float, float, str]], path: Path, fmt: str):
    if fmt == "srt":
        lines = []
        for idx, (start, end, text) in enumerate(segments, 1):
            lines.append(str(idx))
            lines.append(f"{_fmt_srt(start)} --> {_fmt_srt(end)}")
            lines.append(text)
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")

    elif fmt == "vtt":
        lines = ["WEBVTT", ""]
        for start, end, text in segments:
            lines.append(f"{_fmt_vtt(start)} --> {_fmt_vtt(end)}")
            lines.append(text)
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")

    elif fmt == "json":
        data = {
            "segments": [
                {"start": round(s, 3), "end": round(e, 3), "text": t}
                for s, e, t in segments
            ],
            "text": " ".join(t for _, _, t in segments),
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    else:
        lines = []
        for start, end, text in segments:
            lines.append(f"[{_fmt_vtt(start)} --> {_fmt_vtt(end)}]  {text}")
        path.write_text("\n".join(lines), encoding="utf-8")


class BatchProcessor(QThread):
    progress = Signal(int, int, str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, files: List[Path], settings: TranscriptionSettings,
                 model_info: Dict[str, Any], model_manager):
        super().__init__()
        self.files = files
        self.settings = settings
        self.model_info = model_info
        self.model_manager = model_manager
        self.stop_requested = Event()

    def request_stop(self):
        self.stop_requested.set()

    def _emit_progress(self, file_idx, total_files, chunk_idx, total_chunks, filename):
        file_pct = (chunk_idx / max(total_chunks, 1)) * 100
        overall_done = file_idx - 1 + (chunk_idx / max(total_chunks, 1))
        overall_pct = (overall_done / max(total_files, 1)) * 100
        msg = (
            f"File {file_idx}/{total_files} | "
            f"Overall: {overall_pct:.0f}% | "
            f"Current: {file_pct:.0f}% | "
            f"{filename}"
        )
        self.progress.emit(file_idx, total_files, msg)

    def run(self):
        timer = QElapsedTimer()
        timer.start()
        errors = []

        try:
            self.progress.emit(0, len(self.files), "Loading model...")

            load_timer = QElapsedTimer()
            load_timer.start()

            model = self.model_manager.get_or_load_model(
                self.settings.model_key,
                self.settings.device,
                self.model_info['precision'],
            )

            load_time = load_timer.elapsed() / 1000.0

            if not model:
                detail = getattr(self.model_manager, '_last_error', None)
                msg = f"Failed to load model: {detail}" if detail else "Failed to load model"
                self.error.emit(msg)
                return

            transcribe_timer = QElapsedTimer()
            transcribe_timer.start()

            total_files = len(self.files)
            use_timestamps = self.settings.word_timestamps
            fmt = self.settings.output_format

            for idx, audio_file in enumerate(self.files, 1):
                if self.stop_requested.is_set():
                    break

                try:
                    self._emit_progress(idx, total_files, 0, 1, f"Loading {audio_file.name}")
                    audio = load_audio(str(audio_file))

                    if use_timestamps:
                        segments = self._transcribe_with_timestamps(
                            model, audio, audio_file.name, idx, total_files,
                        )
                        out_path = audio_file.with_suffix(f".{fmt}")
                        write_segments(segments, out_path, fmt)
                    else:
                        text = self._transcribe_text(
                            model, audio, audio_file.name, idx, total_files,
                        )
                        out_path = audio_file.with_suffix(".txt")
                        out_path.write_text(text, encoding="utf-8")

                    self._emit_progress(idx, total_files, 1, 1, f"Completed {audio_file.name}")

                except Exception as e:
                    msg = f"Error processing {audio_file.name}: {e}"
                    logger.error(msg, exc_info=True)
                    errors.append(msg)
                    self.error.emit(msg)

        except Exception as e:
            msg = f"Processing failed: {e}"
            logger.error(msg, exc_info=True)
            errors.append(msg)
            self.error.emit(msg)

        finally:
            total = timer.elapsed() / 1000.0
            try:
                transcribe_time = transcribe_timer.elapsed() / 1000.0
            except Exception:
                transcribe_time = 0.0
            try:
                lt = load_time
            except Exception:
                lt = total
            if errors:
                summary = f"Finished with {len(errors)} error(s) | Model load: {lt:.2f}s | Transcription: {transcribe_time:.2f}s"
            else:
                summary = f"Model load: {lt:.2f}s | Transcription: {transcribe_time:.2f}s"
            self.finished.emit(summary)

    def _get_chunks(self, audio: np.ndarray):
        segment_samples = self.settings.segment_length * SR
        audio_length = len(audio)
        if audio_length > segment_samples:
            num_chunks = (audio_length + segment_samples - 1) // segment_samples
            chunks = []
            for i in range(num_chunks):
                start = i * segment_samples
                end = min(start + segment_samples, audio_length)
                chunks.append((audio[start:end], start / SR))
            return chunks
        return [(audio, 0.0)]

    def _transcribe_text(self, model, audio: np.ndarray, filename: str,
                         file_idx: int, total_files: int) -> str:
        chunks = self._get_chunks(audio)
        all_texts = []
        for i, (chunk_audio, _offset) in enumerate(chunks):
            if self.stop_requested.is_set():
                break
            self._emit_progress(file_idx, total_files, i, len(chunks), filename)
            with torch.inference_mode():
                output = model.transcribe(
                    [chunk_audio], batch_size=1, timestamps=False, verbose=False,
                )
            all_texts.append(self._extract_text(output))
        return " ".join(t for t in all_texts if t)

    def _transcribe_with_timestamps(self, model, audio: np.ndarray, filename: str,
                                     file_idx: int, total_files: int
                                     ) -> List[Tuple[float, float, str]]:
        chunks = self._get_chunks(audio)
        all_segments = []
        for i, (chunk_audio, time_offset) in enumerate(chunks):
            if self.stop_requested.is_set():
                break
            self._emit_progress(file_idx, total_files, i, len(chunks), filename)
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

    def _extract_text(self, output) -> str:
        if not output:
            return ""
        if isinstance(output, (list, tuple)):
            first = output[0]
            if isinstance(first, str):
                return first.strip()
            if hasattr(first, 'text'):
                return first.text.strip()
            return str(first).strip()
        return str(output).strip()

    def _extract_timestamp_segments(self, output, time_offset: float
                                     ) -> List[Tuple[float, float, str]]:
        segments = []
        if not output or not isinstance(output, (list, tuple)):
            return segments
        hyp = output[0]
        if not hasattr(hyp, 'timestamp') or not hyp.timestamp:
            return segments
        ts = hyp.timestamp
        if not isinstance(ts, dict):
            return segments
        word_ts = ts.get('word', [])
        if not word_ts:
            word_ts = ts.get('segment', [])
        for entry in word_ts:
            start = entry.get('start', 0) + time_offset
            end = entry.get('end', start) + time_offset
            text = entry.get('word', '') or entry.get('char', '') or entry.get('segment', '')
            if text.strip():
                segments.append((start, end, text.strip()))
        if segments:
            segments = self._group_words_into_segments(
                segments, max_duration=float(self.settings.segment_duration)
            )
        return segments

    def _group_words_into_segments(self, word_segments: List[Tuple[float, float, str]],
                                    max_duration: float = 10.0
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
