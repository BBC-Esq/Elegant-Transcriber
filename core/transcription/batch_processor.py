from __future__ import annotations

from pathlib import Path
from threading import Event

from PySide6.QtCore import QThread, Signal, QElapsedTimer

from core.audio.audio_loader import load_audio_for_parakeet, PARAKEET_SAMPLE_RATE
from core.output.writers import SegmentData, TranscriptionResult, write_output
from core.text.curation import curate_text
from core.transcription.parakeet_core import (
    transcribe_canary,
    transcribe_text,
    transcribe_with_timestamps,
)
from core.logging_config import get_logger

logger = get_logger(__name__)


def _is_oom_error(exc: Exception) -> bool:
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, AttributeError):
        pass
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        if "out of memory" in msg or ("cuda" in msg and "alloc" in msg):
            return True
    return False


def _deduplicated_output_path(output_dir: Path, stem: str, suffix: str,
                               seen: dict[str, int]) -> Path:
    key = stem.lower()
    if key in seen:
        seen[key] += 1
        return output_dir / f"{stem}_{seen[key]}{suffix}"
    else:
        seen[key] = 0
        return output_dir / f"{stem}{suffix}"


class BatchProcessor(QThread):
    progress = Signal(int, int, str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        files: list[Path],
        model,
        output_format: str,
        output_directory: str | None,
        segment_length: int,
        segment_duration: int,
        include_timestamps: bool,
        model_type: str = "parakeet",
        curate_enabled: bool = False,
    ):
        super().__init__()
        self.files = files
        self.model = model
        self.output_format = output_format
        self.output_directory = output_directory
        self.segment_length = int(segment_length)
        self.segment_duration = int(segment_duration)
        self.include_timestamps = bool(include_timestamps)
        self.model_type = model_type or "parakeet"
        self.curate_enabled = bool(curate_enabled)
        self.stop_requested = Event()

    def request_stop(self) -> None:
        self.stop_requested.set()

    def _emit(self, current: int, total: int, msg: str):
        self.progress.emit(current, total, msg)

    def run(self) -> None:
        timer = QElapsedTimer()
        timer.start()

        seen_names: dict[str, int] = {}

        include_timestamps = self.include_timestamps
        if self.model_type == "canary":
            include_timestamps = False
        elif self.output_format in ("srt", "vtt"):
            include_timestamps = True

        try:
            total_files = len(self.files)

            for idx, audio_file in enumerate(self.files, 1):
                if self.stop_requested.is_set():
                    break

                self._emit(idx, total_files, f"Loading {audio_file.name}")

                try:
                    audio = load_audio_for_parakeet(str(audio_file), target_sr=PARAKEET_SAMPLE_RATE)
                    duration = len(audio) / PARAKEET_SAMPLE_RATE

                    def _cancel_check():
                        return self.stop_requested.is_set()

                    def _progress(chunk_idx: int, total_chunks: int):
                        if total_chunks:
                            pct = min(100, int(((chunk_idx + 1) / total_chunks) * 100))
                            self._emit(idx, total_files,
                                       f"Processing {audio_file.name} ({pct}%)")

                    segment_data_list: list[SegmentData] = []
                    if self.model_type == "canary":
                        text = transcribe_canary(
                            self.model, audio,
                            segment_length=self.segment_length,
                            cancel_check=_cancel_check,
                            progress_callback=_progress,
                        )
                        if self.stop_requested.is_set():
                            break
                        if self.curate_enabled:
                            try:
                                text = curate_text(text)
                            except Exception as e:
                                logger.warning(f"Text curation failed for {audio_file.name}: {e}")
                    elif include_timestamps:
                        ts_segments = transcribe_with_timestamps(
                            self.model, audio,
                            segment_length=self.segment_length,
                            segment_duration=self.segment_duration,
                            cancel_check=_cancel_check,
                            progress_callback=_progress,
                        )
                        if self.stop_requested.is_set():
                            break
                        for s, e, t in ts_segments:
                            segment_data_list.append(SegmentData(start=s, end=e, text=t))
                        text = " ".join(t for _, _, t in ts_segments)
                    else:
                        text = transcribe_text(
                            self.model, audio,
                            segment_length=self.segment_length,
                            cancel_check=_cancel_check,
                            progress_callback=_progress,
                        )
                        if self.stop_requested.is_set():
                            break
                        if self.curate_enabled:
                            try:
                                text = curate_text(text)
                            except Exception as e:
                                logger.warning(f"Text curation failed for {audio_file.name}: {e}")

                    result = TranscriptionResult(
                        text=text,
                        segments=segment_data_list,
                        language=None,
                        duration=duration,
                        source_file=audio_file,
                    )

                    out_suffix = f".{self.output_format}"
                    if self.output_directory:
                        out_dir = Path(self.output_directory)
                        out_dir.mkdir(parents=True, exist_ok=True)
                        output_file = _deduplicated_output_path(
                            out_dir, audio_file.stem, out_suffix, seen_names
                        )
                    else:
                        output_file = audio_file.with_suffix(out_suffix)

                    write_output(result, output_file, self.output_format)

                    self._emit(idx, total_files, f"Completed {audio_file.name}")

                except Exception as e:
                    if _is_oom_error(e):
                        self.error.emit(
                            f"GPU out of memory processing {audio_file.name}: {e}\n"
                            "Stopping batch. Try a smaller precision (float16/bfloat16) "
                            "or shorter chunk length."
                        )
                        logger.error("OOM error, stopping batch: %s", e)
                        break
                    self.error.emit(f"Error processing {audio_file.name}: {e}")
                    logger.error("Error processing %s: %s", audio_file.name, e, exc_info=True)

        except Exception as e:
            self.error.emit(f"Processing failed: {e}")
            logger.exception("Batch processing failed")

        finally:
            elapsed = timer.elapsed() / 1000.0
            self.finished.emit(f"Processing time: {elapsed:.2f} seconds")
