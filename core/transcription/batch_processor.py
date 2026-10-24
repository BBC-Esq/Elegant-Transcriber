from __future__ import annotations

from pathlib import Path
from threading import Event

from PySide6.QtCore import QThread, Signal, QElapsedTimer

from core.audio.audio_loader import load_audio_for_parakeet, PARAKEET_SAMPLE_RATE
from core.output.writers import SegmentData, TranscriptionResult, write_output
from core.text.curation import curate_text
from core.models.metadata import ModelMetadata
from core.transcription.dispatch import transcribe_audio
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


def _deduplicated_output_path(target: Path, seen: set[str]) -> Path:
    # Key on the full output path, not just the stem, so two inputs that share
    # a stem in the same directory (e.g. interview.mp3 + interview.wav, or a
    # case-only difference on Windows) don't clobber each other, while inputs
    # with the same stem in different directories are left untouched.
    candidate = target
    n = 0
    while str(candidate).lower() in seen:
        n += 1
        candidate = target.with_name(f"{target.stem}_{n}{target.suffix}")
    seen.add(str(candidate).lower())
    return candidate


class BatchProcessor(QThread):
    progress = Signal(int, int, str)
    completed = Signal(str)
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
        language: str = "en",
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
        self.language = language
        self.stop_requested = Event()

    def request_stop(self) -> None:
        self.stop_requested.set()

    def _emit(self, current: int, total: int, msg: str):
        self.progress.emit(current, total, msg)

    def run(self) -> None:
        timer = QElapsedTimer()
        timer.start()

        seen_paths: set[str] = set()

        include_timestamps = self.include_timestamps
        if not ModelMetadata.get_family_for_type(self.model_type).supports_timestamps:
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

                    text, ts_segments, language = transcribe_audio(
                        self.model, self.model_type, audio,
                        segment_length=self.segment_length,
                        segment_duration=self.segment_duration,
                        include_timestamps=include_timestamps,
                        language=self.language,
                        cancel_check=_cancel_check,
                        progress_callback=_progress,
                    )
                    if self.stop_requested.is_set():
                        break
                    segment_data_list = [SegmentData(start=s, end=e, text=t) for s, e, t in ts_segments]
                    if not ts_segments and self.curate_enabled:
                        try:
                            text = curate_text(text)
                        except Exception as e:
                            logger.warning(f"Text curation failed for {audio_file.name}: {e}")

                    result = TranscriptionResult(
                        text=text,
                        segments=segment_data_list,
                        language=language,
                        duration=duration,
                        source_file=audio_file,
                    )

                    out_suffix = f".{self.output_format}"
                    if self.output_directory:
                        out_dir = Path(self.output_directory)
                        out_dir.mkdir(parents=True, exist_ok=True)
                        target = out_dir / f"{audio_file.stem}{out_suffix}"
                    else:
                        target = audio_file.with_suffix(out_suffix)
                    output_file = _deduplicated_output_path(target, seen_paths)

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
            self.completed.emit(f"Processing time: {elapsed:.2f} seconds")
