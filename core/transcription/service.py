from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QObject, Signal, QRunnable, QThreadPool

from core.audio.audio_loader import load_audio_for_parakeet, PARAKEET_SAMPLE_RATE
from core.temp_file_manager import temp_file_manager
from core.text.curation import curate_text
from core.output.writers import SegmentData, TranscriptionResult
from core.transcription.parakeet_core import (
    transcribe_canary,
    transcribe_text,
    transcribe_with_timestamps,
)
from core.models.metadata import ModelMetadata
from core.logging_config import get_logger

logger = get_logger(__name__)


class _TranscriberSignals(QObject):
    transcription_done = Signal(str)
    transcription_done_with_result = Signal(object)
    progress_updated = Signal(int, int, float)
    error_occurred = Signal(str)
    cancelled = Signal()


class _TranscriptionRunnable(QRunnable):
    def __init__(
        self,
        model,
        model_version: str,
        audio_file: str | Path,
        is_temp_file: bool = True,
        segment_length: int = 90,
        segment_duration: int = 10,
        include_timestamps: bool = False,
        model_type: str = "parakeet",
        get_current_version_func=None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self.model = model
        self.model_version = model_version
        self.audio_file = Path(audio_file)
        self.is_temp_file = is_temp_file
        self.segment_length = segment_length
        self.segment_duration = segment_duration
        self.include_timestamps = include_timestamps
        self.model_type = model_type
        self.get_current_version = get_current_version_func
        self.cancel_event = cancel_event or threading.Event()
        self.signals = _TranscriberSignals()

    def _is_cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def run(self) -> None:
        try:
            if self._is_cancelled():
                logger.info("Transcription cancelled before starting")
                self.signals.cancelled.emit()
                return

            if self.get_current_version and self.get_current_version() != self.model_version:
                logger.warning("Model changed during transcription setup")
                self.signals.cancelled.emit()
                return

            logger.info(f"Starting transcription: {self.audio_file}")

            audio = load_audio_for_parakeet(self.audio_file, target_sr=PARAKEET_SAMPLE_RATE)
            total_duration = len(audio) / PARAKEET_SAMPLE_RATE

            def _cancel_check():
                return self.cancel_event.is_set()

            def _progress(chunk_idx: int, total_chunks: int):
                if total_chunks > 0:
                    pct = min(100.0, ((chunk_idx + 1) / total_chunks) * 100.0)
                    self.signals.progress_updated.emit(chunk_idx + 1, total_chunks, pct)

            segment_data_list: list[SegmentData] = []
            if self.model_type == "canary":
                text = transcribe_canary(
                    self.model, audio,
                    segment_length=self.segment_length,
                    cancel_check=_cancel_check,
                    progress_callback=_progress,
                )
                if self._is_cancelled():
                    self.signals.cancelled.emit()
                    return
            elif self.include_timestamps:
                ts_segments = transcribe_with_timestamps(
                    self.model, audio,
                    segment_length=self.segment_length,
                    segment_duration=self.segment_duration,
                    cancel_check=_cancel_check,
                    progress_callback=_progress,
                )
                if self._is_cancelled():
                    self.signals.cancelled.emit()
                    return
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
                if self._is_cancelled():
                    self.signals.cancelled.emit()
                    return

            logger.info(f"Transcription completed ({len(segment_data_list)} segments, {len(text)} chars)")

            result = TranscriptionResult(
                text=text,
                segments=segment_data_list,
                language=None,
                duration=total_duration,
                source_file=self.audio_file,
            )

            self.signals.transcription_done.emit(text)
            self.signals.transcription_done_with_result.emit(result)

        except Exception as e:
            if self._is_cancelled():
                logger.info("Transcription cancelled during processing")
                self.signals.cancelled.emit()
            else:
                logger.exception("Transcription failed")
                self.signals.error_occurred.emit(f"Transcription failed: {e}")
        finally:
            if self.is_temp_file:
                temp_file_manager.release(self.audio_file)


class TranscriptionService(QObject):
    transcription_started = Signal()
    transcription_completed = Signal(str)
    transcription_completed_with_result = Signal(object)
    transcription_progress = Signal(int, int, float)
    transcription_error = Signal(str)
    transcription_cancelled = Signal()

    def __init__(self, curate_text_enabled: bool = False):
        super().__init__()
        self.curate_enabled = curate_text_enabled
        self._thread_pool = QThreadPool.globalInstance()
        self._get_model_version_func = None
        self._cancel_event: threading.Event | None = None
        self._is_transcribing = False
        self._segment_length: int = 90
        self._segment_duration: int = 10
        self._include_timestamps: bool = False
        self._timestamps_override: bool | None = None
        self._model_type: str = "parakeet"

    def set_model_version_provider(self, func) -> None:
        self._get_model_version_func = func

    def set_parakeet_params(self, segment_length: int, segment_duration: int, include_timestamps: bool) -> None:
        self._segment_length = int(segment_length)
        self._segment_duration = int(segment_duration)
        self._include_timestamps = bool(include_timestamps)
        logger.debug(
            f"Parakeet params: segment_length={self._segment_length}s, "
            f"segment_duration={self._segment_duration}s, "
            f"timestamps={self._include_timestamps}"
        )

    def set_model_type(self, model_type: str) -> None:
        self._model_type = model_type or "parakeet"
        logger.debug(f"Active model_type: {self._model_type}")

    def set_timestamps_override(self, include_timestamps: bool | None) -> None:
        self._timestamps_override = include_timestamps

    def is_transcribing(self) -> bool:
        return self._is_transcribing

    def cancel_transcription(self) -> bool:
        if self._cancel_event and self._is_transcribing:
            logger.info("Cancellation requested")
            self._cancel_event.set()
            return True
        return False

    def transcribe_file(self, model, model_version: str, audio_file: str | Path,
                        is_temp_file: bool = True) -> None:
        if not model:
            error_msg = "No model available for transcription"
            logger.error(error_msg)
            self.transcription_error.emit(error_msg)
            if is_temp_file:
                temp_file_manager.release(Path(audio_file))
            return

        try:
            self._cancel_event = threading.Event()
            self._is_transcribing = True

            include_timestamps = self._include_timestamps
            if self._timestamps_override is not None:
                include_timestamps = self._timestamps_override
            self._timestamps_override = None

            effective_segment_length = self._segment_length
            if self._model_type == "canary":
                effective_segment_length = ModelMetadata.get_chunk_length("Canary-Qwen 2.5B")

            runnable = _TranscriptionRunnable(
                model=model,
                model_version=model_version,
                audio_file=str(audio_file),
                is_temp_file=is_temp_file,
                segment_length=effective_segment_length,
                segment_duration=self._segment_duration,
                include_timestamps=include_timestamps,
                model_type=self._model_type,
                get_current_version_func=self._get_model_version_func,
                cancel_event=self._cancel_event,
            )
            runnable.signals.transcription_done.connect(self._on_transcription_done)
            runnable.signals.transcription_done_with_result.connect(
                self._on_transcription_done_with_result
            )
            runnable.signals.progress_updated.connect(self._on_progress_updated)
            runnable.signals.error_occurred.connect(self._on_transcription_error)
            runnable.signals.cancelled.connect(self._on_transcription_cancelled)
            self._thread_pool.start(runnable)
            self.transcription_started.emit()
        except Exception as e:
            logger.exception("Failed to start transcription")
            self._is_transcribing = False
            self.transcription_error.emit(f"Failed to start transcription: {e}")
            if is_temp_file:
                temp_file_manager.release(Path(audio_file))

    def _on_transcription_done(self, text: str) -> None:
        self._is_transcribing = False
        if self.curate_enabled:
            try:
                text = curate_text(text)
            except Exception as e:
                logger.warning(f"Text curation failed: {e}")

        self.transcription_completed.emit(text)

    def _on_transcription_done_with_result(self, result: object) -> None:
        self.transcription_completed_with_result.emit(result)

    def _on_progress_updated(self, segment_num: int, total_segments: int, percent: float) -> None:
        self.transcription_progress.emit(segment_num, total_segments, percent)

    def _on_transcription_error(self, error: str) -> None:
        self._is_transcribing = False
        logger.error(f"Transcription error: {error}")
        self.transcription_error.emit(error)

    def _on_transcription_cancelled(self) -> None:
        self._is_transcribing = False
        logger.info("Transcription was cancelled")
        self.transcription_cancelled.emit()

    def set_curation_enabled(self, enabled: bool) -> None:
        self.curate_enabled = enabled

    def cleanup(self) -> None:
        if self._cancel_event:
            self._cancel_event.set()
        self._thread_pool.waitForDone(5000)
        logger.debug("TranscriptionService cleanup complete")
