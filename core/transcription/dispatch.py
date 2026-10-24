from __future__ import annotations

from typing import List, Tuple

import numpy as np

from core.transcription.parakeet_core import (
    transcribe_canary,
    transcribe_text,
    transcribe_with_timestamps,
)
from core.transcription.turboctc_core import (
    transcribe_turboctc,
    transcribe_turboctc_with_timestamps,
)


_TEXT_TRANSCRIBERS = {
    "parakeet": transcribe_text,
    "canary": transcribe_canary,
    "turboctc": transcribe_turboctc,
}

_TIMESTAMP_TRANSCRIBERS = {
    "parakeet": transcribe_with_timestamps,
    "turboctc": transcribe_turboctc_with_timestamps,
}


def transcribe_audio(
    model,
    model_type: str,
    audio: np.ndarray,
    segment_length: int,
    segment_duration: float,
    include_timestamps: bool,
    cancel_check=None,
    progress_callback=None,
) -> Tuple[str, List[Tuple[float, float, str]]]:
    with_timestamps = _TIMESTAMP_TRANSCRIBERS.get(model_type) if include_timestamps else None
    if with_timestamps is not None:
        segments = with_timestamps(
            model, audio,
            segment_length=segment_length,
            segment_duration=segment_duration,
            cancel_check=cancel_check,
            progress_callback=progress_callback,
        )
        return " ".join(t for _, _, t in segments), segments
    transcribe = _TEXT_TRANSCRIBERS.get(model_type, transcribe_text)
    text = transcribe(
        model, audio,
        segment_length=segment_length,
        cancel_check=cancel_check,
        progress_callback=progress_callback,
    )
    return text, []
