from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class TranscriptionSettings:
    model_key: str
    device: str
    segment_length: int
    segment_duration: int
    output_format: str
    word_timestamps: bool
    recursive: bool = False
    selected_extensions: List[str] = field(default_factory=list)
