from dataclasses import dataclass
from typing import List


@dataclass
class TranscriptionSettings:
    model_key: str
    device: str
    segment_length: int
    segment_duration: int
    output_format: str
    word_timestamps: bool
    recursive: bool
    selected_extensions: List[str]

    def validate(self) -> List[str]:
        warnings = []
        is_canary = "canary" in self.model_key.lower()
        if self.device.lower() == "cpu":
            if is_canary:
                warnings.append(
                    "CPU inference with Canary-Qwen is extremely slow. "
                    "A CUDA-capable GPU with at least 12 GB VRAM is strongly recommended."
                )
            else:
                warnings.append(
                    "CPU inference with Parakeet is very slow. "
                    "A CUDA-capable GPU is strongly recommended."
                )
        return warnings
