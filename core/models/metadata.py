from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List


PARAKEET_CHUNK_LENGTH = 90
CANARY_CHUNK_LENGTH = 40


@dataclass
class ModelInfo:
    name: str
    model_id: str
    description: str
    model_type: str = "parakeet"
    precisions: list[str] = field(default_factory=lambda: ["bfloat16", "float16", "float32"])


class ModelMetadata:

    _MODELS: list[ModelInfo] = [
        ModelInfo(
            name="Parakeet TDT 0.6B v2",
            model_id="nvidia/parakeet-tdt-0.6b-v2",
            description="English only. Higher accuracy for English transcription than v3.",
            model_type="parakeet",
        ),
        ModelInfo(
            name="Parakeet TDT 0.6B v3",
            model_id="nvidia/parakeet-tdt-0.6b-v3",
            description="Multilingual. Slightly lower English accuracy than v2.",
            model_type="parakeet",
        ),
        ModelInfo(
            name="Canary-Qwen 2.5B",
            model_id="nvidia/canary-qwen-2.5b",
            description=(
                "English only. Higher accuracy than Parakeet but ~30x slower and "
                "requires ~11 GB VRAM. Does not support timestamps."
            ),
            model_type="canary",
            precisions=["bfloat16", "float16"],
        ),
    ]

    _MODEL_MAP: Dict[str, ModelInfo] = OrderedDict((m.name, m) for m in _MODELS)

    @classmethod
    def get_all_model_names(cls) -> List[str]:
        return [m.name for m in cls._MODELS]

    @classmethod
    def get_model_info(cls, name: str) -> ModelInfo | None:
        return cls._MODEL_MAP.get(name)

    @classmethod
    def get_model_id(cls, name: str) -> str | None:
        info = cls._MODEL_MAP.get(name)
        return info.model_id if info else None

    @classmethod
    def get_model_type(cls, name: str) -> str:
        info = cls._MODEL_MAP.get(name)
        return info.model_type if info else "parakeet"

    @classmethod
    def supports_timestamps(cls, name: str) -> bool:
        return cls.get_model_type(name) != "canary"

    @classmethod
    def get_chunk_length(cls, name: str) -> int:
        return CANARY_CHUNK_LENGTH if cls.get_model_type(name) == "canary" else PARAKEET_CHUNK_LENGTH

    @classmethod
    def get_description(cls, name: str) -> str:
        info = cls._MODEL_MAP.get(name)
        return info.description if info else ""

    @classmethod
    def get_precision_options(
        cls, model_name: str, device: str, supported_precisions: Dict[str, List[str]]
    ) -> List[str]:
        info = cls._MODEL_MAP.get(model_name)
        if info is None:
            return []

        hw_supported = set(supported_precisions.get(device, []))
        options = [p for p in info.precisions if p in hw_supported]

        if device == "cpu":
            options = [opt for opt in options if opt == "float32"]

        return options

    @classmethod
    def get_all_models_with_precisions(cls) -> Dict[str, Dict[str, object]]:
        result: Dict[str, Dict[str, object]] = {}
        for m in cls._MODELS:
            for prec in m.precisions:
                key = f"{m.name} - {prec}"
                result[key] = {
                    "name": m.name,
                    "model_id": m.model_id,
                    "precision": prec,
                    "model_type": m.model_type,
                    "default_segment_length": cls.get_chunk_length(m.name),
                    "avg_vram_usage": cls._vram_estimate(m.model_type, prec),
                }
        return result

    @staticmethod
    def _vram_estimate(model_type: str, precision: str) -> str:
        if model_type == "canary":
            return "~10.9 GB" if precision in ("bfloat16", "float16") else "~18 GB"
        return "~1.5 GB" if precision in ("bfloat16", "float16") else "~2.5 GB"

    @classmethod
    def resolve_model_key(cls, name: str, precision: str) -> str | None:
        for m in cls._MODELS:
            if m.name == name and precision in m.precisions:
                return f"{m.name} - {precision}"
        return None
