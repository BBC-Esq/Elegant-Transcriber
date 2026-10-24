from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List


PARAKEET_CHUNK_LENGTH = 90
CANARY_CHUNK_LENGTH = 40
TURBOCTC_CHUNK_LENGTH = 45
QWEN3_CHUNK_LENGTH = 120
QWEN3_MAX_CHUNK_LENGTH = 1200

LANGUAGE_AUTO = "auto"
LANGUAGES: Dict[str, str] = {
    "ar": "Arabic",
    "yue": "Cantonese",
    "zh": "Chinese",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "fil": "Filipino",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mk": "Macedonian",
    "ms": "Malay",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "es": "Spanish",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
}


@dataclass(frozen=True)
class ModelFamily:
    chunk_length: int
    min_chunk_length: int
    max_chunk_length: int
    supports_timestamps: bool
    cpu_allowed: bool
    configurable_chunk_length: bool = False
    supports_language: bool = False


FAMILIES: Dict[str, ModelFamily] = {
    "parakeet": ModelFamily(
        chunk_length=PARAKEET_CHUNK_LENGTH, min_chunk_length=10, max_chunk_length=100,
        supports_timestamps=True, cpu_allowed=True, configurable_chunk_length=True,
    ),
    "canary": ModelFamily(
        chunk_length=CANARY_CHUNK_LENGTH, min_chunk_length=6, max_chunk_length=CANARY_CHUNK_LENGTH,
        supports_timestamps=False, cpu_allowed=False,
    ),
    "turboctc": ModelFamily(
        chunk_length=TURBOCTC_CHUNK_LENGTH, min_chunk_length=10, max_chunk_length=100,
        supports_timestamps=True, cpu_allowed=True,
    ),
    "qwen3asr": ModelFamily(
        chunk_length=QWEN3_CHUNK_LENGTH, min_chunk_length=10, max_chunk_length=QWEN3_MAX_CHUNK_LENGTH,
        supports_timestamps=False, cpu_allowed=False, supports_language=True,
    ),
}


@dataclass
class ModelInfo:
    name: str
    model_id: str
    description: str
    model_type: str = "parakeet"
    precisions: list[str] = field(default_factory=lambda: ["bfloat16", "float16", "float32"])
    vram: dict[str, str] = field(default_factory=dict)


class ModelMetadata:

    _MODELS: list[ModelInfo] = [
        ModelInfo(
            name="Parakeet TDT 0.6B v2",
            model_id="nvidia/parakeet-tdt-0.6b-v2",
            description="English only. Higher accuracy for English transcription than v3.",
            model_type="parakeet",
            vram={"bfloat16": "~1.5 GB", "float16": "~1.5 GB", "float32": "~2.5 GB"},
        ),
        ModelInfo(
            name="Parakeet TDT 0.6B v3",
            model_id="nvidia/parakeet-tdt-0.6b-v3",
            description="Multilingual. Slightly lower English accuracy than v2.",
            model_type="parakeet",
            vram={"bfloat16": "~1.5 GB", "float16": "~1.5 GB", "float32": "~2.5 GB"},
        ),
        ModelInfo(
            name="Canary-Qwen 2.5B",
            model_id="nvidia/canary-qwen-2.5b",
            description=(
                "English only. Higher accuracy than Parakeet but ~30x slower and "
                "requires ~6 GB VRAM. Does not support timestamps."
            ),
            model_type="canary",
            precisions=["bfloat16", "float16"],
            vram={"bfloat16": "~5.9 GB", "float16": "~5.9 GB"},
        ),
        ModelInfo(
            name="Granite Speech 5.0 470M TurboCTC",
            model_id="ibm-granite/granite-speech-5.0-470m-turboctc",
            description=(
                "Multilingual (English, French, German, Spanish, Portuguese, Italian and "
                "Japanese), detected automatically. The fastest model, with slightly lower "
                "English accuracy than Parakeet v2."
            ),
            model_type="turboctc",
            precisions=["bfloat16", "float32"],
            vram={"bfloat16": "~1.5 GB", "float32": "~2.5 GB"},
        ),
        ModelInfo(
            name="Qwen3-ASR 0.6B",
            model_id="Qwen/Qwen3-ASR-0.6B-hf",
            description=(
                "Multilingual (30 languages; choose one under Language or let it detect it). "
                "More accurate than Canary and a bit faster, but much slower than Parakeet. "
                "Does not support timestamps."
            ),
            model_type="qwen3asr",
            vram={"bfloat16": "~3.3 GB", "float16": "~3.3 GB", "float32": "~6 GB"},
        ),
        ModelInfo(
            name="Qwen3-ASR 1.7B",
            model_id="Qwen/Qwen3-ASR-1.7B-hf",
            description=(
                "Multilingual (30 languages; choose one under Language or let it detect it). "
                "The most accurate model, at the same speed as the 0.6B. Does not support timestamps."
            ),
            model_type="qwen3asr",
            vram={"bfloat16": "~6.3 GB", "float16": "~6.3 GB", "float32": "~11 GB"},
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

    @staticmethod
    def get_family_for_type(model_type: str) -> ModelFamily:
        return FAMILIES.get(model_type, FAMILIES["parakeet"])

    @classmethod
    def get_family(cls, name: str) -> ModelFamily:
        return cls.get_family_for_type(cls.get_model_type(name))

    @classmethod
    def supports_timestamps(cls, name: str) -> bool:
        return cls.get_family(name).supports_timestamps

    @classmethod
    def is_cpu_allowed(cls, name: str) -> bool:
        return cls.get_family(name).cpu_allowed

    @classmethod
    def supports_language(cls, name: str) -> bool:
        return cls.get_family(name).supports_language

    @staticmethod
    def resolve_language(value: str | None) -> str | None:
        key = (value or "").strip().lower()
        if key == LANGUAGE_AUTO or key in LANGUAGES:
            return key
        for code, language in LANGUAGES.items():
            if language.lower() == key:
                return code
        return None

    @classmethod
    def get_chunk_length(cls, name: str) -> int:
        return cls.get_family(name).chunk_length

    @classmethod
    def effective_chunk_length(cls, model_type: str, configured: int) -> int:
        family = cls.get_family_for_type(model_type)
        return int(configured) if family.configurable_chunk_length else family.chunk_length

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
            if not cls.is_cpu_allowed(model_name):
                return []
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
                    "avg_vram_usage": m.vram.get(prec, ""),
                    "supports_timestamps": cls.supports_timestamps(m.name),
                }
        return result

    @classmethod
    def resolve_model_key(cls, name: str, precision: str) -> str | None:
        for m in cls._MODELS:
            if m.name == name and precision in m.precisions:
                return f"{m.name} - {precision}"
        return None
