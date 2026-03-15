from collections import OrderedDict

_MODEL_SPECS = [
    ("Parakeet TDT 0.6B v2", "nvidia/parakeet-tdt-0.6b-v2", "bfloat16", 80, "~1.5 GB"),
    ("Parakeet TDT 0.6B v2", "nvidia/parakeet-tdt-0.6b-v2", "float16",  80, "~1.5 GB"),
    ("Parakeet TDT 0.6B v2", "nvidia/parakeet-tdt-0.6b-v2", "float32",  80, "~2.5 GB"),
    ("Parakeet TDT 0.6B v3", "nvidia/parakeet-tdt-0.6b-v3", "bfloat16", 80, "~1.5 GB"),
    ("Parakeet TDT 0.6B v3", "nvidia/parakeet-tdt-0.6b-v3", "float16",  80, "~1.5 GB"),
    ("Parakeet TDT 0.6B v3", "nvidia/parakeet-tdt-0.6b-v3", "float32",  80, "~2.5 GB"),
]

PARAKEET_MODELS = {
    f"{name} - {prec}": {
        'name': name,
        'precision': prec,
        'model_id': model_id,
        'default_segment_length': seg_len,
        'avg_vram_usage': vram,
    }
    for name, model_id, prec, seg_len, vram in _MODEL_SPECS
}

MODEL_NAMES = list(OrderedDict.fromkeys(name for name, *_ in _MODEL_SPECS))

MODEL_PRECISIONS = {}
for name, _, prec, *_ in _MODEL_SPECS:
    MODEL_PRECISIONS.setdefault(name, []).append(prec)

MODEL_TOOLTIPS = {
    "Parakeet TDT 0.6B v2": "English only. Higher accuracy for English transcription than v3.",
    "Parakeet TDT 0.6B v3": "Multilingual with translation support. Slightly lower English accuracy than v2.",
}

SUPPORTED_AUDIO_EXTENSIONS = [
    ".aac", ".amr", ".asf", ".avi", ".flac", ".m4a",
    ".mkv", ".mp3", ".mp4", ".wav", ".webm", ".wma"
]

TIMESTAMP_FORMATS = ["txt", "srt", "vtt", "json"]

DEFAULT_SEGMENT_LENGTH = 80
DEFAULT_SEGMENT_DURATION = 10
DEFAULT_OUTPUT_FORMAT = "txt"
