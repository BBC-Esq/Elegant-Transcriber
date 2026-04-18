from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from utils import get_resource_path
from core.logging_config import get_logger
from core.exceptions import ConfigurationError

logger = get_logger(__name__)

_FLUSH_DELAY_MS = 500


class ConfigManager:

    VALID_OPTIONS = {
        "device_types": {"cpu", "cuda"},
        "precisions": {"float16", "float32", "bfloat16"},
        "output_formats": {"txt", "srt", "vtt", "json"},
        "single_file_output_modes": {
            "clipboard", "save_to_source", "save_and_clipboard", "save_to_custom"
        },
    }

    DEFAULT_CONFIG = {
        "model_name": "Parakeet TDT 0.6B v2",
        "precision": "bfloat16",
        "device_type": "cuda",
        "show_clipboard_window": False,
        "supported_precisions": {"cpu": ["float32"], "cuda": ["bfloat16", "float16", "float32"]},
        "curate_transcription": True,
        "clipboard_append_mode": False,
        "include_timestamps": False,
        "segment_length": 90,
        "segment_duration": 10,
        "output_format": "txt",
        "single_file_output_mode": "clipboard",
        "output_directory": "",
        "batch_recursive": False,
        "batch_extensions": [
            ".aac", ".amr", ".asf", ".avi", ".flac", ".m4a",
            ".mkv", ".mp3", ".mp4", ".wav", ".webm", ".wma",
        ],
    }

    VALIDATION_SCHEMA = {
        "model_name": {"type": str, "validator": "_validate_model_name"},
        "device_type": {"type": str, "options": "device_types", "lowercase": True},
        "precision": {"type": str, "options": "precisions"},
        "show_clipboard_window": {"type": bool},
        "curate_transcription": {"type": bool},
        "clipboard_append_mode": {"type": bool},
        "include_timestamps": {"type": bool},
        "segment_length": {"type": int, "validator": "_validate_segment_length"},
        "segment_duration": {"type": int, "validator": "_validate_segment_duration"},
        "output_format": {"type": str, "options": "output_formats"},
        "single_file_output_mode": {"type": str, "options": "single_file_output_modes"},
        "output_directory": {"type": str},
        "batch_recursive": {"type": bool},
    }

    def __init__(self):
        self._config_path = Path(get_resource_path("config.yaml"))
        self._config_cache: dict[str, Any] | None = None
        self._dirty = False
        self._flush_timer = None

    def _ensure_flush_timer(self) -> None:
        if self._flush_timer is not None:
            return
        try:
            from PySide6.QtCore import QTimer
            self._flush_timer = QTimer()
            self._flush_timer.setSingleShot(True)
            self._flush_timer.setInterval(_FLUSH_DELAY_MS)
            self._flush_timer.timeout.connect(self._flush)
        except Exception:
            self._flush_timer = None

    def _schedule_flush(self) -> None:
        self._ensure_flush_timer()
        if self._flush_timer is not None:
            self._flush_timer.start()

    def _flush(self) -> None:
        if not self._dirty:
            return
        if self._config_cache is None:
            self._dirty = False
            return
        try:
            self._save_to_disk(self._config_cache)
            self._dirty = False
        except Exception as e:
            logger.error(f"Deferred config flush failed: {e}")

    def flush_sync(self) -> None:
        if self._flush_timer is not None:
            self._flush_timer.stop()
        self._flush()

    def _ensure_cache(self) -> dict[str, Any]:
        if self._config_cache is None:
            self._config_cache = self._load_from_file()
        return self._config_cache

    def _apply_to_cache(self, updates: dict[str, Any]) -> None:
        cache = self._ensure_cache()
        self._deep_update(cache, updates)
        self._dirty = True
        self._schedule_flush()

    @property
    def config_path(self) -> Path:
        return self._config_path

    @staticmethod
    def _get_valid_model_names() -> set[str]:
        try:
            from core.models.metadata import ModelMetadata
            return set(ModelMetadata.get_all_model_names())
        except ImportError:
            logger.warning("Could not import ModelMetadata for validation")
            return set()

    def _validate_model_name(self, value: Any) -> str:
        valid_models = self._get_valid_model_names()
        if valid_models and value not in valid_models:
            return self.DEFAULT_CONFIG["model_name"]
        return value

    def _validate_segment_length(self, value: Any) -> int:
        if isinstance(value, int) and 10 <= value <= 100:
            return value
        return self.DEFAULT_CONFIG["segment_length"]

    def _validate_segment_duration(self, value: Any) -> int:
        if isinstance(value, int) and 1 <= value <= 90:
            return value
        return self.DEFAULT_CONFIG["segment_duration"]

    def load_config(self) -> dict[str, Any]:
        return copy.deepcopy(self._ensure_cache())

    def _load_from_file(self) -> dict[str, Any]:
        loaded_config = {}

        try:
            if self._config_path.exists():
                with self._config_path.open() as f:
                    loaded_config = yaml.safe_load(f) or {}
            else:
                logger.info("Config file not found, creating with defaults")
        except (yaml.YAMLError, Exception) as e:
            logger.error(f"Error loading config: {e}. Reverting to defaults.")
            loaded_config = {}

        if not isinstance(loaded_config, dict):
            loaded_config = {}

        merged_config = copy.deepcopy(self.DEFAULT_CONFIG)
        filtered = {k: v for k, v in loaded_config.items() if k in self.DEFAULT_CONFIG}
        self._deep_update(merged_config, filtered)
        self._validate_and_sanitize(merged_config)

        return merged_config

    def _validate_and_sanitize(self, config: dict[str, Any]) -> None:
        for key, schema in self.VALIDATION_SCHEMA.items():
            value = config.get(key)
            default = self.DEFAULT_CONFIG[key]

            if not isinstance(value, schema["type"]):
                logger.warning(f"Invalid type for {key}, reverting to default")
                config[key] = default
                continue

            if schema.get("lowercase") and isinstance(value, str):
                value = value.lower()
                config[key] = value

            if "options" in schema:
                valid_opts = self.VALID_OPTIONS[schema["options"]]
                if value not in valid_opts:
                    logger.warning(f"Invalid {key} '{value}', reverting to default")
                    config[key] = default
                    continue

            if "validator" in schema:
                config[key] = getattr(self, schema["validator"])(value)

        self._validate_supported_precisions(config)

    def _validate_supported_precisions(self, config: dict[str, Any]) -> None:
        key = "supported_precisions"
        if not isinstance(config.get(key), dict):
            config[key] = copy.deepcopy(self.DEFAULT_CONFIG[key])
            return

        for device in self.VALID_OPTIONS["device_types"]:
            if device not in config[key] or not isinstance(config[key][device], list):
                config[key][device] = []
            else:
                config[key][device] = [
                    p for p in config[key][device]
                    if isinstance(p, str) and p in self.VALID_OPTIONS["precisions"]
                ]

    def _save_to_disk(self, config: dict[str, Any]) -> None:
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            validated_config = copy.deepcopy(config)
            self._validate_and_sanitize(validated_config)

            with self._config_path.open("w") as f:
                yaml.safe_dump(validated_config, f, sort_keys=False)

            self._config_cache = validated_config
            logger.debug("Configuration saved successfully")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise ConfigurationError(f"Failed to save configuration: {e}") from e

    def save_config(self, config: dict[str, Any]) -> None:
        self._save_to_disk(config)
        self._dirty = False
        if self._flush_timer is not None:
            self._flush_timer.stop()

    def update_config(self, updates: dict[str, Any]) -> None:
        self._apply_to_cache(updates)

    def get_value(self, key: str, default: Any = None) -> Any:
        return self._ensure_cache().get(key, default)

    def set_value(self, key: str, value: Any) -> None:
        self._apply_to_cache({key: value})

    def get_model_settings(self) -> dict[str, str]:
        cache = self._ensure_cache()
        return {k: cache[k] for k in ["model_name", "precision", "device_type"]}

    def set_model_settings(self, model_name: str, precision: str, device_type: str) -> None:
        self._apply_to_cache({
            "model_name": model_name,
            "precision": precision,
            "device_type": device_type,
        })

    def get_supported_precisions(self) -> dict[str, list[str]]:
        value = self.get_value("supported_precisions", {"cpu": [], "cuda": []})
        return copy.deepcopy(value) if isinstance(value, dict) else {"cpu": [], "cuda": []}

    def set_supported_precisions(self, device: str, precisions: list[str]) -> None:
        if device not in self.VALID_OPTIONS["device_types"]:
            return
        current = self.get_supported_precisions()
        current[device] = precisions
        self.set_value("supported_precisions", current)

    def invalidate_cache(self) -> None:
        self._config_cache = None

    @staticmethod
    def _deep_update(base_dict: dict[str, Any], update_dict: dict[str, Any]) -> None:
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                ConfigManager._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value


config_manager = ConfigManager()
