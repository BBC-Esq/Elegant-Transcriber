from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QSettings

from core.logging_config import get_logger
from utils import get_settings_dir

logger = get_logger(__name__)

_LEGACY_ORG = "ElegantRecorder"
_LEGACY_APP = "Recorder"


def open_ui_settings() -> QSettings:
    path = Path(get_settings_dir()) / "settings.ini"
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    settings = QSettings(str(path), QSettings.IniFormat)
    if is_new:
        _migrate_legacy_settings(settings)
    return settings


def _migrate_legacy_settings(settings: QSettings) -> None:
    legacy = QSettings(_LEGACY_ORG, _LEGACY_APP)
    keys = legacy.allKeys()
    if not keys:
        return
    for key in keys:
        settings.setValue(key, legacy.value(key))
    settings.sync()
    if settings.status() != QSettings.NoError:
        logger.warning(f"Could not write {settings.fileName()}; left the old settings in place")
        return
    legacy.clear()
    legacy.sync()
    _delete_legacy_registry_key()
    logger.info(f"Moved {len(keys)} window and panel settings to {settings.fileName()}")


def _delete_legacy_registry_key() -> None:
    if sys.platform != "win32":
        return
    import winreg
    for subkey in (rf"Software\{_LEGACY_ORG}\{_LEGACY_APP}", rf"Software\{_LEGACY_ORG}"):
        try:
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, subkey)
        except OSError:
            pass
