import os
import sys

APP_FOLDER_NAME = "Elegant Transcriber"


def get_resource_path(relative_path: str) -> str:
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, relative_path)


def get_settings_dir() -> str:
    base = os.environ.get("APPDATA") or os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, APP_FOLDER_NAME)


def get_data_dir() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, f"{APP_FOLDER_NAME} Data")


def get_install_dir() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    if os.path.basename(script_dir) == "app" and os.path.exists(os.path.join(parent_dir, "compute_type.txt")):
        return parent_dir
    return script_dir


def is_gpu_install() -> bool:
    install_dir = get_install_dir()
    compute_type_file = os.path.join(install_dir, "compute_type.txt")

    if os.path.exists(compute_type_file):
        try:
            with open(compute_type_file, 'r') as f:
                return f.read().strip().lower() == "gpu"
        except Exception:
            pass

    return True
