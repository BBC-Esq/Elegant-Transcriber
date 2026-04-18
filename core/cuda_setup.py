from __future__ import annotations

import os
import sys
import platform
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def set_cuda_paths() -> bool:
    if platform.system() != "Windows":
        return False

    try:
        venv_base = Path(sys.executable).parent.parent
        nvidia_base_path = venv_base / 'Lib' / 'site-packages' / 'nvidia'

        if not nvidia_base_path.exists():
            logger.debug("Skipping CUDA path setup: nvidia packages not found")
            return False

        cuda_paths = [
            nvidia_base_path / 'cuda_runtime' / 'bin',
            nvidia_base_path / 'cuda_runtime' / 'lib' / 'x64',
            nvidia_base_path / 'cuda_runtime' / 'include',
            nvidia_base_path / 'cublas' / 'bin',
            nvidia_base_path / 'cudnn' / 'bin',
            nvidia_base_path / 'cuda_nvrtc' / 'bin',
            nvidia_base_path / 'cuda_nvcc' / 'bin',
        ]

        existing_paths = [p for p in cuda_paths if p.exists()]
        if not existing_paths:
            logger.debug("Skipping CUDA path setup: no library paths exist")
            return False

        path_strings = [str(p) for p in existing_paths]

        current_path = os.environ.get('PATH', '')
        new_paths = os.pathsep.join(path_strings + ([current_path] if current_path else []))
        os.environ['PATH'] = new_paths

        triton_cuda_path = nvidia_base_path / 'cuda_runtime'
        current_cuda_path = os.environ.get('CUDA_PATH', '')
        new_cuda_path = os.pathsep.join(
            [str(triton_cuda_path)] + ([current_cuda_path] if current_cuda_path else [])
        )
        os.environ['CUDA_PATH'] = new_cuda_path

        if hasattr(os, 'add_dll_directory'):
            for path in existing_paths:
                try:
                    os.add_dll_directory(str(path))
                except OSError:
                    pass

        logger.info("CUDA paths configured successfully")
        return True

    except Exception as e:
        logger.debug(f"CUDA setup failed: {e}")
        return False
