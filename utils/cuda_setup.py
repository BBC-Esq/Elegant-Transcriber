import os
import sys
import platform
from pathlib import Path


def setup_cuda_paths():
    if platform.system() != "Windows":
        return

    try:
        venv_base = Path(sys.executable).parent.parent
        nvidia_base_path = venv_base / 'Lib' / 'site-packages' / 'nvidia'

        if not nvidia_base_path.exists():
            return

        cuda_paths = [
            nvidia_base_path / 'cuda_runtime' / 'bin',
            nvidia_base_path / 'cuda_runtime' / 'lib' / 'x64',
            nvidia_base_path / 'cuda_runtime' / 'include',
            nvidia_base_path / 'cublas' / 'bin',
            nvidia_base_path / 'cudnn' / 'bin',
            nvidia_base_path / 'cuda_nvrtc' / 'bin',
            nvidia_base_path / 'cuda_nvcc' / 'bin',
        ]

        paths_to_add = [str(p) for p in cuda_paths]

        current_path = os.environ.get('PATH', '')
        new_paths = os.pathsep.join(
            paths_to_add + ([current_path] if current_path else [])
        )
        os.environ['PATH'] = new_paths

        triton_cuda_path = nvidia_base_path / 'cuda_runtime'
        current_cuda_path = os.environ.get('CUDA_PATH', '')
        new_cuda_path = os.pathsep.join(
            [str(triton_cuda_path)]
            + ([current_cuda_path] if current_cuda_path else [])
        )
        os.environ['CUDA_PATH'] = new_cuda_path

        if hasattr(os, 'add_dll_directory'):
            for path in cuda_paths:
                if path.exists():
                    try:
                        os.add_dll_directory(str(path))
                    except OSError:
                        pass

    except Exception as e:
        print(f"Warning: Could not setup CUDA paths: {e}")
