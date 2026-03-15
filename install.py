import sys
import subprocess
import time
import tkinter as tk
from tkinter import messagebox


PLATFORM = "linux" if sys.platform.startswith("linux") else "windows"

torch_urls = {
    "windows": {
        "cp310": {
            "GPU": "https://download.pytorch.org/whl/cu128/torch-2.9.0%2Bcu128-cp310-cp310-win_amd64.whl#sha256=397bfff20d46d22692726ca3450f9194a687244fce8fc01b755bf29d715485ee",
            "CPU": "https://download.pytorch.org/whl/cpu/torch-2.9.0%2Bcpu-cp310-cp310-win_amd64.whl#sha256=96f3f7aa4eb9e7fc5af8a722eaf1e5e32e3039dbafe817178d7b90a8566be32d",
        },
        "cp311": {
            "GPU": "https://download.pytorch.org/whl/cu128/torch-2.9.0%2Bcu128-cp311-cp311-win_amd64.whl#sha256=dc6f6c6e7d7eed20c687fc189754a6ea6bf2da9c64eff59fd6753b80ed4bca05",
            "CPU": "https://download.pytorch.org/whl/cpu/torch-2.9.0%2Bcpu-cp311-cp311-win_amd64.whl#sha256=389e1e0b8083fd355f7caf5ba82356b5e01c318998bd575dbf2285a0d8137089",
        },
        "cp312": {
            "GPU": "https://download.pytorch.org/whl/cu128/torch-2.9.0%2Bcu128-cp312-cp312-win_amd64.whl#sha256=c97dc47a1f64745d439dd9471a96d216b728d528011029b4f9ae780e985529e0",
            "CPU": "https://download.pytorch.org/whl/cpu/torch-2.9.0%2Bcpu-cp312-cp312-win_amd64.whl#sha256=e438061b87ec7dd6018fca9f975219889aa0a3f6cdc3ea10dd0ae2bc7f1c47ce",
        },
        "cp313": {
            "GPU": "https://download.pytorch.org/whl/cu128/torch-2.9.0%2Bcu128-cp313-cp313-win_amd64.whl#sha256=9cba9f0fa2e1b70fffdcec1235a1bb727cbff7e7b118ba111b2b7f984b7087e2",
            "CPU": "https://download.pytorch.org/whl/cpu/torch-2.9.0%2Bcpu-cp313-cp313-win_amd64.whl#sha256=728372e3f58c5826445f677746e5311c1935c1a7c59599f73a49ded850e038e8",
        },
    },
    "linux": {
        "cp310": {
            "GPU": "https://download.pytorch.org/whl/cu128/torch-2.9.0%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl#sha256=816540286fce245a8af3904a194a83af9c9292ad7452eb79160b7a3b1cefb7e3",
            "CPU": "https://download.pytorch.org/whl/cpu/torch-2.9.0%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl#sha256=bd2a257e670ede9fc01c6d76dccdc473040913b8e9328169bf177dbdc38e2484",
        },
        "cp311": {
            "GPU": "https://download.pytorch.org/whl/cu128/torch-2.9.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl#sha256=e97c264478c9fc48f91832749d960f1e349aeb214224ebe65fb09435dd64c59a",
            "CPU": "https://download.pytorch.org/whl/cpu/torch-2.9.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl#sha256=add3e93ecc1eeaa6853f6a973ce60ffb3cb14ed2e80f5055e139b09385dce0a7",
        },
        "cp312": {
            "GPU": "https://download.pytorch.org/whl/cu128/torch-2.9.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=87c62d3b95f1a2270bd116dbd47dc515c0b2035076fbb4a03b4365ea289e89c4",
            "CPU": "https://download.pytorch.org/whl/cpu/torch-2.9.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=28f6eb31b08180a5c5e98d5bc14eef6909c9f5a1dbff9632c3e02a8773449349",
        },
        "cp313": {
            "GPU": "https://download.pytorch.org/whl/cu128/torch-2.9.0%2Bcu128-cp313-cp313-manylinux_2_28_x86_64.whl#sha256=97def0087f8ef171b9002ea500baffdd440c7bdd559c23c38bbf8781b67e9364",
            "CPU": "https://download.pytorch.org/whl/cpu/torch-2.9.0%2Bcpu-cp313-cp313-manylinux_2_28_x86_64.whl#sha256=6c9b217584400963d5b4daddb3711ec7a3778eab211e18654fba076cce3b8682",
        },
    },
}

torchaudio_urls = {
    "windows": {
        "cp310": {
            "GPU": "https://download-r2.pytorch.org/whl/cu128/torchaudio-2.9.0%2Bcu128-cp310-cp310-win_amd64.whl#sha256=1bd69bed6b447079b7ea738236af1e4b24f8efd5178d7ba99bec7a2d9a2c9493",
            "CPU": "https://download-r2.pytorch.org/whl/cpu/torchaudio-2.9.0%2Bcpu-cp310-cp310-win_amd64.whl#sha256=fb17c9fad41099337f817c597e616bd1396a3f638af391d447a7736833c351ed",
        },
        "cp311": {
            "GPU": "https://download.pytorch.org/whl/cu128/torchaudio-2.9.0%2Bcu128-cp311-cp311-win_amd64.whl",
            "CPU": "https://download.pytorch.org/whl/cpu/torchaudio-2.9.0%2Bcpu-cp311-cp311-win_amd64.whl",
        },
        "cp312": {
            "GPU": "https://download.pytorch.org/whl/cu128/torchaudio-2.9.0%2Bcu128-cp312-cp312-win_amd64.whl",
            "CPU": "https://download.pytorch.org/whl/cpu/torchaudio-2.9.0%2Bcpu-cp312-cp312-win_amd64.whl",
        },
        "cp313": {
            "GPU": "https://download.pytorch.org/whl/cu128/torchaudio-2.9.0%2Bcu128-cp313-cp313-win_amd64.whl",
            "CPU": "https://download.pytorch.org/whl/cpu/torchaudio-2.9.0%2Bcpu-cp313-cp313-win_amd64.whl",
        },
    },
    "linux": {
        "cp310": {
            "GPU": "https://download-r2.pytorch.org/whl/cu128/torchaudio-2.9.0%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl#sha256=2d924f6b919a25841eedba3a7921b38e3bab8b86b2cf23841e330633dc2ec4df",
            "CPU": "https://download-r2.pytorch.org/whl/cpu/torchaudio-2.9.0%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl#sha256=1b3c522589ae3f09e95eabe9cd49e0a13d06fca41ccba6f9eba1b6b746a9ba45",
        },
        "cp311": {
            "GPU": "https://download-r2.pytorch.org/whl/cu128/torchaudio-2.9.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl#sha256=b0f04dec9117779a6377c5501c86fc069a427af002c85f0846943d684bba2f23",
            "CPU": "https://download-r2.pytorch.org/whl/cpu/torchaudio-2.9.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl#sha256=bd47fa5f76602b30b7b7278be3536899e12e66a26658beb5bac72c49b32f6f65",
        },
        "cp312": {
            "GPU": "https://download-r2.pytorch.org/whl/cu128/torchaudio-2.9.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=ff838b3171be6ef4e4564e2814533816242a6dbea48517b3722785687169376b",
            "CPU": "https://download-r2.pytorch.org/whl/cpu/torchaudio-2.9.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=541c558c90e0781e8ba3d36319a3484acc5da0f502a605af19b0adc2848556a3",
        },
        "cp313": {
            "GPU": "https://download-r2.pytorch.org/whl/cu128/torchaudio-2.9.0%2Bcu128-cp313-cp313-manylinux_2_28_x86_64.whl#sha256=ea76a708a5a94a5c6cbbf3229e584830268b4765fb01ae8230f1b6744253a20f",
            "CPU": "https://download-r2.pytorch.org/whl/cpu/torchaudio-2.9.0%2Bcpu-cp313-cp313-manylinux_2_28_x86_64.whl#sha256=20c0585786f69a182687ee6c3ab76c0ca8e5bf7b209015022e74ee67cda38af4",
        },
    },
}

gpu_libs = [
    "nvidia-cuda-runtime-cu12==12.8.90",
    "nvidia-cublas-cu12==12.8.4.1",
    "nvidia-cudnn-cu12==9.10.2.21",
    "nvidia-ml-py",
]

app_libs = [
    "av",
    "psutil",
    "pyside6",
]

version_overrides = [
    "transformers>=5",
    "huggingface-hub>=1.3.0",
    "fsspec>=2024.12.0",
    "protobuf>=6.33",
]


start_time = time.time()

def enable_ansi_colors():
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        stdout_handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_ulong()
        kernel32.GetConsoleMode(stdout_handle, ctypes.byref(mode))
        mode.value |= 0x0004
        kernel32.SetConsoleMode(stdout_handle, mode)

def has_nvidia_gpu():
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False

python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
hardware_type = "GPU" if has_nvidia_gpu() else "CPU"

def tkinter_message_box(title, message, type="info", yes_no=False):
    root = tk.Tk()
    root.withdraw()
    if yes_no:
        result = messagebox.askyesno(title, message)
    elif type == "error":
        messagebox.showerror(title, message)
        result = False
    else:
        messagebox.showinfo(title, message)
        result = True
    root.destroy()
    return result

def check_python_version_and_confirm():
    major, minor = map(int, sys.version.split()[0].split('.')[:2])
    if major == 3 and minor in [10, 11, 12, 13]:
        return tkinter_message_box("Confirmation", f"Python version {sys.version.split()[0]} was detected, which is compatible.\n\nClick YES to proceed or NO to exit.", yes_no=True)
    else:
        tkinter_message_box("Python Version Error", "This program requires Python 3.10, 3.11, 3.12 or 3.13\n\nPython versions prior to 3.10 or after 3.13 are not supported.\n\nExiting the installer...", type="error")
        return False

def upgrade_pip_setuptools_wheel(max_retries=5, delay=3):
    for package in ["pip", "setuptools", "wheel"]:
        command = [sys.executable, "-m", "pip", "install", "--upgrade", package, "--no-cache-dir"]
        for attempt in range(max_retries):
            try:
                print(f"\nAttempt {attempt + 1} of {max_retries}: Upgrading {package}...")
                subprocess.run(command, check=True, capture_output=True, text=True, timeout=480)
                print(f"\033[92mSuccessfully upgraded {package}\033[0m")
                break
            except subprocess.CalledProcessError as e:
                print(f"Attempt {attempt + 1} failed. Error: {e.stderr.strip()}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)

def install_libraries(libraries, max_retries=5, delay=3):
    command = ["uv", "pip", "install"] + libraries

    for attempt in range(max_retries):
        try:
            print(f"\nAttempt {attempt + 1} of {max_retries}: Installing {len(libraries)} libraries...")
            subprocess.run(command, check=True, text=True, timeout=1800)
            print(f"\033[92mSuccessfully installed all {len(libraries)} libraries\033[0m")
            return True, attempt + 1
        except subprocess.CalledProcessError as e:
            print(f"Attempt {attempt + 1} failed.")
            if e.stderr:
                print(f"Error: {e.stderr.strip()}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)

    return False, max_retries

def install_no_deps_libraries(libraries, max_retries=5, delay=3):
    command = ["uv", "pip", "install", "--no-deps"] + libraries

    for attempt in range(max_retries):
        try:
            print(f"\nAttempt {attempt + 1} of {max_retries}: Installing {len(libraries)} libraries (--no-deps)...")
            subprocess.run(command, check=True, text=True, timeout=1800)
            print(f"\033[92mSuccessfully installed all {len(libraries)} --no-deps libraries\033[0m")
            return True, attempt + 1
        except subprocess.CalledProcessError as e:
            print(f"Attempt {attempt + 1} failed.")
            if e.stderr:
                print(f"Error: {e.stderr.strip()}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)

    return False, max_retries

def run_nemo_patches():
    print("\n\033[92mApplying NeMo compatibility patches:\033[0m")
    try:
        result = subprocess.run(
            [sys.executable, "patch_nemo.py"],
            check=True, text=True, timeout=60,
            capture_output=True
        )
        print(result.stdout)
        if result.returncode == 0:
            print("\033[92mNeMo patches applied successfully\033[0m")
            return True
        else:
            print(f"\033[91mNeMo patches failed\033[0m")
            if result.stderr:
                print(result.stderr)
            return False
    except subprocess.CalledProcessError as e:
        print(f"\033[91mFailed to apply NeMo patches: {e}\033[0m")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return False
    except FileNotFoundError:
        print("\033[91mpatch_nemo.py not found\033[0m")
        return False

def main():
    enable_ansi_colors()

    if not check_python_version_and_confirm():
        sys.exit(1)

    nvidia_gpu_detected = has_nvidia_gpu()
    message = "An NVIDIA GPU has been detected.\n\nDo you want to proceed with the installation?" if nvidia_gpu_detected else \
              "No NVIDIA GPU has been detected. CPU version will be installed.\n\nDo you want to proceed?"

    if not tkinter_message_box("Hardware Detection", message, yes_no=True):
        sys.exit(1)

    print(f"\033[92mPlatform: {PLATFORM} | Python: {python_version} | Hardware: {hardware_type}\033[0m")

    print("\033[92mInstalling uv:\033[0m")
    subprocess.run(["pip", "install", "uv"], check=True)

    print("\033[92mUpgrading pip, setuptools, and wheel:\033[0m")
    upgrade_pip_setuptools_wheel()

    torch_libs = [
        torch_urls[PLATFORM][python_version][hardware_type],
        torchaudio_urls[PLATFORM][python_version][hardware_type],
    ]
    if hardware_type == "GPU":
        torch_libs += gpu_libs

    print(f"\033[92mStep 1: Installing PyTorch + torchaudio ({hardware_type}):\033[0m")
    success, attempts = install_libraries(torch_libs)
    if not success:
        print(f"\033[91mPyTorch installation failed after {attempts} attempts.\033[0m")
        sys.exit(1)

    print(f"\n\033[92mStep 2: Installing NeMo toolkit (with all dependencies):\033[0m")
    success_nemo, attempts_nemo = install_libraries(["nemo_toolkit[asr]"])
    if not success_nemo:
        print(f"\033[91mNeMo installation failed after {attempts_nemo} attempts.\033[0m")
        sys.exit(1)

    print(f"\n\033[92mStep 3: Upgrading transformers, huggingface-hub, fsspec, protobuf (--no-deps):\033[0m")
    success_overrides, attempts_overrides = install_no_deps_libraries(version_overrides)
    if not success_overrides:
        print(f"\033[91mVersion overrides failed after {attempts_overrides} attempts.\033[0m")
        sys.exit(1)

    print(f"\n\033[92mStep 4: Installing app libraries (av, psutil, pyside6):\033[0m")
    success_app, attempts_app = install_libraries(app_libs)
    if not success_app:
        print(f"\033[91mApp library installation failed after {attempts_app} attempts.\033[0m")
        sys.exit(1)

    patch_success = run_nemo_patches()

    print("\n----- Installation Summary -----")
    print(f"\033[92mPyTorch + torchaudio installed successfully.\033[0m")
    print(f"\033[92mNeMo toolkit installed successfully.\033[0m")
    print(f"\033[92mVersion overrides applied (transformers 5+).\033[0m")
    print(f"\033[92mApp libraries installed successfully.\033[0m")
    if patch_success:
        print(f"\033[92mNeMo patches applied successfully.\033[0m")
    else:
        print(f"\033[93mNeMo patches had issues — check output above.\033[0m")

    end_time = time.time()
    total_time = end_time - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\033[92m\nTotal installation time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}\033[0m")

if __name__ == "__main__":
    main()
