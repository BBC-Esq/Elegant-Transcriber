import sys
import subprocess
import time
import tkinter as tk
from tkinter import messagebox


PLATFORM = "linux" if sys.platform.startswith("linux") else "windows"
AUTO_YES = "--yes" in sys.argv or "-y" in sys.argv

torch_urls = {
    "windows": {
        "cp310": {
            "GPU": "https://download.pytorch.org/whl/cu130/torch-2.14.1%2Bcu130-cp310-cp310-win_amd64.whl#sha256=af506e69a93c4769f00a3b1ab06f6e5c3a0c81d8dcea68663331a1b74a21f1fc",
            "CPU": "https://download.pytorch.org/whl/cpu/torch-2.14.1%2Bcpu-cp310-cp310-win_amd64.whl#sha256=bda7043ec45539b1b61538af468e19854b72ac6be8d6fde2f51dfd0c91721568",
        },
        "cp311": {
            "GPU": "https://download.pytorch.org/whl/cu130/torch-2.14.1%2Bcu130-cp311-cp311-win_amd64.whl#sha256=e98711737150b5a6451cbbfdb425abd5d1bea77f4a1a5d29c48d5ec41cd54949",
            "CPU": "https://download.pytorch.org/whl/cpu/torch-2.14.1%2Bcpu-cp311-cp311-win_amd64.whl#sha256=d14b0c8a66345b7ba43adfa86fc20d89087a0a92e440004755d49d2468ddc6e0",
        },
        "cp312": {
            "GPU": "https://download.pytorch.org/whl/cu130/torch-2.14.1%2Bcu130-cp312-cp312-win_amd64.whl#sha256=0a09031e93632d14ef49553acd6211ece3ae9cdc0969a369b8c24e3cc4d15be9",
            "CPU": "https://download.pytorch.org/whl/cpu/torch-2.14.1%2Bcpu-cp312-cp312-win_amd64.whl#sha256=c52adc3bd526ff7fc5bdcc58fc4a76e6214d09951b646ffe5256ebd7ba60cef1",
        },
        "cp313": {
            "GPU": "https://download.pytorch.org/whl/cu130/torch-2.14.1%2Bcu130-cp313-cp313-win_amd64.whl#sha256=f064c797bb497daf2b2b5e32b6d794abef087e3f24e758e6c6f8010bd2698319",
            "CPU": "https://download.pytorch.org/whl/cpu/torch-2.14.1%2Bcpu-cp313-cp313-win_amd64.whl#sha256=4eec5a4fadd7a18f9c75b97fdb374fbb579ce4b1604bf66eeb14dc2201d54def",
        },
    },
    "linux": {
        "cp310": {
            "GPU": "https://download.pytorch.org/whl/cu130/torch-2.14.1%2Bcu130-cp310-cp310-manylinux_2_28_x86_64.whl#sha256=524039bee1fdb7e96f943768335ba964a8ad2a7d5e41525da6dc4897ae27ed16",
            "CPU": "https://download.pytorch.org/whl/cpu/torch-2.14.1%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl#sha256=4ce2a6ae7d302bba56ef7c5c92d9f752728896cfe2626a7f3e5293f879e36650",
        },
        "cp311": {
            "GPU": "https://download.pytorch.org/whl/cu130/torch-2.14.1%2Bcu130-cp311-cp311-manylinux_2_28_x86_64.whl#sha256=9023b6a538714e370fe9e027fc20c5c600bbe3759b79d9a1a6f07e73c40db8d0",
            "CPU": "https://download.pytorch.org/whl/cpu/torch-2.14.1%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl#sha256=5e38154c8896d426a5df58bf2276b603e82184dfae4f54d77dbb212f36e34f37",
        },
        "cp312": {
            "GPU": "https://download.pytorch.org/whl/cu130/torch-2.14.1%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=5359b255ac68c0cabb8293952b38e00f0d1a570815cd87f19b61af702c071410",
            "CPU": "https://download.pytorch.org/whl/cpu/torch-2.14.1%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=5a6363570c753812540a05eb82380e329469cbe668643e88111414c12627711f",
        },
        "cp313": {
            "GPU": "https://download.pytorch.org/whl/cu130/torch-2.14.1%2Bcu130-cp313-cp313-manylinux_2_28_x86_64.whl#sha256=3a78ea0a4d6605c1e351b33ab8aabe44ddb815239cec00cc6f85555d26c0a5a5",
            "CPU": "https://download.pytorch.org/whl/cpu/torch-2.14.1%2Bcpu-cp313-cp313-manylinux_2_28_x86_64.whl#sha256=331fa474e26e428e2e6af1143b2fdab19cee492d7d21897b8899ce1c14755662",
        },
    },
}

torchaudio_urls = {
    "windows": {
        "cp310": {
            "GPU": "https://download.pytorch.org/whl/cu130/torchaudio-2.11.0%2Bcu130-cp310-cp310-win_amd64.whl#sha256=9bbd4470c74172be32d0e11efbcf5e8dc785f7403b8232c07aac575c8d96715f",
            "CPU": "https://download.pytorch.org/whl/cpu/torchaudio-2.11.0%2Bcpu-cp310-cp310-win_amd64.whl#sha256=fae220128cbfe060c663fbb2a1967b88c605f9eb49b535903cca694011662534",
        },
        "cp311": {
            "GPU": "https://download.pytorch.org/whl/cu130/torchaudio-2.11.0%2Bcu130-cp311-cp311-win_amd64.whl#sha256=04aadf807c27df785c6e3463e70bb1a657ed2d1a16b5578114c9fdf7e8c1ac10",
            "CPU": "https://download.pytorch.org/whl/cpu/torchaudio-2.11.0%2Bcpu-cp311-cp311-win_amd64.whl#sha256=abbff04127caf6e16e3a7c3e1ca2739b68899e14926d7cbec4bd8b40f72375d5",
        },
        "cp312": {
            "GPU": "https://download.pytorch.org/whl/cu130/torchaudio-2.11.0%2Bcu130-cp312-cp312-win_amd64.whl#sha256=f74949f9ace1e4a6cf9468bdb3211b9cfa0af6ea348125471ac71c8621d6c77d",
            "CPU": "https://download.pytorch.org/whl/cpu/torchaudio-2.11.0%2Bcpu-cp312-cp312-win_amd64.whl#sha256=95d517bd1a0a28dacd1c37550ced95cab64f3a7a4ef9b8219b41049388a71163",
        },
        "cp313": {
            "GPU": "https://download.pytorch.org/whl/cu130/torchaudio-2.11.0%2Bcu130-cp313-cp313-win_amd64.whl#sha256=ce09a7b144b7982b46c8fe399cf5f91d43dda571e9d6ddba67e928567551f614",
            "CPU": "https://download.pytorch.org/whl/cpu/torchaudio-2.11.0%2Bcpu-cp313-cp313-win_amd64.whl#sha256=fa0c22f58f60c3537b7d28ed2501e0995acb2a65d9af2708f21edaad67186cd8",
        },
    },
    "linux": {
        "cp310": {
            "GPU": "https://download.pytorch.org/whl/cu130/torchaudio-2.11.0%2Bcu130-cp310-cp310-manylinux_2_28_x86_64.whl#sha256=91497b6525569dcd0871991d32bb4beda90775bcbe9050328ea7b8022106e2fa",
            "CPU": "https://download.pytorch.org/whl/cpu/torchaudio-2.11.0%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl#sha256=64c0a3591a5d343640ca5fc5ccdac3ef9d8bcb5928e24c97acfb9f84cb8487a9",
        },
        "cp311": {
            "GPU": "https://download.pytorch.org/whl/cu130/torchaudio-2.11.0%2Bcu130-cp311-cp311-manylinux_2_28_x86_64.whl#sha256=f6e1e3ab31abfa81f5c8801a0eab769d5308514dc644bb39dcf0a9057f364f16",
            "CPU": "https://download.pytorch.org/whl/cpu/torchaudio-2.11.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl#sha256=2ddcb49458aa9783121222c535c0eb9cd54445ec44bb8dd22c9e52cdb730dbc2",
        },
        "cp312": {
            "GPU": "https://download.pytorch.org/whl/cu130/torchaudio-2.11.0%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=3fba988f4301fe13547fe5e99c76d9ae36a27e19ded82eeffed9d2456e12edef",
            "CPU": "https://download.pytorch.org/whl/cpu/torchaudio-2.11.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=2354248848d06a9ae1e7a12165f800f0dda7df60ecac9fca892322b722b922c0",
        },
        "cp313": {
            "GPU": "https://download.pytorch.org/whl/cu130/torchaudio-2.11.0%2Bcu130-cp313-cp313-manylinux_2_28_x86_64.whl#sha256=e9c07cfdab691454092ff12d21dd1407a4bb8ad081d38f222cf6fcf6abcc18c8",
            "CPU": "https://download.pytorch.org/whl/cpu/torchaudio-2.11.0%2Bcpu-cp313-cp313-manylinux_2_28_x86_64.whl#sha256=3c0175d0ed054bf0dc3b154a744b1a127c94291b3f3b7bdd0639b4b238c89445",
        },
    },
}

gpu_libs = [
    "cuda-bindings==13.0.3",
    "nvidia-cuda-nvrtc==13.0.88",
    "nvidia-ml-py",
]

app_libs = [
    "av",
    "psutil",
    "pyside6",
    "peft>=0.20.0",
    "soundfile",
    "sounddevice",
    "pynput",
    "nltk",
    "fastapi",
    "uvicorn[standard]",
    "python-multipart",
    "pytest",
    "requests",
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
    if AUTO_YES:
        print(f"[{title}] {message}")
        if yes_no:
            return True
        return type != "error"
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
        return tkinter_message_box(
            "Confirmation",
            f"Python version {sys.version.split()[0]} was detected, which is compatible.\n\nClick YES to proceed or NO to exit.",
            yes_no=True,
        )
    else:
        tkinter_message_box(
            "Python Version Error",
            "This program requires Python 3.10, 3.11, 3.12 or 3.13\n\nPython versions prior to 3.10 or after 3.13 are not supported.\n\nExiting the installer...",
            type="error",
        )
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


def _run_install_command(command, label, max_retries, delay):
    for attempt in range(max_retries):
        try:
            print(f"\nAttempt {attempt + 1} of {max_retries}: {label}...")
            subprocess.run(command, check=True, text=True, timeout=1800)
            print(f"\033[92mSuccessfully completed: {label}\033[0m")
            return True, attempt + 1
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"Attempt {attempt + 1} failed: {type(e).__name__}")
            stderr = getattr(e, "stderr", None)
            if stderr:
                print(f"Error: {stderr.strip()}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)

    return False, max_retries


def install_libraries(libraries, max_retries=5, delay=3):
    command = [sys.executable, "-m", "uv", "pip", "install", "--link-mode", "copy"] + libraries
    return _run_install_command(
        command, f"Installing {len(libraries)} libraries", max_retries, delay
    )


def run_nemo_patches():
    print("\n\033[92mApplying NeMo compatibility patches:\033[0m")
    try:
        result = subprocess.run(
            [sys.executable, "patch_nemo.py"],
            check=True, text=True, timeout=60,
            capture_output=True,
        )
        print(result.stdout)
        if result.returncode == 0:
            print("\033[92mNeMo patches applied successfully\033[0m")
            return True
        else:
            print("\033[91mNeMo patches failed\033[0m")
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
    message = ("An NVIDIA GPU has been detected.\n\nDo you want to proceed with the installation?"
               if nvidia_gpu_detected else
               "No NVIDIA GPU has been detected. CPU version will be installed.\n\nDo you want to proceed?")

    if not tkinter_message_box("Hardware Detection", message, yes_no=True):
        sys.exit(1)

    print(f"\033[92mPlatform: {PLATFORM} | Python: {python_version} | Hardware: {hardware_type}\033[0m")

    print("\033[92mInstalling uv:\033[0m")
    subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)

    print("\033[92mUpgrading pip, setuptools, and wheel:\033[0m")
    upgrade_pip_setuptools_wheel()

    torch_libs = [
        torch_urls[PLATFORM][python_version][hardware_type],
        torchaudio_urls[PLATFORM][python_version][hardware_type],
    ]
    if hardware_type == "GPU":
        torch_libs += gpu_libs

    print(f"\033[92mStep 1: Installing PyTorch ({hardware_type}):\033[0m")
    success, attempts = install_libraries(torch_libs)
    if not success:
        print(f"\033[91mPyTorch installation failed after {attempts} attempts.\033[0m")
        sys.exit(1)

    print("\n\033[92mStep 2: Installing NeMo toolkit (with all dependencies):\033[0m")
    success_nemo, attempts_nemo = install_libraries(["nemo_toolkit[asr]>=3.0.0,<3.1.0"])
    if not success_nemo:
        print(f"\033[91mNeMo installation failed after {attempts_nemo} attempts.\033[0m")
        sys.exit(1)

    print("\n\033[92mStep 3: Installing app libraries (av, psutil, pyside6, peft, soundfile, sounddevice, pynput, nltk):\033[0m")
    success_app, attempts_app = install_libraries(app_libs)
    if not success_app:
        print(f"\033[91mApp library installation failed after {attempts_app} attempts.\033[0m")
        sys.exit(1)

    patch_success = run_nemo_patches()

    print("\n\033[92mStep 4: Downloading Parakeet TDT 0.6B v2 model:\033[0m")
    try:
        subprocess.run(
            [sys.executable, "download_model.py", "nvidia/parakeet-tdt-0.6b-v2"],
            check=True, text=True, timeout=600,
        )
        model_download_success = True
    except Exception as e:
        print(f"\033[93mModel download failed: {e}. Model will download on first run.\033[0m")
        model_download_success = False

    print("\n----- Installation Summary -----")
    print("\033[92mPyTorch installed successfully.\033[0m")
    print("\033[92mNeMo toolkit installed successfully.\033[0m")
    print("\033[92mApp libraries installed successfully.\033[0m")
    if patch_success:
        print("\033[92mNeMo patches applied successfully.\033[0m")
    else:
        print("\033[93mNeMo patches had issues - check output above.\033[0m")
    if model_download_success:
        print("\033[92mParakeet TDT v2 model downloaded successfully.\033[0m")
    else:
        print("\033[93mModel download had issues - will download on first run.\033[0m")

    end_time = time.time()
    total_time = end_time - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\033[92m\nTotal installation time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}\033[0m")


if __name__ == "__main__":
    main()
