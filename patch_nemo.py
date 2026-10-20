import importlib.util
import sys
from pathlib import Path


def _find_package_root(package_name: str) -> Path:
    spec = importlib.util.find_spec(package_name)
    if spec and spec.submodule_search_locations:
        return Path(spec.submodule_search_locations[0])
    raise RuntimeError(f"Cannot locate installed {package_name} package")


NEMO_ROOT = _find_package_root("nemo")

try:
    PEFT_ROOT = _find_package_root("peft")
except RuntimeError:
    PEFT_ROOT = None

PATCHES = []


def patch(file_path, old, new, description):
    if isinstance(old, str):
        old = [old]
    PATCHES.append((file_path, old, new, description))


patch(
    NEMO_ROOT / "core" / "utils" / "cuda_python_utils.py",
    """    buf = b" " * size
    (err,) = nvrtc.nvrtcGetProgramLog(prog, buf)""",
    """    buf = bytearray(size)
    (err,) = nvrtc.nvrtcGetProgramLog(prog, buf)""",
    "cuda_python_utils.py: give NVRTC a writable log buffer instead of the shared b' ' object",
)

patch(
    NEMO_ROOT / "core" / "utils" / "cuda_python_utils.py",
    """    ptx = b" " * ptxSize
    (err,) = nvrtc.nvrtcGetPTX(prog, ptx)
    assert_drv(err)

    ptx = np.char.array(ptx)""",
    """    ptx = bytearray(ptxSize)
    (err,) = nvrtc.nvrtcGetPTX(prog, ptx)
    assert_drv(err)

    ptx = np.char.array(bytes(ptx))""",
    "cuda_python_utils.py: give NVRTC a writable PTX buffer",
)


def apply_patches():
    applied = 0
    skipped = 0
    failed = 0

    for file_path, old_patterns, new, description in PATCHES:
        if not file_path.exists():
            print(f"  SKIP (file not found): {description}")
            print(f"        {file_path}")
            skipped += 1
            continue

        content = file_path.read_text(encoding="utf-8")

        if new in content:
            print(f"  SKIP (already patched): {description}")
            skipped += 1
            continue

        matched_old = None
        for old in old_patterns:
            if old in content:
                matched_old = old
                break

        if matched_old is None:
            print(f"  FAIL (pattern not found): {description}")
            print(f"        {file_path}")
            failed += 1
            continue

        content = content.replace(matched_old, new, 1)
        file_path.write_text(content, encoding="utf-8")
        print(f"  OK: {description}")
        applied += 1

    return applied, skipped, failed


if __name__ == "__main__":
    print(f"NeMo location: {NEMO_ROOT}")
    print(f"peft location: {PEFT_ROOT or 'not installed'}")
    print(f"Patches to apply: {len(PATCHES)}")
    print()

    applied, skipped, failed = apply_patches()

    print()
    print(f"Results: {applied} applied, {skipped} skipped, {failed} failed")

    if failed > 0:
        print("\nSome patches failed — the installed NeMo version may differ from expected.")
        sys.exit(1)
    else:
        print("\nAll patches processed successfully.")
