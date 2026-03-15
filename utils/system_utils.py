import torch


def get_compute_and_platform_info() -> list[str]:
    devices = ["cpu"]

    if torch.cuda.is_available():
        devices.append('cuda')

    return devices


def has_bfloat16_support() -> bool:
    if not torch.cuda.is_available():
        return False

    capability = torch.cuda.get_device_capability()
    return capability >= (8, 6)
