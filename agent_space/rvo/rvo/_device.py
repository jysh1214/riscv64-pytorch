"""The Python device module behind torch.rvo.*  -- the minimum names PyTorch and
torch.accelerator look up by convention. Each forwards to the C extension."""

import rvo._C as _C


def device_count() -> int:
    return _C.get_device_count()


def current_device() -> int:
    return _C.get_device()


def set_device(device) -> None:
    if int(device) >= 0:
        _C.set_device(int(device))


def is_available() -> bool:
    return device_count() > 0


def is_initialized() -> bool:
    return True


def init() -> None:
    pass


def _is_in_bad_fork() -> bool:  # needed for torch.manual_seed routing
    return False
