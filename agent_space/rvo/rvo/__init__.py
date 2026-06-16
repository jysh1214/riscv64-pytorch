"""RVO: a minimal out-of-tree PrivateUse1 device.

Importing this package registers the device as a side effect (the 3-line
incantation below). The C++ static registrars (guard/allocator/hooks/ops) fire
when rvo._C is loaded.
"""

import rvo._C

import torch

from . import _device


torch.utils.rename_privateuse1_backend("rvo")  # 1. claim + name the key
torch._register_device_module("rvo", _device)  # 2. torch.rvo.*
torch.utils.generate_methods_for_privateuse1_backend(
    for_storage=True
)  # 3. tensor.rvo()


def _autoload():
    # entry-point target; the real work is the import side effects above.
    pass
