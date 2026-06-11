# Test driver ONLY — no math, no registrations here (oneDNN-aligned: the
# implementation lives in skl_ops.cpp; Inductor wiring lives in skl_b.patch).
import sys

import torch
import torch._inductor.config as ind
from torch.utils import cpp_extension


ind.force_disable_caches = True
ind.max_autotune = True
ind.max_autotune_gemm = True

mode = sys.argv[1] if len(sys.argv) > 1 else "race"
if mode == "solo":
    ind.max_autotune_gemm_backends = ""  # SKL becomes the only contestant

# Load the extension BEFORE the first compile (the patched mm.py references
# torch.ops.skl.mm at module level). In production: import skl_torch / load_library.
cpp_extension.load(
    name="skl_ops",
    sources=["agent_space/skl_b/skl_ops.cpp"],
    is_python_module=False,
    verbose=False,
)

a, b = torch.randn(256, 256), torch.randn(256, 256)
f = torch.compile(lambda x, w: x @ w)
out = f(a, b)
ref = (a.unsqueeze(2) * b.unsqueeze(0)).sum(1)
print(f"[{mode}] correct:", torch.allclose(out, ref, atol=1e-3))
