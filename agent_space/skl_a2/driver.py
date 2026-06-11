import torch
import torch._inductor.config as ind
from torch.utils import cpp_extension


ind.force_disable_caches = True


def cpu_slot():
    return [
        l
        for l in torch._C._dispatch_dump("aten::mm").splitlines()
        if l.startswith("CPU")
    ]


print("BEFORE:", cpu_slot())
cpp_extension.load(
    name="skl_aten_overrides",
    sources=["agent_space/skl_a2/skl_aten_overrides.cpp"],
    is_python_module=False,
    verbose=False,
)
print("AFTER: ", cpu_slot())

a, b = torch.randn(32, 16), torch.randn(16, 8)
ref = (a.unsqueeze(2) * b.unsqueeze(0)).sum(1)  # mm-free reference
print("eager a@b correct:", torch.allclose(a @ b, ref, atol=1e-4))

f = torch.compile(lambda x, w: torch.relu(x @ w))
print("compiled correct:", torch.allclose(f(a, b), torch.relu(ref), atol=1e-4))
