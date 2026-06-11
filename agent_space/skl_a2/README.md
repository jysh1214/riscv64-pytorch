# A2: route aten::mm to SKL via dispatcher override (out-of-tree, zero PyTorch changes)

Verified live on pytorch@0a327fa (CPU build): the override took the (aten::mm, CPU)
dispatch slot from torchgen's generated wrapper; eager `a @ b` and compiled
`extern_kernels.mm` both landed in it. See the book, Ch7, card A.

## Files
- `skl_aten_overrides.cpp` — the override. `skl_sgemm()` is a STAND-IN (naive loops);
  for real SKL, delete it, declare the vendor symbol, and link with `-lskl`.
- `driver.py` — build + load + verify (dispatch dump before/after, eager + compiled checks).

## Use
JIT (compile + dlopen in one step):
    python driver.py
    # the built .so is cached (and reused across runs; recompiled only on source change) at:
    #   $TORCH_EXTENSIONS_DIR or ~/.cache/torch_extensions/py<ver>_cpu/skl_aten_overrides/skl_aten_overrides.so
AOT (no compiler needed once the .so exists — verified in a fresh process):
    torch.ops.load_library("~/.cache/torch_extensions/py314_cpu/skl_aten_overrides/skl_aten_overrides.so")
    # for deliberate deployment: set TORCH_EXTENSIONS_DIR / pass build_directory= to load(),
    # or package properly via setup.py + CppExtension and ship that .so

No further API: TORCH_LIBRARY_IMPL is a static initializer — loading IS registering.

## Reproduce (exact commands, as verified on 2026-06-10)

    cd /home/alex/pytorch
    source .venv/bin/activate        # CPU-only torch 2.13.0a0+git0a327fa built in this venv

    # 1) JIT route: compile + load + verify in one step (~60-90s first run, cached after)
    python agent_space/skl_a2/driver.py

    # 2) AOT route: prove a FRESH process needs only load_library (no compiler)
    python - <<'PY'
    import torch
    torch.ops.load_library(
        "/home/alex/.cache/torch_extensions/py314_cpu/skl_aten_overrides/skl_aten_overrides.so")
    print([l for l in torch._C._dispatch_dump("aten::mm").splitlines() if l.startswith("CPU:")][0])
    a, b = torch.randn(4, 3), torch.randn(3, 2)
    print("correct:", torch.allclose(a @ b, (a.unsqueeze(2)*b.unsqueeze(0)).sum(1), atol=1e-5))
    PY

Expected success markers:
  - driver.py:  BEFORE -> "registered at .../build/aten/src/ATen/RegisterCPU_0.cpp:3234"
                Warning "Overriding a previously registered kernel ..."
                AFTER  -> "registered at .../agent_space/skl_a2/skl_aten_overrides.cpp:32"
                "[SKL] sgemm 32x16x8" twice (eager + compiled), both "correct: True"
  - AOT step:   CPU slot at skl_aten_overrides.cpp:32, "[SKL] sgemm 4x3x2", "correct: True"

## Verify it took effect
    print(torch._C._dispatch_dump("aten::mm"))   # CPU slot should point at this file
Also: the dispatcher warns "Overriding a previously registered kernel ..." once.

## Notes
- Override BOTH "mm" (eager) and "mm.out" (extern_kernels.mm path) — done here.
- Never call at::mm/at::matmul inside the override (re-dispatches into yourself).
- No register_fake needed: aten::mm's Meta kernel still serves compile-time shapes.
