# B: SKL as an autotune contestant — aligned with the oneDNN/BLAS architecture

The implementation does NOT live in Python: like oneDNN, the math lives in a C++
extension registered via TORCH_LIBRARY, and PyTorch only holds pointers to it.

## Architecture (the oneDNN analogy)

    oneDNN                                   SKL (this package)
    ------------------------------           ------------------------------
    libdnnl.so          vendor math          libskl.so          (stand-in: naive loop in
                                                                 skl_ops.cpp; swap point marked)
    native/mkldnn/Linear.cpp                 skl_ops.cpp        C++ glue:
      TORCH_LIBRARY_IMPL(mkldnn, ...)          TORCH_LIBRARY(skl) m.def("mm(...)")
                                               TORCH_LIBRARY_IMPL(skl, CPU)  -> skl_mm_cpu
                                               TORCH_LIBRARY_IMPL(skl, Meta) -> shape fn
                                               (Meta impl = the C++ analog of register_fake)
    mkldnn_lowerings.py ExternKernelChoice   skl_b.patch        IN-TREE (2 files, 22 lines):
      + in-tree heuristics                     kernel/mm.py: skl_mm choice + race entry
                                               template_heuristics/aten.py: uid heuristic
    (tests only)                             driver.py          test ONLY: load ext, compile, check

Three registries: (1) c10 dispatcher <- skl_ops.cpp; (2) autotune choices +
extern_kernels namespace <- mm.py patch; (3) template-heuristic registry <- aten.py patch.
Ordering: the extension must be LOADED before the first compile imports the patched
mm.py (its module level references torch.ops.skl.mm). In-tree endgame (USE_SKL build
flag, third_party/skl) would remove even that constraint, exactly like oneDNN.

## Verified results (pytorch@0a327fa, CPU build, 256x256 fp32)
- solo (max_autotune_gemm_backends=""): call() emitted
  `buf0 = extern_kernels.skl_mm(arg0_1, arg1_1)`; correct: True.
- race (default backends): mm 0.7659 ms vs skl_mm 24.9273 ms -> ATen won.
  (The naive-loop stand-in loses ~32x to oneDNN/Eigen — the stopwatch doing its job;
  a real SKL gemm competes on equal terms.) correct: True.

## Why two modes
- solo  = existence proof: SKL CAN be selected and emitted (wiring end-to-end).
- race  = mechanism proof: SKL IS entered and timed; fastest wins per shape.

## Hard-won requirements (2 of 3 found only by running)
1. (reading)  name="skl_mm" in ExternKernelChoice — kernel.__name__ is "mm", collides
   with extern_kernels.mm.
2. (running)  every contestant uid needs a template heuristic or it SILENTLY vanishes
   from the race (NoValidChoicesError when alone) — now part of the in-tree patch.
3. (running)  the heuristic registers against the namespaced uid "aten::skl_mm",
   not the bare name (the registry error log prints failed + available keys).

## Reproduce (exact commands, as verified on 2026-06-10)

    cd /home/alex/pytorch
    source .venv/bin/activate                       # CPU-only torch 2.13.0a0+git0a327fa
    git apply agent_space/skl_b/skl_b.patch         # TEMPORARY in-tree patch (2 files)

    TORCH_LOGS=output_code python agent_space/skl_b/driver.py solo 2>&1 | grep -E 'skl_mm|correct'
    TORCH_LOGS="output_code,+torch._inductor.select_algorithm" \
        python agent_space/skl_b/driver.py race 2>&1 | grep -E ' ms|extern_kernels|correct'

    git checkout -- torch/_inductor/                # REVERT — leave the tree clean

Expected: solo -> "extern_kernels.skl_mm(arg0_1, arg1_1)", correct: True
          race -> "mm ... ms" + "skl_mm ... ms" timings, ATen wins, correct: True
