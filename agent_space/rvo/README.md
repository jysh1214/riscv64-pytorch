# RVO — a minimal PrivateUse1 device, and TinyLlama on it

Companion package for **Chapter 8** of `books/device-registration.html`.
RVO ("Reference Virtual Octa", a made-up accelerator) is the *smallest backend that
works*: its "device memory" is just host memory, so the ~10 core operators are
one-liners and every real model op (matmul, attention, RMSNorm, softmax) rides the
CPU fallback. The model runs **correctly** — at CPU speed — while genuinely
dispatching as `device="rvo"`.

## What actually ran (captured)

`run_tinyllama.py` ran the **real, ungated TinyLlama-1.1B-Chat (1.10B params)** end to
end on the built **openreg** backend (RVO's in-tree twin), at the book's pinned commit:

    loaded TinyLlama/TinyLlama-1.1B-Chat-v1.0: 1.10B params
    model on device: openreg:0
    output tensor device: openreg:0
    === GENERATED TEXT ===
    The capital of France is Paris.
    === END ===
    greedy-decoded 8 tokens on 'openreg' in 77.7s (0.10 tok/s, no KV cache)

That answer is bit-for-bit the plain-CPU answer (`RVO_DEV=cpu`) — see
`tinyllama_openreg.log`.

## The one caveat: SDPA

An op-by-op probe (`dispatch_probe.log`-style) shows every op is bit-exact on openreg
**except** the fused `scaled_dot_product_attention` (maxdiff ~2.6 vs CPU). So the model
is loaded with `attn_implementation="eager"` (plain matmul + softmax, which the fallback
reproduces exactly). A real backend would implement/fix SDPA. With default (fused) SDPA
the model still *runs* but emits garbage — a good reminder that "runs" and "correct" are
different milestones.

## What is real vs. stand-in

- **Real:** every API call is the genuine PrivateUse1 API; the TinyLlama run above is a
  real end-to-end execution on the openreg twin at this commit.
- **Stand-in:** `csrc/rvo.cpp` is a *reading skeleton* (lift the complete, compilable
  bodies from openreg's `csrc/`). To get a real `rvo` you copy openreg's `csrc/` and
  rename.

## Run it yourself

    pip install transformers safetensors
    cd test/cpp_extensions/open_registration_extension/torch_openreg
    python -m pip install -e . --no-build-isolation     # builds openreg = RVO's twin
    cd -                                                 # back to repo root
    RVO_BACKEND=torch_openreg RVO_DEV=openreg python agent_space/rvo/run_tinyllama.py
    # plain-CPU reference: RVO_BACKEND=torch RVO_DEV=cpu python agent_space/rvo/run_tinyllama.py

## Files

    rvo/__init__.py        the 3-line registration + minimal device module (real)
    rvo/_device.py         device_count / current_device / set_device / is_available
    csrc/rvo.cpp           guard + allocator + hooks + ~10 ops + CPU fallback (skeleton, anchored)
    setup.py               build the native lib + _C extension + autoload entry point
    run_tinyllama.py       real TinyLlama-1.1B end to end (the live proof)
    run_tiny_llama.py      tiny from-scratch Llama-arch model, instant smoke test (no download)
    *.log                  captured outputs
