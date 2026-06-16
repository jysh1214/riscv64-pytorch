"""Run the real TinyLlama-1.1B-Chat on a minimal PrivateUse1 device, end to end.

TinyLlama is an open (ungated) Llama-architecture checkpoint, so this actually
downloads and runs -- no Hugging Face gating. Every op rides the CPU fallback
except the ~10 core memory/layout ops, so it runs correctly at CPU speed while
dispatching as the chosen device.

    pip install transformers safetensors
    # live capture against RVO's in-tree twin:
    RVO_BACKEND=torch_openreg RVO_DEV=openreg python run_tinyllama.py
"""

import os
import time

import torch


BACKEND = os.environ.get("RVO_BACKEND", "rvo")
DEV = os.environ.get("RVO_DEV", "rvo")
NAME = os.environ.get("RVO_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
NEW = int(os.environ.get("RVO_NEW_TOKENS", "24"))
__import__(BACKEND)  # import side effect = device registration

from transformers import AutoModelForCausalLM, AutoTokenizer


tok = AutoTokenizer.from_pretrained(NAME)
# Use eager attention: the bare CPU fallback miscomputes the fused
# scaled_dot_product_attention, but eager attention is plain matmul + softmax,
# which the fallback reproduces bit-exactly. (A real backend would implement SDPA.)
model = AutoModelForCausalLM.from_pretrained(
    NAME, dtype=torch.float32, attn_implementation="eager"
).eval()
print(f"loaded {NAME}: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")

model = model.to(DEV)  # weights now live in device memory
print(f"model on device: {next(model.parameters()).device}")

# TinyLlama-1.1B-Chat uses the Zephyr-style chat format.
prompt = "<|user|>\nWhat is the capital of France?</s>\n<|assistant|>\n"
ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)

# 5-line manual greedy decode. (HF generate() mixes a CPU pad-token tensor with
# device tensors in its post-processing -- a rough edge for brand-new backends --
# so we drive the loop ourselves; it uses only model(...), argmax, cat, .item().)
t = time.time()
gen, eos = ids, tok.eos_token_id
with torch.no_grad():
    for _ in range(NEW):
        nxt = model(gen).logits[:, -1, :].argmax(-1, keepdim=True)
        gen = torch.cat([gen, nxt], dim=1)
        if int(nxt.item()) == eos:
            break
dt = time.time() - t
ntok = gen.shape[1] - ids.shape[1]

print(f"output tensor device: {gen.device}")
print("=== GENERATED TEXT ===")
print(tok.decode(gen[0, ids.shape[1] :], skip_special_tokens=True))
print("=== END ===")
print(
    f"greedy-decoded {ntok} tokens on '{DEV}' in {dt:.1f}s ({ntok / dt:.2f} tok/s, no KV cache)"
)
