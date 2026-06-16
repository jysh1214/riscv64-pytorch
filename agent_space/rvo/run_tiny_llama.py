"""Run a tiny Llama-architecture model on a minimal PrivateUse1 device.

Default targets device "rvo"; for the live capture against RVO's in-tree twin:
    RVO_BACKEND=torch_openreg RVO_DEV=openreg python run_tiny_llama.py

Every op rides the CPU fallback except the ~10 core ops (empty/copy/view/...),
so the model runs correctly while genuinely dispatching as the chosen device.
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


BACKEND = os.environ.get(
    "RVO_BACKEND", "rvo"
)  # python module that registers the device
DEV = os.environ.get("RVO_DEV", "rvo")  # device string
__import__(BACKEND)  # import side effect = registration


class Cfg:
    vocab, dim, n_layers, n_heads, max_seq, eps = 320, 128, 2, 4, 64, 1e-5


class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


def rope_tables(seq, hd):
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
    f = torch.outer(torch.arange(seq).float(), inv)  # seq, hd/2
    return torch.cos(f), torch.sin(f)


def apply_rope(x, cos, sin):  # x: B,H,T,hd
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h, self.hd = c.n_heads, c.dim // c.n_heads
        self.qkv = nn.Linear(c.dim, 3 * c.dim, bias=False)
        self.o = nn.Linear(c.dim, c.dim, bias=False)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
        mask = torch.full((T, T), float("-inf"), device=x.device).triu(1)
        att = torch.softmax(att + mask, dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, D)
        return self.o(y)


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        h = 4 * c.dim
        self.w1 = nn.Linear(c.dim, h, bias=False)
        self.w3 = nn.Linear(c.dim, h, bias=False)
        self.w2 = nn.Linear(h, c.dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1 = RMSNorm(c.dim, c.eps)
        self.attn = Attn(c)
        self.n2 = RMSNorm(c.dim, c.eps)
        self.mlp = MLP(c)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.mlp(self.n2(x))


class TinyLlama(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.emb = nn.Embedding(c.vocab, c.dim)
        cos, sin = rope_tables(c.max_seq, c.dim // c.n_heads)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layers)])
        self.norm = RMSNorm(c.dim, c.eps)
        self.head = nn.Linear(c.dim, c.vocab, bias=False)

    def forward(self, ids):
        T = ids.shape[1]
        x = self.emb(ids)
        cos, sin = self.cos[:T][None, None], self.sin[:T][None, None]
        for b in self.blocks:
            x = b(x, cos, sin)
        return self.head(self.norm(x))


torch.manual_seed(0)
model = TinyLlama(Cfg).eval().to(DEV)
ids = torch.randint(0, Cfg.vocab, (1, 8)).to(DEV)
with torch.no_grad():
    logits = model(ids)
print(f"forward: logits.device={logits.device}  shape={tuple(logits.shape)}")

cur = ids
for _ in range(5):  # greedy generate 5 tokens
    with torch.no_grad():
        nxt = model(cur)[:, -1].argmax(-1, keepdim=True)
    cur = torch.cat([cur, nxt], dim=1)
print(f"generated ids: {cur.cpu().tolist()[0]}")
print(
    f"OK: TinyLlama ({sum(p.numel() for p in model.parameters()) / 1e3:.0f}K params) "
    f"ran on device '{DEV}' via minimal backend '{BACKEND}'"
)
