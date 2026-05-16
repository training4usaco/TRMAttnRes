import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def stable_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
    logits = logits - logits.amax(dim=-1, keepdim=True)
    return F.cross_entropy(logits, targets, ignore_index=ignore_index)

# class RMSNorm(nn.Module):
#     def __init__(self, d_model: int, eps=1e-8):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(d_model))
#         self.eps = eps
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
#         return x / rms * self.weight

def rms_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.to(dtype)

def calculate_rotary_cis(head_dim: int, max_L: int, base: float = 10_000.0) -> torch.Tensor:
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(max_L, dtype=torch.float32)
    angles = torch.outer(pos, theta)    # (context_len, head_dim // 2): all pairwise products of pos and theta

    return torch.polar(torch.ones_like(angles), angles) # complex values with magnitude 1 and angle angles

def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, rotary_cis: torch.Tensor):
    L = q.shape[2]
    rc = rotary_cis[:L].unsqueeze(0).unsqueeze(0)   # (1, 1, L, head_dim // 2)

    def rot(x):
        B, n_heads, L, head_dim = x.shape
        d_type = x.dtype

        x_pairs = x.float().reshape(B, n_heads, L, head_dim // 2, 2)
        x_complex = torch.view_as_complex(x_pairs)

        return torch.view_as_real(x_complex * rc).flatten(-2).to(d_type)    # flatten back to original dimensions

    return rot(q), rot(k)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        hidden_dim = 2 * d_ff // 3

        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        data = self.up_proj(x)
        return self.down_proj(gate * data)

class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, rotary_cis: torch.Tensor | None = None) -> torch.Tensor:
        B, L, D = x.shape   # (batch size, context len, embed dim)

        k = self.wk(x).view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, n_heads, L, head_dim)
        q = self.wq(x).view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.wv(x).view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        if rotary_cis is not None:
            q, k = apply_rotary_emb(q, k, rotary_cis)

        attn = (q @ k.permute(0, 1, 3, 2)) / self.scale  # (B, n_heads, head_dim, L)
        attn = F.softmax(attn, dim=-1)
        out = attn @ v      # (B, n_heads, L, head_dim)
        out = out.transpose(1, 2).reshape(B, L, D)

        return self.wo(out)

class MLPMixer(nn.Module):
    def __init__(self, d_model: int, context_len: int):
        super().__init__()

        hidden_dim = 2 * context_len

        self.layer1 = nn.Linear(context_len, hidden_dim, bias=False)
        self.layer2 = nn.Linear(hidden_dim, context_len, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)   # (B, D, L)
        x = self.layer2(F.gelu(self.layer1(x)))
        return x.transpose(1, 2)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, context_len: int, use_attention: bool = True):
        super().__init__()

        self.use_attention = use_attention
        # self.norm1 = RMSNorm(d_model)
        # self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

        if self.use_attention:
            self.block = MultiHeadedAttention(d_model, n_heads)
        else:
            self.block = MLPMixer(d_model, context_len)

    def forward(self, x: torch.Tensor, rotary_cis: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_attention:
            x = rms_norm(x + self.block(x, rotary_cis))
        else:
            x = rms_norm(x + self.block(x))
        x = rms_norm(x + self.ffn(x))
        return x
