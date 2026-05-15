import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import TransformerBlock, calculate_rotary_cis, stable_cross_entropy

class TRMNet(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, context_len: int, use_attention: bool = True):
        super().__init__()

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, context_len, use_attention=use_attention) for _ in range(2)
        ])

        head_dim = d_model // n_heads

        self.register_buffer(
            "rotary_cis",
            calculate_rotary_cis(head_dim, max_L=max(context_len, 2048))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rc = self.rotary_cis.to(x.device)

        for block in self.blocks:
            x = block(x, rc)
        return x

class TRM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 512,
                 n_heads: int = 8,
                 d_ff: int = 1024,
                 context_len: int = 81,
                 n: int = 6,
                 T: int = 3,
                 n_sup: int = 16,
                 use_attention: bool = True):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.context_len = context_len
        self.n = n
        self.T = T
        self.n_sup = n_sup
        self.use_attention = use_attention

        self.embed_scale = math.sqrt(d_model)
        embed_init_std = 1.0 / self.embed_scale

        self.input_embeddings = nn.Embedding(vocab_size, d_model)

        self.y_init = nn.Parameter(torch.empty(1, 1, d_model))
        self.z_init = nn.Parameter(torch.empty(1, 1, d_model))

        self.net = TRMNet(d_model, n_heads, d_ff, context_len, use_attention)

        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        self.q_head = nn.Linear(d_model, 1, bias=False)

        nn.init.trunc_normal_(self.y_init, std = 1.0)
        nn.init.trunc_normal_(self.z_init, std = 1.0)
        nn.init.normal_(self.input_embeddings.weight, std=embed_init_std)
        nn.init.normal_(self.output_head.weight, std=0.02)
        nn.init.zeros_(self.q_head.weight)

    def embed_input(self, x_tokens: torch.Tensor) -> torch.Tensor:
        return self.embed_scale * self.input_embeddings(x_tokens)

    def latent_recursion(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for _ in range(self.n):
            z = self.net(x + y + z)
        y = self.net(y + z)
        return y, z

    def deep_recursion(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        with torch.no_grad():
            for j in range(self.T - 1):
                y, z = self.latent_recursion(x, y, z)

        y, z = self.latent_recursion(x, y, z)
        return y, z

    def get_output(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.output_head(y)   # (B, L, vocab_size)
        q = self.q_head(y.mean(dim=1)) # (B, 1)

        return logits, q

    def forward(self, x_tokens: torch.Tensor, y_tokens: torch.Tensor | None = None):
        """
        Training (y provided): returns (total_loss, final logits)
        Inference (y not provided): returns a list of logits (n_sup, B, L, vocab_size)
        """
        B, L = x_tokens.shape

        x = self.embed_input(x_tokens)
        y = self.y_init.expand(B, L, self.d_model)
        z = self.z_init.expand(B, L, self.d_model)

        if self.training and y_tokens is not None:
            return self._train_forward(x, y, z, y_tokens)
        return self._infer_forward(x, y, z)

    def _train_forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, y_tokens: torch.Tensor):
        losses = []
        final_logits = None

        for step in range(self.n_sup):
            y, z = self.deep_recursion(x, y, z)
            logits, q = self.get_output(y)

            pred_loss = stable_cross_entropy(
                logits.reshape(-1, self.vocab_size),    # 3D -> 2D
                y_tokens.reshape(-1),                   # 2D -> 1D
                ignore_index=-1
            )

            with torch.no_grad():
                preds = logits.argmax(-1)   # (B, L)
                is_correct = (preds == y_tokens).all(dim=1).float().unsqueeze(1)    # (B, 1)
            halt_loss = F.binary_cross_entropy_with_logits(q, is_correct)

            losses.append(pred_loss + 0.1 * halt_loss)
            final_logits = logits.detach()

            # Detach computational graph but pass refined state to next supervision step
            y = y.detach()
            z = z.detach()

            if q.detach().mean().item() > 0:
                break

        return torch.stack(losses).sum(), final_logits

    def _infer_forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        logits_list = []

        for step in range(self.n_sup):
            y, z = self.deep_recursion(x, y, z)
            logits, q = self.get_output(y)

            logits_list.append(logits)

            y = y.detach()
            z = z.detach()

        return logits_list

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
