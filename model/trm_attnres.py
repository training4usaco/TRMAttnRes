import torch
import torch.nn as nn
import torch.nn.functional as F

from .trm import TRM
from .layers import RMSNorm

class SupervisionStepAttnRes(nn.Module):
    def __init__(self, d_model: int, n_sup: int):
        super().__init__()

        self.d_model = d_model
        self.n_sup = n_sup

        self.wqy = nn.Parameter(torch.zeros(self.n_sup, self.d_model))
        self.wqz = nn.Parameter(torch.zeros(self.n_sup, self.d_model))

        self.key_norm_y = RMSNorm(self.d_model)
        self.key_norm_z = RMSNorm(self.d_model)

    @staticmethod
    def _softmax_attend(query: torch.Tensor,    # (D)
                        sources: list,          # N x (B, L, D)
                        key_norm: RMSNorm) -> torch.Tensor:
        N = len(sources)
        value = torch.stack(sources, dim=0)
        key = key_norm(value)

        logits = torch.einsum('d, nbld -> nbl', query, key)  # (N, B, L)
        weights = F.softmax(logits, dim=0)

        out = torch.einsum('nbl, nbld -> bld', weights, value) # weighted sum
        return out

    def forward(self,
                step: int,
                y_init: torch.Tensor,
                z_init: torch.Tensor,
                block_y: list[torch.Tensor],
                block_z: list[torch.Tensor],
                partial_y: torch.Tensor | None = None,
                partial_z: torch.Tensor | None = None,) -> tuple[torch.Tensor, torch.Tensor]:
        sources_y = [y_init] + list(block_y)
        sources_z = [z_init] + list(block_z)

        if partial_y is not None:
            sources_y.append(partial_y)
            sources_z.append(partial_z)

        y_new = self._softmax_attend(self.wqy[step], sources_y, self.key_norm_y)
        z_new = self._softmax_attend(self.wqz[step], sources_z, self.key_norm_z)

        return y_new, z_new

class TRMAttnRes(nn.Module):
    def __init__(self, trm: TRM, block_size: int = 4):
        super().__init__()

        self.trm = trm
        self.block_size = block_size
        self.attn_res = SupervisionStepAttnRes(trm.d_model, trm.n_sup)

    def forward(self, x_tokens: torch.Tensor, y_tokens: torch.Tensor | None = None):
        B, L = x_tokens.shape
        x = self.trm.input_embeddings(x_tokens)

        y_init = self.trm.y_init.expand(B, L, self.trm.d_model)
        z_init = self.trm.z_init.expand(B, L, self.trm.d_model)

        if self.training and y_tokens is not None:
            return self._train_forward(x, y_init, z_init, y_tokens)
        return self._infer_forward(x, y_init, z_init)

    def _update_block_state(self,
                            step: int,
                            y: torch.Tensor,
                            z: torch.Tensor,
                            block_y: list[torch.Tensor],
                            block_z: list[torch.Tensor],
                            partial_y: torch.Tensor | None = None,
                            partial_z: torch.Tensor | None = None):
        partial_y = y if partial_y is None else partial_y + y
        partial_z = z if partial_z is None else partial_z + z

        if (step + 1) % self.block_size == 0:
            block_y.append(partial_y)
            block_z.append(partial_z)
            partial_y = None
            partial_z = None

        return block_y, block_z, partial_y, partial_z

    def _train_forward(self, x: torch.Tensor, y_init: torch.Tensor, z_init: torch.Tensor, y_tokens: torch.Tensor):
        block_y = []
        block_z = []
        partial_y = None
        partial_z = None

        total_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
        final_logits = None

        for step in range(self.trm.n_sup):
            y, z = self.attn_res(step, y_init, z_init, block_y, block_z, partial_y, partial_z)

            y, z = self.trm.deep_recursion(x, y, z)
            logits, q = self.trm.get_output(y)

            pred_loss = F.cross_entropy(
                logits.reshape(-1, self.trm.vocab_size),  # 3D -> 2D
                y_tokens.reshape(-1),  # 2D -> 1D
                ignore_index = -1
            )

            with torch.no_grad():
                preds = logits.argmax(-1)   # (B, L)
                is_correct = (preds == y_tokens).all(dim=1).float().unsqueeze(1)    # (B, 1)
            halt_loss = F.binary_cross_entropy_with_logits(q, is_correct)

            total_loss = total_loss + pred_loss + 0.1 * halt_loss
            final_logits = logits.detach()

            # Detach before adding into blocks
            y = y.detach()
            z = z.detach()

            block_y, block_z, partial_y, partial_z = self._update_block_state(step, y, z, block_y, block_z, partial_y, partial_z)

            if q.detach().mean().item() > 0:
                break

        return total_loss, final_logits

    def _infer_forward(self, x: torch.Tensor, y_init: torch.Tensor, z_init: torch.Tensor):
        block_y = []
        block_z = []
        partial_y = None
        partial_z = None

        logits_list = []

        for step in range(self.trm.n_sup):
            y, z = self.attn_res(step, y_init, z_init, block_y, block_z, partial_y, partial_z)

            y, z = self.trm.deep_recursion(x, y, z)
            logits, q = self.trm.get_output(y)
            logits_list.append(logits)

            y = y.detach()
            z = z.detach()

            block_y, block_z, partial_y, partial_z = self._update_block_state(step, y, z, block_y, block_z, partial_y, partial_z)

        return logits_list

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
