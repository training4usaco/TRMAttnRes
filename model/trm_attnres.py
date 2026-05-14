import torch
import torch.nn as nn
import torch.nn.functional as F

from .trm import TRM
from .layers import stable_cross_entropy, rms_norm


class SupervisionStepAttnRes(nn.Module):
    def __init__(self, d_model: int, n_sup: int):
        super().__init__()

        self.d_model = d_model
        self.n_sup = n_sup

        self.wqy = nn.Parameter(torch.zeros(self.n_sup, self.d_model))
        self.wqz = nn.Parameter(torch.zeros(self.n_sup, self.d_model))

    @staticmethod
    def _softmax_attend(query: torch.Tensor,    # (D)
                        sources: list          # N x (B, L, D)
                        ) -> torch.Tensor:
        N = len(sources)
        value = torch.stack(sources, dim=0)
        key = rms_norm(value)

        logits = torch.einsum('d, nbld -> nbl', query, key)  # (N, B, L)
        weights = F.softmax(logits, dim=0)

        out = torch.einsum('nbl, nbld -> bld', weights, value) # weighted sum
        return out

    def forward(self,
                step: int,
                y_init: torch.Tensor,
                z_init: torch.Tensor,
                y_history: list[torch.Tensor],
                z_history:list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        sources_y = [y_init] + y_history
        sources_z = [z_init] + z_history

        y_new = self._softmax_attend(self.wqy[step], sources_y)
        z_new = self._softmax_attend(self.wqz[step], sources_z)

        return y_new, z_new

class TRMAttnRes(nn.Module):
    def __init__(self, trm: TRM):
        super().__init__()

        self.trm = trm
        self.attn_res = SupervisionStepAttnRes(trm.d_model, trm.n_sup)

    def forward(self, x_tokens: torch.Tensor, y_tokens: torch.Tensor | None = None):
        B, L = x_tokens.shape
        x = self.trm.embed_input(x_tokens)

        y_init = self.trm.y_init.expand(B, L, self.trm.d_model)
        z_init = self.trm.z_init.expand(B, L, self.trm.d_model)

        if self.training and y_tokens is not None:
            return self._train_forward(x, y_init, z_init, y_tokens)
        return self._infer_forward(x, y_init, z_init)

    # Note: This never gets used because the backward pass needs to be done each supervision step
    def _train_forward(self, x: torch.Tensor, y_init: torch.Tensor, z_init: torch.Tensor, y_tokens: torch.Tensor):
        y_history = []
        z_history = []

        losses = []
        final_logits = None

        for step in range(self.trm.n_sup):
            y, z = self.attn_res(step, y_init, z_init, y_history, z_history)

            y, z = self.trm.deep_recursion(x, y, z)
            logits, q = self.trm.get_output(y)

            pred_loss = stable_cross_entropy(
                logits.reshape(-1, self.trm.vocab_size),  # 3D -> 2D
                y_tokens.reshape(-1),  # 2D -> 1D
                ignore_index = -1
            )

            with torch.no_grad():
                preds = logits.argmax(-1)   # (B, L)
                is_correct = (preds == y_tokens).all(dim=1).float().unsqueeze(1)    # (B, 1)
            halt_loss = F.binary_cross_entropy_with_logits(q, is_correct)

            losses.append(pred_loss + 0.1 * halt_loss)
            final_logits = logits.detach()

            # Detach before adding into blocks
            y = y.detach()
            z = z.detach()

            y_history.append(y)
            z_history.append(z)

            if q.detach().mean().item() > 0:
                break

        return torch.stack(losses).sum(), final_logits

    def _infer_forward(self, x: torch.Tensor, y_init: torch.Tensor, z_init: torch.Tensor):
        y_history = []
        z_history = []

        logits_list = []

        for step in range(self.trm.n_sup):
            y, z = self.attn_res(step, y_init, z_init, y_history, z_history)

            y, z = self.trm.deep_recursion(x, y, z)
            logits, q = self.trm.get_output(y)
            logits_list.append(logits)

            y = y.detach()
            z = z.detach()

            y_history.append(y)
            z_history.append(z)

        return logits_list

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
