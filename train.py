import time
import random
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from adam_atan2_pytorch import AdamAtan2

from config import SudokuConfig, MazeConfig, ARCConfig, TrainConfig, ModelConfig
from model.layers import stablemax_cross_entropy
from model.trm import TRM
from model.trm_attnres import TRMAttnRes
from benchmarks.sudoku import SudokuBenchmark
from benchmarks.maze import MazeBenchmark
from benchmarks.arc_agi import ARCBenchmark

import math

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# --- Device ---------------------------------------------------------------

def get_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested in ("mps", "cuda") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --- EMA ------------------------------------------------------------------

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self._backup = {}

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * param.data
                )

    def apply(self):
        self._backup = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self._backup[name])


# --- LR schedule ----------------------------------------------------------

def get_lr(step: int, base_lr: float, warmup_steps: int, total_steps: int = 0) -> float:
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    if total_steps > 0:
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr


# --- Model factory --------------------------------------------------------

def build_model(model_cfg: ModelConfig) -> nn.Module:
    trm = TRM(
        vocab_size=model_cfg.vocab_size,
        d_model=model_cfg.d_model,
        n_heads=model_cfg.n_heads,
        d_ff=model_cfg.d_ff,
        context_len=model_cfg.context_len,
        n=model_cfg.n,
        T=model_cfg.T,
        n_sup=model_cfg.n_sup,
        use_attention=model_cfg.use_attention,
    )
    if model_cfg.use_attn_res:
        model = TRMAttnRes(trm)
    else:
        model = trm

    model.trm.net if hasattr(model, 'trm') else model.net
    inner = model.trm.net if hasattr(model, 'trm') else model.net
    inner = torch.compile(inner)
    if hasattr(model, 'trm'):
        model.trm.net = inner
    else:
        model.net = inner

    return model


# --- Optimizer factory ----------------------------------------------------

def build_optimizer(model: nn.Module, train_cfg: TrainConfig) -> torch.optim.Optimizer:
    if train_cfg.embed_lr is not None:
        embed_params, other_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "embedding" in name:
                embed_params.append(param)
            else:
                other_params.append(param)
        param_groups = [
            {"params": other_params, "lr": train_cfg.lr},
            {"params": embed_params, "lr": train_cfg.embed_lr},
        ]
    else:
        param_groups = [
            {"params": [p for p in model.parameters() if p.requires_grad],
             "lr": train_cfg.lr}
        ]

    return AdamAtan2(
        param_groups,
        betas=(train_cfg.beta1, train_cfg.beta2),
        weight_decay=train_cfg.weight_decay,
    )


# --- Checkpointing --------------------------------------------------------

def save_checkpoint(model, ema, optimizer, step, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "ema_shadow": ema.shadow,
        "optimizer": optimizer.state_dict(),
    }, path)


def load_checkpoint(path, model, ema, optimizer):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    ema.shadow = ckpt["ema_shadow"]
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["step"]


def load_checkpoint_eval(path, model, ema):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    ema.shadow = ckpt["ema_shadow"]
    return ckpt["step"]


# --- Training loop --------------------------------------------------------

def train(benchmark, model_cfg: ModelConfig, train_cfg: TrainConfig, run_name: str, resume_path: str = None):
    device = get_device(train_cfg.device)

    model = build_model(model_cfg).to(device)
    ema = EMA(model, decay=train_cfg.ema_decay)
    optimizer = build_optimizer(model, train_cfg)

    start_step = 0
    if resume_path is not None:
        start_step = load_checkpoint(resume_path, model, ema, optimizer)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        for name in ema.shadow:
            ema.shadow[name] = ema.shadow[name].to(device)
        print(f"Resumed from {resume_path} at step {start_step}")

    train_loader = benchmark.get_train_loader(train_cfg.batch_size)
    train_loader.pin_memory = True

    trm = model.trm if hasattr(model, 'trm') else model

    print(f"Parameters: {model.num_parameters():,}")
    print(f"Training on {device} for {train_cfg.total_steps} steps (starting at {start_step})")

    def infinite_loader(loader):
        while True:
            yield from loader

    data_iter = infinite_loader(train_loader)

    carry_y = None
    carry_z = None
    carry_x = None
    carry_labels = None
    sup_steps = None
    halted = None
    min_halt_steps = None

    running_loss = 0.0
    t0 = time.time()

    model.train()
    for step in range(start_step, train_cfg.total_steps):
        batch_x, batch_y = next(data_iter)
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        B, L = batch_x.shape
        seq_len = L + trm.n_prefix_tokens

        if carry_y is None:
            halted = torch.ones(B, dtype=torch.bool, device=device)
            sup_steps = torch.zeros(B, dtype=torch.long, device=device)
            min_halt_steps = torch.zeros(B, dtype=torch.long, device=device)
            carry_x = batch_x
            carry_labels = batch_y
            carry_y = trm.y_init.expand(B, seq_len, trm.d_model).clone()
            carry_z = trm.z_init.expand(B, seq_len, trm.d_model).clone()

        if halted.any():
            h1 = halted.unsqueeze(-1)                  # (B, 1)
            h3 = halted.unsqueeze(-1).unsqueeze(-1)     # (B, 1, 1)

            carry_x = torch.where(h1, batch_x, carry_x)
            carry_labels = torch.where(h1, batch_y, carry_labels)

            carry_y = torch.where(h3, trm.y_init.expand(B, seq_len, -1), carry_y)
            carry_z = torch.where(h3, trm.z_init.expand(B, seq_len, -1), carry_z)

            sup_steps = torch.where(halted, 0, sup_steps)

            exploring = torch.rand(B, device=device) < 0.1
            rand_min = torch.randint(2, trm.n_sup + 1, (B,), device=device)
            min_halt_steps = torch.where(
                halted & exploring, rand_min,
                torch.where(halted, torch.zeros_like(min_halt_steps), min_halt_steps)
            )

        lr = get_lr(step, train_cfg.lr, train_cfg.warmup_steps, train_cfg.total_steps)
        for pg in optimizer.param_groups:
            if not pg.get("is_embed", False):
                pg["lr"] = lr

        x_emb = trm.embed_input(carry_x)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            y_new, z_new = trm.deep_recursion(x_emb, carry_y, carry_z)
            logits, q = trm.get_output(y_new)

            pred_loss = stablemax_cross_entropy(
                logits.reshape(-1, trm.vocab_size),
                carry_labels.reshape(-1),
                ignore_index=-1,
            )

            with torch.no_grad():
                is_correct = (logits.argmax(-1) == carry_labels).all(dim=1).float()

            halt_loss = F.binary_cross_entropy_with_logits(q[:, 0], is_correct)
            loss = pred_loss + 0.5 * halt_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        ema.update()

        running_loss += loss.item()

        carry_y = y_new.detach()
        carry_z = z_new.detach()
        sup_steps = sup_steps + 1

        with torch.no_grad():
            is_last = sup_steps >= trm.n_sup
            wants_halt = q[:, 0] > 0  # sigmoid > 0.5
            halted = is_last | (wants_halt & (sup_steps >= min_halt_steps))

        if step % train_cfg.log_every == 0 and step > 0:
            elapsed = time.time() - t0
            steps_per_sec = train_cfg.log_every / elapsed
            remaining = (train_cfg.total_steps - step) / steps_per_sec
            avg_steps = sup_steps.float().mean().item()
            halt_frac = halted.float().mean().item()
            print(
                f"Step {step:6d} | "
                f"loss {running_loss / train_cfg.log_every:.4f} | "
                f"avg_sup {avg_steps:.1f} | "
                f"halt% {halt_frac:.2f} | "
                f"{steps_per_sec:.2f} steps/sec | "
                f"eta {remaining / 3600:.1f}h"
            )
            running_loss = 0.0
            t0 = time.time()

        if step % train_cfg.checkpoint_every == 0 and step > 0:
            path = os.path.join(train_cfg.checkpoint_dir, run_name, f"step_{step}.pt")
            save_checkpoint(model, ema, optimizer, step, path)

    path = os.path.join(train_cfg.checkpoint_dir, run_name, "final.pt")
    save_checkpoint(model, ema, optimizer, train_cfg.total_steps, path)
    print("Training complete.")
    return model, ema


# --- Entry point ----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["sudoku", "maze", "arc1", "arc2", "arc3"], required=True)
    parser.add_argument("--use_attn_res", action="store_true")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    if args.benchmark == "sudoku":
        cfg = SudokuConfig()
        benchmark = SudokuBenchmark(cfg)
    elif args.benchmark == "maze":
        cfg = MazeConfig()
        benchmark = MazeBenchmark(cfg)
    elif args.benchmark in ("arc1", "arc2", "arc3"):
        cfg = ARCConfig()
        cfg.version = int(args.benchmark[-1])
        benchmark = ARCBenchmark(cfg)

    cfg.model.use_attn_res = args.use_attn_res
    run_name = args.run_name or f"{args.benchmark}_{'attnres' if args.use_attn_res else 'base'}"
    train(benchmark, cfg.model, cfg.train, run_name, resume_path=args.resume)


if __name__ == "__main__":
    main()
