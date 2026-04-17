import os
import argparse
import torch
import torch.nn as nn

from config import SudokuConfig, MazeConfig, ARCConfig, TrainConfig, ModelConfig
from model.trm import TRM
from model.trm_attnres import TRMAttnRes
from benchmarks.sudoku import SudokuBenchmark
from benchmarks.maze import MazeBenchmark
from benchmarks.arc_agi import ARCBenchmark

torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(os.cpu_count())

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

def get_lr(step: int, base_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
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
        return TRMAttnRes(trm, block_size=model_cfg.block_size)
    return trm


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
            {"params": other_params, "lr": train_cfg.lr, "is_embed": False},
            {"params": embed_params, "lr": train_cfg.embed_lr, "is_embed": True},
        ]
    else:
        param_groups = [
            {"params": [p for p in model.parameters() if p.requires_grad],
             "lr": train_cfg.lr, "is_embed": False}
        ]

    return torch.optim.AdamW(
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

def train(benchmark, model_cfg: ModelConfig, train_cfg: TrainConfig, run_name: str):
    device = get_device(train_cfg.device)       # was: torch.device(... cuda check ...)

    model = build_model(model_cfg).to(device)
    ema = EMA(model, decay=train_cfg.ema_decay)
    optimizer = build_optimizer(model, train_cfg)

    train_loader = benchmark.get_train_loader(train_cfg.batch_size)

    print(f"Parameters: {model.num_parameters():,}")
    print(f"Training on {device} for {train_cfg.total_steps} steps")

    def infinite_loader(loader):
        while True:
            yield from loader

    data_iter = infinite_loader(train_loader)
    running_loss = 0.0

    for step in range(train_cfg.total_steps):
        model.train()
        x_tokens, y_tokens = next(data_iter)
        x_tokens = x_tokens.to(device)
        y_tokens = y_tokens.to(device)

        import time
        t0 = time.time()

        lr = get_lr(step, train_cfg.lr, train_cfg.warmup_steps)
        for pg in optimizer.param_groups:
            if not pg.get("is_embed", False):
                pg["lr"] = lr

        optimizer.zero_grad()
        loss, _ = model(x_tokens, y_tokens)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        ema.update()

        running_loss += loss.item()

        if step % train_cfg.log_every == 0 and step > 0:
            elapsed = time.time() - t0
            steps_per_sec = train_cfg.log_every / elapsed
            remaining = (train_cfg.total_steps - step) / steps_per_sec
            print(
                f"Step {step:6d} | "
                f"loss {running_loss / train_cfg.log_every:.4f} | "
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
    train(benchmark, cfg.model, cfg.train, run_name)


if __name__ == "__main__":
    main()