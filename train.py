import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import SudokuConfig, MazeConfig, ARCConfig, TrainConfig, ModelConfig
from model.layers import stable_cross_entropy
from model.trm import TRM
from model.trm_attnres import TRMAttnRes
from benchmarks.sudoku import SudokuBenchmark
from benchmarks.maze import MazeBenchmark
from benchmarks.arc_agi import ARCBenchmark

try:
    _n_cpus = len(os.sched_getaffinity(0))  # actual CPUs allocated to this process
except AttributeError:
    _n_cpus = os.cpu_count()
torch.set_num_threads(_n_cpus)

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
        return TRMAttnRes(trm)
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


# --- NaN diagnostics ------------------------------------------------------

def _scan_params_for_nan(model: nn.Module) -> list[str]:
    """Return names of parameters/gradients that contain NaN or Inf."""
    issues = []
    for name, param in model.named_parameters():
        if torch.isnan(param.data).any() or torch.isinf(param.data).any():
            issues.append(f"  PARAM {name}: nan={torch.isnan(param.data).sum().item()} inf={torch.isinf(param.data).sum().item()}")
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                issues.append(f"  GRAD  {name}: nan={torch.isnan(param.grad).sum().item()} inf={torch.isinf(param.grad).sum().item()}")
    return issues


def _find_first_nan_module(model: nn.Module, x_tokens: torch.Tensor, y_tokens: torch.Tensor) -> dict:
    """Register forward hooks and run a no-grad pass to find the first module that outputs NaN."""
    first_nan: dict = {}
    hooks = []

    def make_hook(name):
        def hook(module, inp, output):
            if first_nan:          # already found one — skip noise
                return
            out = output[0] if isinstance(output, tuple) else output
            if not isinstance(out, torch.Tensor):
                return
            if torch.isnan(out).any() or torch.isinf(out).any():
                inp_stats = []
                for t in inp:
                    if isinstance(t, torch.Tensor):
                        inp_stats.append(f"min={t.min():.3g} max={t.max():.3g} nan={torch.isnan(t).sum().item()}")
                first_nan['module'] = name
                first_nan['output_shape'] = tuple(out.shape)
                first_nan['nan_frac'] = torch.isnan(out).float().mean().item()
                first_nan['inf_frac'] = torch.isinf(out).float().mean().item()
                first_nan['input_stats'] = inp_stats
        return hook

    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            model(x_tokens, y_tokens)
    finally:
        for h in hooks:
            h.remove()

    return first_nan


def _debug_nan(model: nn.Module, x_tokens: torch.Tensor, y_tokens: torch.Tensor,
               loss: torch.Tensor, step: int) -> None:
    print(f"\n{'='*60}")
    print(f"[NaN DEBUG] Step {step} — loss={loss.item()}")

    # 1. Param / grad scan
    issues = _scan_params_for_nan(model)
    if issues:
        print("[NaN DEBUG] NaN/Inf in parameters or gradients:")
        for line in issues:
            print(line)
    else:
        print("[NaN DEBUG] No NaN/Inf in parameters or gradients.")

    # 2. Find first NaN module
    model.eval()
    nan_info = _find_first_nan_module(model, x_tokens, y_tokens)
    model.train()
    if nan_info:
        print(f"[NaN DEBUG] First NaN/Inf output at module: {nan_info['module']}")
        print(f"            output shape : {nan_info['output_shape']}")
        print(f"            nan fraction : {nan_info['nan_frac']:.4f}")
        print(f"            inf fraction : {nan_info['inf_frac']:.4f}")
        print(f"            input stats  : {nan_info['input_stats']}")
    else:
        print("[NaN DEBUG] No NaN/Inf found in no-grad eval pass (may be training-specific).")
    print('='*60 + '\n')


# --- Training loop --------------------------------------------------------

def train(benchmark, model_cfg: ModelConfig, train_cfg: TrainConfig, run_name: str):
    device = get_device(train_cfg.device)

    model = build_model(model_cfg).to(device)
    ema = EMA(model, decay=train_cfg.ema_decay)
    optimizer = build_optimizer(model, train_cfg)

    train_loader = benchmark.get_train_loader(train_cfg.batch_size)

    print(f"Parameters: {model.num_parameters():,}")
    print(f"Training on {device} for {train_cfg.total_steps} steps")

    def infinite_loader(loader):
        while True:
            yield from loader

    import time
    data_iter = infinite_loader(train_loader)
    running_loss = 0.0
    t0 = time.time()

    for step in range(train_cfg.total_steps):
        model.train()
        x_tokens, y_tokens = next(data_iter)
        x_tokens = x_tokens.to(device)
        y_tokens = y_tokens.to(device)

        lr = get_lr(step, train_cfg.lr, train_cfg.warmup_steps)
        for pg in optimizer.param_groups:
            if not pg.get("is_embed", False):
                pg["lr"] = lr

        # --- Deep supervision: backward + step at EACH supervision step ---
        step_loss = _deep_supervision_step(model, optimizer, x_tokens, y_tokens, train_cfg)

        ema.update()
        running_loss += step_loss

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


def _deep_supervision_step(model, optimizer, x_tokens, y_tokens, train_cfg):
    B, L = x_tokens.shape
    device_type = x_tokens.device.type

    if hasattr(model, 'trm'):
        trm = model.trm
    else:
        trm = model

    y = trm.y_init.expand(B, L, trm.d_model)
    z = trm.z_init.expand(B, L, trm.d_model)
    history_y, history_z = [], []

    total_loss_value = 0.0
    use_amp = device_type == "cuda"

    for sup_step in range(trm.n_sup):
        if hasattr(model, 'attn_res'):
            y_init = trm.y_init.expand(B, L, trm.d_model)
            z_init = trm.z_init.expand(B, L, trm.d_model)
            y, z = model.attn_res(sup_step, y_init, z_init, history_y, history_z)

        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=use_amp):
            x = trm.embed_input(x_tokens)
            y, z = trm.deep_recursion(x, y, z)
            logits, q = trm.get_output(y)

            pred_loss = stable_cross_entropy(
                logits.reshape(-1, trm.vocab_size),
                y_tokens.reshape(-1),
                ignore_index=-1,
            )
            with torch.no_grad():
                preds = logits.argmax(-1)
                is_correct = (preds == y_tokens).all(dim=1).float().unsqueeze(1)
            halt_loss = F.binary_cross_entropy_with_logits(q.float(), is_correct)
            loss = pred_loss + 0.1 * halt_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        total_loss_value += loss.item()

        y = y.detach()
        z = z.detach()

        if hasattr(model, 'attn_res'):
            history_y.append(y)
            history_z.append(z)

        if q.detach().mean().item() > 0:
            break

    return total_loss_value


# --- Entry point ----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["sudoku", "maze", "arc1", "arc2", "arc3"], required=True)
    parser.add_argument("--use_attn_res", action="store_true")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, help="Override device (cuda, mps, cpu)")
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
    if args.device is not None:
        cfg.train.device = args.device
    run_name = args.run_name or f"{args.benchmark}_{'attnres' if args.use_attn_res else 'base'}"
    train(benchmark, cfg.model, cfg.train, run_name)


if __name__ == "__main__":
    main()