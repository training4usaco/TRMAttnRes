import argparse
import torch

from config import SudokuConfig, MazeConfig, ARCConfig
from train import build_model, load_checkpoint_eval, EMA, get_device
from benchmarks.sudoku import SudokuBenchmark
from benchmarks.maze import MazeBenchmark
from benchmarks.arc_agi import ARCBenchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["sudoku", "maze", "arc1", "arc2", "arc3"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--use_attn_res", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = get_device(args.device)            # was: torch.device(... cuda check ...)

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
    model = build_model(cfg.model).to(device)
    ema = EMA(model, decay=cfg.train.ema_decay)

    load_checkpoint_eval(args.checkpoint, model, ema)
    ema.apply()

    metrics = benchmark.evaluate(model, device)
    print(f"Results on {args.benchmark}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    ema.restore()


if __name__ == "__main__":
    main()