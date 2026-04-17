import json
import os
import numpy as np
import torch
from collections import Counter
from torch.utils.data import DataLoader

from .base import Benchmark, BenchmarkDataset

PAD_TOKEN = 10
MAX_GRID = 30
MAX_GRID_FLAT = MAX_GRID * MAX_GRID  # 900


def _pad_grid(grid: list[list[int]], pad_token: int = PAD_TOKEN) -> np.ndarray:
    out = np.full((MAX_GRID, MAX_GRID), pad_token, dtype=np.int64)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            out[r, c] = val
    return out


def _apply_color_perm(grid: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Permute color tokens 0–9; leave PAD_TOKEN unchanged."""
    out = grid.copy()
    mask = out != PAD_TOKEN
    out[mask] = perm[out[mask]]
    return out


def _apply_dihedral(grid: np.ndarray, k: int) -> np.ndarray:
    g = grid.copy()
    if k >= 4:
        g = np.fliplr(g)
    for _ in range(k % 4):
        g = np.rot90(g)
    return g


def _apply_translation(grid: np.ndarray, dr: int, dc: int) -> np.ndarray:
    return np.roll(np.roll(grid, dr, axis=0), dc, axis=1)


def _augment_task(task: dict, rng: np.random.Generator) -> tuple[dict, np.ndarray]:
    """
    Apply a consistent random augmentation across all grids in a task.
    Returns (augmented_task, color_perm) so the caller can invert the
    color permutation when mapping predictions back to the original color space.
    """
    color_perm = rng.permutation(10).astype(np.int64)
    dihedral_k = int(rng.integers(8))
    dr, dc = rng.integers(-2, 3, size=2)

    def transform(grid_data):
        g = grid_data if isinstance(grid_data, np.ndarray) else _pad_grid(grid_data)
        g = _apply_color_perm(g, color_perm)
        g = _apply_dihedral(g, dihedral_k)
        g = _apply_translation(g, int(dr), int(dc))
        return g

    aug = {"train": [], "test": []}
    for pair in task["train"]:
        aug["train"].append({
            "input": transform(pair["input"]),
            "output": transform(pair["output"]),
        })
    for pair in task["test"]:
        aug["test"].append({
            "input": transform(pair["input"]),
            "output": transform(pair["output"]) if "output" in pair else None,
        })

    return aug, color_perm


def _encode_task(task: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode all demonstration pairs and the test input into flat token arrays.

    Layout: [demo1_in | demo1_out | demo2_in | demo2_out | ... | test_in]
    y_tokens: demo inputs are -1 (ignored), demo outputs are supervised,
              test cells where output is known are supervised, rest are -1.
    """
    grids_x = []
    grids_y = []

    for pair in task["train"]:
        inp = pair["input"] if isinstance(pair["input"], np.ndarray) else _pad_grid(pair["input"])
        out = pair["output"] if isinstance(pair["output"], np.ndarray) else _pad_grid(pair["output"])
        grids_x.append(inp.flatten())
        grids_y.append(np.full(MAX_GRID_FLAT, -1, dtype=np.int64))  # demo inputs not supervised
        grids_x.append(out.flatten())
        grids_y.append(out.flatten())

    for pair in task["test"]:
        inp = pair["input"] if isinstance(pair["input"], np.ndarray) else _pad_grid(pair["input"])
        grids_x.append(inp.flatten())
        if pair.get("output") is not None:
            out = pair["output"] if isinstance(pair["output"], np.ndarray) else _pad_grid(pair["output"])
            grids_y.append(out.flatten())
        else:
            grids_y.append(np.full(MAX_GRID_FLAT, -1, dtype=np.int64))

    return np.concatenate(grids_x).astype(np.int64), np.concatenate(grids_y).astype(np.int64)


class ARCDataset(BenchmarkDataset):
    def __init__(self, tasks: list[dict], n_augmentations: int = 1):
        self.samples = []
        rng = np.random.default_rng(42)

        for task in tasks:
            for _ in range(n_augmentations):
                aug_task, _ = _augment_task(task, rng)  # color_perm not needed at train time
                x, y = _encode_task(aug_task)
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    def augment(self, sample):
        raise NotImplementedError


class ARCBenchmark(Benchmark):
    """
    Supports ARC-AGI versions 1, 2, and 3.

    Expected data layout:
        data/arc/v1/training/   data/arc/v1/evaluation/
        data/arc/v2/training/   data/arc/v2/evaluation/
        data/arc/v3/training/   data/arc/v3/evaluation/  <- placeholder until released
    """

    VERSION_DIRS = {1: "v1", 2: "v2", 3: "v3"}

    def __init__(self, cfg):
        self.cfg = cfg
        self._load_data()

    def _load_tasks(self, split: str) -> list[dict]:
        folder = os.path.join(self.cfg.data_dir, self.VERSION_DIRS[self.cfg.version], split)
        if not os.path.exists(folder):
            raise FileNotFoundError(
                f"ARC-AGI v{self.cfg.version} {split} data not found at {folder}."
                + (" (v3 not yet released as of April 2026)" if self.cfg.version == 3 else "")
            )
        tasks = []
        for fname in sorted(os.listdir(folder)):
            if fname.endswith(".json"):
                with open(os.path.join(folder, fname)) as f:
                    tasks.append(json.load(f))
        return tasks

    def _load_data(self):
        self.train_tasks = self._load_tasks("training")
        self.test_tasks = self._load_tasks("evaluation")
        self.train_dataset = ARCDataset(self.train_tasks, n_augmentations=self.cfg.n_augmentations)
        self.test_dataset = ARCDataset(self.test_tasks, n_augmentations=1)

    def get_train_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=self.cfg.num_workers)

    def get_test_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=self.cfg.num_workers)

    def evaluate(self, model, device) -> dict:
        """
        For each test task, generate n_test_votes augmented predictions, invert each
        prediction's color permutation back to the original color space, then take
        a majority vote. Comparing predictions in augmented space against ground truth
        in original space would produce silently wrong results.
        """
        model.eval()
        rng = np.random.default_rng(0)
        n_correct = 0

        # Ground truth in original (un-augmented) color space
        gt_encoded = [_encode_task(task) for task in self.test_tasks]

        with torch.no_grad():
            for task_idx, task in enumerate(self.test_tasks):
                votes = []

                for _ in range(self.cfg.n_test_votes):
                    aug_task, color_perm = _augment_task(task, rng)
                    inv_color_perm = np.argsort(color_perm).astype(np.int64)

                    x, _ = _encode_task(aug_task)
                    x_tensor = torch.tensor(x, dtype=torch.long).unsqueeze(0).to(device)

                    logits_list = model(x_tensor)
                    pred = logits_list[-1].argmax(-1).squeeze(0).cpu().numpy()

                    # Invert color permutation so the vote is in original color space
                    pred_original = pred.copy()
                    mask = pred_original != PAD_TOKEN
                    pred_original[mask] = inv_color_perm[pred_original[mask]]

                    votes.append(tuple(pred_original.tolist()))

                best_pred = np.array(Counter(votes).most_common(1)[0][0])

                _, gt_y = gt_encoded[task_idx]
                supervised_mask = gt_y != -1
                n_correct += int(np.array_equal(best_pred[supervised_mask], gt_y[supervised_mask]))

        return {"accuracy": n_correct / len(self.test_tasks)}