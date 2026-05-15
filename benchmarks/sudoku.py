import numpy as np
import torch
from torch.utils.data import DataLoader

from .base import Benchmark, BenchmarkDataset


def _generate_sudoku_transform():
    """
    Generate parameters for a valid Sudoku-preserving transformation.
    Keeping generation separate from application ensures the same transform
    can be applied consistently to both the clue and solution grids.
    """
    color_perm = np.array([0] + list(np.random.permutation(9) + 1))  # 0 stays 0 (empty cell)
    band_order = np.random.permutation(3)
    stack_order = np.random.permutation(3)
    row_perms = [np.random.permutation(3) for _ in range(3)]
    col_perms = [np.random.permutation(3) for _ in range(3)]
    return color_perm, band_order, stack_order, row_perms, col_perms


def _apply_sudoku_transform(grid, color_perm, band_order, stack_order, row_perms, col_perms):
    """
    Apply a pre-generated Sudoku-preserving transformation to a flat 81-element grid.

    Note: `[band * 3 + np.arange(3) for band in band_order]` groups all rows within
    a band together before moving to the next band, which is the correct formulation.
    The incorrect `[band_order * 3 + i for i in range(3)]` interleaves rows from
    different bands and produces an invalid Sudoku transformation.
    """
    g = grid.reshape(9, 9).copy()

    # Permute digit labels (0 maps to 0, preserving empty cells)
    g = color_perm[g]

    # Permute bands (groups of 3 rows) and stacks (groups of 3 cols)
    rows = np.concatenate([b * 3 + np.arange(3) for b in band_order])
    cols = np.concatenate([s * 3 + np.arange(3) for s in stack_order])
    g = g[rows][:, cols]

    # Permute rows within each band and cols within each stack
    for b in range(3):
        g[b*3:(b+1)*3] = g[b*3:(b+1)*3][row_perms[b]]
    for s in range(3):
        g[:, s*3:(s+1)*3] = g[:, s*3:(s+1)*3][:, col_perms[s]]

    return g.flatten()


class SudokuDataset(BenchmarkDataset):
    """
    Augmentation is applied on-the-fly in __getitem__ so that each epoch
    sees a fresh random transformation, rather than pre-computing a fixed
    augmented set that would consume excessive memory.
    """

    def __init__(self, clues, solutions, do_augment = True, n_augmentations = 1000):
        self.clues = clues
        self.solutions = solutions
        self.do_augment = do_augment
        self.n_augmentations = n_augmentations if do_augment else 1

    def __len__(self):
        return len(self.clues) * self.n_augmentations

    def __getitem__(self, idx):
        real_idx = idx % len(self.clues)
        clue = self.clues[real_idx]
        sol = self.solutions[real_idx]

        if self.do_augment:
            params = _generate_sudoku_transform()
            clue = _apply_sudoku_transform(clue, *params)
            sol = _apply_sudoku_transform(sol, *params)

        return torch.tensor(clue, dtype=torch.long), torch.tensor(sol, dtype=torch.long)

    def augment(self, sample):
        clue, sol = sample
        params = _generate_sudoku_transform()
        aug_clue = _apply_sudoku_transform(clue.numpy(), *params)
        aug_sol = _apply_sudoku_transform(sol.numpy(), *params)
        return torch.tensor(aug_clue, dtype=torch.long), torch.tensor(aug_sol, dtype=torch.long)


class SudokuBenchmark(Benchmark):
    def __init__(self, cfg):
        self.cfg = cfg
        self._load_data()

    def _load_data(self):
        import os
        train_data = np.load(os.path.join(self.cfg.data_dir, "train.npz"))
        test_data = np.load(os.path.join(self.cfg.data_dir, "test.npz"))

        self.train_dataset = SudokuDataset(
            train_data["clues"], train_data["solutions"], do_augment=True
        )
        self.test_dataset = SudokuDataset(
            test_data["clues"], test_data["solutions"], do_augment=False
        )

    def get_train_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=self.cfg.num_workers, pin_memory=True)

    def get_test_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=self.cfg.num_workers, pin_memory=True)

    def evaluate(self, model, device) -> dict:
        model.eval()
        loader = self.get_test_loader(batch_size=512)
        n_correct = 0
        n_total = 0
        n_batches = len(loader)

        with torch.no_grad():
            for i, (x_tokens, y_tokens) in enumerate(loader):
                x_tokens = x_tokens.to(device)
                y_tokens = y_tokens.to(device)

                logits_list = model(x_tokens)
                preds = logits_list[-1].argmax(-1)
                n_correct += (preds == y_tokens).all(dim=1).sum().item()
                n_total += x_tokens.shape[0]

                if (i + 1) % 10 == 0 or (i + 1) == n_batches:
                    print(f"  [{i+1}/{n_batches}] {n_total} samples, running acc {n_correct/n_total:.4f}")

        return {"accuracy": n_correct / n_total}
