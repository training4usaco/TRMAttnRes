import numpy as np
import torch
from torch.utils.data import DataLoader

from .base import Benchmark, BenchmarkDataset

WALL, EMPTY, START, END, PATH = 0, 1, 2, 3, 4


def _dihedral_transforms(grid: np.ndarray) -> list[np.ndarray]:
    transforms = []
    g = grid.copy()
    for _ in range(4):
        transforms.append(g.copy())
        transforms.append(np.fliplr(g).copy())
        g = np.rot90(g)
    return transforms


class MazeDataset(BenchmarkDataset):
    """
    `do_augment` is used instead of `augment` to avoid shadowing the
    inherited `augment` method from BenchmarkDataset.
    """

    def __init__(self, inputs: np.ndarray, solutions: np.ndarray, do_augment: bool = True):
        self.samples = []
        for inp, sol in zip(inputs, solutions):
            if do_augment:
                for aug_inp, aug_sol in zip(_dihedral_transforms(inp), _dihedral_transforms(sol)):
                    self.samples.append((aug_inp.flatten(), aug_sol.flatten()))
            else:
                self.samples.append((inp.flatten(), sol.flatten()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    def augment(self, sample):
        x, y = sample
        idx = np.random.randint(8)
        transforms_x = _dihedral_transforms(x.numpy().reshape(30, 30))
        transforms_y = _dihedral_transforms(y.numpy().reshape(30, 30))
        return (
            torch.tensor(transforms_x[idx].flatten(), dtype=torch.long),
            torch.tensor(transforms_y[idx].flatten(), dtype=torch.long),
        )


class MazeBenchmark(Benchmark):
    def __init__(self, cfg):
        self.cfg = cfg
        self._load_data()

    def _load_data(self):
        import os
        train_data = np.load(os.path.join(self.cfg.data_dir, "train.npz"))
        test_data = np.load(os.path.join(self.cfg.data_dir, "test.npz"))

        self.train_dataset = MazeDataset(train_data["inputs"], train_data["solutions"], do_augment=True)
        self.test_dataset = MazeDataset(test_data["inputs"], test_data["solutions"], do_augment=False)

    def get_train_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=self.cfg.num_workers)

    def get_test_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=self.cfg.num_workers)

    def evaluate(self, model, device) -> dict:
        model.eval()
        loader = self.get_test_loader(batch_size=128)
        n_correct = 0
        n_total = 0

        with torch.no_grad():
            for x_tokens, y_tokens in loader:
                x_tokens = x_tokens.to(device)
                y_tokens = y_tokens.to(device)

                logits_list = model(x_tokens)
                preds = logits_list[-1].argmax(-1)
                n_correct += (preds == y_tokens).all(dim=1).sum().item()
                n_total += x_tokens.shape[0]

        return {"accuracy": n_correct / n_total}