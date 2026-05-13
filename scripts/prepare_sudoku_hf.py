# scripts/prepare_sudoku_hf.py
import numpy as np
import pandas as pd
import os
from huggingface_hub import hf_hub_download

os.makedirs("data/sudoku", exist_ok=True)

train_path = hf_hub_download(repo_id="sapientinc/sudoku-extreme", filename="train.csv", repo_type="dataset")
test_path = hf_hub_download(repo_id="sapientinc/sudoku-extreme", filename="test.csv", repo_type="dataset")

train_df = pd.read_csv(train_path, dtype=str)
test_df = pd.read_csv(test_path, dtype=str)

def to_arrays(df):
    clues = np.array([[int(c) if c != '.' else 0 for c in row] for row in df["question"]], dtype=np.int64)
    solutions = np.array([[int(c) for c in row] for row in df["answer"]], dtype=np.int64)
    return clues, solutions

train_clues, train_solutions = to_arrays(train_df)
test_clues, test_solutions = to_arrays(test_df)

indices = np.random.default_rng(42).choice(len(train_clues), 1000, replace=False)
np.savez("data/sudoku/train.npz", clues=train_clues[indices], solutions=train_solutions[indices])
np.savez("data/sudoku/test.npz", clues=test_clues, solutions=test_solutions)

print(f"Train: {len(indices)} puzzles")
print(f"Test: {len(test_clues)} puzzles")