import argparse
import numpy as np
import pandas as pd
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to sudoku.csv")
    parser.add_argument("--output_dir", type=str, default="data/sudoku")
    parser.add_argument("--n_train", type=int, default=1000)
    parser.add_argument("--max_clues", type=int, default=23, help="Max given clues to keep (<=23 = extreme difficulty)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input, dtype=str)

    # Filter for extreme difficulty: puzzles with <= args.max_clues given digits
    clue_counts = df["quizzes"].apply(lambda s: sum(c != '0' for c in s))
    df = df[clue_counts <= args.max_clues].reset_index(drop=True)
    print(f"Kept {len(df)} extreme-difficulty puzzles (≤{args.max_clues} clues)")

    clues = np.array([[int(c) for c in row] for row in df["quizzes"]], dtype=np.int64)
    solutions = np.array([[int(c) for c in row] for row in df["solutions"]], dtype=np.int64)

    n_test = min(len(clues) - args.n_train, args.n_train // 10)

    np.savez(os.path.join(args.output_dir, "train.npz"),
             clues=clues[:args.n_train],
             solutions=solutions[:args.n_train])
    np.savez(os.path.join(args.output_dir, "test.npz"),
             clues=clues[args.n_train:args.n_train + n_test],
             solutions=solutions[args.n_train:args.n_train + n_test])

    print(f"Saved {args.n_train} train and {n_test} test examples to {args.output_dir}")


if __name__ == "__main__":
    main()