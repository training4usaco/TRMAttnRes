import argparse
import numpy as np
import pandas as pd
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to sudoku.csv")
    parser.add_argument("--output_dir", type=str, default="data/sudoku")
    parser.add_argument("--n_train", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input, dtype=str)
    clues = np.array([[int(c) for c in row] for row in df["quizzes"]], dtype=np.int64)
    solutions = np.array([[int(c) for c in row] for row in df["solutions"]], dtype=np.int64)

    np.savez(os.path.join(args.output_dir, "train.npz"),
             clues=clues[:args.n_train],
             solutions=solutions[:args.n_train])
    np.savez(os.path.join(args.output_dir, "test.npz"),
             clues=clues[args.n_train:],
             solutions=solutions[args.n_train:])

    print(f"Saved {args.n_train} train and {len(clues) - args.n_train} test examples to {args.output_dir}")


if __name__ == "__main__":
    main()