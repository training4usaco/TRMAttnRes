from dataclasses import dataclass, field
from typing import Optional

ARC_CONTEXT_LEN = 10_800


@dataclass
class ModelConfig:
    vocab_size: int = 10
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2400
    context_len: int = 81
    n: int = 6
    T: int = 3
    n_sup: int = 16
    use_attention: bool = True
    use_attn_res: bool = False


@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 1e-4
    embed_lr: Optional[float] = None
    weight_decay: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 2000
    total_steps: int = 50_000
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    checkpoint_every: int = 1000
    checkpoint_dir: str = "checkpoints"
    log_every: int = 10
    device: str = "mps"


@dataclass
class SudokuConfig:
    model: ModelConfig = field(default_factory=lambda: ModelConfig(
        vocab_size=10,
        context_len=81,
        n=6,                        # L_cycles=6 for sudoku
        T=3,                        # H_cycles=3
        use_attention=False,        # mlp_t=True in official
    ))
    train: TrainConfig = field(default_factory=lambda: TrainConfig(
        weight_decay=1.0,
        total_steps=50_000,         # was 60k, official uses 50k
        batch_size=32,              # official uses 768, reduced for MPS
    ))
    n_augmentations: int = 1000
    data_dir: str = "data/sudoku"
    num_workers: int = 0            # safest on macOS


@dataclass
class MazeConfig:
    model: ModelConfig = field(default_factory=lambda: ModelConfig(
        vocab_size=5,
        context_len=900,
        n=4,                        # L_cycles=4 for maze (NOT 6)
        T=3,
        use_attention=True,
    ))
    train: TrainConfig = field(default_factory=lambda: TrainConfig(
        weight_decay=1.0,
        total_steps=50_000,
        batch_size=16,              # official uses 128 on 4 GPUs, reduced for MPS
    ))
    n_augmentations: int = 8
    data_dir: str = "data/maze"
    num_workers: int = 0


@dataclass
class ARCConfig:
    model: ModelConfig = field(default_factory=lambda: ModelConfig(
        vocab_size=11,
        context_len=ARC_CONTEXT_LEN,
        n=4,                        # L_cycles=4 for ARC (NOT 6)
        T=3,
        use_attention=True,
    ))
    train: TrainConfig = field(default_factory=lambda: TrainConfig(
        lr=1e-4,
        embed_lr=1e-2,
        weight_decay=0.1,
        total_steps=100_000,
        batch_size=4,               # very long sequences, keep tiny for MPS
    ))
    n_augmentations: int = 1000
    n_test_votes: int = 1000
    version: int = 1
    data_dir: str = "data/arc"
    num_workers: int = 0