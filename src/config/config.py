from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

# Project paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments"
CHECKPOINTS_DIR: Path = PROJECT_ROOT / "checkpoints"

EXPERIMENTS_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR.mkdir(exist_ok=True)

# Algorithms we support
ALGOS: List[str] = ["ppo", "sac", "td3"]

# Default seeds to use
SEEDS: List[int] = [0, 10, 20]

# Common hyperparameters across algorithms
COMMON_CFG: Dict[str, Any] = {
    "total_timesteps": 2_000_000,
    "gamma": 0.99,
    "learning_rate": 3e-4,
    "eval_freq": 50_000,
    "n_eval_episodes": 5,
}

# Algorithm-specific hyperparameters
ALGO_CFG: Dict[str, Dict[str, Any]] = {
    "ppo": {
        "n_steps": 4096,
        "batch_size": 256,
        "n_epochs": 10,
        "ent_coef": 0.0,
    },
    "sac": {
        "batch_size": 256,
        "buffer_size": 1_000_000,
        "train_freq": 1,
        "gradient_steps": 1,
        "tau": 0.005,
    },
    "td3": {
        "batch_size": 256,
        "buffer_size": 1_000_000,
        "train_freq": 1,
        "gradient_steps": 1,
        "tau": 0.005,
        "policy_delay": 2,
        "noise_std": 0.1,
    },
}

# Reward shaping configs.
# You can add more later (e.g. "upright", "smooth_gait" etc.)
SHAPING_CONFIGS: Dict[str, Dict[str, Any]] = {
    "none": {},
    # Penalize large torques → encourages smoother, less jerky motion
    "torque_penalty": {
        "torque_coef": 0.01,
    },
}
