import argparse
from typing import List

from config.config import (
    ALGOS,
    SEEDS,
    SHAPING_CONFIGS,
    COMMON_CFG,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RL agents (PPO / SAC / TD3) on Humanoid-v5."
    )

    parser.add_argument(
        "--algo",
        type=str,
        choices=ALGOS + ["all"],
        default="all",
        help="Which algorithm to train. Use 'all' to train all configured algorithms.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=SEEDS,
        help=f"Random seeds to use. Default: {SEEDS}",
    )
    parser.add_argument(
        "--shaping",
        type=str,
        choices=list(SHAPING_CONFIGS.keys()),
        default="none",
        help="Reward shaping configuration to use.",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=COMMON_CFG["total_timesteps"],
        help=f"Total environment timesteps per run "
             f"(default: {COMMON_CFG['total_timesteps']}).",
    )
    return parser.parse_args()


def resolve_algos(choice: str) -> List[str]:
    if choice == "all":
        return ALGOS
    return [choice]
