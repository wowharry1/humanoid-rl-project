from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd


def load_monitor_csv(exp_dir: Path) -> pd.DataFrame:
    """
    Load Monitor CSV produced by gymnasium.wrappers.Monitor.

    The first row is a JSON header, so we skip it.
    """
    monitor_files = list(exp_dir.glob("*.monitor.csv"))
    if not monitor_files:
        raise FileNotFoundError(f"No monitor.csv file found in {exp_dir}")

    # Typically there is a single file; if multiple, pick the first
    monitor_path = monitor_files[0]
    print(f"[INFO] Loading monitor data from: {monitor_path}")
    df = pd.read_csv(monitor_path, skiprows=1)
    # Columns: l (length), r (return), t (time)
    return df


def load_evaluations_npz(exp_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load evaluation history from evaluations.npz (created by EvalCallback).

    Returns:
        timesteps: np.ndarray of shape (N,)
        returns:   np.ndarray of shape (N,)
    """
    eval_path = exp_dir / "evaluations.npz"
    if not eval_path.exists():
        raise FileNotFoundError(f"evaluations.npz not found in {exp_dir}")

    print(f"[INFO] Loading evaluations from: {eval_path}")
    data = np.load(eval_path)
    timesteps = data["timesteps"].squeeze()
    results = data["results"].squeeze()

    # results can be (N, 1) or (N,), so squeeze once
    if results.ndim > 1:
        # assume shape (N, K), average over second axis
        returns = results.mean(axis=1)
    else:
        returns = results

    return timesteps, returns


def summarize_runs(
    algo_name: str,
    seeds: List[int],
    shaping_name: str,
) -> Dict[int, Dict[str, float]]:
    """
    Quick summary: for each seed, load eval curve and compute:
      - best_return
      - final_return
    """
    from eval_utils import build_run_dirs  # local import to avoid cycles

    summary: Dict[int, Dict[str, float]] = {}

    for seed in seeds:
        exp_dir, _ = build_run_dirs(algo_name, seed, shaping_name)
        try:
            timesteps, returns = load_evaluations_npz(exp_dir)
        except FileNotFoundError:
            print(f"[WARN] No evaluations.npz for {algo_name} seed={seed} shaping={shaping_name}")
            continue

        best_ret = float(returns.max())
        final_ret = float(returns[-1])
        summary[seed] = {
            "best_return": best_ret,
            "final_return": final_ret,
        }

    return summary
