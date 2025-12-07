from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Callable

import csv

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config.config import COMMON_CFG


class ShapedHumanoidEnv(gym.Wrapper):
    """
    Simple reward-shaping wrapper for Humanoid-v5.

    Currently implements:
      - torque penalty: penalize large squared torques (smooths motion)

    You can extend this later (e.g. torso uprightness, head stability, etc.)
    by extracting the right features from the observation vector.
    """

    def __init__(self, env: gym.Env, shaping_cfg: Dict[str, Any]):
        super().__init__(env)
        self.shaping_cfg = shaping_cfg

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        shaped_reward = reward

        # 1) Torque penalty shaping
        torque_coef = float(self.shaping_cfg.get("torque_coef", 0.0))
        if torque_coef != 0.0:
            # action is typically a numpy array or list of floats
            a = np.asarray(action, dtype=np.float32)
            torque_cost = float((a ** 2).sum())
            shaped_reward += -torque_coef * torque_cost

        # Keep track for debugging / analysis
        info = dict(info)  # copy to avoid mutating underlying dict
        info["original_reward"] = reward
        info["shaped_reward"] = shaped_reward

        return obs, shaped_reward, terminated, truncated, info


class CSVLogger(gym.Wrapper):
    """
    Minimal CSV logger that mimics the old Monitor-style episode logging.

    It expects that a previous wrapper (RecordEpisodeStatistics) has already
    added info["episode"]["r"] (return) and info["episode"]["l"] (length).
    """

    def __init__(self, env: gym.Env, log_file: Path):
        super().__init__(env)
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Open in append mode; create file if it doesn't exist
        self._file = self.log_file.open("a", newline="")
        self._writer = csv.writer(self._file)

        # If file is new/empty, write header
        if self.log_file.stat().st_size == 0:
            self._writer.writerow(["episode", "length", "return"])

        self._episode_idx = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # After RecordEpisodeStatistics, Gymnasium puts episode info in info["episode"]
        if "episode" in info:
            ep_info = info["episode"]
            self._episode_idx += 1
            ep_len = ep_info.get("l", 0)
            ep_ret = ep_info.get("r", 0.0)
            self._writer.writerow([self._episode_idx, ep_len, ep_ret])
            self._file.flush()

        return obs, reward, terminated, truncated, info

    def close(self):
        try:
            self._file.close()
        except Exception:
            pass
        return super().close()


def _make_single_env(
    log_dir: Path,
    seed: int,
    shaping_cfg: Optional[Dict[str, Any]] = None,
    training: bool = True,
) -> Callable[[], gym.Env]:
    """
    Factory that creates a single Humanoid-v5 env with optional reward shaping.

    If training=True:
      - wraps with RecordEpisodeStatistics
      - wraps with CSVLogger → writes monitor.csv in log_dir

    If training=False:
      - no CSV logging (used for evaluation VecNormalize structure)
    """

    def _init() -> gym.Env:
        env = gym.make("Humanoid-v5")

        # Optional reward shaping
        if shaping_cfg is not None and len(shaping_cfg) > 0:
            env = ShapedHumanoidEnv(env, shaping_cfg)

        if training:
            # Track episode stats in info["episode"]
            env = RecordEpisodeStatistics(env)
            # Log to CSV for plotting (similar to old Monitor)
            log_file = log_dir / "monitor.csv"
            env = CSVLogger(env, log_file)

        env.reset(seed=seed)
        return env

    return _init


def make_vec_envs(
    log_dir: Path,
    seed: int,
    shaping_cfg: Optional[Dict[str, Any]] = None,
    training: bool = True,
) -> VecNormalize:
    """
    Create a vectorized + normalized environment (DummyVecEnv + VecNormalize)
    for either training or evaluation.

    Args:
        log_dir: where CSVLogger will write monitor.csv (if training=True).
        seed: random seed for env.
        shaping_cfg: reward shaping configuration (or None / {}).
        training: if True, env is in training mode (rewards normalized,
                  and CSV logs are written).
                  if False, no CSV logging is applied and rewards are not normalized.
    """
    env_fns = [_make_single_env(log_dir, seed, shaping_cfg, training)]
    vec_env = DummyVecEnv(env_fns)

    if training:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
        )
    else:
        # For eval, we disable reward normalization and do not write CSV logs
        vec_env = VecNormalize(
            vec_env,
            training=False,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )

    return vec_env
