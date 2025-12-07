from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import glob
import os
import argparse
from config.config import ALGOS, SEEDS, SHAPING_CONFIGS, PROJECT_ROOT, EXPERIMENTS_DIR

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from config.config import (
    EXPERIMENTS_DIR,
    CHECKPOINTS_DIR,
    SHAPING_CONFIGS,
)
from envs.envs import ShapedHumanoidEnv

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained RL agents on Humanoid-v5 and plot curves."
    )

    parser.add_argument(
        "--algo",
        type=str,
        choices=ALGOS + ["all"],
        default="all",
        help="Which algorithm to evaluate. Use 'all' to evaluate all.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=SEEDS,
        help=f"Random seeds to evaluate. Default: {SEEDS}",
    )
    parser.add_argument(
        "--shaping",
        type=str,
        choices=list(SHAPING_CONFIGS.keys()),
        default="none",
        help="Reward shaping configuration to evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        choices=["best", "final"],
        default="best",
        help="Which checkpoint to use for evaluation/video: 'best' or 'final'.",
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes per run.",
    )
    parser.add_argument(
        "--render",
        type=str,
        choices=["none", "human", "video"],
        default="none",
        help=(
            "Render mode:\n"
            "  none  - just compute returns\n"
            "  human - open a window (for local machine)\n"
            "  video - record a single episode to MP4"
        ),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If set, also load evaluations.npz and plot eval curves.",
    )
    return parser.parse_args()


def resolve_algos(choice: str) -> List[str]:
    if choice == "all":
        return ALGOS
    return [choice]


MODEL_CLASSES = {
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
}


def build_run_dirs(
    algo_name: str,
    seed: int,
    shaping_name: str,
) -> Tuple[Path, Path]:
    """
    Reconstruct experiment and checkpoint directories for a given run.

    Must match the naming used in train_one_run().
    """
    exp_dir = EXPERIMENTS_DIR / f"{algo_name}_seed{seed}_shape_{shaping_name}"
    ckpt_dir = CHECKPOINTS_DIR / f"{algo_name}_seed{seed}_shape_{shaping_name}"
    return exp_dir, ckpt_dir


def pick_checkpoint(
    algo_name: str,
    ckpt_dir: Path,
    checkpoint_type: str = "best",
) -> Path:
    """
    Decide which checkpoint zip to load.

    checkpoint_type: "best" or "final"
    """
    if checkpoint_type == "best":
        best_path = ckpt_dir / "best_model.zip"
        if best_path.exists():
            return best_path
        # fallback if best_model.zip not found
        final_path = ckpt_dir / f"{algo_name}_final.zip"
        if final_path.exists():
            print(f"[WARN] best_model.zip not found, using final model: {final_path.name}")
            return final_path
        raise FileNotFoundError(
            f"No best_model.zip or final model found in {ckpt_dir}"
        )
    elif checkpoint_type == "final":
        final_path = ckpt_dir / f"{algo_name}_final.zip"
        if final_path.exists():
            return final_path
        raise FileNotFoundError(
            f"Final model not found: {final_path}"
        )
    else:
        raise ValueError(f"Unknown checkpoint_type: {checkpoint_type}")


def make_eval_vec_env(
    algo_name: str,
    seed: int,
    shaping_name: str,
    render_mode: Optional[str] = None,
) -> Tuple[VecNormalize, Path, Path]:
    """
    Create a DummyVecEnv + VecNormalize for evaluation.

    - Uses Humanoid-v5 with optional ShapedHumanoidEnv (reward shaping)
    - Loads VecNormalize statistics from training if available
    - Sets training=False, norm_reward=False for proper evaluation

    render_mode:
        None       -> no visual output
        "human"    -> open a window
        "rgb_array"-> used for video recording
    """
    exp_dir, ckpt_dir = build_run_dirs(algo_name, seed, shaping_name)
    shaping_cfg: Dict[str, Any] = SHAPING_CONFIGS[shaping_name]

    def _init():
        env = gym.make("Humanoid-v5", render_mode=render_mode)
        if shaping_cfg:
            env = ShapedHumanoidEnv(env, shaping_cfg)
        env.reset(seed=seed + 10_000)  # different seed from training
        return env

    base_venv = DummyVecEnv([_init])

    vecnorm_path = ckpt_dir / f"vecnormalize_{algo_name}.pkl"
    if vecnorm_path.exists():
        print(f"[INFO] Loading VecNormalize stats from {vecnorm_path.name}")
        venv = VecNormalize.load(str(vecnorm_path), base_venv)
    else:
        print(f"[WARN] VecNormalize stats not found at {vecnorm_path}, "
              f"creating new eval VecNormalize (obs normalized, rewards not normalized).")
        venv = VecNormalize(
            base_venv,
            training=False,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )

    venv.training = False
    venv.norm_reward = False
    return venv, exp_dir, ckpt_dir


def load_model(
    algo_name: str,
    ckpt_path: Path,
) -> Any:
    """
    Load a trained model for given algorithm from ckpt_path.
    """
    if algo_name not in MODEL_CLASSES:
        raise ValueError(f"Unsupported algorithm: {algo_name}")
    ModelCls = MODEL_CLASSES[algo_name]
    print(f"[INFO] Loading {algo_name.upper()} model from: {ckpt_path}")
    model = ModelCls.load(ckpt_path)
    return model


def evaluate_episodes(
    algo_name: str,
    seed: int,
    shaping_name: str,
    checkpoint_type: str = "best",
    n_episodes: int = 5,
    render_mode: Optional[str] = None,
) -> List[float]:
    """
    Run evaluation episodes and return list of episode returns.

    If render_mode=="human", a render window will be opened.
    """
    venv, exp_dir, ckpt_dir = make_eval_vec_env(
        algo_name=algo_name,
        seed=seed,
        shaping_name=shaping_name,
        render_mode=render_mode,
    )
    ckpt_path = pick_checkpoint(algo_name, ckpt_dir, checkpoint_type)
    model = load_model(algo_name, ckpt_path)

    returns: List[float] = []

    for ep in range(n_episodes):
        obs = venv.reset()
        done = [False]
        ep_ret = 0.0

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = venv.step(action)
            ep_ret += float(reward[0])

            if render_mode == "human":
                # For human render, explicitly call render on the vec env.
                venv.render()

        returns.append(ep_ret)
        print(f"[INFO] Episode {ep} return: {ep_ret:.2f}")

    venv.close()
    return returns


def record_video_for_run(
    algo_name: str,
    seed: int,
    shaping_name: str,
    checkpoint_type: str = "best",
    video_root: Optional[Path] = None,
    video_prefix: Optional[str] = None,
) -> Optional[Path]:
    """
    Record a single evaluation episode to an MP4 video file.

    Uses render_mode="rgb_array" + RecordVideo wrapper.

    Returns:
        Path to the recorded video file, or None if something went wrong.
    """
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    VIDEO_DIR = PROJECT_ROOT / "videos"

    video_root = VIDEO_DIR / algo_name / shaping_name
    video_root.mkdir(parents=True, exist_ok=True)

    exp_dir, ckpt_dir = build_run_dirs(algo_name, seed, shaping_name)
    if video_prefix is None:
        video_prefix = f"{algo_name}_seed{seed}_shape_{shaping_name}_{checkpoint_type}"

    shaping_cfg: Dict[str, Any] = SHAPING_CONFIGS[shaping_name]

    def _init():
        env = gym.make("Humanoid-v5", render_mode="rgb_array")
        if shaping_cfg:
            env = ShapedHumanoidEnv(env, shaping_cfg)

        env = RecordVideo(
            env,
            video_folder=str(video_root),
            episode_trigger=lambda ep_id: True,  # record first episode
            name_prefix=video_prefix,
        )
        env.reset(seed=seed + 20_000)
        return env

    base_venv = DummyVecEnv([_init])

    vecnorm_path = ckpt_dir / f"vecnormalize_{algo_name}.pkl"
    if vecnorm_path.exists():
        print(f"[INFO] Loading VecNormalize stats from {vecnorm_path.name}")
        venv = VecNormalize.load(str(vecnorm_path), base_venv)
    else:
        print(f"[WARN] VecNormalize stats not found at {vecnorm_path}, "
              f"creating eval VecNormalize with default stats.")
        venv = VecNormalize(
            base_venv,
            training=False,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )

    venv.training = False
    venv.norm_reward = False

    ckpt_path = pick_checkpoint(algo_name, ckpt_dir, checkpoint_type)
    model = load_model(algo_name, ckpt_path)

    # Run a single episode
    obs = venv.reset()
    done = [False]
    total_reward = 0.0

    print(f"[INFO] Recording one episode for {algo_name} (seed={seed}, shaping={shaping_name}) ...")
    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = venv.step(action)
        total_reward += float(reward[0])

    venv.close()
    print(f"[INFO] Recorded episode return: {total_reward:.2f}")

    # Find the newest video file
    pattern = str(video_root / f"{video_prefix}*.mp4")
    candidates = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not candidates:
        print("[WARN] No video file found after recording.")
        return None

    best_video_path = Path(candidates[-1])
    print(f"[INFO] Video saved at: {best_video_path}")
    return best_video_path

# ==============================
# Logging / plotting helpers
# ==============================

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

    # results can be (N, K) or (N,)
    if results.ndim > 1:
        # shape (N, K) → mean over K eval episodes
        returns = results.mean(axis=1)
    else:
        returns = results

    return timesteps, returns


def plot_eval_curves(
    algos: List[str],
    seeds: List[int],
    shaping_name: str,
    output_dir: Path,
    title_suffix: str = "",
) -> Path:
    """
    For each algo and seed, load evaluations.npz and plot
    eval return vs timesteps. All curves go on one figure.

    Returns:
        Path to the saved PNG file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    any_plotted = False

    for algo in algos:
        for seed in seeds:
            exp_dir, _ = build_run_dirs(algo, seed, shaping_name)
            try:
                timesteps, returns = load_evaluations_npz(exp_dir)
            except FileNotFoundError:
                print(f"[WARN] No evaluations.npz for {algo} seed={seed} shaping={shaping_name}")
                continue

            label = f"{algo}_seed{seed}"
            ax.plot(timesteps, returns, label=label)
            any_plotted = True

    if not any_plotted:
        print("[WARN] No evaluation curves found, skipping plot.")
        plt.close(fig)
        return output_dir / "no_plot.png"

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Evaluation return")
    base_title = f"Evaluation curves (shaping={shaping_name})"
    if title_suffix:
        ax.set_title(f"{base_title} - {title_suffix}")
    else:
        ax.set_title(base_title)

    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()

    out_path = output_dir / f"eval_curves_shaping_{shaping_name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[INFO] Saved evaluation curves to: {out_path}")
    return out_path
