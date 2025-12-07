from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize

from config.config import (
    PROJECT_ROOT,
    EXPERIMENTS_DIR,
    CHECKPOINTS_DIR,
    COMMON_CFG,
    ALGO_CFG,
    SHAPING_CONFIGS,
)
from envs.envs import make_vec_envs


def _build_paths(
    algo_name: str,
    seed: int,
    shaping_name: str,
) -> Dict[str, Path]:
    """
    Helper to build experiment & checkpoint directories for a given run.
    """
    exp_dir = EXPERIMENTS_DIR / f"{algo_name}_seed{seed}_shape_{shaping_name}"
    ckpt_dir = CHECKPOINTS_DIR / f"{algo_name}_seed{seed}_shape_{shaping_name}"

    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    return {"exp_dir": exp_dir, "ckpt_dir": ckpt_dir}


def _create_model(
    algo_name: str,
    train_env: VecNormalize,
    seed: int,
) -> Any:
    """
    Create a PPO / SAC / TD3 model with common + algo-specific hyperparameters.
    """
    algo_cfg = ALGO_CFG[algo_name]

    common_kwargs = dict(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        seed=seed,
        learning_rate=COMMON_CFG["learning_rate"],
        gamma=COMMON_CFG["gamma"],
    )

    if algo_name == "ppo":
        model = PPO(
            **common_kwargs,
            n_steps=algo_cfg["n_steps"],
            batch_size=algo_cfg["batch_size"],
            n_epochs=algo_cfg["n_epochs"],
            ent_coef=algo_cfg["ent_coef"],
        )
    elif algo_name == "sac":
        model = SAC(
            **common_kwargs,
            batch_size=algo_cfg["batch_size"],
            buffer_size=algo_cfg["buffer_size"],
            train_freq=algo_cfg["train_freq"],
            gradient_steps=algo_cfg["gradient_steps"],
            tau=algo_cfg["tau"],
        )
    elif algo_name == "td3":
        model = TD3(
            **common_kwargs,
            batch_size=algo_cfg["batch_size"],
            buffer_size=algo_cfg["buffer_size"],
            train_freq=algo_cfg["train_freq"],
            gradient_steps=algo_cfg["gradient_steps"],
            tau=algo_cfg["tau"],
            policy_delay=algo_cfg["policy_delay"],
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    return model


def train_one_run(
    algo_name: str,
    seed: int,
    shaping_name: str = "none",
) -> None:
    """
    Train a single run for (algo_name, seed, shaping).

    Steps:
      1. Build experiment/checkpoint directories
      2. Create training & evaluation VecNormalize envs
      3. Setup EvalCallback (best_model.zip + evaluations.npz)
      4. Train for COMMON_CFG['total_timesteps']
      5. Save final model + VecNormalize stats
    """
    assert algo_name in ALGO_CFG, f"Unsupported algorithm: {algo_name}"
    assert shaping_name in SHAPING_CONFIGS, f"Unknown shaping config: {shaping_name}"

    paths = _build_paths(algo_name, seed, shaping_name)
    exp_dir: Path = paths["exp_dir"]
    ckpt_dir: Path = paths["ckpt_dir"]

    shaping_cfg: Dict[str, Any] = SHAPING_CONFIGS[shaping_name]

    # -------------------------
    # 1. Create environments
    # -------------------------
    # Training env (with logging)
    train_env = make_vec_envs(
        log_dir=exp_dir,
        seed=seed,
        shaping_cfg=shaping_cfg,
        training=True,
    )

    # Evaluation env:
    #   - different seed
    #   - reward not normalized
    eval_env = make_vec_envs(
        log_dir=exp_dir,
        seed=seed + 100,
        shaping_cfg=shaping_cfg,
        training=False,
    )

    # -------------------------
    # 2. Eval callback
    # -------------------------
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(ckpt_dir),
        log_path=str(exp_dir),
        eval_freq=COMMON_CFG["eval_freq"],
        n_eval_episodes=COMMON_CFG["n_eval_episodes"],
        deterministic=True,
        render=False,
    )

    # -------------------------
    # 3. Create model
    # -------------------------
    model = _create_model(
        algo_name=algo_name,
        train_env=train_env,
        seed=seed,
    )

    # -------------------------
    # 4. Train
    # -------------------------
    model.learn(
        total_timesteps=COMMON_CFG["total_timesteps"],
        callback=eval_callback,
    )

    # -------------------------
    # 5. Save model & VecNormalize
    # -------------------------
    final_model_path = ckpt_dir / f"{algo_name}_final"
    model.save(final_model_path)
    # Save VecNormalize statistics for later evaluation
    train_env.save(str(ckpt_dir / f"vecnormalize_{algo_name}.pkl"))

    print(f"[INFO] Finished training {algo_name} (seed={seed}, shaping={shaping_name}).")
    print(f"[INFO] Final model saved to: {final_model_path}.zip")
    print(f"[INFO] VecNormalize stats saved to: {ckpt_dir / f'vecnormalize_{algo_name}.pkl'}")
