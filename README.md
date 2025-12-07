# Humanoid-v5 Reinforcement Learning Project

This repository contains an implementation and comparison of three reinforcement learning algorithms  
(**PPO**, **SAC**, **TD3**) on the **Humanoid-v5** continuous control environment using **Gymnasium + MuJoCo + Stable-Baselines3**.

The goal of the project is to evaluate the learning efficiency, final performance, and training stability  
of state-of-the-art algorithms under *identical environment settings and reward shaping*.

trained model files: https://drive.google.com/file/d/1XcDYU4tPQ49NddsTBB-qe5Aw-w8umRvC/view?usp=sharing
Report file: https://github.com/wowharry1/humanoid-rl-project/releases/tag/v1.0

---

## Project Overview

| Component | Specification |
|----------|-------------|
| Environment | Gymnasium **Humanoid-v5** |
| Simulator | **MuJoCo** |
| State Space | 376-dimensional continuous state |
| Action Space | 17-dimensional continuous torque control |
| Reward | Default reward + **torque penalty shaping** |
| Metric | **Episode Return** (sum of rewards per episode) |

This repository trains PPO, SAC, and TD3 for **300,000 timesteps each**,  saves the **best checkpoint (best_model.zip)** during training,  and evaluates the agents both **quantitatively (evaluation curves)** and  **qualitatively (recorded videos)**.

---

## Repository Structure
project_root/

│

├─ src/

│ ├─ train.py # Training entry point

│ ├─ eval.py # Evaluation & video recording entry point

│ ├─ utils/ # Training/Evaluation utilities/Algorithm builders (PPO/SAC/TD3)

│ ├─ config/ # Config

│ └─ envs/ # Setting Environment

│

├─ checkpoints/ # Saved checkpoints (best_model.zip)

├─ plots/ # Evaluation curves

├─ videos/ # Recorded performance videos

├─ README.md

└─ .gitignore

## Install Dependencies
pip install -r requirements.txt

## Train (all algorithms)
python train.py --algo all --shaping torque_penalty --total_timesteps 300000

## Evaluate and Save Videos
python eval.py --algo all --shaping torque_penalty --render video

