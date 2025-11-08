# AtariTrainer

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Framework: SB3](https://img.shields.io/badge/Framework-Stable_Baselines3-brightgreen)](https://stable-baselines3.readthedocs.io/en/master/)
[![Gymnasium](https://img.shields.io/badge/Env-Gymnasium-ffC000)](https://gymnasium.farama.org/)

A robust, configurable, and extensible training pipeline for Atari agents using Gymnasium and Stable-Baselines3. This repository provides a full framework for training, evaluating, optimizing, and watching agents on any Atari game.

---

## ✨ Key Features

* **Multi-Algorithm Support:** Train state-of-the-art agents using PPO, DQN, or A2C right out of the box.
* **Hyperparameter Optimization:** Use the `optimize.py` script to run Optuna-based studies and find the best hyperparameters for your agents.
* **Config-Driven:** All hyperparameters are managed in simple `*.yaml` files for easy and reproducible experiments.
* **Flexible CLI:** Override any config setting (like `algo`, `env_id`, or `total_timesteps`) directly from the command line.
* **Resume Training:** Stop and resume training from any saved checkpoint (`.zip` file).
* **Cloud-Ready:** Includes a `modal_app.py` for running large-scale training jobs on cloud GPUs.
* **Generic Watch Script:** A single `game.py` script can load any trained model and play back its performance for any Atari game.

---

## 💻 Project Structure

```
AtariTrainer/
├── configs/
│   ├── base.yaml
│   ├── breakout_dqn.yaml
│   ├── pacman.yaml
│   └── smooth.yaml
├── trainer/
│   ├── __init__.py
│   ├── callbacks.py
│   ├── envs.py
│   ├── fast_atari_wrappers.py
│   ├── policy.py
│   ├── runner.py
│   └── utils.py
├── .gitignore
├── .python-version
├── eval.py
├── examples.md
├── game.py
├── modal_app.py
├── optimize.py
├── pyproject.toml
├── README.md
├── requirements.txt
├── train.py
└── uv.lock
```

---

## 🚀 Getting Started

### 1. Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Vink0217/AtariTrainer.git
    cd AtariTrainer
    ```

2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### 2. Monitoring with TensorBoard

Before you start training, open a separate terminal and run TensorBoard to watch your agent learn in real-time. All logs are stored in the `runs/` directory (which is ignored by Git).

```bash
tensorboard --logdir ./runs
```

---

## 🎮 How to Use

This pipeline is run from the command line. For a comprehensive list of commands for training, optimization, evaluation, and more, please see the detailed examples file:

### ➡️ [**Command Examples (examples.md)**](./examples.md) ⬅️

This document provides examples for:
- **Training** agents with different configurations.
- **Resuming** training from checkpoints.
- **Optimizing** hyperparameters with Optuna.
- **Evaluating** model performance.
- **Watching** your trained agents play.
- **Cloud Training** with Modal.

---

## 📈 Future Work (Roadmap)

This project is the perfect foundation for more advanced RL concepts.

- [x] **Hyperparameter Sweeps:** Integrated Optuna to find the optimal hyperparameters for each game.
- [ ] **AI-vs-AI Arena:** Modify the environment to enable self-play between two policies.
- [ ] **Continuous Integration:** Add a GitHub Action to automatically run a short test on every push.
- [ ] **Web UI Dashboard:** Build a simple Flask/FastAPI app to display results.