# AtariTrainer Command Examples

This file provides a comprehensive list of common and advanced command-line examples for training, evaluating, and watching agents with the AtariTrainer framework.

---

## 1. Basic Training

These commands cover the most common use cases for starting new training runs.

### Train with a Config File (Recommended)

Using a configuration file is the best way to ensure reproducible results. All settings are loaded from the YAML file, and you can override specific ones from the command line.

**Train Breakout with DQN:**
```bash
python train.py --config configs/breakout_dqn.yaml --device cuda
```
*This command loads all settings from `breakout_dqn.yaml` and runs the training on a CUDA-enabled GPU.*

**Train Pac-Man with PPO (and override timesteps):**
```bash
python train.py --config configs/pacman.yaml --total-timesteps 5000000 --device cuda
```
*This uses the `pacman.yaml` config but reduces the training duration to 5 million steps.*

### Train without a Config File (Quick Tests)

You can also run training by specifying all parameters on the command line. This is useful for quick experiments.

**Train Pong with PPO:**
```bash
python train.py --env "ALE/Pong-v5" --algo PPO --total-timesteps 1000000 --n-envs 8 --device cuda
```

**Train Space Invaders with A2C:**
```bash
python train.py --env "ALE/SpaceInvaders-v5" --algo A2C --total-timesteps 2000000 --n-envs 16 --device cpu
```

---

## 2. Resuming Training

If a run is interrupted or you want to fine-tune a model, you can resume from a saved checkpoint.

**Resume from `best_model.zip`:**
```bash
# The --load-model path should point to your checkpoint file
python train.py --config configs/pacman.yaml --load-model "runs/ALE_MsPacman-v5_PPO_1700000000/best_model.zip" --total-timesteps 2000000
```
*This command loads the `best_model.zip` and continues training for an additional 2 million timesteps. A new `resume_run` folder will be created inside the original log directory to store the new logs and models.*

---

## 3. Watching a Trained Agent

Use `game.py` to see your trained agent in action.

**Watch a PPO Pong Agent:**
```bash
# The first argument is the path to the model
python game.py "runs/ALE_Pong-v5_PPO_1700000000/best_model.zip"
```

**Watch a DQN Breakout Agent:**
```bash
# You must specify the environment if it's not the default (Pong)
python game.py "runs/ALE_Breakout-v5_DQN_1700000000/best_model.zip" --env "ALE/Breakout-v5"
```

---

## 4. Evaluation

Use `eval.py` to formally evaluate a model's performance over a set number of episodes.

**Evaluate the Best Pac-Man Model:**
```bash
python eval.py "runs/ALE_MsPacman-v5_PPO_1700000000/best_model.zip" --env "ALE/MsPacman-v5" --n-eval-episodes 20
```
*This will run the agent for 20 full episodes and print the mean and standard deviation of the rewards.*

---

## 5. Hyperparameter Optimization

Use `optimize.py` to run a hyperparameter search with Optuna. This script automatically trains multiple agent versions to find the best hyperparameter combination for a given algorithm and environment.

**Run a PPO Optimization Study for Pong:**
```bash
python optimize.py --config configs/base.yaml --n-trials 50 --n-timesteps 200000 --study-name "PPO_Pong_Opt"
```
*This command starts an optimization study for the PPO algorithm, using `configs/base.yaml` as the base configuration (which defaults to the Pong environment).*
- It will run `50` trials.
- Each trial will train an agent for `200,000` timesteps.
- The logs and results for this study will be saved in a directory named `PPO_Pong_Opt` inside your logs folder.

**Run a DQN Optimization for Breakout:**
```bash
python optimize.py --config configs/breakout_dqn.yaml --n-trials 30 --n-timesteps 300000
```
*This runs a study for the DQN algorithm on the Breakout environment, as specified in the `breakout_dqn.yaml` file.*

---

## 6. Using Modal for Cloud Training

To run a training job on a powerful cloud GPU, use the `modal` CLI.

**Run the Default Pac-Man Training on an A100 GPU:**
```bash
modal run modal_app.py:run_training
```
*This command executes the `run_training` function in `modal_app.py`, which is pre-configured to use the `pacman.yaml` config.*

**Check GPU Availability on Modal:**
```bash
modal run modal_app.py:gpu_check
```
*This is a utility function to verify that the Modal environment is correctly set up with a GPU and the right libraries.*
