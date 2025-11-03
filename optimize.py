import optuna
import os
import yaml
from trainer.runner import Trainer, load_config
from trainer.utils import make_run_dir, save_config, pretty_print_cfg

# --- CONFIGURATION ---
BASE_CONFIG_FILE = "configs/base.yaml"
N_TRIALS = 25              # Total number of experiments to run
N_TIMESTEPS_PER_TRIAL = 500_000 # Timesteps for each *single* experiment

def objective(trial: optuna.Trial) -> float:
    """
    This is the main function that Optuna will call for each trial.
    It returns the mean reward, which Optuna will try to maximize.
    """
    
    # 1. Load the base config file
    cfg = load_config(BASE_CONFIG_FILE)

    # 2. Define the hyperparameters you want to search
    # We'll focus on PPO's most important parameters
    cfg["algo_params"]["PPO"]["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    cfg["algo_params"]["PPO"]["n_steps"] = trial.suggest_int("n_steps", 32, 512, step=32)
    cfg["algo_params"]["PPO"]["batch_size"] = trial.suggest_int("batch_size", 32, 512, step=32)
    cfg["algo_params"]["PPO"]["n_epochs"] = trial.suggest_int("n_epochs", 1, 10)
    cfg["algo_params"]["PPO"]["ent_coef"] = trial.suggest_float("ent_coef", 1e-8, 1e-1, log=True)
    cfg["algo_params"]["PPO"]["clip_range"] = trial.suggest_float("clip_range", 0.1, 0.4)

    # 3. Set other configs for this specific trial
    cfg["total_timesteps"] = N_TIMESTEPS_PER_TRIAL
    cfg["early_stop"]["enabled"] = False  # Disable early stopping during optimization
    cfg["device"] = "cuda"                # Hard-code to cuda for speed
    
    # 4. Create a unique log directory for this trial
    # We'll name it based on the trial number
    run_dir = os.path.join(cfg.get("base_log_dir", "./runs"), "optuna", f"trial_{trial.number}")
    
    print(f"\n--- Starting Trial {trial.number}/{N_TRIALS} ---")
    print(f"Logging to: {run_dir}")
    pretty_print_cfg(cfg["algo_params"]["PPO"])

    # 5. Run the training
    try:
        trainer = Trainer(cfg, log_dir=run_dir)
        # This now returns the best mean reward thanks to our change in runner.py
        best_mean_reward = trainer.train()
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        # Prune trial if it fails (e.g., invalid hyperparams)
        raise optuna.exceptions.TrialPruned()

    print(f"--- Trial {trial.number} Finished. Best Reward: {best_mean_reward} ---")
    return best_mean_reward


if __name__ == "__main__":
    # 1. Create the "study"
    # We're maximizing the reward, so direction is "maximize"
    study = optuna.create_study(direction="maximize")

    # 2. Start the optimization
    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        print("Optimization stopped early by user.")

    # 3. Print the results
    print("\n--- Optimization Finished ---")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best reward (score): {study.best_value}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")