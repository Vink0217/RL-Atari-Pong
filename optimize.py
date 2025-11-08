import optuna
import os
import argparse
from trainer.runner import Trainer, load_config
from trainer.utils import pretty_print_cfg

def create_objective(args):
    """
    Factory function to create the Optuna objective function,
    closing over the command-line arguments.
    """
    def objective(trial: optuna.Trial) -> float:
        """
        Main function for Optuna to call, returning the mean reward.
        """
        cfg = load_config(args.config)

        # Define hyperparameters to search
        algo_upper = cfg.get("algorithm", "PPO").upper()
        
        if algo_upper == "PPO":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "n_steps": trial.suggest_int("n_steps", 32, 512, step=32),
                "batch_size": trial.suggest_int("batch_size", 32, 512, step=32),
                "n_epochs": trial.suggest_int("n_epochs", 1, 10),
                "ent_coef": trial.suggest_float("ent_coef", 1e-8, 1e-1, log=True),
                "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
            }
        elif algo_upper == "DQN":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "buffer_size": trial.suggest_int("buffer_size", 50000, 200000),
                "exploration_fraction": trial.suggest_float("exploration_fraction", 0.05, 0.3),
                "exploration_final_eps": trial.suggest_float("exploration_final_eps", 0.01, 0.1),
            }
        else:
            # Default or A2C - can be expanded
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "n_steps": trial.suggest_int("n_steps", 4, 64),
            }

        cfg["algo_params"][algo_upper] = params
        cfg["total_timesteps"] = args.n_timesteps
        cfg["early_stop"]["enabled"] = False
        cfg["device"] = args.device

        # Create a unique log directory for this trial
        study_name = args.study_name or f"optuna_{algo_upper}_{os.path.basename(args.config).split('.')[0]}"
        run_dir = os.path.join(args.log_dir, study_name, f"trial_{trial.number}")
        
        print(f"\n--- Starting Trial {trial.number}/{args.n_trials} for Study '{study_name}' ---")
        print(f"Logging to: {run_dir}")
        pretty_print_cfg(params)

        try:
            trainer = Trainer(cfg, log_dir=run_dir)
            best_mean_reward = trainer.train()
        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
            raise optuna.exceptions.TrialPruned()

        print(f"--- Trial {trial.number} Finished. Best Reward: {best_mean_reward} ---")
        return best_mean_reward
    
    return objective

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization with Optuna")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Base config file for the study")
    parser.add_argument("--n-trials", type=int, default=25, help="Number of optimization trials to run")
    parser.add_argument("--n-timesteps", type=int, default=500_000, help="Timesteps per trial")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--log-dir", type=str, default="./runs", help="Base directory for saving logs")
    parser.add_argument("--study-name", type=str, default=None, help="Name for the Optuna study (defaults to config name)")
    args = parser.parse_args()

    # Create the objective function with the parsed args
    objective_fn = create_objective(args)

    study = optuna.create_study(direction="maximize")
    try:
        study.optimize(objective_fn, n_trials=args.n_trials)
    except KeyboardInterrupt:
        print("\nOptimization stopped early by user.")

    print("\n--- Optimization Finished ---")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best reward (score): {study.best_value}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")