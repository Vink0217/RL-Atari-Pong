"""
PPO Atari Pong Training Script
---------------------------------
Train a PPO agent to play Atari Pong using Gymnasium + Stable-Baselines3.
Compatible with Windows (spawn-safe) multiprocessing.
"""

import os
import sys
import gymnasium as gym
import ale_py
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Import your utils
try:
    from trainer.runner import load_config, Trainer
    from trainer.utils import make_run_dir, save_config, pretty_print_cfg
except Exception as e:
    print(f"Error importing trainer modules: {e}")
    load_config = None
    Trainer = None
    make_run_dir = None
    save_config = None
    pretty_print_cfg = None


def make_env():
    """Factory function for creating an Atari Pong environment."""
    def _init():
        gym.register_envs(ale_py)
        env = gym.make("ALE/Pong-v5", frameskip=4)
        env = Monitor(env)  # track rewards, episode lengths, etc.
        return env
    return _init


if __name__ == "__main__":
    # ✅ Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train PPO on Atari Pong")

    parser.add_argument("--env", type=str, default="ALE/Pong-v5", help="Environment ID")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000, help="Total timesteps for training")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--device", type=str, default="cuda", help="Device: 'cuda' or 'cpu'")
    parser.add_argument("--log-dir", type=str, default="./ppo_pong_logs", help="Logging directory (fallback)")
    parser.add_argument("--save-freq", type=int, default=100_000, help="Checkpoint save frequency")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file (optional)")
    parser.add_argument("--algo", type=str, default=None, help="Algorithm to use (PPO, DQN, A2C). Overrides config.")

    args = parser.parse_args()

    # If a YAML config is provided, prefer using the Trainer in trainer/runner.py
    if args.config:
        if load_config is None or Trainer is None or make_run_dir is None or save_config is None or pretty_print_cfg is None:
            raise RuntimeError("trainer.runner or trainer.utils module not available - cannot load YAML config")

        # Load YAML config
        cfg = load_config(args.config)

        # --- CLI Overrides ---
        # 2. ADD THE 'algo' MAPPING TO THIS DICTIONARY
        cli_to_cfg = {
            "total-timesteps": "total_timesteps",
            "n-envs": "n_envs",
            "device": "device",
            "env": "env_id",
            "log-dir": "base_log_dir", # CLI --log-dir overrides base_log_dir in YAML
            "save-freq": "checkpoint_freq",
            "algo": "algorithm"  # <-- THIS IS THE NEW, CORRECTED LINE
        }

        def cli_passed(flag: str) -> bool:
            # Helper to check if a flag was actually passed on the command line
            for a in sys.argv[1:]:
                if a == f"--{flag}" or a.startswith(f"--{flag}="):
                    return True
            return False

        for cli_flag, cfg_key in cli_to_cfg.items():
            if cli_passed(cli_flag):
                dest = cli_flag.replace("-", "_")
                val = getattr(args, dest)
                # Only override if the flag was passed (val is not None)
                if val is not None:
                    cfg[cfg_key] = val
        # --- End Overrides ---

        # Get values from config (now with CLI overrides applied)
        base_log_dir = cfg.get("base_log_dir", args.log_dir) 
        env_id = cfg.get("env_id", "ALE/Pong-v5")
        algo = cfg.get("algorithm", "PPO") # This will now be set by --algo if passed
        
        # Use your helper to create a unique dir
        run_dir = make_run_dir(base_log_dir, env_id, algo)
        
        print(f"[train.py] Starting Trainer with config: {args.config}")
        print(f"[train.py] Algorithm: {algo}")
        print(f"[train.py] Logging to: {run_dir}")
        pretty_print_cfg(cfg) # Use your pretty printer!
        
        # Save the *final* config (with overrides) to the run dir
        save_config(cfg, os.path.join(run_dir, "config.json"))

        # Pass the unique run_dir to the Trainer
        trainer = Trainer(cfg, log_dir=run_dir) 
        trainer.train()
        sys.exit(0)

    # --- Fallback (non-config) execution ---
    print("[train.py] No --config provided, running with CLI args and hardcoded PPO.")
    
    num_envs = args.n_envs
    use_subproc = True 

    def make_env_custom():
        def _init():
            gym.register_envs(ale_py)
            env = gym.make(args.env, frameskip=4)
            env = Monitor(env)
            return env
        return _init

    if use_subproc:
        env = SubprocVecEnv([make_env_custom() for _ in range(num_envs)])
    else:
        env = DummyVecEnv([make_env_custom() for _ in range(num_envs)])

    os.makedirs(args.log_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.save_freq // num_envs),
        save_path=args.log_dir,
        name_prefix="ppo_pong_checkpoint",
    )

    new_logger = configure(args.log_dir, ["stdout", "csv", "tensorboard"])

    model = PPO(
        policy="CnnPolicy",
        env=env,
        verbose=1,
        n_steps=128,
        batch_size=256,
        n_epochs=4,
        learning_rate=2.5e-4,
        clip_range=0.1,
        ent_coef=0.01,
        device=args.device,
    )
    model.set_logger(new_logger)

    print(f"🚀 Starting PPO training on {args.env} for {args.total_timesteps} timesteps "
          f"using {num_envs} parallel envs on {args.device.upper()}.")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    model.save(os.path.join(args.log_dir, "ppo_pong_final"))
    env.close()

    print(f"✅ Training complete! Model saved to: {args.log_dir}")