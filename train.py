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

# Optional: use the Trainer and load_config helper when a YAML is supplied
try:
    from trainer.runner import load_config, Trainer
except Exception:
    # Keep backward compatibility if trainer package isn't importable for some reason
    load_config = None
    Trainer = None


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
    parser.add_argument("--log-dir", type=str, default="./ppo_pong_logs", help="Logging directory")
    parser.add_argument("--save-freq", type=int, default=100_000, help="Checkpoint save frequency")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file (optional)")

    args = parser.parse_args()

    # If a YAML config is provided, prefer using the Trainer in trainer/runner.py
    if args.config:
        if load_config is None or Trainer is None:
            raise RuntimeError("trainer.runner module not available - cannot load YAML config")

        # Load YAML config
        cfg = load_config(args.config)

        # Map CLI flags to config keys (CLI should override YAML when explicitly passed)
        cli_to_cfg = {
            "total-timesteps": "total_timesteps",
            "n-envs": "n_envs",
            "device": "device",
            "env": "env_id",
            "log-dir": "log_dir",
            "save-freq": "checkpoint_freq",
        }

        # Helper to check whether a CLI flag was explicitly provided
        def cli_passed(flag: str) -> bool:
            for a in sys.argv[1:]:
                if a == f"--{flag}" or a.startswith(f"--{flag}="):
                    return True
            return False

        # Apply overrides from CLI for flags the user explicitly passed
        for cli_flag, cfg_key in cli_to_cfg.items():
            if cli_passed(cli_flag):
                # argparse destination uses underscores instead of dashes
                dest = cli_flag.replace("-", "_")
                val = getattr(args, dest)
                # convert types if necessary (keep YAML values types otherwise)
                cfg[cfg_key] = val

        # Determine log_dir: config beats default, but CLI --log-dir overrides above
        log_dir = cfg.get("log_dir", args.log_dir)

        print(f"[train.py] Starting Trainer with config: {args.config} | log_dir={log_dir}")
        trainer = Trainer(cfg, log_dir=log_dir)
        trainer.train()
        sys.exit(0)

    # ✅ Windows requires the "spawn" guard
    num_envs = args.n_envs
    use_subproc = True  # change to False if you get Windows spawn issues

    def make_env_custom():
        """Return an env factory that uses CLI env_id."""
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

    # ✅ Logging and checkpoints
    os.makedirs(args.log_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.save_freq // num_envs),
        save_path=args.log_dir,
        name_prefix="ppo_pong_checkpoint",
    )

    new_logger = configure(args.log_dir, ["stdout", "csv", "tensorboard"])

    # ✅ PPO model setup
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

    # ✅ Training (uses your CLI value)
    print(f"🚀 Starting PPO training on {args.env} for {args.total_timesteps} timesteps "
          f"using {num_envs} parallel envs on {args.device.upper()}.")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    # ✅ Save final model
    model.save(os.path.join(args.log_dir, "ppo_pong_final"))
    env.close()

    print(f"✅ Training complete! Model saved to: {args.log_dir}")
