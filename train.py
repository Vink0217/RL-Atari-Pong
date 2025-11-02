"""
PPO Atari Pong Training Script
---------------------------------
Train a PPO agent to play Atari Pong using Gymnasium + Stable-Baselines3.
Compatible with Windows (spawn-safe) multiprocessing.
"""

import os
import gymnasium as gym
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure


def make_env():
    """Factory function for creating an Atari Pong environment."""
    def _init():
        gym.register_envs(ale_py)
        env = gym.make("ALE/Pong-v5", frameskip=4)
        env = Monitor(env)  # track rewards, episode lengths, etc.
        return env
    return _init


if __name__ == "__main__":
    # ✅ Windows requires the "spawn" guard
    num_envs = 8
    use_subproc = True  # change to False if you get Windows spawn issues

    if use_subproc:
        env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    else:
        env = DummyVecEnv([make_env() for _ in range(num_envs)])

    # ✅ Logging and checkpoints
    log_dir = "./ppo_pong_logs"
    os.makedirs(log_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // num_envs,  # save every 100k timesteps
        save_path=log_dir,
        name_prefix="ppo_pong_checkpoint",
    )

    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

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
        device="cuda"  # or "cpu" if no GPU
    )

    model.set_logger(new_logger)

    # ✅ Training
    total_timesteps = 10_000_000  # ~5M = good baseline agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    # ✅ Save final model
    model.save(os.path.join(log_dir, "ppo_pong_final"))
    env.close()

    print("✅ Training complete! Model saved to:", log_dir)
