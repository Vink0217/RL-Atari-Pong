import os
import yaml
import time
import gymnasium as gym
import ale_py

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage

from .envs import make_atari_env
from .callbacks import EarlyStopOnReward


class Trainer:
    """
    Trainer orchestrates environment creation, model instantiation, callbacks, and training.
    Reads a config dict (usually from YAML) and writes logs/checkpoints to `log_dir`.
    """

    def __init__(self, cfg: dict, log_dir: str):
        self.cfg = cfg
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def _make_model(self, env):
        """Create the PPO model with standard hyperparameters."""
        model = PPO(
            policy="CnnPolicy",
            env=env,
            learning_rate=self.cfg.get("learning_rate", 2.5e-4),
            n_steps=self.cfg.get("n_steps", 128),
            batch_size=self.cfg.get("batch_size", 256),
            n_epochs=self.cfg.get("n_epochs", 4),
            gamma=self.cfg.get("gamma", 0.99),
            ent_coef=self.cfg.get("ent_coef", 0.01),
            clip_range=self.cfg.get("clip_range", 0.1),
            verbose=1,
            device=self.cfg.get("device", "cpu"),
        )
        return model

    def train(self):
        """Main training loop — builds envs, model, callbacks, and runs PPO.learn()."""
        env_id = self.cfg.get("env_id", "ALE/Pong-v5")
        n_envs = int(self.cfg.get("n_envs", 8))
        frame_stack = int(self.cfg.get("frame_stack", 4))
        seed = int(self.cfg.get("seed", 0))
        use_subproc = bool(self.cfg.get("use_subproc", True))

        print(f"[Trainer] Building envs: {env_id} (n_envs={n_envs}, frame_stack={frame_stack})")
        env = make_atari_env(env_id, n_envs=n_envs, seed=seed, frame_stack=frame_stack, use_subproc=use_subproc)

        # Set up the logger: stdout, CSV, and TensorBoard
        new_logger = configure(self.log_dir, ["stdout", "csv", "tensorboard"])

        # Initialize PPO model
        model = self._make_model(env)
        model.set_logger(new_logger)

        # --- Callbacks setup ---

        # ✅ Checkpoint every `checkpoint_freq` timesteps
        checkpoint_freq = int(self.cfg.get("checkpoint_freq", 100_000))
        checkpoint_freq_per_env = max(1, checkpoint_freq // max(1, n_envs))
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq_per_env,
            save_path=self.log_dir,
            name_prefix="checkpoint",
        )

        # ✅ Evaluation every `eval_freq` timesteps
        eval_freq = int(self.cfg.get("eval_freq", 200_000))
        eval_freq_per_env = max(1, eval_freq // max(1, n_envs))
        n_eval_episodes = int(self.cfg.get("n_eval_episodes", 10))

        # Create eval env with the same vectorization backend as training to avoid
        # "Training and eval env are not of the same type" warnings from SB3.
        eval_env = make_atari_env(
            env_id, n_envs=1, seed=seed + 42, frame_stack=frame_stack, use_subproc=use_subproc
        )

        # Ensure eval env has the same top-level Vec wrapper as the training env.
        # SB3 sometimes wraps the training env in VecTransposeImage (to reorder
        # image channels) when the model is created; if so, wrap the eval_env
        # the same way to avoid the SB3 warning about mismatched env types.
        try:
            # If SB3 wrapped the training env in VecTransposeImage (it does this
            # automatically for image observations), wrap the eval_env the same
            # way so the top-level types match and SB3 won't warn.
            if isinstance(model.get_env(), VecTransposeImage) or (
                "VecTransposeImage" in repr(model.get_env())
            ):
                eval_env = VecTransposeImage(eval_env)
        except Exception:
            # Best-effort only; if this fails, we still continue and SB3 will
            # emit its warning if types differ.
            pass

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=self.log_dir,
            log_path=self.log_dir,
            eval_freq=eval_freq_per_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
        )

        # ✅ Early stopping (optional)
        # Allow YAML to set `early_stop: null` — coerce None -> {} so `.get` works below
        early_cfg = self.cfg.get("early_stop") or {}
        early_cb = None
        if early_cfg.get("enabled", True):
            early_cb = EarlyStopOnReward(
                eval_callback=eval_callback,
                patience=int(early_cfg.get("patience", 5)),
                min_delta=float(early_cfg.get("min_delta", 1.0)),
            )

        # Combine all callbacks
        callbacks = [checkpoint_callback, eval_callback]
        if early_cb:
            callbacks.append(early_cb)

        # --- Start training ---
        total_timesteps = int(self.cfg.get("total_timesteps", 5_000_000))
        print(f"[Trainer] Starting training for {total_timesteps:,} timesteps...")
        start_time = time.time()

        model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)

        print(f"[Trainer] Training completed in {(time.time() - start_time) / 3600:.2f} hours.")
        env.close()
        eval_env.close()
        model.save(os.path.join(self.log_dir, "final_model"))

        print(f"[Trainer] Final model saved to {self.log_dir}/final_model.zip")


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


if __name__ == "__main__":
    # Example for direct execution
    cfg = load_config("trainer/config.yaml")
    trainer = Trainer(cfg, log_dir="./runs/example_run")
    trainer.train()
