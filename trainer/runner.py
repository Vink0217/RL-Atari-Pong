import os
import yaml
import time
import gymnasium as gym
import ale_py

# 1. IMPORT THE NEW ALGORITHMS
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage

from .envs import make_atari_env
from .callbacks import EarlyStopOnReward

# 2. CREATE A MAP OF ALGORITHM NAMES TO THEIR CLASSES
ALGO_MAP = {
    "PPO": PPO,
    "A2C": A2C,
    "DQN": DQN,
}


class Trainer:
    """
    Trainer orchestrates environment creation, model instantiation, callbacks, and training.
    Reads a config dict (usually from YAML) and writes logs/checkpoints to `log_dir`.
    """

    def __init__(self, cfg: dict, log_dir: str):
        self.cfg = cfg
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    # 3. THIS IS THE MODIFIED MODEL FACTORY
    def _make_model(self, env):
        """Create the SB3 model based on the config."""
        
        # Get the algorithm name and class (now set by CLI or YAML)
        algo_name = self.cfg.get("algorithm", "PPO")
        if algo_name not in ALGO_MAP:
            raise ValueError(f"Unknown algorithm: {algo_name}. Must be one of {list(ALGO_MAP.keys())}")
        
        AlgoClass = ALGO_MAP[algo_name]
        
        # Get the nested hyperparameters for that specific algorithm
        try:
            algo_params = self.cfg.get("algo_params", {}).get(algo_name, {})
        except Exception:
            raise ValueError(f"No 'algo_params.{algo_name}' found in config.")

        if not algo_params:
             print(f"[Trainer] Warning: No hyperparameters found for {algo_name} in config, using defaults.")

        print(f"[Trainer] Initializing model for {algo_name} with params: {algo_params}")
        
        # Create the model, unpacking the params dict
        model = AlgoClass(
            env=env,
            verbose=1,
            device=self.cfg.get("device", "cpu"), 
            tensorboard_log=self.log_dir, # Pass the log_dir here
            **algo_params  
        )
        return model

    def train(self):
        """Main training loop — builds envs, model, callbacks, and runs .learn()."""
        # --- Config reading ---
        env_id = self.cfg.get("env_id", "ALE/Pong-v5")
        n_envs = int(self.cfg.get("n_envs", 8))
        frame_stack = int(self.cfg.get("frame_stack", 4))
        seed = int(self.cfg.get("seed", 0))
        use_subproc = bool(self.cfg.get("use_subproc", True))

        print(f"[Trainer] Building envs: {env_id} (n_envs={n_envs}, frame_stack={frame_stack})")
        
        # --- Env creation ---
        env = make_atari_env(env_id, n_envs=n_envs, seed=seed, frame_stack=frame_stack, use_subproc=use_subproc)

        # --- Logger ---
        # Note: We pass tensorboard_log to the model, but configure() sets up CSV/stdout
        new_logger = configure(self.log_dir, ["stdout", "csv", "tensorboard"])

        # --- Model creation ---
        model = self._make_model(env)
        model.set_logger(new_logger) # Connect the CSV/stdout logger

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

        eval_env = make_atari_env(
            env_id, n_envs=1, seed=seed + 42, frame_stack=frame_stack, use_subproc=use_subproc
        )

        try:
            if isinstance(model.get_env(), VecTransposeImage) or (
                "VecTransposeImage" in repr(model.get_env())
            ):
                eval_env = VecTransposeImage(eval_env)
        except Exception:
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
        early_cfg = self.cfg.get("early_stop") or {}
        early_cb = None
        if early_cfg.get("enabled", True):
            early_cb = EarlyStopOnReward(
                eval_callback=eval_callback,
                patience=int(early_cfg.get("patience", 5)),
                min_delta=float(early_cfg.get("min_delta", 1.0)),
            )

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