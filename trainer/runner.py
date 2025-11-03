import os
import yaml
import time
import gymnasium as gym
import ale_py

from stable_baselines3 import PPO, A2C, DQN
# 1. Change the CheckpointCallback import
from stable_baselines3.common.callbacks import EvalCallback # Removed CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage

# 2. Import our NEW callback
from .envs import make_atari_env
from .callbacks import EarlyStopOnReward, RotatingCheckpointCallback 

ALGO_MAP = {
    "PPO": PPO,
    "A2C": A2C,
    "DQN": DQN,
}

class Trainer:
    # ... __init__ and _make_model are unchanged ...
    def __init__(self, cfg: dict, log_dir: str):
        self.cfg = cfg
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def _make_model(self, env):
        algo_name = self.cfg.get("algorithm", "PPO")
        if algo_name not in ALGO_MAP:
            raise ValueError(f"Unknown algorithm: {algo_name}. Must be one of {list(ALGO_MAP.keys())}")
        AlgoClass = ALGO_MAP[algo_name]
        try:
            algo_params = self.cfg.get("algo_params", {}).get(algo_name, {})
        except Exception:
            raise ValueError(f"No 'algo_params.{algo_name}' found in config.")
        if not algo_params:
             print(f"[Trainer] Warning: No hyperparameters found for {algo_name} in config, using defaults.")
        print(f"[Trainer] Initializing model for {algo_name} with params: {algo_params}")
        model = AlgoClass(
            env=env,
            verbose=1,
            device=self.cfg.get("device", "cpu"), 
            tensorboard_log=self.log_dir,
            **algo_params  
        )
        return model

    def train(self):
        # ... Config reading is unchanged ...
        env_id = self.cfg.get("env_id", "ALE/Pong-v5")
        n_envs = int(self.cfg.get("n_envs", 8))
        frame_stack = int(self.cfg.get("frame_stack", 4))
        seed = int(self.cfg.get("seed", 0))
        use_subproc = bool(self.cfg.get("use_subproc", True))
        print(f"[Trainer] Building envs: {env_id} (n_envs={n_envs}, frame_stack={frame_stack})")
        
        env = make_atari_env(env_id, n_envs=n_envs, seed=seed, frame_stack=frame_stack, use_subproc=use_subproc)
        new_logger = configure(self.log_dir, ["stdout", "csv", "tensorboard"])

        # ... Model loading logic is unchanged ...
        model_path_to_load = self.cfg.get("load_model")
        if model_path_to_load:
            print(f"[Trainer] Loading and resuming from: {model_path_to_load}")
            algo_name = self.cfg.get("algorithm", "PPO")
            AlgoClass = ALGO_MAP[algo_name]
            model = AlgoClass.load(model_path_to_load, env=env)
            model.set_logger(new_logger)
        else:
            print("[Trainer] Creating new model from scratch.")
            model = self._make_model(env)
            model.set_logger(new_logger)

        # --- Callbacks setup ---

        # 3. USE THE NEW CALLBACK
        checkpoint_freq = int(self.cfg.get("checkpoint_freq", 100_000))
        checkpoint_freq_per_env = max(1, checkpoint_freq // max(1, n_envs))
        
        # We can add a new config setting for how many to keep, default to 5
        keep_last = int(self.cfg.get("checkpoints_to_keep", 5)) 
        
        checkpoint_callback = RotatingCheckpointCallback(
            keep_last=keep_last, # <-- New parameter
            save_freq=checkpoint_freq_per_env,
            save_path=self.log_dir,
            name_prefix="checkpoint",
            verbose=1 # So it tells us when it's deleting
        )

        # ... EvalCallback is unchanged ...
        eval_freq = int(self.cfg.get("eval_freq", 200_000))
        eval_freq_per_env = max(1, eval_freq // max(1, n_envs))
        n_eval_episodes = int(self.cfg.get("n_eval_episodes", 10))
        eval_env = make_atari_env(
            env_id, n_envs=1, seed=seed + 42, frame_stack=frame_stack, use_subproc=use_subproc
        )
        try:
            if isinstance(model.get_env(), VecTransposeImage) or ("VecTransposeImage" in repr(model.get_env())):
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

        # ... EarlyStopping is unchanged ...
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

        # ... model.learn() is unchanged ...
        total_timesteps = int(self.cfg.get("total_timesteps", 5_000_000))
        print(f"[Trainer] Starting training for {total_timesteps:,} timesteps...")
        start_time = time.time()
        model.learn(
            total_timesteps=total_timesteps, 
            callback=callbacks, 
            progress_bar=True, 
            reset_num_timesteps=False 
        )

        # ... Rest of file is unchanged ...
        print(f"[Trainer] Training completed in {(time.time() - start_time) / 3600:.2f} hours.")
        env.close()
        eval_env.close()
        model.save(os.path.join(self.log_dir, "final_model"))
        print(f"[Trainer] Final model saved to {self.log_dir}/final_model.zip")

# ... load_config and __main__ are unchanged ...
def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

if __name__ == "__main__":
    cfg = load_config("trainer/config.yaml") 
    trainer = Trainer(cfg, log_dir="./runs/example_run")
    trainer.train()