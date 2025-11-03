import os
import yaml
import time
import gymnasium as gym
import ale_py
import torch  # <-- Import PyTorch

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from stable_baselines3.common.policies import ActorCriticCnnPolicy # <-- Import base policy

from .envs import make_atari_env
from .callbacks import EarlyStopOnReward, RotatingCheckpointCallback 
# --- 1. Import our new custom policy ---
from .policy import GpuPreprocessedCnnPolicy

ALGO_MAP = {
    "PPO": PPO,
    "A2C": A2C,
    "DQN": DQN,
}

class Trainer:
    def __init__(self, cfg: dict, log_dir: str):
        self.cfg = cfg
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # --- 2. Add hardware-aware flags ---
        self.is_gpu_available = torch.cuda.is_available() and self.cfg.get("device", "auto") != "cpu"
        self.use_cpu_preprocessing = not self.is_gpu_available
        self.device = self.cfg.get("device", "auto")

        if self.is_gpu_available:
            print("[Trainer] CUDA device detected. Using GPU Preprocessing.")
        else:
            print("[Trainer] No CUDA device. Using CPU Parallel Preprocessing.")

    def _make_model(self, env):
        algo_name = self.cfg.get("algorithm", "PPO")
        if algo_name not in ALGO_MAP:
            raise ValueError(f"Unknown algorithm: {algo_name}.")
        
        AlgoClass = ALGO_MAP[algo_name]
        algo_params = self.cfg.get("algo_params", {}).get(algo_name, {})

        # --- 3. Smartly select the policy ---
        policy_to_use = "CnnPolicy" # Default
        if self.is_gpu_available:
            # If we're on GPU, use our custom fast policy
            policy_to_use = GpuPreprocessedCnnPolicy
            print("[Trainer] Using custom GpuPreprocessedCnnPolicy")
        else:
            # If on CPU, use the standard policy
            policy_to_use = "CnnPolicy"
            print("[Trainer] Using standard CnnPolicy")
        
        print(f"[Trainer] Initializing model for {algo_name}")
        
        model = AlgoClass(
            policy=policy_to_use, # <-- Use the selected policy
            env=env,
            verbose=1,
            device=self.device, 
            tensorboard_log=self.log_dir,
            **algo_params  
        )
        return model

    def train(self):
        # --- Config reading ---
        env_id = self.cfg.get("env_id", "ALE/Pong-v5")
        n_envs = int(self.cfg.get("n_envs", 8))
        frame_stack = int(self.cfg.get("frame_stack", 4))
        seed = int(self.cfg.get("seed", 0))
        
        # --- 4. Use our hardware-aware flags ---
        # If we're on CPU, we must use SubprocVecEnv for speed.
        # If we're on GPU, it doesn't matter as much, but Subproc is still good.
        use_subproc = bool(self.cfg.get("use_subproc", True)) or not self.is_gpu_available

        print(f"[Trainer] Building envs: {env_id} (n_envs={n_envs}, frame_stack={frame_stack})")
        
        env = make_atari_env(
            env_id, 
            n_envs=n_envs, 
            seed=seed, 
            frame_stack=frame_stack, 
            use_subproc=use_subproc,
            use_cpu_preprocessing=self.use_cpu_preprocessing # <-- Pass the flag!
        )
        
        new_logger = configure(self.log_dir, ["stdout", "csv", "tensorboard"])

        # --- Model loading logic (unchanged) ---
        model_path_to_load = self.cfg.get("load_model")
        if model_path_to_load:
            print(f"[Trainer] Loading and resuming from: {model_path_to_load}")
            algo_name = self.cfg.get("algorithm", "PPO")
            AlgoClass = ALGO_MAP[algo_name]
            
            # --- 5. IMPORTANT: Tell .load() which policy to use ---
            policy_to_use = GpuPreprocessedCnnPolicy if self.is_gpu_available else "CnnPolicy"
            
            model = AlgoClass.load(
                model_path_to_load, 
                env=env, 
                custom_objects={"policy": policy_to_use} # <-- Fix for loading
            )
            model.set_logger(new_logger)
        else:
            print("[Trainer] Creating new model from scratch.")
            model = self._make_model(env) # This already selects the right policy
            model.set_logger(new_logger)

        # --- Callbacks setup ---
        # ... (Your callback logic is unchanged and fine) ...
        checkpoint_freq = int(self.cfg.get("checkpoint_freq", 100_000))
        checkpoint_freq_per_env = max(1, checkpoint_freq // max(1, n_envs))
        keep_last = int(self.cfg.get("checkpoints_to_keep", 5)) 
        checkpoint_callback = RotatingCheckpointCallback(
            keep_last=keep_last,
            save_freq=checkpoint_freq_per_env,
            save_path=self.log_dir,
            name_prefix="checkpoint",
            verbose=1
        )
        eval_freq = int(self.cfg.get("eval_freq", 200_000))
        eval_freq_per_env = max(1, eval_freq // max(1, n_envs))
        n_eval_episodes = int(self.cfg.get("n_eval_episodes", 10))
        
        # --- 6. Create the eval_env with the SAME flags ---
        eval_env = make_atari_env(
            env_id, 
            n_envs=1, 
            seed=seed + 42, 
            frame_stack=frame_stack, 
            use_subproc=False, # Eval env is always 1, so no subproc
            use_cpu_preprocessing=self.use_cpu_preprocessing
        )
        # ... (VecTransposeImage check is unchanged) ...
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

        # --- Start training (unchanged) ---
        total_timesteps = int(self.cfg.get("total_timesteps", 5_000_000))
        print(f"[Trainer] Starting training for {total_timesteps:,} timesteps...")
        start_time = time.time()
        model.learn(
            total_timesteps=total_timesteps, 
            callback=callbacks, 
            progress_bar=True, 
            reset_num_timesteps=False 
        )

        print(f"[Trainer] Training completed in {(time.time() - start_time) / 3600:.2f} hours.")
        env.close()
        eval_env.close()
        model.save(os.path.join(self.log_dir, "final_model"))
        print(f"[Trainer] Final model saved to {self.log_dir}/final_model.zip")

        # --- Return statement (unchanged) ---
        if hasattr(eval_callback, 'best_mean_reward'):
            print(f"[Trainer] Best mean reward: {eval_callback.best_mean_reward}")
            return eval_callback.best_mean_reward
        else:
            return -float('inf')


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

if __name__ == "__main__":
    cfg = load_config("trainer/config.yaml") 
    trainer = Trainer(cfg, log_dir="./runs/example_run")
    trainer.train()