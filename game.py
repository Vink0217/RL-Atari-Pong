import gymnasium as gym
import ale_py
import os
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation
from gymnasium import Wrapper
from stable_baselines3 import PPO, A2C, DQN

# 1. Import the SB3 wrappers we ACTUALLY use in training
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor

# --- Configuration ---
# ⚠️ UPDATE THIS PATH to a model you train with your 'base.yaml'
MODEL_PATH = r"C:\Vinayak\Programmes\Python\RL learning\runs\ALE_Pong-v5_PPO_20251103_033437\final_model.zip"

ALGO_MAP = {"PPO": PPO, "A2C": A2C, "DQN": DQN}

# --- Wrapper copied from trainer/envs.py ---
# This makes the paddle "smoother" instead of "erratic"
class RepeatAction(Wrapper):
    def __init__(self, env, repeat: int = 2):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        for _ in range(self.repeat):
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += reward
            terminated = terminated or term
            truncated = truncated or trunc
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info

# --- Main Script ---

if not os.path.exists(MODEL_PATH):
    print("="*50)
    print(f"ERROR: Model not found at: {MODEL_PATH}")
    print("Please update the 'MODEL_PATH' variable")
    print("="*50)
    exit()

# Try to guess algorithm from path, default to PPO
algo_name = "PPO"
if "DQN" in MODEL_PATH.upper():
    algo_name = "DQN"
elif "A2C" in MODEL_PATH.upper():
    algo_name = "A2C"

print(f"Loading {algo_name} model from: {MODEL_PATH}")
ModelClass = ALGO_MAP[algo_name]
model = ModelClass.load(MODEL_PATH)

# Create the *factory function* to build the env
def make_env():
    def _init():
        gym.register_envs(ale_py)
        env = gym.make("ALE/Pong-v5", render_mode="human")
        # Apply wrappers in the EXACT SAME order as envs.py
        env = RepeatAction(env, repeat=2)
        env = Monitor(env) 
        env = ResizeObservation(env, (84, 84))
        env = GrayscaleObservation(env, keep_dim=True)
        return env
    return _init

# Create a DummyVecEnv
print("Applying wrappers...")
env = DummyVecEnv([make_env()])

# Wrap it with VecFrameStack
# This MUST match 'frame_stack: 4' in your base.yaml
env = VecFrameStack(env, 4) 

print("Wrappers applied. Starting game...")

# --- THIS IS THE CORRECTED GAME LOOP ---
obs = env.reset() 
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    
    # 1. Expect 4 values from the SB3 VecEnv (obs, reward, dones, info)
    obs, reward, dones, info = env.step(action)
    
    # 2. Check the boolean array 'dones'
    done = dones[0]

print("Game over.")
env.close()