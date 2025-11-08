import gymnasium as gym
import ale_py
import os
import argparse
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor

# Import the wrapper used during training
from trainer.fast_atari_wrappers import FastAtariWrapper

# --- Configuration ---
ALGO_MAP = {"PPO": PPO, "A2C": A2C, "DQN": DQN}

# --- Main Script ---

parser = argparse.ArgumentParser(description="Watch a trained agent play Atari.")
parser.add_argument("model", help="Path to the saved SB3 model zip file")
parser.add_argument("--env", default="ALE/SpaceInvaders-v5", help="Gym environment ID (e.g., ALE/Breakout-v5)")
args = parser.parse_args()

MODEL_PATH = args.model
ENV_ID = args.env

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

# Create the factory function to build the env, using the correct wrappers
def make_env(env_id_to_make, seed=0):
    def _init():
        gym.register_envs(ale_py)
        # IMPORTANT: Set render_mode to "human" and use the same settings as training
        env = gym.make(env_id_to_make, render_mode="human", obs_type="rgb", frameskip=1)
        env.reset(seed=seed)
        env = Monitor(env)
        # Use the same wrapper as in training, but disable terminal_on_life_loss for viewing
        env = FastAtariWrapper(env, frame_skip=4, screen_size=84, terminal_on_life_loss=False)
        return env
    return _init

# Create a DummyVecEnv
print(f"Applying wrappers and creating env: {ENV_ID}")
env = DummyVecEnv([make_env(ENV_ID)])

# Wrap it with VecFrameStack, this MUST match training config (e.g., frame_stack: 4)
env = VecFrameStack(env, 4) 

print("Wrappers applied. Starting game...")

# --- THIS IS THE CORRECTED GAME LOOP ---
# This part is unchanged and correct.
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