import gymnasium as gym
import ale_py
from stable_baselines3 import PPO

gym.register_envs(ale_py)
env = gym.make("ALE/Pong-v5", render_mode="human")
model = PPO.load("./ppo_pong_logs/ppo_pong_final")

obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
env.close()
