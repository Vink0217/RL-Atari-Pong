# --- (Keep your RepeatAction wrapper class at the top) ---
import gymnasium as gym
import ale_py
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation
from gymnasium import Wrapper

class RepeatAction(Wrapper):
    # (Your class code is here, no changes)
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


# --- REPLACE YOUR OLD make_atari_env WITH THIS ---
def make_atari_env(
    env_id: str, 
    n_envs: int = 1, 
    frame_stack: int = 4, 
    use_subproc: bool = True, 
    seed: int = None,
    use_cpu_preprocessing: bool = True  # <-- THE NEW FLAG
):
    """
    Create a vectorized Atari environment.
    If use_cpu_preprocessing=True, adds Resize/Grayscale wrappers.
    If use_cpu_preprocessing=False, skips them.
    """
    gym.register_envs(ale_py)
    
    def make_env(rank: int = 0):
        def _init():
            env = gym.make(env_id)
            if seed is not None:
                try:
                    env.reset(seed=seed + rank)
                except Exception:
                    pass
            
            env = RepeatAction(env, repeat=2)
            env = Monitor(env)
            
            # --- THIS IS THE NEW LOGIC ---
            if use_cpu_preprocessing:
                # If we're on CPU, we add these wrappers
                env = ResizeObservation(env, (84, 84))
                env = GrayscaleObservation(env, keep_dim=True)
            # --- END NEW LOGIC ---
            
            return env
        return _init
    
    env_fns = [make_env(i) for i in range(n_envs)]
    
    if not use_subproc:
        vec = DummyVecEnv(env_fns)
    else:
        vec = SubprocVecEnv(env_fns)
    
    if frame_stack and frame_stack > 1:
        vec = VecFrameStack(vec, n_stack=frame_stack)
        
    return vec