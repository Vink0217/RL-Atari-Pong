import gymnasium as gym
import ale_py
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation
from gymnasium import Wrapper

class RepeatAction(Wrapper):
    """
    Custom wrapper that repeats the chosen action for `repeat` frames.
    The cumulative reward across those frames is returned.
    Useful to smooth out twitchy agent behavior (like in Pong).
    """
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

def make_atari_env(env_id: str, n_envs: int = 1, frame_stack: int = 4, use_subproc: bool = True, seed: int = None):
    """Create a vectorized Atari environment with common wrappers.

    Args:
        env_id: Gym env id.
        n_envs: Number of parallel envs.
        frame_stack: Number of frames to stack.
        use_subproc: Whether to use SubprocVecEnv.
        seed: Optional base seed. If provided, each worker will be seeded with seed + rank.
    """
    gym.register_envs(ale_py)
    def make_env(rank: int = 0):
        def _init():
            env = gym.make(env_id)
            # Seed the environment if requested (gymnasium supports seeding via reset)
            if seed is not None:
                try:
                    env.reset(seed=seed + rank)
                except Exception:
                    # fallback for envs that may not support reset(seed=...)
                    try:
                        env.seed(seed + rank)
                    except Exception:
                        pass
            env = RepeatAction(env, repeat=2)
            env = Monitor(env)
            # optional preprocessing: resize + grayscale
            try:
                env = ResizeObservation(env, (84, 84))
                env = GrayscaleObservation(env, keep_dim=True)
            except Exception:
                pass
            return env
        return _init
    env_fns = [make_env(i) for i in range(n_envs)]
    # Prefer the requested vectorization backend. Previously we forced
    # DummyVecEnv when n_envs == 1 which can produce a mismatch when the
    # training env uses SubprocVecEnv. Use the user's `use_subproc` choice
    # to decide which vec wrapper to create even for a single env.
    if not use_subproc:
        vec = DummyVecEnv(env_fns)
    else:
        vec = SubprocVecEnv(env_fns)
    
    if frame_stack and frame_stack > 1:
        vec = VecFrameStack(vec, n_stack=frame_stack)
    return vec