# --- (Keep your RepeatAction wrapper class at the top) ---
import gymnasium as gym
import ale_py
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from .fast_atari_wrappers import FastAtariWrapper


def make_atari_env(
    env_id: str, 
    n_envs: int = 1, 
    frame_stack: int = 4, 
    use_subproc: bool = True, 
    seed: int = None,
    terminal_on_life_loss: bool = True,
    use_efficient_wrappers: bool = True,
):
    """
    Create a vectorized Atari environment with optimized wrappers.
    - Efficient cv2-based frame processing
    - Built-in frameskip (4 frames)
    - Max-pooling over pairs of frames
    - Episode end on life loss (helps exploration)
    """
    gym.register_envs(ale_py)
    
    def make_env(rank: int = 0):
        def _init():
            # Make env with explicit frameskip setting
            env = gym.make(
                env_id,
                frameskip=1,  # We'll handle frame skip in wrapper
                render_mode=None,
                obs_type="rgb",  # Ensure RGB observations
            )
            # Seed if provided
            if seed is not None:
                try:
                    env.reset(seed=seed + rank)
                except Exception:
                    pass
            
            # Add performance monitoring
            env = Monitor(env)
            
            # Add optimized preprocessing
            env = FastAtariWrapper(
                env,
                noop_max=30,  # Random noops on reset
                frame_skip=4,  # Built-in efficient frame skip
                screen_size=84,
                terminal_on_life_loss=terminal_on_life_loss,
            )
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