"""
Optimized Atari wrappers matching SB3-Zoo's high-performance setup.
Uses cv2-based frame processing and efficient frame skip handling.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2
cv2.ocl.setUseOpenCL(False)  # disable OpenCL to prevent thread/GPU conflicts

class FastAtariWrapper(gym.Wrapper):
    """
    Optimized Atari preprocessing wrapper combining multiple operations:
    - Convert to grayscale and resize in a single cv2 operation
    - Handle frameskip at the lowest level
    - Optimize memory allocation
    """
    def __init__(
        self,
        env: gym.Env,
        noop_max: int = 30,
        frame_skip: int = 4,
        screen_size: int = 84,
        terminal_on_life_loss: bool = True,
    ):
        super().__init__(env)
        assert frame_skip > 0
        assert screen_size > 0

        self.noop_max = noop_max
        self.frame_skip = frame_skip
        self.screen_size = screen_size
        self.terminal_on_life_loss = terminal_on_life_loss
        
        # Keep RGB channels for GPU preprocessing
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(screen_size, screen_size, 3),
            dtype=np.uint8,
        )
        
        # Preallocate buffers for max efficiency
        self.game_over = False
        self._lives = 0
        self._last_frame = None
        self.max_frame = None  # for maxing over pairs
        
    def _get_ob(self, frame):
        """Resize frame maintaining RGB channels."""
        # Efficient resize keeping RGB channels
        frame = cv2.resize(
            frame, (self.screen_size, self.screen_size),
            interpolation=cv2.INTER_AREA
        )
        return frame  # Keep RGB channels (H, W, 3)

    def reset(self, **kwargs):
        """Do no-op resets and handle lives."""
        obs, info = self.env.reset(**kwargs)
        self._lives = info.get('lives', 0)
        
        if self.noop_max > 0:
            # Do random number of no-ops on reset
            noops = self.env.unwrapped.np_random.integers(1, self.noop_max + 1)
            for _ in range(noops):
                obs, _, terminated, truncated, info = self.env.step(0)
                if terminated or truncated:
                    obs, info = self.env.reset(**kwargs)
        
        self.game_over = False
        self._last_frame = self._get_ob(obs)
        return self._last_frame, info

    def step(self, action):
        """Execute action and process frames."""
        total_reward = 0.0
        self.max_frame = None

        for i in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            current_lives = info.get('lives', 0)
            
            # Max pool over last two frames (only if not final frame)
            if i == self.frame_skip - 2:
                self.max_frame = self._get_ob(obs)
            elif i == self.frame_skip - 1:
                if self.max_frame is not None:
                    processed = self._get_ob(obs)
                    self._last_frame = np.maximum(self.max_frame, processed)
                else:
                    self._last_frame = self._get_ob(obs)
            
            done = terminated or truncated
            if self.terminal_on_life_loss:
                lost_life = current_lives < self._lives
                self._lives = current_lives
                done = done or lost_life
            
            if done:
                break

        return self._last_frame, total_reward, terminated, truncated, info