import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision import transforms

class GpuPreprocessingFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom features extractor that performs image preprocessing
    (grayscale, resize) on the GPU using torchvision transforms.
    Expects (N, H, W, 12) (4 stacked RGB frames).
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        # Be robust to different observation_space shapes. Channels may be
        # the last dimension (H, W, C) or the middle one (H, C, W) depending
        # on how the Vec/FrameStack wrappers were applied. Look for a small
        # channel-like dimension (1, 3, 4, or 12). If none match, fall back
        # to the last dimension and include the full shape in the error for
        # easier debugging.
        shape = observation_space.shape
        channel_candidates = [d for d in shape if d in (1, 3, 4, 12)]
        if channel_candidates:
            n_input_channels = channel_candidates[0]
        else:
            n_input_channels = shape[-1]

        # Now we expect 12 channels (4 stacked frames × 3 RGB channels)
        expected_channels = 12
        if n_input_channels != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} input channels (4 frames x 3 RGB), got {n_input_channels}. "
                f"Observation space shape: {shape}. Make sure frame_stack=4 and RGB frames."
            )
        
        super().__init__(observation_space, features_dim)

        self.preprocessor = nn.Sequential(
            _StackGrayscale(),  # Converts 12 channels (4×RGB) to 4 channels
            nn.BatchNorm2d(4),  # Normalize for stable training
        )

        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_obs = torch.zeros(
                (1, *observation_space.shape), 
                dtype=torch.float32
            )
            preprocessed = self._forward_pre(dummy_obs)
            cnn_out = self.cnn(preprocessed)
            # cnn_out.shape[1] is the flattened output size of the CNN. We
            # need to remember it for the linear layer, but the BaseFeaturesExtractor
            # expects `self._features_dim` to be the FINAL output size of the
            # features extractor (i.e. `features_dim`), so keep them separate.
            self._cnn_flattened_dim = cnn_out.shape[1]

        # Tell the base class the output feature dim (what the policy will see).
        self._features_dim = features_dim

        # Linear maps from the flattened CNN output to the desired features_dim.
        self.linear = nn.Sequential(nn.Linear(self._cnn_flattened_dim, features_dim), nn.ReLU())

    def _forward_pre(self, observations: torch.Tensor) -> torch.Tensor:
        # Accept either (N, H, W, C) or (N, C, H, W). Normalize to float and
        # convert to channel-first (N, C, H, W) which our preprocessor expects.
        if observations.ndim != 4:
            raise ValueError(f"Unexpected observation tensor shape: {observations.shape}")

        # If channels already in dim=1 (N, C, H, W), keep as-is. Otherwise,
        # assume (N, H, W, C) and permute.
        if observations.shape[1] in (1, 3, 4, 12):
            x = observations
        else:
            x = observations.permute(0, 3, 1, 2)

        x = x.float() / 255.0
        return self.preprocessor(x)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        preprocessed_obs = self._forward_pre(observations)
        return self.linear(self.cnn(preprocessed_obs))

class _StackGrayscale(nn.Module):
    """
    Helper module to apply Grayscale to a stack of 4 RGB images.
    Input: (N, 12, H, W) -> Output: (N, 4, H, W)
    """
    def __init__(self):
        super().__init__()
        self.grayscale = transforms.Grayscale(num_output_channels=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1, c2, c3, c4 = torch.split(x, 3, dim=1)
        g1 = self.grayscale(c1)
        g2 = self.grayscale(c2)
        g3 = self.grayscale(c3)
        g4 = self.grayscale(c4)
        return torch.cat([g1, g2, g3, g4], dim=1)


class GpuPreprocessedCnnPolicy(ActorCriticCnnPolicy):
    """
    A CnnPolicy that uses our custom GpuPreprocessingFeaturesExtractor.
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule,
        *args,
        **kwargs,
    ):
        kwargs["features_extractor_class"] = GpuPreprocessingFeaturesExtractor
        kwargs["features_extractor_kwargs"] = dict(features_dim=512)
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )