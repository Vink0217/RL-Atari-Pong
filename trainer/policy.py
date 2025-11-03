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
        n_input_channels = observation_space.shape[2]
        if n_input_channels != 12:
            raise ValueError(
                f"Expected 12 input channels (4 stacked RGB frames), got {n_input_channels}"
            )
        
        super().__init__(observation_space, features_dim)

        self.preprocessor = nn.Sequential(
            _StackGrayscale(),
            transforms.Resize((84, 84), antialias=True)
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
            self._features_dim = cnn_out.shape[1]

        self.linear = nn.Sequential(nn.Linear(self._features_dim, features_dim), nn.ReLU())

    def _forward_pre(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations.permute(0, 3, 1, 2)
        observations = observations.float() / 255.0
        return self.preprocessor(observations)

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