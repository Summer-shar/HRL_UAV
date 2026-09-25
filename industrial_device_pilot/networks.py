"""PPO set encoders with device-order invariance inside each fixed source."""

import torch
from torch import nn
from torch.nn import functional as F

from .observation import (
    DEVICE_BLOCK_DIM,
    DEVICE_FEATURE_DIM,
    DEVICES_PER_SOURCE,
    GLOBAL_DIM,
    MACRO_DIM,
    SOURCE_COUNT,
)


NETWORK_ARCHITECTURE = "device_set_mean_max_phi32_head256_v1"
DEVICE_HIDDEN_DIM = 32
POOLED_DIM = SOURCE_COUNT * 2 * DEVICE_HIDDEN_DIM
HEAD_INPUT_DIM = POOLED_DIM + GLOBAL_DIM


class _SetBackbone(nn.Module):
    """Preserve each device's geometry/workload relation before pooling.

    Each source's device embeddings are reduced by mean and componentwise max;
    the three source groups remain ordered. This is not UAV permutation
    equivariance or a claim that pooling is a sufficient Markov statistic.
    """

    def __init__(self, obs_dim):
        super().__init__()
        if obs_dim != MACRO_DIM:
            raise ValueError(f"{NETWORK_ARCHITECTURE} requires {MACRO_DIM} input features")
        self.obs_dim = obs_dim
        self.device_phi = nn.Sequential(
            nn.Linear(DEVICE_FEATURE_DIM, DEVICE_HIDDEN_DIM), nn.ReLU(),
            nn.Linear(DEVICE_HIDDEN_DIM, DEVICE_HIDDEN_DIM), nn.ReLU(),
        )
        self.in_to_y1 = nn.Linear(HEAD_INPUT_DIM, 256)
        self.y1_to_y2 = nn.Linear(256, 256)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0., std=0.1)

    def pooled_features(self, inputstate):
        if inputstate.ndim < 1 or inputstate.shape[-1] != self.obs_dim:
            raise ValueError(f"Expected last observation dimension {self.obs_dim}")
        leading = inputstate.shape[:-1]
        devices = inputstate[..., :DEVICE_BLOCK_DIM].reshape(
            *leading, SOURCE_COUNT, DEVICES_PER_SOURCE, DEVICE_FEATURE_DIM)
        embeddings = self.device_phi(devices)
        # All operations inherit input/parameter device and dtype. No CPU helper
        # tensor is constructed in the forward path.
        pooled = torch.cat((embeddings.mean(dim=-2), embeddings.amax(dim=-2)), dim=-1)
        pooled = pooled.reshape(*leading, POOLED_DIM)
        return torch.cat((pooled, inputstate[..., DEVICE_BLOCK_DIM:]), dim=-1)

    def _hidden(self, inputstate):
        value = F.relu(self.in_to_y1(self.pooled_features(inputstate)))
        return F.relu(self.y1_to_y2(value))


class SetActorNet(_SetBackbone):
    def __init__(self, obs_dim=MACRO_DIM, action_dim=6):
        if action_dim != 6:
            raise ValueError("Device pilot requires the existing six movement actions")
        super().__init__(obs_dim)
        self.out = nn.Linear(256, action_dim)
        nn.init.normal_(self.out.weight, mean=0., std=0.1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.8, dtype=torch.float32))

    def forward(self, inputstate):
        mean = torch.tanh(self.out(self._hidden(inputstate)))
        std = self.log_std.clamp(-5., 1.).exp().expand_as(mean)
        return mean, std


class SetCriticNet(_SetBackbone):
    def __init__(self, obs_dim=MACRO_DIM, out_dim=1):
        if out_dim != 1:
            raise ValueError("Device pilot uses a scalar state value")
        super().__init__(obs_dim)
        self.out = nn.Linear(256, out_dim)
        nn.init.normal_(self.out.weight, mean=0., std=0.1)

    def forward(self, inputstate):
        return self.out(self._hidden(inputstate))
