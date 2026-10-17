"""Fixed-vector candidate networks; graph networks remain untouched."""
import math
import torch
from torch import nn
from .networks import initialize, mlp
from .hybrid_simplex import concentration_from_raw
from .lite_observation import LOWER_DIM, UPPER_DIM, TERMINAL_DIM, GLOBAL_DIM


class LiteEncoder(nn.Module):
    kind = 'lite'
    def __init__(self, upper=False):
        super().__init__()
        self.upper = upper
        if upper:
            self.terminal = nn.Sequential(mlp(TERMINAL_DIM, 64, 64), nn.ReLU())
            self.body = nn.Sequential(mlp(3 * 128 + GLOBAL_DIM, 256, 256), nn.ReLU())

    def reset_inference(self):
        pass  # EMA is parameter-free and belongs to the collector's observer.

    def forward(self, obs):
        if not self.upper:
            return obs
        # 3 x K terminal rows (K = 13 in v2.4.0), then the global part; the
        # per-region mean/max pooling makes the weights independent of K.
        k = (obs.shape[-1] - GLOBAL_DIM) // (3 * TERMINAL_DIM)
        terminals = obs[..., :3 * k * TERMINAL_DIM].reshape(*obs.shape[:-1], 3, k, TERMINAL_DIM)
        embedded = self.terminal(terminals)
        pooled = torch.cat((embedded.mean(-2), embedded.max(-2).values), dim=-1)
        return self.body(torch.cat((pooled.flatten(-2), obs[..., -GLOBAL_DIM:]), dim=-1))


class VectorNetwork(nn.Module):
    def tensor(self, obs):
        # Online input is an immutable float32 vector; copy avoids writable
        # NumPy alias warnings. Training batches already own contiguous tensors.
        return torch.tensor(obs, dtype=torch.float32, device=next(self.parameters()).device)

    def batch_tensor(self, states):
        return states.tensor(next(self.parameters()).device)


class LiteLowerActor(VectorNetwork):
    def __init__(self):
        super().__init__()
        self.encoder = LiteEncoder()
        self.body = nn.Sequential(mlp(LOWER_DIM, 256, 256), nn.ReLU())
        self.support = nn.Linear(256, 6 * 7)
        self.concentration = nn.Linear(256, 6 * 7 * 3)
        initialize(self)
        nn.init.zeros_(self.support.weight)
        nn.init.zeros_(self.support.bias)
        nn.init.zeros_(self.concentration.weight)
        nn.init.constant_(self.concentration.bias, math.log(math.expm1(1)))

    def decode(self, obs):
        h = self.body(obs)
        logits = self.support(h).reshape(-1, 6, 7).to('cpu', dtype=torch.float64)
        raw = self.concentration(h).reshape(-1, 6, 7, 3).to('cpu')
        return logits, concentration_from_raw(raw)

    def forward(self, obs, history=None, **kwargs):
        logits, kappa = self.decode(self.tensor(obs)[None])
        return logits[0], kappa[0]

    def batch(self, states):
        return self.decode(self.batch_tensor(states))


class LiteQ(VectorNetwork):
    def __init__(self):
        super().__init__()
        self.encoder = LiteEncoder()
        self.head = mlp(LOWER_DIM + 36, 256, 256, 1)
        initialize(self)
        nn.init.orthogonal_(self.head[-1].weight, 1.)
        self.last_action_cast_error = 0.

    def encode_batch(self, states):
        return self.batch_tensor(states)

    def state_projection(self, encoded):
        return nn.functional.linear(encoded, self.head[0].weight[:, :LOWER_DIM], self.head[0].bias)

    def value_features(self, encoded, features, projected=None):
        projected = self.state_projection(encoded) if projected is None else projected
        if features.ndim == 3:
            projected = projected[:, None]
        action = nn.functional.linear(features, self.head[0].weight[:, LOWER_DIM:])
        return self.head[1:](projected + action).squeeze(-1)


class LiteUpperActor(VectorNetwork):
    def __init__(self):
        super().__init__()
        self.encoder = LiteEncoder(upper=True)
        self.mean = nn.Linear(256, 6)
        self.std = nn.Linear(256, 6)
        initialize(self)
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.std.weight)
        nn.init.constant_(self.std.bias, math.atanh((math.log(.15) + 2.5) / 2.5))

    def decode(self, obs):
        h = self.encoder(obs)
        return self.mean(h), -2.5 + 2.5 * self.std(h).tanh()

    def forward(self, obs, history=None, **kwargs):
        mean, std = self.decode(self.tensor(obs)[None])
        return mean[0], std[0]

    def batch(self, states):
        return self.decode(self.batch_tensor(states))


class LiteValue(VectorNetwork):
    def __init__(self):
        super().__init__()
        self.encoder = LiteEncoder(upper=True)
        self.head = nn.Linear(256, 1)
        initialize(self)
        nn.init.orthogonal_(self.head.weight, 1.)

    def forward(self, obs, history=None, **kwargs):
        return self.head(self.encoder(self.tensor(obs)[None])).squeeze()

    def batch(self, states):
        return self.head(self.encoder(self.batch_tensor(states))).squeeze(-1)
