"""TQC value learning with the existing direct routing and Gaussian actor.

Algorithm: Kuznetsov et al., ICML 2020, https://arxiv.org/abs/2005.04269.
Loss reduction follows SB3-Contrib TQC (mean over all pairwise terms).
The HRL update cadence, optimizer settings and routing projection stay local.
"""

import torch
from torch import nn

from .sac_agent import CriticNet, device, value_lr


def truncated_target(q1, q2, drop_per_critic):
    if q1.ndim != 2 or q1.shape != q2.shape:
        raise ValueError("TQC expects two [batch, quantiles] predictions")
    if type(drop_per_critic) is not int or not 0 <= drop_per_critic < q1.shape[1]:
        raise ValueError("TQC truncation must retain at least one quantile per critic")
    ordered = torch.cat((q1, q2), dim=1).sort(dim=1).values
    return ordered[:, :ordered.shape[1] - 2 * drop_per_critic]


def quantile_loss(q1, q2, target):
    if q1.ndim != 2 or q1.shape != q2.shape or target.ndim != 2 or target.shape[0] != q1.shape[0]:
        raise ValueError("Invalid TQC batch/quantile shapes")
    prediction = torch.stack((q1, q2), dim=1)
    error = target.detach()[:, None, None, :] - prediction[..., None]
    magnitude = error.abs()
    huber = torch.where(magnitude <= 1, 0.5 * error.square(), magnitude - 0.5)
    tau = (torch.arange(q1.shape[1], device=q1.device, dtype=q1.dtype) + 0.5) / q1.shape[1]
    weight = (tau[None, None, :, None] - (error.detach() < 0).to(q1.dtype)).abs()
    return (weight * huber).mean()


def actor_value(q1, q2):
    # TQC's actor uses the mean of ALL current quantiles, without truncation.
    return torch.stack((q1, q2), dim=1).mean(dim=(1, 2)).unsqueeze(1)


class QuantileCritic:
    def __init__(self, state_dim, action_dim, route_width, quantiles, drop_per_critic):
        if type(quantiles) is not int or quantiles < 2:
            raise ValueError("TQC requires at least two quantiles per critic")
        if type(drop_per_critic) is not int or not 0 <= drop_per_critic < quantiles:
            raise ValueError("Invalid TQC drop_per_critic")
        self.quantiles = quantiles
        self.drop_per_critic = drop_per_critic
        self.critic_v = CriticNet(state_dim, action_dim, route_width, quantiles).to(device)
        self.target_critic_v = CriticNet(state_dim, action_dim, route_width, quantiles).to(device)
        self.target_critic_v.load_state_dict(self.critic_v.state_dict())
        self.optimizer = torch.optim.Adam(self.critic_v.parameters(), lr=value_lr, eps=1e-5)

    def get_v(self, state, action):
        return self.critic_v(state, action)

    def learn(self, q1, q2, target, max_grad_norm=0.0):
        loss = quantile_loss(q1, q2, target)
        self.optimizer.zero_grad()
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.critic_v.parameters(), max_grad_norm if max_grad_norm > 0 else float("inf"))
        self.optimizer.step()
        return float(norm)
