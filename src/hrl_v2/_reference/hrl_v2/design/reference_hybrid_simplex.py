"""Executable mathematical reference, NOT an integrated UAV training program.

Seven non-empty supports for a three-destination routing row. Conditional
ratios use Dirichlet distributions; probability density is relative to the
explicit reference measure: uniform support and uniform density on each face.
No Euclidean projection, Gumbel straight-through estimator, ratio threshold,
or renormalization after sampling is used.

All core reference calculations use float64. This module requires PyTorch.
Caller owns RNG state and optimizer management. The row-enumerated objective
is a gradient surrogate, not the numerical value of the joint actor objective.
"""
from typing import Callable, Optional, Tuple
import math
import torch
from torch import Tensor
from torch.distributions import Categorical, Dirichlet

SUPPORTS: Tuple[Tuple[int, ...], ...] = (
    (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)
)


def concentration_from_raw(raw: Tensor) -> Tensor:
    """Proposed v2 uses kappa > 1 for numerical regularity, not sparse ratios."""
    return 1.0 + torch.nn.functional.softplus(raw.to(torch.float64))


def _validate(logits: Tensor, concentration: Tensor) -> None:
    if logits.ndim != 3 or logits.shape[1:] != (6, 7):
        raise ValueError('logits must have shape [batch, 6, 7]')
    if concentration.shape != (*logits.shape, 3):
        raise ValueError('concentration must have shape [batch, 6, 7, 3]')
    if logits.dtype != torch.float64 or concentration.dtype != torch.float64:
        raise TypeError('reference functions require float64 parameters')
    if logits.device != concentration.device:
        raise ValueError('parameters must use the same device')
    if not torch.isfinite(logits).all() or not torch.isfinite(concentration).all():
        raise ValueError('parameters must be finite')
    if not (concentration > 0).all():
        raise ValueError('concentrations must be positive')


def all_face_samples(concentration: Tensor) -> Tuple[Tensor, Tensor]:
    """Return A[B,6,7,3] and log f/f_uniform [B,6,7].

    Generate one pathwise sample per non-singleton face, in SUPPORTS order.
    Singleton faces have exactly zero continuous log-density and no random draw.
    """
    if concentration.ndim != 4 or concentration.shape[1:] != (6, 7, 3):
        raise ValueError('wrong concentration shape')
    actions, logs = [], []
    b, r = concentration.shape[:2]
    for m, support in enumerate(SUPPORTS):
        k = len(support)
        if k == 1:
            action = concentration.new_zeros((b, r, 3))
            action[..., support[0]] = 1.0
            log_cont = concentration.new_zeros((b, r))
        else:
            dist = Dirichlet(concentration[:, :, m, list(support)], validate_args=True)
            part = dist.rsample()
            # Do not clamp or silently remove an underflowed positive fraction.
            if not torch.isfinite(part).all() or not (part > 0).all():
                raise FloatingPointError('invalid Dirichlet sample; no route repair applied')
            action = concentration.new_zeros((b, r, 3))
            action[..., list(support)] = part
            log_cont = dist.log_prob(part) - math.lgamma(k)
        actions.append(action)
        logs.append(log_cont)
    return torch.stack(actions, dim=2), torch.stack(logs, dim=2)


def sample_joint(logits: Tensor, concentration: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Return routes[B,6,3], support_ids[B,6], relative row log-density[B,6].

    The reference deliberately samples all conditional faces before gathering
    chosen supports. Optimized implementations may sample chosen faces only,
    but must record that RNG schedule and cannot claim bitwise equivalence.
    """
    _validate(logits, concentration)
    all_actions, all_cont_logs = all_face_samples(concentration)
    ids = Categorical(logits=logits).sample()
    routes = all_actions.gather(2, ids[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
    continuous = all_cont_logs.gather(2, ids[..., None]).squeeze(2)
    logq = logits.log_softmax(-1).gather(2, ids[..., None]).squeeze(2)
    return routes, ids, logq + math.log(7.0) + continuous


def log_relative_density(logits: Tensor, concentration: Tensor,
                         routes: Tensor, ids: Tensor) -> Tensor:
    """Recompute per-row density, checking the explicitly stored support."""
    _validate(logits, concentration)
    if routes.shape != (*logits.shape[:2], 3) or ids.shape != logits.shape[:2]:
        raise ValueError('wrong action/support shape')
    if ((ids < 0) | (ids > 6)).any():
        raise ValueError('invalid support id')
    if not torch.isfinite(routes).all() or (routes < 0).any():
        raise ValueError('invalid route')
    if not torch.allclose(routes.sum(-1), torch.ones_like(routes[..., 0]), atol=1e-12, rtol=0):
        raise ValueError('row sums differ from one')
    result = logits.log_softmax(-1).gather(2, ids[..., None]).squeeze(2) + math.log(7.0)
    for m, support in enumerate(SUPPORTS):
        mask = ids == m
        if not mask.any():
            continue
        selected = routes[mask]
        outside = [j for j in range(3) if j not in support]
        if outside and not (selected[:, outside] == 0).all():
            raise ValueError('nonzero ratio outside selected support')
        part = selected[:, list(support)]
        if not (part > 0).all():
            raise ValueError('support contains zero ratio')
        if len(support) > 1:
            pars = concentration[:, :, m, :][mask][:, list(support)]
            result = result.clone()
            result[mask] = result[mask] + Dirichlet(pars).log_prob(part) - math.lgamma(len(support))
    return result


def deterministic_decode(logits: Tensor, concentration: Tensor) -> Tensor:
    """Secondary evaluation only: argmax support, conditional Dirichlet mean.

    torch.argmax breaks exact ties by the first SUPPORTS entry. It is not the
    expectation of the full mixture and is not used for primary sampled tests.
    """
    _validate(logits, concentration)
    ids = logits.argmax(-1)
    routes = concentration.new_zeros((*logits.shape[:2], 3))
    for m, support in enumerate(SUPPORTS):
        mask = ids == m
        if not mask.any():
            continue
        part = concentration[:, :, m, :][mask][:, list(support)]
        part = part / part.sum(-1, keepdim=True)
        rows = concentration.new_zeros((part.shape[0], 3))
        rows[:, list(support)] = part
        routes[mask] = rows
    return routes


def row_enumerated_actor_surrogate(
    logits: Tensor,
    concentration: Tensor,
    critic: Callable[[Tensor], Tensor],
    alpha: float,
    sample_weights: Optional[Tensor] = None,
) -> Tensor:
    """Gradient estimator for E[alpha*sum_r ell_r - Q(s,A)].

    Each row enumerates its seven supports; other rows are sampled and detached.
    SUM the six row contributions (do not average by six). Non-action critic
    inputs and critic parameters must be frozen by the caller. Continuous face
    samples are reparameterized. Exactly 42 Q calls in this readable reference.
    A production version should cache the frozen critic report embedding once
    per state and batch these small Q-head calls; do not repeat the graph encoder.

    Its expected gradient equals the joint-objective actor gradient under the
    factorized routing policy, subject to ordinary pathwise regularity. Its
    scalar value counts the Q term six times and is NOT a reporting objective.
    """
    _validate(logits, concentration)
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError('alpha must be finite and nonnegative')
    faces, log_cont = all_face_samples(concentration)
    logq = logits.log_softmax(-1)
    q = logq.exp()
    ids = Categorical(logits=logits).sample()
    base = faces.gather(2, ids[..., None, None].expand(-1, -1, 1, 3)).squeeze(2).detach()
    loss = logits.new_zeros(logits.shape[0])
    for row in range(6):
        for support_id in range(7):
            route = base.clone()
            route[:, row, :] = faces[:, row, support_id, :]
            value = critic(route)
            if value.shape != (logits.shape[0],):
                raise ValueError('critic must return [batch]')
            ell = logq[:, row, support_id] + math.log(7.0) + log_cont[:, row, support_id]
            loss = loss + q[:, row, support_id] * (alpha * ell - value)
    if sample_weights is not None:
        if sample_weights.shape != loss.shape or not torch.isfinite(sample_weights).all():
            raise ValueError('invalid sample weights')
        loss = loss * sample_weights.detach()
    return loss.mean()


def physical_frame_reward(costs: Tensor, scale: float = 100.0) -> Tensor:
    """New finite-horizon contract: five undiscounted 0.1 s costs per frame."""
    if costs.shape[-1] != 5 or not torch.isfinite(costs).all() or (costs < 0).any():
        raise ValueError('expected five finite nonnegative physical costs')
    if scale <= 0:
        raise ValueError('scale must be positive')
    return -costs.sum(-1) / scale


def finite_horizon_target(reward: Tensor, done: Tensor, next_soft_value: Tensor) -> Tensor:
    """gamma=1; handle terminal states without evaluating a terminal policy."""
    if not (reward.shape == done.shape == next_soft_value.shape):
        raise ValueError('shape mismatch')
    if done.dtype != torch.bool:
        raise TypeError('done must be boolean')
    # Select terminal zeros before arithmetic to avoid 0 * NaN.
    continuation = torch.where(done, torch.zeros_like(next_soft_value), next_soft_value)
    result = reward + continuation
    if not torch.isfinite(result).all():
        raise FloatingPointError('nonfinite nonterminal target')
    return result
