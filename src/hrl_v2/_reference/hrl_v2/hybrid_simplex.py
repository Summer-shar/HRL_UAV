"""CPU float64 hybrid action distribution with isolated sampling streams."""
import math
import torch
from .design.reference_hybrid_simplex import (
    SUPPORTS, concentration_from_raw, all_face_samples, log_relative_density,
    deterministic_decode, physical_frame_reward, finite_horizon_target, _validate,
)


def sample(logits, concentration, streams, stream=0):
    _validate(logits, concentration)
    if logits.device.type != "cpu":
        raise ValueError("V2 distributions are on CPU")
    with streams.fork("ratio", stream):
        faces, cont = all_face_samples(concentration)
    ids = torch.multinomial(logits.softmax(-1).reshape(-1,7), 1,
                           generator=streams.generator("support",stream)).reshape(logits.shape[:2])
    actions = faces.gather(2,ids[...,None,None].expand(-1,-1,1,3)).squeeze(2)
    ell = logits.log_softmax(-1).gather(2,ids[...,None]).squeeze(2)+math.log(7)
    ell = ell + cont.gather(2,ids[...,None]).squeeze(2)
    return actions, ids, ell


def action_features(actions, device):
    a = actions.to(device=device, dtype=torch.float32)
    error = (a.to(device="cpu",dtype=torch.float64)-actions.to("cpu")).abs().max()
    if error.item() > 1e-7 or ((actions > 0) & (a.to("cpu") == 0)).any():
        raise FloatingPointError("Execution action loses support or precision in Q")
    return torch.cat((a.flatten(-2), (a.detach()>0).float().flatten(-2)), -1), error.item()


def surrogate(logits, concentration, critic, alpha, streams, weights=None):
    """42 candidate heads in one call; six row contributions are SUMMED."""
    _validate(logits, concentration)
    with streams.fork("ratio",100):
        faces, cont = all_face_samples(concentration)
    logq = logits.log_softmax(-1)
    q = logq.exp()
    ids = torch.multinomial(q.reshape(-1,7),1,generator=streams.generator("support",100)).reshape(-1,6)
    base = faces.gather(2,ids[...,None,None].expand(-1,-1,1,3)).squeeze(2).detach()
    candidates = base[:,None,None].expand(-1,6,7,-1,-1).clone()
    for row in range(6):
        candidates[:,row,:,row] = faces[:,row]
    values = critic(candidates.flatten(1,2))
    if values.shape != (logits.shape[0],42):
        raise ValueError("Candidate critic must return B x 42")
    loss = (q * (alpha*(logq+math.log(7)+cont)-values.reshape(-1,6,7))).sum((1,2))
    if weights is not None:
        loss = loss * weights.detach().to(loss)
    return loss.mean()
