import numpy as np





































































































def project_rows_to_simplex(values):
    """Euclidean projection of the final axis onto the probability simplex."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] <= 0:
        raise ValueError("values must have a non-empty final axis.")
    if np.any(~np.isfinite(array)):
        raise ValueError("values must be finite.")

    width = array.shape[-1]
    flat = array.reshape(-1, width)
    ordered = np.sort(flat, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1) - 1.0
    ranks = np.arange(1, width + 1, dtype=np.float64)
    active = ordered - cumulative / ranks > 0.0
    support_size = np.maximum(np.sum(active, axis=1), 1)
    threshold = cumulative[
        np.arange(flat.shape[0]),
        support_size - 1,
    ] / support_size
    projected = np.maximum(flat - threshold[:, None], 0.0)
    projected /= np.maximum(
        np.sum(projected, axis=1, keepdims=True),
        1e-12,
    )
    return projected.reshape(array.shape)


def project_rows_to_simplex_tensor(values):
    """Differentiable counterpart of the physical projection, off its kinks."""
    import torch

    width = values.shape[-1]
    ordered = values.sort(dim=-1, descending=True).values
    cumulative = ordered.cumsum(dim=-1)-1.0
    ranks = torch.arange(1, width+1, device=values.device, dtype=values.dtype)
    support = (ordered-cumulative/ranks > 0).sum(dim=-1, keepdim=True).clamp_min(1)
    threshold = cumulative.gather(-1, support-1)/support.to(values.dtype)
    projected = (values-threshold).clamp_min(0)
    return projected/projected.sum(dim=-1, keepdim=True).clamp_min(1e-12)
