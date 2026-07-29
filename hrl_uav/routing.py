import numpy as np


def centered_stick_breaking_to_simplex(values, logit_scale=4.0):
    """Map N-1 bounded latent values bijectively into an N-way simplex."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] <= 0:
        raise ValueError("values must have a non-empty final axis.")
    if np.any(~np.isfinite(array)):
        raise ValueError("values must be finite.")
    scale = float(logit_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("logit_scale must be finite and positive.")

    width = array.shape[-1]
    logits = np.clip(array * scale, -40.0, 40.0)
    remaining_categories = np.arange(
        width,
        0,
        -1,
        dtype=np.float64,
    )
    centered_logits = logits - np.log(remaining_categories)
    fractions = np.empty_like(centered_logits)
    positive = centered_logits >= 0.0
    fractions[positive] = 1.0 / (
        1.0 + np.exp(-centered_logits[positive])
    )
    exp_logits = np.exp(centered_logits[~positive])
    fractions[~positive] = exp_logits / (1.0 + exp_logits)

    output = np.empty(array.shape[:-1] + (width + 1,), dtype=np.float64)
    remaining = np.ones(array.shape[:-1], dtype=np.float64)
    for index in range(width):
        output[..., index] = remaining * fractions[..., index]
        remaining = remaining * (1.0 - fractions[..., index])
    output[..., -1] = remaining
    output /= np.sum(output, axis=-1, keepdims=True)
    return output


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
