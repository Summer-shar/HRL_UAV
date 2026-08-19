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


def priority_gated_to_simplex(
    values,
    priority_local_biases,
    logit_scale=4.0,
):
    """Map priority-aware latent actions to routing simplices.

    The first latent variable of each source row is an explicit local-service
    gate. Remaining variables distribute the offloaded share among peers via
    centered stick breaking. ``priority_local_biases`` encodes only the prior
    operating point; every gate remains controllable by the learned action.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3 or array.shape[0] <= 0 or array.shape[1] <= 1:
        raise ValueError(
            "values must have shape (priorities, sources, sources - 1)"
        )
    priorities, sources, width = array.shape
    if width != sources - 1:
        raise ValueError(
            "The final latent width must equal the number of peers."
        )
    if np.any(~np.isfinite(array)):
        raise ValueError("values must be finite.")
    biases = np.asarray(priority_local_biases, dtype=np.float64).reshape(-1)
    if biases.size != priorities or np.any(~np.isfinite(biases)):
        raise ValueError("priority_local_biases must contain one finite value per priority.")
    scale = float(logit_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("logit_scale must be finite and positive.")

    gate_logits = np.clip(
        biases[:, None] + scale * array[..., 0],
        -40.0,
        40.0,
    )
    local_fraction = 1.0 / (1.0 + np.exp(-gate_logits))
    remote_count = sources - 1
    if remote_count == 1:
        remote_distribution = np.ones(
            (priorities, sources, 1),
            dtype=np.float64,
        )
    else:
        remote_distribution = centered_stick_breaking_to_simplex(
            array[..., 1:],
            logit_scale=scale,
        )

    output = np.zeros((priorities, sources, sources), dtype=np.float64)
    for source in range(sources):
        peers = [destination for destination in range(sources) if destination != source]
        output[:, source, source] = local_fraction[:, source]
        output[:, source, peers] = (
            (1.0 - local_fraction[:, source, None])
            * remote_distribution[:, source, :]
        )
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
