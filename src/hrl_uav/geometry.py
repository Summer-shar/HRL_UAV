"""Load-aligned coordinates; this rotates actions, it does not prescribe them."""

import numpy as np


def cooperation_aligned_basis(centroids, uav_coords, offered_loads):
    centroids = np.asarray(centroids, dtype=np.float64)
    uav_coords = np.asarray(uav_coords, dtype=np.float64)
    offered_loads = np.asarray(offered_loads, dtype=np.float64).reshape(-1)
    if centroids.shape != uav_coords.shape or centroids.ndim != 2:
        raise ValueError("centroids and uav_coords must have matching (N,2) shapes")
    if offered_loads.size != len(centroids):
        raise ValueError("one current offered load is required per UAV")
    heavy, light = int(np.argmax(offered_loads)), int(np.argmin(offered_loads))
    partners = np.full(len(centroids), heavy, dtype=np.int64)
    partners[heavy] = light
    parallel = uav_coords[partners] - centroids
    norms = np.linalg.norm(parallel, axis=1, keepdims=True)
    fallback = np.zeros_like(parallel)
    fallback[:,0] = 1.0
    parallel = np.where(norms > 1e-8, parallel/np.maximum(norms,1e-8), fallback)
    perpendicular = np.stack((-parallel[:,1],parallel[:,0]),axis=1)
    return np.stack((parallel,perpendicular),axis=2).astype(np.float32), partners
