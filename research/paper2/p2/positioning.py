"""Does the upper layer position UAVs for cooperation, beyond tracking its users?

Uses the recorded test trajectories (UAV and user positions every 5 s) and the
per-run computation shares.  For every run and decision time t:

* tracking error      |u_k - c_k|, c_k = centroid of the users served by UAV k;
* spread ratio        mean pairwise UAV distance / mean pairwise centroid
                      distance (< 1: UAVs stay closer to each other than their
                      user groups are, i.e. shorter U2U links);
* lean toward partner projection of (u_k - c_k) onto the unit vector from c_k to
                      the centroid of the partner region (heaviest <-> lightest
                      arrival load of that run), in metres; > 0 means the UAV
                      sits on the side of its users facing the partner.
The first 60 s are skipped (UAVs start at fixed positions).
"""
import numpy as np

from . import common as C

WARMUP_S = 60.


def run_metrics(npz, cycles_by_source):
    t, uav, user, src = npz['time_s'], npz['uav_xy'], npz['user_xy'], npz['user_source']
    keep = t >= min(WARMUP_S, t[-1]/2)   # short smoke windows keep their second half
    cent = np.stack([user[:, src == k].mean(axis=1) for k in range(uav.shape[1])], axis=1)  # T x 3 x 2
    track = np.linalg.norm(uav-cent, axis=-1)                                              # T x 3
    pairs = [(0, 1), (0, 2), (1, 2)]
    d_uav = np.mean([np.linalg.norm(uav[:, i]-uav[:, j], axis=-1) for i, j in pairs], axis=0)
    d_cen = np.mean([np.linalg.norm(cent[:, i]-cent[:, j], axis=-1) for i, j in pairs], axis=0)
    out = dict(time_s=t, spread_ratio=d_uav/np.maximum(d_cen, 1e-9), tracking_m=track.mean(axis=1))
    if cycles_by_source:
        heavy, light = int(np.argmax(cycles_by_source)), int(np.argmin(cycles_by_source))

        def lean(k, partner):
            direction = cent[:, partner]-cent[:, k]
            direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-9)
            return ((uav[:, k]-cent[:, k])*direction).sum(-1)
        out.update(lean_light_m=lean(light, heavy), lean_heavy_m=lean(heavy, light),
                   heavy=heavy, light=light)
    out['summary'] = {k: float(np.mean(v[keep])) for k, v in out.items()
                      if isinstance(v, np.ndarray) and k != 'time_s'}
    return out


def collect(root, seeds, methods):
    rows = []
    for seed in seeds:
        for m in methods:
            folder = C.seed_dir(root, seed)/'test'/m
            for path in sorted((folder/'traj').glob('*.npz')):
                rec = folder/f'{path.stem}.json'
                shares = C.read_json(rec).get('cycles_by_source') if rec.exists() else None
                with np.load(path) as npz:
                    r = run_metrics(npz, shares)
                rows.append(dict(seed=seed, method=m, scenario=int(path.stem), **r))
    return rows


def summary(rows, methods):
    out = {}
    for m in methods:
        rs = [r['summary'] for r in rows if r['method'] == m]
        if rs:
            out[m] = {k: float(np.mean([r[k] for r in rs if k in r])) for k in rs[0]}
            out[m]['runs'] = len(rs)
    return out
