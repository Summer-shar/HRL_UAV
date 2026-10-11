"""Read-only follow-up: how much the upper layer can change upload delay.

Upload is a per-terminal FIFO at a fixed rate (dedicated B/39 per terminal), independent of
routing, so the upload stage is replayed exactly from arrivals for any fixed UAV position.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hrl_uav.industrial_hotspot_env import HotspotIndustrialEnv
from hrl_v2.evaluate import gate_manifest
from analyze import BASE

OUT = Path(__file__).resolve().parent
scene_cfg = json.loads((ROOT/'src/hrl_v2/configs/physical_scene.json').read_text(encoding='utf-8'))
T, W = 900., (10., 1.)


def upload_residence(physics, arrivals, uav_xy):
    """Weighted mean upload residence (s) with fixed UAV positions, continuous-time FIFO per terminal."""
    rates = physics.compute_u2g_rate_bps(uav_coords=np.asarray(uav_xy, float))
    total = pop = 0.
    free = np.zeros(39)
    for a in arrivals:  # already in birth order, task_id order within a frame
        start = max(free[a.user_id], a.birth_s)
        free[a.user_id] = start + a.input_bits/rates[a.user_id]
        w = W[a.priority]
        total += w*(min(free[a.user_id], T)-a.birth_s); pop += w
    return total/pop, rates


def main():
    env = HotspotIndustrialEnv(scene_cfg)
    init = np.array(scene_cfg['base_scene']['uav_initial_positions_m'], float)
    rows = []
    for sc in gate_manifest(20)['scenarios']:
        sid = sc['scenario']
        env.reset(int(sc['arrival_seed']), int(sc['layout_seed']), int(sc['load_seed']))
        arrivals = [a for k in range(1800) for a in env._scene.frame(k)]
        users, src = env.physics.user_coords, env.physics.user_allocations
        bits = np.zeros(39)
        for a in arrivals: bits[a.user_id] += a.input_bits
        cen = np.array([np.average(users[src == r], axis=0, weights=bits[src == r]) for r in range(3)])
        up_init, r_init = upload_residence(env.physics, arrivals, init)
        up_cen, _ = upload_residence(env.physics, arrivals, cen)
        best = cen.copy()
        for _ in range(2):  # coordinate descent; regions are independent (own terminals only)
            for r in range(3):
                grid = [best[r] + np.array([dx, dy]) for dx in range(-30, 31, 5) for dy in range(-30, 31, 5)]
                scores = []
                for g in grid:
                    trial = best.copy(); trial[r] = g
                    scores.append(upload_residence(env.physics, arrivals, trial)[0])
                best[r] = grid[int(np.argmin(scores))]
        up_best, _ = upload_residence(env.physics, arrivals, best)
        m = json.loads((BASE/f'baselines/hover_equal_{sid:02d}_1.json').read_text())['metrics']
        c = json.loads((BASE/f'baselines/centroid_equal_{sid:02d}_1.json').read_text())['metrics']
        rows.append(dict(scenario=sid, measured_upload_hover_equal=m['stage_delay_s']['upload'],
                         measured_upload_centroid_equal=c['stage_delay_s']['upload'],
                         measured_delay_centroid_equal=c['truncated_delay_s'],
                         replay_upload_initial=up_init, replay_upload_bit_centroid=up_cen,
                         replay_upload_best_static=up_best,
                         uav_best_static=best.tolist(), uav_bit_centroid=cen.tolist(),
                         dist_init_to_best_m=np.linalg.norm(best-init, axis=1).tolist(),
                         uplink_mbps_initial=[float(r_init.min()/1e6), float(r_init.mean()/1e6), float(r_init.max()/1e6)]))
        print(sid, {k: (round(v, 4) if isinstance(v, float) else [round(x, 1) for x in v])
                    for k, v in rows[-1].items() if k.startswith(('measured', 'replay', 'dist'))}, flush=True)
    (OUT/'upper_upload.json').write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
