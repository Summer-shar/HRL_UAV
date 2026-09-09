"""Read saved routing and resource ledgers; never load or update a model."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "results/ppo_only_unlock_seed1"
PURE = ROOT / "results/split_mdp_actor_lr3e5_resume600_to800_seed1"
ROLES = ("heavy", "medium", "light")


def inspect(folder, label, seeds):
    routes, role_routes, raw_scores, per_seed = [], [], [], []
    hashes = {}
    for seed in seeds:
        paths = [folder / f"{label}_{seed}_trajectory.npz", folder / f"{label}_{seed}_uavs.csv"]
        hashes.update({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
        with np.load(paths[0]) as data:
            route = data["executed_route"].copy()
            scores = data["raw_action"].reshape(-1, 2, 3, 3).copy()
        meter = pd.read_csv(paths[1])
        service = meter[meter.service_steps > 0].sort_values(["time_s", "uav"])
        assert len(service) == 3 * len(route)
        assert (service.groupby("time_s").size() == 3).all()
        assert np.allclose(route.sum(-1), 1, atol=1e-6)
        assert np.all(route >= -1e-8)
        role_ids = np.array([ROLES.index(role) for role in service.role]).reshape(-1, 3)
        order = np.argsort(role_ids, axis=1)
        aligned = np.stack([route[i][:, indices][:, :, indices] for i, indices in enumerate(order)])
        routes.append(route)
        role_routes.append(aligned)
        raw_scores.append(scores)
        record = {"seed": seed, "by_role": {}, "by_fixed_id": {}, "phase_destination_loads": {}}
        for name, group_key in (("by_role", "role"), ("by_fixed_id", "uav")):
            for key, rows in meter.groupby(group_key):
                seconds = len(rows) * 0.5
                record[name][str(key)] = {
                    "assigned_mbps_wall": float(rows.assigned_destination_mbit.sum() / seconds),
                    "computed_mbps_wall": float(rows.computed_mbit.sum() / seconds),
                    "mean_low_compute_queue_mbit": float(rows.queue_compute_low_mbit.mean()),
                    "mean_transfer_in_queue_mbit": float(rows.queue_transfer_in_mbit.mean()),
                    "low_cpu_utilization_service": float(rows.cpu_low_utilization_sum.sum() / rows.service_steps.sum()),
                }
        for heavy in range(3):
            times = meter[(meter.uav == heavy) & (meter.role == "heavy")].time_s
            phase = meter[meter.time_s.isin(times)]
            record["phase_destination_loads"][str(heavy)] = {
                str(uid): float(rows.assigned_destination_mbit.sum() / (len(rows) * 0.5))
                for uid, rows in phase.groupby("uav")
            }
        per_seed.append(record)
    all_routes, all_roles, all_scores = map(np.concatenate, (routes, role_routes, raw_scores))
    averaged = {}
    for group in ("by_role", "by_fixed_id", "phase_destination_loads"):
        averaged[group] = {
            key: {metric: float(np.mean([row[group][key][metric] for row in per_seed]))
                  for metric in per_seed[0][group][key]}
            for key in per_seed[0][group]
        }
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest for p, digest in hashes.items())
    if label == "uniform_test":
        assert np.allclose(all_routes, 1 / 3, atol=1e-6)
    return {
        "route_fixed_id_mean": all_routes.mean(0).tolist(),
        "route_role_mean": all_roles.mean(0).tolist(),
        "single_destination_row_fraction_high_low": ((all_routes > 1e-8).sum(-1) == 1).mean((0, 2)).tolist(),
        "raw_abs_over_095_fraction_high_low": (np.abs(all_scores) > 0.95).mean((0, 2, 3)).tolist(),
        "fixed_row_mean_standard_deviation_high_low": all_routes.std(0).mean((1, 2)).tolist(),
        "mean_across_seeds": averaged,
        "per_seed": per_seed,
        "source_hashes": hashes,
    }


def main():
    current_report = json.loads((CURRENT / "report.json").read_text())
    pure_report = json.loads((PURE / "report.json").read_text())
    seeds = [row["seed"] for row in current_report["final"]]
    assert seeds == [row["seed"] for row in pure_report["final"]]
    arms = {
        "uniform": inspect(CURRENT, "uniform_test", seeds),
        "pure_sac": inspect(PURE, pure_report["final_label"], seeds),
        "joint_sac_centroid": inspect(CURRENT, "centroid_same_sac", seeds),
        "joint_sac_ppo": inspect(CURRENT, "final", seeds),
    }
    report = {
        "scope": "Saved actions only; one training seed, three development evaluation scenes. No new rollouts or training.",
        "matrix_semantics": "High then low; rows source, columns destination. Decision weighted, not traffic weighted.",
        "role_order": list(ROLES),
        "rate_semantics": "Actual assigned/computed bits per wall second, includes deployment; arithmetic mean of scene-wise means.",
        "arms": arms,
    }
    output = CURRENT / "uniform_gap_saved_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({name: {key: value for key, value in arm.items() if key not in ("source_hashes", "per_seed")}
                      for name, arm in arms.items()}, indent=2))


if __name__ == "__main__":
    main()
