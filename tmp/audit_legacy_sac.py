"""Read-only historical Actor replay; never trains or calls a teacher."""

from argparse import Namespace
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tmp/legacy_sac_audit_sources"
sys.path[:0] = [str(SOURCE), str(SOURCE / "scripts")]

import numpy as np
import torch

import compare_common as common
import hrl_uav.agents.sac_agent as sac


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=serial) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(state):
    result = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        result.update(key.encode())
        result.update(tensor.detach().cpu().numpy().tobytes())
    return result.hexdigest()


def evaluate(actor, saved_args, seed, macro_steps, output):
    args = Namespace(**saved_args)
    args.macro_steps = macro_steps
    args.eval_episodes = 1
    args.eval_base_seed = seed
    env = common.make_env_from_args(args, record_trajectory=False)
    # Same four assignments as historical configure_macro_objective().
    env.macro_shaping_lambda = float(getattr(args, "macro_shaping_final", 0.0))
    for key in ("macro_move_penalty_weight", "macro_propulsion_tiebreak_weight",
                "macro_propulsion_energy_norm"):
        setattr(env, key, float(getattr(args, key)))
    assert env.sac_obs_dim == actor.action_net.in_to_y1.in_features
    assert int(env.micro_action_space.shape[0]) == actor.action_net.out.out_features
    assert args.micro_sca_residual_scale == 0
    assert not args.micro_autonomous_residual_gate
    base = env.env
    rows = []
    raw_actions = []
    routes = []
    role_routes = []
    before_weights = digest(actor.action_net.state_dict())
    original = env.step_resource_frame

    def observed_step(action):
        old_flow = base.coop_flow_mbits.copy()
        frame = original(action)
        info = frame["infos"][-1]
        compute = float(np.sum(base.Q_loc_high) + np.sum(base.Q_loc_low))
        transfer = float(np.sum(base.Q_tx_high) + np.sum(base.Q_tx_low))
        upload = float(sum(p.remaining_upload_mbits for p in base.parent_tasks.values()))
        counters = base.get_task_cumulative_counters()
        loads = np.array([base._get_zone_target_load_mbps(i) for i in range(base.N)])
        order = np.argsort(-loads)
        if frame["phase"] == "service":
            route = np.stack([info["split_ratios_high"], info["split_ratios_low"]])
            raw_actions.append(np.asarray(action).copy())
            routes.append(route.copy())
            role_routes.append(route[:, order][:, :, order])
        delta = base.coop_flow_mbits - old_flow
        row = {
            "time_s": float(counters["time_s"]), "phase": frame["phase"],
            "q_compute_mbits": compute, "q_transfer_mbits": transfer,
            "q_upload_mbits": upload, "q_total_mbits": compute + transfer + upload,
            "arrived_count": counters["arrived_count"],
            "completed_count": counters["completed_count"],
            "completed_high_count": counters["completed_high_count"],
            "completed_low_count": counters["completed_low_count"],
            "completed_high_delay_sum_s": counters["completed_high_delay_sum_s"],
            "completed_low_delay_sum_s": counters["completed_low_delay_sum_s"],
            "cross_routed_mbits": float(delta.sum() - np.trace(delta)),
            "all_routed_mbits": float(delta.sum()),
        }
        for i in range(base.N):
            row[f"load_uav{i}_mbps"] = float(loads[i])
            row[f"destination_uav{i}_mbits"] = float(delta[:, i].sum())
        rows.append(row)
        return frame

    env.step_resource_frame = observed_step
    started = time.perf_counter()
    with torch.inference_mode():
        metrics = common.evaluate_policy(
            env, args,
            lambda current, local, deterministic: np.zeros((current.N, 2), np.float32),
            lambda current, deterministic: actor.choose_action(
                current.get_sac_state(), deterministic=True),
        )
    assert before_weights == digest(actor.action_net.state_dict())
    assert abs(rows[-1]["time_s"] - macro_steps * 10) < 1e-6
    path = output / f"seed{seed}_{macro_steps * 10}s"
    write_csv(path.with_suffix(".csv"), rows)
    np.savez_compressed(path.with_suffix(".npz"), raw_actions=np.asarray(raw_actions),
                        routes=np.asarray(routes), role_routes=np.asarray(role_routes))
    windows = []
    counter_keys = [k for k in rows[0] if k.startswith("completed_") or k == "arrived_count"]
    previous = {key: 0.0 for key in counter_keys}
    for start in range(0, macro_steps * 10, 300):
        subset = [r for r in rows if start < r["time_s"] <= start + 300 + 1e-6]
        last = subset[-1]
        dh = last["completed_high_delay_sum_s"] - previous["completed_high_delay_sum_s"]
        dl = last["completed_low_delay_sum_s"] - previous["completed_low_delay_sum_s"]
        nh = last["completed_high_count"] - previous["completed_high_count"]
        nl = last["completed_low_count"] - previous["completed_low_count"]
        windows.append({
            "start_s": start, "end_s": start + 300,
            "completed_in_window_weighted_delay_ms": 1000 * (10 * dh + dl) / max(10 * nh + nl, 1),
            "mean_compute_queue_mbits": float(np.mean([r["q_compute_mbits"] for r in subset])),
            "mean_transfer_queue_mbits": float(np.mean([r["q_transfer_mbits"] for r in subset])),
            "mean_upload_queue_mbits": float(np.mean([r["q_upload_mbits"] for r in subset])),
            "terminal_total_queue_mbits": last["q_total_mbits"],
            "cross_routed_mbps": sum(r["cross_routed_mbits"] for r in subset) / 300,
        })
        previous = {key: last[key] for key in counter_keys}
    route_mean = np.mean(routes, axis=0)
    result = {
        "seed": seed, "duration_s": macro_steps * 10,
        "elapsed_wall_s": time.perf_counter() - started,
        "task_delay_weighted_ms": metrics["task_delay_weighted_ms"],
        "task_delay_high_ms": metrics["task_delay_high_ms"],
        "task_delay_low_ms": metrics["task_delay_low_ms"],
        "task_completion_rate": metrics["task_completion_rate"],
        "queue_total_mbits": metrics["q_high"] + metrics["q_low"],
        "queue_high_mbits": metrics["q_high"], "queue_low_mbits": metrics["q_low"],
        "coop_ratio": metrics["coop_ratio"],
        "terminal_total_queue_mbits": rows[-1]["q_total_mbits"],
        "service_decision_mean_routes": route_mean,
        "service_decision_mean_routes_by_load_role": np.mean(role_routes, axis=0),
        "role_order": ["heavy", "medium", "light"],
        "priority_order": ["high", "low"],
        "high_local_ratio": float(np.diag(route_mean[0]).mean()),
        "low_local_ratio": float(np.diag(route_mean[1]).mean()),
        "weights_unchanged": True, "windows": windows,
    }
    save_json(path.with_suffix(".json"), result)
    print(json.dumps({k: result[k] for k in ("seed", "duration_s", "task_delay_weighted_ms",
                                           "queue_total_mbits", "terminal_total_queue_mbits",
                                           "coop_ratio")}), flush=True)
    return result


def main():
    torch.set_num_threads(1)
    sac.device = torch.device("cpu")
    checkpoint = SOURCE / "result_archive/v66_same_scene_direct_sac_gate/direct_sac_final.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["sac_actor"]
    actor = sac.Actor(state["in_to_y1.weight"].shape[1], state["out.weight"].shape[0])
    actor.action_net.load_state_dict(state)
    actor.action_net.eval().requires_grad_(False)
    output = ROOT / "results/legacy_sac_readonly_audit"
    if (output / "report.json").exists() or list(output.glob("seed*")):
        raise FileExistsError("Refusing to overwrite completed replay data")
    output.mkdir(exist_ok=True)
    save_json(output / "manifest.json", {
        "source_snapshot": str(SOURCE), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "saved_args": payload["args"], "training_performed": False,
        "teacher_called": False, "historical_initialization_was_teacher_assisted": True,
        "baseline_seed": 8000, "extension_seeds": [8000, 8001, 8002],
        "baseline_duration_s": 300, "extension_duration_s": 900,
        "note": "Exploratory frozen replay, not training robustness or infinite-horizon stability proof.",
    })
    baseline = evaluate(actor, payload["args"], 8000, 30, output)
    archive = SOURCE / "result_archive/v66_same_scene_direct_sac_gate/per_seed_raw.csv"
    with archive.open(encoding="utf-8-sig", newline="") as stream:
        expected = next(r for r in csv.DictReader(stream)
                        if r["method"] == "direct_sac_final" and int(r["seed"]) == 8000)
    keys = ["task_delay_weighted_ms", "task_delay_high_ms", "task_delay_low_ms",
            "queue_total_mbits", "task_completion_rate", "coop_ratio"]
    delta = {key: float(baseline[key]) - float(expected[key]) for key in keys}
    reproduced = all(abs(v) < 0.02 for v in delta.values())
    report = {"baseline_delta": delta, "baseline_reproduced": reproduced, "runs": [baseline]}
    save_json(output / "report.json", report)
    if not reproduced:
        raise RuntimeError(f"Historical baseline mismatch: {delta}")
    for seed in (8000, 8001, 8002):
        report["runs"].append(evaluate(actor, payload["args"], seed, 90, output))
        save_json(output / "report.json", report)


if __name__ == "__main__":
    main()
