"""Frozen-policy resource interventions; no optimizer or physics changes."""

import copy
import csv
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
from scipy import stats
import torch

from .experiment import append_csv, evaluate, isolated_rng, snapshot, write_json
from .mdp import QueueMDP
from .agents.sac_agent import device


def bandwidth_config(config, mhz):
    result = copy.deepcopy(config)
    result["environment"]["u2u_bandwidth_mhz"] = float(mhz)
    return result


def cpu_config(config, ghz):
    result = copy.deepcopy(config)
    result["cpu_ghz"] = float(ghz)
    return result


class MeteredMDP(QueueMDP):
    """Read existing integration-step service counters and post-frame queues."""

    def __init__(self, config, record_links=False):
        super().__init__(config)
        self.readings = []
        self.uav_readings = []
        self.record_links = record_links
        self.link_readings = []
        self.compute_admitted = np.zeros(self.N)
        enqueue = self.env._enqueue_compute_chunk

        def metered_enqueue(chunk):
            self.compute_admitted[chunk.target_uav] += chunk.remaining_mbits
            return enqueue(chunk)

        self.env._enqueue_compute_chunk = metered_enqueue

    def step_frame(self, action, gamma_frame=0.99):
        self.compute_admitted.fill(0)
        before_compute = self.env.Q_loc_high.copy()+self.env.Q_loc_low.copy()
        before_transfer = self.env.Q_tx_high.copy()+self.env.Q_tx_low.copy()
        before_flow = self.env.coop_flow_mbits.copy()
        frame = super().step_frame(action, gamma_frame)
        service = [info for info in frame["infos"] if not info["service_paused"]]
        compute_steps = [info for info in frame["infos"]
                         if not info.get("compute_service_paused", info["service_paused"])]
        dt = self.env.micro_dt
        active_count = bandwidth_sum = rate_sum = tx = capacity = compute = cpu = 0.0
        for info in service:
            bandwidth = info["u2u_link_bandwidth_hz"]
            active = bandwidth > 0
            active_count += float(active.sum())
            bandwidth_sum += float(bandwidth.sum()) / 1e6
            rate_sum += float(info["u2u_rate_bps"].sum()) / 1e6
            tx += float(info["tx_service_matrix_mbits"].sum())
            capacity += float(info["u2u_rate_bps"].sum()) * dt / 1e6
        for info in compute_steps:
            compute += float(info["compute_service_per_uav_mbits"].sum())
            cpu += float(info["cpu_utilization_mean"])
        _, _, upload, _ = self.task_aggregates()
        self.readings.append({
            "time_s": self.env._current_time(),
            "queue_compute_mbit": float(self.env.Q_loc_high.sum()+self.env.Q_loc_low.sum()),
            "queue_transfer_mbit": float(self.env.Q_tx_high.sum()+self.env.Q_tx_low.sum()),
            "queue_upload_mbit": float(upload.sum()),
            "queue_compute_low_mbit": float(self.env.Q_loc_low.sum()),
            "queue_transfer_low_mbit": float(self.env.Q_tx_low.sum()),
            "service_steps": len(service), "compute_steps": len(compute_steps),
            "active_link_steps": active_count,
            "active_link_bandwidth_mhz_sum": bandwidth_sum,
            "active_link_rate_mbps_sum": rate_sum,
            "tx_service_mbit": tx, "tx_capacity_mbit": capacity,
            "compute_service_mbit": compute, "cpu_utilization_sum": cpu,
        })
        assigned = self.env.coop_flow_mbits-before_flow
        computed = sum((info["compute_service_per_uav_mbits"] for info in compute_steps), np.zeros(self.N))
        transmitted = sum((info["tx_service_matrix_mbits"] for info in service), np.zeros((self.N,self.N)))
        after_compute = self.env.Q_loc_high+self.env.Q_loc_low
        after_transfer = self.env.Q_tx_high+self.env.Q_tx_low
        # A transferred chunk joins compute only after its entire transmission.
        # Count enqueue events, not partially received bytes.
        admitted_compute = self.compute_admitted.copy()
        admitted_transfer = assigned.copy()
        np.fill_diagonal(admitted_transfer, 0)
        compute_error = after_compute-before_compute-admitted_compute+computed
        transfer_error = after_transfer-before_transfer-admitted_transfer+transmitted
        if max(np.abs(compute_error).max(), np.abs(transfer_error).max()) > 1e-5:
            raise AssertionError("Physical queue flow balance failed")
        if self.record_links:
            for source in range(self.N):
                for destination in range(self.N):
                    if source == destination:
                        continue
                    rates = [float(info["u2u_rate_bps"][source, destination])/1e6 for info in service]
                    bandwidths = [float(info["u2u_link_bandwidth_hz"][source, destination])/1e6 for info in service]
                    self.link_readings.append({
                        "time_s": self.env._current_time(), "source": source, "destination": destination,
                        "service_steps": len(service), "active_steps": sum(b > 0 for b in bandwidths),
                        "bandwidth_mhz_sum": sum(bandwidths), "rate_mbps_sum": sum(rates),
                        "allocated_capacity_mbit": sum(rates)*dt,
                        "admitted_mbit": float(admitted_transfer[source, destination]),
                        "served_mbit": float(transmitted[source, destination]),
                        "queue_mbit": float(after_transfer[source, destination]),
                        "balance_error": float(transfer_error[source, destination]),
                    })
        cpu_sum = sum((info["cpu_utilization_per_uav"] for info in compute_steps), np.zeros(self.N))
        low_cpu_sum = sum((info["cpu_low_utilization_per_uav"] for info in compute_steps), np.zeros(self.N))
        order = np.argsort(-self.env.zone_expected_offered_load_mbps)
        roles = {int(uav): role for uav, role in zip(order, ("heavy", "medium", "light"))}
        for uav in range(self.N):
            self.uav_readings.append({
                "time_s": self.env._current_time(), "uav": uav, "role": roles[uav],
                "service_steps": len(service),
                "compute_steps": len(compute_steps),
                "expected_source_load_mbps": float(self.env.zone_expected_offered_load_mbps[uav]),
                "assigned_source_mbit": float(assigned[uav].sum()),
                "assigned_destination_mbit": float(assigned[:,uav].sum()),
                "assigned_local_mbit": float(assigned[uav,uav]),
                "actual_tx_in_mbit": float(transmitted[:,uav].sum()),
                "actual_tx_out_mbit": float(transmitted[uav].sum()),
                "actual_compute_input_mbit": float(admitted_compute[uav]),
                "completed_transfer_compute_input_mbit": float(admitted_compute[uav]-assigned[uav,uav]),
                "computed_mbit": float(computed[uav]),
                "queue_compute_mbit": float(after_compute[uav]),
                "queue_compute_low_mbit": float(self.env.Q_loc_low[uav]),
                "queue_transfer_in_mbit": float(after_transfer[:,uav].sum()),
                "cpu_utilization_sum": float(cpu_sum[uav]),
                "cpu_low_utilization_sum": float(low_cpu_sum[uav]),
                "compute_balance_error": float(compute_error[uav]),
                "transfer_balance_error_max": float(np.abs(transfer_error[:,uav]).max()),
            })
        return frame


def summarize_meter(rows, duration):
    sums = {key: sum(row[key] for row in rows) for key in rows[0]}
    active = max(sums["active_link_steps"], 1)
    service = max(sums["service_steps"], 1)
    compute_steps = max(sums.get("compute_steps", sums["service_steps"]), 1)
    result = {f"mean_{key}": sums[key]/len(rows) for key in rows[0] if key.startswith("queue_")}
    result.update(
        mean_active_links_service=sums["active_link_steps"]/service,
        mean_bandwidth_per_active_link_mhz=sums["active_link_bandwidth_mhz_sum"]/active,
        mean_rate_per_active_link_mbps=sums["active_link_rate_mbps_sum"]/active,
        tx_served_mbit=sums["tx_service_mbit"],
        tx_throughput_mbps_wall=sums["tx_service_mbit"]/duration,
        compute_throughput_mbps_wall=sums["compute_service_mbit"]/duration,
        cpu_utilization_service=sums["cpu_utilization_sum"]/compute_steps,
        allocated_tx_capacity_utilization=sums["tx_service_mbit"]/max(sums["tx_capacity_mbit"], 1e-12),
    )
    return result


class FrozenMicro:
    def __init__(self, model):
        from .reference import ActorView
        self.sac_actor = ActorView(model.actor)

    def macro_action(self, state, enabled, deterministic=False):
        if enabled:
            raise ValueError("Resource audit keeps centroid deployment fixed")
        return np.zeros(6, dtype=np.float32), 0.0


class UniformRouting:
    """Zero raw scores reproduce the fresh deterministic actor's uniform route."""

    def __init__(self):
        self.sac_actor = self

    def choose_action(self, state, deterministic=False):
        return np.zeros(18, dtype=np.float32)

    def macro_action(self, state, enabled, deterministic=False):
        if enabled:
            raise ValueError("Uniform control uses centroid deployment")
        return np.zeros(6, dtype=np.float32), 0.0


def summarize_links(rows, duration, seed, mhz):
    result = []
    for source, destination in sorted({(r["source"], r["destination"]) for r in rows}):
        selected = [r for r in rows if (r["source"], r["destination"]) == (source, destination)]
        active = sum(r["active_steps"] for r in selected)
        capacity = sum(r["allocated_capacity_mbit"] for r in selected)
        result.append({
            "seed": seed, "bandwidth_mhz": mhz, "source": source, "destination": destination,
            "admitted_mbps_wall": sum(r["admitted_mbit"] for r in selected)/duration,
            "served_mbps_wall": sum(r["served_mbit"] for r in selected)/duration,
            "allocated_capacity_mbps_wall": capacity/duration,
            "bandwidth_mhz_when_active": sum(r["bandwidth_mhz_sum"] for r in selected)/max(active, 1),
            "rate_mbps_when_active": sum(r["rate_mbps_sum"] for r in selected)/max(active, 1),
            "mean_queue_mbit": float(np.mean([r["queue_mbit"] for r in selected])),
            "terminal_queue_mbit": selected[-1]["queue_mbit"],
            "allocated_capacity_utilization": sum(r["served_mbit"] for r in selected)/max(capacity, 1e-12),
        })
    return result


def run_scene_resource_check(config, output):
    """Fixed-rule 3/6 MHz check on the restored scene; never trains a policy."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    files = [p for folder in ("hrl_uav", "scripts", "configs") for p in (root/folder).rglob("*")
             if p.is_file() and p.suffix in (".py", ".json")]
    sources = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    seeds = list(config["qualification"]["test_seeds"])
    macros = int(config["test_macros"])
    with isolated_rng():
        env = QueueMDP(config).env
        env.reset(seed=seeds[0])
        env.set_zone_target_loads(config["phase_loads"][0])
        duty = (1-env.deployment_dt/env.macro_dt
                if env.pause_service_during_flight and not env.compute_during_deployment else 1.0)
        capacity = env.F_max_array/(env.C*1e6)*duty
        loads = env.zone_expected_offered_load_mbps.copy()
    budget = {
        "phase_zero_offered_load_mbps": loads.tolist(),
        "ideal_cpu_capacity_mbps_wall_per_uav": capacity.tolist(),
        "ideal_cpu_utilization_total": float(loads.sum()/capacity.sum()),
        "source_cpu_excess_mbps": np.maximum(loads-capacity, 0).tolist(),
        "uniform_expected_transfer_mbps_wall": float(loads.sum()*(1-1/env.N)),
        "scope": "CPU-only nominal accounting; excess must leave the source under sustained load. Not a wireless feasibility or queue-stability certificate.",
    }
    arms = {f"uniform_b{mhz}": bandwidth_config(config, mhz) for mhz in (3, 6)}
    write_json(output/"manifest.json", {
        "status": "FROZEN_BEFORE_EVALUATION", "arm_configs": arms, "source_hashes": sources,
        "seeds": seeds, "macros": macros, "budget": budget,
        "training": False, "training_seed_count": 0, "checkpoint": None, "teacher": False,
        "policy": "Centroid deployment, equal one-third routing to all destinations for both priorities.",
        "only_difference_between_arms": "environment.u2u_bandwidth_mhz",
        "decision_rule": "Keep mainline at 6 MHz; report both arms without selecting policy, changing rewards, or starting training.",
        "scope": "Exploratory 900 s resource diagnostic with reused evaluation seeds; not SAC performance or infinite-horizon stability proof.",
    })
    started = time.perf_counter()
    all_rows, all_links = [], []
    max_balance_error = 0.0
    for label, cfg in arms.items():
        mhz = cfg["environment"]["u2u_bandwidth_mhz"]
        for seed in seeds:
            instances = []

            def factory(settings):
                instance = MeteredMDP(settings, record_links=True)
                instances.append(instance)
                return instance

            with isolated_rng(), patch("hrl_uav.experiment.QueueMDP", factory):
                row = evaluate(cfg, UniformRouting(), [seed], macros, output, label, macro_enabled=False)[0]
            meter = instances[0]
            append_csv(output/f"{label}_{seed}_service.csv", meter.readings)
            append_csv(output/f"{label}_{seed}_links.csv", meter.link_readings)
            append_csv(output/f"{label}_{seed}_uavs.csv", meter.uav_readings)
            row.update(bandwidth_mhz=mhz, **summarize_meter(meter.readings, row["time_s"]))
            all_rows.append(row)
            all_links.extend(summarize_links(meter.link_readings, row["time_s"], seed, mhz))
            max_balance_error = max(max_balance_error, max(abs(r["balance_error"]) for r in meter.link_readings),
                                    max(abs(r["compute_balance_error"]) for r in meter.uav_readings))
            append_csv(output/"per_seed.csv", [row])
            print(f"[scene {mhz:g} MHz seed {seed}] delay={row['task_delay_weighted_ms']:.1f}ms "
                  f"Qtx={row['mean_queue_transfer_mbit']:.2f} Qcpu={row['mean_queue_compute_mbit']:.2f} Mbit", flush=True)
    for seed in seeds:
        with np.load(output/f"uniform_b3_{seed}_trajectory.npz") as base, np.load(output/f"uniform_b6_{seed}_trajectory.npz") as other:
            for key in ("uav", "users", "centroid", "executed_route"):
                if not np.array_equal(base[key], other[key]):
                    raise AssertionError(f"Geometry/routing pairing failed: {seed} {key}")
        with (output/f"uniform_b3_{seed}_queues.csv").open(encoding="utf-8-sig") as f0, (output/f"uniform_b6_{seed}_queues.csv").open(encoding="utf-8-sig") as f1:
            rows0, rows1 = list(csv.DictReader(f0)), list(csv.DictReader(f1))
            if len(rows0) != len(rows1) or any(a["task_arrived_count"] != b["task_arrived_count"] for a, b in zip(rows0, rows1)):
                raise AssertionError("Arrival pairing failed")
    unchanged = all(hashlib.sha256((root/p).read_bytes()).hexdigest() == digest for p, digest in sources.items())
    if not unchanged:
        raise AssertionError("Source changed during scene resource check")
    keys = [k for k, v in all_rows[0].items() if isinstance(v, (float, int)) and k not in ("seed", "bandwidth_mhz")]
    report = {
        "status": "RESOURCE_DIAGNOSTIC_COMPLETE_NOT_TRAINING", "budget": budget,
        "means": {str(mhz): {k: float(np.mean([r[k] for r in all_rows if r["bandwidth_mhz"] == mhz])) for k in keys} for mhz in (3, 6)},
        "rows": all_rows, "link_summaries": all_links, "queue_balance_max_error": max_balance_error,
        "crn_geometry_routes_arrivals_equal": True, "source_unchanged": unchanged,
        "minutes": (time.perf_counter()-started)/60,
        "limitations": "Only a fixed diagnostic policy and three finite scenes. Failure does not prove resource infeasibility; success does not prove learned-policy convergence or general stability. Queue volumes are not time-delay components.",
    }
    append_csv(output/"link_summary.csv", all_links)
    write_json(output/"report.json", report)
    return report


def run_bandwidth(source, output, cpu_check=False):
    from stable_baselines3 import SAC

    source, output = Path(source).resolve(), Path(output).resolve()
    manifest = json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    checkpoint = source/"sb3/final.zip"
    if config["environment"]["u2u_bandwidth_mhz"] != 3:
        raise ValueError("This paired diagnostic was specified for a 3 MHz source")
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    for relative, digest in manifest["source_hashes"].items():
        if relative.replace("\\", "/") in ("hrl_uav/env.py", "hrl_uav/mdp.py", "hrl_uav/experiment.py"):
            if hashlib.sha256((root/relative).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Evaluation source changed: {relative}")
    treatments = [(6, 10), (6, 12)] if cpu_check else [(3, config["cpu_ghz"]), (6, config["cpu_ghz"])]
    axis = "cpu_ghz" if cpu_check else "bandwidth_mhz"
    levels = [pair[1 if cpu_check else 0] for pair in treatments]
    write_json(output/"manifest.json", {
        "config": config, "source": str(source), "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "treatments_bandwidth_mhz_cpu_ghz": treatments, "comparison_axis": axis,
        "seeds": config["test_seeds"],
        "macro_count": config["test_macros"], "training": False,
        "only_config_change_between_arms": "cpu_ghz" if cpu_check else "environment.u2u_bandwidth_mhz",
        "scope": "Frozen SB3 final reacts to changed queues; not a fixed-action direct effect or retraining.",
        "queue_sampling": "Post-frame 0.5 s equal-time samples; backlog volumes are not delay components.",
        "decision_rule": "Report all paired readings; no resource or policy selection based on results.",
    })
    torch.set_num_threads(1)
    model = SAC.load(checkpoint, device=device)
    model.policy.set_training_mode(False)
    before = snapshot(model.policy)
    agent = FrozenMicro(model)
    started = time.perf_counter()
    all_rows = []
    baseline_max_error = 0.0
    baseline_path = source.parent/"bandwidth_check" if cpu_check else source/"sb3"
    if cpu_check:
        previous = json.loads((baseline_path/"manifest.json").read_text(encoding="utf-8"))
        if previous["config"] != config or previous["checkpoint_sha256"] != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
            raise ValueError("CPU audit reference does not match the frozen bandwidth audit")
    with (baseline_path/("per_seed.csv" if cpu_check else "final_evaluation.csv")).open(encoding="utf-8-sig", newline="") as handle:
        old = {int(row["seed"]): row for row in csv.DictReader(handle) if not cpu_check or float(row["bandwidth_mhz"]) == 6}
    all_uav = []
    baseline_label = None
    for arm_index, (mhz, ghz) in enumerate(treatments):
        cfg = cpu_config(bandwidth_config(config, mhz), ghz)
        label = f"b{mhz}c{ghz:g}" if cpu_check else f"b{mhz}"
        if baseline_label is None:
            baseline_label = label
        for seed in config["test_seeds"]:
            instances = []

            def factory(settings):
                instance = MeteredMDP(settings)
                instances.append(instance)
                return instance

            with isolated_rng(), patch("hrl_uav.experiment.QueueMDP", factory):
                summary = evaluate(cfg, agent, [seed], config["test_macros"], output,
                                   label, macro_enabled=False)[0]
            meter = instances[0].readings
            append_csv(output/f"{label}_{seed}_service.csv", meter)
            uav_meter = instances[0].uav_readings
            append_csv(output/f"{label}_{seed}_uavs.csv", uav_meter)
            all_uav.extend({"seed":seed, "level":levels[arm_index], **row} for row in uav_meter)
            summary.update(bandwidth_mhz=mhz, cpu_ghz=ghz, **summarize_meter(meter, summary["time_s"]))
            if arm_index == 0:
                for key in ("task_delay_weighted_ms", "queue_total_mbit", "task_completion_rate"):
                    baseline_max_error = max(baseline_max_error, abs(summary[key]-float(old[seed][key])))
                if baseline_max_error > 0.01:
                    raise ValueError(f"Baseline not reproduced: max error {baseline_max_error}")
            else:
                with np.load(output/f"{baseline_label}_{seed}_trajectory.npz") as base, np.load(output/f"{label}_{seed}_trajectory.npz") as treated:
                    for key in ("uav", "users", "centroid"):
                        if not np.array_equal(base[key], treated[key]):
                            raise ValueError(f"CRN geometry mismatch: {seed} {key}")
                with (output/f"{baseline_label}_{seed}_queues.csv").open(encoding="utf-8-sig") as f0, (output/f"{label}_{seed}_queues.csv").open(encoding="utf-8-sig") as f1:
                    rows0, rows1 = list(csv.DictReader(f0)), list(csv.DictReader(f1))
                    if len(rows0) != len(rows1):
                        raise ValueError("CRN rollout length mismatch")
                    for r0, r1 in zip(rows0, rows1):
                        if r0["task_arrived_count"] != r1["task_arrived_count"]:
                            raise ValueError("CRN arrivals mismatch")
            all_rows.append(summary)
            append_csv(output/"per_seed.csv", [summary])
            print(f"[resource {mhz} MHz {ghz} GHz seed {seed}] delay={summary['task_delay_weighted_ms']:.1f}ms "
                  f"Q={summary['queue_total_mbit']:.1f}Mbit elapsed={(time.perf_counter()-started)/60:.1f}min", flush=True)
    delta = max((value-before[key]).abs().max().item() for key, value in snapshot(model.policy).items())
    if delta != 0:
        raise ValueError("Frozen policy parameters changed")
    base = [row for row in all_rows if row[axis] == levels[0]]
    treatment = [row for row in all_rows if row[axis] == levels[1]]
    differences = [b["task_delay_weighted_ms"]-a["task_delay_weighted_ms"] for a, b in zip(base, treatment)]
    mean = float(np.mean(differences))
    half = float(stats.t.ppf(.975, len(differences)-1)*stats.sem(differences))
    numeric = [key for key, value in all_rows[0].items() if isinstance(value, (int, float)) and key not in ("seed", "bandwidth_mhz")]
    uav_summaries = []
    for level in levels:
        for seed in config["test_seeds"]:
            for grouping, groups in (("role", ("heavy", "medium", "light")), ("uav", (0,1,2))):
                for group in groups:
                    selected = [row for row in all_uav if row["level"] == level and row["seed"] == seed and row[grouping] == group]
                    duration = len(selected)*config["environment"]["resource_frame_dt"]
                    count = max(sum(row.get("compute_steps", row["service_steps"]) for row in selected), 1)
                    result = {"level":level, "seed":seed, "grouping":grouping, "group":group}
                    for key in ("assigned_source_mbit", "assigned_destination_mbit", "assigned_local_mbit", "actual_tx_in_mbit", "actual_tx_out_mbit", "actual_compute_input_mbit", "computed_mbit"):
                        result[key.replace("_mbit", "_mbps")] = sum(row[key] for row in selected)/duration
                    for key in ("queue_compute_mbit", "queue_compute_low_mbit", "queue_transfer_in_mbit"):
                        result[f"mean_{key}"] = float(np.mean([row[key] for row in selected]))
                    result["cpu_utilization_service"] = sum(row["cpu_utilization_sum"] for row in selected)/count
                    result["cpu_low_utilization_service"] = sum(row["cpu_low_utilization_sum"] for row in selected)/count
                    uav_summaries.append(result)
    append_csv(output/"uav_summary.csv", uav_summaries)
    report = {
        "status": "FROZEN_CPU_DIAGNOSTIC_NOT_TRAINING" if cpu_check else "FROZEN_BANDWIDTH_DIAGNOSTIC_NOT_TRAINING", "rows": all_rows,
        "comparison_axis": axis, "levels": levels,
        "means": {str(level): {key: float(np.mean([row[key] for row in all_rows if row[axis] == level])) for key in numeric} for level in levels},
        "paired_delay_ms_treatment_minus_baseline": differences, "paired_mean_ms": mean,
        "paired_t95_ci_ms": [mean-half, mean+half],
        "baseline_max_error": baseline_max_error, "parameter_max_delta": delta,
        "crn_geometry_and_arrivals_equal": True, "training_seed_count": 1,
        "evaluation_seed_count": len(differences), "minutes": (time.perf_counter()-started)/60,
        "queue_balance_max_error": max(max(abs(row["compute_balance_error"]), row["transfer_balance_error_max"]) for row in all_uav),
        "uav_summaries": uav_summaries,
        "limitations": "Three repeated scenes, one trained policy; no claim of training convergence, long-run stability, or exclusive root cause.",
    }
    write_json(output/"report.json", report)
    return report


def run_uniform_check(source, reference, output):
    source, reference, output = map(lambda path: Path(path).resolve(), (source, reference, output))
    manifest = json.loads((reference/"manifest.json").read_text(encoding="utf-8"))
    prior = json.loads((reference/"report.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    if manifest["source"] != str(source) or prior["status"] != "FROZEN_CPU_DIAGNOSTIC_NOT_TRAINING":
        raise ValueError("Expected the matched 6 MHz CPU diagnostic")
    root = Path(__file__).resolve().parents[1]
    original = json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    for relative, digest in original["source_hashes"].items():
        if relative.replace("\\", "/") in ("hrl_uav/env.py", "hrl_uav/mdp.py", "hrl_uav/experiment.py"):
            if hashlib.sha256((root/relative).read_bytes()).hexdigest() != digest:
                raise ValueError("Physical evaluation sources changed")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"manifest.json", {
        "config":config, "reference":str(reference),
        "reference_report_sha256":hashlib.sha256((reference/"report.json").read_bytes()).hexdigest(),
        "bandwidth_mhz":6, "cpu_ghz":[10,12], "seeds":config["test_seeds"],
        "macro_count":config["test_macros"], "training":False,
        "policy":"Fixed one-third route to each destination for both priorities, matching zero actor output.",
        "scope":"Diagnostic fixed-rule control only; not a teacher, new learned method, or proof of optimality.",
    })
    torch.set_num_threads(1)
    started = time.perf_counter()
    all_rows = []
    for ghz in (10,12):
        cfg = cpu_config(bandwidth_config(config,6),ghz)
        for seed in config["test_seeds"]:
            instances = []

            def factory(settings):
                instance = MeteredMDP(settings)
                instances.append(instance)
                return instance

            label = f"uniform_c{ghz}"
            with isolated_rng(), patch("hrl_uav.experiment.QueueMDP", factory):
                summary = evaluate(cfg,UniformRouting(),[seed],config["test_macros"],output,label,macro_enabled=False)[0]
            meter = instances[0]
            append_csv(output/f"{label}_{seed}_service.csv",meter.readings)
            append_csv(output/f"{label}_{seed}_uavs.csv",meter.uav_readings)
            summary.update(cpu_ghz=ghz,bandwidth_mhz=6,**summarize_meter(meter.readings,summary["time_s"]))
            with np.load(reference/f"b6c{ghz}_{seed}_trajectory.npz") as base, np.load(output/f"{label}_{seed}_trajectory.npz") as control:
                for key in ("uav","users","centroid"):
                    if not np.array_equal(base[key],control[key]):
                        raise ValueError("CRN geometry mismatch")
            with (reference/f"b6c{ghz}_{seed}_queues.csv").open(encoding="utf-8-sig") as f0, (output/f"{label}_{seed}_queues.csv").open(encoding="utf-8-sig") as f1:
                r0,r1 = list(csv.DictReader(f0)),list(csv.DictReader(f1))
                if len(r0)!=len(r1) or any(a["task_arrived_count"]!=b["task_arrived_count"] for a,b in zip(r0,r1)):
                    raise ValueError("CRN arrival count mismatch")
            append_csv(output/"per_seed.csv",[summary])
            all_rows.append(summary)
            print(f"[uniform 6 MHz {ghz} GHz seed {seed}] delay={summary['task_delay_weighted_ms']:.1f}ms "
                  f"Q={summary['queue_total_mbit']:.1f}Mbit",flush=True)
    comparisons = {}
    for ghz in (10,12):
        learned = {row["seed"]:row for row in prior["rows"] if row["cpu_ghz"]==ghz}
        rows = [row for row in all_rows if row["cpu_ghz"]==ghz]
        differences = [row["task_delay_weighted_ms"]-learned[row["seed"]]["task_delay_weighted_ms"] for row in rows]
        half = float(stats.t.ppf(.975,len(rows)-1)*stats.sem(differences))
        mean = float(np.mean(differences))
        comparisons[str(ghz)] = {"uniform_minus_learned_delay_ms":differences,"mean_ms":mean,"t95_ci_ms":[mean-half,mean+half]}
    keys = [key for key,value in all_rows[0].items() if isinstance(value,(int,float)) and key!="seed"]
    report = {
        "status":"FIXED_UNIFORM_ROUTING_DIAGNOSTIC_NOT_TRAINING","rows":all_rows,
        "means":{str(ghz):{key:float(np.mean([row[key] for row in all_rows if row["cpu_ghz"]==ghz])) for key in keys} for ghz in (10,12)},
        "paired_comparisons":comparisons,"crn_geometry_and_arrivals_equal":True,
        "minutes":(time.perf_counter()-started)/60,"training_seed_count":1,"evaluation_seed_count":3,
        "scope":"Descriptive control against the existing trained checkpoint, not a training-method comparison or stability proof.",
    }
    write_json(output/"report.json",report)
    return report
