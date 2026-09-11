"""Bounded role-offset trajectory diagnosis with deterministic, read-only SAC."""

from contextlib import ExitStack, contextmanager
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import numpy as np
import torch

from .agents.sac_agent import Actor, Critic, Entroy, Memory
from .bandwidth import summarize_links, summarize_meter
from .experiment import append_csv, isolated_rng, rollout, snapshot, write_json
from .geometry import cooperation_aligned_basis
from .mdp import QueueMDP
from .ppo_frozen import DEFAULT_SOURCE, SOURCE_DIGESTS, MacroPairMeter, exogenous_records
from .priority_grid import file_hash
from .resumption import state_equal
from .scene_prescreen import ROOT, assert_crn_equal, digest_update, role_summary, source_hashes
from .wireless_check import ENERGY, PRIORITY_MEASURES, read_csv
from .wireless_pilot import PilotMeter, route_statistics


SAFE_HEAD = "ab5287d60dee7b5cb371e65dc60514d543728669"
REVIEWED_HEAD = "b4908d3e78d4c807013c8ff6991637dbae546129"
EXPLORATION = (33600, 33601)
RECHECK = (33610, 33611, 33612)
ROLES = ("heavy", "medium", "light")
HISTORY = ROOT / "results/ppo_frozen_sac_pilot120_seed101"
J_ATOL, J_RTOL = 1e-10, 1e-12
MEASURES = (
    "J", "J_holding", "J_energy", "task_delay_weighted_ms", "task_delay_high_ms",
    "task_delay_low_ms", "task_completion_rate", *PRIORITY_MEASURES, "unfinished",
    "mean_queue_total_mbit", "mean_queue_upload_mbit", "mean_queue_compute_mbit",
    "mean_queue_transfer_mbit", "queue_total_mbit", "queue_upload_mbit",
    "queue_compute_mbit", "queue_transfer_mbit", "late_queue_slope_mbit_s",
    "power_w", *ENERGY, "objective_energy_j", "propulsion_increment_energy_j",
    "tx_served_mbit", "tx_throughput_mbps_wall", "mean_active_links_service",
    "mean_rate_per_active_link_mbps", "u2u_tx_time_s", "u2g_served_mbit",
    "u2g_tx_time_s", "u2g_rate_time_weighted_mbps", "compute_throughput_mbps_wall",
    "cpu_utilization_service", "mean_uav_centroid_distance_m", "fleet_distance_m",
    "mean_position_delta_vs_zero_m", "max_position_delta_vs_zero_m",
)


def candidates():
    result = [{"name": "zero", "offsets_m": [[0, 0] for _ in ROLES]}]
    directions = (("xp", (20, 0)), ("xm", (-20, 0)), ("yp", (0, 20)), ("ym", (0, -20)))
    for index, role in enumerate(ROLES):
        for direction, value in directions:
            offsets = [[0, 0] for _ in ROLES]
            offsets[index] = list(value)
            result.append({"name": f"{role}_{direction}", "offsets_m": offsets})
    for direction, x in (("xp", 20), ("xm", -20)):
        result.append({"name": f"all_{direction}", "offsets_m": [[x, 0] for _ in ROLES]})
    for item in result:
        item["offset_norm_m"] = float(np.linalg.norm(item["offsets_m"]))
    return result


def role_action(loads, offsets, radius):
    loads, offsets = np.asarray(loads), np.asarray(offsets, dtype=np.float64)
    if loads.shape != (3,) or offsets.shape != (3, 2) or radius != 60:
        raise ValueError("This fixed protocol requires 3 UAVs and the existing 60 m radius")
    if not np.isfinite(loads).all() or not np.isfinite(offsets).all():
        raise ValueError("Nonfinite role/action input")
    order = np.argsort(-loads, kind="stable")
    raw = np.zeros((3, 2), dtype=np.float32)
    raw[order] = offsets / radius
    return raw.ravel()


def choose_candidate(rows):
    by_name = {row["policy"]: row for row in rows}
    if set(by_name) != {c["name"] for c in candidates()} or len(rows) != 15:
        raise ValueError("Selection requires all 15 candidates")
    zero = by_name["zero"]["J"]
    valid = [c for c in candidates()[1:] if by_name[c["name"]]["J"] < zero
             and not np.isclose(by_name[c["name"]]["J"], zero, rtol=J_RTOL, atol=J_ATOL)]
    if not valid:
        return None
    best = min(by_name[c["name"]]["J"] for c in valid)
    ties = [c for c in valid if np.isclose(by_name[c["name"]]["J"], best, rtol=J_RTOL, atol=J_ATOL)]
    return min(ties, key=lambda c: (c["offset_norm_m"], c["name"]))


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def source_bundle():
    paths = {"checkpoint": DEFAULT_SOURCE, "manifest.json": DEFAULT_SOURCE.parent.parent / "manifest.json",
             "report.json": DEFAULT_SOURCE.parent.parent / "report.json"}
    for key, path in paths.items():
        if file_hash(path) != SOURCE_DIGESTS[key]:
            raise ValueError(f"Frozen source digest mismatch: {path}")
    manifest, report = load_json(paths["manifest.json"]), load_json(paths["report.json"])
    saved = torch.load(DEFAULT_SOURCE, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    if (manifest["status"] != "LONG_COMPLETE" or report["status"] != "LONG_COMPLETE"
            or manifest["training_initialization_seed"] != 101 or report["final_window"] != 2000
            or cfg != manifest["config"] or saved["stage"] != "pretrain"
            or saved["updates"]["ppo"] != 0 or saved["updates"]["ppo_gradients"] != 0
            or saved["run_state"]["completed_windows"] != 2000
            or not saved["run_state"]["at_rollout_boundary"]):
        raise ValueError("Source is not the declared completed SAC-only window2000")
    if not any(c["window"] == 2000 and c["sha256"] == SOURCE_DIGESTS["checkpoint"]
               for c in manifest["checkpoints"]):
        raise ValueError("Source checkpoint not registered in manifest")
    for path, digest in manifest["source_hashes"].items():
        if file_hash(ROOT / path) != digest:
            raise ValueError(f"Existing source changed since SAC101: {path}")
    env = cfg["environment"]
    if (env["wireless_model"] != "psd_per_uav_budget" or env["u2u_bandwidth_mhz"] != 6
            or not env["compute_during_deployment"] or not env["pause_service_during_flight"]
            or cfg["training"]["pretrain_windows"] != 2000 or cfg["training"]["joint_windows"] != 0
            or cfg["training"]["seed"] != 101 or cfg["waypoint_radius_m"] != 60
            or cfg["max_displacement_m"] != 20
            or cfg["observation"] != {"layout": "split_v1", "routing_ready": False}):
        raise ValueError("Unexpected effective configuration")
    history = load_json(HISTORY / "manifest.json")
    historical_config = copy.deepcopy(history["config"])
    historical_config["training"]["joint_windows"] = 0
    if (historical_config != cfg or history["source_checkpoint_sha256"] != SOURCE_DIGESTS["checkpoint"]
            or history["actual_head"] != REVIEWED_HEAD):
        raise ValueError("Existing centroid reference has different provenance")
    if git("rev-parse", "codex/u2u-model") != SAFE_HEAD:
        raise ValueError("Safety branch has changed")
    # New diagnosis files need not exist in the source manifest; original files must match.
    for path, digest in history["source_hashes"].items():
        if file_hash(ROOT / path) != digest:
            raise ValueError(f"Reviewed PPO-pilot source changed: {path}")
    return copy.deepcopy(cfg), saved, manifest


class ReadOnlySAC:
    """Load SAC components only. No replay instance and no PPO network exists."""

    def __init__(self, saved):
        cfg = saved["config"]
        self.sac_actor = Actor(saved["micro_state_dim"], 18)
        width = 3 if cfg["training"]["critic_action_representation"] == "executed_route" else 0
        self.critic = Critic(saved["micro_state_dim"], 18, width)
        self.entropy = Entroy(18, cfg["training"]["initial_alpha"])
        nets = (self.sac_actor.action_net, self.critic.critic_v, self.critic.target_critic_v)
        for network, key in zip(nets, ("sac_actor", "sac_critic", "sac_target")):
            network.load_state_dict(saved[key], strict=True)
        with torch.no_grad():
            self.entropy.log_alpha.copy_(saved["log_alpha"].to(self.entropy.log_alpha.device))
        self.entropy.alpha = self.entropy.log_alpha.exp()
        self.optimizers = (self.sac_actor.optimizer, self.critic.optimizer, self.entropy.optimizer)
        for optimizer, state in zip(self.optimizers, saved["optimizers"][:3]):
            optimizer.load_state_dict(state)
        self.updates = copy.deepcopy(saved["updates"])
        self.observations = copy.deepcopy(saved["observations"])
        self.reference = self.state()
        for key in ("sac_actor", "sac_critic", "sac_target", "log_alpha"):
            if not state_equal(self.reference[key], saved[key]):
                raise AssertionError(f"SAC restoration failed: {key}")
        if not state_equal(self.reference["optimizers"], saved["optimizers"][:3]):
            raise AssertionError("SAC optimizer restore failed")

    def state(self):
        nets = (self.sac_actor.action_net, self.critic.critic_v, self.critic.target_critic_v)
        return {
            **{key: snapshot(net) for key, net in zip(("sac_actor", "sac_critic", "sac_target"), nets)},
            "log_alpha": self.entropy.log_alpha.detach().cpu().clone(),
            "alpha": self.entropy.alpha.detach().cpu().clone(),
            "optimizers": copy.deepcopy([o.state_dict() for o in self.optimizers]),
            "updates": copy.deepcopy(self.updates),
            "behavior": (self.sac_actor.min_action, self.sac_actor.max_action, self.entropy.target_entropy,
                         [[m.training for m in net.modules()] for net in nets]),
            "gradients": [[None if p.grad is None else p.grad.detach().cpu().clone()
                           for p in net.parameters()] for net in nets],
            "alpha_grad": None if self.entropy.log_alpha.grad is None else self.entropy.log_alpha.grad.detach().cpu().clone(),
        }

    def assert_unchanged(self):
        now = self.state()
        checks = {key: state_equal(value, now[key]) for key, value in self.reference.items()}
        if not all(checks.values()):
            raise AssertionError(f"Frozen SAC state changed: {checks}")
        return checks


def forbidden(*args, **kwargs):
    raise AssertionError("Learning, backward and SAC replay access are forbidden in this diagnosis")


@contextmanager
def no_learning(sac):
    with ExitStack() as stack:
        for optimizer in sac.optimizers:
            stack.enter_context(patch.object(optimizer, "step", forbidden))
        for method in ("store_transition", "sample", "load_state_dict"):
            stack.enter_context(patch.object(Memory, method, forbidden))
        stack.enter_context(patch("torch.autograd.backward", forbidden))
        stack.enter_context(patch("torch.autograd.grad", forbidden))
        yield
    sac.assert_unchanged()


class RolePolicy:
    def __init__(self, sac, candidate):
        self.sac_actor = sac.sac_actor
        self.candidate = copy.deepcopy(candidate)
        self.meter = None
        self.macro_calls = 0

    def macro_action(self, state, enabled=True, deterministic=False):
        if not deterministic:
            raise AssertionError("Only deterministic evaluation is allowed")
        self.macro_calls += 1
        if not enabled:
            if self.candidate["name"] != "zero":
                raise ValueError("A nonzero candidate must enter the macro action path")
            return np.zeros(6, np.float32), 0.0
        # Only current expected load determines roles. Geometry rotation is done by start_macro.
        return role_action(self.meter.env.zone_expected_offered_load_mbps,
                           self.candidate["offsets_m"], self.meter.config["waypoint_radius_m"]), 0.0

    store = update_sac = update_ppo = forbidden


class OpportunityMeter(MacroPairMeter, PilotMeter):
    """Compose existing CRN, route, parent/energy and link meters without new dynamics."""

    def __init__(self, config):
        super().__init__(config)
        self.centroid_digest = hashlib.sha256()
        self.deployments = []
        self.uplink_readings = []
        self.objective_holding = self.objective_energy = 0.0
        self.reset_calls = 0
        integrate = self.env._integrate_user_positions

        def record_centroids():
            result = integrate()
            digest_update(self.centroid_digest, self.env.global_micro_step, self.env.weighted_centroids)
            return result

        self.env._integrate_user_positions = record_centroids

    def reset(self, seed, macro_count):
        self.reset_calls += 1
        state = super().reset(seed, macro_count)
        self.initial_positions = self.env.uav_coords.copy()
        digest_update(self.centroid_digest, self.env.global_micro_step, self.env.weighted_centroids)
        return state

    def start_macro(self, action):
        before, centroids = self.env.uav_coords.copy(), self.env.weighted_centroids.copy()
        loads = self.env.zone_expected_offered_load_mbps.copy()
        basis, partners = cooperation_aligned_basis(centroids, before, loads)
        raw = np.asarray(action).reshape(3, 2)
        local = self.config["waypoint_radius_m"] * raw
        target = np.clip(centroids + np.einsum("nij,nj->ni", basis, local), 0, 1000)
        info = super().start_macro(action)
        roles = {int(i): role for i, role in zip(np.argsort(-loads, kind="stable"), ROLES)}
        for i in range(3):
            self.deployments.append({
                "macro": self.macro_index, "time_s": self.env._current_time(), "uav": i,
                "role": roles[i], "phase_age": self.phase_age, "partner": int(partners[i]),
                "load_mbps": float(loads[i]), "raw_x": float(raw[i, 0]), "raw_y": float(raw[i, 1]),
                "local_x_m": float(local[i, 0]), "local_y_m": float(local[i, 1]),
                "target_x_m": float(target[i, 0]), "target_y_m": float(target[i, 1]),
                "before_x_m": float(before[i, 0]), "before_y_m": float(before[i, 1]),
                "after_x_m": None, "after_y_m": None,
                "executed_dx_m": None, "executed_dy_m": None, "distance_m": None,
                "execution_complete": False,
            })
        return info

    def step_frame(self, action, gamma_frame=.99):
        frame = super().step_frame(action, gamma_frame)
        # macro_step schedules motion; actual coordinates advance in deployment micro-steps.
        if frame["phase"] == "deployment" and self.env.macro_service_pause_steps_remaining == 0:
            for row in self.deployments[-self.N:]:
                after = self.env.uav_coords[row["uav"]]
                delta = after - np.array([row["before_x_m"], row["before_y_m"]])
                distance = float(np.linalg.norm(delta))
                if distance > self.config["max_displacement_m"] + 1e-5:
                    raise AssertionError("Actual movement limit violated")
                row.update(after_x_m=float(after[0]), after_y_m=float(after[1]),
                           executed_dx_m=float(delta[0]), executed_dy_m=float(delta[1]),
                           distance_m=distance, execution_complete=True)
        record, rcfg = self.readings[-1], self.reward_config
        holding = sum(float(i["task_holding_cost_weighted"]) for i in frame["infos"])
        energy = sum(float(i["total_energy"]) + self.propulsion_delta_share for i in frame["infos"])
        holding_cost = rcfg["delay_weight"] * holding / rcfg["holding_norm_task_s"]
        energy_cost = rcfg["energy_weight"] * energy / rcfg["energy_norm_j"]
        np.testing.assert_allclose(holding_cost+energy_cost, frame["cost"], rtol=1e-12, atol=1e-10)
        self.objective_holding += holding_cost
        self.objective_energy += energy_cost
        record.update(holding_weighted_task_s=holding, holding_cost_sum=holding_cost,
                      energy_cost_sum=energy_cost, objective_energy_j=energy,
                      propulsion_increment_energy_j=self.propulsion_delta_share*len(frame["infos"]))
        self.crn_frames[-1]["centroids_microsteps_sha256"] = self.centroid_digest.hexdigest()
        service = [i for i in frame["infos"] if not i["service_paused"]]
        for source in range(3):
            bits = tx_time = rate_time_bits = 0.0
            for info in service:
                selected = info["user_associations"] == source
                work = np.asarray(info["uplink_service_mbits_per_user"], dtype=np.float64)[selected]
                active = np.asarray(info["uplink_active_time_s_per_user"], dtype=np.float64)[selected]
                rate = np.asarray(info["uplink_rate_bps_per_user"], dtype=np.float64)[selected]
                np.testing.assert_allclose(rate*active/1e6, work, rtol=2e-6, atol=1e-7)
                bits += float(work.sum())
                tx_time += float(active.sum())
                rate_time_bits += float((rate*active/1e6).sum())
            self.uplink_readings.append({"time_s": record["time_s"], "source": source,
                                        "served_mbit": bits, "tx_time_s": tx_time,
                                        "actual_rate_time_product_mbit": rate_time_bits})
        current_roles = {r["uav"]: r["role"] for r in self.uav_readings[-3:]}
        for link in self.link_readings[-6:]:
            link["source_role"] = current_roles[link["source"]]
            link["destination_role"] = current_roles[link["destination"]]
        return frame


def crn_records(meter):
    records = exogenous_records(meter)
    for item, original in zip(records, meter.crn_frames):
        item["centroids_microsteps_sha256"] = original["centroids_microsteps_sha256"]
    return records


def evaluate_rule(cfg, sac, candidate, seed, macros, directory, split, *, native_zero=False):
    if seed not in EXPLORATION + RECHECK or macros not in (2, 90):
        raise ValueError("Scene/duration outside the bounded protocol")
    directory.mkdir(parents=True, exist_ok=False)
    policy = RolePolicy(sac, candidate)
    instances = []

    def factory(config):
        meter = OpportunityMeter(config)
        policy.meter = meter
        instances.append(meter)
        return meter

    started = time.perf_counter()
    with isolated_rng(), no_learning(sac), patch("hrl_uav.experiment.QueueMDP", factory):
        summary, rows, trajectory, routes, meter = rollout(
            cfg, policy, seed, macros, macro_enabled=not native_zero,
            train=False, complete_tail=False, record=True)
    if (len(instances) != 1 or meter.reset_calls != 1 or policy.macro_calls != macros
            or summary["time_s"] != macros*10 or summary["boundary_transitions"] != 0
            or not all(r["execution_complete"] for r in meter.deployments)):
        raise AssertionError("Unexpected reset, tail or transition collection")
    tail = rows[len(rows)*2//3:]
    slope = float(np.polyfit([r["time_s"] for r in tail], [r["queue_total_mbit"] for r in tail], 1)[0]) if len(tail) > 1 else 0.0
    summary.update(summarize_meter(meter.readings, summary["time_s"]))
    last = meter.readings[-1]
    summary.update({k: last[k] for k in PRIORITY_MEASURES})
    summary.update({k: sum(r[k] for r in meter.readings) for k in
                    (*ENERGY, "objective_energy_j", "propulsion_increment_energy_j")})
    summary["J_holding"] = meter.objective_holding / meter.integration_steps
    summary["J_energy"] = meter.objective_energy / meter.integration_steps
    np.testing.assert_allclose(summary["J"], summary["J_holding"]+summary["J_energy"], rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(summary["power_w"]*summary["time_s"], sum(summary[k] for k in ENERGY), rtol=1e-12, atol=1e-6)
    uav, users, centroids, macro = map(np.asarray, zip(*trajectory))
    raw, executed = map(np.asarray, zip(*routes))
    if len(routes) != macros*meter.env.service_resource_frames:
        raise AssertionError("Unexpected SAC decision count")
    summary.update(
        mean_queue_total_mbit=sum(summary[f"mean_queue_{q}_mbit"] for q in ("upload", "compute", "transfer")),
        late_queue_slope_mbit_s=slope, fleet_distance_m=sum(r["distance_m"] for r in meter.deployments),
        mean_uav_centroid_distance_m=float(np.linalg.norm(uav-centroids, axis=-1).mean()),
        mean_position_delta_vs_zero_m=0.0, max_position_delta_vs_zero_m=0.0,
        u2u_tx_time_s=sum(r["tx_time_s"] for r in meter.link_readings),
        u2g_served_mbit=sum(r["served_mbit"] for r in meter.uplink_readings),
        u2g_tx_time_s=sum(r["tx_time_s"] for r in meter.uplink_readings),
    )
    summary["u2g_rate_time_weighted_mbps"] = summary["u2g_served_mbit"]/max(summary["u2g_tx_time_s"], 1e-12)
    if not np.isfinite([summary[k] for k in MEASURES]).all():
        raise AssertionError("Nonfinite diagnosis output")
    row = {"split": split, "policy": candidate["name"], "seed": seed, **summary}
    append_csv(directory/"queues.csv", rows)
    append_csv(directory/"frames.csv", meter.readings)
    append_csv(directory/"uavs.csv", meter.uav_readings)
    append_csv(directory/"links.csv", meter.link_readings)
    append_csv(directory/"uplink.csv", meter.uplink_readings)
    append_csv(directory/"routes.csv", meter.route_readings)
    append_csv(directory/"deployments.csv", meter.deployments)
    append_csv(directory/"crn.csv", crn_records(meter))
    np.savez_compressed(directory/"trajectory.npz", uav=uav, users=users, centroid=centroids,
                        macro_action=macro, raw_action=raw, executed_route=executed,
                        initial_uav=meter.initial_positions)
    write_json(directory/"metrics.json", row)
    audit = {"seed": seed, "policy": candidate["name"], "duration_s": summary["time_s"],
             "reset_count": meter.reset_calls, "macro_calls": policy.macro_calls,
             "sac_decisions": len(routes), "learning_unchanged": sac.assert_unchanged(),
             "replay_instantiated": False, "replay_reads": 0, "replay_writes": 0,
             "training": False, "complete_tail": False, "elapsed_seconds": time.perf_counter()-started,
             "cost_decomposition_error": abs(summary["J"]-summary["J_holding"]-summary["J_energy"]),
             "queue_balance_error_max_mbit": max(max(abs(r["compute_balance_error"]), r["transfer_balance_error_max"])
                                                  for r in meter.uav_readings)}
    write_json(directory/"audit.json", audit)
    return row, meter


def pair(reference, other, reference_dir, other_dir, *, same_uav=False):
    left, right = read_csv(reference_dir/"crn.csv"), read_csv(other_dir/"crn.csv")
    assert_crn_equal(left, right)
    with np.load(reference_dir/"trajectory.npz") as a, np.load(other_dir/"trajectory.npz") as b:
        for key in ("users", "centroid") + (("uav", "raw_action", "executed_route") if same_uav else ()):
            np.testing.assert_array_equal(a[key], b[key])
        position = np.linalg.norm(b["uav"]-a["uav"], axis=-1)
    other["mean_position_delta_vs_zero_m"] = float(position.mean())
    other["max_position_delta_vs_zero_m"] = float(position.max())
    write_json(other_dir/"metrics.json", other)
    if same_uav:
        for key in MEASURES:
            np.testing.assert_allclose(reference[key], other[key], rtol=1e-12, atol=1e-9, err_msg=key)
    result = {"split": other["split"], "policy": other["policy"], "seed": other["seed"],
              "crn_equal": True, **{f"delta_{k}": other[k]-reference[k] for k in MEASURES}}
    return result


def historical_zero(row, directory):
    history = load_json(HISTORY/"report.json")
    records = history["validation_history"][0]["centroid"]
    old = next(r for r in records if r["seed"] == row["seed"])
    keys = [k for k in MEASURES if k in old and k not in ("mean_position_delta_vs_zero_m", "max_position_delta_vs_zero_m")]
    errors = {k: abs(row[k]-old[k]) for k in keys}
    if any(value > 1e-9 for value in errors.values()):
        raise AssertionError(f"Zero baseline differs from pilot reference: {errors}")
    prefix = HISTORY/f"validation_centroid_w0_{row['seed']}"
    old_crn, new_crn = read_csv(f"{prefix}_crn.csv"), read_csv(directory/"crn.csv")
    keys_crn = ("time_s", "task_arrived_count", "arrivals_sha256", "load_sha256",
                "schedule_sha256", "users_microsteps_sha256")
    assert_crn_equal([{k: r[k] for k in keys_crn} for r in old_crn],
                     [{k: r[k] for k in keys_crn} for r in new_crn])
    with np.load(f"{prefix}_trajectory.npz") as a, np.load(directory/"trajectory.npz") as b:
        for k in ("uav", "users", "centroid", "raw_action", "executed_route"):
            np.testing.assert_array_equal(a[k], b[k])
    return {"seed": row["seed"], "metric_errors": errors, "crn_and_trajectories_equal": True,
            "source_files": {str(p): file_hash(p) for p in (HISTORY/"report.json",
                Path(f"{prefix}_crn.csv"), Path(f"{prefix}_trajectory.npz"))}}


def prepare(output):
    cfg, saved, source = source_bundle()
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"purpose": "bounded feasible trajectory opportunity check, not optimization or training",
                "candidates": candidates(), "roles": ROLES, "exploration": EXPLORATION,
                "recheck": RECHECK, "duration_s": 900, "smoke_seconds": 20,
                "training": False, "replay_instantiated": False, "deterministic_sac": True,
                "complete_tail": False, "max_exploration_runs": 30, "max_recheck_runs": 6,
                "selection": "lowest mean J on exploration, only if below zero beyond numerical tolerance",
                "tie_break": "smaller combined role offset L2 norm, then lexical name",
                "J_atol": J_ATOL, "J_rtol": J_RTOL,
                "macro_actions": "role local offset / 60 m, then unmodified QueueMDP.start_macro",
                "controller_inputs": "current expected loads; geometry uses current UAVs and current centroids",
                "exogenous_checks": "parent births, every physical-step user and centroid positions, load schedule",
                "forbidden_scenes": list(range(33700, 33710)), "scope": "reused development scenes only"}
    write_json(output/"protocol.json", protocol)
    write_json(output/"candidates.json", candidates())
    manifest = {"status": "PREPARED", "actual_head": git("rev-parse", "HEAD"),
                "reviewed_head": REVIEWED_HEAD, "safe_head": SAFE_HEAD, "branch": git("branch", "--show-current"),
                "worktree_status": git("status", "--short"),
                "diff_from_reviewed": git("diff", REVIEWED_HEAD, "--", "hrl_uav", "scripts", "tests", "configs"),
                "config": cfg, "source_checkpoint": str(DEFAULT_SOURCE), "source_digests": SOURCE_DIGESTS,
                "source_run_hashes": source["source_hashes"], "source_hashes_before": source_hashes(),
                "protocol_sha256": file_hash(output/"protocol.json"),
                "candidates_sha256": file_hash(output/"candidates.json"), "training": False,
                "training_seed_count": 0, "sac_source_training_seed": 101, "sac_source_window": 2000,
                "counts": {"exploration": 0, "recheck": 0}, "reserved_scenes_executed": [],
                "checkpoint_update_counts": saved["updates"], "prepared_unix": time.time()}
    write_json(output/"manifest.json", manifest)
    print(json.dumps({"status": "PREPARED", "output": str(output), "candidates": 15}), flush=True)


def assert_manifest(output, status):
    manifest = load_json(output/"manifest.json")
    if manifest["status"] != status or manifest["source_hashes_before"] != source_hashes():
        raise ValueError("Status/source fingerprint changed; do not mix outputs")
    if file_hash(output/"protocol.json") != manifest["protocol_sha256"] or candidates() != load_json(output/"candidates.json"):
        raise ValueError("Frozen candidates/protocol changed")
    return manifest


def smoke(output):
    manifest = assert_manifest(output, "PREPARED")
    cfg, saved, _ = source_bundle()
    sac = ReadOnlySAC(saved)
    del saved
    root = output/"smoke"
    native_dir = root/"native_zero"
    zero, _ = evaluate_rule(cfg, sac, candidates()[0], EXPLORATION[0], 2, native_dir, "smoke", native_zero=True)
    checks = []
    for candidate in candidates():
        directory = root/candidate["name"]
        row, meter = evaluate_rule(cfg, sac, candidate, EXPLORATION[0], 2, directory, "smoke")
        checks.append(pair(zero, row, native_dir, directory, same_uav=candidate["name"] == "zero"))
        if candidate["name"] != "zero" and row["max_position_delta_vs_zero_m"] <= 1e-3:
            raise AssertionError("Nonzero offset did not cause actual motion")
        if not np.isclose(sum(meter.readings[-1][f"task_arrived_{p}_count"] for p in ("high", "low")), row["task_arrived_count"]):
            raise AssertionError("Priority counts disagree")
    write_json(root/"report.json", {"status": "SMOKE_COMPLETE", "checks": checks,
                                    "frozen": sac.assert_unchanged(), "learning_calls": 0, "replay_accesses": 0})
    manifest.update(status="SMOKE_COMPLETE", smoke_runs=16)
    write_json(output/"manifest.json", manifest)
    print(json.dumps({"status": "SMOKE_COMPLETE", "runs": 16, "source_unchanged": True}), flush=True)


def aggregate(rows, split):
    result = []
    for candidate in candidates():
        selected = [r for r in rows if r["split"] == split and r["policy"] == candidate["name"]]
        if selected:
            result.append({"split": split, "policy": candidate["name"], "scene_count": len(selected),
                           **{key: float(np.mean([r[key] for r in selected])) for key in MEASURES}})
    zero = next(r for r in result if r["policy"] == "zero")
    for row in result:
        row["delta_J"] = row["J"]-zero["J"]
    return sorted(result, key=lambda r: r["J"])


def save_meter_summaries(output, row, meter):
    identity = {k: row[k] for k in ("split", "policy", "seed")}
    append_csv(output/"route_summary.csv", route_statistics(meter.route_readings, row["policy"], row["seed"], 2000, row["split"]))
    append_csv(output/"role_summary.csv", [{**r, "split": row["split"]} for r in role_summary(meter.uav_readings, .5, row["policy"], row["seed"])])
    append_csv(output/"link_summary.csv", [{**r, **identity} for r in summarize_links(meter.link_readings, 900, row["seed"], 6)])
    role_links = []
    for source in ROLES:
        for dest in ROLES:
            if source == dest:
                continue
            rows = [r for r in meter.link_readings if r["source_role"] == source and r["destination_role"] == dest]
            active = sum(r["active_steps"] for r in rows)
            role_links.append({**identity, "source_role": source, "destination_role": dest,
                               "served_mbit": sum(r["served_mbit"] for r in rows),
                               "admitted_mbit": sum(r["admitted_mbit"] for r in rows),
                               "tx_time_s": sum(r["tx_time_s"] for r in rows),
                               "tx_energy_j": sum(r["tx_energy_j"] for r in rows),
                               "rate_mbps_when_active": sum(r["rate_mbps_sum"] for r in rows)/max(active, 1)})
    append_csv(output/"role_link_summary.csv", role_links)


def run(output):
    manifest = assert_manifest(output, "SMOKE_COMPLETE")
    cfg, saved, _ = source_bundle()
    sac = ReadOnlySAC(saved)
    del saved
    started = time.perf_counter()
    manifest["status"] = "RUNNING"
    write_json(output/"manifest.json", manifest)
    all_rows, pairs, history_checks = [], [], []
    try:
        def stage(split, rules, seeds):
            references = {}
            for candidate in rules:
                for seed in seeds:
                    directory = output/split/candidate["name"]/str(seed)
                    row, meter = evaluate_rule(cfg, sac, candidate, seed, 90, directory, split)
                    if candidate["name"] == "zero":
                        references[seed] = (row, directory)
                        if split == "exploration":
                            history_checks.append(historical_zero(row, directory))
                    ref, ref_dir = references[seed]
                    paired = pair(ref, row, ref_dir, directory)
                    all_rows.append(row)
                    pairs.append(paired)
                    append_csv(output/"per_seed.csv", [{k: row[k] for k in ("split", "policy", "seed", *MEASURES)}])
                    append_csv(output/"paired_deltas.csv", [paired])
                    save_meter_summaries(output, row, meter)
                    manifest["counts"][split] += 1
                    write_json(output/"manifest.json", manifest)
                    print(f"[{split} {candidate['name']} {seed}] J={row['J']:.9f} delta={paired['delta_J']:+.9f} delay={row['task_delay_weighted_ms']:.6f}ms", flush=True)
                    del meter

        stage("exploration", candidates(), EXPLORATION)
        exploration = aggregate(all_rows, "exploration")
        append_csv(output/"summary.csv", exploration)
        selected = choose_candidate(exploration)
        selection = {"selected": selected, "basis": "exploration mean J only", "ranking": exploration,
                     "frozen_before_recheck_unix": time.time(), "protocol_sha256": manifest["protocol_sha256"]}
        write_json(output/"selection.json", selection)
        selection_hash = file_hash(output/"selection.json")
        if selected is not None:
            stage("recheck", [candidates()[0], selected], RECHECK)
            append_csv(output/"summary.csv", aggregate(all_rows, "recheck"))
        if file_hash(output/"selection.json") != selection_hash:
            raise AssertionError("Selection changed during development recheck")
        if manifest["counts"] != {"exploration": 30, "recheck": 6 if selected else 0}:
            raise AssertionError("Run-count budget mismatch")
        cfg_after, _, _ = source_bundle()
        after = source_hashes()
        if after != manifest["source_hashes_before"] or cfg_after != cfg:
            raise AssertionError("Sources/config changed during diagnosis")
        manifest.update(status="COMPLETE", source_hashes_after=after, source_unchanged=True,
                        selection_sha256=selection_hash, selected=selected,
                        elapsed_seconds=time.perf_counter()-started, frozen_checks=sac.assert_unchanged())
        write_json(output/"baseline_checks.json", history_checks)
        write_json(output/"manifest.json", manifest)
        print(json.dumps({"status": "COMPLETE", "counts": manifest["counts"], "selected": selected}), flush=True)
    except Exception as error:
        manifest.update(status="FAILED", error=repr(error))
        write_json(output/"manifest.json", manifest)
        raise
