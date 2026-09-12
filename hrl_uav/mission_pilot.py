"""Fixed ten-run mission pilot: deterministic frozen SAC, no learning path."""

import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import numpy as np
import torch

from .bandwidth import summarize_links, summarize_meter
from .experiment import append_csv, isolated_rng, rollout, write_json
from .mission_provenance import PHYSICS_BASE_COMMIT, PHYSICS_FILES, assert_committed_sources, physics_provenance
from .mission_smoke import (
    CONTROLLERS, ROOT, MissionMeter, MissionPolicy, load_source, load_spec,
    mission_config, paired_records,
)
from .priority_grid import file_hash
from .scene_prescreen import assert_crn_equal, digest_update, role_summary
from .trajectory_check import ReadOnlySAC, no_learning
from .wireless_check import ENERGY, PRIORITY_MEASURES
from .wireless_pilot import route_statistics

SCENES = (33600, 33601)
DURATION_S = 900
RUN_FILES = tuple(dict.fromkeys((*PHYSICS_FILES,
    "hrl_uav/__init__.py", "hrl_uav/agents/__init__.py", "hrl_uav/qualification.py",
    "hrl_uav/resumption.py", "hrl_uav/priority_grid.py",
    "hrl_uav/mission_smoke.py", "hrl_uav/mission_provenance.py", "hrl_uav/mission_pilot.py",
    "hrl_uav/mission_pilot_analysis.py", "scripts/run_mission_rpgm_v1_pilot.py",
    "tests/test_mission_rpgm_v1.py", "tests/test_mission_pilot.py",
    "docs/mission_rpgm_v1_pilot.md")))


def protocol():
    return dict(scenes=list(SCENES), controllers=list(CONTROLLERS), duration_s=DURATION_S,
                evaluations=10, training=False, evaluation_train=False, complete_tail=False,
                stage="macro tracking development pilot only",
                sac="seed101 window2000 deterministic frozen zero-shot",
                macro_enabled="true for fixed controller dispatch; no PPO network",
                progressive_coefficient=.5, predictive_lead_s=10,
                blind_patrol="world x axis, +2 m/s for 30s, -2 m/s for 30s, repeat; diagnostic only",
                excluded_scenes=[33610, 33611, 33612, *range(33700, 33710)],
                saturation_tolerance_m=1e-9, numerical_ordering_tolerance=1e-10,
                late_queue_slope="OLS of 0.5s frame-end total queue, 600<t<=900s",
                tracking_sampling="9001 physical-boundary positions including t=0; UAV-row pooled or per UAV",
                communication_denominators="rate: active link-step mean or sum(Mbit)/sum(actual sender seconds); throughput: /900s",
                selection="all fixed rules, no early outcome stopping or extra evaluations")


def validate_case(name, seed, seconds):
    if name not in CONTROLLERS or seed not in SCENES or seconds != DURATION_S:
        raise ValueError("Only five declared controllers x scenes33600/33601 x900s are authorized")


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return dict(mean=float(a.mean()), median=float(np.median(a)), p90=float(np.percentile(a, 90)), max=float(a.max()))


class ObservedPolicy(MissionPolicy):
    def macro_action(self, state, enabled=True, deterministic=False):
        # Read-only pre-cap request. Execution remains the archived MissionPolicy.
        if self.name == "hover":
            request = np.zeros((3, 2))
        elif self.name == "blind_patrol":
            request = np.tile([20.0 if (self.macro_calls // 3) % 2 == 0 else -20.0, 0.0], (3, 1))
        else:
            request = self.meter.env.weighted_centroids - self.meter.env.uav_coords
            if self.name == "progressive_centroid":
                request = .5 * request
            elif self.name == "predictive_centroid":
                request = request + 10.0 * self.meter.centroid_velocity
        result = super().macro_action(state, enabled, deterministic)
        expected = request / np.maximum(np.linalg.norm(request, axis=1, keepdims=True) / 20, 1) / 20
        np.testing.assert_array_equal(result[0].reshape(3, 2), expected)
        self.meter.requested_displacement = request.copy()
        return result


class MissionPilotMeter(MissionMeter):
    def __init__(self, config):
        super().__init__(config)
        self.uplink_readings = []
        self.innovation_digest = hashlib.sha256()
        self.mobility_states = 0
        update = self.env._update_gauss_markov_mobility

        def observed_mobility():
            result = update()
            digest_update(self.innovation_digest, self.env.global_micro_step,
                          self.env.user_local_velocities, self.env.mobility_group_velocities)
            self.mobility_states += 1
            return result

        self.env._update_gauss_markov_mobility = observed_mobility

    def start_macro(self, action):
        before = self.env.uav_coords.copy()
        raw = np.asarray(action).reshape(3, 2)
        mapped = np.clip(raw, -1, 1)
        mapped /= np.maximum(np.linalg.norm(mapped, axis=1, keepdims=True), 1)
        world = before + 20 * mapped
        clipped = np.clip(world, 0, 1000)
        info = super().start_macro(action)
        for i, row in enumerate(self.commands[-3:]):
            request = self.requested_displacement[i]
            row.update(requested_distance_m=float(np.linalg.norm(request)),
                       cap_triggered=int(np.linalg.norm(request) > 20+1e-9),
                       unit_action_saturated=int(np.linalg.norm(raw[i]) >= 1-1e-9),
                       actual_at_20m=int(row["distance_m"] >= 20-1e-9),
                       map_clipped=int(np.linalg.norm(clipped[i]-world[i]) > 1e-9),
                       safety_changed=int(np.linalg.norm(info["projected_uav_coords"][i]-clipped[i]) > 1e-9),
                       minimum_flight_separation_m=info["minimum_flight_separation_m"])
            for prefix, xy in (("requested_target", before[i]+request), ("pre_map_target", world[i]),
                               ("post_map_target", clipped[i])):
                row.update({f"{prefix}_{axis}": float(xy[j]) for j, axis in enumerate(("x", "y"))})
        return info

    def step_frame(self, action, gamma_frame=.99):
        frame = super().step_frame(action, gamma_frame)
        record = self.readings[-1]
        record["queue_total_mbit"] = sum(record[f"queue_{q}_mbit"] for q in ("upload", "compute", "transfer"))
        rng = self.env.np_random.get_state()
        digest = hashlib.sha256(rng[0].encode())
        digest_update(digest, rng[1], rng[2], rng[3], rng[4])
        self.crn_frames[-1].update(environment_rng_sha256=digest.hexdigest(),
                                  mobility_velocity_states_sha256=self.innovation_digest.hexdigest())
        for source in range(3):
            bits = active_time = 0.0
            for info in frame["infos"]:
                mask = info["user_associations"] == source
                work = np.asarray(info["uplink_service_mbits_per_user"], dtype=np.float64)[mask]
                duration = np.asarray(info["uplink_active_time_s_per_user"], dtype=np.float64)[mask]
                rate = np.asarray(info["uplink_rate_bps_per_user"], dtype=np.float64)[mask]
                np.testing.assert_allclose(rate*duration/1e6, work, rtol=2e-6, atol=1e-7)
                # Archived info exports times as float32: allow one float32 ULP.
                time_tolerance = float(np.spacing(np.float32(self.env.micro_dt)))
                if np.any(duration < 0) or np.any(duration > self.env.micro_dt+time_tolerance):
                    raise AssertionError("U2G sending duration outside physical step")
                bits += float(work.sum())
                active_time += float(duration.sum())
            self.uplink_readings.append(dict(time_s=record["time_s"], source=source,
                                            served_mbit=bits, tx_time_s=active_time))
        roles = {r["uav"]: r["role"] for r in self.uav_readings[-3:]}
        for row in self.link_readings[-6:]:
            row.update(source_role=roles[row["source"]], destination_role=roles[row["destination"]])
        return frame


def crn_records(meter):
    records = paired_records(meter)
    for row, original in zip(records, meter.crn_frames):
        for key in ("environment_rng_sha256", "mobility_velocity_states_sha256"):
            row[key] = original[key]
    return records


def trajectory_statistics(meter, name, seed):
    q, c, g = (np.asarray([r[k] for r in meter.positions]) for k in (1, 3, 4))
    errors = {"business_error_m": np.linalg.norm(q-c, axis=2), "group_error_m": np.linalg.norm(q-g, axis=2)}
    rows = []
    for uav in ("all", 0, 1, 2):
        commands = [r for r in meter.commands if uav == "all" or r["uav"] == uav]
        row = dict(policy=name, seed=seed, uav=uav, macro_count=len({r["macro"] for r in commands}),
                   uav_row_count=len(commands), distance_m=sum(r["distance_m"] for r in commands),
                   minimum_flight_separation_m=min(r["minimum_flight_separation_m"] for r in commands))
        for field, values in errors.items():
            selected = values.ravel() if uav == "all" else values[:, uav]
            row.update({f"{field}_{key}": value for key, value in stats(selected).items()})
        row.update({f"speed_mps_{key}": value for key, value in stats([r["speed_mps"] for r in commands]).items()})
        for flag in ("cap_triggered", "unit_action_saturated", "actual_at_20m", "map_clipped", "safety_changed"):
            selected = [r for r in commands if r[flag]]
            row[f"{flag}_rows"] = len(selected)
            row[f"{flag}_macros"] = len({r["macro"] for r in selected})
            row[f"{flag}_uavs"] = len({r["uav"] for r in selected})
            row[f"{flag}_row_fraction"] = len(selected)/len(commands)
            row[f"{flag}_macro_fraction"] = row[f"{flag}_macros"]/row["macro_count"]
        rows.append(row)
    return rows


def evaluate_one(cfg, sac, name, seed, *, _short_test=False):
    validate_case(name, seed, DURATION_S)
    seconds = 20 if _short_test else DURATION_S
    meter = MissionPilotMeter(cfg)
    policy = ObservedPolicy(sac, name)
    policy.meter = meter
    with isolated_rng(), no_learning(sac), torch.no_grad(), \
         patch("hrl_uav.experiment.QueueMDP", return_value=meter), \
         patch("hrl_uav.experiment.DirectHRL", side_effect=AssertionError("No learning agent")):
        row, macro_rows, _, _, _ = rollout(cfg, policy, seed, seconds//10, macro_enabled=True,
                                           train=False, complete_tail=False, record=True)
    if (meter.reset_count != 1 or meter.integration_steps != seconds*10 or row["time_s"] != seconds
            or len(meter.positions) != seconds*10+1 or policy.macro_calls != seconds//10
            or len(meter.micro_states) != seconds*2 or meter.mobility_states != seconds*2
            or row["boundary_transitions"] or row["final_pending_dropped"]
            or meter.env.macro_fly_steps_remaining):
        raise AssertionError("Clock/reset/decision/boundary conservation failed")
    row.update(policy=name, seed=seed, duration_s=seconds, **summarize_meter(meter.readings, seconds))
    row.update({key: meter.readings[-1][key] for key in PRIORITY_MEASURES})
    row.update({key: sum(r[key] for r in meter.readings) for key in ENERGY})
    row.update({f"queue_{q}_mbit": meter.readings[-1][f"queue_{q}_mbit"] for q in ("upload", "compute", "transfer")})
    tail = [r for r in meter.readings if r["time_s"] > seconds*2/3]
    row.update(J_holding=sum(r["holding_cost"] for r in meter.readings)/meter.integration_steps,
               J_energy=sum(r["energy_cost"] for r in meter.readings)/meter.integration_steps,
               propulsion_increment_j=sum(r["propulsion_increment_j"] for r in meter.readings),
               late_queue_slope_mbit_s=float(np.polyfit([r["time_s"] for r in tail], [r["queue_total_mbit"] for r in tail], 1)[0]),
               u2g_served_mbit=sum(r["served_mbit"] for r in meter.uplink_readings),
               u2g_tx_time_s=sum(r["tx_time_s"] for r in meter.uplink_readings),
               u2u_tx_time_s=sum(r["tx_time_s"] for r in meter.link_readings),
               boundary_user_hits=meter.env.mission_user_boundary_hits,
               sac_decisions=len(meter.micro_states), macro_decisions=policy.macro_calls,
               sac_update_increment=0, ppo_updates=0, replay_writes=0, replay_samples=0)
    row["u2g_rate_time_weighted_mbps"] = row["u2g_served_mbit"]/max(row["u2g_tx_time_s"], 1e-12)
    row["u2u_rate_time_weighted_mbps"] = row["tx_served_mbit"]/max(row["u2u_tx_time_s"], 1e-12)
    np.testing.assert_allclose(row["J"], row["J_holding"]+row["J_energy"], rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(row["power_w"]*seconds, sum(row[k] for k in ENERGY), rtol=1e-12, atol=1e-6)
    np.testing.assert_allclose(row["queue_total_mbit"], sum(row[f"queue_{q}_mbit"] for q in ("upload", "compute", "transfer")), atol=1e-7)
    trajectory = trajectory_statistics(meter, name, seed)
    row.update({key: value for key, value in trajectory[0].items() if key not in ("policy", "seed", "uav")})
    row["fleet_distance_m"] = row["distance_m"]
    expected = meter.env.mission_routes_m[:, 0] + min(seconds/900, 1)*(meter.env.mission_routes_m[:, 1]-meter.env.mission_routes_m[:, 0])
    row["group_endpoint_max_error_m"] = float(np.linalg.norm(meter.env.mobility_group_centers-expected, axis=1).max())
    np.testing.assert_allclose(meter.env.mobility_group_centers, expected, rtol=0, atol=1e-8)
    for key, value in row.items():
        if isinstance(value, (int, float)) and not np.isfinite(value):
            raise AssertionError(f"Nonfinite metric: {key}")
    return row, meter, trajectory, macro_rows


def save_case(directory, name, seed, row, meter, trajectory, macro_rows, sac):
    directory.mkdir(parents=True, exist_ok=False)
    for filename, rows in (("queues", meter.readings), ("commands", meter.commands), ("uplink", meter.uplink_readings),
                           ("links", meter.link_readings), ("uavs", meter.uav_readings), ("routes", meter.route_readings),
                           ("macro_metrics", macro_rows), ("trajectory_summary", trajectory),
                           ("role_summary", role_summary(meter.uav_readings, .5, name, seed)),
                           ("route_summary", route_statistics(meter.route_readings, name, seed, 2000, "mission_pilot")),
                           ("link_summary", summarize_links(meter.link_readings, row["duration_s"], seed, 6))):
        append_csv(directory / f"{filename}.csv", rows)
    write_json(directory / "crn.json", crn_records(meter))
    write_json(directory / "freeze_check.json", sac.assert_unchanged())
    write_json(directory / "summary.json", row)
    write_json(directory / "group_endpoints.json", dict(
        time_s=row["duration_s"], actual_m=meter.env.mobility_group_centers.tolist(),
        declared_m=meter.env.mission_routes_m[:, 1].tolist(),
        error_m=np.linalg.norm(meter.env.mobility_group_centers-meter.env.mission_routes_m[:, 1], axis=1).tolist()))
    np.savez_compressed(directory / "trajectory.npz", time_s=[r[0] for r in meter.positions],
                        uav=[r[1] for r in meter.positions], users=[r[2] for r in meter.positions],
                        centroid=[r[3] for r in meter.positions], groups=[r[4] for r in meter.positions],
                        micro_state=meter.micro_states)
    errors = []
    for t, q, _, c, g in meter.positions:
        for i in range(3):
            errors.append(dict(time_s=t, uav=i, business_error_m=float(np.linalg.norm(q[i]-c[i])),
                               group_error_m=float(np.linalg.norm(q[i]-g[i]))))
    append_csv(directory / "tracking_error.csv", errors)


def run(output):
    from .mission_pilot_analysis import analyze
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Pilot output must be new; no overwrite or automatic rerun")
    physics = physics_provenance()
    head = assert_committed_sources(RUN_FILES)
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if branch != "codex/mission_rpgm_v1":
        raise ValueError("Pilot must run on codex/mission_rpgm_v1")
    spec = load_spec()
    saved, source_manifest = load_source(spec)
    cfg = mission_config(saved["config"], spec)
    hashes = {name: file_hash(ROOT / name) for name in RUN_FILES}
    with isolated_rng():
        sac = ReadOnlySAC(saved)
    del saved
    gc.collect()
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="RUNNING", actual_head=head, physics_base_commit=PHYSICS_BASE_COMMIT,
                    branch=branch, physics_hashes=physics, config=cfg, protocol=protocol(),
                    source_hashes_before=hashes, source_checkpoint=spec["source_checkpoint"],
                    source_checkpoint_sha256=spec["checkpoint_sha256"], source_config=source_manifest["config"],
                    source_manifest_sha256=file_hash((ROOT / spec["source_checkpoint"]).parent.parent / "manifest.json"),
                    training=False, training_seed_count=0, source_training_initializations=[101],
                    deterministic=True, zero_shot=True, reserved_scenes_executed=[],
                    completed_cases=[], scene_count=2, scope="reused development scenes, not final test or significance",
                    denominators=protocol()["communication_denominators"])
    write_json(output / "manifest.json", manifest)
    rows, tracking = [], []
    started = time.perf_counter()
    try:
        for seed in SCENES:
            reference = None
            for name in CONTROLLERS:
                print(f"[mission pilot start] seed={seed} policy={name} seconds=900", flush=True)
                row, meter, trajectory, macro_rows = evaluate_one(cfg, sac, name, seed)
                records = crn_records(meter)
                if reference is None:
                    reference = records
                assert_crn_equal(reference, records)
                save_case(output / f"{name}_{seed}", name, seed, row, meter, trajectory, macro_rows, sac)
                append_csv(output / "per_seed.csv", [row])
                append_csv(output / "trajectory_summary.csv", trajectory)
                rows.append(row)
                tracking.extend(trajectory)
                manifest["completed_cases"].append(dict(seed=seed, policy=name, crn_equal=True, elapsed_seconds=time.perf_counter()-started))
                write_json(output / "manifest.json", manifest)
                print(f"[mission pilot {len(rows)}/10] J={row['J']:.6f} holding={row['J_holding']:.6f} power={row['power_w']:.3f}W", flush=True)
                del meter, macro_rows
                gc.collect()
        if len(rows) != 10 or {(r["seed"], r["policy"]) for r in rows} != {(s, p) for s in SCENES for p in CONTROLLERS}:
            raise AssertionError("Pilot evaluation matrix mismatch")
        analyze(output, rows, manifest)
        manifest.update(source_hashes_after={name: file_hash(ROOT / name) for name in RUN_FILES},
                        frozen_sac=sac.assert_unchanged(), crn_all_equal=True, elapsed_seconds=time.perf_counter()-started)
        manifest["source_unchanged"] = hashes == manifest["source_hashes_after"]
        if not manifest["source_unchanged"] or physics != physics_provenance():
            raise AssertionError("Source/physics changed during pilot")
        if file_hash(ROOT / spec["source_checkpoint"]) != spec["checkpoint_sha256"]:
            raise AssertionError("Source checkpoint changed")
        manifest["status"] = "COMPLETE_TEN_CASES_STOPPED_NO_TRAINING"
        write_json(output / "manifest.json", manifest)
    except Exception as exc:
        manifest.update(status="FAILED_STOPPED", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "manifest.json", manifest)
        raise
    return dict(output=str(output), status=manifest["status"], evaluations=len(rows))
