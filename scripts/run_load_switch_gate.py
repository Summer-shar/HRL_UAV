import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap

PROJECT_ROOT = Path(bootstrap())

from search_position_oracle import load_checkpoint_components
from search_position_oracle import target_centroid_waypoint_action


LOAD_PHASES_MBPS = np.asarray(
    [
        [12.5, 5.5, 3.5],
        [3.5, 12.5, 5.5],
        [5.5, 3.5, 12.5],
    ],
    dtype=np.float64,
)
POLICY_NAMES = ("P0", "P1", "P2")
POLICY_LABELS_ZH = {
    "P0": "P0 纯质心跟踪",
    "P1": "P1 因果负载感知",
    "P2": "P2 非因果预知上限",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the frozen load-switching position-value gate."
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "results/tcom_dynamic_u2u_sparse_v23_target_noise/pilot/"
            "seed1/joint_td3_sac/checkpoints/best.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/load_switch_gate",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=65000)
    parser.add_argument("--macro-steps", type=int, default=30)
    parser.add_argument("--switch-period", type=int, default=10)
    parser.add_argument("--offset-m", type=float, default=40.0)
    parser.add_argument("--anticipation-steps", type=int, default=5)
    parser.add_argument("--practical-gain-ms", type=float, default=5.0)
    parser.add_argument("--power-budget-w", type=float, default=840.0)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
    )
    parser.add_argument(
        "--policies",
        default=",".join(POLICY_NAMES),
        help="Comma-separated subset of P0,P1,P2; formal gate requires all.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Regenerate statistics and figures from an existing completed run.",
    )
    return parser.parse_args()


def resolve_output_dir(path_value):
    output = Path(path_value)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    return output.resolve()


def phase_index_for_step(macro_index, switch_period=10):
    return min(int(macro_index) // int(switch_period), len(LOAD_PHASES_MBPS) - 1)


def switch_happens_at_step(macro_index, switch_period=10):
    macro_index = int(macro_index)
    return macro_index > 0 and macro_index % int(switch_period) == 0


def weighted_node_queue_scores(base_env, queue_norm):
    q_high = (
        np.asarray(base_env.Q_loc_high, dtype=np.float64)
        + np.sum(np.asarray(base_env.Q_tx_high, dtype=np.float64), axis=0)
        + np.sum(np.asarray(base_env.Q_tx_high, dtype=np.float64), axis=1)
    )
    q_low = (
        np.asarray(base_env.Q_loc_low, dtype=np.float64)
        + np.sum(np.asarray(base_env.Q_tx_low, dtype=np.float64), axis=0)
        + np.sum(np.asarray(base_env.Q_tx_low, dtype=np.float64), axis=1)
    )
    denominator = max(float(queue_norm), 1e-12)
    return np.clip(q_high / denominator, 0.0, 1.0) + np.clip(
        q_low / denominator,
        0.0,
        1.0,
    )


def choose_support_uav(base_env, heavy_uav, queue_norm):
    scores = weighted_node_queue_scores(base_env, queue_norm)
    candidates = [index for index in range(int(base_env.N)) if index != heavy_uav]
    return min(candidates, key=lambda index: (float(scores[index]), int(index)))


class GatePolicyController:
    """Stateful implementation of the frozen P0/P1/P2 rules."""

    def __init__(
        self,
        name,
        switch_period=10,
        offset_m=40.0,
        anticipation_steps=5,
    ):
        if name not in POLICY_NAMES:
            raise ValueError(f"Unknown gate policy: {name}")
        self.name = name
        self.switch_period = int(switch_period)
        self.offset_m = float(offset_m)
        self.anticipation_steps = int(anticipation_steps)
        self.active_heavy = None
        self.active_support = None
        self.pending_role = None
        self.pending_activation_step = None
        self.previous_observed_heavy = None
        self.last_event = ""

    @staticmethod
    def observed_heavy_uav(base_env):
        loads = np.asarray(
            base_env.zone_expected_offered_load_mbps,
            dtype=np.float64,
        )
        return int(np.argmax(loads))

    def _role_for_heavy(self, env, heavy_uav):
        support = choose_support_uav(
            env.env,
            heavy_uav,
            env.macro_queue_norm,
        )
        return int(heavy_uav), int(support)

    def reset(self, env):
        observed_heavy = self.observed_heavy_uav(env.env)
        self.previous_observed_heavy = observed_heavy
        self.pending_role = None
        self.pending_activation_step = None
        if self.name == "P0":
            self.active_heavy = None
            self.active_support = None
            self.last_event = "centroid_only"
        else:
            self.active_heavy, self.active_support = self._role_for_heavy(
                env,
                observed_heavy,
            )
            self.last_event = "initial_role"

    def before_macro(self, env, macro_index, switched):
        macro_index = int(macro_index)
        observed_heavy = self.observed_heavy_uav(env.env)
        detected_switch = (
            self.previous_observed_heavy is not None
            and observed_heavy != self.previous_observed_heavy
        )
        if bool(switched) != bool(detected_switch):
            raise RuntimeError(
                "Load-switch flag and observable load ranking disagree at "
                f"macro step {macro_index}: flag={switched}, "
                f"detected={detected_switch}."
            )
        self.previous_observed_heavy = observed_heavy

        if self.name == "P0":
            self.last_event = "centroid_only"
            return

        if self.name == "P1":
            if switched:
                self.pending_role = self._role_for_heavy(env, observed_heavy)
                self.pending_activation_step = macro_index + 1
                self.active_heavy = None
                self.active_support = None
                self.last_event = "switch_detected_wait_one_period"
                return
            if (
                self.pending_role is not None
                and macro_index >= int(self.pending_activation_step)
            ):
                self.active_heavy, self.active_support = self.pending_role
                self.pending_role = None
                self.pending_activation_step = None
                self.last_event = "new_role_activated"
            else:
                self.last_event = "maintain_role"
            return

        anticipation_offset = macro_index % self.switch_period
        anticipation_start = self.switch_period - self.anticipation_steps
        if (
            anticipation_offset == anticipation_start
            and macro_index + self.anticipation_steps < 30
        ):
            next_phase = phase_index_for_step(
                macro_index + self.anticipation_steps,
                self.switch_period,
            )
            next_heavy = int(np.argmax(LOAD_PHASES_MBPS[next_phase]))
            self.active_heavy, self.active_support = self._role_for_heavy(
                env,
                next_heavy,
            )
            self.last_event = "noncausal_anticipation_started"
        elif switched:
            if self.active_heavy != observed_heavy:
                self.active_heavy, self.active_support = self._role_for_heavy(
                    env,
                    observed_heavy,
                )
                self.last_event = "fallback_role_at_switch"
            else:
                self.last_event = "anticipated_role_continues"
        else:
            self.last_event = "maintain_role"

    def action(self, env):
        if env.macro_action_mode != "centroid_waypoint":
            raise ValueError(
                "The frozen gate requires macro_action_mode=centroid_waypoint."
            )
        raw_state = env.env.get_raw_state()
        centroids = np.asarray(raw_state["weighted_centroids"], dtype=np.float64)
        targets = centroids.copy()
        if self.active_support is not None and self.active_heavy is not None:
            support = int(self.active_support)
            heavy = int(self.active_heavy)
            toward_heavy = (
                np.asarray(raw_state["uav_coords"][heavy], dtype=np.float64)
                - centroids[support]
            )
            norm = float(np.linalg.norm(toward_heavy))
            if norm > 1e-12:
                targets[support] += toward_heavy / norm * self.offset_m
        targets = np.clip(targets, 0.0, 1000.0)
        return target_centroid_waypoint_action(
            targets,
            centroids,
            env.macro_waypoint_radius_m,
        )

    def metadata(self):
        return {
            "heavy_uav": -1 if self.active_heavy is None else self.active_heavy,
            "support_uav": (
                -1 if self.active_support is None else self.active_support
            ),
            "event": self.last_event,
        }


def _hash_array(hasher, value):
    array = np.ascontiguousarray(np.asarray(value))
    hasher.update(str(array.dtype).encode("ascii"))
    hasher.update(str(array.shape).encode("ascii"))
    hasher.update(array.tobytes())


def update_external_trace_hash(hasher, env, raw_state, info, phase_loads):
    _hash_array(hasher, raw_state["user_coords"])
    _hash_array(hasher, info.get("arrival_high_per_uav", np.zeros(env.env.N)))
    _hash_array(hasher, info.get("arrival_low_per_uav", np.zeros(env.env.N)))
    _hash_array(hasher, phase_loads)
    counters = env.env.get_task_cumulative_counters()
    _hash_array(hasher, counters["arrived_source_high_count"])
    _hash_array(hasher, counters["arrived_source_low_count"])
    rng = env.env.np_random
    if hasattr(rng, "bit_generator"):
        rng_state = json.dumps(
            rng.bit_generator.state,
            sort_keys=True,
            separators=(",", ":"),
        )
        hasher.update(rng_state.encode("utf-8"))
    else:
        state_name, state_values, state_position, has_gauss, cached_gauss = (
            rng.get_state()
        )
        hasher.update(str(state_name).encode("ascii"))
        _hash_array(hasher, state_values)
        hasher.update(str(int(state_position)).encode("ascii"))
        hasher.update(str(int(has_gauss)).encode("ascii"))
        hasher.update(repr(float(cached_gauss)).encode("ascii"))


def task_window_delta(before, after, high_weight, low_weight):
    high_count = int(after["completed_high_count"] - before["completed_high_count"])
    low_count = int(after["completed_low_count"] - before["completed_low_count"])
    high_delay = float(
        after["completed_high_delay_sum_s"]
        - before["completed_high_delay_sum_s"]
    )
    low_delay = float(
        after["completed_low_delay_sum_s"]
        - before["completed_low_delay_sum_s"]
    )
    numerator = high_weight * high_delay + low_weight * low_delay
    denominator = high_weight * high_count + low_weight * low_count
    return {
        "weighted_delay_numerator_s": float(numerator),
        "weighted_delay_denominator": float(denominator),
        "completed_high_count": high_count,
        "completed_low_count": low_count,
    }


def run_gate_episode(
    env,
    checkpoint_args,
    micro_policy,
    policy_name,
    seed,
    macro_steps=30,
    switch_period=10,
    offset_m=40.0,
    anticipation_steps=5,
):
    base_env = env.env
    base_env.set_zone_target_loads(LOAD_PHASES_MBPS[0])
    env.reset(seed=int(seed))
    base_env.set_zone_target_loads(LOAD_PHASES_MBPS[0])

    controller = GatePolicyController(
        policy_name,
        switch_period=switch_period,
        offset_m=offset_m,
        anticipation_steps=anticipation_steps,
    )
    controller.reset(env)

    external_hasher = hashlib.sha256()
    total_service_energy_j = 0.0
    total_macro_energy_j = 0.0
    total_u2g_active_time_s = 0.0
    total_u2u_active_time_s = 0.0
    total_flow = np.zeros((base_env.N, base_env.N), dtype=np.float64)
    phase_flow = np.zeros(
        (len(LOAD_PHASES_MBPS), base_env.N, base_env.N),
        dtype=np.float64,
    )
    phase_u2g_time = np.zeros(len(LOAD_PHASES_MBPS), dtype=np.float64)
    phase_u2u_time = np.zeros(len(LOAD_PHASES_MBPS), dtype=np.float64)
    transition_numerator_s = 0.0
    transition_denominator = 0.0
    micro_decision_count = 0
    macro_records = []
    uav_trajectory = [np.asarray(base_env.uav_coords, dtype=np.float64).copy()]
    centroid_trajectory = [
        np.asarray(base_env.weighted_centroids, dtype=np.float64).copy()
    ]
    trajectory_phases = [0]

    high_weight = float(base_env.w_D_high)
    low_weight = float(base_env.w_D_low)
    started_at = time.perf_counter()
    done = False

    for macro_index in range(int(macro_steps)):
        phase_index = phase_index_for_step(macro_index, switch_period)
        switched = switch_happens_at_step(macro_index, switch_period)
        if macro_index == 0 or switched:
            base_env.set_zone_target_loads(LOAD_PHASES_MBPS[phase_index])

        controller.before_macro(env, macro_index, switched)
        local_state, global_state = env.get_macro_state()
        macro_action = np.asarray(controller.action(env), dtype=np.float32)
        counters_before = base_env.get_task_cumulative_counters()
        uav_before = np.asarray(base_env.uav_coords, dtype=np.float64).copy()
        centroids_before = np.asarray(
            base_env.weighted_centroids,
            dtype=np.float64,
        ).copy()

        _, _, mac_info = env.macro_step(macro_action)
        total_macro_energy_j += float(mac_info.get("E_macro_total", 0.0))
        macro_flow = np.zeros((base_env.N, base_env.N), dtype=np.float64)
        macro_u2g_time_s = 0.0
        macro_u2u_time_s = 0.0

        for _ in range(base_env.resource_frames_per_macro):
            service_paused = bool(
                base_env.pause_service_during_flight
                and base_env.macro_service_pause_steps_remaining > 0
            )
            if service_paused:
                micro_action = np.zeros(
                    env.micro_action_space.shape[0],
                    dtype=np.float32,
                )
            else:
                env.prepare_resource_frame()
                micro_action = np.asarray(micro_policy(env, True), dtype=np.float32)
                micro_decision_count += 1
            frame = env.step_resource_frame(micro_action)
            done = bool(frame["done"])
            for raw_state, info in zip(frame["raw_states"], frame["infos"]):
                total_service_energy_j += float(info.get("total_energy", 0.0))
                tx_flow = np.asarray(
                    info.get(
                        "tx_service_matrix_mbits",
                        np.zeros((base_env.N, base_env.N)),
                    ),
                    dtype=np.float64,
                )
                u2g_time = float(
                    np.sum(
                        np.asarray(
                            info.get("uplink_active_time_s_per_user", 0.0),
                            dtype=np.float64,
                        )
                    )
                )
                u2u_time = float(
                    np.sum(
                        np.asarray(
                            info.get("u2u_active_time_s", 0.0),
                            dtype=np.float64,
                        )
                    )
                )
                macro_flow += tx_flow
                macro_u2g_time_s += u2g_time
                macro_u2u_time_s += u2u_time
                update_external_trace_hash(
                    external_hasher,
                    env,
                    raw_state,
                    info,
                    LOAD_PHASES_MBPS[phase_index],
                )
            if done:
                break

        macro_reward = float(env.get_macro_reward())
        macro_reward_info = env.get_last_macro_reward_info()
        counters_after = base_env.get_task_cumulative_counters()
        window_delta = task_window_delta(
            counters_before,
            counters_after,
            high_weight,
            low_weight,
        )
        in_transition_window = (
            switch_period <= macro_index < switch_period + 5
            or 2 * switch_period <= macro_index < 2 * switch_period + 5
        )
        if in_transition_window:
            transition_numerator_s += window_delta[
                "weighted_delay_numerator_s"
            ]
            transition_denominator += window_delta[
                "weighted_delay_denominator"
            ]

        total_flow += macro_flow
        phase_flow[phase_index] += macro_flow
        total_u2g_active_time_s += macro_u2g_time_s
        total_u2u_active_time_s += macro_u2u_time_s
        phase_u2g_time[phase_index] += macro_u2g_time_s
        phase_u2u_time[phase_index] += macro_u2u_time_s
        metadata = controller.metadata()
        uav_after = np.asarray(base_env.uav_coords, dtype=np.float64).copy()
        centroids_after = np.asarray(
            base_env.weighted_centroids,
            dtype=np.float64,
        ).copy()
        macro_records.append(
            {
                "seed": int(seed),
                "policy": policy_name,
                "macro_index": int(macro_index),
                "phase_index": int(phase_index),
                "switch_flag": int(switched),
                "expected_loads_mbps": np.asarray(
                    base_env.zone_expected_offered_load_mbps,
                    dtype=np.float64,
                ).copy(),
                "heavy_uav": int(metadata["heavy_uav"]),
                "support_uav": int(metadata["support_uav"]),
                "policy_event": str(metadata["event"]),
                "macro_action": macro_action.copy(),
                "uav_before_m": uav_before,
                "uav_after_m": uav_after,
                "centroids_before_m": centroids_before,
                "centroids_after_m": centroids_after,
                "tx_flow_mbits": macro_flow.copy(),
                "u2g_active_time_s": macro_u2g_time_s,
                "u2u_active_time_s": macro_u2u_time_s,
                "macro_reward": float(macro_reward),
                "average_micro_reward": float(
                    macro_reward_info.get("micro_reward_mean", 0.0)
                ),
                "global_state_dim": int(np.asarray(global_state).size),
            }
        )
        uav_trajectory.append(uav_after)
        centroid_trajectory.append(centroids_after)
        trajectory_phases.append(phase_index)
        if done:
            break

    if len(macro_records) != int(macro_steps):
        raise RuntimeError(
            f"Episode ended after {len(macro_records)} macro steps; "
            f"expected {macro_steps}."
        )

    task_stats = base_env.get_task_delay_stats()
    episode_duration_s = max(float(macro_steps) * float(base_env.macro_dt), 1e-12)
    total_energy_j = total_service_energy_j + total_macro_energy_j
    cross_flow = float(total_flow.sum() - np.trace(total_flow))
    flow_sum = float(total_flow.sum())
    transition_delay_ms = (
        1000.0 * transition_numerator_s / transition_denominator
        if transition_denominator > 0.0
        else float("nan")
    )
    result = {
        "seed": int(seed),
        "policy": policy_name,
        "task_delay_weighted_ms": float(
            task_stats.get("task_delay_weighted_ms", 0.0)
        ),
        "task_delay_high_ms": float(task_stats.get("task_delay_high_ms", 0.0)),
        "task_delay_low_ms": float(task_stats.get("task_delay_low_ms", 0.0)),
        "transition_window_delay_weighted_ms": float(transition_delay_ms),
        "transition_window_weighted_completed": float(transition_denominator),
        "average_power_w": float(total_energy_j / episode_duration_s),
        "total_energy_j": float(total_energy_j),
        "u2g_active_time_s": float(total_u2g_active_time_s),
        "u2u_active_time_s": float(total_u2u_active_time_s),
        "coop_ratio": float(cross_flow / max(flow_sum, 1e-12)),
        "cross_flow_mbits": cross_flow,
        "task_arrived_count": int(task_stats.get("task_arrived_count", 0)),
        "task_completed_count": int(task_stats.get("task_completed_count", 0)),
        "external_trace_hash": external_hasher.hexdigest(),
        "micro_decision_count": int(micro_decision_count),
        "elapsed_s": float(time.perf_counter() - started_at),
    }
    trace = {
        "macro_records": macro_records,
        "uav_trajectory_m": np.stack(uav_trajectory, axis=0),
        "centroid_trajectory_m": np.stack(centroid_trajectory, axis=0),
        "trajectory_phases": np.asarray(trajectory_phases, dtype=np.int64),
        "phase_flow_mbits": phase_flow,
        "total_flow_mbits": total_flow,
        "phase_u2g_active_time_s": phase_u2g_time,
        "phase_u2u_active_time_s": phase_u2u_time,
    }
    return result, trace


RAW_FIELDNAMES = (
    "seed",
    "policy",
    "task_delay_weighted_ms",
    "task_delay_high_ms",
    "task_delay_low_ms",
    "transition_window_delay_weighted_ms",
    "transition_window_weighted_completed",
    "average_power_w",
    "total_energy_j",
    "u2g_active_time_s",
    "u2u_active_time_s",
    "coop_ratio",
    "cross_flow_mbits",
    "task_arrived_count",
    "task_completed_count",
    "external_trace_hash",
    "micro_decision_count",
    "elapsed_s",
)


def write_csv_atomic(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    os.replace(temporary, path)


def read_raw_rows(path):
    integer_fields = {
        "seed",
        "task_arrived_count",
        "task_completed_count",
        "micro_decision_count",
    }
    string_fields = {"policy", "external_trace_hash"}
    rows = []
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            row = {}
            for key, value in raw.items():
                if key in string_fields:
                    row[key] = value
                elif key in integer_fields:
                    row[key] = int(value)
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def load_saved_traces(output_dir, rows):
    traces = {}
    for row in rows:
        policy = str(row["policy"])
        seed = int(row["seed"])
        path = output_dir / "trajectories" / f"{policy}_seed{seed}.npz"
        with np.load(path) as payload:
            traces[(policy, seed)] = {
                "uav_trajectory_m": np.asarray(
                    payload["uav_trajectory_m"], dtype=np.float64
                ),
                "centroid_trajectory_m": np.asarray(
                    payload["centroid_trajectory_m"], dtype=np.float64
                ),
                "trajectory_phases": np.asarray(
                    payload["trajectory_phases"], dtype=np.int64
                ),
                "phase_flow_mbits": np.asarray(
                    payload["phase_flow_mbits"], dtype=np.float64
                ),
            }
    return traces


def write_json_atomic(path, payload):
    def ready(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, dict):
            return {key: ready(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [ready(item) for item in value]
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(ready(payload), handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def save_macro_trace(path, records, num_uavs):
    rows = []
    for record in records:
        row = {
            "seed": record["seed"],
            "policy": record["policy"],
            "macro_index": record["macro_index"],
            "phase_index": record["phase_index"],
            "switch_flag": record["switch_flag"],
            "heavy_uav": record["heavy_uav"],
            "support_uav": record["support_uav"],
            "policy_event": record["policy_event"],
            "u2g_active_time_s": record["u2g_active_time_s"],
            "u2u_active_time_s": record["u2u_active_time_s"],
            "macro_reward": record["macro_reward"],
            "average_micro_reward": record["average_micro_reward"],
        }
        for index, value in enumerate(record["expected_loads_mbps"]):
            row[f"load_uav{index + 1}_mbps"] = float(value)
        for uav in range(num_uavs):
            for axis, axis_name in enumerate(("x", "y")):
                row[f"uav{uav + 1}_{axis_name}_m"] = float(
                    record["uav_after_m"][uav, axis]
                )
                row[f"centroid{uav + 1}_{axis_name}_m"] = float(
                    record["centroids_after_m"][uav, axis]
                )
                row[f"action_uav{uav + 1}_{axis_name}"] = float(
                    record["macro_action"][uav, axis]
                )
        rows.append(row)
    fieldnames = list(rows[0].keys()) if rows else []
    write_csv_atomic(path, rows, fieldnames)


def paired_summary(candidate_rows, reference_rows, metric):
    candidate_by_seed = {int(row["seed"]): float(row[metric]) for row in candidate_rows}
    reference_by_seed = {int(row["seed"]): float(row[metric]) for row in reference_rows}
    seeds = sorted(set(candidate_by_seed) & set(reference_by_seed))
    values = np.asarray(
        [candidate_by_seed[seed] - reference_by_seed[seed] for seed in seeds],
        dtype=np.float64,
    )
    count = int(values.size)
    mean = float(np.mean(values)) if count else float("nan")
    std = float(np.std(values, ddof=1)) if count > 1 else 0.0
    if count > 1:
        half_width = float(student_t.ppf(0.975, count - 1)) * std / math.sqrt(count)
    else:
        half_width = float("inf")
    effect = mean / std if std > 1e-12 else float("nan")
    return {
        "seeds": seeds,
        "values": values,
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "cohen_dz": effect,
        "sample_count": count,
    }


def configure_chinese_font():
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    candidates = ("Microsoft YaHei", "SimHei", "STSong")
    available = {font.name for font in fm.fontManager.ttflist}
    for candidate in candidates:
        if candidate in available:
            plt.rcParams["font.sans-serif"] = [candidate, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def plot_trajectories(output_dir, representative_traces):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    configure_chinese_font()
    phase_colors = ("#4472C4", "#ED7D31", "#70AD47")
    uav_markers = ("o", "s", "^")
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.0), sharex=True, sharey=True)
    for axis, policy_name in zip(axes, POLICY_NAMES):
        trace = representative_traces[policy_name]
        uav = np.asarray(trace["uav_trajectory_m"], dtype=np.float64)
        centroids = np.asarray(trace["centroid_trajectory_m"], dtype=np.float64)
        phases = np.asarray(trace["trajectory_phases"], dtype=np.int64)
        for uav_index in range(uav.shape[1]):
            axis.plot(
                centroids[:, uav_index, 0],
                centroids[:, uav_index, 1],
                color="#9E9E9E",
                linestyle=":",
                linewidth=1.0,
                alpha=0.75,
            )
            for step in range(uav.shape[0] - 1):
                phase = int(phases[min(step + 1, phases.size - 1)])
                axis.plot(
                    uav[step : step + 2, uav_index, 0],
                    uav[step : step + 2, uav_index, 1],
                    color=phase_colors[phase],
                    linewidth=2.0,
                    alpha=0.9,
                )
            axis.scatter(
                uav[:, uav_index, 0],
                uav[:, uav_index, 1],
                marker=uav_markers[uav_index],
                s=20,
                c=[phase_colors[int(value)] for value in phases],
                edgecolors="white",
                linewidths=0.35,
                zorder=3,
            )
            axis.scatter(
                uav[0, uav_index, 0],
                uav[0, uav_index, 1],
                marker="D",
                s=48,
                color="black",
                zorder=4,
            )
            axis.scatter(
                uav[-1, uav_index, 0],
                uav[-1, uav_index, 1],
                marker="*",
                s=110,
                color="black",
                zorder=4,
            )
        axis.set_title(POLICY_LABELS_ZH[policy_name], fontsize=12)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        axis.set_xlabel("X 坐标 (m)")
    axes[0].set_ylabel("Y 坐标 (m)")
    all_points = np.concatenate(
        [trace["uav_trajectory_m"] for trace in representative_traces.values()],
        axis=0,
    )
    lower = np.maximum(np.min(all_points, axis=(0, 1)) - 70.0, 0.0)
    upper = np.minimum(np.max(all_points, axis=(0, 1)) + 70.0, 1000.0)
    for axis in axes:
        axis.set_xlim(lower[0], upper[0])
        axis.set_ylim(lower[1], upper[1])
    handles = [
        Line2D([0], [0], color=color, lw=2.5, label=f"负载阶段 {index + 1}")
        for index, color in enumerate(phase_colors)
    ]
    handles.extend(
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor="#666666",
            markeredgecolor="white",
            markersize=7,
            label=f"UAV-{index + 1}",
        )
        for index, marker in enumerate(uav_markers)
    )
    handles.extend(
        [
            Line2D(
                [0], [0], color="#9E9E9E", linestyle=":", lw=1.5,
                label="业务质心轨迹",
            ),
            Line2D(
                [0], [0], marker="D", color="none", markerfacecolor="black",
                markersize=6, label="起点",
            ),
            Line2D(
                [0], [0], marker="*", color="none", markerfacecolor="black",
                markersize=9, label="终点",
            ),
        ]
    )
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("负载切换下三种固定策略的无人机轨迹", fontsize=15)
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.93))
    target = output_dir / "trajectories" / "load_switch_trajectories"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_gate_summary(output_dir, grouped, delay_p1, delay_p2, power_budget_w):
    import matplotlib.pyplot as plt

    configure_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))

    delta_sets = [
        np.asarray(delay_p1["values"], dtype=np.float64),
        np.asarray(delay_p2["values"], dtype=np.float64),
    ]
    colors = ("#4472C4", "#ED7D31")
    labels = ("P1−P0", "P2−P0")
    for index, (values, color, label) in enumerate(
        zip(delta_sets, colors, labels), start=1
    ):
        jitter = np.linspace(-0.08, 0.08, values.size)
        axes[0].scatter(
            np.full(values.size, index) + jitter,
            values,
            s=28,
            facecolors="white",
            edgecolors=color,
            linewidths=1.0,
            alpha=0.9,
        )
        mean = float(np.mean(values))
        half_width = float(student_t.ppf(0.975, values.size - 1)) * float(
            np.std(values, ddof=1)
        ) / math.sqrt(values.size)
        axes[0].errorbar(
            index,
            mean,
            yerr=half_width,
            fmt="o",
            color=color,
            capsize=5,
            markersize=7,
            linewidth=2.0,
            label=label,
        )
    axes[0].axhline(0.0, color="#333333", linewidth=1.0)
    axes[0].axhline(-5.0, color="#C00000", linestyle="--", linewidth=1.2)
    axes[0].text(
        0.98,
        -5.0,
        "−5 ms 门槛",
        transform=axes[0].get_yaxis_transform(),
        va="center",
        ha="right",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
    )
    axes[0].set_xticks([1, 2], labels)
    axes[0].set_ylabel("相对 P0 的配对时延差 (ms)")
    axes[0].set_title("(a) 时延门槛：通过")
    axes[0].grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)

    policy_colors = ("#A5A5A5", "#4472C4", "#ED7D31")
    for index, (policy, color) in enumerate(
        zip(POLICY_NAMES, policy_colors), start=1
    ):
        values = np.asarray(
            [float(row["average_power_w"]) for row in grouped[policy]],
            dtype=np.float64,
        )
        jitter = np.linspace(-0.08, 0.08, values.size)
        axes[1].scatter(
            np.full(values.size, index) + jitter,
            values,
            s=27,
            facecolors="white",
            edgecolors=color,
            linewidths=1.0,
            alpha=0.9,
        )
        half_width = float(student_t.ppf(0.975, values.size - 1)) * float(
            np.std(values, ddof=1)
        ) / math.sqrt(values.size)
        axes[1].errorbar(
            index,
            float(np.mean(values)),
            yerr=half_width,
            fmt="o",
            color=color,
            capsize=5,
            markersize=7,
            linewidth=2.0,
        )
    axes[1].axhline(
        float(power_budget_w), color="#C00000", linestyle="--", linewidth=1.2
    )
    axes[1].text(
        0.98,
        float(power_budget_w),
        "840 W 预算",
        transform=axes[1].get_yaxis_transform(),
        va="center",
        ha="right",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
    )
    axes[1].set_xticks([1, 2, 3], POLICY_NAMES)
    axes[1].set_ylabel("系统平均功率 (W)")
    axes[1].set_title("(b) 功率门槛：不通过")
    axes[1].grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    fig.suptitle("负载切换门禁：时延收益与功率可行性", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    target = output_dir / "gate_summary"
    fig.savefig(target.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_flow_heatmaps(output_dir, mean_phase_flows):
    import matplotlib.pyplot as plt

    configure_chinese_font()
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 9.0))
    vmax = max(
        float(np.max(matrix))
        for policy_flows in mean_phase_flows.values()
        for matrix in policy_flows
    )
    image = None
    for row, policy_name in enumerate(POLICY_NAMES):
        for column in range(3):
            matrix = np.asarray(mean_phase_flows[policy_name][column])
            axis = axes[row, column]
            image = axis.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=vmax)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    axis.text(
                        j,
                        i,
                        f"{matrix[i, j]:.1f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="black",
                    )
            axis.set_xticks(range(matrix.shape[1]), ["U1", "U2", "U3"])
            axis.set_yticks(range(matrix.shape[0]), ["U1", "U2", "U3"])
            if row == 0:
                axis.set_title(f"负载阶段 {column + 1}")
            if column == 0:
                axis.set_ylabel(POLICY_LABELS_ZH[policy_name])
    colorbar_axis = fig.add_axes([0.91, 0.13, 0.018, 0.72])
    fig.colorbar(image, cax=colorbar_axis, label="实际 U2U 流量 (Mbit)")
    fig.suptitle("各策略与负载阶段的平均链路流量矩阵", fontsize=15)
    fig.subplots_adjust(left=0.10, right=0.87, bottom=0.07, top=0.92, wspace=0.28, hspace=0.30)
    target = output_dir / "flow_matrices" / "mean_flow_heatmaps"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def summarize_and_write(output_dir, rows, traces_by_key, args, checkpoint_path, env):
    grouped = {
        policy: [row for row in rows if row["policy"] == policy]
        for policy in POLICY_NAMES
    }
    missing = [policy for policy in POLICY_NAMES if not grouped[policy]]
    if missing:
        raise RuntimeError(f"Formal gate summary is missing policies: {missing}")

    delay_p1 = paired_summary(
        grouped["P1"], grouped["P0"], "task_delay_weighted_ms"
    )
    delay_p2 = paired_summary(
        grouped["P2"], grouped["P0"], "task_delay_weighted_ms"
    )
    delay_p2_vs_p1 = paired_summary(
        grouped["P2"], grouped["P1"], "task_delay_weighted_ms"
    )
    transition_p1 = paired_summary(
        grouped["P1"],
        grouped["P0"],
        "transition_window_delay_weighted_ms",
    )
    transition_p2 = paired_summary(
        grouped["P2"],
        grouped["P0"],
        "transition_window_delay_weighted_ms",
    )
    power_p1 = paired_summary(
        grouped["P1"], grouped["P0"], "average_power_w"
    )
    power_p2 = paired_summary(
        grouped["P2"], grouped["P0"], "average_power_w"
    )

    paired_rows = []
    by_policy_seed = {
        (row["policy"], int(row["seed"])): row
        for row in rows
    }
    for seed in delay_p1["seeds"]:
        p0 = by_policy_seed[("P0", seed)]
        p1 = by_policy_seed[("P1", seed)]
        p2 = by_policy_seed[("P2", seed)]
        paired_rows.append(
            {
                "row_type": "seed",
                "seed": seed,
                "delta_p1_minus_p0_ms": (
                    float(p1["task_delay_weighted_ms"])
                    - float(p0["task_delay_weighted_ms"])
                ),
                "delta_p2_minus_p0_ms": (
                    float(p2["task_delay_weighted_ms"])
                    - float(p0["task_delay_weighted_ms"])
                ),
                "transition_delta_p1_minus_p0_ms": (
                    float(p1["transition_window_delay_weighted_ms"])
                    - float(p0["transition_window_delay_weighted_ms"])
                ),
                "transition_delta_p2_minus_p0_ms": (
                    float(p2["transition_window_delay_weighted_ms"])
                    - float(p0["transition_window_delay_weighted_ms"])
                ),
            }
        )
    paired_rows.append(
        {
            "row_type": "summary",
            "seed": "",
            "delta_p1_minus_p0_ms": delay_p1["mean"],
            "delta_p2_minus_p0_ms": delay_p2["mean"],
            "transition_delta_p1_minus_p0_ms": transition_p1["mean"],
            "transition_delta_p2_minus_p0_ms": transition_p2["mean"],
            "p1_ci95_low_ms": delay_p1["ci95_low"],
            "p1_ci95_high_ms": delay_p1["ci95_high"],
            "p1_cohen_dz": delay_p1["cohen_dz"],
            "p2_ci95_low_ms": delay_p2["ci95_low"],
            "p2_ci95_high_ms": delay_p2["ci95_high"],
            "p2_cohen_dz": delay_p2["cohen_dz"],
            "delta_p2_minus_p1_ms": delay_p2_vs_p1["mean"],
            "p2_vs_p1_ci95_low_ms": delay_p2_vs_p1["ci95_low"],
            "p2_vs_p1_ci95_high_ms": delay_p2_vs_p1["ci95_high"],
            "transition_p1_ci95_low_ms": transition_p1["ci95_low"],
            "transition_p1_ci95_high_ms": transition_p1["ci95_high"],
            "transition_p2_ci95_low_ms": transition_p2["ci95_low"],
            "transition_p2_ci95_high_ms": transition_p2["ci95_high"],
        }
    )
    paired_fields = sorted({key for row in paired_rows for key in row})
    write_csv_atomic(output_dir / "paired_deltas.csv", paired_rows, paired_fields)

    mean_power = {
        policy: float(np.mean([row["average_power_w"] for row in records]))
        for policy, records in grouped.items()
    }
    mean_u2g_time = {
        policy: float(np.mean([row["u2g_active_time_s"] for row in records]))
        for policy, records in grouped.items()
    }
    mean_u2u_time = {
        policy: float(np.mean([row["u2u_active_time_s"] for row in records]))
        for policy, records in grouped.items()
    }
    p1_better_seed_count = int(np.sum(delay_p1["values"] < 0.0))
    p1_relative_gain_pct = float(
        -delay_p1["mean"]
        / np.mean([row["task_delay_weighted_ms"] for row in grouped["P0"]])
        * 100.0
    )
    delay_pass = bool(
        delay_p1["mean"] < -float(args.practical_gain_ms)
        and delay_p1["ci95_high"] < 0.0
    )
    power_pass = bool(
        mean_power["P1"] <= float(args.power_budget_w)
        and mean_power["P2"] <= float(args.power_budget_w)
    )
    gate_pass = bool(delay_pass and power_pass)
    transition_significant = bool(
        transition_p1["mean"] < 0.0 and transition_p1["ci95_high"] < 0.0
    )
    p2_effective = bool(
        delay_p2["mean"] < -float(args.practical_gain_ms)
        and delay_p2["ci95_high"] < 0.0
    )
    if gate_pass:
        recommendation = "通过：冻结场景并进入正式训练。"
    elif delay_pass and not power_pass:
        recommendation = (
            "未通过：时延门槛成立，但 P1/P2 均违反 840 W 功率门槛。"
            "第 7 节的 A/B 均不是针对该功率失败的直接修复，故本轮不自动复测；"
            "应先冻结当前结论并另行预注册唯一调整。"
        )
    elif transition_significant:
        recommendation = "未通过；辅助结果支持第 7 节选项 A（切换周期 10→6）。"
    elif p2_effective:
        recommendation = "未通过；P2 有效而 P1 无效，正式训练需先定义因果负载预测状态。"
    else:
        recommendation = "未通过；P1/P2 均未显示足够收益，若复测则选择第 7 节选项 B。"

    mean_phase_flows = {}
    flow_rows = []
    for policy in POLICY_NAMES:
        policy_traces = [
            traces_by_key[(policy, int(row["seed"]))]
            for row in grouped[policy]
        ]
        matrices = np.mean(
            np.stack([trace["phase_flow_mbits"] for trace in policy_traces], axis=0),
            axis=0,
        )
        mean_phase_flows[policy] = matrices
        for phase_index, matrix in enumerate(matrices):
            matrix_path = (
                output_dir
                / "flow_matrices"
                / f"{policy}_phase{phase_index + 1}_mean_F_mn.csv"
            )
            matrix_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(
                matrix_path,
                matrix,
                delimiter=",",
                fmt="%.9f",
                header="to_UAV1,to_UAV2,to_UAV3",
                comments="from_UAV_rows,",
            )
            for source in range(matrix.shape[0]):
                for target in range(matrix.shape[1]):
                    flow_rows.append(
                        {
                            "policy": policy,
                            "phase_index": phase_index,
                            "source_uav": source,
                            "target_uav": target,
                            "mean_flow_mbits": float(matrix[source, target]),
                        }
                    )
    write_csv_atomic(
        output_dir / "flow_matrices" / "mean_F_mn_long.csv",
        flow_rows,
        ("policy", "phase_index", "source_uav", "target_uav", "mean_flow_mbits"),
    )

    representative_seed = int(args.base_seed)
    representative = {
        policy: traces_by_key[(policy, representative_seed)]
        for policy in POLICY_NAMES
    }
    plot_trajectories(output_dir, representative)
    plot_flow_heatmaps(output_dir, mean_phase_flows)
    plot_gate_summary(
        output_dir,
        grouped,
        delay_p1,
        delay_p2,
        args.power_budget_w,
    )

    verdict = {
        "checkpoint": str(checkpoint_path),
        "gate_pass": gate_pass,
        "delay_pass": delay_pass,
        "power_pass": power_pass,
        "recommendation": recommendation,
        "p1_minus_p0_delay_ms": delay_p1,
        "p2_minus_p0_delay_ms": delay_p2,
        "p2_minus_p1_delay_ms": delay_p2_vs_p1,
        "p1_minus_p0_transition_ms": transition_p1,
        "p2_minus_p0_transition_ms": transition_p2,
        "p1_minus_p0_power_w": power_p1,
        "p2_minus_p0_power_w": power_p2,
        "mean_power_w": mean_power,
        "mean_u2g_active_time_s": mean_u2g_time,
        "mean_u2u_active_time_s": mean_u2u_time,
        "p1_better_seed_count": p1_better_seed_count,
        "p1_relative_delay_gain_pct": p1_relative_gain_pct,
        "configuration": {
            "episodes": int(args.episodes),
            "base_seed": int(args.base_seed),
            "macro_steps": int(args.macro_steps),
            "switch_period": int(args.switch_period),
            "offset_m": float(args.offset_m),
            "anticipation_steps": int(args.anticipation_steps),
            "practical_gain_ms": float(args.practical_gain_ms),
            "power_budget_w": float(args.power_budget_w),
            "load_phases_mbps": LOAD_PHASES_MBPS,
            "macro_load_state": bool(env.macro_load_state),
            "macro_flow_state": bool(env.macro_flow_state),
            "macro_capacity_state": bool(env.macro_capacity_state),
            "macro_state_dim": int(env.get_macro_state()[1].size),
            "episode_progress_feature_present": False,
            "episode_progress_feature_audit": (
                "get_macro_state contains position, queue, arrival, topology, "
                "flow, expected-load, capacity, and energy-price features; "
                "it does not append macro index or episode progress."
            ),
        },
    }
    write_json_atomic(output_dir / "gate_verdict.json", verdict)

    crn_ok = True
    for seed in delay_p1["seeds"]:
        hashes = {
            by_policy_seed[(policy, seed)]["external_trace_hash"]
            for policy in POLICY_NAMES
        }
        crn_ok = crn_ok and len(hashes) == 1
    checks = [
        ("整段平均收益超过 5 ms", delay_p1["mean"] < -args.practical_gain_ms),
        ("整段配对 95% CI 排除 0", delay_p1["ci95_high"] < 0.0),
        ("P1 平均功率不超过 840 W", mean_power["P1"] <= args.power_budget_w),
        ("P2 平均功率不超过 840 W", mean_power["P2"] <= args.power_budget_w),
        ("三策略外生随机轨迹逐种子一致", crn_ok),
    ]
    lines = [
        "# 负载切换门禁判决",
        "",
        f"- **最终判决：{'通过' if gate_pass else '不通过'}**",
        f"- P1−P0 整段时延差：`{delay_p1['mean']:.3f} ms`，"
        f"95% CI `[{delay_p1['ci95_low']:.3f}, {delay_p1['ci95_high']:.3f}] ms`，"
        f"Cohen dz=`{delay_p1['cohen_dz']:.3f}`。",
        f"- P2−P0 整段时延差：`{delay_p2['mean']:.3f} ms`，"
        f"95% CI `[{delay_p2['ci95_low']:.3f}, {delay_p2['ci95_high']:.3f}] ms`。",
        f"- P2−P1 整段时延差：`{delay_p2_vs_p1['mean']:.3f} ms`，"
        f"95% CI `[{delay_p2_vs_p1['ci95_low']:.3f}, {delay_p2_vs_p1['ci95_high']:.3f}] ms`。",
        f"- P1−P0 过渡窗时延差：`{transition_p1['mean']:.3f} ms`，"
        f"95% CI `[{transition_p1['ci95_low']:.3f}, {transition_p1['ci95_high']:.3f}] ms`。",
        f"- 平均功率：P0 `{mean_power['P0']:.3f} W`，P1 `{mean_power['P1']:.3f} W`，"
        f"P2 `{mean_power['P2']:.3f} W`。",
        f"- P1−P0 功率差：`{power_p1['mean']:.3f} W`，"
        f"95% CI `[{power_p1['ci95_low']:.3f}, {power_p1['ci95_high']:.3f}] W`。",
        f"- P1 在 `{p1_better_seed_count}/{delay_p1['sample_count']}` 个配对种子中降低时延，"
        f"平均相对收益为 `{p1_relative_gain_pct:.2f}%`。",
        f"- 累计 U2G 活动时间：P0 `{mean_u2g_time['P0']:.3f} s`，"
        f"P1 `{mean_u2g_time['P1']:.3f} s`；累计 U2U 活动时间：P0 "
        f"`{mean_u2u_time['P0']:.3f} s`，P1 `{mean_u2u_time['P1']:.3f} s`。",
        f"- 后续动作：{recommendation}",
        "",
        "## 逐条核验",
        "",
    ]
    for label, passed in checks:
        lines.append(f"- [{'x' if passed else ' '}] {label}")
    lines.extend(
        [
            "",
            "## 口径说明",
            "",
            "- 主时延沿用四位置审计的任务级优先级加权端到端时延。",
            "- 过渡窗为宏观步 10--14 与 20--24，在窗口内完成任务的完整端到端时延。",
            "- P2 是非因果诊断上限，不进入主表，也不作为正式位置价值证据。",
            "- 本次只做冻结微观 SAC 的确定性重放，没有训练或参数调节。",
        ]
    )
    (output_dir / "gate_verdict.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8-sig",
    )
    return verdict


def main():
    args = parse_args()
    policies = tuple(
        item.strip().upper()
        for item in str(args.policies).split(",")
        if item.strip()
    )
    invalid = sorted(set(policies) - set(POLICY_NAMES))
    if invalid:
        raise ValueError(f"Unknown policies: {invalid}")
    if int(args.macro_steps) != 30:
        raise ValueError("The frozen gate requires exactly 30 macro steps.")
    if int(args.switch_period) != 10:
        raise ValueError("The frozen first-round gate requires switch_period=10.")
    if not math.isclose(float(args.offset_m), 40.0, abs_tol=1e-12):
        raise ValueError("The frozen P1/P2 offset is exactly 40 m.")

    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path, checkpoint_args, env, _actor, micro_policy, device = (
        load_checkpoint_components(args.checkpoint, args.device)
    )
    required_flags = {
        "macro_load_state": bool(env.macro_load_state),
        "macro_flow_state": bool(env.macro_flow_state),
        "macro_capacity_state": bool(env.macro_capacity_state),
    }
    if not all(required_flags.values()):
        raise RuntimeError(
            "The frozen gate requires all macro state flags enabled: "
            f"{required_flags}."
        )
    if env.macro_action_mode != "centroid_waypoint":
        raise RuntimeError(
            "The frozen gate requires centroid_waypoint macro actions."
        )
    if not math.isclose(float(env.power_budget_w), args.power_budget_w, abs_tol=1e-6):
        raise RuntimeError(
            f"Checkpoint power budget is {env.power_budget_w} W, expected "
            f"{args.power_budget_w} W."
        )
    if not math.isclose(float(checkpoint_args.load_scale), 1.15, abs_tol=1e-9):
        raise RuntimeError("The frozen gate requires load_scale=1.15.")
    if not math.isclose(float(checkpoint_args.cpu_ghz), 10.0, abs_tol=1e-9):
        raise RuntimeError("The frozen gate requires CPU=10 GHz.")
    if str(env.power_constraint_scope) != "system":
        raise RuntimeError("The frozen gate requires a system-average power budget.")

    metadata = {
        "status": "FROZEN",
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "policies": list(policies),
        "episodes": int(args.episodes),
        "seeds": [int(args.base_seed) + index for index in range(args.episodes)],
        "macro_steps": int(args.macro_steps),
        "switch_period": int(args.switch_period),
        "offset_m": float(args.offset_m),
        "anticipation_steps": int(args.anticipation_steps),
        "load_phases_mbps": LOAD_PHASES_MBPS,
        "macro_state_flags": required_flags,
        "training_performed": False,
    }
    write_json_atomic(output_dir / "gate_configuration.json", metadata)

    if args.summarize_only:
        rows = read_raw_rows(output_dir / "per_seed_raw.csv")
        traces_by_key = load_saved_traces(output_dir, rows)
        verdict = summarize_and_write(
            output_dir,
            rows,
            traces_by_key,
            args,
            checkpoint_path,
            env,
        )
        print(
            f"[VERDICT] {'PASS' if verdict['gate_pass'] else 'FAIL'}: "
            f"{verdict['recommendation']}"
        )
        print(f"[OUTPUT] {output_dir}")
        return

    rows = []
    traces_by_key = {}
    for episode_index in range(int(args.episodes)):
        seed = int(args.base_seed) + episode_index
        seed_hashes = {}
        for policy_name in policies:
            print(
                f"[gate {episode_index + 1:02d}/{args.episodes:02d}] "
                f"seed={seed} policy={policy_name}",
                flush=True,
            )
            result, trace = run_gate_episode(
                env,
                checkpoint_args,
                micro_policy,
                policy_name,
                seed,
                macro_steps=args.macro_steps,
                switch_period=args.switch_period,
                offset_m=args.offset_m,
                anticipation_steps=args.anticipation_steps,
            )
            rows.append(result)
            traces_by_key[(policy_name, seed)] = trace
            seed_hashes[policy_name] = result["external_trace_hash"]
            save_macro_trace(
                output_dir / "macro_traces" / f"{policy_name}_seed{seed}.csv",
                trace["macro_records"],
                int(checkpoint_args.num_agents),
            )
            (output_dir / "trajectories").mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output_dir / "trajectories" / f"{policy_name}_seed{seed}.npz",
                uav_trajectory_m=trace["uav_trajectory_m"],
                centroid_trajectory_m=trace["centroid_trajectory_m"],
                trajectory_phases=trace["trajectory_phases"],
                phase_flow_mbits=trace["phase_flow_mbits"],
            )
            write_csv_atomic(output_dir / "per_seed_raw.csv", rows, RAW_FIELDNAMES)
            print(
                f"  delay={result['task_delay_weighted_ms']:.3f} ms "
                f"transition={result['transition_window_delay_weighted_ms']:.3f} ms "
                f"power={result['average_power_w']:.3f} W "
                f"elapsed={result['elapsed_s']:.1f}s",
                flush=True,
            )
        if len(set(seed_hashes.values())) != 1:
            raise RuntimeError(
                f"CRN audit failed for seed {seed}: {seed_hashes}."
            )

    if set(policies) == set(POLICY_NAMES) and int(args.episodes) == 20:
        verdict = summarize_and_write(
            output_dir,
            rows,
            traces_by_key,
            args,
            checkpoint_path,
            env,
        )
        print(
            f"[VERDICT] {'PASS' if verdict['gate_pass'] else 'FAIL'}: "
            f"{verdict['recommendation']}",
            flush=True,
        )
    else:
        print(
            "[DONE] Smoke/partial run completed; the frozen formal verdict "
            "requires P0/P1/P2 and exactly 20 CRN seeds."
        )
    print(f"[OUTPUT] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
