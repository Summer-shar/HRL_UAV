"""Audit the physical chain behind the v44 learned-trajectory gain.

The audit is observational: it wraps task lifecycle methods on a fresh
evaluation environment without changing routing, service, or policy actions.
Completed-task delay is decomposed along each parent task's critical chunk.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_joint_hrl_v1_matched import load_policies, make_eval_env, mean_ci95  # noqa: E402
from compare_common import evaluate_policy  # noqa: E402
from search_position_oracle import make_centroid_offset_policy  # noqa: E402


DEFAULT_CHECKPOINT = (
    ROOT
    / "results"
    / "tcom_v44_joint_cooperation_finetune"
    / "formal"
    / "seed1"
    / "joint_td3_sac"
    / "checkpoints"
    / "final.pt"
)
DEFAULT_OUTPUT = ROOT / "result_archive" / "v44_physical_causal_chain"
POLICIES = ("centroid", "learned")
COMPONENTS = (
    "upload_tx_s",
    "upload_queue_s",
    "routing_wait_s",
    "u2u_tx_s",
    "u2u_queue_s",
    "compute_exec_s",
    "compute_queue_s",
    "deployment_pause_s",
    "residual_s",
)


def parse_seed_spec(value: str) -> list[int]:
    seeds: list[int] = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = (int(item) for item in token.split("-", 1))
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(token))
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class PhysicalAudit:
    """Attach read-only lifecycle and link accounting to one environment."""

    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.env = wrapper.env
        self.completed_records: list[dict] = []
        self.parents: dict[int, object] = {}
        self.chunks: dict[int, list[object]] = {}
        self._reset_link_totals()
        self._patch()

    def _reset_link_totals(self) -> None:
        n = int(self.env.N)
        self.u2g_service_mbits = 0.0
        self.u2g_active_s = 0.0
        self.u2g_service_distance_horizontal_m = 0.0
        self.u2g_service_distance_3d_m = 0.0
        self.u2g_service_rate_mbps = 0.0
        self.u2g_all_distance_horizontal_m = 0.0
        self.u2g_all_distance_3d_m = 0.0
        self.u2g_all_count = 0
        self.u2u_service_mbits = np.zeros((n, n), dtype=np.float64)
        self.u2u_active_s = np.zeros((n, n), dtype=np.float64)
        self.u2u_service_distance_m = np.zeros((n, n), dtype=np.float64)
        self.u2u_service_rate_mbps = np.zeros((n, n), dtype=np.float64)
        self.u2u_service_spectral_efficiency = np.zeros((n, n), dtype=np.float64)
        self.u2u_service_bandwidth_mhz = np.zeros((n, n), dtype=np.float64)
        self.u2u_pair_distance_sum_m = np.zeros((n, n), dtype=np.float64)
        self.u2u_pair_distance_count = 0
        self.path_length_m = np.zeros(n, dtype=np.float64)
        self.move_event_count = np.zeros(n, dtype=np.int64)

    def _patch(self) -> None:
        env = self.env

        original_init = env._init_task_tracker

        def init_tracker(*args, **kwargs):
            result = original_init(*args, **kwargs)
            self.completed_records.clear()
            self.parents.clear()
            self.chunks.clear()
            self._reset_link_totals()
            return result

        env._init_task_tracker = init_tracker

        original_create_parent = env._create_parent_task

        def create_parent(*args, **kwargs):
            parent = original_create_parent(*args, **kwargs)
            parent._audit_upload_active_s = 0.0
            parent._audit_upload_pause_s = 0.0
            self.parents[int(parent.task_id)] = parent
            self.chunks[int(parent.task_id)] = []
            return parent

        env._create_parent_task = create_parent

        original_create_chunk = env._create_chunk

        def create_chunk(*args, **kwargs):
            chunk = original_create_chunk(*args, **kwargs)
            chunk._audit_created_time = float(env._current_time())
            chunk._audit_compute_enter_time = None
            chunk._audit_finish_time = None
            chunk._audit_u2u_active_s = 0.0
            chunk._audit_compute_active_s = 0.0
            chunk._audit_u2u_pause_s = 0.0
            chunk._audit_compute_pause_s = 0.0
            self.chunks.setdefault(int(chunk.parent_task_id), []).append(chunk)
            return chunk

        env._create_chunk = create_chunk

        original_enqueue_compute = env._enqueue_compute_chunk

        def enqueue_compute(chunk):
            old_stage = str(chunk.current_stage)
            if chunk._audit_compute_enter_time is None:
                enter_time = float(env._current_time())
                if old_stage == "u2u_queue":
                    enter_time += float(env.micro_dt)
                chunk._audit_compute_enter_time = enter_time
            return original_enqueue_compute(chunk)

        env._enqueue_compute_chunk = enqueue_compute

        original_consume_compute = env._consume_compute_queue

        def consume_compute(
            queue,
            capacity_mbits,
            finish_time,
            service_by_priority=None,
            service_by_type=None,
        ):
            before = [(chunk, float(chunk.remaining_mbits)) for chunk in list(queue)]
            result = original_consume_compute(
                queue,
                capacity_mbits,
                finish_time,
                service_by_priority=service_by_priority,
                service_by_type=service_by_type,
            )
            rate_mbps = float(capacity_mbits) / max(float(env.micro_dt), 1e-12)
            if rate_mbps > 1e-12:
                for chunk, remaining_before in before:
                    remaining_after = max(float(chunk.remaining_mbits), 0.0)
                    served = max(remaining_before - remaining_after, 0.0)
                    chunk._audit_compute_active_s += served / rate_mbps
            return result

        env._consume_compute_queue = consume_compute

        original_consume_tx = env._consume_tx_queue

        def consume_tx(queue, capacity_mbits):
            before = [
                (chunk, float(chunk.remaining_mbits), str(chunk.current_stage))
                for chunk in list(queue)
            ]
            result = original_consume_tx(queue, capacity_mbits)
            spectral = env.compute_u2u_spectral_efficiency()
            bandwidth = np.asarray(
                env.last_u2u_link_bandwidth_matrix_hz,
                dtype=np.float64,
            )
            rate_mbps = spectral * bandwidth / 1e6
            for chunk, remaining_before, stage_before in before:
                if stage_before != "u2u_queue":
                    continue
                if str(chunk.current_stage) != "u2u_queue":
                    served = remaining_before
                else:
                    served = max(remaining_before - float(chunk.remaining_mbits), 0.0)
                link_rate = float(rate_mbps[chunk.source_uav, chunk.target_uav])
                if link_rate > 1e-12:
                    chunk._audit_u2u_active_s += served / link_rate
            return result

        env._consume_tx_queue = consume_tx

        original_serve_uplink = env._serve_uplink_task_queues

        def serve_uplink():
            before = []
            for user_id, queue in enumerate(env.uplink_task_queues):
                for task_id in list(queue):
                    parent = env.parent_tasks.get(task_id)
                    if parent is not None:
                        before.append(
                            (parent, int(user_id), float(parent.remaining_upload_mbits))
                        )
            result = original_serve_uplink()
            rates = np.asarray(result["effective_rate_bps_per_user"], dtype=np.float64)
            for parent, user_id, remaining_before in before:
                remaining_after = max(float(parent.remaining_upload_mbits), 0.0)
                served = max(remaining_before - remaining_after, 0.0)
                rate_bps = float(rates[user_id])
                if rate_bps > 1e-12:
                    parent._audit_upload_active_s += served * 1e6 / rate_bps
            return result

        env._serve_uplink_task_queues = serve_uplink

        original_mark_chunk = env._mark_chunk_completed

        def mark_chunk(chunk, finish_time):
            chunk._audit_finish_time = float(finish_time)
            return original_mark_chunk(chunk, finish_time)

        env._mark_chunk_completed = mark_chunk

        original_record_parent = env._record_completed_parent_task

        def record_parent(parent):
            self._record_parent_components(parent)
            return original_record_parent(parent)

        env._record_completed_parent_task = record_parent

        original_pause = env._advance_macro_transition_step

        def advance_pause():
            dt = float(env.micro_dt)
            for parent in list(env.parent_tasks.values()):
                if parent.finish_time is not None:
                    continue
                if parent.upload_finish_time is None:
                    parent._audit_upload_pause_s += dt
            for chunk_list in self.chunks.values():
                for chunk in chunk_list:
                    stage = str(chunk.current_stage)
                    if stage == "u2u_queue":
                        chunk._audit_u2u_pause_s += dt
                    elif stage == "compute_queue":
                        chunk._audit_compute_pause_s += dt
            return original_pause()

        env._advance_macro_transition_step = advance_pause

        original_micro_step = env.micro_step

        def micro_step(*args, **kwargs):
            users_before = np.asarray(env.user_coords, dtype=np.float64).copy()
            uavs_before = np.asarray(env.uav_coords, dtype=np.float64).copy()
            raw_state, info = original_micro_step(*args, **kwargs)
            self._record_link_step(users_before, uavs_before, info)
            return raw_state, info

        env.micro_step = micro_step

        original_macro_step = env.macro_step

        def macro_step(*args, **kwargs):
            before = np.asarray(env.uav_coords, dtype=np.float64).copy()
            result = original_macro_step(*args, **kwargs)
            projected = np.asarray(result[2]["projected_uav_coords"], dtype=np.float64)
            distances = np.linalg.norm(projected - before, axis=1)
            self.path_length_m += distances
            self.move_event_count += (distances > 1e-8).astype(np.int64)
            return result

        env.macro_step = macro_step

    def _record_parent_components(self, parent) -> None:
        finish_time = float(parent.finish_time)
        birth_time = float(parent.birth_time)
        upload_finish = float(parent.upload_finish_time or birth_time)
        chunks = self.chunks.get(int(parent.task_id), [])
        completed = [
            chunk
            for chunk in chunks
            if getattr(chunk, "_audit_finish_time", None) is not None
        ]
        if not completed:
            return
        critical = max(completed, key=lambda chunk: float(chunk._audit_finish_time))
        created = float(critical._audit_created_time)
        compute_enter = float(critical._audit_compute_enter_time or created)
        upload_total = max(upload_finish - birth_time, 0.0)
        upload_active = float(parent._audit_upload_active_s)
        upload_pause = min(float(parent._audit_upload_pause_s), upload_total)
        upload_queue = max(upload_total - upload_active - upload_pause, 0.0)
        routing_wait = max(created - upload_finish, 0.0)
        is_remote = int(critical.target_uav) != int(critical.source_uav)
        u2u_total = max(compute_enter - created, 0.0) if is_remote else 0.0
        u2u_active = float(critical._audit_u2u_active_s) if is_remote else 0.0
        u2u_pause = min(float(critical._audit_u2u_pause_s), u2u_total)
        u2u_queue = max(u2u_total - u2u_active - u2u_pause, 0.0)
        compute_total = max(finish_time - compute_enter, 0.0)
        compute_active = float(critical._audit_compute_active_s)
        compute_pause = min(float(critical._audit_compute_pause_s), compute_total)
        compute_queue = max(compute_total - compute_active - compute_pause, 0.0)
        deployment_pause = upload_pause + u2u_pause + compute_pause
        components = {
            "upload_tx_s": upload_active,
            "upload_queue_s": upload_queue,
            "routing_wait_s": routing_wait,
            "u2u_tx_s": u2u_active,
            "u2u_queue_s": u2u_queue,
            "compute_exec_s": compute_active,
            "compute_queue_s": compute_queue,
            "deployment_pause_s": deployment_pause,
        }
        total_s = max(finish_time - birth_time, 0.0)
        accounted = float(sum(components.values()))
        components["residual_s"] = total_s - accounted
        self.completed_records.append(
            {
                "task_id": int(parent.task_id),
                "priority": str(parent.priority),
                "task_type": str(parent.task_type),
                "source_uav": int(parent.source_uav),
                "critical_target_uav": int(critical.target_uav),
                "critical_remote": int(is_remote),
                "total_s": total_s,
                **components,
            }
        )

    def _record_link_step(self, users_before, uavs_before, info) -> None:
        if bool(info.get("service_paused", False)):
            return
        associations = np.asarray(
            info.get("user_associations", self.env.user_allocations),
            dtype=np.int64,
        )
        horizontal = np.linalg.norm(users_before - uavs_before[associations], axis=1)
        distance_3d = np.sqrt(horizontal**2 + float(self.env.H) ** 2)
        self.u2g_all_distance_horizontal_m += float(np.sum(horizontal))
        self.u2g_all_distance_3d_m += float(np.sum(distance_3d))
        self.u2g_all_count += int(horizontal.size)

        service = np.asarray(
            info.get("uplink_service_mbits_per_user", np.zeros(self.env.M)),
            dtype=np.float64,
        )
        active = np.asarray(
            info.get("uplink_active_time_s_per_user", np.zeros(self.env.M)),
            dtype=np.float64,
        )
        rates = np.asarray(
            info.get("uplink_rate_bps_per_user", np.zeros(self.env.M)),
            dtype=np.float64,
        ) / 1e6
        self.u2g_service_mbits += float(np.sum(service))
        self.u2g_active_s += float(np.sum(active))
        self.u2g_service_distance_horizontal_m += float(np.sum(service * horizontal))
        self.u2g_service_distance_3d_m += float(np.sum(service * distance_3d))
        self.u2g_service_rate_mbps += float(np.sum(service * rates))

        tx_service = np.asarray(
            info.get("tx_service_matrix_mbits", np.zeros((self.env.N, self.env.N))),
            dtype=np.float64,
        )
        tx_active = np.asarray(
            info.get("u2u_active_time_s", np.zeros_like(tx_service)),
            dtype=np.float64,
        )
        tx_rates = np.asarray(
            info.get("u2u_rate_bps", np.zeros_like(tx_service)),
            dtype=np.float64,
        ) / 1e6
        tx_bandwidth_mhz = np.asarray(
            info.get("u2u_link_bandwidth_hz", np.zeros_like(tx_service)),
            dtype=np.float64,
        ) / 1e6
        spectral_efficiency = np.divide(
            tx_rates,
            tx_bandwidth_mhz,
            out=np.zeros_like(tx_rates),
            where=tx_bandwidth_mhz > 1e-12,
        )
        coords = np.asarray(self.env.uav_coords, dtype=np.float64)
        distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
        self.u2u_service_mbits += tx_service
        self.u2u_active_s += tx_active
        self.u2u_service_distance_m += tx_service * distances
        self.u2u_service_rate_mbps += tx_service * tx_rates
        self.u2u_service_spectral_efficiency += tx_service * spectral_efficiency
        self.u2u_service_bandwidth_mhz += tx_service * tx_bandwidth_mhz
        pair_mask = ~np.eye(self.env.N, dtype=bool)
        self.u2u_pair_distance_sum_m[pair_mask] += distances[pair_mask]
        self.u2u_pair_distance_count += 1

    def _weighted_component_means(self, records: list[dict]) -> dict[str, float]:
        if not records:
            return {component: 0.0 for component in COMPONENTS}
        weights = np.asarray(
            [self.env.w_D_high if row["priority"] == "high" else self.env.w_D_low for row in records],
            dtype=np.float64,
        )
        denominator = max(float(np.sum(weights)), 1e-12)
        return {
            component: float(
                np.sum(weights * np.asarray([row[component] for row in records]))
                / denominator
            )
            for component in COMPONENTS
        }

    def summary(self) -> dict[str, float]:
        weighted = self._weighted_component_means(self.completed_records)
        high = [row for row in self.completed_records if row["priority"] == "high"]
        low = [row for row in self.completed_records if row["priority"] == "low"]

        def ordinary(records, field):
            return float(np.mean([row[field] for row in records])) if records else 0.0

        u2g_volume = max(self.u2g_service_mbits, 1e-12)
        u2g_active = max(self.u2g_active_s, 1e-12)
        u2u_volume = max(float(np.sum(self.u2u_service_mbits)), 1e-12)
        u2u_active = max(float(np.sum(self.u2u_active_s)), 1e-12)
        result = {
            "stage_completed_count": len(self.completed_records),
            "stage_weighted_total_ms": 1000.0 * float(
                sum(weighted[component] for component in COMPONENTS)
            ),
            "stage_high_total_ms": 1000.0 * ordinary(high, "total_s"),
            "stage_low_total_ms": 1000.0 * ordinary(low, "total_s"),
            "critical_remote_fraction": float(
                np.mean([row["critical_remote"] for row in self.completed_records])
            ) if self.completed_records else 0.0,
            "u2g_geometric_horizontal_m": self.u2g_all_distance_horizontal_m
            / max(self.u2g_all_count, 1),
            "u2g_geometric_3d_m": self.u2g_all_distance_3d_m
            / max(self.u2g_all_count, 1),
            "u2g_flow_weighted_horizontal_m": self.u2g_service_distance_horizontal_m
            / u2g_volume,
            "u2g_flow_weighted_3d_m": self.u2g_service_distance_3d_m / u2g_volume,
            "u2g_flow_weighted_rate_mbps": self.u2g_service_rate_mbps / u2g_volume,
            "u2g_effective_rate_mbps": self.u2g_service_mbits / u2g_active,
            "u2g_service_mbits": self.u2g_service_mbits,
            "u2g_active_link_time_s": self.u2g_active_s,
            "u2u_flow_weighted_distance_m": float(np.sum(self.u2u_service_distance_m))
            / u2u_volume,
            "u2u_flow_weighted_rate_mbps": float(np.sum(self.u2u_service_rate_mbps))
            / u2u_volume,
            "u2u_flow_weighted_spectral_efficiency": float(
                np.sum(self.u2u_service_spectral_efficiency)
            ) / u2u_volume,
            "u2u_flow_weighted_bandwidth_mhz": float(
                np.sum(self.u2u_service_bandwidth_mhz)
            ) / u2u_volume,
            "u2u_effective_rate_mbps": float(np.sum(self.u2u_service_mbits))
            / u2u_active,
            "u2u_service_mbits": float(np.sum(self.u2u_service_mbits)),
            "u2u_active_link_time_s": float(np.sum(self.u2u_active_s)),
            "path_length_total_m": float(np.sum(self.path_length_m)),
            **{f"path_length_uav{i + 1}_m": float(value) for i, value in enumerate(self.path_length_m)},
            **{f"move_events_uav{i + 1}": int(value) for i, value in enumerate(self.move_event_count)},
        }
        for component, value in weighted.items():
            result[f"stage_weighted_{component[:-2]}_ms"] = 1000.0 * value
        for priority, records in (("high", high), ("low", low)):
            for component in COMPONENTS:
                result[f"stage_{priority}_{component[:-2]}_ms"] = 1000.0 * ordinary(
                    records, component
                )
        for i in range(self.env.N):
            for j in range(self.env.N):
                if i == j:
                    continue
                volume = float(self.u2u_service_mbits[i, j])
                result[f"u2u_flow_{i + 1}_{j + 1}_mbits"] = volume
                result[f"u2u_distance_{i + 1}_{j + 1}_m"] = (
                    float(self.u2u_service_distance_m[i, j]) / max(volume, 1e-12)
                )
                result[f"u2u_rate_{i + 1}_{j + 1}_mbps"] = (
                    float(self.u2u_service_rate_mbps[i, j]) / max(volume, 1e-12)
                )
                result[f"u2u_eta_{i + 1}_{j + 1}"] = (
                    float(self.u2u_service_spectral_efficiency[i, j])
                    / max(volume, 1e-12)
                )
                result[f"u2u_bandwidth_{i + 1}_{j + 1}_mhz"] = (
                    float(self.u2u_service_bandwidth_mhz[i, j])
                    / max(volume, 1e-12)
                )
                result[f"u2u_active_{i + 1}_{j + 1}_s"] = float(
                    self.u2u_active_s[i, j]
                )
        return result


def evaluate_arm(checkpoint_args, macro_policy, micro_policy, seed: int) -> dict:
    env = make_eval_env(checkpoint_args, record_trajectory=False)
    audit = PhysicalAudit(env)
    eval_args = argparse.Namespace(**vars(checkpoint_args))
    eval_args.eval_episodes = 1
    eval_args.eval_base_seed = int(seed)
    metrics = evaluate_policy(env, eval_args, macro_policy, micro_policy)
    audit_metrics = audit.summary()
    return {
        "task_delay_weighted_ms": float(metrics["task_delay_weighted_ms"]),
        "task_delay_high_ms": float(metrics["task_delay_high_ms"]),
        "task_delay_low_ms": float(metrics["task_delay_low_ms"]),
        "queue_mbits": float(metrics["q_high"] + metrics["q_low"]),
        "queue_high_mbits": float(metrics["q_high"]),
        "queue_low_mbits": float(metrics["q_low"]),
        "average_power_w": float(metrics["average_power_w"]),
        "pressure_imbalance": float(metrics["pressure_imbalance"]),
        **audit_metrics,
    }


def summarize(rows: list[dict], seeds: list[int]) -> dict:
    fields = [key for key in rows[0] if key not in {"seed", "policy"}]
    by_policy = {
        policy: {
            int(row["seed"]): row for row in rows if row["policy"] == policy
        }
        for policy in POLICIES
    }
    means = {
        policy: {
            field: float(np.mean([float(by_policy[policy][seed][field]) for seed in seeds]))
            for field in fields
        }
        for policy in POLICIES
    }
    paired = {
        field: mean_ci95(
            [
                float(by_policy["centroid"][seed][field])
                - float(by_policy["learned"][seed][field])
                for seed in seeds
            ]
        )
        for field in fields
    }
    component_delta_sum = sum(
        paired[f"stage_weighted_{component[:-2]}_ms"]["mean"]
        for component in COMPONENTS
    )
    return {
        "seeds": seeds,
        "difference_definition": "centroid minus learned; positive means learned is lower",
        "means": means,
        "paired_differences": paired,
        "component_delta_sum_ms": float(component_delta_sum),
        "measured_e2e_delta_ms": float(paired["task_delay_weighted_ms"]["mean"]),
        "decomposition_closure_error_ms": float(
            component_delta_sum - paired["task_delay_weighted_ms"]["mean"]
        ),
        "scope_note": (
            "Stage components use the critical chunk of each completed parent task. "
            "They add arithmetically to completed-task E2E delay, but queueing makes "
            "them coupled rather than independent causal effects."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="8000-8019")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    seeds = parse_seed_spec(args.seeds)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_args, learned_policy, micro_policy = load_policies(
        args.checkpoint.resolve(), args.device
    )
    policies = {
        "centroid": make_centroid_offset_policy(np.zeros((3, 2))),
        "learned": learned_policy,
    }
    rows: list[dict] = []
    task_rows: list[dict] = []
    for seed in seeds:
        for policy_name in POLICIES:
            env = make_eval_env(checkpoint_args, record_trajectory=False)
            audit = PhysicalAudit(env)
            eval_args = argparse.Namespace(**vars(checkpoint_args))
            eval_args.eval_episodes = 1
            eval_args.eval_base_seed = int(seed)
            metrics = evaluate_policy(
                env,
                eval_args,
                policies[policy_name],
                micro_policy,
            )
            result = {
                "task_delay_weighted_ms": float(metrics["task_delay_weighted_ms"]),
                "task_delay_high_ms": float(metrics["task_delay_high_ms"]),
                "task_delay_low_ms": float(metrics["task_delay_low_ms"]),
                "queue_mbits": float(metrics["q_high"] + metrics["q_low"]),
                "queue_high_mbits": float(metrics["q_high"]),
                "queue_low_mbits": float(metrics["q_low"]),
                "average_power_w": float(metrics["average_power_w"]),
                "pressure_imbalance": float(metrics["pressure_imbalance"]),
                **audit.summary(),
            }
            rows.append({"seed": seed, "policy": policy_name, **result})
            for task in audit.completed_records:
                task_rows.append({"seed": seed, "policy": policy_name, **task})
            print(
                f"seed={seed} policy={policy_name} "
                f"delay={result['task_delay_weighted_ms']:.2f}ms "
                f"closure={result['stage_weighted_total_ms'] - result['task_delay_weighted_ms']:+.4f}ms",
                flush=True,
            )
    write_csv(output / "per_seed_raw.csv", rows)
    write_csv(output / "per_task_stage_components.csv", task_rows)
    report = summarize(rows, seeds)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
