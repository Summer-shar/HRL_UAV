"""Causal aggregated observations and direct routing for the clean HRL mainline."""

import numpy as np

from .env import UAV_MEC_Env
from .geometry import cooperation_aligned_basis
from .load_switching import (
    build_cluster_share_schedule,
    build_load_switch_schedule,
    parse_cluster_share_phases,
)
from .routing import project_rows_to_simplex


SHARED_FEATURES = (
    "compute_queue", "transfer_queue", "upload_queue", "unfinished_tasks",
    "oldest_age", "unfinished_chunks", "assigned_cpu_seconds", "current_arrivals",
    "current_load", "cpu_capacity", "u2u_spectral_efficiency", "u2g_mean_rate",
    "uav_coordinates", "centroids", "centroid_offset", "centroid_velocity",
    "phase_age", "macro_clock",
)
MICRO_FEATURES = (
    "compute_queue", "transfer_queue", "upload_queue", "unfinished_tasks",
    "oldest_age", "current_arrivals", "current_load", "cpu_capacity",
    "u2u_spectral_efficiency", "u2g_mean_rate", "phase_age", "macro_clock",
)
MACRO_FEATURES = (
    "compute_queue", "inbound_transfer_queue", "upload_queue", "unfinished_tasks",
    "oldest_age", "current_load", "cpu_capacity", "u2u_spectral_efficiency",
    "u2g_mean_rate", "uav_coordinates", "centroid_offset", "centroid_velocity",
    "phase_age",
)
HOTSPOT_MACRO_FEATURES = (
    "compute_queue", "inbound_transfer_queue", "unfinished_tasks", "oldest_age",
    "u2u_pair_quality", "uav_coordinates", "centroid_velocity", "phase_age",
    "cluster_offset", "cluster_offered_load", "cluster_upload",
)


def compress(values, scale):
    """Monotone, non-saturating encoding; scale has the units of values."""
    values = np.asarray(values, dtype=np.float64)
    return np.sign(values) * np.log1p(np.abs(values) / scale)


def holding_energy_cost(holding, energy, reward_config):
    delay = float(holding) / reward_config["holding_norm_task_s"]
    energy = float(energy) / reward_config["energy_norm_j"]
    total = reward_config["delay_weight"] * delay + reward_config["energy_weight"] * energy
    if not np.isfinite(total):
        raise FloatingPointError("Non-finite physical objective")
    return total


class QueueMDP:
    """900 s sampling streams, without a queue reset at 300 s boundaries.

    Observations are causal aggregates, not a claim of a fully observed MDP.
    Raw SAC scores are the policy actions; simplex projection is deterministic
    environment dynamics. Entropy uses raw scores; the Critic can encode their
    executed routes, as selected by the training configuration.
    """

    def __init__(self, config, *, environment_factory=UAV_MEC_Env):
        self.config = config
        self.env = environment_factory(**config["environment"])
        self.env.F_max_array[:] = config["cpu_ghz"] * 1e9
        self.N = self.env.N
        self.micro_action_dim = 2 * self.N * self.N
        self.macro_action_dim = 2 * self.N
        self.reward_config = config["reward"]
        self.observe_routing_ready = config.get("observation", {}).get("routing_ready", False)
        self.observation_layout = config.get("observation", {}).get("layout", "shared_v1")
        if self.observation_layout not in ("shared_v1", "split_v1"):
            raise ValueError(f"Unknown observation layout: {self.observation_layout}")
        self.cluster_count = int(self.env.cluster_count)
        self.observe_clusters = self.cluster_count > 1
        self.cluster_schedule = None
        # Cluster aggregates extend only the macro policy input; the micro
        # observation layout stays compatible; its values can change.
        self.macro_features = MACRO_FEATURES
        if self.observe_clusters:
            self.macro_features = MACRO_FEATURES + (
                "cluster_load_share",
                "cluster_backlog",
            )
        macro_layout = config.get("macro_observation_layout", "legacy")
        if macro_layout not in ("legacy", "hotspot_upload_v2"):
            raise ValueError(f"Unknown macro observation layout: {macro_layout}")
        if macro_layout == "hotspot_upload_v2":
            if not self.observe_clusters or self.observation_layout != "split_v1":
                raise ValueError("hotspot_upload_v2 requires clusters and split_v1")
            self.macro_features = HOTSPOT_MACRO_FEATURES
        self.phase_age = 0
        self.macro_index = 0
        self.centroid_velocity = np.zeros((self.N, 2))
        self.propulsion_share = 0.0
        self.propulsion_delta_share = 0.0
        self.total_power_energy = 0.0
        self.cost_sum = 0.0
        self.integration_steps = 0
        self.micro_feature_slices = {}
        self.macro_feature_slices = {}
        self.feature_slices = self.micro_feature_slices

    def reset(self, seed, macro_count):
        self.env.reset(seed=int(seed))
        self.schedule = build_load_switch_schedule(
            "cyclic_random", self.config["phase_loads"], macro_count + 1, seed,
            *self.config["phase_dwell"],
        )
        cluster_phases = self.config.get("phase_cluster_shares")
        if cluster_phases is not None:
            if not self.observe_clusters:
                raise ValueError(
                    "phase_cluster_shares requires environment cluster_count > 1"
                )
            shares = parse_cluster_share_phases(
                cluster_phases, self.N, self.cluster_count
            )
            dwell = self.config.get("phase_cluster_dwell", self.config["phase_dwell"])
            self.cluster_schedule = build_cluster_share_schedule(
                "cyclic_random", shares, macro_count + 1, seed, *dwell
            )
            self.env.set_cluster_target_shares(self.cluster_schedule.shares(0))
        else:
            self.cluster_schedule = None
        self.phase_age = 0
        self.macro_index = 0
        self.centroid_velocity.fill(0.0)
        self.previous_centroid = self.env.weighted_centroids.copy()
        self.env.set_zone_target_loads(self.schedule.loads(0))
        self.propulsion_share = self.propulsion_delta_share = 0.0
        self.total_power_energy = self.cost_sum = 0.0
        self.integration_steps = 0
        return self.observation()

    def advance_macro_clock(self):
        self.macro_index += 1
        self.centroid_velocity = (
            self.env.weighted_centroids - self.previous_centroid
        ) / self.env.macro_dt
        self.previous_centroid = self.env.weighted_centroids.copy()
        if self.schedule.switched(self.macro_index):
            self.phase_age = 0
            self.env.set_zone_target_loads(self.schedule.loads(self.macro_index))
        else:
            self.phase_age += 1
        if self.cluster_schedule is not None and self.cluster_schedule.switched(
            self.macro_index
        ):
            self.env.set_cluster_target_shares(
                self.cluster_schedule.shares(self.macro_index)
            )

    def task_aggregates(self):
        counts = np.zeros((2, self.N))
        oldest = np.zeros_like(counts)
        upload = np.zeros_like(counts)
        incomplete_chunks = np.zeros_like(counts)
        now = self.env._current_time()
        for parent in self.env.parent_tasks.values():
            if parent.finish_time is not None:
                continue
            priority = 0 if parent.priority == "high" else 1
            index = (priority, parent.source_uav)
            counts[index] += 1
            oldest[index] = max(oldest[index], now - parent.birth_time)
            upload[index] += parent.remaining_upload_mbits
            incomplete_chunks[index] += max(parent.total_chunks - parent.completed_chunks, 0)
        return counts, oldest, upload, incomplete_chunks

    def routing_ready_aggregates(self):
        """Read only tasks already uploaded and awaiting the next routing step."""
        work = np.zeros((2, self.N))
        counts = np.zeros_like(work)
        for source, task_ids in enumerate(self.env.completed_upload_tasks_by_uav):
            for task_id in task_ids:
                parent = self.env.parent_tasks.get(task_id)
                if parent is None or parent.finish_time is not None:
                    continue
                if (parent.source_uav != source or parent.remaining_upload_mbits > 1e-10
                        or parent.total_chunks != 0):
                    raise ValueError("Pending routing list contains a task in the wrong stage")
                priority = 0 if parent.priority == "high" else 1
                work[priority, source] += parent.size_mbits
                counts[priority, source] += 1
        return work, counts

    def _observation_parts(self):
        raw = self.env.get_raw_state()
        counts, oldest, upload, incomplete = self.task_aggregates()
        q_loc = np.stack([raw["Q_loc_high"], raw["Q_loc_low"]])
        q_tx = np.stack([raw["Q_tx_high"], raw["Q_tx_low"]])
        capacity = self.env.F_max_array / (self.env.C * 1e6)
        # This includes already assigned in-flight work, not hypothetical future arrivals.
        assigned_work = q_loc + q_tx.sum(axis=1)
        # PSD mode keeps fixed-reference quality here, not allocated service rates.
        # Feature name, 9D flattening, and normalization remain unchanged.
        u2u_eta = self.env.compute_u2u_reference_quality()
        u2g_rates = self.env.compute_u2g_rate_bps() / 1e6
        associations = raw["user_associations"]
        uplink = np.zeros(self.N)
        for source in range(self.N):
            selected = u2g_rates[associations == source]
            uplink[source] = float(np.mean(selected)) if selected.size else 0.0
        parts = {
            "compute_queue": compress(q_loc, 10.0),
            "transfer_queue": compress(q_tx, 10.0),
            "upload_queue": compress(upload, 10.0),
            "unfinished_tasks": compress(counts, 10.0),
            "oldest_age": compress(oldest, 10.0),
            "unfinished_chunks": compress(incomplete, 10.0),
            "assigned_cpu_seconds": compress(assigned_work / capacity[None, :], 1.0),
            "current_arrivals": compress(np.stack([raw["A_t_high"], raw["A_t_low"]]), 1.0),
            "current_load": compress(self.env.zone_expected_offered_load_mbps, 10.0),
            "cpu_capacity": capacity / 10.0,
            "u2u_spectral_efficiency": compress(u2u_eta, 1.0),
            "u2g_mean_rate": compress(uplink, 1.0),
            "uav_coordinates": raw["uav_coords"] / 1000.0,
            "centroids": raw["weighted_centroids"] / 1000.0,
            "centroid_offset": (raw["weighted_centroids"] - raw["uav_coords"]) / 60.0,
            "centroid_velocity": self.centroid_velocity,
            "phase_age": np.asarray([self.phase_age / 10.0]),
            "macro_clock": np.asarray([(self.env.global_micro_step % self.env.micro_steps_per_macro) / self.env.micro_steps_per_macro]),
        }
        if self.observe_routing_ready:
            ready_work, ready_counts = self.routing_ready_aggregates()
            parts["routing_ready_work"] = compress(ready_work, 10.0)
            parts["routing_ready_count"] = compress(ready_counts, 10.0)
        if self.observe_clusters:
            cluster_loads = self.env.cluster_expected_offered_load_mbps.astype(
                np.float64
            )
            cluster_totals = cluster_loads.sum(axis=1, keepdims=True)
            parts["cluster_load_share"] = np.divide(
                cluster_loads,
                cluster_totals,
                out=np.zeros_like(cluster_loads),
                where=cluster_totals > 0.0,
            )
            cluster_backlog = np.zeros((self.N, self.cluster_count))
            for parent in self.env.parent_tasks.values():
                if parent.finish_time is not None:
                    continue
                cluster_backlog[
                    parent.source_uav,
                    self.env.user_cluster_ids[parent.user_id],
                ] += 1.0
            parts["cluster_backlog"] = compress(cluster_backlog, 10.0)
            if self.config.get("macro_observation_layout") == "hotspot_upload_v2":
                offsets = np.zeros((self.N, self.cluster_count, 2))
                for source in range(self.N):
                    for cluster in range(self.cluster_count):
                        users = ((self.env.user_allocations == source)
                                 & (self.env.user_cluster_ids == cluster))
                        if not np.any(users):
                            raise ValueError("Hotspot observation requires nonempty clusters")
                        offsets[source, cluster] = (
                            self.env.user_coords[users].mean(axis=0)
                            - self.env.uav_coords[source]
                        )
                upload_by_cluster = np.zeros((2, self.N, self.cluster_count))
                for parent in self.env.parent_tasks.values():
                    if parent.finish_time is None:
                        priority = 0 if parent.priority == "high" else 1
                        upload_by_cluster[priority, parent.source_uav,
                                          self.env.user_cluster_ids[parent.user_id]] += (
                            parent.remaining_upload_mbits
                        )
                if not np.allclose(u2u_eta, u2u_eta.T, rtol=0, atol=1e-12):
                    raise ValueError("Pair encoding requires symmetric U2U reference quality")
                parts["u2u_pair_quality"] = compress(u2u_eta[np.triu_indices(self.N, 1)], 1.0)
                parts["cluster_offset"] = offsets / 60.0
                parts["cluster_offered_load"] = compress(cluster_loads, 10.0)
                parts["cluster_upload"] = compress(upload_by_cluster, 10.0)
        # Sum physical bits by destination before applying nonlinear encoding.
        parts["inbound_transfer_queue"] = compress(q_tx.sum(axis=1), 10.0)
        return parts

    def _observation(self, level):
        parts = self._observation_parts()
        keys = SHARED_FEATURES if self.observation_layout == "shared_v1" else (
            MICRO_FEATURES if level == "micro" else self.macro_features
        )
        if self.observe_routing_ready and (level == "micro" or self.observation_layout == "shared_v1"):
            keys += ("routing_ready_work", "routing_ready_count")
        slices = self.micro_feature_slices if level == "micro" else self.macro_feature_slices
        slices.clear()
        offset = 0
        for key in keys:
            value = parts[key]
            size = np.asarray(value).size
            slices[key] = [offset, offset + size]
            offset += size
        state = np.concatenate([np.asarray(parts[k]).ravel() for k in keys]).astype(np.float32)
        if not np.all(np.isfinite(state)):
            raise FloatingPointError("Non-finite observation")
        return state

    def micro_observation(self):
        """Routing observation at a service decision, after current arrivals."""
        return self._observation("micro")

    def macro_observation(self):
        """Deployment observation before motion, using past/current aggregates."""
        return self._observation("macro")

    def observation(self):
        """Compatibility alias for the micro observation, never the PPO input."""
        return self.micro_observation()

    def observation_spec(self):
        micro, macro = self.micro_observation(), self.macro_observation()
        return {
            "layout": self.observation_layout,
            "micro": {"dim": micro.size, "slices": dict(self.micro_feature_slices)},
            "macro": {"dim": macro.size, "slices": dict(self.macro_feature_slices)},
        }

    def prepare_service(self):
        self.env.prepare_service_resource_frame()
        return self.micro_observation()

    def route(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.size != self.micro_action_dim or not np.all(np.isfinite(action)):
            raise ValueError("Direct SAC requires finite 2*N*N routing scores")
        if np.any(np.abs(action) > 1.000001):
            raise ValueError("SAC scores must be bounded in [-1,1]")
        return project_rows_to_simplex(action.reshape(2, self.N, self.N))

    def start_macro(self, action):
        if self.env.continuous_mission:
            action = np.asarray(action, dtype=np.float64)
            if action.shape not in ((self.N, 2), (self.macro_action_dim,)) or not np.isfinite(action).all():
                raise ValueError("Mission macro action must contain finite world-coordinate scores")
            if self.config["max_displacement_m"] != 20:
                raise ValueError("mission_rpgm_v1 requires the fixed 20 m displacement limit")
            action = np.clip(action.reshape(self.N, 2), -1.0, 1.0)
            action /= np.maximum(np.linalg.norm(action, axis=1, keepdims=True), 1.0)
            _, _, info = self.env.macro_step(20.0 * action)
            self.propulsion_share = info["E_macro_total"] / self.env.micro_steps_per_macro
            self.propulsion_delta_share = info["E_propulsion_increment"] / self.env.micro_steps_per_macro
            return info
        action = np.asarray(action, dtype=np.float64).reshape(self.N, 2)
        action = np.clip(action, -1.0, 1.0)
        action /= np.maximum(np.linalg.norm(action, axis=1, keepdims=True), 1.0)
        basis, _ = cooperation_aligned_basis(
            self.env.weighted_centroids, self.env.uav_coords,
            self.env.zone_expected_offered_load_mbps,
        )
        offset = np.einsum("nij,nj->ni", basis, action)
        target = np.clip(self.env.weighted_centroids + self.config["waypoint_radius_m"] * offset, 0, 1000)
        delta = target - self.env.uav_coords
        distance = np.linalg.norm(delta, axis=1, keepdims=True)
        delta *= np.minimum(distance, self.config["max_displacement_m"]) / np.maximum(distance, 1e-12)
        delta[distance[:, 0] <= 0.5] = 0.0
        _, _, info = self.env.macro_step(delta)
        self.propulsion_share = info["E_macro_total"] / self.env.micro_steps_per_macro
        hover_fly = info["E_hover"] * self.env.deployment_dt / (self.env.macro_dt - self.env.deployment_dt)
        self.propulsion_delta_share = (info["E_flying"] - hover_fly) / self.env.micro_steps_per_macro
        return info

    def step_frame(self, action, gamma_frame=0.99):
        paused = self.env.macro_service_pause_steps_remaining > 0
        if not paused:
            self.env.prepare_service_resource_frame()
        route = self.route(action)
        costs, infos = [], []
        for _ in range(self.env.resource_frame_steps):
            _, info = self.env.micro_step(route[0], route[1])
            cost = holding_energy_cost(
                info["task_holding_cost_weighted"],
                info["total_energy"] + self.propulsion_delta_share,
                self.reward_config,
            )
            costs.append(cost)
            infos.append(info)
            self.total_power_energy += info["total_energy"] + self.propulsion_share
            self.cost_sum += cost
            self.integration_steps += 1
        gamma_step = gamma_frame ** (1.0 / self.env.resource_frame_steps)
        reward = -sum(gamma_step ** k * cost for k, cost in enumerate(costs))
        return {"reward": reward, "cost": sum(costs), "infos": infos, "route": route, "phase": "deployment" if paused else "service"}

    def metrics(self):
        result = self.env.get_task_delay_stats()
        counts, ages, upload, _ = self.task_aggregates()
        q_compute = float(self.env.Q_loc_high.sum() + self.env.Q_loc_low.sum())
        q_transfer = float(self.env.Q_tx_high.sum() + self.env.Q_tx_low.sum())
        result = {key: float(value) for key, value in result.items() if np.isscalar(value)}
        result.update(
            time_s=self.env._current_time(), queue_compute_mbit=q_compute,
            queue_transfer_mbit=q_transfer, queue_upload_mbit=float(upload.sum()),
            queue_total_mbit=q_compute + q_transfer + float(upload.sum()),
            unfinished=int(counts.sum()), oldest_age_s=float(ages.max()),
            J=self.cost_sum / max(self.integration_steps, 1),
            power_w=self.total_power_energy / max(self.env._current_time(), 1e-12),
            residual_m=float(np.linalg.norm(self.env.uav_coords - self.env.weighted_centroids, axis=1).mean()),
        )
        return result
