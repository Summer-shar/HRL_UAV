"""Causal, immutable deployment reports and the versioned split observation contract."""
import copy
from dataclasses import dataclass
import math
import numpy as np

from .industrial_observation import ControllerView, SCHEMA as BASE_REPORT_SCHEMA

REPORT_SCHEMA = "industrial_deployment_reports_v3"
OBSERVATION_SCHEMA = "industrial_deployment_v3"
LINKS = tuple((i, j) for i in range(3) for j in range(3) if i != j)


@dataclass(frozen=True)
class DeploymentControllerView:
    base: ControllerView
    macro_remaining_s: float
    deployment_phase: str
    schema: str = REPORT_SCHEMA


class DeploymentObservationEncoder:
    """Read reports and fixed configuration only; aggregation is intentionally lossy."""
    def __init__(self, config, deployment_config):
        self.config = copy.deepcopy(config)
        self.deployment_config = copy.deepcopy(deployment_config)
        length, history, speed = (config["map_size_m"], config["time"]["history_s"],
                                  config["motion"]["max_speed_mps"])
        macro = deployment_config["macro_period_s"]
        self.macro_fields = (
            ("positions", 6, length, False), ("upload_pressure", 6, 10., True),
            ("upload_offset", 12, length, False), ("recent_bit_rate", 6, 1e6, True),
            ("recent_offset", 12, length, False), ("upstream_cycles", 6, 1e9, True),
            ("recent_cycle_rate", 6, 1e9, True), ("compute_cycles", 6, 1e9, True),
            ("transfer_bits", 12, 1e6, True), ("inbound_cycles", 6, 1e9, True),
            ("unfinished_count", 6, 10., True), ("history_duration", 1, history, False))
        self.micro_fields = (
            ("ready_bits", 6, 1e6, True), ("ready_cycles", 6, 1e9, True),
            ("ready_count", 6, 10., True), ("compute_cycles", 6, 1e9, True),
            ("transfer_bits", 12, 1e6, True), ("inbound_cycles", 6, 1e9, True),
            ("upload_pressure", 6, 10., True), ("ground_cycles", 6, 1e9, True),
            ("recent_bit_rate", 6, 1e6, True), ("recent_cycle_rate", 6, 1e9, True),
            ("unfinished_count", 6, 10., True), ("positions", 6, length, False),
            ("executed_velocity", 6, speed, False), ("macro_remaining", 1, macro, False),
            ("history_duration", 1, history, False))

    @staticmethod
    def _field_spec(fields):
        cursor, rows, slices = 0, [], {}
        for name, size, scale, logged in fields:
            rows.append(dict(name=name, slice=[cursor, cursor+size], dim=size,
                             scale=float(scale), log1p=logged))
            slices[name] = [cursor, cursor+size]
            cursor += size
        return dict(dim=cursor, fields=rows, slices=slices)

    def spec(self):
        return dict(schema=OBSERVATION_SCHEMA, layout="split_v1",
                    macro=self._field_spec(self.macro_fields), micro=self._field_spec(self.micro_fields),
                    raw_report_schema=REPORT_SCHEMA, base_report_schema=BASE_REPORT_SCHEMA,
                    priorities=["high", "low"], directed_links_zero_based=[list(x) for x in LINKS],
                    array_order="priority, node; spatial offsets priority, source, coordinate",
                    encoding="nonnegative log1p workload; fixed physical scales; no missing report fallback",
                    time_semantics="macro_remaining is time until next upper decision after commitment",
                    deployment_contract=copy.deepcopy(self.deployment_config))

    @staticmethod
    def _array(value, shape, name, nonnegative=True):
        a = np.asarray(value, dtype=np.float64)
        if a.shape != shape or not np.isfinite(a).all() or (nonnegative and np.any(a < 0)):
            raise ValueError(f"Invalid report field: {name}")
        return a

    def _validated(self, wrapper):
        if not isinstance(wrapper, DeploymentControllerView) or wrapper.schema != REPORT_SCHEMA:
            raise ValueError("Expected industrial_deployment_reports_v3")
        v = wrapper.base
        if not isinstance(v, ControllerView) or v.schema != BASE_REPORT_SCHEMA:
            raise ValueError("Unexpected base report schema")
        if not math.isfinite(v.time_s) or v.time_s < 0 or v.revision < 0:
            raise ValueError("Invalid report clock")
        if len(v.terminals) != 39 or sorted(t.user_id for t in v.terminals) != list(range(39)):
            raise ValueError("Reports must identify each of the 39 terminals exactly once")
        positions = self._array(v.positions_m, (3, 2), "positions")
        if np.any(positions > self.config["map_size_m"]):
            raise ValueError("UAV position outside map")
        velocity = self._array(v.velocities_mps, (3, 2), "velocities", False)
        if np.any(np.linalg.norm(velocity, axis=1) > self.config["motion"]["max_speed_mps"]+1e-8):
            raise ValueError("Velocity exceeds configured speed")
        b, flight = wrapper.macro_remaining_s, v.motion_remaining_s
        h, m = self.deployment_config["macro_period_s"], self.deployment_config["adjustment_s"]
        if not math.isfinite(b) or not 0 <= b <= h+1e-9 or not math.isfinite(flight) or flight < 0:
            raise ValueError("Invalid deployment clock")
        expected_flight = max(0., b-(h-m))
        expected_phase = "awaiting_decision" if b == 0 else "adjusting" if expected_flight > 1e-9 else "holding"
        if not math.isclose(flight, expected_flight, rel_tol=0., abs_tol=1e-8):
            raise ValueError("Flight and macro clocks disagree")
        if wrapper.deployment_phase != expected_phase:
            raise ValueError("Deployment phase and clocks disagree")
        if expected_flight == 0 and np.any(velocity != 0):
            raise ValueError("Holding and uncommitted reports must have zero velocity")
        rates = self._array(v.uplink_rate_bps, (39,), "uplink rates")
        if np.any(rates <= 0):
            raise ValueError("Upload-pressure encoding requires positive uplink rates")
        duration = min(v.time_s, self.config["time"]["history_s"])
        terminal_positions, source = np.zeros((39, 2)), np.zeros(39, dtype=int)
        raw = {key: np.zeros((39, 2)) for key in ("upload_bits", "pending_cycles", "recent_bits", "recent_cycles")}
        for t in v.terminals:
            k = t.user_id
            if not t.valid or not math.isfinite(t.sample_time_s) or abs(t.sample_time_s-v.time_s) > 1e-9:
                raise ValueError("Missing, stale or future terminal report")
            if t.source != k//13:
                raise ValueError("Terminal source differs from fixed source association")
            if (not math.isfinite(t.history_duration_s) or abs(t.history_duration_s-duration) > 1e-8 or
                    bool(t.history_valid) != (duration > 0)):
                raise ValueError("Inconsistent synchronous history window")
            terminal_positions[k] = self._array(t.position_m, (2,), "terminal position")
            if np.any(terminal_positions[k] > self.config["map_size_m"]):
                raise ValueError("Terminal position outside map")
            source[k] = t.source
            for key in ("upload_bits", "pending_cycles", "pending_count", "oldest_age_s",
                        "recent_bits", "recent_cycles", "recent_count"):
                a = self._array(getattr(t, key), (2,), key)
                if key in raw:
                    raw[key][k] = a
        compute = self._array(v.compute_cycles, (2, 3), "compute cycles")
        transfer = self._array(v.transfer_bits, (2, 3, 3), "transfer bits")
        tx_cycles = self._array(v.transfer_cycles, (2, 3, 3), "transfer cycles")
        for a in (transfer, tx_cycles):
            if np.any(a[:, np.arange(3), np.arange(3)] != 0):
                raise ValueError("Local work cannot occupy U2U queues")
        counts = self._array(v.unfinished_counts, (2, 3), "unfinished counts")
        ready, ids = np.zeros((3, 2, 3)), set()
        for task in v.ready_tasks:
            if task.task_id in ids or task.source not in range(3) or task.priority not in range(2):
                raise ValueError("Invalid or duplicate ready task")
            ids.add(task.task_id)
            self._array((task.input_bits, task.compute_cycles, task.age_s), (3,), "ready task")
            ready[:, task.priority, task.source] += (task.input_bits, task.compute_cycles, 1)
        return v, positions, velocity, rates, terminal_positions, source, raw, duration, compute, transfer, tx_cycles, counts, ready

    def _features(self, wrapper):
        v, positions, velocity, rates, z, source, raw, duration, compute, transfer, tx_cycles, counts, ready = self._validated(wrapper)

        def aggregate(a):
            return np.stack([a[source == i].sum(axis=0) for i in range(3)], axis=1)

        def offsets(weights):
            result = np.zeros((2, 3, 2))
            for i in range(3):
                terminal_z, mass = z[source == i], weights[source == i]
                for p in range(2):
                    total = mass[:, p].sum()
                    center = (mass[:, p, None]*terminal_z).sum(axis=0)/total if total > 0 else terminal_z.mean(axis=0)
                    result[p, i] = center-positions[i]
            return result

        pressure = raw["upload_bits"]/rates[:, None]
        bit_rate = raw["recent_bits"]/duration if duration > 0 else np.zeros((39, 2))
        cycle_rate = raw["recent_cycles"]/duration if duration > 0 else np.zeros((39, 2))
        ground = aggregate(raw["pending_cycles"])
        return dict(positions=positions, executed_velocity=velocity, macro_remaining=[wrapper.macro_remaining_s],
                    history_duration=[duration], upload_pressure=aggregate(pressure), upload_offset=offsets(pressure),
                    recent_bit_rate=aggregate(bit_rate), recent_cycle_rate=aggregate(cycle_rate),
                    recent_offset=offsets(bit_rate), ground_cycles=ground, upstream_cycles=ground+ready[1],
                    ready_bits=ready[0], ready_cycles=ready[1], ready_count=ready[2], compute_cycles=compute,
                    transfer_bits=np.stack([transfer[:, i, j] for i, j in LINKS], axis=1),
                    inbound_cycles=tx_cycles.sum(axis=1), unfinished_count=counts)

    def _encode(self, wrapper, fields):
        features, parts = self._features(wrapper), []
        for name, size, scale, logged in fields:
            value = np.asarray(features[name], dtype=float).ravel()/scale
            if value.size != size or not np.isfinite(value).all() or (logged and np.any(value < 0)):
                raise ValueError(f"Invalid encoded feature {name}")
            parts.append(np.log1p(value) if logged else value)
        result = np.concatenate(parts).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Observation overflow")
        return result

    def encode_macro(self, view):
        return self._encode(view, self.macro_fields)

    def encode_micro(self, view):
        return self._encode(view, self.micro_fields)
