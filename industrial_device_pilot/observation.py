"""Causal per-device PPO reports; the existing SAC observation is unchanged.

Device identity is checked as report integrity metadata only. Within each source,
the incoming report order is retained, so permutation invariance is a property of
the set networks rather than an identity-based reordering in this encoder.
"""

import copy
import math

import numpy as np

from hrl_uav.industrial_deployment_observation import (
    DeploymentControllerView,
    DeploymentObservationEncoder,
    LINKS,
    OBSERVATION_SCHEMA as LEGACY_OBSERVATION_SCHEMA,
    REPORT_SCHEMA,
)
from hrl_uav.industrial_observation import ControllerView, SCHEMA as BASE_REPORT_SCHEMA


OBSERVATION_SCHEMA = "industrial_device_observation_v1"
LAYOUT_VERSION = "source_device12_global55_v1"
SOURCE_COUNT = 3
DEVICES_PER_SOURCE = 13
DEVICE_FEATURE_DIM = 12
DEVICE_BLOCK_DIM = SOURCE_COUNT * DEVICES_PER_SOURCE * DEVICE_FEATURE_DIM
GLOBAL_DIM = 55
MACRO_DIM = DEVICE_BLOCK_DIM + GLOBAL_DIM
MICRO_DIM = 86


class DeviceObservationEncoder:
    """Encode legal synchronous reports without using a radio model for PPO."""

    def __init__(self, config, deployment_config):
        self.config = copy.deepcopy(config)
        self.deployment_config = copy.deepcopy(deployment_config)
        self.legacy_encoder = DeploymentObservationEncoder(config, deployment_config)
        length = float(config["map_size_m"])
        history = float(config["time"]["history_s"])
        if not math.isfinite(length) or length <= 0 or not math.isfinite(history) or history <= 0:
            raise ValueError("Invalid fixed observation scales")
        self.terminal_fields = (
            ("relative_position", 2, length, False),
            ("upload_bits", 2, 1e6, True),
            ("pending_cycles", 2, 1e9, True),
            ("pending_count", 2, 10., True),
            ("recent_bit_rate", 2, 1e6, True),
            ("recent_cycle_rate", 2, 1e9, True),
        )
        self.global_fields = (
            ("positions", 6, length, False),
            ("ready_bits", 6, 1e6, True),
            ("ready_cycles", 6, 1e9, True),
            ("ready_count", 6, 10., True),
            ("compute_cycles", 6, 1e9, True),
            ("transfer_bits", 12, 1e6, True),
            ("inbound_cycles", 6, 1e9, True),
            ("unfinished_count", 6, 10., True),
            ("history_duration", 1, history, False),
        )

    def spec(self):
        terminal = DeploymentObservationEncoder._field_spec(self.terminal_fields)
        global_local = DeploymentObservationEncoder._field_spec(self.global_fields)
        macro_fields = []
        for source in range(SOURCE_COUNT):
            for slot in range(DEVICES_PER_SOURCE):
                for name, size, scale, logged in self.terminal_fields:
                    macro_fields.append((f"source_{source}.device_slot_{slot}.{name}", size, scale, logged))
        macro_fields.extend((f"global.{name}", size, scale, logged)
                            for name, size, scale, logged in self.global_fields)
        macro = DeploymentObservationEncoder._field_spec(macro_fields)
        macro.update(
            device_block=dict(slice=[0, DEVICE_BLOCK_DIM], shape=[3, 13, 12],
                              order="source, report-order device slot, feature",
                              device_features=terminal),
            global_block=dict(slice=[DEVICE_BLOCK_DIM, MACRO_DIM],
                              fields_relative_to_block=global_local),
        )
        old_spec = self.legacy_encoder.spec()
        return dict(
            schema=OBSERVATION_SCHEMA,
            layout=LAYOUT_VERSION,
            macro=macro,
            micro=copy.deepcopy(old_spec["micro"]),
            micro_reference_schema=LEGACY_OBSERVATION_SCHEMA,
            micro_reference_layout=old_spec["layout"],
            micro_array_order=old_spec["array_order"],
            raw_report_schema=REPORT_SCHEMA,
            base_report_schema=BASE_REPORT_SCHEMA,
            priorities=["high", "low"],
            directed_links_zero_based=[list(link) for link in LINKS],
            array_order=("device block: source, input-order device slot, feature; "
                         "device priority pairs: high, low; global workloads: priority, node; "
                         "global positions: node, coordinate; transfer: priority, directed link"),
            encoding="nonnegative log1p workload; fixed physical scales; no missing report fallback",
            time_semantics=("recent arrivals over past min(time_s, history_s), divided by that "
                            "observed duration; zero rate at time zero; current queues separate"),
            identity_semantics=("user_id only validates unique fixed source association; "
                                "no identity or template number is encoded"),
            radio_semantics=("macro uses public geometry and workload only; "
                             "micro delegates the unchanged legacy encoder"),
            deployment_contract=copy.deepcopy(self.deployment_config),
        )

    @staticmethod
    def _array(value, shape, name, nonnegative=True):
        return DeploymentObservationEncoder._array(value, shape, name, nonnegative)

    def _validated_macro(self, wrapper):
        # Do not call legacy _validated/_features: those inspect radio rates.
        if not isinstance(wrapper, DeploymentControllerView) or wrapper.schema != REPORT_SCHEMA:
            raise ValueError("Expected industrial_deployment_reports_v3")
        view = wrapper.base
        if not isinstance(view, ControllerView) or view.schema != BASE_REPORT_SCHEMA:
            raise ValueError("Unexpected base report schema")
        if not math.isfinite(view.time_s) or view.time_s < 0 or view.revision < 0:
            raise ValueError("Invalid report clock")
        if len(view.terminals) != 39 or sorted(t.user_id for t in view.terminals) != list(range(39)):
            raise ValueError("Reports must identify each of the 39 terminals exactly once")
        positions = self._array(view.positions_m, (3, 2), "positions")
        if np.any(positions > self.config["map_size_m"]):
            raise ValueError("UAV position outside map")
        velocity = self._array(view.velocities_mps, (3, 2), "velocities", False)
        if np.any(np.linalg.norm(velocity, axis=1) > self.config["motion"]["max_speed_mps"] + 1e-8):
            raise ValueError("Velocity exceeds configured speed")
        remaining, flight = wrapper.macro_remaining_s, view.motion_remaining_s
        period = self.deployment_config["macro_period_s"]
        adjustment = self.deployment_config["adjustment_s"]
        if (not math.isfinite(remaining) or not 0 <= remaining <= period + 1e-9
                or not math.isfinite(flight) or flight < 0):
            raise ValueError("Invalid deployment clock")
        expected_flight = max(0., remaining - (period - adjustment))
        expected_phase = ("awaiting_decision" if remaining == 0 else
                          "adjusting" if expected_flight > 1e-9 else "holding")
        if not math.isclose(flight, expected_flight, rel_tol=0., abs_tol=1e-8):
            raise ValueError("Flight and macro clocks disagree")
        if wrapper.deployment_phase != expected_phase:
            raise ValueError("Deployment phase and clocks disagree")
        if expected_flight == 0 and np.any(velocity != 0):
            raise ValueError("Holding and uncommitted reports must have zero velocity")

        duration = min(view.time_s, self.config["time"]["history_s"])
        devices = [[] for _ in range(SOURCE_COUNT)]
        for report in view.terminals:
            if (not report.valid or not math.isfinite(report.sample_time_s)
                    or abs(report.sample_time_s - view.time_s) > 1e-9):
                raise ValueError("Missing, stale or future terminal report")
            if report.source != report.user_id // DEVICES_PER_SOURCE:
                raise ValueError("Terminal source differs from fixed source association")
            if (not math.isfinite(report.history_duration_s)
                    or abs(report.history_duration_s - duration) > 1e-8
                    or bool(report.history_valid) != (duration > 0)):
                raise ValueError("Inconsistent synchronous history window")
            terminal_position = self._array(report.position_m, (2,), "terminal position")
            if np.any(terminal_position > self.config["map_size_m"]):
                raise ValueError("Terminal position outside map")
            raw = {name: self._array(getattr(report, name), (2,), name)
                   for name in ("upload_bits", "pending_cycles", "pending_count", "oldest_age_s",
                                "recent_bits", "recent_cycles", "recent_count")}
            device_features = dict(
                relative_position=terminal_position - positions[report.source],
                upload_bits=raw["upload_bits"],
                pending_cycles=raw["pending_cycles"],
                pending_count=raw["pending_count"],
                recent_bit_rate=raw["recent_bits"] / duration if duration > 0 else np.zeros(2),
                recent_cycle_rate=raw["recent_cycles"] / duration if duration > 0 else np.zeros(2),
            )
            devices[report.source].append(device_features)
        if any(len(group) != DEVICES_PER_SOURCE for group in devices):
            raise ValueError("Expected 13 devices per fixed source")

        compute = self._array(view.compute_cycles, (2, 3), "compute cycles")
        transfer = self._array(view.transfer_bits, (2, 3, 3), "transfer bits")
        transfer_cycles = self._array(view.transfer_cycles, (2, 3, 3), "transfer cycles")
        for array in (transfer, transfer_cycles):
            if np.any(array[:, np.arange(3), np.arange(3)] != 0):
                raise ValueError("Local work cannot occupy U2U queues")
        counts = self._array(view.unfinished_counts, (2, 3), "unfinished counts")
        ready, ids = np.zeros((3, 2, 3)), set()
        for task in view.ready_tasks:
            if task.task_id in ids or task.source not in range(3) or task.priority not in range(2):
                raise ValueError("Invalid or duplicate ready task")
            ids.add(task.task_id)
            self._array((task.input_bits, task.compute_cycles, task.age_s), (3,), "ready task")
            ready[:, task.priority, task.source] += (task.input_bits, task.compute_cycles, 1)
        global_features = dict(
            positions=positions,
            ready_bits=ready[0], ready_cycles=ready[1], ready_count=ready[2],
            compute_cycles=compute,
            transfer_bits=np.stack([transfer[:, i, j] for i, j in LINKS], axis=1),
            inbound_cycles=transfer_cycles.sum(axis=1),
            unfinished_count=counts,
            history_duration=[duration],
        )
        return devices, global_features

    @staticmethod
    def _encode_fields(features, fields):
        parts = []
        for name, size, scale, logged in fields:
            value = np.asarray(features[name], dtype=np.float64).ravel() / scale
            if value.size != size or not np.isfinite(value).all() or (logged and np.any(value < 0)):
                raise ValueError(f"Invalid encoded feature {name}")
            parts.append(np.log1p(value) if logged else value)
        return np.concatenate(parts)

    def encode_macro(self, view):
        devices, global_features = self._validated_macro(view)
        parts = [self._encode_fields(device, self.terminal_fields)
                 for group in devices for device in group]
        parts.append(self._encode_fields(global_features, self.global_fields))
        result = np.concatenate(parts).astype(np.float32)
        if result.shape != (MACRO_DIM,) or not np.isfinite(result).all():
            raise ValueError("Invalid device observation or overflow")
        return result

    def encode_micro(self, view):
        return self.legacy_encoder.encode_micro(view)
