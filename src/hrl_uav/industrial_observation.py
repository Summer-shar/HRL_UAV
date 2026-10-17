"""Immutable causal reports and a versioned, public-information-only encoding."""
from dataclasses import dataclass
import numpy as np

SCHEMA = "industrial_reports_v1"


def tuples(value):
    a = np.asarray(value)
    return tuple(float(x) for x in a) if a.ndim == 1 else tuple(tuples(row) for row in a)


@dataclass(frozen=True)
class TerminalReport:
    user_id: int
    source: int
    position_m: tuple
    sample_time_s: float
    valid: bool
    upload_bits: tuple
    pending_cycles: tuple
    pending_count: tuple
    oldest_age_s: tuple
    recent_bits: tuple
    recent_cycles: tuple
    recent_count: tuple
    history_duration_s: float
    history_valid: bool


@dataclass(frozen=True)
class ReadyTask:
    task_id: int
    source: int
    priority: int
    input_bits: float
    compute_cycles: float
    age_s: float


@dataclass(frozen=True)
class ControllerView:
    time_s: float
    revision: int
    phase: str
    terminals: tuple
    positions_m: tuple
    velocities_mps: tuple
    motion_remaining_s: float
    ready_tasks: tuple
    compute_cycles: tuple
    transfer_bits: tuple
    transfer_cycles: tuple
    unfinished_counts: tuple
    oldest_age_s: tuple
    cpu_hz: tuple
    uplink_rate_bps: tuple
    u2u_current_rate_bps: tuple
    u2u_all_active_rate_bps: tuple
    u2u_all_active_power_w: tuple
    schema: str = SCHEMA


class ObservationEncoder:
    """Identical legal information for both layers; no latent phase or deadline."""
    def __init__(self, config):
        self.map_scale = config["map_size_m"]
        self.speed_scale = config["motion"]["max_speed_mps"]
        self.history_scale = config["time"]["history_s"]
        self.slices = {}

    def encode(self, view):
        if view.schema != SCHEMA or len(view.terminals) != 39:
            raise ValueError("Unsupported report schema")
        parts, cursor = [], 0
        def add(name, value, scale, log=False):
            nonlocal cursor
            a = np.asarray(value, dtype=np.float64).ravel() / scale
            if log:
                a = np.sign(a) * np.log1p(np.abs(a))
            if not np.isfinite(a).all():
                raise ValueError(f"Nonfinite observation {name}")
            self.slices[name] = (cursor, cursor + a.size)
            cursor += a.size
            parts.append(a)
        for index, report in enumerate(view.terminals):
            if report.user_id != index or report.sample_time_s > view.time_s + 1e-9:
                raise ValueError("Misordered or future report")
            prefix = f"terminal_{index}."
            add(prefix + "position", report.position_m if report.valid else (0, 0), self.map_scale)
            add(prefix + "identity_validity", (report.source / 2, report.valid,
                max(0, view.time_s-report.sample_time_s) / self.history_scale,
                report.history_duration_s / self.history_scale, report.history_valid and report.valid), 1)
            for name, scale in (("upload_bits", 1e6), ("pending_cycles", 1e9), ("pending_count", 10),
                                ("oldest_age_s", 10), ("recent_bits", 1e6), ("recent_cycles", 1e9), ("recent_count", 10)):
                add(prefix + name, getattr(report, name) if report.valid else (0, 0), scale, True)
        ready = np.zeros((3, 2, 3))
        for task in view.ready_tasks:
            ready[:, task.priority, task.source] += (task.input_bits, task.compute_cycles, 1)
        for name, value, scale, log in (
                ("uav_positions", view.positions_m, self.map_scale, False),
                ("executed_velocity", view.velocities_mps, self.speed_scale, False),
                ("motion_remaining", [view.motion_remaining_s], self.history_scale, False),
                ("ready_bits", ready[0], 1e6, True), ("ready_cycles", ready[1], 1e9, True),
                ("ready_count", ready[2], 10, True),
                ("compute_cycles", view.compute_cycles, 1e9, True),
                ("transfer_bits", view.transfer_bits, 1e6, True),
                ("transfer_cycles", view.transfer_cycles, 1e9, True),
                ("unfinished_counts", view.unfinished_counts, 10, True),
                ("oldest_age", view.oldest_age_s, 10, True),
                ("cpu_hz", view.cpu_hz, 1e10, False),
                ("uplink_rates", view.uplink_rate_bps, 1e6, True),
                ("current_u2u_rates", view.u2u_current_rate_bps, 1e6, True),
                ("all_active_u2u_rates", view.u2u_all_active_rate_bps, 1e6, True),
                ("all_active_u2u_power", view.u2u_all_active_power_w, 1, False)):
            add(name, value, scale, log)
        return np.concatenate(parts).astype(np.float32)

    def spec(self, view):
        encoded = self.encode(view)
        return dict(schema=SCHEMA, macro=dict(dim=encoded.size, slices=dict(self.slices)),
                    micro=dict(dim=encoded.size, slices=dict(self.slices)),
                    encoding="fixed terminal order; signed log1p workload scaling; explicit missing masks")
