"""Static terminals and slotted exogenous activity/arrivals, independent of control."""
from dataclasses import dataclass, asdict
from pathlib import Path
import copy
import hashlib
import json
import math
import numpy as np

TYPES = ("sensor", "controller", "vision")


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return validate_config(cfg)


def validate_config(cfg):
    c = copy.deepcopy(cfg)
    if c.get("schema") != "industrial_static_v1" or (c["num_uavs"], c["num_users"]) != (3, 39):
        raise ValueError("Expected industrial_static_v1 with 3 UAVs and 39 fixed terminals")
    if set(c) - {"schema", "status", "num_uavs", "num_users", "map_size_m", "height_m", "regions_m", "device_counts", "uav_initial_positions_m", "region_multipliers", "activity", "task_types", "time", "motion", "resources", "objective", "parameter_sources"}:
        raise ValueError("Unknown top-level configuration key (legacy overrides are forbidden)")
    def positive(x):
        a = np.asarray(x, dtype=float)
        return np.isfinite(a).all() and np.all(a > 0)
    for x in (c["map_size_m"], c["height_m"], list(c["time"].values()), list(c["motion"].values()),
              c["region_multipliers"], list(c["activity"].values())):
        if not positive(x):
            raise ValueError("Nonpositive or nonfinite physical parameter")
    t = c["time"]
    for large, small in ((t["routing_s"], t["physical_s"]), (t["motion_s"], t["routing_s"])):
        if not math.isclose(large / small, round(large / small), abs_tol=1e-9):
            raise ValueError("Decision clocks must be integer multiples of the physical clock")
    regions = np.asarray(c["regions_m"], float)
    positions = np.asarray(c["uav_initial_positions_m"], float)
    counts = np.asarray(c["device_counts"])
    if regions.shape != (3, 2, 2) or not np.isfinite(regions).all() or np.any(regions[:, 1] <= regions[:, 0]):
        raise ValueError("Invalid region rectangles")
    if np.any(regions < 0) or np.any(regions > c["map_size_m"]):
        raise ValueError("Region outside map")
    if positions.shape != (3, 2) or not np.isfinite(positions).all() or np.any(positions < 0) or np.any(positions > c["map_size_m"]):
        raise ValueError("Invalid initial UAV positions")
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
    if np.any(distances[~np.eye(3, dtype=bool)] < c["motion"]["min_separation_m"]):
        raise ValueError("Initial UAV separation is unsafe")
    if counts.shape != (3, 3) or np.any(counts < 0) or np.any(counts != np.floor(counts)) or np.any(counts.sum(axis=1) != 13):
        raise ValueError("Each region must contain 13 explicitly typed devices")
    if np.shape(c["region_multipliers"]) != (3,) or set(c["task_types"]) != set(TYPES):
        raise ValueError("Invalid region or device profiles")
    for p in c["task_types"].values():
        if set(p) != {"rate_per_s", "high_probability", "input_bits", "cycles_per_bit"}:
            raise ValueError("Unknown task profile key")
        if not np.isfinite(p["rate_per_s"]) or p["rate_per_s"] < 0 or not 0 <= p["high_probability"] <= 1:
            raise ValueError("Invalid task rate/probability")
        if np.shape(p["input_bits"]) != (2,) or np.shape(p["cycles_per_bit"]) != (2,) or not positive(p["input_bits"]) or not positive(p["cycles_per_bit"]):
            raise ValueError("Expected high/low bit and cycles profiles")
    r = c["resources"]
    if np.shape(r["cpu_hz"]) != (3,) or not positive(r["cpu_hz"]) or not positive([v for k, v in r.items() if k != "cpu_hz"]):
        raise ValueError("Invalid resource budget")
    o = c["objective"]
    if o["energy_mode"] != "full" or not 0 < o["gamma_per_routing_frame"] <= 1 or not 0 <= o["holding_weight"] <= 1:
        raise ValueError("Invalid objective or energy definition")
    if np.shape(o["priority_weights"]) != (2,) or not positive(o["priority_weights"]) or not positive([o["holding_reference_task_s"], o["energy_reference_j"]]):
        raise ValueError("Invalid cost normalization")
    return c


@dataclass(frozen=True)
class Terminal:
    user_id: int
    source: int
    task_type: str
    position_m: tuple


@dataclass(frozen=True)
class Arrival:
    task_id: int
    user_id: int
    source: int
    task_type: str
    priority: int
    birth_s: float
    input_bits: float
    compute_cycles: float


class IndustrialScene:
    def __init__(self, config, seed, layout_seed=None):
        self.config = validate_config(config)
        self.seed = int(seed)
        self.layout_seed = self.seed if layout_seed is None else int(layout_seed)
        rng = np.random.default_rng(np.random.SeedSequence([self.layout_seed, 1001]))
        terminals = []
        for region, counts in enumerate(self.config["device_counts"]):
            lo, hi = self.config["regions_m"][region]
            for kind, count in zip(TYPES, counts):
                for _ in range(count):
                    terminals.append(Terminal(len(terminals), region, kind, tuple(rng.uniform(lo, hi))))
        self.terminals = tuple(terminals)
        self._activity_rng = [np.random.default_rng(np.random.SeedSequence([self.seed, 2001, k])) for k in range(39)]
        self._arrival_rng = [np.random.default_rng(np.random.SeedSequence([self.seed, 3001, k])) for k in range(39)]
        self._mark_rng = [np.random.default_rng(np.random.SeedSequence([self.seed, 4001, k])) for k in range(39)]
        a, dt = self.config["activity"], self.config["time"]["routing_s"]
        self._switch_p = np.array([-math.expm1(-dt / a["normal_mean_s"]), -math.expm1(-dt / a["busy_mean_s"])])
        busy_probability = self._switch_p[0] / self._switch_p.sum()
        self._busy = np.array([r.random() < busy_probability for r in self._activity_rng])
        self._frame = -1
        self._task_id = 0
        self.event_digest = hashlib.sha256()

    def frame(self, frame_index):
        if frame_index != self._frame + 1:
            raise ValueError("Exogenous frames must advance exactly once in physical order")
        dt = self.config["time"]["routing_s"]
        events = []
        for k, terminal in enumerate(self.terminals):
            if frame_index > 0 and self._activity_rng[k].random() < self._switch_p[int(self._busy[k])]:
                self._busy[k] = not self._busy[k]
            profile = self.config["task_types"][terminal.task_type]
            activity = self.config["activity"]["busy_multiplier" if self._busy[k] else "normal_multiplier"]
            rate = profile["rate_per_s"] * self.config["region_multipliers"][terminal.source] * activity
            for _ in range(self._arrival_rng[k].poisson(rate * dt)):
                priority = 0 if self._mark_rng[k].random() < profile["high_probability"] else 1
                bits = float(profile["input_bits"][priority])
                arrival = Arrival(self._task_id, k, terminal.source, terminal.task_type, priority,
                                  frame_index * dt, bits, bits * profile["cycles_per_bit"][priority])
                self._task_id += 1
                events.append(arrival)
                self.event_digest.update(json.dumps(asdict(arrival), sort_keys=True).encode())
        self._frame = frame_index
        return tuple(events)


def workload_summary(config):
    c = validate_config(config)
    a, dt = c["activity"], c["time"]["routing_s"]
    p_n, p_b = (-math.expm1(-dt / a[k]) for k in ("normal_mean_s", "busy_mean_s"))
    busy = p_n / (p_n + p_b)
    multiplier = (1 - busy) * a["normal_multiplier"] + busy * a["busy_multiplier"]
    rows = []
    for region, counts in enumerate(c["device_counts"]):
        bit_rate = cycle_rate = 0.
        for kind, count in zip(TYPES, counts):
            p = c["task_types"][kind]
            probabilities = np.array([p["high_probability"], 1 - p["high_probability"]])
            bits = np.array(p["input_bits"])
            frequency = count * p["rate_per_s"] * c["region_multipliers"][region] * multiplier
            bit_rate += frequency * float(probabilities @ bits)
            cycle_rate += frequency * float(probabilities @ (bits * p["cycles_per_bit"]))
        rows.append(dict(region=region, input_mbps=bit_rate / 1e6, compute_gcycles_s=cycle_rate / 1e9))
    return dict(busy_probability=busy, normal_mean_discrete_s=dt / p_n,
                busy_mean_discrete_s=dt / p_b, regions=rows,
                total_input_mbps=sum(r["input_mbps"] for r in rows),
                total_compute_gcycles_s=sum(r["compute_gcycles_s"] for r in rows),
                note="Analytical expected workload, not queue stability or performance evidence")
