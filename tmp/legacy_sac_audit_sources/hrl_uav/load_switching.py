"""Deterministic episode-level load-switch schedules for HRL experiments."""

from dataclasses import dataclass

import numpy as np


LOAD_SWITCH_RNG_TAG = 0x4C5357


@dataclass(frozen=True)
class LoadSwitchSchedule:
    mode: str
    phases_mbps: np.ndarray
    phase_index_by_macro: np.ndarray
    switch_flag_by_macro: np.ndarray
    switch_steps: tuple
    dwell_steps: tuple
    episode_seed: int

    @property
    def enabled(self):
        return self.mode != "off"

    @property
    def macro_steps(self):
        return int(self.phase_index_by_macro.size)

    def phase_index(self, macro_index):
        return int(self.phase_index_by_macro[int(macro_index)])

    def loads(self, macro_index):
        return self.phases_mbps[self.phase_index(macro_index)].copy()

    def switched(self, macro_index):
        return bool(self.switch_flag_by_macro[int(macro_index)])


def parse_load_switch_phases(value, num_uavs):
    """Parse ``a,b,c;d,e,f`` into a positive phase matrix."""
    if value is None or not str(value).strip():
        return np.empty((0, int(num_uavs)), dtype=np.float64)
    rows = []
    for phase in str(value).split(";"):
        values = [
            float(item.strip())
            for item in phase.split(",")
            if item.strip()
        ]
        if len(values) != int(num_uavs):
            raise ValueError(
                "Each load-switch phase must contain exactly "
                f"{int(num_uavs)} loads."
            )
        rows.append(values)
    phases = np.asarray(rows, dtype=np.float64)
    if phases.shape[0] < 2:
        raise ValueError("Load switching requires at least two phases.")
    if not np.all(np.isfinite(phases)) or np.any(phases <= 0.0):
        raise ValueError("Load-switch phases must be finite and positive.")
    reference_total = float(np.sum(phases[0]))
    if not np.allclose(
        np.sum(phases, axis=1),
        reference_total,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            "All load-switch phases must preserve the same total load."
        )
    return phases


def build_load_switch_schedule(
    mode,
    phases_mbps,
    macro_steps,
    episode_seed,
    dwell_min=8,
    dwell_max=12,
):
    """Build a cyclic schedule without consuming any agent/environment RNG."""
    mode = str(mode).lower()
    macro_steps = int(macro_steps)
    episode_seed = int(episode_seed)
    if macro_steps <= 0:
        raise ValueError("macro_steps must be positive.")

    phases = np.asarray(phases_mbps, dtype=np.float64)
    if phases.ndim != 2 or phases.shape[0] == 0:
        raise ValueError("phases_mbps must be a non-empty 2D matrix.")

    if mode == "off":
        return LoadSwitchSchedule(
            mode=mode,
            phases_mbps=phases.copy(),
            phase_index_by_macro=np.zeros(macro_steps, dtype=np.int64),
            switch_flag_by_macro=np.zeros(macro_steps, dtype=np.bool_),
            switch_steps=tuple(),
            dwell_steps=(macro_steps,),
            episode_seed=episode_seed,
        )
    if mode != "cyclic_random":
        raise ValueError(f"Unsupported load-switch mode: {mode}")

    dwell_min = int(dwell_min)
    dwell_max = int(dwell_max)
    if dwell_min <= 0 or dwell_max < dwell_min:
        raise ValueError("Invalid load-switch dwell interval.")

    rng = np.random.default_rng(
        np.random.SeedSequence([episode_seed, LOAD_SWITCH_RNG_TAG])
    )
    phase_indices = np.empty(macro_steps, dtype=np.int64)
    switch_flags = np.zeros(macro_steps, dtype=np.bool_)
    switch_steps = []
    dwell_steps = []
    cursor = 0
    phase_index = 0
    while cursor < macro_steps:
        sampled_dwell = int(rng.integers(dwell_min, dwell_max + 1))
        segment_end = min(cursor + sampled_dwell, macro_steps)
        phase_indices[cursor:segment_end] = phase_index
        dwell_steps.append(segment_end - cursor)
        cursor = segment_end
        if cursor < macro_steps:
            switch_flags[cursor] = True
            switch_steps.append(cursor)
            phase_index = (phase_index + 1) % phases.shape[0]

    return LoadSwitchSchedule(
        mode=mode,
        phases_mbps=phases.copy(),
        phase_index_by_macro=phase_indices,
        switch_flag_by_macro=switch_flags,
        switch_steps=tuple(int(item) for item in switch_steps),
        dwell_steps=tuple(int(item) for item in dwell_steps),
        episode_seed=episode_seed,
    )


def apply_load_switch_phase(wrapper, schedule, macro_index, force=False):
    """Apply the current phase before the macro policy reads its state."""
    macro_index = int(macro_index)
    should_apply = bool(force) or schedule.switched(macro_index)
    if should_apply:
        base_env = getattr(wrapper, "env", wrapper)
        base_env.set_zone_target_loads(schedule.loads(macro_index))
    return schedule.switched(macro_index)

