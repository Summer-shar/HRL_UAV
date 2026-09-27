"""Bounded fixed-state task-splitting mechanisms using production physics.

Importing this module performs no experiment. run_mechanisms() requests exactly
nine cases of 5 seconds (50 ticks with the reviewed production configuration).
The caller owns the global physical-step budget and may instrument micro_step.

These are controlled already-uploaded initial states, not arrival-to-completion
experiments: every parent has birth=0 and upload_finish=0 as the observation
origin convention. Initial upload work is represented in conservation counters,
but its pre-window delay/energy is excluded. No CPU or U2U work is forgiven.
"""
import json
import math

import numpy as np

from hrl_uav.industrial_hotspot_env import HotspotIndustrialEnv
from hrl_uav.industrial_scene import Arrival
from .paths import default_config


WINDOW_SECONDS = 5.0
BACKGROUND_PARENTS = 10
TARGET_BITS = 1_400_000.0
TARGET_CYCLES = 2_100_000_000.0


def _empty_arrival_tape(env):
    """Install a known empty tape, without changing any scene parameter."""
    scene = env._scene

    def frame(index):
        if type(index) is not int or index != scene._frame + 1:
            raise ValueError("Mechanism arrival frames must advance exactly once")
        scene._frame = index
        return ()

    scene.frame = frame


def _admit_ready(env, *, source, priority):
    """Declare initial already-uploaded work and reconcile its full bit ledger."""
    terminal = next(t for t in env._scene.terminals
                    if t.source == source and t.task_type == "vision")
    pid = len(env.parents)
    arrival = Arrival(pid, terminal.user_id, source, "vision", priority,
                      0.0, TARGET_BITS, TARGET_CYCLES)
    env._admit(arrival)
    parent = env.parents[pid]
    env.upload[terminal.user_id].remove(pid)
    parent.upload_remaining = 0.0
    parent.upload_finish_s = 0.0
    env.upload_served_bits += arrival.input_bits
    env.ready.append(pid)
    env._invalidate_view()
    return pid


def _conservation(env):
    errors = {key: float(value) for key, value in env.conservation().items()}
    for key, value in errors.items():
        scale = env.generated_cycles if "cycles" in key else env.generated_bits
        if not math.isfinite(value) or abs(value) > max(1e-7, scale * 1e-12):
            raise AssertionError(f"Mechanism conservation failed: {key}={value}")
    for parent in env.parents.values():
        if parent.route_time_s is None:
            continue
        fragments = [env.fragments[fid] for fid in parent.fragment_ids]
        if not fragments:
            raise AssertionError("A routed parent has no positive fragment")
        for field, total in (("input_bits", parent.arrival.input_bits),
                             ("compute_cycles", parent.arrival.compute_cycles)):
            if not math.isclose(sum(getattr(f, field) for f in fragments), total,
                                rel_tol=1e-12, abs_tol=1e-7):
                raise AssertionError("Fragment allocations do not conserve parent work")
        complete = all(f.finish_s is not None for f in fragments)
        if complete != (parent.finish_s is not None):
            raise AssertionError("Parent completion differs from positive fragments")
        if complete and parent.finish_s != max(f.finish_s for f in fragments):
            raise AssertionError("Parent did not finish with its last fragment")
    return errors


def _fragment_record(env, fragment):
    events = [row for row in env.ledger
              if row.get("fragment_id") == fragment.fragment_id]
    cpu = [row for row in events if row["kind"] == "cpu_service"]
    transfer = [row for row in events if row["kind"] == "u2u_service"]
    completed_transfer = [row for row in events if row["kind"] == "u2u_complete"]
    route_time = env.parents[fragment.parent_id].route_time_s
    remote = fragment.source != fragment.destination
    transfer_complete = completed_transfer[0]["time_s"] if completed_transfer else None
    cpu_enter = transfer_complete if remote else route_time
    first_cpu = min((row["start_s"] for row in cpu), default=None)
    first_transfer = min((row["start_s"] for row in transfer), default=None)
    last_transfer_service = max(
        (row["start_s"] + row["duration_s"] for row in transfer), default=None)
    cpu_duration = sum(row["duration_s"] for row in cpu)
    observed_end = env.time_s if fragment.finish_s is None else fragment.finish_s
    cpu_elapsed = None if cpu_enter is None else observed_end - cpu_enter
    return dict(
        fragment_id=fragment.fragment_id, parent_id=fragment.parent_id,
        source=fragment.source, destination=fragment.destination,
        ratio=fragment.input_bits / TARGET_BITS,
        bits=fragment.input_bits, cycles=fragment.compute_cycles,
        finish_s=fragment.finish_s, censored=fragment.finish_s is None,
        transfer_complete_s=transfer_complete,
        transfer_service_complete_s=last_transfer_service,
        transfer_first_service_s=first_transfer,
        transfer_wait_to_first_service_s=(
            None if first_transfer is None else first_transfer - route_time),
        transfer_stage_elapsed_s=(
            (env.time_s if transfer_complete is None else transfer_complete) - route_time
            if remote else 0.0),
        transfer_service_duration_s=sum(row["duration_s"] for row in transfer),
        cpu_enter_s=cpu_enter, cpu_first_service_s=first_cpu,
        cpu_wait_to_first_service_s=(
            None if first_cpu is None or cpu_enter is None else first_cpu - cpu_enter),
        cpu_stage_elapsed_s=cpu_elapsed, cpu_service_duration_s=cpu_duration,
        cpu_nonservice_elapsed_s=(
            None if cpu_elapsed is None else max(0.0, cpu_elapsed - cpu_duration)),
        cpu_nonservice_note="Includes any grid rounding and service gaps; not pure FIFO wait",
        remaining_transfer_bits=fragment.transfer_remaining,
        remaining_compute_cycles=fragment.compute_remaining,
        cpu_energy_j=sum(row["energy_j"] for row in cpu),
        u2u_energy_j=sum(row["energy_j"] for row in transfer),
    )


def _case(scene, condition, action_name, allocation):
    env = HotspotIndustrialEnv(scene, ledger_enabled=True)
    _empty_arrival_tape(env)
    local = np.tile(np.eye(3), (2, 1, 1)).astype(np.float64)
    background_ids = []
    if condition != "idle":
        source = 0 if condition == "source_busy" else 1
        background_ids = [_admit_ready(env, source=source, priority=0)
                          for _ in range(BACKGROUND_PARENTS)]
    target_id = _admit_ready(env, source=0, priority=1)
    if tuple(env.ready) != tuple(background_ids + [target_id]):
        raise AssertionError("Initial ready FIFO must put the target after background")
    # Routing initializes CPU backlog; all assigned cycles remain unserved.
    for pid in background_ids:
        if env.ready.popleft() != pid:
            raise AssertionError("Background ready order changed")
    env._route(tuple(background_ids), local)
    if env.cpu_served_cycles != 0.0 or env.tx_served_bits != 0.0:
        raise AssertionError("Initial-state construction performed physical service")
    if any(env.parents[pid].finish_s is not None for pid in background_ids):
        raise AssertionError("Background work was silently completed")
    env._publish("mechanism_initial_ready")
    initial_cpu = [
        [sum(env.fragments[fid].compute_remaining for fid in env.compute[p][node])
         for node in range(3)] for p in range(2)]
    initial_errors = _conservation(env)
    maximum_errors = {key: abs(value) for key, value in initial_errors.items()}
    route = local.copy()
    route[1, 0] = np.asarray(allocation, dtype=np.float64)
    count = round(WINDOW_SECONDS / env.dt)
    if count != 50 or not math.isclose(count * env.dt, WINDOW_SECONDS):
        raise ValueError("This mechanism design requires the reviewed 0.1-second clock")
    readings = []
    for _ in range(count):
        if env.step_index % env.macro_steps == 0:
            env.start_motion(np.zeros(6))
        if env.step_index % env.frame_steps == 0:
            env.prepare_frame()
        readings.append(env.micro_step(route))
        for key, value in _conservation(env).items():
            maximum_errors[key] = max(maximum_errors[key], abs(value))
    target = env.parents[target_id]
    children = [_fragment_record(env, env.fragments[fid])
                for fid in target.fragment_ids]
    completed_children = [row for row in children if row["finish_s"] is not None]
    tails = [row["fragment_id"] for row in completed_children
             if target.finish_s is not None and row["finish_s"] == target.finish_s]
    first_child = min((row["finish_s"] for row in completed_children), default=None)
    target_residence = env.time_s if target.finish_s is None else target.finish_s
    holding_from_parents = sum(
        env.config["objective"]["priority_weights"][parent.arrival.priority] *
        ((env.time_s if parent.finish_s is None else parent.finish_s) - parent.arrival.birth_s)
        for parent in env.parents.values())
    if not math.isclose(holding_from_parents, env.holding, rel_tol=1e-11, abs_tol=1e-8):
        raise AssertionError("Mechanism parent holding identity failed")
    cost = sum(row["cost"] for row in readings)
    if not math.isclose(cost, .09 * env.holding + sum(env.energy.values()) / 140.,
                        rel_tol=1e-11, abs_tol=1e-8):
        raise AssertionError("Mechanism physical cost identity failed")
    return dict(
        condition=condition, action=action_name, allocation=list(allocation),
        window_seconds=WINDOW_SECONDS, physical_steps=count,
        initial_state="Already-uploaded parents; birth=0 is the observation origin",
        excluded_prehistory="Upload delay and upload energy before this controlled ready state",
        task_size_profile="Production low-priority vision size; background priority explicitly set high",
        future_arrival_tape=[], target_id=target_id, background_ids=background_ids,
        task_bits=TARGET_BITS, task_cycles=TARGET_CYCLES,
        initial_uav_positions_m=env.config["uav_initial_positions_m"],
        initial_cpu_cycles_by_priority_destination=initial_cpu,
        target_completed=target.finish_s is not None, target_censored=target.finish_s is None,
        target_finish_s=target.finish_s,
        target_ready_to_complete_s=target.finish_s,
        target_cutoff_age_s=target_residence if target.finish_s is None else None,
        target_holding_task_s=target_residence,
        target_tail_fragment_ids=tails,
        target_tail_destinations=[env.fragments[fid].destination for fid in tails],
        target_first_child_finish_s=first_child,
        target_tail_gap_s=(
            None if target.finish_s is None or first_child is None
            else target.finish_s - first_child),
        children=children, holding_task_s=env.holding,
        holding_identity_error=holding_from_parents - env.holding,
        energy_j=dict(env.energy), cost_sum=cost, cost_per_s=cost / WINDOW_SECONDS,
        target_cpu_energy_j=sum(row["cpu_energy_j"] for row in children),
        target_u2u_energy_j=sum(row["u2u_energy_j"] for row in children),
        target_energy_note="Ledger allocation at actual aggregate CPU frequency; not marginal counterfactual energy",
        completed_parents=sum(parent.finish_s is not None for parent in env.parents.values()),
        unfinished_parents=len(env._active_parents),
        parent_completion_event_counts={
            pid: sum(row["kind"] == "parent_complete" and row.get("parent_id") == pid
                     for row in env.ledger) for pid in env.parents},
        initial_conservation_errors=initial_errors,
        conservation_errors=_conservation(env),
        maximum_absolute_conservation_errors=maximum_errors,
        generated_bits=env.generated_bits, initial_uploaded_bits=env.upload_served_bits,
        generated_cycles=env.generated_cycles, cpu_served_cycles=env.cpu_served_cycles,
        u2u_served_bits=env.tx_served_bits,
        terminated=False, truncated=True, research_training_updates=0,
    )


def run_mechanisms():
    """Return nine fixed-window cases; caller instruments the total 450 ticks.

    No rates, CPU parameters, geometry, objective weights, or service methods
    are replaced. Only the external arrival tape is fixed to empty after the
    disclosed initial ready-state workload. Background always routes locally.
    """
    scene = json.loads(default_config().with_name("physical_scene.json").read_text(encoding="utf-8"))
    profiles = scene["base_scene"]["task_types"]["vision"]
    if (profiles["input_bits"][1] != TARGET_BITS or
            profiles["input_bits"][1] * profiles["cycles_per_bit"][1] != TARGET_CYCLES):
        raise ValueError("Production vision task size changed; review the mechanism design")
    actions = (
        ("local", (1.0, 0.0, 0.0)),
        ("equal", (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)),
        ("remote1", (0.0, 1.0, 0.0)),
    )
    return [_case(scene, condition, name, allocation)
            for condition in ("idle", "source_busy", "destination_busy")
            for name, allocation in actions]
