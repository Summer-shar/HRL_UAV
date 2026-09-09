"""Reference implementation of Xu et al. (TCOM 2024), Algorithm 1.

The paper's short-timescale optimizer jointly allocates per-task inter-UAV
offloading ratios and per-task CPU frequencies.  This module implements the
paper's one-frame problem independently from the queueing environment so that
the reference method is not confused with the route-only projected optimizer
in ``scripts/compare_sca.py``.

Frequencies are represented in GHz and task sizes in Mbit for numerical
conditioning.  Reported delays and energies retain SI units (seconds/joules).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cvxpy as cp
import numpy as np


@dataclass(frozen=True)
class Xu2024SCAInput:
    """One short-timescale frame of the paper's resource-allocation problem."""

    task_mbits: np.ndarray
    cycles_per_bit: np.ndarray | float
    source_uav: np.ndarray
    u2g_rate_mbps: np.ndarray
    u2u_rate_mbps: np.ndarray
    f_max_ghz: np.ndarray
    capacitance_coeff: np.ndarray | float
    frame_seconds: float
    gt_tx_power_w: float
    uav_tx_power_w: float
    delay_weight: float
    energy_weight: float
    task_delay_weights: Optional[np.ndarray] = None
    route_wait_cost_s: Optional[np.ndarray] = None
    u2u_capacity_mbits: Optional[np.ndarray] = None
    enforce_service_stability: bool = False
    propulsion_energy_j: float = 0.0


@dataclass(frozen=True)
class Xu2024SCASolution:
    rho: np.ndarray
    cpu_ghz: np.ndarray
    completion_time_s: np.ndarray
    objective: float
    delay_sum_s: float
    energy_sum_j: float
    iterations: int
    status: str
    converged: bool
    objective_history: np.ndarray
    max_change_history: np.ndarray


def nearest_uav_association(
    user_coords: np.ndarray,
    uav_coords: np.ndarray,
) -> np.ndarray:
    """Implement the greedy nearest-UAV association in paper equation (30)."""

    users = np.asarray(user_coords, dtype=np.float64)
    uavs = np.asarray(uav_coords, dtype=np.float64)
    if users.ndim != 2 or users.shape[1] != 2:
        raise ValueError("user_coords must have shape (K, 2)")
    if uavs.ndim != 2 or uavs.shape[1] != 2:
        raise ValueError("uav_coords must have shape (M, 2)")
    horizontal_sq = np.sum(
        (users[:, None, :] - uavs[None, :, :]) ** 2,
        axis=2,
    )
    return np.argmin(horizontal_sq, axis=1).astype(np.int64)


def _broadcast_task_vector(value, task_count: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 0:
        result = np.full(task_count, float(result), dtype=np.float64)
    if result.shape != (task_count,):
        raise ValueError(f"{name} must be scalar or have shape ({task_count},)")
    return result


def _broadcast_uav_vector(value, uav_count: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 0:
        result = np.full(uav_count, float(result), dtype=np.float64)
    if result.shape != (uav_count,):
        raise ValueError(f"{name} must be scalar or have shape ({uav_count},)")
    return result


def _validated_input(problem: Xu2024SCAInput) -> dict[str, np.ndarray | float]:
    task_mbits = np.asarray(problem.task_mbits, dtype=np.float64).reshape(-1)
    task_count = int(task_mbits.size)
    if task_count == 0 or np.any(task_mbits <= 0.0):
        raise ValueError("task_mbits must contain at least one positive task")

    source_uav = np.asarray(problem.source_uav, dtype=np.int64).reshape(-1)
    if source_uav.shape != (task_count,):
        raise ValueError("source_uav must contain one association per task")

    u2u_rate_mbps = np.asarray(problem.u2u_rate_mbps, dtype=np.float64)
    if u2u_rate_mbps.ndim != 2 or u2u_rate_mbps.shape[0] != u2u_rate_mbps.shape[1]:
        raise ValueError("u2u_rate_mbps must be a square matrix")
    uav_count = int(u2u_rate_mbps.shape[0])
    if np.any(source_uav < 0) or np.any(source_uav >= uav_count):
        raise ValueError("source_uav contains an invalid UAV index")

    u2g_rate_mbps = np.asarray(problem.u2g_rate_mbps, dtype=np.float64).reshape(-1)
    if u2g_rate_mbps.shape != (task_count,) or np.any(u2g_rate_mbps <= 0.0):
        raise ValueError("u2g_rate_mbps must be positive with one value per task")

    off_diagonal = ~np.eye(uav_count, dtype=bool)
    if np.any(u2u_rate_mbps[off_diagonal] <= 0.0):
        raise ValueError("all off-diagonal U2U rates must be positive")

    f_max_ghz = _broadcast_uav_vector(problem.f_max_ghz, uav_count, "f_max_ghz")
    if np.any(f_max_ghz <= 0.0):
        raise ValueError("f_max_ghz must be positive")
    cycles_per_bit = _broadcast_task_vector(
        problem.cycles_per_bit,
        task_count,
        "cycles_per_bit",
    )
    if np.any(cycles_per_bit <= 0.0):
        raise ValueError("cycles_per_bit must be positive")
    capacitance = _broadcast_uav_vector(
        problem.capacitance_coeff,
        uav_count,
        "capacitance_coeff",
    )
    if np.any(capacitance < 0.0):
        raise ValueError("capacitance_coeff must be non-negative")

    task_delay_weights = (
        np.ones(task_count, dtype=np.float64)
        if problem.task_delay_weights is None
        else _broadcast_task_vector(
            problem.task_delay_weights,
            task_count,
            "task_delay_weights",
        )
    )
    if np.any(task_delay_weights < 0.0):
        raise ValueError("task_delay_weights must be non-negative")
    route_wait_cost_s = (
        np.zeros((task_count, uav_count), dtype=np.float64)
        if problem.route_wait_cost_s is None
        else np.asarray(problem.route_wait_cost_s, dtype=np.float64)
    )
    if route_wait_cost_s.shape != (task_count, uav_count):
        raise ValueError(
            "route_wait_cost_s must have shape "
            f"({task_count}, {uav_count})"
        )
    if (
        np.any(~np.isfinite(route_wait_cost_s))
        or np.any(route_wait_cost_s < 0.0)
    ):
        raise ValueError("route_wait_cost_s must be finite and non-negative")
    u2u_capacity_mbits = (
        None
        if problem.u2u_capacity_mbits is None
        else np.asarray(problem.u2u_capacity_mbits, dtype=np.float64)
    )
    if u2u_capacity_mbits is not None:
        if u2u_capacity_mbits.shape != (uav_count, uav_count):
            raise ValueError(
                "u2u_capacity_mbits must have shape "
                f"({uav_count}, {uav_count})"
            )
        if (
            np.any(~np.isfinite(u2u_capacity_mbits))
            or np.any(u2u_capacity_mbits < 0.0)
        ):
            raise ValueError(
                "u2u_capacity_mbits must be finite and non-negative"
            )
    if problem.frame_seconds <= 0.0:
        raise ValueError("frame_seconds must be positive")
    if problem.delay_weight < 0.0 or problem.energy_weight < 0.0:
        raise ValueError("objective weights must be non-negative")

    # Mbit * cycles/bit / GHz = seconds after this 1e-3 conversion.
    workload_gcycles = task_mbits * cycles_per_bit * 1e-3
    # phi * 1e27 converts workload[Gcycle] * f[GHz]^2 to joules.
    capacitance_scaled = capacitance * 1e27
    return {
        "task_mbits": task_mbits,
        "source_uav": source_uav,
        "u2g_rate_mbps": u2g_rate_mbps,
        "u2u_rate_mbps": u2u_rate_mbps,
        "f_max_ghz": f_max_ghz,
        "workload_gcycles": workload_gcycles,
        "capacitance_scaled": capacitance_scaled,
        "task_delay_weights": task_delay_weights,
        "route_wait_cost_s": route_wait_cost_s,
        "u2u_capacity_mbits": u2u_capacity_mbits,
        "task_count": task_count,
        "uav_count": uav_count,
    }


def _initial_point(data: dict[str, np.ndarray | float]) -> tuple[np.ndarray, np.ndarray]:
    task_count = int(data["task_count"])
    uav_count = int(data["uav_count"])
    source_uav = np.asarray(data["source_uav"], dtype=np.int64)
    workload = np.asarray(data["workload_gcycles"], dtype=np.float64)
    f_max = np.asarray(data["f_max_ghz"], dtype=np.float64)

    if uav_count == 1:
        rho = np.ones((task_count, 1), dtype=np.float64)
    else:
        rho = np.full(
            (task_count, uav_count),
            0.2 / (uav_count - 1),
            dtype=np.float64,
        )
        rho[np.arange(task_count), source_uav] = 0.8

    cpu = np.zeros((task_count, uav_count), dtype=np.float64)
    for destination in range(uav_count):
        demand = workload * rho[:, destination]
        demand += max(float(np.mean(demand)), 1e-6) * 1e-3
        cpu[:, destination] = f_max[destination] * demand / np.sum(demand)
    return rho, cpu


def _actual_metrics(
    problem: Xu2024SCAInput,
    data: dict[str, np.ndarray | float],
    rho: np.ndarray,
    cpu_ghz: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    task_mbits = np.asarray(data["task_mbits"], dtype=np.float64)
    workload = np.asarray(data["workload_gcycles"], dtype=np.float64)
    source = np.asarray(data["source_uav"], dtype=np.int64)
    u2g_rate = np.asarray(data["u2g_rate_mbps"], dtype=np.float64)
    u2u_rate = np.asarray(data["u2u_rate_mbps"], dtype=np.float64)
    capacitance = np.asarray(data["capacitance_scaled"], dtype=np.float64)
    task_weights = np.asarray(data["task_delay_weights"], dtype=np.float64)
    route_wait = np.asarray(data["route_wait_cost_s"], dtype=np.float64)
    u2u_capacity = data["u2u_capacity_mbits"]
    task_count = int(data["task_count"])
    uav_count = int(data["uav_count"])

    upload = task_mbits / u2g_rate
    compute = workload[:, None] * rho / np.maximum(cpu_ghz, 1e-12)
    branch = compute + route_wait * rho
    transmit_energy = 0.0
    for task in range(task_count):
        for destination in range(uav_count):
            if destination == source[task]:
                continue
            transfer = (
                task_mbits[task]
                * rho[task, destination]
                / u2u_rate[source[task], destination]
            )
            branch[task, destination] += transfer
            transmit_energy += float(problem.uav_tx_power_w) * transfer
    completion = upload + np.max(branch, axis=1)
    upload_energy = float(problem.gt_tx_power_w) * float(np.sum(upload))
    compute_energy = float(
        np.sum(
            workload[:, None]
            * rho
            * np.square(cpu_ghz)
            * capacitance[None, :]
        )
    )
    delay_sum = float(np.dot(task_weights, completion))
    energy_sum = (
        upload_energy
        + transmit_energy
        + compute_energy
        + float(problem.propulsion_energy_j)
    )
    objective = (
        float(problem.delay_weight) * delay_sum
        + float(problem.energy_weight) * energy_sum
    )
    return completion, delay_sum, energy_sum, objective


def solve_xu2024_sca(
    problem: Xu2024SCAInput,
    *,
    rho_init: Optional[np.ndarray] = None,
    cpu_init_ghz: Optional[np.ndarray] = None,
    max_iterations: int = 10,
    tolerance: float = 0.01,
    proximal_rho: float = 1e-3,
    proximal_cpu: float = 1e-3,
    enforce_frame_deadline: bool = True,
    solver: str = "CLARABEL",
    verbose: bool = False,
) -> Xu2024SCASolution:
    """Solve the paper's greedy-association + SCA/CVX resource subproblem.

    Equations (31)-(33) are implemented as tangent convex surrogates.  The
    paper only specifies that the two proximal constants are positive; their
    numerical values are therefore explicit arguments and part of the result
    provenance rather than hidden implementation defaults.
    """

    data = _validated_input(problem)
    task_count = int(data["task_count"])
    uav_count = int(data["uav_count"])
    source = np.asarray(data["source_uav"], dtype=np.int64)
    task_mbits = np.asarray(data["task_mbits"], dtype=np.float64)
    workload = np.asarray(data["workload_gcycles"], dtype=np.float64)
    u2g_rate = np.asarray(data["u2g_rate_mbps"], dtype=np.float64)
    u2u_rate = np.asarray(data["u2u_rate_mbps"], dtype=np.float64)
    f_max = np.asarray(data["f_max_ghz"], dtype=np.float64)
    capacitance = np.asarray(data["capacitance_scaled"], dtype=np.float64)
    task_weights = np.asarray(data["task_delay_weights"], dtype=np.float64)
    route_wait = np.asarray(data["route_wait_cost_s"], dtype=np.float64)
    u2u_capacity = data["u2u_capacity_mbits"]

    default_rho, default_cpu = _initial_point(data)
    rho_anchor = (
        default_rho
        if rho_init is None
        else np.asarray(rho_init, dtype=np.float64).reshape(task_count, uav_count)
    )
    cpu_anchor = (
        default_cpu
        if cpu_init_ghz is None
        else np.asarray(cpu_init_ghz, dtype=np.float64).reshape(task_count, uav_count)
    )
    rho_anchor = np.maximum(rho_anchor, 0.0)
    rho_anchor /= np.maximum(np.sum(rho_anchor, axis=1, keepdims=True), 1e-12)
    cpu_floor = min(
        1e-4,
        float(np.min(f_max)) / max(10.0 * task_count, 1.0),
    )
    cpu_anchor = np.maximum(cpu_anchor, cpu_floor)
    for destination in range(uav_count):
        total = float(np.sum(cpu_anchor[:, destination]))
        if total > f_max[destination]:
            cpu_anchor[:, destination] *= f_max[destination] / total

    upload_delay = task_mbits / u2g_rate
    upload_energy = float(problem.gt_tx_power_w) * float(np.sum(upload_delay))
    objective_history: list[float] = []
    max_change_history: list[float] = []
    last_status = "not-solved"
    converged = False

    for _ in range(max(int(max_iterations), 1)):
        rho = cp.Variable((task_count, uav_count), nonneg=True)
        cpu = cp.Variable((task_count, uav_count))
        completion_tail = cp.Variable(task_count, nonneg=True)
        constraints = [
            cp.sum(rho, axis=1) == 1.0,
            cpu >= cpu_floor,
        ]
        for destination in range(uav_count):
            constraints.append(
                cp.sum(cpu[:, destination]) <= f_max[destination]
            )
            if bool(problem.enforce_service_stability):
                constraints.append(
                    workload @ rho[:, destination]
                    <= float(problem.frame_seconds)
                    * cp.sum(cpu[:, destination])
                )
        if u2u_capacity is not None:
            capacity = np.asarray(u2u_capacity, dtype=np.float64)
            for origin in range(uav_count):
                source_mask = (source == origin).astype(np.float64)
                source_work = task_mbits * source_mask
                for destination in range(uav_count):
                    if destination == origin:
                        continue
                    constraints.append(
                        source_work @ rho[:, destination]
                        <= capacity[origin, destination]
                    )

        compute_upper: list[list[cp.Expression]] = []
        compute_energy_terms: list[cp.Expression] = []
        transfer_energy_terms: list[cp.Expression] = []
        for task in range(task_count):
            task_compute_upper = []
            for destination in range(uav_count):
                rho0 = float(rho_anchor[task, destination])
                f0 = max(float(cpu_anchor[task, destination]), cpu_floor)
                # Equation (32): convex term plus the first-order upper
                # approximation of the concave part in equation (31).
                product_upper = (
                    0.5 * cp.square(rho[task, destination] + cp.inv_pos(cpu[task, destination]))
                    - 0.5 * rho0**2
                    - 0.5 / f0**2
                    - rho0 * (rho[task, destination] - rho0)
                    + (cpu[task, destination] - f0) / f0**3
                )
                branch_compute = workload[task] * product_upper
                task_compute_upper.append(branch_compute)

                # Equation (33), with the anchor constant restored so the
                # surrogate is value- and gradient-consistent at the anchor.
                energy_product_upper = (
                    rho0 * cp.square(cpu[task, destination])
                    + rho[task, destination] * f0**2
                    - rho0 * f0**2
                    + 0.5
                    * float(proximal_rho)
                    * cp.square(rho[task, destination] - rho0)
                    + 0.5
                    * float(proximal_cpu)
                    * cp.square(cpu[task, destination] - f0)
                )
                compute_energy_terms.append(
                    workload[task]
                    * capacitance[destination]
                    * energy_product_upper
                )

                if destination == source[task]:
                    constraints.append(
                        completion_tail[task]
                        >= branch_compute
                        + route_wait[task, destination]
                        * rho[task, destination]
                    )
                else:
                    transfer = (
                        task_mbits[task]
                        / u2u_rate[source[task], destination]
                        * rho[task, destination]
                    )
                    constraints.append(
                        completion_tail[task]
                        >= transfer
                        + branch_compute
                        + route_wait[task, destination]
                        * rho[task, destination]
                    )
                    transfer_energy_terms.append(
                        float(problem.uav_tx_power_w) * transfer
                    )
            compute_upper.append(task_compute_upper)
            if enforce_frame_deadline:
                constraints.append(
                    upload_delay[task] + completion_tail[task]
                    <= float(problem.frame_seconds)
                )

        weighted_delay = cp.sum(
            cp.multiply(task_weights, upload_delay + completion_tail)
        )
        variable_energy = cp.sum(cp.hstack(compute_energy_terms))
        if transfer_energy_terms:
            variable_energy += cp.sum(cp.hstack(transfer_energy_terms))
        total_energy = (
            variable_energy
            + upload_energy
            + float(problem.propulsion_energy_j)
        )
        objective = (
            float(problem.delay_weight) * weighted_delay
            + float(problem.energy_weight) * total_energy
        )
        cvx_problem = cp.Problem(cp.Minimize(objective), constraints)
        if not cvx_problem.is_dcp():
            raise RuntimeError("Xu2024 SCA subproblem is not DCP")
        cvx_problem.solve(solver=solver, verbose=verbose)
        last_status = str(cvx_problem.status)
        if last_status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise RuntimeError(
                f"Xu2024 SCA subproblem failed with status {last_status!r}"
            )

        rho_value = np.maximum(np.asarray(rho.value, dtype=np.float64), 0.0)
        rho_value /= np.maximum(
            np.sum(rho_value, axis=1, keepdims=True),
            1e-12,
        )
        cpu_value = np.maximum(
            np.asarray(cpu.value, dtype=np.float64),
            cpu_floor,
        )
        completion, delay_sum, energy_sum, actual_objective = _actual_metrics(
            problem,
            data,
            rho_value,
            cpu_value,
        )
        rho_change = float(np.max(np.abs(rho_value - rho_anchor)))
        cpu_change = float(
            np.max(
                np.abs(cpu_value - cpu_anchor)
                / np.maximum(f_max[None, :], 1e-12)
            )
        )
        max_change = max(rho_change, cpu_change)
        objective_history.append(float(actual_objective))
        max_change_history.append(max_change)
        rho_anchor = rho_value
        cpu_anchor = cpu_value
        if max_change <= float(tolerance):
            converged = True
            break

    completion, delay_sum, energy_sum, actual_objective = _actual_metrics(
        problem,
        data,
        rho_anchor,
        cpu_anchor,
    )
    return Xu2024SCASolution(
        rho=rho_anchor.astype(np.float64),
        cpu_ghz=cpu_anchor.astype(np.float64),
        completion_time_s=completion.astype(np.float64),
        objective=float(actual_objective),
        delay_sum_s=float(delay_sum),
        energy_sum_j=float(energy_sum),
        iterations=len(objective_history),
        status=last_status,
        converged=bool(converged),
        objective_history=np.asarray(objective_history, dtype=np.float64),
        max_change_history=np.asarray(max_change_history, dtype=np.float64),
    )
