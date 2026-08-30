"""Xu et al. (TCOM 2024) Algorithm-1 baseline for this repository.

The mathematical solver is the paper-faithful one-frame implementation in
``hrl_uav.xu2024_sca``.  This script supplies an explicitly adapted execution
bridge for the repository's persistent-queue, two-priority environment:

* the environment's fixed source association is retained for comparability;
* per-task paper decisions are aggregated into source/priority route rows;
* per-task CPU frequencies are aggregated into high/low UAV frequencies; and
* the paper's within-frame deadline is optional because queued tasks in this
  environment are allowed to carry across frames.

These adaptations are written to the result metadata and must not be described
as a bit-exact reproduction of the paper's simulation model.
"""

from __future__ import annotations

import argparse
import time

from cvxpy.error import SolverError
import numpy as np

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap

bootstrap()

from compare_common import add_common_args
from compare_common import run_fixed_policy_benchmark
from compare_sca import centroid_chase_macro_policy
from hrl_uav.routing import project_rows_to_simplex
from hrl_uav.xu2024_sca import Xu2024SCAInput
from hrl_uav.xu2024_sca import solve_xu2024_sca


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Xu et al. 2024 Algorithm-1 SCA/CVX baseline adapted to the "
            "repository queueing environment"
        ),
    )
    add_common_args(parser, default_results_subdir="xu2024_sca_adapted")
    parser.add_argument("--xu_sca_max_iter", type=int, default=10)
    parser.add_argument("--xu_sca_tol", type=float, default=0.01)
    parser.add_argument("--xu_sca_proximal_rho", type=float, default=1e-3)
    parser.add_argument("--xu_sca_proximal_cpu", type=float, default=1e-3)
    parser.add_argument(
        "--xu_sca_warm_start",
        type=int,
        choices=[0, 1],
        default=0,
        help="Initialize each frame from the previous feasible SCA solution.",
    )
    parser.add_argument(
        "--xu_sca_task_aggregation",
        choices=["per_task", "source_priority"],
        default="source_priority",
        help=(
            "per_task preserves Algorithm 1 variables before the execution "
            "bridge; source_priority solves the six route rows that the "
            "current environment can actually execute."
        ),
    )
    parser.add_argument(
        "--xu_sca_enforce_frame_deadline",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "Use paper constraint D_k<=delta. Keep disabled for the current "
            "persistent-queue model; enable only in a paper-matched scenario."
        ),
    )
    parser.add_argument(
        "--xu_sca_use_priority_delay_weights",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "Weight high/low virtual tasks by the current environment's "
            "priority weights. Set 0 for the paper's unprioritized task sum."
        ),
    )
    parser.add_argument(
        "--xu_sca_queue_aware",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "Add causal compute-queue and U2U-queue waiting times to the "
            "adapted one-frame objective. This is disabled for the native "
            "paper baseline and enabled only for the persistent-queue "
            "adapter."
        ),
    )
    parser.set_defaults(
        episodes=1,
        eval_interval=0,
        eval_episodes=10,
        trajectory_plot_interval=0,
        micro_solver="xu_sca",
    )
    return parser.parse_args()


def configure_xu2024_wrapper(env, args):
    env.xu_sca_max_iter = int(args.xu_sca_max_iter)
    env.xu_sca_tol = float(args.xu_sca_tol)
    env.xu_sca_proximal_rho = float(args.xu_sca_proximal_rho)
    env.xu_sca_proximal_cpu = float(args.xu_sca_proximal_cpu)
    env.xu_sca_warm_start = bool(
        int(getattr(args, "xu_sca_warm_start", 0))
    )
    env.xu_sca_task_aggregation = str(
        getattr(args, "xu_sca_task_aggregation", "source_priority")
    )
    env.xu_sca_enforce_frame_deadline = bool(
        int(args.xu_sca_enforce_frame_deadline)
    )
    env.xu_sca_use_priority_delay_weights = bool(
        int(args.xu_sca_use_priority_delay_weights)
    )
    env.xu_sca_queue_aware = bool(
        int(getattr(args, "xu_sca_queue_aware", 0))
    )
    env.xu_sca_solver = str(getattr(args, "sca_solver", "CLARABEL"))
    env.xu_sca_delay_weight = float(
        getattr(
            args,
            "sca_delay_weight",
            getattr(args, "delay_objective_weight", 0.3),
        )
    )
    env.xu_sca_energy_weight = float(
        getattr(
            args,
            "sca_energy_weight",
            getattr(args, "energy_objective_weight", 0.7),
        )
    )


def _current_tasks(base_env):
    tasks = []
    for source_task_ids in base_env.current_arrival_tasks_by_uav:
        for task_id in source_task_ids:
            parent = base_env.parent_tasks.get(task_id)
            if parent is not None and parent.size_mbits > 0.0:
                tasks.append(parent)
    return tasks


def _expected_pseudo_tasks(base_env):
    """Fallback only for a resource frame with no sampled task arrivals."""

    tasks = []
    horizon = float(base_env.resource_frame_dt)
    for user_id in range(base_env.M):
        source = int(base_env.user_allocations[user_id])
        for priority, rate, size in (
            (
                "high",
                float(base_env.user_lambda_high[user_id]),
                float(base_env.user_task_size_high_mbits[user_id]),
            ),
            (
                "low",
                float(base_env.user_lambda_low[user_id]),
                float(base_env.user_task_size_low_mbits[user_id]),
            ),
        ):
            expected_mbits = rate * size * horizon
            if expected_mbits <= 1e-9:
                continue
            tasks.append(
                {
                    "user_id": user_id,
                    "source_uav": source,
                    "priority": priority,
                    "size_mbits": expected_mbits,
                }
            )
    return tasks


def _task_field(task, name):
    return task[name] if isinstance(task, dict) else getattr(task, name)


def _aggregate_source_priority_tasks(
    priorities,
    source_uav,
    task_mbits,
    u2g_rate_mbps,
    num_uavs,
):
    aggregated_priorities = []
    aggregated_sources = []
    aggregated_mbits = []
    aggregated_u2g_rates = []
    for priority in ("high", "low"):
        for source in range(num_uavs):
            mask = (priorities == priority) & (source_uav == source)
            if not np.any(mask):
                continue
            bits = np.asarray(task_mbits[mask], dtype=np.float64)
            rates = np.asarray(u2g_rate_mbps[mask], dtype=np.float64)
            total_bits = float(np.sum(bits))
            # Preserve the exact sum of individual upload times:
            # L_group / R_eff = sum_k L_k / R_k.
            effective_rate = total_bits / max(
                float(np.sum(bits / np.maximum(rates, 1e-12))),
                1e-12,
            )
            aggregated_priorities.append(priority)
            aggregated_sources.append(source)
            aggregated_mbits.append(total_bits)
            aggregated_u2g_rates.append(effective_rate)
    return (
        np.asarray(aggregated_priorities, dtype=object),
        np.asarray(aggregated_sources, dtype=np.int64),
        np.asarray(aggregated_mbits, dtype=np.float64),
        np.asarray(aggregated_u2g_rates, dtype=np.float64),
    )


def _queue_wait_costs_s(
    base_env,
    priorities,
    source_uav,
    u2u_rate_mbps,
):
    """Causal fluid waiting-time estimate for the persistent-queue adapter.

    High-priority work waits behind existing high-priority work. Low-priority
    work waits behind both queues because the environment serves high traffic
    first. Existing U2U chunks cannot be rerouted; their waiting time is used
    only as a congestion price for newly admitted work.
    """

    priorities = np.asarray(priorities, dtype=object)
    source_uav = np.asarray(source_uav, dtype=np.int64)
    rates = np.asarray(u2u_rate_mbps, dtype=np.float64)
    f_max_hz = np.asarray(base_env.F_max_array, dtype=np.float64)
    compute_rate_mbps = f_max_hz / (float(base_env.C) * 1e6)
    q_loc_high = np.asarray(base_env.Q_loc_high, dtype=np.float64)
    q_loc_low = np.asarray(base_env.Q_loc_low, dtype=np.float64)
    q_tx_high = np.asarray(base_env.Q_tx_high, dtype=np.float64)
    q_tx_low = np.asarray(base_env.Q_tx_low, dtype=np.float64)

    wait = np.zeros((priorities.size, base_env.N), dtype=np.float64)
    for task_index, (priority, source) in enumerate(
        zip(priorities, source_uav)
    ):
        if priority == "high":
            compute_queue_ahead = q_loc_high
            tx_queue_ahead = q_tx_high[int(source)]
        else:
            compute_queue_ahead = q_loc_high + q_loc_low
            tx_queue_ahead = (
                q_tx_high[int(source)] + q_tx_low[int(source)]
            )
        wait[task_index] = compute_queue_ahead / np.maximum(
            compute_rate_mbps,
            1e-12,
        )
        for destination in range(base_env.N):
            if destination != int(source):
                wait[task_index, destination] += (
                    tx_queue_ahead[destination]
                    / max(rates[int(source), destination], 1e-12)
                )
    return wait


def _route_to_action(env, route):
    route = np.asarray(route, dtype=np.float64).reshape(2, env.N, env.N)
    diagonal_bias = np.eye(env.N, dtype=np.float64)[None, ...]
    diagonal_bias *= float(env.micro_diag_bias)
    scale = max(float(env.micro_action_scale), 1e-12)
    action = np.clip((route - diagonal_bias) / scale, -1.0, 1.0)
    reconstructed = project_rows_to_simplex(
        action * scale + diagonal_bias
    )
    if not np.allclose(reconstructed, route, atol=2e-6):
        raise RuntimeError(
            "Xu2024 route cannot be represented by the configured micro "
            "action mapping; use micro_diag_bias<=1 and action_scale>=1."
        )
    return action.reshape(-1).astype(np.float32)


def _previous_frame_initial_point(
    env,
    priorities,
    source_uav,
    task_mbits,
):
    if not bool(getattr(env, "xu_sca_warm_start", True)):
        return None, None
    if not hasattr(env, "_xu2024_last_route"):
        return None, None

    route = np.asarray(env._xu2024_last_route, dtype=np.float64)
    if route.shape != (2, env.N, env.N):
        return None, None
    rho_init = np.empty((task_mbits.size, env.N), dtype=np.float64)
    for task_index, (priority, source) in enumerate(
        zip(priorities, source_uav)
    ):
        priority_index = 0 if priority == "high" else 1
        rho_init[task_index] = route[priority_index, int(source)]

    # Task identities and counts change every frame. Reusing an aggregated
    # high/low CPU allocation as a per-task initial point is numerically
    # fragile, so only the feasible route is carried forward. The reference
    # solver rebuilds a demand-proportional CPU initial point for this frame.
    return rho_init, None


def solve_xu2024_micro_action(env, deterministic=False):
    del deterministic
    base_env = env.env
    tasks = _current_tasks(base_env)
    if not tasks:
        base_env.clear_external_cpu_frequencies()
        if hasattr(env, "_xu2024_last_route"):
            route = np.asarray(env._xu2024_last_route, dtype=np.float64)
        else:
            route = np.repeat(
                np.eye(env.N, dtype=np.float64)[None, ...],
                2,
                axis=0,
            )
        env._xu2024_used_expected_fallback = False
        env._xu2024_empty_arrival_frames = int(
            getattr(env, "_xu2024_empty_arrival_frames", 0)
        ) + 1
        return _route_to_action(env, route)

    user_ids = np.asarray(
        [_task_field(task, "user_id") for task in tasks],
        dtype=np.int64,
    )
    source_uav = np.asarray(
        [_task_field(task, "source_uav") for task in tasks],
        dtype=np.int64,
    )
    priorities = np.asarray(
        [_task_field(task, "priority") for task in tasks],
        dtype=object,
    )
    task_mbits = np.asarray(
        [_task_field(task, "size_mbits") for task in tasks],
        dtype=np.float64,
    )
    user_coords = np.asarray(base_env.user_coords[user_ids], dtype=np.float64)
    u2g_rate_mbps = (
        base_env.compute_u2g_rate_bps(
            user_coords=user_coords,
            uav_coords=base_env.uav_coords,
            associations=source_uav,
        )
        / 1e6
    )
    u2u_rate_mbps = (
        base_env.compute_u2u_rate_bps(
            uav_coords=base_env.uav_coords,
        )
        / 1e6
    )
    if str(
        getattr(env, "xu_sca_task_aggregation", "source_priority")
    ) == "source_priority":
        (
            priorities,
            source_uav,
            task_mbits,
            u2g_rate_mbps,
        ) = _aggregate_source_priority_tasks(
            priorities,
            source_uav,
            task_mbits,
            u2g_rate_mbps,
            env.N,
        )
    task_delay_weights = np.ones(task_mbits.size, dtype=np.float64)
    if bool(getattr(env, "xu_sca_use_priority_delay_weights", True)):
        task_delay_weights = np.where(
            priorities == "high",
            float(base_env.w_D_high),
            float(base_env.w_D_low),
        )
    route_wait_cost_s = None
    queue_aware = bool(getattr(env, "xu_sca_queue_aware", False))
    if queue_aware:
        route_wait_cost_s = _queue_wait_costs_s(
            base_env,
            priorities,
            source_uav,
            u2u_rate_mbps,
        )

    problem = Xu2024SCAInput(
        task_mbits=task_mbits,
        cycles_per_bit=float(base_env.C),
        source_uav=source_uav,
        u2g_rate_mbps=u2g_rate_mbps,
        u2u_rate_mbps=u2u_rate_mbps,
        f_max_ghz=np.asarray(base_env.F_max_array, dtype=np.float64) / 1e9,
        capacitance_coeff=float(base_env.kappa),
        frame_seconds=float(base_env.resource_frame_dt),
        gt_tx_power_w=float(base_env.p_gt),
        uav_tx_power_w=float(base_env.p_u2u),
        delay_weight=float(getattr(env, "xu_sca_delay_weight", 0.3)),
        energy_weight=float(getattr(env, "xu_sca_energy_weight", 0.7)),
        task_delay_weights=task_delay_weights,
        route_wait_cost_s=route_wait_cost_s,
        enforce_service_stability=False,
    )
    rho_init, cpu_init = _previous_frame_initial_point(
        env,
        priorities,
        source_uav,
        task_mbits,
    )
    started = time.perf_counter()
    solve_kwargs = {
        "max_iterations": int(getattr(env, "xu_sca_max_iter", 10)),
        "tolerance": float(getattr(env, "xu_sca_tol", 0.01)),
        "proximal_rho": float(
            getattr(env, "xu_sca_proximal_rho", 1e-3)
        ),
        "proximal_cpu": float(
            getattr(env, "xu_sca_proximal_cpu", 1e-3)
        ),
        "enforce_frame_deadline": bool(
            getattr(env, "xu_sca_enforce_frame_deadline", False)
        ),
        "solver": str(getattr(env, "xu_sca_solver", "CLARABEL")),
    }
    try:
        solution = solve_xu2024_sca(
            problem,
            rho_init=rho_init,
            cpu_init_ghz=cpu_init,
            **solve_kwargs,
        )
    except (RuntimeError, SolverError):
        if rho_init is None and cpu_init is None:
            raise
        env._xu2024_cold_start_fallbacks = int(
            getattr(env, "_xu2024_cold_start_fallbacks", 0)
        ) + 1
        solution = solve_xu2024_sca(problem, **solve_kwargs)

    route = np.zeros((2, env.N, env.N), dtype=np.float64)
    for priority_index, priority in enumerate(("high", "low")):
        for source in range(env.N):
            mask = (priorities == priority) & (source_uav == source)
            if np.any(mask):
                weights = task_mbits[mask]
                route[priority_index, source] = np.average(
                    solution.rho[mask],
                    axis=0,
                    weights=weights,
                )
            elif hasattr(env, "_xu2024_last_route"):
                route[priority_index, source] = env._xu2024_last_route[
                    priority_index,
                    source,
                ]
            else:
                route[priority_index, source, source] = 1.0
    route /= np.maximum(np.sum(route, axis=2, keepdims=True), 1e-12)

    f_high_hz = np.sum(
        solution.cpu_ghz[priorities == "high"],
        axis=0,
    ) * 1e9
    f_low_hz = np.sum(
        solution.cpu_ghz[priorities == "low"],
        axis=0,
    ) * 1e9
    if queue_aware:
        # The paper assumes every task can finish inside its independent
        # frame. Persistent queues permit stochastic bursts above one-frame
        # capacity, so a hard per-frame completion constraint is invalid.
        # Reuse the experiment's common causal queue-aware CPU allocator and
        # let CVX own only the routing decision in this explicit adaptation.
        base_env.clear_external_cpu_frequencies()
    else:
        base_env.set_external_cpu_frequencies(
            f_high_hz,
            f_low_hz,
            duration_steps=base_env.resource_frame_steps,
        )

    env._xu2024_last_route = route.astype(np.float32)
    env._xu2024_last_f_high_hz = f_high_hz.astype(np.float64)
    env._xu2024_last_f_low_hz = f_low_hz.astype(np.float64)
    env._xu2024_last_solution = solution
    env._xu2024_used_expected_fallback = False
    env._xu2024_solve_count = int(
        getattr(env, "_xu2024_solve_count", 0)
    ) + 1
    env._xu2024_total_solve_seconds = float(
        getattr(env, "_xu2024_total_solve_seconds", 0.0)
    ) + (time.perf_counter() - started)
    env._xu2024_total_iterations = int(
        getattr(env, "_xu2024_total_iterations", 0)
    ) + int(solution.iterations)
    return _route_to_action(env, route)


def main():
    args = parse_args()

    def micro_policy(env, deterministic=False):
        configure_xu2024_wrapper(env, args)
        return solve_xu2024_micro_action(env, deterministic=deterministic)

    if str(args.macro_action_mode) in {
        "centroid_waypoint",
        "cooperation_waypoint",
    }:
        def macro_policy(env, _local_obs_batch, _deterministic=False):
            return np.zeros((env.N, 2), dtype=np.float32)
    else:
        macro_policy = centroid_chase_macro_policy

    run_fixed_policy_benchmark(
        experiment_label="Xu2024-SCA-adapted",
        args=args,
        csv_filename="eval_xu2024_sca_adapted.csv",
        macro_policy=macro_policy,
        micro_policy=micro_policy,
    )


if __name__ == "__main__":
    main()
