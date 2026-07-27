import argparse
import time
from typing import Tuple

import numpy as np

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap

bootstrap()

from compare_common import add_common_args
from compare_common import compute_u2u_rate_mbps
from compare_common import run_fixed_policy_benchmark

def parse_args():
    parser = argparse.ArgumentParser(
        description="SCA-based resource allocation baseline for UAV-MEC",
    )
    add_common_args(parser, default_results_subdir="sca")
    parser.set_defaults(
        episodes=1,
        eval_interval=0,
        eval_episodes=10,
        trajectory_plot_interval=0,
    )
    return parser.parse_args()


def _build_zone_priority_task_bits(env) -> np.ndarray:
    """Return one high- and one low-priority virtual task per source zone."""
    base_env = env.env
    high_per_user = (
        base_env.user_lambda_high
        * base_env.user_task_size_high_mbits
        * base_env.arrival_dt
    )
    low_per_user = (
        base_env.user_lambda_low
        * base_env.user_task_size_low_mbits
        * base_env.arrival_dt
    )
    high = np.zeros(base_env.N, dtype=np.float64)
    low = np.zeros(base_env.N, dtype=np.float64)
    for zone_idx in range(base_env.N):
        mask = base_env.user_allocations == zone_idx
        high[zone_idx] = float(np.sum(high_per_user[mask]))
        low[zone_idx] = float(np.sum(low_per_user[mask]))
    return np.concatenate([high, low], axis=0) * 1e6


def _sca_cache_key(env) -> Tuple[int, ...]:
    base_env = env.env
    interval = max(int(getattr(env, "sca_update_interval", 1)), 1)
    resource_step = max(int(getattr(base_env, "resource_frame_steps", 1)), 1)
    resource_frame_index = int(base_env.global_micro_step) // resource_step
    bucket = resource_frame_index // interval
    coords = np.rint(np.asarray(base_env.uav_coords) * 4.0).astype(np.int64)
    load_key = int(round(float(getattr(base_env, "load_scale", 1.0)) * 1000.0))
    cpu_key = tuple(
        np.rint(np.asarray(base_env.F_max_array) / 1e6)
        .astype(np.int64)
        .tolist()
    )
    if str(getattr(env, "power_constraint_scope", "system")) == "per_uav":
        dual_key = tuple(
            np.rint(
                np.asarray(
                    getattr(env, "energy_lagrange_multipliers", np.zeros(env.N)),
                    dtype=np.float64,
                )
                * 1e6
            )
            .astype(np.int64)
            .tolist()
        )
    else:
        dual_key = (
            int(
                round(
                    float(getattr(env, "energy_lagrange_multiplier", 0.0))
                    * 1e6
                )
            ),
        )
    return (
        bucket,
        load_key,
        *dual_key,
        *cpu_key,
        *coords.reshape(-1).tolist(),
    )


def _routing_softmax(action, num_uavs, diagonal_bias, action_scale):
    action = np.asarray(action, dtype=np.float64).reshape(
        2 * num_uavs, num_uavs
    )
    action = np.clip(action, -1.0, 1.0)
    logits = action * float(action_scale)
    logits += np.tile(
        np.eye(num_uavs, dtype=np.float64),
        (2, 1),
    ) * float(diagonal_bias)
    logits -= np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.maximum(
        np.sum(exp_logits, axis=1, keepdims=True),
        1e-12,
    )


def _sca_energy_coefficients(omega_energy, energy_norm, num_uavs):
    weights = np.asarray(omega_energy, dtype=np.float64)
    normalizers = np.asarray(energy_norm, dtype=np.float64)
    if weights.ndim == 0 or weights.shape == (1,):
        weights = np.full(num_uavs, float(weights.reshape(-1)[0]))
    if normalizers.ndim == 0 or normalizers.shape == (1,):
        normalizers = np.full(
            num_uavs,
            float(normalizers.reshape(-1)[0]),
        )
    if weights.shape != (num_uavs,) or normalizers.shape != (num_uavs,):
        raise ValueError(
            "SCA energy weights and normalizers must be scalar or one per UAV."
        )
    if np.any(weights < 0.0) or np.any(normalizers <= 0.0):
        raise ValueError(
            "SCA energy weights must be non-negative and normalizers positive."
        )
    return weights / normalizers


def _routing_surrogate_value_and_gradient(
    action,
    anchor_action,
    high_load,
    low_load,
    queue_high,
    queue_low,
    service_capacity,
    shared_rate,
    peak_compute_energy,
    micro_dt,
    tx_power,
    diagonal_bias,
    action_scale,
    delay_weight_high,
    delay_weight_low,
    omega_D,
    omega_E,
    delay_norm,
    energy_norm,
    proximal_weight,
):
    num_uavs = int(high_load.size)
    energy_coefficients = _sca_energy_coefficients(
        omega_E,
        energy_norm,
        num_uavs,
    )
    action = np.asarray(action, dtype=np.float64).reshape(
        2 * num_uavs,
        num_uavs,
    )
    rho = _routing_softmax(
        action,
        num_uavs,
        diagonal_bias,
        action_scale,
    )
    rho_high = rho[:num_uavs]
    rho_low = rho[num_uavs:]

    assigned_high = high_load @ rho_high
    assigned_low = low_load @ rho_low
    utilization = (
        assigned_high + assigned_low
    ) / service_capacity
    weighted_load_norm = max(
        float(
            delay_weight_high * np.sum(high_load)
            + delay_weight_low * np.sum(low_load)
        ),
        1e-8,
    )

    inverse_rate = 1.0 / np.maximum(shared_rate, 1e-8)
    np.fill_diagonal(inverse_rate, 0.0)
    tx_time_high = high_load[:, None] * inverse_rate
    tx_time_low = low_load[:, None] * inverse_rate

    communication_delay = (
        delay_weight_high * np.sum(tx_time_high * rho_high)
        + delay_weight_low * np.sum(tx_time_low * rho_low)
    ) / weighted_load_norm
    queue_numerator = np.sum(
        float(micro_dt)
        / service_capacity
        * (
            delay_weight_high * queue_high * assigned_high
            + delay_weight_low
            * (queue_high + queue_low)
            * assigned_low
            + delay_weight_high * np.square(assigned_high)
            + delay_weight_low * assigned_high * assigned_low
            + delay_weight_low * np.square(assigned_low)
        )
    )
    queue_delay = queue_numerator / weighted_load_norm
    overload = np.maximum(utilization - 1.0, 0.0)
    overload_cost = float(np.mean(np.square(overload)))
    delay_cost = communication_delay + queue_delay + overload_cost

    transmission_energy_per_source = float(tx_power) * (
        np.sum(tx_time_high * rho_high, axis=1)
        + np.sum(tx_time_low * rho_low, axis=1)
    )
    compute_energy_per_destination = (
        peak_compute_energy * np.power(utilization, 3)
    )
    energy_cost = float(
        np.sum(
            transmission_energy_per_source
            + compute_energy_per_destination
        )
    )
    dual_weighted_energy = float(
        np.dot(
            energy_coefficients,
            transmission_energy_per_source
            + compute_energy_per_destination,
        )
    )

    objective = (
        float(omega_D) * delay_cost / max(float(delay_norm), 1e-8)
        + dual_weighted_energy
        + float(proximal_weight)
        * float(np.sum(np.square(action - anchor_action)))
    )

    grad_delay_high = (
        delay_weight_high * tx_time_high / weighted_load_norm
    )
    grad_delay_low = (
        delay_weight_low * tx_time_low / weighted_load_norm
    )
    queue_scale = (
        float(micro_dt) / service_capacity / weighted_load_norm
    )
    grad_assigned_high = queue_scale * (
        delay_weight_high * queue_high
        + 2.0 * delay_weight_high * assigned_high
        + delay_weight_low * assigned_low
    )
    grad_assigned_low = queue_scale * (
        delay_weight_low * (queue_high + queue_low)
        + delay_weight_low * assigned_high
        + 2.0 * delay_weight_low * assigned_low
    )
    grad_overload_assigned = (
        2.0 * overload / max(num_uavs, 1) / service_capacity
    )
    grad_assigned_high += grad_overload_assigned
    grad_assigned_low += grad_overload_assigned
    grad_delay_high += high_load[:, None] * grad_assigned_high[None, :]
    grad_delay_low += low_load[:, None] * grad_assigned_low[None, :]

    grad_energy_high = (
        energy_coefficients[:, None]
        * float(tx_power)
        * tx_time_high
    )
    grad_energy_low = (
        energy_coefficients[:, None]
        * float(tx_power)
        * tx_time_low
    )
    grad_compute_assigned = (
        3.0
        * peak_compute_energy
        * np.square(utilization)
        / service_capacity
    )
    priced_compute_gradient = energy_coefficients * grad_compute_assigned
    grad_energy_high += high_load[:, None] * priced_compute_gradient[None, :]
    grad_energy_low += low_load[:, None] * priced_compute_gradient[None, :]

    grad_rho = np.vstack(
        [
            float(omega_D)
            * grad_delay_high
            / max(float(delay_norm), 1e-8)
            + grad_energy_high,
            float(omega_D)
            * grad_delay_low
            / max(float(delay_norm), 1e-8)
            + grad_energy_low,
        ]
    )
    grad_action = np.empty_like(action)
    for row_idx in range(2 * num_uavs):
        probability = rho[row_idx]
        row_gradient = grad_rho[row_idx]
        grad_action[row_idx] = float(action_scale) * probability * (
            row_gradient - float(np.dot(row_gradient, probability))
        )
    grad_action += (
        2.0 * float(proximal_weight) * (action - anchor_action)
    )

    components = {
        "delay": float(delay_cost),
        "energy": float(energy_cost),
        "dual_weighted_energy": float(dual_weighted_energy),
        "overload": float(overload_cost),
    }
    return float(objective), grad_action, rho, components


def optimize_routing_logits_SCA(
    expected_load_mbits,
    queue_high_mbits,
    queue_low_mbits,
    service_capacity_mbits,
    nominal_u2u_rate_mbps,
    f_max_hz,
    kappa,
    micro_dt,
    tx_power,
    diagonal_bias,
    action_scale,
    delay_weight_high=10.0,
    delay_weight_low=1.0,
    omega_D=0.7,
    omega_E=0.3,
    delay_norm=10.0,
    energy_norm=14.0,
    action_init=None,
    max_iter=3,
    tol=1e-3,
    proximal_weight=1e-3,
    trust_region=0.5,
    solver="CLARABEL",
    verbose=False,
):
    """Fast projected SCA over the same bounded logits used by SAC.

    At each outer iteration, equal-bandwidth U2U coupling is fixed at the
    current active-link estimate.  A quadratic upper model in the logits is
    minimized by projected gradient with backtracking.  This is the same
    route-only subproblem as the CVXPY reference above without rebuilding a
    cone program at every 0.1-second slot.
    """
    del solver, verbose
    expected_load_mbits = np.asarray(expected_load_mbits, dtype=np.float64)
    if expected_load_mbits.ndim != 2 or expected_load_mbits.shape[0] != 2:
        raise ValueError("expected_load_mbits must have shape (2, N).")
    num_uavs = int(expected_load_mbits.shape[1])
    high_load = np.maximum(expected_load_mbits[0], 1e-8)
    low_load = np.maximum(expected_load_mbits[1], 1e-8)
    queue_high = np.maximum(
        np.asarray(queue_high_mbits, dtype=np.float64).reshape(num_uavs),
        0.0,
    )
    queue_low = np.maximum(
        np.asarray(queue_low_mbits, dtype=np.float64).reshape(num_uavs),
        0.0,
    )
    service_capacity = np.maximum(
        np.asarray(service_capacity_mbits, dtype=np.float64).reshape(num_uavs),
        1e-6,
    )
    nominal_rate = np.maximum(
        np.asarray(nominal_u2u_rate_mbps, dtype=np.float64),
        1e-8,
    )
    f_max_hz = np.asarray(f_max_hz, dtype=np.float64).reshape(num_uavs)
    peak_compute_energy = (
        float(kappa) * np.power(f_max_hz, 3) * float(micro_dt)
    )
    if action_init is None:
        action = np.zeros((2 * num_uavs, num_uavs), dtype=np.float64)
    else:
        action = np.clip(
            np.asarray(action_init, dtype=np.float64).reshape(
                2 * num_uavs,
                num_uavs,
            ),
            -1.0,
            1.0,
        )

    offdiag_mask = ~np.eye(num_uavs, dtype=bool)
    objective_history = []
    max_delta_history = []
    active_link_history = []
    status_history = []
    component_history = []

    for _ in range(max(int(max_iter), 1)):
        outer_start = action.copy()
        rho = _routing_softmax(
            action,
            num_uavs,
            diagonal_bias,
            action_scale,
        )
        offloaded_mbits = (
            high_load[:, None] * rho[:num_uavs]
            + low_load[:, None] * rho[num_uavs:]
        )
        flow_scale = max(
            float(np.sum(high_load + low_load)) * 0.02,
            1e-4,
        )
        active_link_count = max(
            float(
                np.sum(
                    (1.0 - np.exp(-offloaded_mbits / flow_scale))[
                        offdiag_mask
                    ]
                )
            ),
            1.0,
        )
        active_link_history.append(active_link_count)
        shared_rate = nominal_rate / active_link_count
        np.fill_diagonal(shared_rate, np.inf)
        anchor_action = action.copy()

        accepted_any = False
        for _inner in range(8):
            value, gradient, _, components = (
                _routing_surrogate_value_and_gradient(
                    action=action,
                    anchor_action=anchor_action,
                    high_load=high_load,
                    low_load=low_load,
                    queue_high=queue_high,
                    queue_low=queue_low,
                    service_capacity=service_capacity,
                    shared_rate=shared_rate,
                    peak_compute_energy=peak_compute_energy,
                    micro_dt=micro_dt,
                    tx_power=tx_power,
                    diagonal_bias=diagonal_bias,
                    action_scale=action_scale,
                    delay_weight_high=delay_weight_high,
                    delay_weight_low=delay_weight_low,
                    omega_D=omega_D,
                    omega_E=omega_E,
                    delay_norm=delay_norm,
                    energy_norm=energy_norm,
                    proximal_weight=proximal_weight,
                )
            )
            gradient_max = float(np.max(np.abs(gradient)))
            if gradient_max < float(tol):
                break
            step_size = min(1.0, float(trust_region) / gradient_max)
            accepted = False
            for _backtrack in range(12):
                candidate = np.clip(
                    action - step_size * gradient,
                    -1.0,
                    1.0,
                )
                candidate = np.minimum(
                    np.maximum(
                        candidate,
                        anchor_action - float(trust_region),
                    ),
                    anchor_action + float(trust_region),
                )
                candidate_value, _, _, _ = (
                    _routing_surrogate_value_and_gradient(
                        action=candidate,
                        anchor_action=anchor_action,
                        high_load=high_load,
                        low_load=low_load,
                        queue_high=queue_high,
                        queue_low=queue_low,
                        service_capacity=service_capacity,
                        shared_rate=shared_rate,
                        peak_compute_energy=peak_compute_energy,
                        micro_dt=micro_dt,
                        tx_power=tx_power,
                        diagonal_bias=diagonal_bias,
                        action_scale=action_scale,
                        delay_weight_high=delay_weight_high,
                        delay_weight_low=delay_weight_low,
                        omega_D=omega_D,
                        omega_E=omega_E,
                        delay_norm=delay_norm,
                        energy_norm=energy_norm,
                        proximal_weight=proximal_weight,
                    )
                )
                directional_change = float(
                    np.sum(gradient * (candidate - action))
                )
                if candidate_value <= value + 1e-4 * directional_change:
                    action = candidate
                    accepted = True
                    accepted_any = True
                    break
                step_size *= 0.5
            if not accepted:
                break

        final_value, _, final_rho, final_components = (
            _routing_surrogate_value_and_gradient(
                action=action,
                anchor_action=anchor_action,
                high_load=high_load,
                low_load=low_load,
                queue_high=queue_high,
                queue_low=queue_low,
                service_capacity=service_capacity,
                shared_rate=shared_rate,
                peak_compute_energy=peak_compute_energy,
                micro_dt=micro_dt,
                tx_power=tx_power,
                diagonal_bias=diagonal_bias,
                action_scale=action_scale,
                delay_weight_high=delay_weight_high,
                delay_weight_low=delay_weight_low,
                omega_D=omega_D,
                omega_E=omega_E,
                delay_norm=delay_norm,
                energy_norm=energy_norm,
                proximal_weight=proximal_weight,
            )
        )
        outer_delta = float(np.max(np.abs(action - outer_start)))
        objective_history.append(final_value)
        max_delta_history.append(outer_delta)
        component_history.append(final_components)
        status_history.append(
            "projected-optimal" if accepted_any else "stationary"
        )
        if outer_delta < float(tol):
            break

    info = {
        "objective_history": np.asarray(objective_history, dtype=np.float64),
        "max_delta_history": np.asarray(max_delta_history, dtype=np.float64),
        "active_link_history": np.asarray(active_link_history, dtype=np.float64),
        "status_history": np.asarray(status_history, dtype=object),
        "component_history": component_history,
        "rho": final_rho.astype(np.float32),
    }
    return action.reshape(-1).astype(np.float32), info


def solve_sca_micro_action(env, deterministic=False):
    """
    Solve a route-only SCA problem on the current state.

    The returned vector is the same bounded 18-D logit action used by SAC;
    CPU allocation remains the common environment-side priority rule.
    """
    base_env = env.env
    cache_key = _sca_cache_key(env)
    if getattr(env, "_sca_cached_key", None) == cache_key:
        return np.asarray(env._sca_cached_action, dtype=np.float32).copy()

    raw_state = base_env.get_raw_state()
    task_bits = _build_zone_priority_task_bits(env)
    num_uavs = base_env.N
    expected_load_mbits = task_bits.reshape(2, num_uavs) / 1e6
    nominal_u2u_rate_mbps = compute_u2u_rate_mbps(env, raw_state)
    queued_high = np.asarray(raw_state["Q_loc_high"], dtype=np.float64) + np.sum(
        np.asarray(raw_state["Q_tx_high"], dtype=np.float64), axis=0
    )
    queued_low = np.asarray(raw_state["Q_loc_low"], dtype=np.float64) + np.sum(
        np.asarray(raw_state["Q_tx_low"], dtype=np.float64), axis=0
    )
    service_capacity_mbits = (
        base_env.F_max_array.astype(np.float64)
        * float(base_env.resource_frame_dt)
        / max(float(base_env.C) * 1e6, 1e-12)
    )
    constrained_objective = (
        str(getattr(env, "objective_mode", "weighted"))
        == "constrained_delay"
    )
    if constrained_objective:
        sca_delay_weight = float(getattr(env, "J_w_delay", 0.7))
        if str(getattr(env, "power_constraint_scope", "system")) == "per_uav":
            sca_energy_weight = np.asarray(
                env.energy_lagrange_multipliers,
                dtype=np.float64,
            )
            sca_energy_norm = np.maximum(
                np.asarray(env.uav_power_budgets_w, dtype=np.float64)
                * float(base_env.resource_frame_dt),
                1e-8,
            )
        else:
            sca_energy_weight = float(
                getattr(env, "energy_lagrange_multiplier", 0.0)
            )
            sca_energy_norm = max(
                float(getattr(env, "power_budget_w", 0.0))
                * float(base_env.resource_frame_dt),
                1e-8,
            )
    else:
        sca_delay_weight = float(getattr(env, "sca_delay_weight", 0.7))
        sca_energy_weight = float(getattr(env, "sca_energy_weight", 0.3))
        sca_energy_norm = float(getattr(env, "J_energy_norm", 14.0))
    started = time.perf_counter()
    micro_action, info = optimize_routing_logits_SCA(
        expected_load_mbits=expected_load_mbits,
        queue_high_mbits=queued_high,
        queue_low_mbits=queued_low,
        service_capacity_mbits=service_capacity_mbits,
        nominal_u2u_rate_mbps=nominal_u2u_rate_mbps,
        f_max_hz=base_env.F_max_array,
        kappa=float(base_env.kappa),
        micro_dt=float(base_env.resource_frame_dt),
        tx_power=float(base_env.p_u2u),
        diagonal_bias=float(getattr(env, "micro_diag_bias", 0.5)),
        action_scale=float(getattr(env, "micro_action_scale", 1.0)),
        delay_weight_high=float(base_env.w_D_high),
        delay_weight_low=float(base_env.w_D_low),
        omega_D=sca_delay_weight,
        omega_E=sca_energy_weight,
        delay_norm=float(getattr(env, "J_delay_norm", 10.0)),
        energy_norm=sca_energy_norm,
        action_init=getattr(env, "_sca_action_warm", None),
        max_iter=int(getattr(env, "sca_max_iter", 5)),
        tol=float(getattr(env, "sca_tol", 1e-3)),
        proximal_weight=float(getattr(env, "sca_proximal_weight", 1e-4)),
        solver=str(getattr(env, "sca_solver", "CLARABEL")),
        verbose=False,
    )
    env._sca_action_warm = micro_action.copy()
    env._sca_rho_warm = np.asarray(info["rho"], dtype=np.float32).copy()

    env._sca_cached_key = cache_key
    env._sca_cached_action = micro_action.copy()
    env._sca_solve_count = int(getattr(env, "_sca_solve_count", 0)) + 1
    env._sca_total_solve_seconds = float(
        getattr(env, "_sca_total_solve_seconds", 0.0)
    ) + (time.perf_counter() - started)
    env._sca_last_status = (
        str(info["status_history"][-1])
        if len(info.get("status_history", []))
        else "no-iteration"
    )
    return micro_action


def centroid_chase_macro_policy(env, local_obs_batch, deterministic=False):
    actions = np.zeros((env.N, 2), dtype=np.float32)
    max_displacement = max(float(getattr(env, "max_macro_displacement", 20.0)), 1e-8)

    for agent_idx in range(env.N):
        delta_centroid = np.asarray(local_obs_batch[agent_idx][:2], dtype=np.float32)
        delta_centroid_m = delta_centroid * 1000.0
        distance_m = float(np.linalg.norm(delta_centroid_m))
        if distance_m < 1e-6:
            continue

        rho = min(distance_m / max_displacement, 1.0)
        theta = np.arctan2(delta_centroid_m[1], delta_centroid_m[0]) / np.pi
        actions[agent_idx, 0] = np.clip(rho, 0.0, 1.0)
        actions[agent_idx, 1] = np.clip(theta, -1.0, 1.0)

    return actions


def main():
    args = parse_args()
    run_fixed_policy_benchmark(
        experiment_label="SCA",
        args=args,
        csv_filename="eval_sca.csv",
        macro_policy=centroid_chase_macro_policy,
        micro_policy=solve_sca_micro_action,
    )


if __name__ == "__main__":
    main()
