import argparse
import types
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap

bootstrap()

from compare_common import add_common_args
from compare_common import alpha_to_micro_action
from compare_common import compute_u2u_rate_mbps
from compare_common import run_fixed_policy_benchmark

try:
    import cvxpy as cp
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "compare_sca.py requires cvxpy. Please install cvxpy first."
    ) from exc


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


def _ensure_2d_rates(rates, user_to_uav, num_users, num_uavs, eps=1e-12):
    """
    Convert a UAV-level rate matrix into a per-user rate matrix.
    The returned matrix has shape (K, M), where K is the number of users
    and M is the number of UAVs.
    """
    rates = np.asarray(rates, dtype=np.float64)
    user_to_uav = np.asarray(user_to_uav, dtype=np.int64).reshape(-1)

    if rates.shape == (num_users, num_uavs):
        user_rates = rates.copy()
    elif rates.shape == (num_uavs, num_uavs):
        user_rates = rates[user_to_uav]
    else:
        raise ValueError(
            f"Unsupported rate shape {rates.shape}. Expected "
            f"(K, M) or (M, M)."
        )

    user_rates = np.maximum(user_rates, eps)
    return user_rates


def _normalize_rows(matrix, eps=1e-12):
    matrix = np.asarray(matrix, dtype=np.float64)
    row_sum = np.sum(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(row_sum, eps)


def _compute_u2u_rate_bps_from_env(env):
    diff = env.uav_coords[:, None, :] - env.uav_coords[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)

    channel_gain = env.beta_0 / np.maximum(dist_sq, 1e-8)
    np.fill_diagonal(channel_gain, 0.0)

    rate_bps = env.B_u2u_total * np.log2(
        1.0 + (env.p_u2u * channel_gain) / env.noise_power
    )
    np.fill_diagonal(rate_bps, 0.0)
    return rate_bps


def _build_user_task_bits_from_state(env):
    """
    Build a user-level load proxy from the current simulator state.
    We use the current zone-level arrivals first, and fall back to the
    profile-implied expected load if a zone is idle.
    """
    current_zone_load_bits = np.asarray(env.A_t_high + env.A_t_low, dtype=np.float64) * 1e6
    expected_user_load_bits = (
        env.user_lambda_high * env.user_task_size_high_mbits
        + env.user_lambda_low * env.user_task_size_low_mbits
    ) * env.arrival_dt * 1e6

    task_bits = np.zeros(env.M, dtype=np.float64)
    for zone_idx in range(env.N):
        user_idx = np.where(env.user_allocations == zone_idx)[0]
        if user_idx.size == 0:
            continue

        if current_zone_load_bits[zone_idx] > 1e-12:
            per_user_bits = current_zone_load_bits[zone_idx] / float(user_idx.size)
        else:
            per_user_bits = float(np.mean(expected_user_load_bits[user_idx]))

        task_bits[user_idx] = max(per_user_bits, 1e-8)

    return task_bits


def _aggregate_user_solution_to_uav(env, rho, f, task_bits):
    """
    Convert per-user rho/f into the simulator-facing row-stochastic matrices
    and per-UAV CPU budgets.
    """
    split_ratios = np.zeros((env.N, env.N), dtype=np.float64)
    for zone_idx in range(env.N):
        user_idx = np.where(env.user_allocations == zone_idx)[0]
        if user_idx.size == 0:
            split_ratios[zone_idx, zone_idx] = 1.0
            continue

        weights = np.asarray(task_bits[user_idx], dtype=np.float64)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-12:
            weights = np.ones_like(weights)
            weight_sum = float(np.sum(weights))

        split_ratios[zone_idx] = (
            np.sum(rho[user_idx] * weights[:, None], axis=0) / weight_sum
        )
        split_ratios[zone_idx] = np.clip(split_ratios[zone_idx], 1e-12, 1.0)
        split_ratios[zone_idx] /= np.sum(split_ratios[zone_idx])

    f_uav = np.sum(f, axis=0)
    f_uav = np.clip(f_uav, 0.0, env.F_max_array)
    return split_ratios, split_ratios, f_uav


def optimize_allocation_SCA(
    task_bits,
    cpu_cycles_per_bit,
    user_to_uav,
    u2u_rates_bps,
    phi,
    f_max,
    tx_power=1.0,
    omega_D=1.0,
    omega_E=1.0,
    comm_delay=None,
    rho_init=None,
    f_init=None,
    max_iter=10,
    tol=1e-3,
    proximal_weight=1e-4,
    solver=None,
    verbose=False,
    eps=1e-8,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    SCA baseline for the paper-style joint offloading / CPU allocation problem.

    Parameters
    ----------
    task_bits : array_like, shape (K,)
        Task size for each user. In this repository we keep the same unit
        convention as the simulator, so this can also be interpreted as a load
        proxy if you are feeding expected traffic.
    cpu_cycles_per_bit : array_like, shape (K,) or scalar
        CPU cycles required per bit.
    user_to_uav : array_like, shape (K,)
        Fixed region association of each user to its home UAV.
    u2u_rates_bps : array_like, shape (M, M) or (K, M)
        U2U transmission rates. If an (M, M) matrix is given, each user's
        source UAV is used to pick the corresponding row.
    phi : array_like, shape (M,)
        Capacitance/energy factor of each UAV.
    f_max : array_like, shape (M,)
        CPU budget of each UAV.

    Returns
    -------
    rho : ndarray, shape (K, M)
        Forwarding ratio.
    f : ndarray, shape (K, M)
        CPU frequency allocated to each user on each UAV.
    info : dict
        Convergence trace and helper outputs.
    """
    task_bits = np.asarray(task_bits, dtype=np.float64).reshape(-1)
    user_to_uav = np.asarray(user_to_uav, dtype=np.int64).reshape(-1)
    f_max = np.asarray(f_max, dtype=np.float64).reshape(-1)
    phi = np.asarray(phi, dtype=np.float64).reshape(-1)
    cpu_cycles_per_bit = np.asarray(cpu_cycles_per_bit, dtype=np.float64)

    num_users = task_bits.size
    num_uavs = f_max.size

    if user_to_uav.size != num_users:
        raise ValueError(
            f"user_to_uav has length {user_to_uav.size}, but task_bits has "
            f"length {num_users}."
        )

    if cpu_cycles_per_bit.ndim == 0:
        cpu_cycles_per_bit = np.full(num_users, float(cpu_cycles_per_bit), dtype=np.float64)
    else:
        cpu_cycles_per_bit = cpu_cycles_per_bit.reshape(-1)
        if cpu_cycles_per_bit.size != num_users:
            raise ValueError(
                f"cpu_cycles_per_bit has length {cpu_cycles_per_bit.size}, "
                f"but task_bits has length {num_users}."
            )

    if phi.size != num_uavs:
        raise ValueError(f"phi has length {phi.size}, expected {num_uavs}.")

    u2u_rates_bps = _ensure_2d_rates(
        u2u_rates_bps,
        user_to_uav=user_to_uav,
        num_users=num_users,
        num_uavs=num_uavs,
        eps=eps,
    )

    # Local forwarding is free, so we override the diagonal to zero delay.
    for k in range(num_users):
        u2u_rates_bps[k, user_to_uav[k]] = np.inf

    # Constant coefficients in the objective.
    tran_delay_coeff = task_bits[:, None] / np.maximum(u2u_rates_bps, eps)
    tran_delay_coeff[np.arange(num_users), user_to_uav] = 0.0
    comp_coeff = np.broadcast_to((task_bits * cpu_cycles_per_bit)[:, None], (num_users, num_uavs))
    comp_energy_coeff = np.broadcast_to(
        ((task_bits * cpu_cycles_per_bit)[:, None] * phi[None, :]),
        (num_users, num_uavs),
    )

    if comm_delay is None:
        comm_delay = np.zeros(num_users, dtype=np.float64)
    else:
        comm_delay = np.asarray(comm_delay, dtype=np.float64).reshape(-1)
        if comm_delay.size != num_users:
            raise ValueError(
                f"comm_delay has length {comm_delay.size}, expected {num_users}."
            )

    # Feasible initialization.
    if rho_init is None:
        rho_prev = np.full((num_users, num_uavs), 1.0 / num_uavs, dtype=np.float64)
        rho_prev[np.arange(num_users), user_to_uav] += 0.25
        rho_prev = _normalize_rows(rho_prev, eps=eps)
    else:
        rho_prev = np.asarray(rho_init, dtype=np.float64)
        if rho_prev.shape != (num_users, num_uavs):
            raise ValueError(
                f"rho_init has shape {rho_prev.shape}, expected "
                f"({num_users}, {num_uavs})."
            )
        rho_prev = np.clip(rho_prev, 0.0, 1.0)
        rho_prev = _normalize_rows(rho_prev, eps=eps)

    if f_init is None:
        f_prev = np.tile((f_max / max(num_users, 1))[None, :], (num_users, 1))
    else:
        f_prev = np.asarray(f_init, dtype=np.float64)
        if f_prev.shape != (num_users, num_uavs):
            raise ValueError(
                f"f_init has shape {f_prev.shape}, expected "
                f"({num_users}, {num_uavs})."
            )
    f_prev = np.maximum(f_prev, eps)

    objective_history = []
    max_delta_history = []
    status_history = []

    solver_candidates = []
    if solver is not None:
        solver_candidates.append(solver)
    solver_candidates.extend(["ECOS", "SCS"])

    rho_value = rho_prev.copy()
    f_value = f_prev.copy()

    for _ in range(int(max_iter)):
        rho = cp.Variable((num_users, num_uavs), nonneg=True)
        f = cp.Variable((num_users, num_uavs), nonneg=True)

        inv_f_prev = 1.0 / np.maximum(f_prev, eps)
        inv_f_prev_sq = inv_f_prev ** 2

        # First-order Taylor expansion of rho / f around (rho_prev, f_prev).
        comp_delay_hat = cp.multiply(comp_coeff, (
            rho_prev * inv_f_prev
            + cp.multiply(inv_f_prev, rho - rho_prev)
            - cp.multiply(rho_prev * inv_f_prev_sq, f - f_prev)
        ))

        # First-order Taylor expansion of rho * f^2 around (rho_prev, f_prev).
        comp_energy_hat = cp.multiply(comp_energy_coeff, (
            rho_prev * (f_prev ** 2)
            + cp.multiply(f_prev ** 2, rho - rho_prev)
            + cp.multiply(2.0 * rho_prev * f_prev, f - f_prev)
        ))

        tran_delay = cp.multiply(tran_delay_coeff, rho)

        user_delay_terms = []
        for k in range(num_users):
            total_delay_per_destination = tran_delay[k, :] + comp_delay_hat[k, :]
            user_delay_terms.append(float(comm_delay[k]) + cp.max(total_delay_per_destination))

        # Transmission energy is proportional to transmission time.
        total_energy = cp.sum(
            tx_power * tran_delay + comp_energy_hat
        )

        constraints = [
            cp.sum(rho, axis=1) == 1.0,
            rho <= 1.0,
            cp.sum(f, axis=0) <= f_max,
        ]

        # Keep the iterate inside the feasible orthant.
        constraints.append(f <= f_max[None, :])

        objective = cp.Minimize(
            omega_D * cp.sum(cp.hstack(user_delay_terms))
            + omega_E * total_energy
            + proximal_weight
            * (cp.sum_squares(rho - rho_prev) + cp.sum_squares(f - f_prev))
        )

        problem = cp.Problem(objective, constraints)

        solved = False
        last_status = None
        for solver_name in solver_candidates:
            try:
                problem.solve(
                    solver=solver_name,
                    warm_start=True,
                    verbose=verbose,
                )
                last_status = problem.status
                if last_status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                    solved = True
                    break
            except Exception:
                last_status = "error"

        status_history.append(last_status)

        if not solved or rho.value is None or f.value is None:
            break

        rho_new = np.asarray(rho.value, dtype=np.float64)
        f_new = np.asarray(f.value, dtype=np.float64)

        rho_new = np.clip(rho_new, 0.0, 1.0)
        rho_new = _normalize_rows(rho_new, eps=eps)
        f_new = np.clip(f_new, 0.0, None)
        # Numerical cleanup: project tiny violations back to the budget.
        col_sum = np.sum(f_new, axis=0, keepdims=True)
        scale = np.minimum(1.0, f_max[None, :] / np.maximum(col_sum, eps))
        f_new = f_new * scale

        max_delta = max(
            float(np.max(np.abs(rho_new - rho_prev))),
            float(np.max(np.abs(f_new - f_prev))),
        )
        max_delta_history.append(max_delta)
        objective_history.append(float(problem.value) if problem.value is not None else np.nan)

        rho_value = rho_new
        f_value = f_new

        rho_prev = rho_new
        f_prev = np.maximum(f_new, eps)

        if max_delta < tol:
            break

    info = {
        "objective_history": np.asarray(objective_history, dtype=np.float64),
        "max_delta_history": np.asarray(max_delta_history, dtype=np.float64),
        "status_history": np.asarray(status_history, dtype=object),
    }
    return rho_value.astype(np.float32), f_value.astype(np.float32), info


def _sca_micro_step(self, split_ratios_high_unused, split_ratios_low_unused):
    """
    Runtime replacement for UAV_MEC_Env.micro_step.
    The incoming action is ignored; we solve the joint SCA problem directly
    from the current state so that rho and f are both used.
    """
    arrived_high = np.zeros(self.N)
    arrived_low = np.zeros(self.N)

    if self.pending_upload_jobs:
        completed_jobs = [job for job in self.pending_upload_jobs if job["remaining_steps"] <= 0]
        self.pending_upload_jobs = [
            job for job in self.pending_upload_jobs if job["remaining_steps"] > 0
        ]
        for job in completed_jobs:
            target = int(job["target_uav"])
            arrived_high[target] += job["high_bits"]
            arrived_low[target] += job["low_bits"]

    if np.sum(self.A_t_high) + np.sum(self.A_t_low) > 0:
        assigned_uavs = self.user_allocations
        user_offsets = self.user_coords - self.uav_coords[assigned_uavs]
        user_dist_sq = np.sum(user_offsets**2, axis=1) + self.H**2
        user_channel_gain = self.beta_0 / np.maximum(user_dist_sq, 1e-8)
        user_bandwidth = self.B_gt_total / max(self.M, 1)
        user_rate_bps = user_bandwidth * np.log2(
            1.0 + (self.p_gt * user_channel_gain) / self.noise_power
        )
        user_uplink_step_mbits = (user_rate_bps * self.micro_dt) / 1e6

        for i in range(self.N):
            zone_mask = self.user_allocations == i
            zone_capacity = np.sum(user_uplink_step_mbits[zone_mask])
            total_bits = self.A_t_high[i] + self.A_t_low[i]
            if total_bits <= 1e-12:
                continue
            upload_steps = max(1, int(np.ceil(total_bits / max(zone_capacity, 1e-8))))
            self.pending_upload_jobs.append(
                {
                    "target_uav": i,
                    "high_bits": float(self.A_t_high[i]),
                    "low_bits": float(self.A_t_low[i]),
                    "remaining_steps": upload_steps,
                    "active_users": float(self.current_arrival_active_users[i]),
                }
            )

    assigned_anchors = self.anchor_points[self.user_allocations]
    if self.users_static:
        self.user_velocities.fill(0.0)
        self.user_coords = assigned_anchors + self.user_offsets
        self.user_coords = np.clip(self.user_coords, 0.0, 1000.0)
    else:
        w_t = self.np_random.standard_normal((self.M, 2))
        self.user_velocities = (self.alpha_gmmm * self.user_velocities) + (
            np.sqrt(1 - self.alpha_gmmm**2) * w_t
        )

        speeds = np.linalg.norm(self.user_velocities, axis=1, keepdims=True)
        self.user_velocities = np.where(
            speeds > self.max_infantry_speed,
            self.user_velocities * (self.max_infantry_speed / np.maximum(speeds, 1e-6)),
            self.user_velocities,
        )

        self.user_offsets += self.user_velocities * self.micro_dt
        zone_dist = np.linalg.norm(self.user_offsets, axis=1, keepdims=True)
        safe_zone_dist = np.maximum(zone_dist, 1e-8)
        self.user_offsets = np.where(
            zone_dist > self.user_zone_radius,
            self.user_offsets / safe_zone_dist * self.user_zone_radius,
            self.user_offsets,
        )
        self.user_coords = assigned_anchors + self.user_offsets
        self.user_coords = np.clip(self.user_coords, 0.0, 1000.0)

    if self.macro_fly_steps_remaining > 0:
        delta_x_uav = self.current_macro_v * np.cos(self.current_macro_theta) * self.micro_dt
        delta_y_uav = self.current_macro_v * np.sin(self.current_macro_theta) * self.micro_dt
        self.uav_coords[:, 0] += delta_x_uav
        self.uav_coords[:, 1] += delta_y_uav
        self.macro_fly_steps_remaining -= 1
    self.uav_coords = np.clip(self.uav_coords, 0.0, 1000.0)

    task_bits = _build_user_task_bits_from_state(self)
    u2u_rates_bps = _compute_u2u_rate_bps_from_env(self)
    rho, f, _ = optimize_allocation_SCA(
        task_bits=task_bits,
        cpu_cycles_per_bit=np.full(self.M, self.C, dtype=np.float64),
        user_to_uav=self.user_allocations,
        u2u_rates_bps=u2u_rates_bps,
        phi=np.full(self.N, self.kappa, dtype=np.float64),
        f_max=self.F_max_array.astype(np.float64),
        tx_power=float(self.p_u2u),
        omega_D=1.0,
        omega_E=1.0,
        max_iter=10,
        tol=1e-3,
        proximal_weight=1e-4,
        solver=None,
        verbose=False,
    )
    split_ratios_high, split_ratios_low, f_opt_array = _aggregate_user_solution_to_uav(
        self, rho, f, task_bits
    )

    alloc_high = arrived_high[:, None] * split_ratios_high
    alloc_low = arrived_low[:, None] * split_ratios_low

    self.Q_loc_high += np.diag(alloc_high)
    self.Q_loc_low += np.diag(alloc_low)

    tx_high_add, tx_low_add = alloc_high.copy(), alloc_low.copy()
    np.fill_diagonal(tx_high_add, 0.0)
    np.fill_diagonal(tx_low_add, 0.0)
    self.Q_tx_high += tx_high_add
    self.Q_tx_low += tx_low_add

    mu_exe_mbits = (f_opt_array * self.micro_dt) / (self.C * 1e6)
    mu_high_limit = mu_exe_mbits * 0.8
    actual_exe_high = np.minimum(mu_high_limit, self.Q_loc_high)
    self.Q_loc_high -= actual_exe_high

    remaining_mu_exe = mu_exe_mbits - actual_exe_high
    actual_exe_low = np.minimum(remaining_mu_exe, self.Q_loc_low)
    self.Q_loc_low -= actual_exe_low

    energy_comp_per_uav = self.kappa * (f_opt_array**3) * self.micro_dt
    total_E_comp = np.sum(energy_comp_per_uav)

    diff = self.uav_coords[:, None, :] - self.uav_coords[None, :, :]
    dist_2d_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_2d_sq, np.inf)

    channel_gain = self.beta_0 / np.maximum(dist_2d_sq, 1e-8)
    np.fill_diagonal(channel_gain, 0.0)

    tx_demand_mask = (self.Q_tx_high + self.Q_tx_low) > 1e-9
    np.fill_diagonal(tx_demand_mask, False)
    num_active_links = np.count_nonzero(tx_demand_mask)

    link_bandwidth = np.zeros((self.N, self.N))
    if num_active_links > 0:
        link_bandwidth[tx_demand_mask] = self.B_u2u_total / num_active_links

    rate_bps = link_bandwidth * np.log2(
        1.0 + (self.p_u2u * channel_gain) / self.noise_power
    )
    mu_trans_mbits = (rate_bps * self.micro_dt) / 1e6

    actual_trans_high = np.minimum(mu_trans_mbits, self.Q_tx_high)
    self.Q_tx_high -= actual_trans_high

    remaining_mu_trans = mu_trans_mbits - actual_trans_high
    actual_trans_low = np.minimum(remaining_mu_trans, self.Q_tx_low)
    self.Q_tx_low -= actual_trans_low

    self.Q_loc_high += np.sum(actual_trans_high, axis=0)
    self.Q_loc_low += np.sum(actual_trans_low, axis=0)

    energy_uplink_per_uav = np.zeros(self.N, dtype=np.float32)
    for job in self.pending_upload_jobs:
        target = int(job["target_uav"])
        if 0 <= target < self.N:
            energy_uplink_per_uav[target] += (
                job["active_users"] * self.p_gt * self.micro_dt
            )
    total_E_uplink = float(np.sum(energy_uplink_per_uav))
    upload_delay_cost_high = np.sum(
        [job["high_bits"] * job["remaining_steps"] * self.micro_dt for job in self.pending_upload_jobs]
    )
    upload_delay_cost_low = np.sum(
        [job["low_bits"] * job["remaining_steps"] * self.micro_dt for job in self.pending_upload_jobs]
    )
    upload_delay_cost = 5.0 * upload_delay_cost_high + 1.0 * upload_delay_cost_low
    upload_delay_cost_per_uav = np.zeros(self.N, dtype=np.float32)
    for job in self.pending_upload_jobs:
        target = int(job["target_uav"])
        upload_delay_cost_per_uav[target] += (
            5.0 * job["high_bits"] * job["remaining_steps"] * self.micro_dt
            + 1.0 * job["low_bits"] * job["remaining_steps"] * self.micro_dt
        )

    active_u2u_links = ((actual_trans_high > 0) | (actual_trans_low > 0)).astype(float)
    energy_trans_per_uav = np.sum(
        active_u2u_links * self.p_u2u * self.micro_dt,
        axis=1,
    ).astype(np.float32)
    total_E_trans = float(np.sum(energy_trans_per_uav))

    q_high_tot = np.sum(self.Q_loc_high) + np.sum(self.Q_tx_high)
    q_low_tot = np.sum(self.Q_loc_low) + np.sum(self.Q_tx_low)
    total_energy = total_E_comp + total_E_uplink + total_E_trans

    info = {
        "queue_length": q_high_tot + q_low_tot,
        "q_high_tot": q_high_tot,
        "q_low_tot": q_low_tot,
        "total_energy": total_energy,
        "energy_comp": total_E_comp,
        "energy_uplink": total_E_uplink,
        "energy_trans": total_E_trans,
        "energy_comp_per_uav": energy_comp_per_uav.copy(),
        "energy_uplink_per_uav": energy_uplink_per_uav.copy(),
        "energy_trans_per_uav": energy_trans_per_uav.copy(),
        "energy_total_per_uav": (
            energy_comp_per_uav
            + energy_uplink_per_uav
            + energy_trans_per_uav
        ).copy(),
        "upload_delay_cost": upload_delay_cost,
        "upload_delay_cost_high": upload_delay_cost_high,
        "upload_delay_cost_low": upload_delay_cost_low,
        "upload_delay_cost_per_uav": upload_delay_cost_per_uav.copy(),
        "cpu_utilization_mean": np.mean(f_opt_array / np.maximum(self.F_max_array, 1e-8)),
        "cpu_utilization_per_uav": (f_opt_array / np.maximum(self.F_max_array, 1e-8)).copy(),
        "arrival_high_sum": np.sum(self.A_t_high),
        "arrival_low_sum": np.sum(self.A_t_low),
        "arrival_total_per_uav": (self.A_t_high + self.A_t_low).copy(),
        "Q_loc_high_sum": np.sum(self.Q_loc_high),
        "Q_loc_low_sum": np.sum(self.Q_loc_low),
        "Q_tx_high_sum": np.sum(self.Q_tx_high),
        "Q_tx_low_sum": np.sum(self.Q_tx_low),
        "Q_loc_total": np.sum(self.Q_loc_high) + np.sum(self.Q_loc_low),
        "Q_tx_total": np.sum(self.Q_tx_high) + np.sum(self.Q_tx_low),
    }

    for job in self.pending_upload_jobs:
        job["remaining_steps"] -= 1
    self.global_micro_step += 1
    self._generate_traffic()
    self.record_trajectory_snapshot()

    return self.get_raw_state(), info


def patch_joint_sca_micro_step(wrapper, seed):
    """
    Attach the SCA micro-step implementation to the underlying simulator.
    This keeps the public environment code unchanged.
    """
    base_env = wrapper.env
    base_env.micro_step = types.MethodType(_sca_micro_step, base_env)


def dummy_micro_policy(env, deterministic=False):
    return np.zeros((2 * env.N * env.N,), dtype=np.float32)


def _build_expected_user_task_bits(env) -> np.ndarray:
    """
    Build a per-user load proxy from the simulator's traffic parameters.
    The simulator stores task sizes in Mbit-style units, so we convert them
    back to bit units here to match the paper notation.
    """
    base_env = env.env
    expected_load_mbits = (
        base_env.user_lambda_high * base_env.user_task_size_high_mbits
        + base_env.user_lambda_low * base_env.user_task_size_low_mbits
    ) * base_env.arrival_dt
    return expected_load_mbits.astype(np.float64) * 1e6


def solve_sca_micro_action(env, deterministic=False):
    """
    Wrapper used by the benchmark runner.
    It solves the SCA problem on the current state and converts the user-level
    rho solution into the row-stochastic matrix expected by the simulator.
    """
    raw_state = env.env.get_raw_state()
    base_env = env.env

    task_bits = _build_expected_user_task_bits(env)
    cpu_cycles = np.full(base_env.M, base_env.C, dtype=np.float64)
    user_to_uav = base_env.user_allocations.copy()
    u2u_rates_mbps = compute_u2u_rate_mbps(env, raw_state)
    u2u_rates_bps = u2u_rates_mbps.astype(np.float64) * 1e6
    phi = np.full(base_env.N, base_env.kappa, dtype=np.float64)
    f_max = base_env.F_max_array.astype(np.float64)

    rho, f, info = optimize_allocation_SCA(
        task_bits=task_bits,
        cpu_cycles_per_bit=cpu_cycles,
        user_to_uav=user_to_uav,
        u2u_rates_bps=u2u_rates_bps,
        phi=phi,
        f_max=f_max,
        tx_power=float(base_env.p_u2u),
        omega_D=1.0,
        omega_E=1.0,
        max_iter=10,
        tol=1e-3,
        proximal_weight=1e-4,
        solver=None,
        verbose=False,
    )

    # Aggregate user-level rho into the micro-action matrix used by the env.
    alpha = np.eye(env.N, dtype=np.float64)
    for m in range(env.N):
        user_idx = np.where(user_to_uav == m)[0]
        if user_idx.size == 0:
            continue

        weights = task_bits[user_idx]
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-12:
            continue

        alpha[m] = np.sum(rho[user_idx] * weights[:, None], axis=0) / weight_sum
        alpha[m] = np.maximum(alpha[m], 1e-12)
        alpha[m] = alpha[m] / np.sum(alpha[m])

    micro_action = alpha_to_micro_action(
        alpha,
        diagonal_bias=float(getattr(env, "micro_diag_bias", 0.5)),
        action_scale=float(getattr(env, "micro_action_scale", 1.0)),
    )
    return micro_action.astype(np.float32)


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
