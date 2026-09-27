"""Explicit fixed-CPU, aggregate Xu/CVX routing specialization, not full SCA.

See docs/mission_xu_cvx_r.md for the one-batch predictor and its limitations.
Only the SAC observation and static constants enter this controller.
"""

from dataclasses import dataclass
from itertools import product
import time

import cvxpy as cp
import numpy as np

from .mdp import MICRO_FEATURES


WIDTHS = dict(compute_queue=6, transfer_queue=18, upload_queue=6, unfinished_tasks=6,
              oldest_age=6, current_arrivals=6, current_load=3, cpu_capacity=3,
              u2u_spectral_efficiency=9, u2g_mean_rate=3, phase_age=1, macro_clock=1)
SLICES = {}
_offset = 0
for _name in MICRO_FEATURES:
    SLICES[_name] = (_offset, _offset + WIDTHS[_name])
    _offset += WIDTHS[_name]
assert _offset == 68
SUPPORTS = np.asarray([s for s in product((0., 1.), repeat=3) if any(s)])


@dataclass(frozen=True)
class StaticModel:
    bandwidth_hz: float
    noise_psd: float
    tx_budget_w: float
    reference_bandwidth_hz: float
    reference_power_w: float
    cycles_per_bit: float
    kappa: float
    priority_high: float
    priority_low: float
    delay_weight: float
    energy_weight: float
    holding_norm: float
    energy_norm: float

    def __post_init__(self):
        for key, value in vars(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid static parameter {key}")
            if key not in ("kappa", "delay_weight", "energy_weight") and value == 0:
                raise ValueError(f"Zero static parameter {key}")
        if self.delay_weight <= 0:
            raise ValueError("The frozen protocol requires positive delay weight")


def decode_observation(state):
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (68,) or not np.isfinite(state).all():
        raise ValueError("Expected the finite 68D split_v1 SAC observation")
    def extract(name, shape, scale):
        lo, hi = SLICES[name]
        encoded = state[lo:hi].reshape(shape)
        if np.any(encoded < -1e-7):
            raise ValueError(f"Negative nonnegative observation feature: {name}")
        return np.expm1(np.maximum(encoded, 0)) * scale
    result = dict(compute=extract("compute_queue", (2, 3), 10),
                  transfer=extract("transfer_queue", (2, 3, 3), 10),
                  arrivals=extract("current_arrivals", (2, 3), 1),
                  eta=extract("u2u_spectral_efficiency", (3, 3), 1))
    lo, hi = SLICES["cpu_capacity"]
    result["capacity"] = state[lo:hi] * 10
    if np.any(result["capacity"] <= 0) or not all(np.isfinite(v).all() for v in result.values()):
        raise ValueError("Invalid decoded state")
    return result


def hypothetical_full_link_resources(eta, model):
    """All-six-active prediction, not the environment's actual resource allocation."""
    gain = np.expm1(np.asarray(eta) * np.log(2)) * model.noise_psd * model.reference_bandwidth_hz / model.reference_power_w
    active = ~np.eye(3, dtype=bool)
    bandwidth = active * (model.bandwidth_hz / 6)
    power = active * (model.tx_budget_w / 2)
    noise = model.noise_psd * bandwidth
    snr = np.divide(power * gain, noise, out=np.zeros((3, 3)), where=active)
    rates = bandwidth * np.log2(1+snr) / 1e6
    if not np.isfinite(rates).all() or np.any(rates[active] <= 0):
        raise ValueError("Invalid hypothetical link rates")
    return bandwidth, power, noise, rates


def predictor_coefficients(decoded, model):
    capacity = decoded["capacity"]
    _, power, _, rates = hypothetical_full_link_resources(decoded["eta"], model)
    frequencies = capacity * model.cycles_per_bit * 1e6
    wait, duration, energy = (np.zeros((6, 3)) for _ in range(3))
    for priority in range(2):
        compute_ahead = decoded["compute"][:priority+1].sum(axis=0)
        transfer_ahead = decoded["transfer"][:priority+1].sum(axis=0)
        for source in range(3):
            row = priority*3+source
            bits = decoded["arrivals"][priority, source]
            wait[row] = compute_ahead / capacity
            duration[row] = bits / capacity
            energy[row] = model.kappa * model.cycles_per_bit * bits*1e6 * frequencies**2
            for destination in range(3):
                if destination != source:
                    wait[row, destination] += transfer_ahead[source, destination]/rates[source, destination]
                    duration[row, destination] += bits/rates[source, destination]
                    energy[row, destination] += power[source, destination]*bits/rates[source, destination]
    return wait, duration, energy


def proxy_objective(route, wait, duration, energy, delay_weights, energy_weight):
    active = route > 0
    branch_time = np.where(active, wait + duration*route, 0)
    return delay_weights*branch_time.max(axis=1) + energy_weight*(energy*route).sum(axis=1)


class XuCVXRouting:
    def __init__(self, model):
        self.model = model
        self.delay_weights = model.delay_weight/model.holding_norm * np.repeat(
            (model.priority_high, model.priority_low), 3)
        self.energy_weight = model.energy_weight/model.energy_norm
        self.previous = np.tile(np.eye(3), (2, 1, 1)).reshape(6, 3)
        self.calls = 0
        self.records = []
        self.candidate_costs = []
        self.mask = np.tile(SUPPORTS, (6, 1))
        self.wait = cp.Parameter((42, 3), nonneg=True)
        self.duration = cp.Parameter((42, 3), nonneg=True)
        self.energy = cp.Parameter((42, 3), nonneg=True)
        self.rho = cp.Variable((42, 3), nonneg=True)
        self.tail = cp.Variable(42, nonneg=True)
        self.problem = cp.Problem(cp.Minimize(
            np.repeat(self.delay_weights, 7) @ self.tail
            + self.energy_weight*cp.sum(cp.multiply(self.energy, self.rho))),
            [cp.sum(self.rho, axis=1) == 1, self.rho <= self.mask,
             cp.reshape(self.tail, (42, 1), order="C") >= cp.multiply(
                 self.mask, self.wait + cp.multiply(self.duration, self.rho))])
        if not self.problem.is_dcp() or not self.problem.is_dpp():
            raise AssertionError("The fixed-CPU subproblem must be a cached convex LP")

    def solve_coefficients(self, wait, duration, energy, empty_rows=None):
        for values in (wait, duration, energy):
            if np.shape(values) != (6, 3) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("Predictor coefficients must be finite nonnegative 6x3 arrays")
        for parameter, value in ((self.wait, wait), (self.duration, duration), (self.energy, energy)):
            parameter.value = np.repeat(value, 7, axis=0)
        start = time.perf_counter()
        self.problem.solve(solver="CLARABEL", warm_start=True, max_iter=200,
                           tol_gap_abs=1e-8, tol_gap_rel=1e-8, tol_feas=1e-8)
        elapsed = time.perf_counter()-start
        if self.problem.status != cp.OPTIMAL:
            raise RuntimeError(f"CVX solve failed: {self.problem.status}")
        raw = np.asarray(self.rho.value)
        tails = np.asarray(self.tail.value)
        if not np.isfinite(raw).all() or not np.isfinite(tails).all():
            raise FloatingPointError("Nonfinite CVX solution")
        residual = max(float(np.max(np.abs(raw.sum(axis=1)-1))), float(np.max(-raw)),
                       float(np.max(raw-self.mask)), float(np.max(
                           self.mask*(self.wait.value+self.duration.value*raw)-tails[:, None])))
        tolerance = 1e-5*(1+float(np.max(wait+duration)))
        if residual > tolerance:
            raise AssertionError(f"Material LP infeasibility {residual} > {tolerance}")
        cleaned = np.maximum(raw, 0)*self.mask
        cleaned[cleaned < 1e-8] = 0
        cleaned /= cleaned.sum(axis=1, keepdims=True)
        cleanup = float(np.max(np.abs(raw-cleaned)))
        if cleanup > 1e-5:
            raise AssertionError("Excessive numerical route cleanup")
        # Rank the declared-support LPs, not a different objective after cleanup.
        # The separately enumerated smaller support covers zero-weight branches.
        values = (np.repeat(self.delay_weights, 7)
                  * (self.mask*(self.wait.value+self.duration.value*cleaned)).max(axis=1)
                  + self.energy_weight*(self.energy.value*cleaned).sum(axis=1)).reshape(6, 7)
        selected, route = [], np.empty((6, 3))
        for row in range(6):
            minimum = values[row].min()
            tied = np.flatnonzero(values[row] <= minimum + 1e-8*(1+abs(minimum)))
            index = min(tied, key=lambda j: (np.count_nonzero(cleaned[row*7+j]), int(j)))
            selected.append(int(index))
            route[row] = cleaned[row*7+index]
        if empty_rows is not None:
            route[empty_rows] = self.previous[empty_rows]
        self.previous = route.copy()
        self.calls += 1
        record = dict(decision=self.calls, solve_seconds=elapsed, status=self.problem.status,
                      solver_iterations=int(self.problem.solver_stats.num_iters),
                      primal_residual=residual, feasibility_tolerance=tolerance, max_cleanup=cleanup,
                      predicted_cost=float(proxy_objective(route, wait, duration, energy,
                          self.delay_weights, self.energy_weight).sum()),
                      empty_rows=int(np.count_nonzero(empty_rows)) if empty_rows is not None else 0)
        record.update({f"support_row{i}": int(j) for i, j in enumerate(selected)})
        self.records.append(record)
        self.candidate_costs.append(values)
        return route.ravel().copy()

    def choose_action(self, state, deterministic=False):
        if not deterministic:
            raise ValueError("Only the deterministic comparison interface is allowed")
        decoded = decode_observation(state)
        coefficients = predictor_coefficients(decoded, self.model)
        return self.solve_coefficients(*coefficients, empty_rows=decoded["arrivals"].ravel() <= 1e-12)
