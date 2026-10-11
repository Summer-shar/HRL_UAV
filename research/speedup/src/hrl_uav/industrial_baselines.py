"""Causal lower-level baselines. The CVX objective is a declared batch proxy."""
from types import SimpleNamespace
import numpy as np


def local_scores(view):
    return np.tile(np.eye(3), (2, 1, 1)).ravel()


def hover_scores(view):
    return np.zeros(6)


class RandomRouting:
    def __init__(self, seed):
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), 9001]))

    def __call__(self, view):
        return self.rng.dirichlet(np.ones(3), size=6).ravel()


class GreedyRouting:
    """One destination per row, using only current report queues and geometry.

    Each candidate activates its link in addition to currently backlogged links.
    Candidate rows are evaluated against the same report, not a speculative
    sequence of assignments. Ties choose the smallest destination index.
    """
    def __init__(self, config):
        from .industrial_env import IndustrialPhysics
        self.physics = IndustrialPhysics(config, ())

    def estimates(self, view):
        bits = np.zeros((2, 3))
        for task in view.ready_tasks:
            bits[task.priority, task.source] += task.input_bits
        cpu, tx = np.asarray(view.compute_cycles), np.asarray(view.transfer_bits)
        inbound = np.asarray(view.transfer_cycles)
        active = tx.sum(axis=0) > 0
        times = np.zeros((2, 3, 3))
        for p in range(2):
            workload = cpu[:p+1].sum(axis=0) + inbound[:p+1].sum(axis=(0, 1))
            for source in range(3):
                times[p, source] = workload / np.asarray(view.cpu_hz)
                for destination in range(3):
                    if destination == source:
                        continue
                    candidate = active.copy()
                    candidate[source, destination] = True
                    allocation = self.physics.allocate_u2u_resources(candidate, view.positions_m)
                    rate = allocation['u2u_rate_bps'][source, destination]
                    times[p, source, destination] += (tx[:, source, destination].sum() + bits[p, source]) / rate
        return times

    def __call__(self, view):
        destinations = self.estimates(view).argmin(axis=2)
        return np.eye(3)[destinations].ravel()


class CVXRouting:
    def __init__(self, config):
        # Reuse the checked support-enumeration LP, not the old 68D decoder.
        from .xu_cvx_r import XuCVXRouting
        o = config["objective"]
        # Propulsion is action-independent for this lower-routing proxy. Its
        # weighted constant is omitted; CPU/U2U energy coefficients stay intact.
        self.solver = XuCVXRouting(SimpleNamespace(
            priority_high=o["priority_weights"][0], priority_low=o["priority_weights"][1],
            delay_weight=o["holding_weight"], energy_weight=1-o["holding_weight"],
            holding_norm=o["holding_reference_task_s"], energy_norm=o["energy_reference_j"]))
        self.kappa = config["resources"]["cpu_kappa"]
        self.records = []

    def coefficients(self, view):
        bits, cycles = np.zeros((2, 3)), np.zeros((2, 3))
        for task in view.ready_tasks:
            bits[task.priority, task.source] += task.input_bits
            cycles[task.priority, task.source] += task.compute_cycles
        cpu, tx = np.asarray(view.compute_cycles), np.asarray(view.transfer_bits)
        tx_cycles = np.asarray(view.transfer_cycles)
        frequencies = np.asarray(view.cpu_hz)
        rates, power = np.asarray(view.u2u_all_active_rate_bps), np.asarray(view.u2u_all_active_power_w)
        wait, duration, energy = (np.zeros((6, 3)) for _ in range(3))
        for priority in range(2):
            ahead = cpu[:priority+1].sum(axis=0) + tx_cycles[:priority+1].sum(axis=(0, 1))
            for source in range(3):
                row, work, data = priority*3+source, cycles[priority, source], bits[priority, source]
                wait[row] = ahead/frequencies
                duration[row] = work/frequencies
                energy[row] = self.kappa*frequencies**2*work
                for destination in range(3):
                    if destination != source:
                        rate = rates[source, destination]
                        wait[row, destination] += tx[:priority+1, source, destination].sum()/rate
                        duration[row, destination] += data/rate
                        energy[row, destination] += power[source, destination]*data/rate
        return wait, duration, energy, bits.ravel() == 0

    def __call__(self, view):
        wait, duration, energy, empty = self.coefficients(view)
        if np.all(empty):
            self.records.append(dict(status="empty_previous_route", fallback=False))
            return self.solver.previous.ravel().copy()
        try:
            scores = self.solver.solve_coefficients(wait, duration, energy, empty)
            self.records.append(dict(self.solver.records[-1], fallback=False))
            return scores
        except Exception as error:
            # Fixed, visible failure rule, not an unreported performance adjustment.
            scores = local_scores(view)
            self.solver.previous = scores.reshape(6, 3).copy()
            self.records.append(dict(status="solver_failure_local", fallback=True,
                                     error=f"{type(error).__name__}: {error}"))
            return scores
