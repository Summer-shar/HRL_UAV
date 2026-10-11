"""Hotspot reset adapter; deployment and task service remain the audited V3 code."""
from collections import deque

import numpy as np

from .industrial_env import IndustrialPhysics
from .industrial_deployment_env import IndustrialDeploymentEnv
from .industrial_hotspot_scene import HotspotIndustrialScene, validate_hotspot_config
from .industrial_scene import workload_summary


class HotspotIndustrialEnv(IndustrialDeploymentEnv):
    def __init__(self, config, *, ledger_enabled=False):
        # Do not call the old constructor: IndustrialEnv.__init__ invokes reset
        # virtually before subclass initialization, creating a legacy scene.
        self.scene_wrapper = validate_hotspot_config(config)
        self.deployment_config = self.scene_wrapper["deployment"]
        self.ledger_enabled = bool(ledger_enabled)
        t = self.scene_wrapper["base_scene"]["time"]
        self.dt = t["physical_s"]
        self.frame_steps = round(t["routing_s"]/self.dt)
        self.macro_steps = round(self.deployment_config["macro_period_s"]/self.dt)
        self.adjustment_steps = round(self.deployment_config["adjustment_s"]/self.dt)
        self.reset(0)

    def reset(self, seed, layout_seed=None, load_seed=None, realization=None):
        self._scene = HotspotIndustrialScene(self.scene_wrapper, seed, layout_seed, load_seed, realization)
        self.config = self._scene.config
        self.physics = IndustrialPhysics(self.config, self._scene.terminals)
        self.parents, self.fragments = {}, {}
        self._reset_active_state()
        self.upload = [deque() for _ in range(39)]
        self.compute = [[deque() for _ in range(3)] for _ in range(2)]
        self.transfer = [[[deque() for _ in range(3)] for _ in range(3)] for _ in range(2)]
        self.ready, self._recent = deque(), deque()
        self._prepared_frame = -1
        self.step_index = self.revision = self.motion_ticks = self._macro_end_tick = 0
        self.velocity = np.zeros((3, 2))
        self._target = self.physics.uav_coords.copy()
        self.ledger = []
        self.holding = self.cost_sum = 0.
        self.energy = dict(upload=0., u2u=0., cpu=0., propulsion=0.)
        self.generated_bits = self.generated_cycles = self.upload_served_bits = 0.
        self.tx_served_bits = self.cpu_served_cycles = 0.
        self._arrival_totals_by_region = np.zeros((3, 2, 3))
        self._alive_counts = np.zeros(2, dtype=np.int64)
        self.unfinished_parent_time_s_by_priority = np.zeros(2)
        self._expected_workload = workload_summary(self.config)
        self._publish("reset_before_arrivals")
        return self.view()

    def _admit(self, arrival):
        super()._admit(arrival)
        self._arrival_totals_by_region[arrival.source, arrival.priority] += (
            arrival.input_bits, arrival.compute_cycles, 1)
        self._alive_counts[arrival.priority] += 1

    def _complete(self, fragment):
        parent = self.parents[fragment.parent_id]
        was_unfinished = parent.finish_s is None
        super()._complete(fragment)
        if was_unfinished and parent.finish_s is not None:
            self._alive_counts[parent.arrival.priority] -= 1

    def micro_step(self, route):
        # Match the inherited cost's step-start population, including parents
        # finishing at this step's end. Only successful service steps integrate.
        alive_at_start = self._alive_counts.copy()
        result = super().micro_step(route)
        self.unfinished_parent_time_s_by_priority += alive_at_start*self.dt
        return result

    @property
    def realization(self):
        # A detached audit artifact; it is not a public controller observation.
        import copy
        return copy.deepcopy(self._scene.realization)

    def metrics(self):
        result = super().metrics()
        result["canonical_event_sha256"] = self._scene.canonical_event_digest.hexdigest()
        result["realization_sha256"] = self._scene.realization["realization_sha256"]
        integral = self.unfinished_parent_time_s_by_priority
        result["unfinished_parent_time_s_by_priority"] = integral.tolist()
        result["unfinished_parent_time_s"] = float(integral.sum())
        result["average_unfinished_parents"] = float(integral.sum()/self.time_s) if self.time_s else 0.
        rows = []
        for region, priority in enumerate(self._arrival_totals_by_region):
            total = priority.sum(axis=0)
            rows.append(dict(region=region, tasks=int(total[2]), input_bits=float(total[0]),
                compute_cycles=float(total[1]), priority=[dict(priority=p, tasks=int(v[2]),
                input_bits=float(v[0]), compute_cycles=float(v[1])) for p, v in enumerate(priority)]))
        result["arrivals_by_region"] = rows
        import copy
        result["expected_workload"] = copy.deepcopy(self._expected_workload)
        return result
