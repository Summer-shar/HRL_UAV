"""Synchronous adjustment followed by holding, with continuous inherited service."""
import numpy as np

from .industrial_env import IndustrialEnv
from .industrial_deployment_scene import validate_deployment_config
from .industrial_deployment_observation import DeploymentControllerView


class IndustrialDeploymentEnv(IndustrialEnv):
    def __init__(self, config, *, ledger_enabled=False):
        self.scene_wrapper = validate_deployment_config(config)
        self.deployment_config = self.scene_wrapper["deployment"]
        self._macro_end_tick = 0
        super().__init__(self.scene_wrapper["base_scene"], ledger_enabled=ledger_enabled)
        self.adjustment_steps = round(self.deployment_config["adjustment_s"]/self.dt)

    def reset(self, seed, layout_seed=None):
        # Parent construction calls this virtual method before our __init__ returns.
        self._macro_end_tick = 0
        return super().reset(seed, layout_seed)

    def deployment_view(self):
        remaining = max(0, self._macro_end_tick-self.step_index)*self.dt
        phase = ("awaiting_decision" if remaining == 0 else
                 "adjusting" if self.motion_ticks > 0 else "holding")
        return DeploymentControllerView(super().view(), remaining, phase)

    def start_motion(self, action):
        if (self.step_index % self.macro_steps or self.motion_ticks or
                self.step_index < self._macro_end_tick):
            raise ValueError("Deployment requires a completed, uncommitted macro boundary")
        raw = np.asarray(action, dtype=float)
        if raw.size != 6 or not np.isfinite(raw).all():
            raise ValueError("Expected six finite deployment scores")
        scaled = np.clip(raw.reshape(3, 2), -1., 1.)
        scaled /= np.maximum(np.linalg.norm(scaled, axis=1, keepdims=True), 1.)
        start = self.physics.uav_coords.copy()
        requested = np.clip(start + scaled*self.deployment_config["max_target_step_m"],
                            0., self.config["map_size_m"])
        actual, safety_scale = self.physics._project_joint_motion_to_safe_separation(
            start, requested-start)
        flight = self.deployment_config["adjustment_s"]
        self._target = start + actual
        self.velocity = actual/flight
        self.motion_ticks = self.adjustment_steps
        self._macro_end_tick = self.step_index+self.macro_steps
        self._publish(self._phase)
        return dict(raw_action=raw.ravel().copy(), requested_target_m=requested.copy(),
                    target_m=self._target.copy(), displacement_m=actual.copy(),
                    velocity_mps=self.velocity.copy(), flight_duration_s=flight,
                    macro_duration_s=self.deployment_config["macro_period_s"],
                    safety_scale=safety_scale,
                    minimum_separation_m=self.physics._minimum_separation_along_joint_motion(start, actual))
