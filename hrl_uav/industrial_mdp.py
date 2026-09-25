"""Two action clocks with one physical cost and consistent elapsed-time discount."""
import numpy as np
from .industrial_env import IndustrialEnv
from .industrial_observation import ObservationEncoder
from .routing import project_rows_to_simplex


class IndustrialMDP:
    macro_action_dim, micro_action_dim = 6, 18

    def __init__(self, config, *, ledger_enabled=False):
        self.env = IndustrialEnv(config, ledger_enabled=ledger_enabled)
        self.config = self.env.config
        self.encoder = ObservationEncoder(self.config)
        t = self.config["time"]
        self.frames_per_macro = round(t["motion_s"]/t["routing_s"])
        self.gamma_frame = self.config["objective"]["gamma_per_routing_frame"]
        self.gamma_step = self.gamma_frame**(1/self.env.frame_steps)
        self.gamma_macro = self.gamma_frame**self.frames_per_macro

    def reset(self, seed, layout_seed=None):
        return self.encoder.encode(self.env.reset(seed, layout_seed))

    def controller_information(self):
        return self.env.view()

    def observation_spec(self):
        return self.encoder.spec(self.env.view())

    def macro_observation(self):
        return self.encoder.encode(self.env.view())

    def prepare_service(self):
        return self.encoder.encode(self.env.prepare_frame())

    def start_macro(self, action):
        return self.env.start_motion(action)

    def route(self, scores):
        scores = np.asarray(scores, dtype=float)
        if scores.size != 18 or not np.isfinite(scores).all() or np.any(np.abs(scores) > 1+1e-7):
            raise ValueError("Expected 18 finite routing scores within [-1,1]")
        return project_rows_to_simplex(scores.reshape(2, 3, 3))

    def step_frame(self, scores):
        self.env.prepare_frame()
        route = self.route(scores)
        readings = [self.env.micro_step(route) for _ in range(self.env.frame_steps)]
        return dict(reward=-sum(self.gamma_step**i*x["cost"] for i, x in enumerate(readings)),
                    cost=sum(x["cost"] for x in readings), route=route,
                    discount=self.gamma_frame, readings=readings)

    def metrics(self):
        return self.env.metrics()
