"""Separate 85/86-dimensional public observations on the deployment clocks."""
import copy
from .industrial_mdp import IndustrialMDP
from .industrial_deployment_env import IndustrialDeploymentEnv
from .industrial_deployment_observation import DeploymentObservationEncoder


class DeploymentIndustrialMDP(IndustrialMDP):
    def __init__(self, config, *, ledger_enabled=False):
        self.env = IndustrialDeploymentEnv(config, ledger_enabled=ledger_enabled)
        self.config = self.env.config
        self.deployment_config = self.env.deployment_config
        self.scene_wrapper = self.env.scene_wrapper
        self.encoder = DeploymentObservationEncoder(self.config, self.deployment_config)
        self.frames_per_macro = round(self.deployment_config["macro_period_s"]/self.config["time"]["routing_s"])
        self.gamma_frame = self.config["objective"]["gamma_per_routing_frame"]
        self.gamma_step = self.gamma_frame**(1/self.env.frame_steps)
        self.gamma_macro = self.gamma_frame**self.frames_per_macro
        self.last_motion = None

    def reset(self, seed, layout_seed=None):
        self.last_motion = None
        self.env.reset(seed, layout_seed)
        return self.macro_observation()

    def start_macro(self, action):
        motion = self.env.start_motion(action)
        self.last_motion = copy.deepcopy(motion)
        return motion

    def controller_information(self):
        return self.env.deployment_view()

    def observation_spec(self):
        return self.encoder.spec()

    def macro_observation(self):
        return self.encoder.encode_macro(self.env.deployment_view())

    def prepare_service(self):
        self.env.prepare_frame()
        return self.encoder.encode_micro(self.env.deployment_view())
