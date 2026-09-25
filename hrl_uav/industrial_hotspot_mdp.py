"""Hotspot scene adapter retaining the exact public 85/86 observation encoding."""
from .industrial_deployment_mdp import DeploymentIndustrialMDP
from .industrial_deployment_observation import DeploymentObservationEncoder
from .industrial_hotspot_env import HotspotIndustrialEnv


class HotspotIndustrialMDP(DeploymentIndustrialMDP):
    def __init__(self, config, *, ledger_enabled=False):
        self.env = HotspotIndustrialEnv(config, ledger_enabled=ledger_enabled)
        self.config = self.env.config
        self.deployment_config = self.env.deployment_config
        self.scene_wrapper = self.env.scene_wrapper
        self.encoder = DeploymentObservationEncoder(self.config, self.deployment_config)
        self.frames_per_macro = round(self.deployment_config["macro_period_s"]/self.config["time"]["routing_s"])
        self.gamma_frame = self.config["objective"]["gamma_per_routing_frame"]
        self.gamma_step = self.gamma_frame**(1/self.env.frame_steps)
        self.gamma_macro = self.gamma_frame**self.frames_per_macro
        self.last_motion = None

    def reset(self, seed, layout_seed=None, load_seed=None, realization=None):
        self.last_motion = None
        self.env.reset(seed, layout_seed, load_seed, realization)
        self.config = self.env.config
        # Type composition is not read by the encoder; refreshing its fixed
        # configuration nevertheless keeps the adapter's effective state honest.
        self.encoder = DeploymentObservationEncoder(self.config, self.deployment_config)
        return self.macro_observation()
