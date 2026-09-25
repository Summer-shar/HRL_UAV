"""Deployment policy adapters and stage clocks, without importing torch."""
import copy
import time

import numpy as np

from .industrial_rollout import store_transitions


class DeploymentLearningControllers:
    def __init__(self, agent, encoder, *, deterministic=True, macro_enabled=True):
        self.agent = agent
        self.encoder = encoder
        self.deterministic = deterministic
        self.macro_enabled = macro_enabled

    def macro(self, view):
        return self.agent.macro_action(self.encoder.encode_macro(view),
            deterministic=self.deterministic, enabled=self.macro_enabled)

    def micro(self, view):
        return self.agent.sac_actor.choose_action(self.encoder.encode_micro(view),
            deterministic=self.deterministic)


class DeploymentTrainingSchedule:
    """Update PPO before priming the next command, SAC after real successors."""
    def __init__(self, agent, method, protocol, *, on_update=None):
        self.agent, self.method, self.protocol = agent, method, protocol
        self.on_update = on_update
        self.stage = "pretrain"
        self.macros = self.frames = self.physical_steps = 0
        self.stage_macros = {"pretrain": 0, "joint": 0}
        self.stage_frames = {"pretrain": 0, "joint": 0}
        self.ppo_samples = self.consumed_ppo_samples = 0
        self.ppo_seconds = self.sac_seconds = 0.
        self.stage_macro_target = None

    def set_stage(self, stage, target_macros):
        if stage not in ("pretrain", "joint"):
            raise ValueError("Unknown training stage")
        if stage == "pretrain" and self.method != "ppo_sac":
            raise ValueError("Only SAC has a pretraining stage")
        self.stage, self.stage_macro_target = stage, int(target_macros)
        self.agent.set_stage(stage)

    def _emit(self, layer, stats):
        if self.on_update is not None:
            self.on_update(dict(layer=layer, stage=self.stage, macros=self.macros,
                frames=self.frames, stats=copy.deepcopy(stats)))

    def macro_boundary(self, transition):
        self.macros += 1
        self.stage_macros[self.stage] += 1
        if self.stage == "pretrain":
            if np.any(transition.action != 0):
                raise AssertionError("SAC pretraining requires strictly zero upper actions")
            if self.agent.macro_rollout:
                raise AssertionError("Pretraining must not contain PPO samples")
            return
        self.agent.macro_rollout.append((transition.state, transition.action,
            transition.logprob, transition.reward, transition.next_state, transition.trace_continue))
        self.ppo_samples += 1
        before, batch_size = self.agent.ppo_updates, len(self.agent.macro_rollout)
        started = time.perf_counter()
        self.agent.update_ppo(force=self.stage_macros[self.stage] == self.stage_macro_target)
        self.ppo_seconds += time.perf_counter() - started
        if self.agent.ppo_updates != before:
            if self.agent.macro_rollout:
                raise AssertionError("PPO update did not consume its complete behavior batch")
            self.consumed_ppo_samples += batch_size
            self._emit("ppo", self.agent.last_ppo_stats)

    def completed_micro(self, collected):
        self.physical_steps += collected["physical_steps"]
        for transition in collected["micro"]:
            self.frames += 1
            self.stage_frames[self.stage] += 1
            if self.method != "ppo_sac":
                continue
            store_transitions(self.agent, {"micro": [transition]}, macro=False)
            if self.frames % self.protocol["learning"]["sac_update_frames"]:
                continue
            before = self.agent.sac_updates
            started = time.perf_counter()
            self.agent.update_sac()
            self.sac_seconds += time.perf_counter() - started
            if self.agent.sac_updates != before:
                self._emit("sac", self.agent.last_sac_stats)

    def state_dict(self):
        names = ("stage", "macros", "frames", "physical_steps", "stage_macros",
            "stage_frames", "ppo_samples", "consumed_ppo_samples", "ppo_seconds",
            "sac_seconds", "stage_macro_target")
        return {name: copy.deepcopy(getattr(self, name)) for name in names}

    def load_state_dict(self, state):
        for name in self.state_dict():
            if name not in state:
                raise ValueError("Missing schedule state: " + name)
            setattr(self, name, copy.deepcopy(state[name]))
        if self.ppo_samples != self.consumed_ppo_samples + len(self.agent.macro_rollout):
            raise ValueError("PPO sample cursor and pending behavior batch disagree")
