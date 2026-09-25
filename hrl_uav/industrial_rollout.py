"""No-learning collection with real boundary successors, including upper actions."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    discount: float
    trace_continue: float
    truncated: bool
    route: object = None
    logprob: float = 0.


class IndustrialRollout:
    def __init__(self, mdp, macro_controller, micro_controller, *, on_macro_boundary=None):
        self.mdp = mdp
        self.macro_controller, self.micro_controller = macro_controller, micro_controller
        self.on_macro_boundary = on_macro_boundary
        self._macro = None
        self._pending_micro = None
        self.macro_calls = self.micro_calls = 0

    def _prime_macro(self):
        if self._macro is None:
            state = self.mdp.macro_observation()
            result = self.macro_controller(self.mdp.controller_information())
            action, logprob = result if isinstance(result, tuple) else (result, 0.)
            action = np.asarray(action).ravel().copy()
            self.mdp.start_macro(action)
            self._macro = (state, action, float(logprob))
            self.macro_calls += 1

    def run(self, macros, *, truncate=True, collect_metrics=True):
        if type(macros) is not int or macros < 1:
            raise ValueError("Positive whole macro count required")
        high, low = [], []
        start_step = self.mdp.env.step_index
        def finish_micro(next_state, is_truncated=False):
            if self._pending_micro is not None:
                state, action, reward, route = self._pending_micro
                low.append(Transition(state, action, reward, next_state.copy(), self.mdp.gamma_frame,
                                      float(not is_truncated), is_truncated, route))
                self._pending_micro = None
        for macro in range(macros):
            self._prime_macro()
            reward = 0.
            for frame in range(self.mdp.frames_per_macro):
                state = self.mdp.prepare_service()
                finish_micro(state)
                action = np.asarray(self.micro_controller(self.mdp.controller_information())).ravel().copy()
                self.micro_calls += 1
                result = self.mdp.step_frame(action)
                reward += self.mdp.gamma_frame**frame*result["reward"]
                self._pending_micro = (state, action, result["reward"], result["route"].copy())
            state, action, logprob = self._macro
            is_truncated = truncate and macro == macros-1
            high.append(Transition(state, action, reward, self.mdp.macro_observation().copy(),
                self.mdp.gamma_macro, float(not is_truncated), is_truncated, logprob=logprob))
            self._macro = None
            # A future training scheduler may consume this completed macro and
            # update PPO BEFORE selecting the next committed flight command.
            # This avoids stale on-policy actions and stale SAC successor velocity.
            if self.on_macro_boundary is not None:
                self.on_macro_boundary(high[-1])
        # Freeze window metrics BEFORE the next boundary's arrivals. No extra physical step.
        # Metrics are a read-only diagnostic. Callers may skip unused snapshots,
        # but requested snapshots must precede next-boundary arrivals below.
        metrics = self.mdp.metrics() if collect_metrics else None
        self._prime_macro()
        successor = self.mdp.prepare_service()
        finish_micro(successor, truncate)
        return dict(macro=high, micro=low, metrics=metrics,
                    physical_steps=self.mdp.env.step_index-start_step,
                    next_boundary_primed=True, next_boundary_physical_steps=0)


class LearningControllers:
    """Adapters only: no policy update or replay write is triggered by collection."""
    def __init__(self, agent, encoder, *, deterministic=True):
        self.agent, self.encoder, self.deterministic = agent, encoder, deterministic

    def macro(self, view):
        return self.agent.macro_action(self.encoder.encode(view), deterministic=self.deterministic)

    def micro(self, view):
        return self.agent.sac_actor.choose_action(self.encoder.encode(view), deterministic=self.deterministic)


def build_agent(mdp, training_config):
    """Fresh networks only. Training settings must be supplied by a separate protocol."""
    from .learning import DirectHRL
    training = dict(training_config)
    supplied = training.pop("macro_gamma", mdp.gamma_macro)
    if not np.isclose(supplied, mdp.gamma_macro):
        raise ValueError("Macro discount disagrees with physical duration")
    if "macro_gamma_frames" in training and training.pop("macro_gamma_frames") != mdp.frames_per_macro:
        raise ValueError("Legacy macro frame override is incompatible")
    training["macro_gamma"] = mdp.gamma_macro
    training["gamma_per_resource_frame"] = mdp.gamma_frame
    training["critic_action_representation"] = "executed_route"
    cfg = dict(schema="industrial_static_v1", environment=dict(num_uavs=3),
               observation=dict(layout="split_v1"), training=training)
    return DirectHRL.from_mdp(mdp, cfg)


def store_transitions(agent, collected, *, micro=True, macro=True):
    """For later authorized learning runs; collection itself never calls this."""
    if micro:
        for t in collected["micro"]:
            agent.store(t.state, t.action, t.route, t.reward, t.next_state, t.discount)
    if macro:
        for t in collected["macro"]:
            agent.macro_rollout.append((t.state, t.action, t.logprob, t.reward, t.next_state, t.trace_continue))
