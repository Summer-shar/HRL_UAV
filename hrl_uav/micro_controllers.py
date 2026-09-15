"""Unified micro routing controller interface for per-method PPO training.

collect_stream resolves micro_controller=None to policy.sac.sac_actor, so the
archived SAC path is bit-exact. Fixed-rule and solver controllers never learn,
never touch the environment RNG, and emit the same 18-dim score layout that
QueueMDP.route projects onto the per-(priority, source) simplex.

RandomMicro duplicates mission_paper_comparison.RandomRouting with a widened
scene whitelist; the archived comparison module is not modified, so its
recorded provenance hashes stay verifiable.
"""

import numpy as np
import torch

from .bandwidth import UniformRouting
from .scene_prescreen import LocalRouting
from .xu_cvx_r import XuCVXRouting

RANDOM_SEED_ROOT = 86701  # Same SeedSequence root as the archived comparison.
DEFAULT_ALLOWED_SCENES = tuple(range(72000, 72040)) + (33600, 33601)


class SACMicro:
    """Frozen SAC101 wrapper; delegates the freeze audit to ReadOnlySAC."""

    name = "SAC"

    def __init__(self, read_only_sac):
        self.sac = read_only_sac

    def choose_action(self, state, deterministic=True):
        with torch.no_grad():
            return self.sac.sac_actor.choose_action(state, deterministic=deterministic)

    def assert_unchanged(self):
        return self.sac.assert_unchanged()


class FixedRuleMicro:
    """Local (no offloading) or Uniform (fixed equal split); stateless rules."""

    def __init__(self, name):
        if name == "Local":
            self.rule = LocalRouting()
        elif name == "Uniform":
            self.rule = UniformRouting()
        else:
            raise ValueError(f"Unknown fixed micro rule: {name}")
        self.name = name
        self.calls = 0

    def choose_action(self, state, deterministic=True):
        self.calls += 1
        return self.rule.choose_action(state, deterministic=True)

    def assert_unchanged(self):
        return {"fixed_rule": self.name, "calls": self.calls}


class RandomMicro:
    """Independent Dirichlet(1,1,1) per (priority, source) row per decision.

    Sampling is the method's semantics; the deterministic flag does not freeze
    it. The private PCG64 stream is seeded per scene, so environment CRN is
    untouched and each 900 s stream is reproducible.
    """

    name = "Random"

    def __init__(self, scene, allowed_scenes=DEFAULT_ALLOWED_SCENES):
        if scene not in allowed_scenes:
            raise ValueError("RandomMicro scene outside the declared schedule")
        self.scene = scene
        self.rng = np.random.default_rng(np.random.SeedSequence([RANDOM_SEED_ROOT, scene]))
        self.calls = 0

    def choose_action(self, state, deterministic=True):
        self.calls += 1
        return self.rng.dirichlet(np.ones(3), size=(2, 3)).ravel()

    def assert_unchanged(self):
        return {"random_dirichlet": True, "calls": self.calls}


class CVXMicro:
    """Xu-CVX-R adaptation: fixed-CPU aggregate convex routing, no SCA."""

    name = "Xu-CVX-R"

    def __init__(self, actor):
        if not isinstance(actor, XuCVXRouting):
            raise TypeError("CVXMicro requires an XuCVXRouting actor")
        self.actor = actor

    def choose_action(self, state, deterministic=True):
        return self.actor.choose_action(state, deterministic=True)

    def assert_unchanged(self):
        return {"convex_lp": True, "solver_calls": self.actor.calls}


def cvx_micro(probe_meter):
    """Build the CVX controller from a probe meter sharing the stream config."""
    from .mission_xu_cvx_r import static_model  # Lazy: avoids a heavy import chain.
    return CVXMicro(XuCVXRouting(static_model(probe_meter)))
