"""Independent PPO baselines with fixed routing and typed checkpoints."""
import numpy as np

from .baseline_protocol import UPPER_BASELINES, is_paper_caliber
from .baselines import Routing
from .checkpoint import load
from .lite_networks import LiteUpperActor, LiteValue
from .lite_observation import LiteObserver
from .policy import Policy
from .ppo import PPO
from .randomness import derive, initialization
from .schedule import phase

CHECKPOINT_TYPE = 'independent_upper_baseline_v1'


def upper_phases():
    """Use exactly Proposed's 200 upper-episode seed identities, in order."""
    return [p for p in map(phase, range(1, 601)) if p.layer == 'upper']


def baseline_tag(baseline):
    if baseline not in UPPER_BASELINES:
        raise ValueError('Expected NC, TOO or TFOO baseline')
    internal, route = UPPER_BASELINES[baseline]
    return dict(checkpoint_type=CHECKPOINT_TYPE, baseline_type=baseline,
                internal_name=internal, lower_override=route)


def require_checkpoint_type(saved, expected):
    """Add type checks after, never instead of, the strict identity loader."""
    if expected == 'Proposed':
        if (saved.get('checkpoint_type', 'proposed') != 'proposed'
                or any(k in saved for k in ('baseline_type', 'lower_override'))
                or not {'sac', 'ppo'}.issubset(saved.get('learner', {}))):
            raise ValueError('Checkpoint type mismatch: expected Proposed')
    else:
        tag = baseline_tag(expected)
        if any(saved.get(k) != v for k, v in tag.items()) or 'sac' in saved.get('learner', {}):
            raise ValueError('Checkpoint type mismatch: expected '+expected)
    return saved


def load_typed(path, binding, expected):
    return require_checkpoint_type(load(path, binding), expected)


class UpperLearner:
    """Only the same upper actor/value and PPO as Proposed; no SAC exists."""
    def __init__(self, seed, device, config):
        if config.get('encoder') != 'lite' or not is_paper_caliber(config['objective']):
            raise ValueError('Independent upper baselines require paper-caliber Lite config')
        networks = []
        for purpose, cls in (('upper_init', LiteUpperActor), ('value', LiteValue)):
            with initialization(derive(seed, purpose=purpose)):
                network = cls()
            networks.append(network.to(device))
        self.ppo = PPO(*networks, gamma=config['ppo']['gamma'], gae_lambda=config['ppo']['gae_lambda'])
        self.gamma_lower = config['objective']['gamma_micro']

    def state_dict(self):
        ppo = self.ppo
        state = {k: getattr(ppo, k).state_dict() for k in
                 ('actor', 'value', 'actor_optimizer', 'value_optimizer')}
        state.update(counts=dict(ppo.counts), version=ppo.version)
        return dict(ppo=state, discounts=dict(lower=self.gamma_lower, ppo=ppo.gamma, gae_lambda=ppo.gae_lambda))

    def load_state_dict(self, state):
        expected = dict(lower=self.gamma_lower, ppo=self.ppo.gamma, gae_lambda=self.ppo.gae_lambda)
        if set(state) != {'ppo', 'discounts'} or state['discounts'] != expected:
            raise ValueError('Upper checkpoint learner/discount contract mismatch')
        for key, value in state['ppo'].items():
            if key in ('counts', 'version'): setattr(self.ppo, key, value)
            else: getattr(self.ppo, key).load_state_dict(value)


def fixed_upper_policy(ppo, scene, streams, route, *, stochastic, mode='moving', with_value=False):
    """The Lite observer comes from the upper encoder, without a SAC actor."""
    policy = Policy(None, ppo.actor, ppo.value if with_value else None, streams,
                    mode=mode, stochastic=stochastic, deployment=scene['deployment'])
    policy.observer = LiteObserver()
    if route == 'equal':
        policy.lower_override = lambda report, history: (
            np.full((2, 3, 3), 1/3, dtype=np.float64), np.full(6, 6))
    else:
        if route not in ('local', 'cvx'): raise ValueError('Unknown fixed lower override')
        policy.lower_override = Routing(route, scene, streams.identity['master'])
    return policy
