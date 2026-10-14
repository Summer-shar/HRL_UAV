"""Training stages for the paper-2 pipeline (requires PyTorch).

Algorithms are the unchanged v2.4.0 implementations (hrl_v2.sac.SAC,
hrl_v2.ppo.PPO, hybrid simplex lower actor, Lite networks).  What changes is
only how the stages are organised, following the audit of the old schedule:

* M1   SAC pre-training on generic random deployments (not only hover).
* M2   PPO upper training against the FROZEN, DETERMINISTIC M1 SAC, i.e. the
       same lower behaviour that is used at evaluation time.
* M3   SAC fine-tuning on the selected M2 upper (deterministic); alpha, actor
       step size and critic-only warm-up follow the root's fine-tuning recipe
       (common.FINETUNE_RECIPES).
* TFOO / TOO / NC  independent PPO uppers with their fixed lower rule, same
       PPO settings, budget, training scenarios and exploration noise as M2.
* HoverSAC  M1 SAC fine-tuned with all UAVs hovering (ablation).

Every PPO upper starts from the same seed-derived weights with initial std 0.5.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import gc
import math
import time
from pathlib import Path

import numpy as np
import torch

from . import common as C
from . import perf
from hrl_v2.collector import collect
from hrl_v2.policy import Policy
from hrl_v2.randomness import Streams
from hrl_v2.replay import Replay
from hrl_v2.runtime import Learner, parameter_hash
from hrl_v2.upper_baselines import UpperLearner, fixed_upper_policy


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
def device_setup():
    perf.apply()
    torch.set_num_threads(int(os.environ.get('P2_THREADS', '2')))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    name = os.environ.get('P2_DEVICE') or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def set_initial_std(actor, std=C.UPPER_INITIAL_STD):
    """Initial upper exploration std; the std head weight is zero at init."""
    if float(actor.std.weight.detach().abs().max()) != 0.:
        raise ValueError('Initial std can only be set on a freshly initialised upper actor')
    with torch.no_grad():
        actor.std.bias.fill_(math.atanh((math.log(std)+2.5)/2.5))


def _fast_microbatches(learner):
    """Same losses (sums over the same minibatch), far fewer tiny kernel calls.
    v2.4.0 used 16-row microbatches inside 128/150-row minibatches; only the
    float summation order and the grouping of random draws change."""
    learner.ppo.microbatch_size = 150
    if hasattr(learner, 'sac'):
        learner.sac.microbatch_size = 128
        learner.sac.critic_microbatch_size = 128


def proposed_learner(seed, device, config):
    learner = Learner(seed, device, config=config)
    set_initial_std(learner.ppo.actor)
    _fast_microbatches(learner)
    return learner


def baseline_learner(seed, device, config):
    learner = UpperLearner(seed, device, config)
    set_initial_std(learner.ppo.actor)
    _fast_microbatches(learner)
    return learner


def save_state(path, **state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_state(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def load_ppo_part(learner, state):
    """Copy the PPO actor/value (and optimiser) from a saved learner state."""
    for name in ('actor', 'value', 'actor_optimizer', 'value_optimizer'):
        getattr(learner.ppo, name).load_state_dict(state['ppo'][name])
    learner.ppo.counts = dict(state['ppo']['counts'])
    learner.ppo.version = state['ppo']['version']


def budgets(smoke):
    d = C.design(smoke)
    return d['budget'], tuple(d['upper_candidates']), tuple(d['lower_candidates'])


# ---------------------------------------------------------------------------
# Generic exploration upper for M1 (no task knowledge)
# ---------------------------------------------------------------------------
class ExploreUpperPolicy(Policy):
    """Two of three episodes: uniform random scores every macro (random walk).
    Every third episode: full-speed flights toward random waypoints drawn
    uniformly over the map, so that the lower also sees UAV positions (and
    UAV-user distances) far from the starting layout."""

    def __init__(self, *args, random_seed, waypoint, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = np.random.default_rng(random_seed)
        self.waypoint = bool(waypoint)
        self.targets = None

    @torch.no_grad()
    def upper(self, report, history):
        if self.observer is not None:
            self.observer.update(report, history)
        if not self.waypoint:
            return self.rng.uniform(-1., 1., 6), {}
        positions = np.asarray([u.position for u in report.uavs], dtype=float)
        if self.targets is None:
            self.targets = self.rng.uniform(0., report.map_size, (len(positions), 2))
        out = []
        for i, position in enumerate(positions):
            delta = self.targets[i]-position
            if np.linalg.norm(delta) < 20.:
                self.targets[i] = self.rng.uniform(0., report.map_size, 2)
                delta = self.targets[i]-position
            out.extend(delta/max(20., float(np.linalg.norm(delta))))
        return np.asarray(out), {}


# ---------------------------------------------------------------------------
# SAC stages: M1, M3, HoverSAC
# ---------------------------------------------------------------------------
def _sac_policy(name, learner, streams, seed, episode):
    if name == 'M1':
        return ExploreUpperPolicy(learner.sac.actor, learner.ppo.actor, None, streams, mode='moving',
                                  stochastic=True, random_seed=C.derive(seed, C.STAGE['M1'], 0, episode, purpose='random_upper'),
                                  waypoint=episode % 3 == 0)
    if name == 'M3':
        return Policy(learner.sac.actor, learner.ppo.actor, None, streams, mode='moving',
                      stochastic=True, upper_stochastic=False)
    if name == 'HoverSAC':
        return Policy(learner.sac.actor, None, None, streams, mode='hover', stochastic=True)
    raise ValueError(name)


def train_sac_stage(name, seed, root, *, smoke=False, init_paths=None):
    """Train one SAC stage. init_paths: dict(m1=..., upper=...) for M3/HoverSAC."""
    budget, _, lower_candidates = budgets(smoke)
    episodes = budget[name]
    rd = C.run_dir(root, seed, name)
    (rd/'ckpt').mkdir(parents=True, exist_ok=True)
    if (rd/'done.json').exists() and (rd/'final.pt').exists():
        print(f'[{name} seed {seed}] already finished', flush=True)
        return
    config, scene = C.load_config(), C.load_scene(root)
    clk = C.clocks(scene, smoke)
    device = device_setup()
    learner = proposed_learner(seed, device, config)
    recipe = C.finetune(root, smoke, episodes) if name in ('M3', 'HoverSAC') else None
    contract_path = rd/'contract.json'
    if recipe is not None and contract_path.exists():
        started_with = C.read_json(contract_path).get('finetune', dict(recipe='v1'))
        if started_with.get('recipe') != recipe['recipe']:
            raise RuntimeError(f'{rd} was started with fine-tuning recipe {started_with.get("recipe")}, '
                               f'this output root uses {recipe["recipe"]}; use a separate --out')
    replay, completed = Replay(), 0
    state_path = rd/'state.pt'
    if state_path.exists():
        st = load_state(state_path)
        learner.load_state_dict(st['learner'])
        replay, completed = st['replay'], st['completed']
        torch.set_rng_state(st['cpu_rng'])
        print(f'[{name} seed {seed}] resumed after episode {completed}', flush=True)
    else:
        if name in ('M3', 'HoverSAC'):
            learner.load_state_dict(load_state(init_paths['m1'])['learner'])
            if name == 'M3':
                load_ppo_part(learner, load_state(init_paths['upper'])['learner'])
        C.write_json(rd/'contract.json', dict(stage=name, seed=seed, episodes=episodes, smoke=smoke,
            init_paths={k: str(v) for k, v in (init_paths or {}).items()},
            alpha=('M1 schedule '+str(C.ALPHA_M1)) if name == 'M1' else recipe['alpha'],
            actor_frozen_episodes=C.ACTOR_FROZEN_EPISODES['M1'] if name == 'M1' else recipe['actor_frozen_episodes'],
            finetune=recipe, upper_initial_std=C.UPPER_INITIAL_STD,
            training_scenarios=f'derive(seed, stage={C.STAGE[name]}, 0, episode)',
            parameter_hashes=_hashes(learner), device=str(device), torch=torch.__version__))
    if recipe is not None:
        # The learner state loaded from M1 (or a snapshot) carries its Adam
        # param_groups; the recipe's actor step size overrides the stored lr.
        for group in learner.sac.actor_optimizer.param_groups:
            group['lr'] = recipe['actor_lr']
    frozen_upper = parameter_hash(learner.ppo.actor) if name == 'M3' else None
    version = dict(M1='explore-v1', M3='m2-selected-deterministic', HoverSAC='hover-0')[name]
    if replay.upper_version != version:
        replay.set_upper_version(version)
    stage = C.STAGE[name]
    for n in range(completed+1, episodes+1):
        started = time.perf_counter()
        streams = Streams(seed, stage, 0, n)
        adapter = perf.TrainingAdapter(scene, timing=clk)
        adapter.reset(**C.training_seeds(seed, stage, n))
        policy = _sac_policy(name, learner, streams, seed, n)
        alpha = C.alpha_m1(n) if name == 'M1' else recipe['alpha']
        actor_enabled = n > (C.ACTOR_FROZEN_EPISODES['M1'] if name == 'M1' else recipe['actor_frozen_episodes'])
        residuals = []

        def event(streams=streams, alpha=alpha, actor_enabled=actor_enabled, residuals=residuals):
            row = learner.sac.event(replay, alpha, actor_enabled, streams)
            residuals.extend(v for fit in row.get('critics', []) for v in fit['residual_mse'])

        episode = collect(adapter, policy, upper_version=version, lower_version=learner.sac.version,
                          identity=dict(stage=name, seed=seed, episode=n), on_event=event,
                          reward_scale=config['objective']['reward_scale'],
                          gamma_lower=config['objective']['gamma_micro'])
        replay.add(episode)
        row = dict(stage=name, seed=seed, episode=n, alpha=alpha, actor_enabled=actor_enabled,
                   actor_lr=learner.sac.actor_optimizer.param_groups[0]['lr'],
                   reward_physical=float(sum(episode.rewards)),
                   critic_residual_mse=float(np.mean(residuals)) if residuals else None,
                   actor_relative_entropy=policy.entropy_sum/policy.entropy_count if policy.entropy_count else None,
                   sac_counts=dict(learner.sac.counts), wall_seconds=time.perf_counter()-started,
                   **C.summarize(episode.metrics))
        C.append_jsonl(rd/'train.jsonl', row)
        print(f'[{name} seed {seed} {n}/{episodes}] delay={row["truncated_delay_s"]:.4f}s '
              f'cost/s={row["cost_per_s"]:.4f} wall={row["wall_seconds"]:.1f}s', flush=True)
        if name != 'M1' and n in lower_candidates:
            save_state(rd/'ckpt'/f'ep{n:04d}.pt', learner=learner.state_dict(), stage=name, episode=n, seed=seed)
        if n % C.SNAPSHOT_EVERY == 0 or n == episodes:
            save_state(state_path, learner=learner.state_dict(), replay=replay, completed=n,
                       cpu_rng=torch.get_rng_state())
        del adapter, policy, episode
        gc.collect()
    if frozen_upper is not None and parameter_hash(learner.ppo.actor) != frozen_upper:
        raise RuntimeError('M3 must not change the upper actor')
    save_state(rd/'final.pt', learner=learner.state_dict(), stage=name, episode=episodes, seed=seed)
    C.write_json(rd/'done.json', dict(stage=name, seed=seed, episodes=episodes, sac_counts=learner.sac.counts,
                                      parameter_hashes=_hashes(learner)))
    if C.prune_snapshots(root):
        state_path.unlink(missing_ok=True)      # resume snapshot (with replay) no longer needed


# ---------------------------------------------------------------------------
# PPO stages: M2 (frozen deterministic M1 SAC) and the three baselines
# ---------------------------------------------------------------------------
def train_ppo_stage(name, seed, root, *, smoke=False, init_paths=None):
    budget, upper_candidates, _ = budgets(smoke)
    episodes = budget['PPO']
    rd = C.run_dir(root, seed, name)
    (rd/'ckpt').mkdir(parents=True, exist_ok=True)
    if (rd/'done.json').exists() and (rd/'final.pt').exists():
        print(f'[{name} seed {seed}] already finished', flush=True)
        return
    config, scene = C.load_config(), C.load_scene(root)
    clk = C.clocks(scene, smoke)
    device = device_setup()
    proposed = name == 'M2'
    learner = proposed_learner(seed, device, config) if proposed else baseline_learner(seed, device, config)
    rule = None if proposed else C.PPO_BASELINES[name]
    batch, completed = [], 0
    state_path = rd/'state.pt'
    if state_path.exists():
        st = load_state(state_path)
        learner.load_state_dict(st['learner'])
        completed = st['completed']
        torch.set_rng_state(st['cpu_rng'])
        print(f'[{name} seed {seed}] resumed after episode {completed}', flush=True)
    else:
        if proposed:
            m1 = load_state(init_paths['m1'])['learner']
            learner.load_state_dict(m1)   # SAC from M1; PPO weights identical to a fresh init
        C.write_json(rd/'contract.json', dict(stage=name, seed=seed, episodes=episodes, smoke=smoke,
            lower='frozen deterministic M1 SAC' if proposed else 'fixed rule: '+rule,
            init_paths={k: str(v) for k, v in (init_paths or {}).items()},
            upper_initial_std=C.UPPER_INITIAL_STD, ppo=config['ppo'],
            training_scenarios=f'derive(seed, stage={C.STAGE["PPO"]}, 0, episode) (shared by M2/TFOO/TOO/NC)',
            parameter_hashes=_hashes(learner), device=str(device), torch=torch.__version__))
    frozen_sac = parameter_hash(learner.sac.actor) if proposed else None
    stage = C.STAGE['PPO']
    for n in range(completed+1, episodes+1):
        started = time.perf_counter()
        streams = Streams(seed, stage, 0, n)
        adapter = perf.TrainingAdapter(scene, timing=clk)
        adapter.reset(**C.training_seeds(seed, stage, n))
        if proposed:
            policy = Policy(learner.sac.actor, learner.ppo.actor, learner.ppo.value, streams, mode='moving',
                            stochastic=False, upper_stochastic=True)
            lower_version = f'sac-m1-{learner.sac.version}-deterministic'
        else:
            policy = fixed_upper_policy(learner.ppo, scene, streams, rule, stochastic=True, with_value=True)
            lower_version = 'fixed-'+rule
        episode = collect(adapter, policy, upper_version=f'ppo-{learner.ppo.version}-stochastic',
                          lower_version=lower_version, identity=dict(stage=name, seed=seed, episode=n),
                          reward_scale=config['objective']['reward_scale'],
                          gamma_lower=config['objective']['gamma_micro'])
        batch.append(episode)
        logs = None
        if len(batch) == config['ppo']['batch_episodes']:
            logs = learner.ppo.update(batch, streams)
            batch.clear()
        moves = np.linalg.norm(np.asarray([h['displacement'] for h in episode.high]), axis=-1)
        row = dict(stage=name, seed=seed, episode=n, reward_physical=float(sum(episode.rewards)),
                   mean_step_m=float(moves.mean()), ppo=logs, ppo_counts=dict(learner.ppo.counts),
                   wall_seconds=time.perf_counter()-started, **C.summarize(episode.metrics))
        C.append_jsonl(rd/'train.jsonl', row)
        print(f'[{name} seed {seed} {n}/{episodes}] delay={row["truncated_delay_s"]:.4f}s '
              f'cost/s={row["cost_per_s"]:.4f} step={row["mean_step_m"]:.1f}m wall={row["wall_seconds"]:.1f}s', flush=True)
        if n in upper_candidates:
            save_state(rd/'ckpt'/f'ep{n:04d}.pt', learner=learner.state_dict(), stage=name, episode=n, seed=seed)
        if (n % C.SNAPSHOT_EVERY == 0 or n == episodes) and not batch:
            save_state(state_path, learner=learner.state_dict(), completed=n, cpu_rng=torch.get_rng_state())
        del adapter, policy, episode
        gc.collect()
    if batch:
        raise RuntimeError('PPO budget must contain complete batches')
    if frozen_sac is not None and parameter_hash(learner.sac.actor) != frozen_sac:
        raise RuntimeError('M2 must not change the SAC actor')
    save_state(rd/'final.pt', learner=learner.state_dict(), stage=name, episode=episodes, seed=seed)
    C.write_json(rd/'done.json', dict(stage=name, seed=seed, episodes=episodes, ppo_counts=learner.ppo.counts,
                                      parameter_hashes=_hashes(learner)))
    if C.prune_snapshots(root):
        state_path.unlink(missing_ok=True)


def _hashes(learner):
    out = dict(ppo_actor=parameter_hash(learner.ppo.actor), ppo_value=parameter_hash(learner.ppo.value))
    if hasattr(learner, 'sac'):
        out.update(sac_actor=parameter_hash(learner.sac.actor), q1=parameter_hash(learner.sac.q1))
    return out


# ---------------------------------------------------------------------------
# Evaluation policies (deterministic execution for every learned method)
# ---------------------------------------------------------------------------
class LoadedMethod:
    """Factory of evaluation policies for one method / checkpoint choice."""

    def __init__(self, method, seed, *, learner_state=None, upper_state=None, device=None):
        self.method, self.seed = method, seed
        self.device = device or device_setup()
        config = C.load_config()
        spec = C.METHOD_SPEC[method]
        if spec['lower'] == 'sac':
            self.learner = proposed_learner(seed, self.device, config)
            self.learner.load_state_dict(learner_state)
            if upper_state is not None:
                load_ppo_part(self.learner, upper_state)
        elif spec['upper'] == 'ppo':
            self.learner = baseline_learner(seed, self.device, config)
            self.learner.load_state_dict(learner_state)
        else:
            self.learner = None
        for net in self._networks():
            net.eval()

    def _networks(self):
        if self.learner is None:
            return []
        nets = [self.learner.ppo.actor, self.learner.ppo.value]
        if hasattr(self.learner, 'sac'):
            nets += [self.learner.sac.actor]
        return nets

    def policy(self, scene, split, scenario):
        spec = C.METHOD_SPEC[self.method]
        streams = Streams(self.seed, episode=scenario, split=split, policy_stream=1)
        if spec['lower'] == 'sac':
            upper = self.learner.ppo.actor if spec['upper'] == 'ppo' else None
            return Policy(self.learner.sac.actor, upper, None, streams,
                          mode='moving' if spec['upper'] == 'ppo' else 'hover',
                          stochastic=False, upper_stochastic=False, deployment=scene['deployment'])
        if spec['upper'] == 'ppo':
            return fixed_upper_policy(self.learner.ppo, scene, streams, spec['lower'], stochastic=False)
        from .evaluate import ScriptedPolicy
        return ScriptedPolicy(spec['upper'], spec['lower'])
