"""Shared, torch-free helpers for the paper-2 pipeline.

Physics, rewards, observations and the RL algorithms come from
``research/speedup/src``: v2.4.0 plus the v2.5.0 changes (number of terminals
per region as a scene parameter, bitwise identical for 13 per region).  This
package organises training stages, evaluation, sweeps and plots.
"""
import copy
import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np

PAPER2 = Path(__file__).resolve().parent.parent          # research/paper2
REPO = PAPER2.parent.parent                              # repository root
SRC = REPO / 'research' / 'speedup' / 'src'               # v2.4.0 source tree
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CONFIG_PATH = SRC / 'hrl_v2' / 'configs' / 'v2_4_0_lite.json'
SCENE_PATH = SRC / 'hrl_v2' / 'configs' / 'physical_scene.json'

# ---------------------------------------------------------------------------
# Experiment design (fixed before any training; see README.md)
# ---------------------------------------------------------------------------
SEEDS = (4101, 4102, 4103)
METHODS = ('Proposed', 'TFOO', 'TOO', 'NC', 'HoverSAC', 'CA')
# upper: 'ppo' (own independently trained PPO) or 'hover'; lower: 'sac' or a fixed rule
METHOD_SPEC = {
    'Proposed': dict(upper='ppo', lower='sac'),
    'TFOO': dict(upper='ppo', lower='cvx'),      # reference [11]: PPO trajectory + convex allocation
    'TOO': dict(upper='ppo', lower='equal'),     # PPO trajectory + equal three-way split
    'NC': dict(upper='ppo', lower='local'),      # PPO trajectory, no cooperation (all local)
    'HoverSAC': dict(upper='hover', lower='sac'),  # no trajectory optimisation (ablation)
    'CA': dict(upper='hover', lower='equal'),    # optional untrained reference, not in the main table
}
PPO_BASELINES = {'TFOO': 'cvx', 'TOO': 'equal', 'NC': 'local'}

# Training budgets (episodes of 900 s).  Stage codes keep every stage's
# training scenarios distinct; all PPO uppers (Proposed M2 and the three
# baselines) share stage 2 identities, i.e. identical training scenarios and
# identical upper exploration noise streams.
STAGE = dict(M1=1, M2=2, M3=3, HoverSAC=4, PPO=2)
BUDGET = dict(M1=200, M2=400, M3=100, HoverSAC=100, PPO=400)
UPPER_INITIAL_STD = 0.5            # was 0.15 in v2.4.0 (lite_networks.py:101)
ALPHA_M1 = ((100, 0.005), (200, 0.0025))   # (last episode, alpha)
ACTOR_FROZEN_EPISODES = dict(M1=2)  # M3 / HoverSAC: see FINETUNE_RECIPES

# Fine-tuning recipe of the two SAC fine-tuning stages (M3 and HoverSAC; M1,
# M2 and the PPO baselines do not depend on it).  The recipe is a property of
# an output root (``<out>/recipe.json``); a root without that file is v1.
#   v1  original run (pipeline d458a73): alpha dropped 0.0025 -> 0.0005 at the
#       start of fine-tuning, pre-training actor learning rate, 3 critic-only
#       episodes.  Its training logs show entropy collapse in every seed and
#       two divergences in M3 for seed 4102.
#   v2  conservative fine-tuning, fixed from those training diagnostics before
#       any v2 run and judged on the dev split only: keep the final M1 alpha,
#       smaller actor steps, longer critic-only warm-up after the replay flush.
FINETUNE_RECIPES = {
    'v1': dict(alpha=0.0005, actor_lr=1e-4, actor_frozen_episodes=3),
    'v2': dict(alpha=0.0025, actor_lr=3e-5, actor_frozen_episodes=10),
}
ALPHA_FINETUNE = FINETUNE_RECIPES['v1']['alpha']   # kept for old imports; use finetune(root)
UPPER_CANDIDATES = (100, 200, 300, 400)    # PPO checkpoints considered on dev
LOWER_CANDIDATES = (0, 25, 50, 75, 100)    # 0 = M1 SAC without fine-tuning
SNAPSHOT_EVERY = 10                         # resumable state interval (episodes)

DEV_SCENARIOS = tuple(range(1, 9))          # dev split, 8 scenarios
TEST_SCENARIOS = tuple(range(21, 41))       # test split, 20 fresh scenarios (1-20 were used before)
SWEEP_SCENARIOS = tuple(range(21, 29))      # first 8 test scenarios
# Sweeps cover stable operating points only: mean computation demand below
# 95 % of the total UAV CPU capacity (rho < 0.95).  The analytic mean demand is
# 24.725 Gcycles/s at nominal load (identical in every scenario), so 8 GHz
# (rho = 1.03, demand exceeds capacity; every queue diverges) and 1.2x load
# (rho = 0.99) are saturated.  Those two points were run under the first
# design (8-12 GHz, 0.8-1.2x); they are kept and reported separately as the
# saturated regime, never dropped silently.
SWEEPS = dict(cpu_ghz=(9.0, 10.0, 11.0, 12.0, 13.0, 14.0), load=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1),
              u2u_mhz=(1.5, 2.0, 3.0, 4.5, 6.0))
SATURATED = dict(cpu_ghz=(8.0,), load=(1.2,), u2u_mhz=())
MEAN_DEMAND_GCYCLES = 24.725   # hrl_uav workload_summary, nominal load, 3 UAVs share it
NOMINAL = dict(cpu_ghz=10.0, load=1.0, u2u_mhz=3.0, users=13)

# Retrained sweep, design v3 (research/paper2/PROTOCOL.md, fixed before any v3
# run).  Every learned method is trained from scratch at every value with the
# main pipeline; the nominal value of each parameter is the main run.  users =
# terminals per region (3 regions); the expected computation load of every
# business profile is kept at its nominal value (see users_scene), so all users
# points share demand/capacity 0.82 and only the number of terminals changes.
RETRAIN_SWEEPS = dict(cpu_ghz=(9.0, 10.0, 11.0, 12.0, 14.0), load=(0.6, 0.8, 0.9, 1.0, 1.1),
                      users=(7, 10, 13, 16, 19))
RETRAIN_ORDER = ('cpu_ghz', 'load', 'users')   # run order on one computer
PILOT_POINT = ('cpu_ghz', 9.0)                 # hardest point (demand/capacity 0.92), run first
SELECTION_METRIC = 'cost_per_s'             # the objective both layers optimise
TRAJECTORY_SCENARIO = 21


def recipe_name(root):
    path = Path(root)/'recipe.json'
    name = read_json(path)['finetune'] if path.exists() else 'v1'
    if name not in FINETUNE_RECIPES:
        raise ValueError(f'Unknown fine-tuning recipe {name!r} in {path}')
    return name


def finetune(root, smoke=False, episodes=None):
    """Fine-tuning parameters of this output root (smoke runs keep >= 1 actor episode)."""
    r = dict(FINETUNE_RECIPES[recipe_name(root)], recipe=recipe_name(root))
    if smoke and episodes is not None:
        r['actor_frozen_episodes'] = min(r['actor_frozen_episodes'], max(int(episodes)-1, 0))
    return r


def utilization(param, value):
    """Mean computation demand / total UAV CPU capacity at a sweep point
    (users points keep the nominal demand by construction)."""
    cpu = float(value) if param == 'cpu_ghz' else NOMINAL['cpu_ghz']
    load = float(value) if param == 'load' else NOMINAL['load']
    return MEAN_DEMAND_GCYCLES*load/(3*cpu)


def out_root(path=None):
    root = Path(path or os.environ.get('P2_OUT', PAPER2 / 'outputs')).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# Seeds: identical to hrl_v2.randomness.derive (copied so that this module
# stays torch-free; tests/test_identity.py checks equality where torch exists)
# ---------------------------------------------------------------------------
PURPOSES = dict(layout=1, load=2, arrival=3, upper=4, support=5, ratio=6,
                replay=7, actor=8, q1=9, q2=10, upper_init=11, value=12, random_upper=13)


def derive(master, stage=0, cycle=0, episode=0, purpose='arrival', stream=0, split='train'):
    external = purpose in ('layout', 'load', 'arrival') and split != 'train'
    fields = [2026092602, dict(train=10, dev=20, test=30)[split], 0 if external else master,
              0 if split != 'train' else stage, 0 if split != 'train' else cycle,
              episode, PURPOSES[purpose], stream]
    return int(np.random.SeedSequence(fields).generate_state(1, dtype=np.uint64)[0]) % (2**63-1)


def training_seeds(seed, stage, episode):
    return {k+'_seed': derive(seed, stage, 0, episode, purpose=k) for k in ('arrival', 'layout', 'load')}


def scenario_seeds(split, scenario):
    return {k+'_seed': derive(0, episode=scenario, purpose=k, split=split) for k in ('layout', 'load', 'arrival')}


# ---------------------------------------------------------------------------
# Config, scene, clocks
# ---------------------------------------------------------------------------
def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def load_config():
    """Same checks as hrl_v2.run.read_config, without importing torch."""
    config = read_json(CONFIG_PATH)
    if config.get('encoder') != 'lite' or config.get('observation_version') != 'lite_110_729_v2_2_1':
        raise ValueError('Expected the v2.4.0 Lite configuration')
    return config


# A retrained sweep point is an output root with <out>/scene.json =
# {"param": ..., "value": ...}: every stage of that root trains and evaluates
# on the scaled scene.  run.py sets the active root for each command.
ACTIVE_ROOT = None


def set_active_root(root):
    global ACTIVE_ROOT
    ACTIVE_ROOT = Path(root) if root is not None else None


def scene_override(root=None):
    root = root if root is not None else ACTIVE_ROOT
    if root is None:
        return None
    path = Path(root)/'scene.json'
    return read_json(path) if path.exists() else None


def load_base_scene():
    return read_json(SCENE_PATH)


def load_scene(root=None):
    """Nominal scene, or the scaled scene of a retrained sweep point root."""
    base = load_base_scene()
    override = scene_override(root)
    if not override:
        return base
    if override['param'] not in SWEEPS and override['param'] not in RETRAIN_SWEEPS:
        raise ValueError(f'unknown sweep parameter {override["param"]!r}')
    value = override['value']
    return scaled_scene(base, **{override['param']: int(value) if override['param'] == 'users' else float(value)})


def test_scenarios(root, d):
    """The main root and design-v3 retrained points are evaluated on the full
    test split; design-v2 retrained points (scene.json without "design") on the
    sweep scenarios of the earlier, non-retrained sweep."""
    override = scene_override(root)
    if override and override.get('design') != 'v3':
        return tuple(d['sweep_scenarios'])
    return tuple(d['test'])


def prune_snapshots(root):
    override = scene_override(root)
    return bool(override and override.get('prune_snapshots'))


def scaled_scene(scene, cpu_ghz=None, load=None, u2u_mhz=None, users=None):
    """Copy of the scene with every UAV CPU at cpu_ghz, all arrival rates x load,
    the total U2U bandwidth at u2u_mhz and `users` terminals per region."""
    s = copy.deepcopy(scene) if users is None or int(users) == NOMINAL['users'] else users_scene(scene, int(users))
    base = s['base_scene']
    if u2u_mhz is not None and float(u2u_mhz) != NOMINAL['u2u_mhz']:
        base['resources']['u2u_bandwidth_hz'] = float(u2u_mhz)*1e6
    if cpu_ghz is not None and float(cpu_ghz) != NOMINAL['cpu_ghz']:
        base['resources']['cpu_hz'] = [float(cpu_ghz)*1e9]*base['num_uavs']
    if load is not None and float(load) != NOMINAL['load']:
        base['region_multipliers'] = [m*float(load) for m in base['region_multipliers']]
        for profile in s['business_profiles']['catalog']:
            profile['business_multiplier'] *= float(load)
    return s


def clocks(scene, smoke=False):
    from hrl_v2.timing import ClockConfig
    return ClockConfig.from_scene(scene, rollout_seconds=50. if smoke else 900.)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, default=_plain)+'\n', encoding='utf-8')
    os.replace(tmp, path)


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False, default=_plain)+'\n')


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def clean(x):
    """Recursively convert numpy scalars/arrays to Python and non-finite floats to None."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (np.integer, bool, int)) and not isinstance(x, (np.bool_,)):
        return int(x) if not isinstance(x, bool) else x
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if np.isfinite(x) else None
    return x


def _plain(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x).__name__)


def seed_dir(root, seed):
    return Path(root)/f'seed_{seed}'


def run_dir(root, seed, name):
    return seed_dir(root, seed)/'train'/name


def alpha_m1(episode):
    for last, alpha in ALPHA_M1:
        if episode <= last:
            return alpha
    return ALPHA_M1[-1][1]


def summarize(metrics):
    """Flat per-run record used by selection, tables and plots."""
    m = metrics
    seconds = m['seconds']
    energy = {k: v/seconds for k, v in m['energy_j'].items()}
    pr = m.get('priority', [{}, {}])
    row = dict(truncated_delay_s=m.get('truncated_delay_s'), cost_per_s=m.get('cost_per_s'),
               weighted_completed_delay_s=m.get('weighted_completed_delay_s'),
               unfinished_rate=m.get('unfinished_rate'), unfinished_parents=m.get('unfinished_parents'),
               energy_w_total=sum(energy.values()), energy_w=energy,
               energy_w_compute_comm=sum(v for k, v in energy.items() if k != 'propulsion'),
               stage_delay_s=m.get('stage_delay_s'), seconds=seconds,
               mean_terminal_to_uav_m=m.get('terminal_mobility', {}).get('final_mean_terminal_to_serving_uav_m'),
               event_sha256=m.get('event_sha256'))
    for key in ('mean_delay_ms', 'completed_p95_s', 'unfinished', 'truncated_delay_s', 'arrived',
                'completed', 'completion_rate'):
        for i, name in enumerate(('high', 'low')):
            if i < len(pr) and key in pr[i]:
                row[f'{name}_{key}'] = pr[i][key]
    return clean(row)


def design(smoke=False):
    """All experiment sizes in one place; smoke=True is a minutes-long plumbing check."""
    if not smoke:
        return dict(budget=dict(BUDGET), upper_candidates=UPPER_CANDIDATES, lower_candidates=LOWER_CANDIDATES,
                    dev=DEV_SCENARIOS, test=TEST_SCENARIOS, sweep_scenarios=SWEEP_SCENARIOS,
                    sweeps={k: tuple(v) for k, v in SWEEPS.items()}, saturated={k: tuple(v) for k, v in SATURATED.items()},
                    retrain_sweeps={k: tuple(v) for k, v in RETRAIN_SWEEPS.items()}, seeds=SEEDS)
    return dict(budget=dict(M1=4, M2=10, M3=3, HoverSAC=3, PPO=10), upper_candidates=(5, 10),
                lower_candidates=(0, 2, 3), dev=(1, 2), test=(21, 22), sweep_scenarios=(21,),
                sweeps=dict(cpu_ghz=(10.0, 12.0), load=(0.8, 1.0), u2u_mhz=(2.0, 3.0)),
                saturated=dict(cpu_ghz=(), load=(), u2u_mhz=()),
                retrain_sweeps=dict(cpu_ghz=(9.0, 10.0), load=(0.8, 1.0), users=(7, 13)),
                seeds=(4101,))


def point_value(param, value):
    """Canonical value of a sweep point (users = whole terminals per region)."""
    if param == 'users':
        if float(value) != int(float(value)):
            raise ValueError(f'users must be a whole number of terminals per region, got {value}')
        return int(float(value))
    return float(value)


# ---------------------------------------------------------------------------
# Number of terminals per region (design v3 users sweep)
# ---------------------------------------------------------------------------
TYPES = ('sensor', 'controller', 'vision')


def largest_remainder(weights, total):
    """Integers proportional to `weights` summing to `total`; ties go to the
    lower index (deterministic Hamilton apportionment)."""
    w = np.asarray(weights, dtype=float)
    exact = w*total/w.sum()
    out = np.floor(exact).astype(int)
    order = sorted(range(len(w)), key=lambda i: (-(exact[i]-out[i]), i))
    for i in order[:int(total)-int(out.sum())]:
        out[i] += 1
    return [int(x) for x in out]


def cluster_quotas(template, counts, n_cluster):
    """Cluster members by type: the template quotas scaled to n_cluster, capped by
    the available terminals of each type (excess to the type with most spare)."""
    q = largest_remainder(template, n_cluster)
    while any(a > c for a, c in zip(q, counts)):
        excess = sum(max(a-c, 0) for a, c in zip(q, counts))
        q = [min(a, c) for a, c in zip(q, counts)]
        for _ in range(excess):
            spare = [(c-a, -i) for i, (a, c) in enumerate(zip(q, counts))]
            best = max(range(3), key=lambda i: spare[i])
            if spare[best][0] <= 0:
                raise ValueError('not enough terminals for the cluster quotas')
            q[best] += 1
    return q


def expected_profile_cycles(task_types, counts):
    """Expected cycles/s of one business profile before the business multiplier
    and the (profile-independent) activity factor."""
    total = 0.
    for kind, count in zip(TYPES, counts):
        p = task_types[kind]
        prob = np.array([p['high_probability'], 1-p['high_probability']])
        cycles = np.array(p['input_bits'], float)*np.array(p['cycles_per_bit'], float)
        total += count*p['rate_per_s']*float(prob @ cycles)
    return total


def users_scene(scene, per_region):
    """Scene with `per_region` terminals in each of the three regions.

    Per business profile: type counts = nominal counts scaled to per_region
    (largest remainder); cluster members = round(9 per_region / 13), with type
    quotas scaled from the nominal 3/3/3; every slot keeps its type's task
    parameters; and the business multiplier is rescaled so that the profile's
    expected computation load equals its nominal value exactly (the sum over the
    three regions, 24.725 Gcycles/s, is unchanged for every profile permutation).
    Terminal sampling (cluster disk, background rejection), mobility, uplink
    bandwidth per terminal (B_G / number of terminals), seeds and RNG binding
    follow the v2.5.0 simulator.  per_region == 13 returns an unchanged copy.
    """
    k = int(per_region)
    nominal = NOMINAL['users']
    if k == nominal:
        return copy.deepcopy(scene)
    if k < 3:
        raise ValueError('at least three terminals per region')
    s = copy.deepcopy(scene)
    base, geometry, business = s['base_scene'], s['geometry'], s['business_profiles']
    if base['num_users'] != 3*nominal or geometry['cluster_count_per_region'] != 9:
        raise ValueError('users_scene expects the nominal v2.4.0 scene as input')
    n_cluster = int(np.floor(9*k/nominal+0.5))
    task_types = base['task_types']
    counts_all, multipliers = [], []
    for profile in business['catalog']:
        pid = profile['profile_id']
        old_counts = profile['device_counts_by_type']
        counts = largest_remainder(old_counts, k)
        quotas = cluster_quotas(profile['cluster_counts_by_type'], counts, n_cluster)
        scale = expected_profile_cycles(task_types, old_counts)/expected_profile_cycles(task_types, counts)
        multiplier = base['region_multipliers'][pid]*scale
        profile['device_counts_by_type'] = counts
        profile['cluster_counts_by_type'] = quotas
        profile['business_multiplier'] = multiplier
        profile['slots'] = [dict(task_type=kind, parameters=copy.deepcopy(task_types[kind]))
                            for kind, count in zip(TYPES, counts) for _ in range(count)]
        counts_all.append(counts)
        multipliers.append(multiplier)
    base['num_users'] = 3*k
    base['device_counts'] = counts_all
    base['region_multipliers'] = multipliers
    geometry['cluster_count_per_region'] = n_cluster
    geometry['background_count_per_region'] = k-n_cluster
    business['physical_source_rule'] = f'user_id // {k}'
    business['global_device_counts_by_type'] = [int(x) for x in np.asarray(counts_all).sum(axis=0)]
    s['status'] = f'{scene.get("status", "")}; v2.5.0 users sweep: {k} terminals per region, nominal load'
    return s
