"""Shared, torch-free helpers for the paper-2 pipeline.

Physics, rewards, observations and the RL algorithms come unchanged from
``research/speedup/src`` (v2.4.0).  This package only organises training
stages, evaluation, sweeps and plots.  Nothing here edits the v2.4.0 source.
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
METHODS = ('Proposed', 'TFOO', 'TOO', 'NC', 'HoverSAC')
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
ALPHA_FINETUNE = 0.0005            # floor; never 0 (schedule.py:29 used 0)
ACTOR_FROZEN_EPISODES = dict(M1=2, M3=3, HoverSAC=3)
UPPER_CANDIDATES = (100, 200, 300, 400)    # PPO checkpoints considered on dev
LOWER_CANDIDATES = (0, 25, 50, 75, 100)    # 0 = M1 SAC without fine-tuning
SNAPSHOT_EVERY = 10                         # resumable state interval (episodes)

DEV_SCENARIOS = tuple(range(1, 9))          # dev split, 8 scenarios
TEST_SCENARIOS = tuple(range(21, 41))       # test split, 20 fresh scenarios (1-20 were used before)
SWEEP_SCENARIOS = tuple(range(21, 29))      # first 8 test scenarios
SWEEPS = dict(cpu_ghz=(8.0, 9.0, 10.0, 11.0, 12.0), load=(0.8, 0.9, 1.0, 1.1, 1.2))
NOMINAL = dict(cpu_ghz=10.0, load=1.0)
SELECTION_METRIC = 'cost_per_s'             # the objective both layers optimise
TRAJECTORY_SCENARIO = 21


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


def load_scene():
    return read_json(SCENE_PATH)


def scaled_scene(scene, cpu_ghz=None, load=None):
    """Copy of the scene with every UAV CPU at cpu_ghz and all arrival rates x load."""
    s = copy.deepcopy(scene)
    base = s['base_scene']
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
                    sweeps={k: tuple(v) for k, v in SWEEPS.items()}, seeds=SEEDS)
    return dict(budget=dict(M1=4, M2=10, M3=3, HoverSAC=3, PPO=10), upper_candidates=(5, 10),
                lower_candidates=(0, 2, 3), dev=(1, 2), test=(21, 22), sweep_scenarios=(21,),
                sweeps=dict(cpu_ghz=(8.0, 10.0), load=(1.0, 1.2)), seeds=(4101,))
