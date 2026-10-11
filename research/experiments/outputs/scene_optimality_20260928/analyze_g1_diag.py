"""G1 diagnostics on the frozen ep300 checkpoint (read-only: no training, no gate/run writes).

verify  stochastic scenario 1 stream 1 must reproduce the G1 panel file bit-for-bit (event sha)
D1      deterministic lower decode, 20 gate scenarios x stream 0, hover
D2a     high priority = SAC stochastic sample, low priority = equal 1/3 (support 6), streams [1,2]
D2b     high priority = equal, low priority = SAC stochastic sample, streams [1,2]
The SAC sample in D2 consumes exactly the same random draws as the G1 panel at each step.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
import torch
from hrl_v2.checkpoint import identity, load, atomic_json
from hrl_v2.evaluate import gate_manifest
from hrl_v2.run import read_config
from hrl_v2.runtime import configure, Learner
from hrl_v2.policy import Policy
from hrl_v2.randomness import Streams
from hrl_v2.physics_adapter import PhysicsAdapter
from hrl_v2.collector import collect_metrics
from hrl_v2.metrics import PooledMetrics
from hrl_v2.timing import ClockConfig
from hrl_v2.paths import source_root
from hrl_v2.gates import paired, g1

OUT = Path(__file__).resolve().parent/'g1_diag'
RUN = ROOT/'outputs/lite_v2_3_0/training_4101_20260929_002123/runs/Proposed/seed_4101'
CKPT = RUN/'checkpoints/episode-0300-evaluated.pt'
CONFIG = ROOT/'src/hrl_v2/configs/v2_3_0_lite.json'
EQUAL = (np.full((2,3,3),1/3,dtype=np.float64), np.full(6,6))


class MixedPolicy(Policy):
    """Sample the SAC action exactly as Policy.lower does, then replace one priority with equal."""
    def __init__(self, *a, sac_priority, **k):
        super().__init__(*a, **k)
        self.sac_priority = sac_priority

    def lower(self, report, history):
        route, ids = super().lower(report, history)
        route, ids = route.copy(), np.asarray(ids).copy()
        other = 1-self.sac_priority
        route[other] = EQUAL[0][other]
        ids[3*other:3*other+3] = 6
        return route, ids


def main(parts):
    configure()
    config = read_config(CONFIG, ROOT/'outputs')
    binding = identity(ROOT/'outputs', CONFIG)
    saved = load(CKPT, binding)
    learner = Learner(saved['seed'], configure(), config=config); learner.load_state_dict(saved['learner'])
    master = saved['seed']
    reference = json.loads((RUN/'baseline_reference.json').read_text())
    manifest = reference['manifest']
    if manifest != gate_manifest(20): raise ValueError('reference manifest differs from gate_manifest(20)')
    scene = json.loads((source_root()/'hrl_v2/configs/physical_scene.json').read_text(encoding='utf-8'))
    timing = ClockConfig.from_config(config)
    scale = config['objective']['reward_scale']
    plan = dict(verify=(True, None, [1], [1]), D1=(False, None, None, [0]),
                D2a=(True, 0, None, [1,2]), D2b=(True, 1, None, [1,2]))
    summary = {}
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        for part in parts:
            stochastic, sac_priority, only, streams_list = plan[part]
            pooled, per = PooledMetrics(), {}
            for scenario in manifest['scenarios']:
                sid = scenario['scenario']
                if only and sid not in only: continue
                for stream in streams_list:
                    path = OUT/part/f'{sid:02d}_{stream}.json'
                    if path.exists(): metrics = json.loads(path.read_text())['metrics']
                    else:
                        tick = time.perf_counter()
                        streams = Streams(master, episode=sid, split='dev', policy_stream=stream)
                        adapter = PhysicsAdapter(scene, timing=timing)
                        adapter.reset(**{k:v for k,v in scenario.items() if k.endswith('_seed')})
                        args = (learner.sac.actor, learner.ppo.actor, None, streams)
                        kw = dict(mode='hover', stochastic=stochastic, deployment=scene['deployment'])
                        policy = Policy(*args, **kw) if sac_priority is None else MixedPolicy(*args, sac_priority=sac_priority, **kw)
                        item = dict(part=part, checkpoint=str(CKPT), scenario=sid, stream=stream, master=master)
                        metrics = collect_metrics(adapter, policy, identity=item, reward_scale=scale)
                        atomic_json(path, dict(identity=item, metrics=metrics, wall_seconds=time.perf_counter()-tick))
                        print(f'{part} scenario={sid} stream={stream} delay={metrics["truncated_delay_s"]:.6f} '
                              f'cost/s={metrics["cost_per_s"]:.4f} wall={time.perf_counter()-tick:.1f}s', flush=True)
                    pooled.add(metrics); per.setdefault(str(sid), PooledMetrics()).add(metrics)
            per = {k:v.result() for k,v in per.items()}
            if part == 'verify':
                g = json.loads((RUN/'evaluations/episode-0300/stochastic_hover_01_1.json').read_text())['metrics']
                m = json.loads((OUT/'verify/01_1.json').read_text())['metrics']
                ok = all(m[k] == g[k] for k in ('event_sha256', 'canonical_event_sha256', 'truncated_delay_s', 'cost_sum'))
                summary[part] = dict(reproduces_g1_panel=bool(ok), delay=m['truncated_delay_s'], g1_delay=g['truncated_delay_s'])
                print('verify reproduces G1 panel:', ok, flush=True)
                if not ok: raise RuntimeError('Setup does not reproduce the G1 stochastic panel')
                continue
            gate = g1(per, reference['per_scenario'], reference['required_wins'])
            summary[part] = dict(pooled=pooled.result(), per_scenario=per, g1_style=gate,
                                 cost_per_s_vs_equal=paired({k:dict(truncated_delay_s=v['cost_per_s'],unfinished_parents=0) for k,v in per.items()},
                                     {k:dict(truncated_delay_s=v['cost_per_s'],unfinished_parents=0) for k,v in reference['per_scenario']['hover_equal'].items()}))
            e = gate['equal']
            print(f'{part}: mean {e["candidate_mean_s"]:.6f} vs equal {e["reference_mean_s"]:.6f} wins {e["wins"]}/20 '
                  f'unfinished {e["candidate_unfinished"]} vs {e["reference_unfinished"]} passed={gate["passed"]}', flush=True)
            atomic_json(OUT/f'summary_{part}.json', summary[part])
    return summary


if __name__ == '__main__':
    main(sys.argv[1:] or ['verify', 'D1', 'D2a', 'D2b'])
