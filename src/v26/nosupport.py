"""Ablation "NoSupport": the lower policy without support selection.

The proposed lower policy first picks, for every row (source UAV, priority),
one of the seven non-empty destination sets (the support) and then draws the
ratios on that set from a Dirichlet distribution, so a ratio can be exactly 0.
In this ablation the support is fixed to all three UAVs: every row is a
Dirichlet sample on the whole simplex, every ratio is positive, and every task
is split over all three UAVs - the property of a softmax output that section
3.2 of the paper argues against.  Everything else is identical: networks,
initial weights (same seed), SAC, temperature schedule, training scenarios,
fine-tuning recipe, selection rule and evaluation.

Implementation: in the processes of this ablation only, LiteLowerActor.decode
replaces the six support logits other than {0,1,2} by -1e4 (probability
exactly 0 in float64, finite as the distribution code requires).  Then
* sampling and the deterministic decode always pick {0,1,2};
* the support term of the log-density is 0 (+ the constant ln 7, which only
  shifts the soft values by a constant);
* the actor surrogate weights the six other supports by exactly 0.
The v2.5.0 files are not edited.

Pipeline (budgets of the main run: M1 200 windows, M3 100 windows, candidates
0/25/50/75/100):
  M1  stage-1 SAC pre-training from scratch, random deployments (as the main run)
  M3  stage-3 fine-tuning under the main run's selected stage-2 PPO (frozen)
      - no new PPO is trained; the trajectory policy is the main run's, so the
      comparison isolates the representation of the offloading ratios
  dev candidates 0 (M1 + main PPO) and 25-100, selection by dev cost per second
  test on the 20 test scenarios (nominal parameters)
"""
from pathlib import Path

from p2 import common as C

NAME = 'nosupport'
METHOD = 'NoSupport'
FULL = 6                      # index of the support (0, 1, 2) in hrl_v2.hybrid_simplex.SUPPORTS
MASKED = -1e4

_PATCHED = False


def patch():
    """Only in this process: the lower actor can only choose the support {0, 1, 2}."""
    global _PATCHED
    if _PATCHED:
        return
    import torch
    from hrl_v2 import lite_networks
    from hrl_v2.hybrid_simplex import SUPPORTS
    if tuple(SUPPORTS[FULL]) != (0, 1, 2) or len(SUPPORTS) != 7:
        raise RuntimeError('unexpected support order')
    original = lite_networks.LiteLowerActor.decode
    keep = torch.zeros(7, dtype=torch.bool)
    keep[FULL] = True

    def decode(self, obs):
        logits, kappa = original(self, obs)
        return logits.masked_fill(~keep, MASKED), kappa

    lite_networks.LiteLowerActor.decode = decode
    _PATCHED = True
    check()


def check():
    """Runtime check with a fresh actor: only {0,1,2} has probability, every executed ratio is positive."""
    import torch
    from hrl_v2.lite_networks import LiteLowerActor
    from hrl_v2.lite_observation import LOWER_DIM
    from hrl_v2.hybrid_simplex import deterministic_decode
    actor = LiteLowerActor()
    with torch.no_grad():
        for p in actor.parameters():                 # random weights, so the support head is not all zero
            p.add_(0.5*torch.randn_like(p))
        logits, kappa = actor.decode(torch.randn(4, LOWER_DIM))
    q = logits.softmax(-1)
    if not (q[..., FULL] == 1).all() or not (q[..., :FULL] == 0).all() or not torch.isfinite(logits).all():
        raise RuntimeError('NoSupport mask failed')
    routes = deterministic_decode(logits, kappa)
    if not (routes > 0).all():
        raise RuntimeError('NoSupport: a deterministic ratio is not positive')


def root(base):
    return Path(base)/NAME


def prepare(main, base, seeds, smoke=False):
    r = root(base)
    r.mkdir(parents=True, exist_ok=True)
    if (Path(main)/'recipe.json').exists() and not (r/'recipe.json').exists():
        C.write_json(r/'recipe.json', C.read_json(Path(main)/'recipe.json'))
    if C.recipe_name(r) != C.recipe_name(main):
        raise ValueError(f'{r} and {main} use different fine-tuning recipes')
    for seed in seeds:
        upper = main_upper(main, seed)
        if not upper.exists():
            raise FileNotFoundError(f'{upper} (the main run\'s selected stage-2 PPO) is missing')
        target = C.seed_dir(r, seed)/'main_selected.json'
        if not target.exists():
            C.write_json(target, C.read_json(C.seed_dir(main, seed)/'selected.json'))
    if not (r/'nosupport.json').exists():
        C.write_json(r/'nosupport.json', dict(ablation=METHOD, main=str(Path(main).resolve()), smoke=smoke,
                                              note=__doc__.split('\n\n')[0]))


def main_upper(main, seed):
    selected = C.read_json(C.seed_dir(main, seed)/'selected.json')
    return C.run_dir(main, seed, 'M2')/'ckpt'/f'ep{selected["ProposedUpper"]:04d}.pt'


def train_job(base, main, stage, seed, smoke=False):
    patch()
    r = root(base)
    C.set_active_root(r)
    from p2 import learn
    init = {}
    if stage == 'M3':
        init = dict(m1=C.run_dir(r, seed, 'M1')/'final.pt', upper=main_upper(main, seed))
        for path in init.values():
            if not Path(path).exists():
                raise FileNotFoundError(f'M3 needs {path}')
    learn.train_sac_stage(stage, seed, r, smoke=smoke, init_paths=init)


def candidate_paths(base, main, seed, candidate):
    r = root(base)
    if int(candidate) == 0:
        return 'Proposed', C.run_dir(r, seed, 'M1')/'final.pt', main_upper(main, seed)
    return 'Proposed', C.run_dir(r, seed, 'M3')/'ckpt'/f'ep{int(candidate):04d}.pt', None


def _evaluate(base, main, seed, candidate, split, scenarios, folder, smoke, **extra):
    from p2 import evaluate
    method, learner, upper = candidate_paths(base, main, seed, candidate)
    make = evaluate.factory(method, seed, learner, upper)
    scene = C.load_base_scene()
    rows = []
    for sc in scenarios:
        rows.append(evaluate.evaluate_one(make, scene, split, sc, Path(folder)/f'{sc:02d}.json', smoke=smoke,
                                          identity=dict(seed=seed, method=METHOD, candidate=int(candidate),
                                                        learner=str(learner), upper=str(upper) if upper else None,
                                                        **extra)))
    return rows


def dev_job(base, main, seed, candidate, smoke=False):
    patch()
    import numpy as np
    d = C.design(smoke)
    folder = C.seed_dir(root(base), seed)/'dev'/METHOD/str(int(candidate))
    rows = _evaluate(base, main, seed, candidate, 'dev', d['dev'], folder, smoke)
    summary = dict(label=METHOD, candidate=int(candidate), seed=seed, scenarios=list(d['dev']),
                   **{k: float(np.mean([r[k] for r in rows])) for k in (C.SELECTION_METRIC, 'truncated_delay_s',
                                                                          'energy_w_total')})
    C.write_json(folder/'summary.json', summary)
    return summary


def select_job(base, seed, smoke=False):
    """Same rule as the main run (lowest mean dev cost per second, ties: later)."""
    candidates = tuple(C.design(smoke)['lower_candidates'])
    sd = C.seed_dir(root(base), seed)
    scores = {c: C.read_json(sd/'dev'/METHOD/str(c)/'summary.json')[C.SELECTION_METRIC] for c in candidates}
    best = min(candidates, key=lambda c: (scores[c], -c))
    out = dict(metric=C.SELECTION_METRIC, scores={str(c): scores[c] for c in candidates}, selected=best)
    C.write_json(sd/'selection.json', out)
    return out


def test_dir(base, seed):
    return C.seed_dir(root(base), seed)/'test'/METHOD


def test_job(base, main, seed, smoke=False):
    patch()
    d = C.design(smoke)
    candidate = C.read_json(C.seed_dir(root(base), seed)/'selection.json')['selected']
    _evaluate(base, main, seed, candidate, 'test', d['test'], test_dir(base, seed), smoke, experiment='ablation')


def jobs(seeds, Job, smoke=False):
    out = []
    for seed in seeds:
        S = f'--seed={seed}'
        m1, m3 = f'ns_s{seed}_train_M1', f'ns_s{seed}_train_M3'
        out.append(Job(m1, ['ns-train', S, '--stage=M1'], [], 110))
        out.append(Job(m3, ['ns-train', S, '--stage=M3'], [m1], 108))
        devs = []
        for c in C.design(smoke)['lower_candidates']:
            jid = f'ns_s{seed}_dev_{c}'
            out.append(Job(jid, ['ns-dev', S, f'--candidate={c}'], [m1] if int(c) == 0 else [m3], 106))
            devs.append(jid)
        sel = f'ns_s{seed}_select'
        out.append(Job(sel, ['ns-select', S], devs, 105))
        out.append(Job(f'ns_s{seed}_test', ['ns-test', S], [sel], 104))
    return out
