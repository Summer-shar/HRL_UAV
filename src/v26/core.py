"""Version 2.6: longer SAC fine-tuning, with or without region bursts in training.

Nothing of v2.5.0 is edited.  This module calls the unchanged p2 / hrl_v2 code
and changes its behaviour only inside the processes started by run_v26.py:

* ``C.design`` is wrapped so that the fine-tuning stages (M3, HoverSAC) run to
  300 windows instead of 100 and are saved every 25 windows;
* for the variant ``burst`` the training adapter installs, in half of the
  training windows, a region burst (all terminals of one random region at the
  busy intensity for 60-150 s) - the per-terminal activity model is unchanged.

Both variants CONTINUE the main run: the snapshot after window 100 (networks,
optimisers, replay, random state) is copied from the main output root, which is
only read.  Candidates 0-100 of the main run stay in the candidate pool, so the
dev cost of the selected model cannot get worse.

Layout of --out-base:
    ext/, burst/            variant roots (same layout as a main root, stages M3 / HoverSAC only)
    burstdev/<variant>/     burst experiment on the dev scenarios (decision data)
    ablation/NoStage3/      MH-TPCO without stage 3 on the test scenarios
    zeroshot/<variant>/     zero-shot sweep of the adopted variant
    results/                tables and consolidated records (pushed to GitHub)
    V26.md, DECISION.md     status page and the decision
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from p2 import common as C

RUN = C.REPO/'run_v26.py'
VARIANTS = ('ext', 'burst')
LABELS = dict(ProposedLower='M3', HoverSAC='HoverSAC')        # dev label -> fine-tuning stage
METHOD_LABEL = dict(Proposed='ProposedLower', HoverSAC='HoverSAC')
BASE, TOTAL, STEP = 100, 300, 25                              # windows of the main run / of v2.6 / between candidates
SMOKE_SIZES = dict(base=3, total=6, old=(0, 2, 3), new=(5, 6))
BURST_TRAIN = dict(probability=0.5, start=(0.1, 0.7), duration=(1/15, 1/6))   # fractions of the window
TOLERANCE = 0.01                                              # decision rule, see decide()
STATUS_PAGE = 'V26.md'


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def sizes(smoke=False):
    if smoke:
        s = dict(SMOKE_SIZES)
    else:
        s = dict(base=BASE, total=TOTAL, old=tuple(C.LOWER_CANDIDATES),
                 new=tuple(range(BASE+STEP, TOTAL+1, STEP)))
    s['all'] = tuple(s['old'])+tuple(s['new'])
    return s


_ORIGINAL_DESIGN = C.design


def design(smoke=False):
    """C.design with the longer fine-tuning budget and the extended candidate list."""
    d = _ORIGINAL_DESIGN(smoke)
    s = sizes(smoke)
    if tuple(d['lower_candidates']) != tuple(s['old']) or d['budget']['M3'] != s['base'] \
            or d['budget']['HoverSAC'] != s['base']:
        raise RuntimeError('v2.6 expects the fine-tuning design of the main run (100 windows, candidates 0-100)')
    d['budget'] = dict(d['budget'], M3=s['total'], HoverSAC=s['total'])
    d['lower_candidates'] = s['all']
    return d


def patch():
    """Only in this process: the pipeline code sees the v2.6 fine-tuning budget."""
    C.design = design


def variant_root(base, variant):
    if variant not in VARIANTS:
        raise ValueError(f'unknown variant {variant!r}')
    return Path(base)/variant


# ---------------------------------------------------------------------------
# Preparation: copy what the continuation needs from the main root (read only)
# ---------------------------------------------------------------------------
def _copy(src, dst):
    if dst.exists():
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name+'.copying')
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    return 1


def check_roots(main, base):
    main, base = Path(main).resolve(), Path(base).resolve()
    if main == base or main in base.parents or base in main.parents:
        raise ValueError('--out-base must be a separate folder; the main root is only read')
    return main, base


def required(main, seeds, smoke=False):
    s = sizes(smoke)
    need = []
    for seed in seeds:
        sd = C.seed_dir(main, seed)
        need += [sd/'selected.json', C.run_dir(main, seed, 'M1')/'final.pt']
        for label, stage in LABELS.items():
            rd = C.run_dir(main, seed, stage)
            need += [rd/'state.pt', rd/'contract.json', rd/'train.jsonl', rd/'done.json']
            need += [sd/'dev'/label/str(c)/'summary.json' for c in s['old']]
    return need


def prepare(main, base, seeds, smoke=False):
    main, base = check_roots(main, base)
    s = sizes(smoke)
    missing = [str(p) for p in required(main, seeds, smoke) if not p.exists()]
    if missing:
        raise FileNotFoundError('the main root lacks files needed to continue the fine-tuning:\n  ' +
                                '\n  '.join(missing))
    if not smoke and C.recipe_name(main) != 'v2':
        raise ValueError(f'{main} does not use fine-tuning recipe v2 (the recipe of the paper)')
    for seed in seeds:
        for stage in LABELS.values():
            lines = len(C.read_jsonl(C.run_dir(main, seed, stage)/'train.jsonl'))
            if lines != s['base']:
                raise ValueError(f'{C.run_dir(main, seed, stage)/"train.jsonl"} has {lines} windows, '
                                 f'expected {s["base"]}')
    placed = 0
    for variant in VARIANTS:
        root = variant_root(base, variant)
        root.mkdir(parents=True, exist_ok=True)
        if (main/'recipe.json').exists():
            placed += _copy(main/'recipe.json', root/'recipe.json')
        if C.recipe_name(root) != C.recipe_name(main):
            raise ValueError(f'{root} and {main} use different fine-tuning recipes')
        for seed in seeds:
            sd_main, sd = C.seed_dir(main, seed), C.seed_dir(root, seed)
            for stage in LABELS.values():
                for name in ('state.pt', 'contract.json', 'train.jsonl'):
                    placed += _copy(C.run_dir(main, seed, stage)/name, C.run_dir(root, seed, stage)/name)
            for label in LABELS:
                for c in s['old']:
                    for f in sorted((sd_main/'dev'/label/str(c)).glob('*.json')):
                        placed += _copy(f, sd/'dev'/label/str(c)/f.name)
            placed += _copy(sd_main/'selected.json', sd/'selected.json')
            placed += _copy(sd_main/'selected.json', sd/'main_selected.json')
        info = root/'v26.json'
        if not info.exists():
            C.write_json(info, dict(variant=variant, main=str(main), continued_from_window=s['base'],
                                    windows=s['total'], new_candidates=list(s['new']), smoke=smoke,
                                    burst_training=BURST_TRAIN if variant == 'burst' else None,
                                    recipe=C.recipe_name(root), created=now()))
    return placed


# ---------------------------------------------------------------------------
# Region bursts in training (variant "burst")
# ---------------------------------------------------------------------------
def burst_plan(arrival_seed, horizon_s, dt):
    """Deterministic in the window's arrival seed: None, or a burst of one region."""
    rng = np.random.default_rng(np.random.SeedSequence([int(arrival_seed), 26001]))
    if rng.random() >= BURST_TRAIN['probability']:
        return None
    region = int(rng.integers(3))
    start = round(rng.uniform(*BURST_TRAIN['start'])*horizon_s/dt)*dt
    length = max(dt, round(rng.uniform(*BURST_TRAIN['duration'])*horizon_s/dt)*dt)
    return dict(region=region, start_s=float(start), end_s=float(min(start+length, horizon_s)))


def install_region(env, region, start_s, end_s):
    """p2.burst.install for a given region: during [start_s, end_s) every terminal of
    the region generates tasks at the busy intensity.  Only this environment's
    ``_scene.frame`` is wrapped; the multiplier is restored after every frame."""
    scene = env._scene
    members = [k for k, t in enumerate(scene.terminals) if t.source == region]
    others = [k for k, t in enumerate(scene.terminals) if t.source != region]
    profiles = {id(scene._profiles[k]): scene._profiles[k] for k in members}
    if len(profiles) != 1:
        raise RuntimeError('a region is expected to carry exactly one business profile')
    profile = next(iter(profiles.values()))
    act = profile["activity_parameters"]
    if any(scene._profiles[k] is profile or scene._profiles[k]["activity_parameters"] is act for k in others):
        raise RuntimeError('the burst profile is shared with another region')
    dt = scene.config["time"]["routing_s"]
    original = scene.frame

    def frame(index):
        if start_s <= index*dt < end_s:
            saved = act["normal_multiplier"]
            act["normal_multiplier"] = act["busy_multiplier"]
            try:
                return original(index)
            finally:
                act["normal_multiplier"] = saved
        return original(index)

    scene.frame = frame
    return dict(region=int(region), start_s=float(start_s), end_s=float(end_s),
                busy_multiplier=act["busy_multiplier"])


def install_training_bursts(perf, log_path=None):
    """Replace p2.perf.TrainingAdapter (in this process) by a subclass whose reset()
    installs the window's burst plan."""
    base_class = perf.TrainingAdapter

    class BurstTrainingAdapter(base_class):
        def reset(self, *, arrival_seed, layout_seed, load_seed):
            report = super().reset(arrival_seed=arrival_seed, layout_seed=layout_seed, load_seed=load_seed)
            plan = burst_plan(arrival_seed, self.timing.rollout_seconds, self.timing.lower_interval)
            self.v26_burst = install_region(self.env, **plan) if plan is not None else None
            if log_path is not None:
                C.append_jsonl(log_path, dict(arrival_seed=int(arrival_seed), burst=self.v26_burst))
            return report

    perf.TrainingAdapter = BurstTrainingAdapter
    return BurstTrainingAdapter


# ---------------------------------------------------------------------------
# Jobs (each runs in its own process: run_v26.py <command> ...)
# ---------------------------------------------------------------------------
def train_job(base, main, variant, stage, seed, smoke=False):
    patch()
    root = variant_root(base, variant)
    C.set_active_root(root)
    rd = C.run_dir(root, seed, stage)
    if not (rd/'state.pt').exists() and not (rd/'final.pt').exists():
        raise FileNotFoundError(f'{rd/"state.pt"} is missing; run "run_v26.py prepare" first')
    from p2 import learn, perf
    if variant == 'burst':
        install_training_bursts(perf, rd/'bursts.jsonl')
    learn.train_sac_stage(stage, seed, root, smoke=smoke, init_paths={})
    logged = len(C.read_jsonl(rd/'train.jsonl'))
    if logged < sizes(smoke)['total']:
        raise RuntimeError(f'{rd}: {logged} windows logged, expected {sizes(smoke)["total"]}')


def dev_job(base, variant, label, candidate, seed, smoke=False):
    patch()
    root = variant_root(base, variant)
    C.set_active_root(root)
    if int(candidate) not in sizes(smoke)['new']:
        raise ValueError('only the new candidates are evaluated; the old dev records are copied from the main run')
    from p2 import evaluate
    return evaluate.dev_job(root, seed, label, int(candidate), smoke=smoke)


def select_job(base, variant, seed, smoke=False):
    """Same rule as p2.evaluate.select (lowest mean dev cost per second, ties: later),
    over the old and the new candidates."""
    s = sizes(smoke)
    sd = C.seed_dir(variant_root(base, variant), seed)
    selected, main_selected = C.read_json(sd/'selected.json'), C.read_json(sd/'main_selected.json')
    out = {}
    for label in LABELS:
        scores = {c: C.read_json(sd/'dev'/label/str(c)/'summary.json')[C.SELECTION_METRIC] for c in s['all']}
        best = min(s['all'], key=lambda c: (scores[c], -c))
        old_best = min(s['old'], key=lambda c: (scores[c], -c))
        if old_best != main_selected[label]:
            raise RuntimeError(f'seed {seed} {label}: the main run selected {main_selected[label]}, '
                               f'its dev records give {old_best}')
        selected[label] = best
        out[label] = dict(metric=C.SELECTION_METRIC, scores={str(c): scores[c] for c in s['all']},
                          selected=best, main_selected=old_best, changed=best != old_best,
                          dev_cost=scores[best], main_dev_cost=scores[old_best])
    C.write_json(sd/'selected.json', selected)
    C.write_json(sd/'v26_selection.json', out)
    return out


def selection(base, variant, seed):
    return C.read_json(C.seed_dir(variant_root(base, variant), seed)/'v26_selection.json')


def policy_paths(base, main, variant, seed, method, smoke=False):
    """(method, learner checkpoint, upper checkpoint or None) of the dev-selected model.
    variant 'main': the main run.  A variant whose selection is an old candidate
    resolves to the main run's checkpoint (the same model)."""
    from p2 import evaluate
    if variant == 'main' or method not in METHOD_LABEL:
        return evaluate.method_paths(main, seed, method)
    label = METHOD_LABEL[method]
    candidate = selection(base, variant, seed)[label]['selected']
    if candidate <= sizes(smoke)['base']:
        return evaluate.candidate_paths(main, seed, label, candidate, C.read_json(C.seed_dir(main, seed)/'selected.json'))
    return evaluate.candidate_paths(variant_root(base, variant), seed, label, candidate, None)


def _identity(seed, method, variant, learner, upper, **more):
    return dict(seed=seed, method=method, variant=variant, learner=str(learner) if learner else None,
                upper=str(upper) if upper else None, **more)


def _evaluate(base, main, variant, seed, method, smoke, jobs):
    """jobs: iterable of (scene, split, scenario, out_json, extra kwargs of evaluate_one, identity extras)."""
    from p2 import evaluate
    method_, learner, upper = policy_paths(base, main, variant, seed, method, smoke)
    make = evaluate.factory(method_, seed, learner, upper)
    for scene, split, scenario, out_json, kwargs, extra in jobs:
        evaluate.evaluate_one(make, scene, split, scenario, out_json, smoke=smoke,
                              identity=_identity(seed, method, variant, learner, upper, **extra), **kwargs)


def burstdev_dir(base, variant, seed):
    return Path(base)/'burstdev'/variant/f'seed_{seed}'


def burstdev_job(base, main, variant, seed, smoke=False):
    """The burst experiment of the paper on the DEV scenarios (decision data, no test scenario)."""
    from p2 import burst as B
    d, scene = _ORIGINAL_DESIGN(smoke), C.load_base_scene()
    folder = burstdev_dir(base, variant, seed)
    if variant != 'main' and not selection(base, variant, seed)['ProposedLower']['changed']:
        C.write_json(folder/'same_as_main.json', dict(note='the selection is the main run\'s model'))
        return
    _evaluate(base, main, variant, seed, 'Proposed', smoke,
              [(scene, 'dev', sc, folder/f'{sc:02d}.json', dict(burst=B.window(smoke)), dict(experiment='burstdev'))
               for sc in d['dev']])


def ablation_job(base, main, seed, smoke=False):
    """MH-TPCO without stage 3: the stage-1 SAC with the selected stage-2 PPO (candidate 0)."""
    from p2 import evaluate
    d, scene = _ORIGINAL_DESIGN(smoke), C.load_base_scene()
    method, learner, upper = evaluate.candidate_paths(main, seed, 'ProposedLower', 0,
                                                      C.read_json(C.seed_dir(main, seed)/'selected.json'))
    make = evaluate.factory(method, seed, learner, upper)
    folder = Path(base)/'ablation'/'NoStage3'/f'seed_{seed}'
    for sc in d['test']:
        evaluate.evaluate_one(make, scene, 'test', sc, folder/f'{sc:02d}.json', smoke=smoke,
                              identity=_identity(seed, 'NoStage3', 'main', learner, upper, experiment='ablation'))


def test_dir(base, variant, seed, method):
    return C.seed_dir(variant_root(base, variant), seed)/'test'/method


def burst_dir(base, variant, seed, method):
    return C.seed_dir(variant_root(base, variant), seed)/'burst'/method


def zeroshot_dir(base, variant, point, seed, method):
    return Path(base)/'zeroshot'/variant/point/f'seed_{seed}'/method


def should_run(base, variant, seed, method, smoke=False):
    """Evaluation jobs of phase 2 run only for the adopted variant and only where its selected
    model differs from the main run (smoke runs: always, to exercise the path)."""
    decision = C.read_json(Path(base)/'results'/'v26_decision.json')
    if decision['adopted'] != variant:
        return False
    return bool(smoke or changed(base, variant, seed, method))


def _skip(base, variant, seed, method, smoke):
    if should_run(base, variant, seed, method, smoke):
        return False
    print(f'not needed: variant {variant} not adopted or seed {seed} {method} unchanged', flush=True)
    return True


def test_job(base, main, variant, seed, method, smoke=False):
    if _skip(base, variant, seed, method, smoke):
        return
    d, scene = _ORIGINAL_DESIGN(smoke), C.load_base_scene()
    folder = test_dir(base, variant, seed, method)
    _evaluate(base, main, variant, seed, method, smoke,
              [(scene, 'test', sc, folder/f'{sc:02d}.json', dict(traj_npz=folder/'traj'/f'{sc:02d}.npz'), {})
               for sc in d['test']])


def burst_job(base, main, variant, seed, method, smoke=False):
    if _skip(base, variant, seed, method, smoke):
        return
    from p2 import burst as B
    d, scene = _ORIGINAL_DESIGN(smoke), C.load_base_scene()
    folder = burst_dir(base, variant, seed, method)
    _evaluate(base, main, variant, seed, method, smoke,
              [(scene, 'test', sc, folder/f'{sc:02d}.json', dict(burst=B.window(smoke)), dict(experiment='burst'))
               for sc in d['sweep_scenarios']])


def zeroshot_points(smoke=False):
    from p2 import pipeline, retrain
    return [(pipeline.point_name(p, v), p, v) for p, v in retrain.zeroshot_points(smoke)]


def zeroshot_job(base, main, variant, seed, method, point, smoke=False):
    if _skip(base, variant, seed, method, smoke):
        return
    d = _ORIGINAL_DESIGN(smoke)
    name, param, value = next(x for x in zeroshot_points(smoke) if x[0] == point)
    scene = C.scaled_scene(C.load_base_scene(), **{param: value})
    folder = zeroshot_dir(base, variant, name, seed, method)
    _evaluate(base, main, variant, seed, method, smoke,
              [(scene, 'test', sc, folder/f'{sc:02d}.json', {},
                dict(param=param, value=value, trained_at='nominal point')) for sc in d['test']])


# ---------------------------------------------------------------------------
# Decision (dev data only; written before any test record of v2.6 exists)
# ---------------------------------------------------------------------------
def _mean_delay(r):
    nh, nl = r['high_arrived'], r['low_arrived']
    return (nh*r['high_truncated_delay_s']+nl*r['low_truncated_delay_s'])/(nh+nl)


def _burstdev_rows(base, variant, seed):
    folder = burstdev_dir(base, variant, seed)
    if variant != 'main' and (folder/'same_as_main.json').exists():
        folder = burstdev_dir(base, 'main', seed)
    rows = [C.read_json(p) for p in sorted(folder.glob('[0-9][0-9].json'))]
    if not rows:
        raise FileNotFoundError(f'no burst-dev records in {folder}')
    return rows


def decide(base, seeds, smoke=False, force=None):
    """Rule fixed in docs/V26.md before the run:
    J = mean dev cost of the selected MH-TPCO model (nominal dev scenarios, mean over seeds);
    B = mean cost of the same models in the burst experiment on the dev scenarios.
    burst is adopted if its selection differs from the main run, J_burst <= (1+1 %) J_ext and
    B_burst < B_ext; otherwise ext if its selection differs from the main run; otherwise main
    (nothing changes).  J_ext <= J_main and J_burst <= J_main hold by construction."""
    base = Path(base)
    table = {}
    for variant in ('main',)+VARIANTS:
        per_seed = {}
        for seed in seeds:
            if variant == 'main':
                sel = selection(base, 'ext', seed)
                row = dict(selected=sel['ProposedLower']['main_selected'], dev_cost=sel['ProposedLower']['main_dev_cost'],
                           changed=False, hover_selected=sel['HoverSAC']['main_selected'], hover_changed=False)
            else:
                sel = selection(base, variant, seed)
                row = dict(selected=sel['ProposedLower']['selected'], dev_cost=sel['ProposedLower']['dev_cost'],
                           changed=sel['ProposedLower']['changed'], hover_selected=sel['HoverSAC']['selected'],
                           hover_changed=sel['HoverSAC']['changed'])
            rows = _burstdev_rows(base, variant, seed)
            row.update(burst_cost=float(np.mean([r['cost_per_s'] for r in rows])),
                       burst_delay_s=float(np.mean([_mean_delay(r) for r in rows])),
                       burst_high_delay_s=float(np.mean([r['high_truncated_delay_s'] for r in rows])))
            per_seed[str(seed)] = row
        table[variant] = dict(per_seed=per_seed,
                              J=float(np.mean([r['dev_cost'] for r in per_seed.values()])),
                              B=float(np.mean([r['burst_cost'] for r in per_seed.values()])),
                              changed=any(r['changed'] for r in per_seed.values()))
    ext, burst = table['ext'], table['burst']
    if burst['changed'] and burst['J'] <= (1+TOLERANCE)*ext['J'] and burst['B'] < ext['B']:
        adopted, why = 'burst', 'nominal dev cost within 1 % of ext and lower cost in the burst experiment on dev'
    elif ext['changed']:
        adopted, why = 'ext', 'burst does not meet both conditions; ext lowers the nominal dev cost'
    else:
        adopted, why = 'main', 'no variant gives a different, better model on dev; the main results stay'
    forced = None
    if force and adopted != force:
        forced, adopted = adopted, force        # smoke runs only: exercise the evaluation of a variant
    decision = dict(adopted=adopted, reason=why, rule=decide.__doc__, tolerance=TOLERANCE, table=table,
                    forced_from=forced, seeds=list(seeds), smoke=smoke, decided=now())
    C.write_json(base/'results'/'v26_decision.json', decision)
    (base/'DECISION.md').write_text(decision_text(decision), encoding='utf-8')
    return decision


def decision_text(d):
    lines = ['# v2.6 的选择（只用验证场景）', '', f'**采用：{d["adopted"]}**（{d["reason"]}）', '']
    if d.get('forced_from'):
        lines += [f'（冒烟运行：按规则应为 {d["forced_from"]}，为检查评估流程改为 {d["adopted"]}）', '']
    lines += ['| 版本 | 标称验证代价 J | 突发验证代价 B | 与现有模型不同 |', '|---|---|---|---|']
    for v in ('main',)+VARIANTS:
        t = d['table'][v]
        lines.append(f'| {v} | {t["J"]:.4f} | {t["B"]:.4f} | {"是" if t["changed"] else "否"} |')
    lines += ['', '| 版本 | 种子 | MH-TPCO 选中的窗口 | 验证代价 | 突发验证：代价 / 平均时延 / 高优先级时延 | HoverSAC 选中的窗口 |',
              '|---|---|---|---|---|---|']
    for v in ('main',)+VARIANTS:
        for seed, r in d['table'][v]['per_seed'].items():
            lines.append(f'| {v} | {seed} | {r["selected"]} | {r["dev_cost"]:.4f} | {r["burst_cost"]:.3f} / '
                         f'{r["burst_delay_s"]:.3f} s / {r["burst_high_delay_s"]:.3f} s | {r["hover_selected"]} |')
    lines += ['', '规则（运行前写定，见 docs/V26.md）：burst 的模型与现有不同、J_burst ≤ 1.01 J_ext 且 B_burst < B_ext 时采用 burst；'
              '否则 ext 的模型与现有不同时采用 ext；否则保持现有结果。', '']
    return '\n'.join(lines)


def changed(base, variant, seed, method):
    return bool(selection(base, variant, seed)[METHOD_LABEL[method]]['changed'])


# ---------------------------------------------------------------------------
# Job graph and runner
# ---------------------------------------------------------------------------
class Job:
    def __init__(self, jid, args, deps=(), priority=0):
        self.id, self.args, self.deps, self.priority = jid, list(args), list(deps), priority


def plan_phase1(seeds, smoke=False):
    """Training of both variants, dev evaluation of the new candidates, selection,
    burst experiment on dev, and the no-stage-3 ablation."""
    s, jobs = sizes(smoke), []
    for seed in seeds:
        S = f'--seed={seed}'
        jobs.append(Job(f's{seed}_ablation_NoStage3', ['ablation', S], [], 5))
        jobs.append(Job(f's{seed}_burstdev_main', ['burstdev', '--variant=main', S], [], 6))
        for variant in VARIANTS:
            V = f'--variant={variant}'
            devs = []
            for label, stage in LABELS.items():
                first = label == 'ProposedLower'
                train = f'{variant}_s{seed}_train_{stage}'
                jobs.append(Job(train, ['train', V, S, f'--stage={stage}'], [], 100 if first else 90))
                for c in s['new']:
                    jid = f'{variant}_s{seed}_dev_{label}_{c}'
                    jobs.append(Job(jid, ['dev', V, S, f'--label={label}', f'--candidate={c}'], [train],
                                    70 if first else 60))
                    devs.append(jid)
            select = f'{variant}_s{seed}_select'
            jobs.append(Job(select, ['select', V, S], devs, 80))
            jobs.append(Job(f'{variant}_s{seed}_burstdev', ['burstdev', V, S], [select], 75))
    return jobs


def plan_phase2(seeds, smoke=False, deps=('decide',)):
    """Test, burst experiment and zero-shot sweep for both variants; each job checks at run
    time (should_run) whether it is needed, so the whole run is one job graph."""
    jobs = []
    for variant in VARIANTS:
        V_ = f'--variant={variant}'
        for seed in seeds:
            S = f'--seed={seed}'
            for method in METHOD_LABEL:
                M = f'--method={method}'
                jobs.append(Job(f'{variant}_s{seed}_test_{method}', ['test', V_, S, M], deps, 50))
                jobs.append(Job(f'{variant}_s{seed}_burst_{method}', ['burst', V_, S, M], deps, 40))
                for name, _, _ in zeroshot_points(smoke):
                    jobs.append(Job(f'{variant}_s{seed}_zeroshot_{method}_{name}',
                                    ['zeroshot', V_, S, M, f'--point={name}'], deps,
                                    30 if method == 'Proposed' else 20))
    return jobs


def plan_all(seeds, smoke=False):
    """Everything after `prepare` in one job graph: phase 1, the decision (a job that waits for
    phase 1), the phase-2 evaluations, the no-stage-3 ablation and the NoSupport ablation."""
    from . import nosupport
    first = plan_phase1(seeds, smoke)
    decide = Job('decide', ['decide'], [j.id for j in first if not j.id.endswith('_ablation_NoStage3')], 79)
    return first + [decide] + plan_phase2(seeds, smoke) + nosupport.jobs(seeds, Job, smoke)


def run_jobs(jobs, base, main, workers=4, smoke=False, poll=None, poll_seconds=600, seeds=None):
    """Dependency-aware runner (p2.pipeline.run for run_v26.py).  Returns the failed job ids."""
    base = Path(base)
    jd = base/'_jobs'
    jd.mkdir(parents=True, exist_ok=True)
    marker = {j.id: jd/f'{j.id}.done' for j in jobs}
    done = {j.id for j in jobs if marker[j.id].exists()}
    failed, running = set(), {}
    pending = [j for j in jobs if j.id not in done]
    common = [f'--main={main}', f'--out-base={base}'] + (['--smoke'] if smoke else []) + \
        ([f'--seeds={",".join(map(str, seeds))}'] if seeds else [])
    print(f'[{now()}] {len(done)} jobs already done, {len(pending)} to run with {workers} workers; logs in {jd}',
          flush=True)
    last_poll = time.time()
    while pending or running:
        blocked = [j for j in pending if any(dep in failed for dep in j.deps)]
        for j in blocked:
            failed.add(j.id)
            print(f'SKIP {j.id} (dependency failed)', flush=True)
        pending = [j for j in pending if j not in blocked]
        ready = sorted((j for j in pending if all(dep in done for dep in j.deps)), key=lambda j: -j.priority)
        while ready and len(running) < workers:
            j = ready.pop(0)
            pending.remove(j)
            log = (jd/f'{j.id}.log').open('a', encoding='utf-8')
            log.write(f'\n=== start {now()} ===\n')
            log.flush()
            proc = subprocess.Popen([sys.executable, str(RUN)]+j.args+common, stdout=log,
                                    stderr=subprocess.STDOUT, cwd=str(C.REPO))
            running[j.id] = (proc, log, time.time())
            print(f'START {j.id}', flush=True)
        time.sleep(0.2 if smoke else 2)
        for jid, (proc, log, t0) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close()
            del running[jid]
            minutes = (time.time()-t0)/60
            if code == 0:
                marker[jid].write_text(now()+'\n')
                done.add(jid)
                print(f'DONE  {jid} ({minutes:.1f} min) [{len(done)}/{len(jobs)}]', flush=True)
            else:
                failed.add(jid)
                print(f'FAIL  {jid} (exit {code}, {minutes:.1f} min) -> see {jd/(jid+".log")}', flush=True)
        if poll and time.time()-last_poll >= poll_seconds:
            last_poll = time.time()
            poll(dict(done=len(done), total=len(jobs), running=sorted(running), failed=sorted(failed)))
        if not running and pending and not any(all(dep in done for dep in j.deps) for j in pending):
            for j in pending:
                failed.add(j.id)
                print(f'SKIP {j.id} (dependency failed)', flush=True)
            break
    return sorted(failed)
