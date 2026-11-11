"""Torch-free test of the v2.6 extension (src/v26, run_v26.py).

Checks: the v2.5.0 code fingerprint is unchanged by the new files; the wrapped
design (300 windows, candidates up to 300) and the unchanged original design;
the job graphs; the training burst plan (deterministic, about half of the
windows, inside the window) and its installation on the real training adapter
(arrivals identical before the burst, more arrivals in the burst region only,
scene left unchanged); and, with the scripted CA policy standing in for the
networks, the whole flow on a fake main root: prepare (main root untouched),
selection over old and new candidates, burst experiment on dev, the decision
rule (all three outcomes), test / burst / zero-shot evaluation, the ablation,
the report, and the job runner (dependencies, failures, resume).
Usage:  python tests/test_v26.py
"""
import hashlib
import importlib
import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
import numpy as np  # noqa: E402

from p2 import common as C  # noqa: E402

try:
    importlib.import_module('torch')
except ImportError:   # collector imports SUPPORTS from a torch module; stub it
    stub = types.ModuleType('hrl_v2.hybrid_simplex')
    stub.SUPPORTS = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2))
    sys.modules['hrl_v2.hybrid_simplex'] = stub

from p2 import evaluate, perf, retrain  # noqa: E402
from v26 import core as V, report as R  # noqa: E402

FINGERPRINT_V250 = 'f3aab1118cdf1b2e2ccc23b79630d55cb7893589d81ba3108cad1b1bb02c8e0f'   # branch paper2-release, 6a5d4bc


def tree_hash(root):
    h = hashlib.sha256()
    for f in sorted(Path(root).rglob('*')):
        if f.is_file():
            h.update(str(f.relative_to(root)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def test_design_and_plans():
    assert retrain.code_fingerprint()[0] == FINGERPRINT_V250, 'a v2.5.0 file was changed'
    d0, d1 = V._ORIGINAL_DESIGN(), V.design()
    assert d0['budget']['M3'] == 100 and tuple(d0['lower_candidates']) == (0, 25, 50, 75, 100)
    assert d1['budget']['M3'] == d1['budget']['HoverSAC'] == 300 and d1['budget']['M1'] == d0['budget']['M1']
    assert tuple(d1['lower_candidates']) == tuple(range(0, 301, 25))
    assert d1['dev'] == d0['dev'] and d1['test'] == d0['test'] and C.design is V._ORIGINAL_DESIGN
    jobs = V.plan_phase1(d0['seeds'])
    ids = [j.id for j in jobs]
    assert len(ids) == len(set(ids)) and all(dep in ids for j in jobs for dep in j.deps)
    assert sum('_train_' in i for i in ids) == 12 and sum('_dev_' in i for i in ids) == 96, len(ids)
    assert sum(i.endswith('_select') for i in ids) == 6 and sum('burstdev' in i for i in ids) == 9
    print(f'phase 1: {len(ids)} jobs')


def test_burst_plan():
    plans = [V.burst_plan(seed, 900., .5) for seed in range(2000)]
    assert plans == [V.burst_plan(seed, 900., .5) for seed in range(2000)]
    hit = [p for p in plans if p is not None]
    assert 0.45 < len(hit)/len(plans) < 0.55, len(hit)
    assert {p['region'] for p in hit} == {0, 1, 2}
    assert all(90 <= p['start_s'] <= 630 and 60 <= p['end_s']-p['start_s'] <= 150 and p['end_s'] <= 900 for p in hit)
    print(f'burst plan: {len(hit)}/{len(plans)} windows, mean length '
          f'{np.mean([p["end_s"]-p["start_s"] for p in hit]):.0f} s')


def _arrivals(adapter, policy):
    from hrl_v2.collector import collect_metrics
    collect_metrics(adapter, policy, identity={}, reward_scale=100.)
    rows = [(p.arrival.birth_s, p.arrival.source) for p in adapter.env.parents.values()]
    return np.asarray(rows)


def test_training_adapter():
    perf.apply()
    scene = C.load_base_scene()
    before = [dict(p['activity_parameters']) for p in scene['business_profiles']['catalog']]
    clk = C.clocks(scene, smoke=True)
    original = perf.TrainingAdapter
    try:
        burst_class = V.install_training_bursts(perf)
        assert perf.TrainingAdapter is burst_class and issubclass(burst_class, original)
    finally:
        perf.TrainingAdapter = original
    seeds = next(C.training_seeds(4101, 3, n) for n in range(101, 400)
                 if V.burst_plan(C.training_seeds(4101, 3, n)['arrival_seed'], 50., .5) is not None)
    quiet = next(C.training_seeds(4101, 3, n) for n in range(101, 400)
                 if V.burst_plan(C.training_seeds(4101, 3, n)['arrival_seed'], 50., .5) is None)
    policy = lambda: evaluate.ScriptedPolicy('hover', 'equal')        # noqa: E731
    plain = original(scene, timing=clk)
    plain.reset(**seeds)
    a = _arrivals(plain, policy())
    hot = burst_class(scene, timing=clk)
    hot.reset(**seeds)
    plan = hot.v26_burst
    b = _arrivals(hot, policy())
    assert plan is not None and plan['busy_multiplier'] == 3.0
    pre = lambda x: x[x[:, 0] < plan['start_s']]                      # noqa: E731
    assert np.array_equal(pre(a), pre(b)), 'arrivals before the burst differ'
    inside = lambda x, region: int(((x[:, 0] >= plan['start_s']) & (x[:, 0] < plan['end_s']) &      # noqa: E731
                                    (x[:, 1] == region)).sum())
    assert inside(b, plan['region']) > 1.2*inside(a, plan['region']), (inside(a, plan['region']), inside(b, plan['region']))
    for region in set(range(3))-{plan['region']}:
        assert inside(b, region) == inside(a, region), 'another region changed during the burst'
    assert [p['activity_parameters'] for p in scene['business_profiles']['catalog']] == before, 'scene modified'
    calm = burst_class(scene, timing=clk)
    calm.reset(**quiet)
    reference = original(scene, timing=clk)
    reference.reset(**quiet)
    assert calm.v26_burst is None and np.array_equal(_arrivals(calm, policy()), _arrivals(reference, policy()))
    print(f'training burst: region {plan["region"]}, {plan["start_s"]:.1f}-{plan["end_s"]:.1f} s, arrivals in the '
          f'region {inside(a, plan["region"])} -> {inside(b, plan["region"])}')


def fake_main(root, seed, scores):
    """A main root with the files prepare() needs and CA records standing in for every method."""
    s = V.sizes(smoke=True)
    d = V._ORIGINAL_DESIGN(smoke=True)
    sd = C.seed_dir(root, seed)
    C.write_json(root/'recipe.json', dict(finetune='v2'))
    C.write_json(sd/'selected.json', dict(ProposedUpper=5, TFOO=5, TOO=5, NC=5,
                                          **{label: min(s['old'], key=lambda c: (scores[label][c], -c))
                                             for label in V.LABELS}))
    (C.run_dir(root, seed, 'M1')).mkdir(parents=True)
    (C.run_dir(root, seed, 'M1')/'final.pt').write_bytes(b'm1')
    for label, stage in V.LABELS.items():
        rd = C.run_dir(root, seed, stage)
        rd.mkdir(parents=True)
        (rd/'state.pt').write_bytes(b'state after window 3')
        C.write_json(rd/'contract.json', dict(stage=stage, finetune=dict(recipe='v2')))
        C.write_json(rd/'done.json', dict(stage=stage))
        for n in range(1, s['base']+1):
            C.append_jsonl(rd/'train.jsonl', dict(stage=stage, seed=seed, episode=n, cost_per_s=7., actor_relative_entropy=-3.))
        for c in s['old']:
            C.write_json(sd/'dev'/label/str(c)/'summary.json', {C.SELECTION_METRIC: scores[label][c], 'candidate': c})
    make = evaluate.factory('CA', seed, None, None)
    scene = C.load_base_scene()
    from p2 import burst as B
    for method in ('CA', 'TFOO', 'TOO', 'Proposed', 'HoverSAC'):
        for sc in d['test']:
            evaluate.evaluate_one(make, scene, 'test', sc, sd/'test'/method/f'{sc:02d}.json', smoke=True,
                                  identity=dict(seed=seed, method=method))
        for sc in d['sweep_scenarios']:
            evaluate.evaluate_one(make, scene, 'test', sc, sd/'burst'/method/f'{sc:02d}.json', smoke=True,
                                  burst=B.window(True), identity=dict(seed=seed, method=method))


def fake_training_and_dev(base, seed, scores):
    s = V.sizes(smoke=True)
    for variant in V.VARIANTS:
        root = V.variant_root(base, variant)
        for label, stage in V.LABELS.items():
            rd = C.run_dir(root, seed, stage)
            assert (rd/'state.pt').read_bytes() == b'state after window 3'
            for n in range(s['base']+1, s['total']+1):
                C.append_jsonl(rd/'train.jsonl', dict(stage=stage, seed=seed, episode=n, cost_per_s=6.5,
                                                      actor_relative_entropy=-2.5))
            for c in s['new']:
                (rd/'ckpt').mkdir(exist_ok=True)
                (rd/'ckpt'/f'ep{c:04d}.pt').write_bytes(b'ckpt')
                C.write_json(C.seed_dir(root, seed)/'dev'/label/str(c)/'summary.json',
                             {C.SELECTION_METRIC: scores[variant][label][c], 'candidate': c})


def test_flow():
    tmp = Path(tempfile.mkdtemp(prefix='p2v26_'))
    main, base = tmp/'main', tmp/'v26'
    seed = V._ORIGINAL_DESIGN(smoke=True)['seeds'][0]
    seeds = (seed,)
    old = dict(ProposedLower={0: 6.2, 2: 6.4, 3: 6.0}, HoverSAC={0: 10.3, 2: 10.2, 3: 10.4})
    fake_main(main, seed, old)
    snapshot = tree_hash(main)
    for bad in (main, main/'inside'):
        try:
            V.check_roots(main, bad)
        except ValueError:
            pass
        else:
            raise AssertionError('an output folder inside the main root must be refused')
    (C.run_dir(main, seed, 'M3')/'state.pt').rename(C.run_dir(main, seed, 'M3')/'state.hidden')
    try:
        V.prepare(main, base, seeds, smoke=True)
    except FileNotFoundError as e:
        print('refused as expected:', str(e).splitlines()[0])
    else:
        raise AssertionError('a missing snapshot must stop the preparation')
    (C.run_dir(main, seed, 'M3')/'state.hidden').rename(C.run_dir(main, seed, 'M3')/'state.pt')
    placed = V.prepare(main, base, seeds, smoke=True)
    assert placed > 0 and V.prepare(main, base, seeds, smoke=True) == 0
    assert C.recipe_name(V.variant_root(base, 'ext')) == 'v2'
    new = dict(ext=dict(ProposedLower={5: 5.9, 6: 5.8}, HoverSAC={5: 10.25, 6: 10.3}),
               burst=dict(ProposedLower={5: 5.85, 6: 6.1}, HoverSAC={5: 10.1, 6: 10.3}))
    fake_training_and_dev(base, seed, new)
    sel = {v: V.select_job(base, v, seed, smoke=True) for v in V.VARIANTS}
    assert sel['ext']['ProposedLower']['selected'] == 6 and sel['ext']['ProposedLower']['main_selected'] == 3
    assert sel['ext']['HoverSAC']['selected'] == 2 and not sel['ext']['HoverSAC']['changed']
    assert sel['burst']['ProposedLower']['selected'] == 5 and sel['burst']['HoverSAC']['selected'] == 5
    assert C.read_json(C.seed_dir(V.variant_root(base, 'ext'), seed)/'selected.json')['TFOO'] == 5

    # the networks are replaced by the scripted CA policy
    paths_seen = []
    real_paths, real_candidate = V.policy_paths, evaluate.candidate_paths

    def fake_paths(base_, main_, variant, seed_, method, smoke=False):
        paths_seen.append((variant, method, real_paths_checked(base_, main_, variant, seed_, method)))
        return 'CA', None, None

    def real_paths_checked(base_, main_, variant, seed_, method):
        saved = evaluate.method_paths, evaluate.candidate_paths
        evaluate.method_paths = lambda root, s_, m: ('main', C.seed_dir(root, s_), None)
        evaluate.candidate_paths = real_candidate
        try:
            return str(real_paths(base_, main_, variant, seed_, method, smoke=True)[1])
        finally:
            evaluate.method_paths, evaluate.candidate_paths = saved

    V.policy_paths = fake_paths
    evaluate.candidate_paths = lambda root, seed_, label, candidate, selected=None: ('CA', None, None)
    try:
        for variant in ('main',)+V.VARIANTS:
            V.burstdev_job(base, main, variant, seed, smoke=True)
        decision = V.decide(base, seeds, smoke=True)
        # CA is the same policy in all three "variants": equal burst cost -> burst is not better -> ext
        assert decision['adopted'] == 'ext' and decision['table']['ext']['J'] == 5.8, decision
        assert decision['table']['main']['J'] == 6.0 and (base/'DECISION.md').exists()
        jobs = V.plan_phase2(base, decision, seeds, smoke=True)
        points = V.zeroshot_points(smoke=True)
        assert [j.id for j in jobs] == [f'ext_s{seed}_test_Proposed', f'ext_s{seed}_burst_Proposed'] + \
            [f'ext_s{seed}_zeroshot_Proposed_{n}' for n, _, _ in points], [j.id for j in jobs]
        assert len(V.plan_phase2(base, decision, seeds, smoke=True, force=True)) == 2*len(jobs)
        assert V.plan_phase2(base, dict(adopted='main'), seeds, smoke=True) == []
        V.test_job(base, main, 'ext', seed, 'Proposed', smoke=True)
        V.burst_job(base, main, 'ext', seed, 'Proposed', smoke=True)
        for name, _, _ in points:
            V.zeroshot_job(base, main, 'ext', seed, 'Proposed', name, smoke=True)
        V.ablation_job(base, main, seed, smoke=True)
    finally:
        V.policy_paths, evaluate.candidate_paths = real_paths, real_candidate
    # new candidates resolve to the variant root, the main run to the main root
    resolved = {(v, m): p for v, m, p in paths_seen}
    assert 'ep0006.pt' in resolved[('ext', 'Proposed')] and str(base) in resolved[('ext', 'Proposed')]
    assert 'ep0005.pt' in resolved[('burst', 'Proposed')] and str(main) in resolved[('main', 'Proposed')]
    record = C.read_json(V.zeroshot_dir(base, 'ext', points[0][0], seed, 'Proposed')/'21.json')
    assert record['param'] == points[0][1] and record['variant'] == 'ext'

    # decision rule: the two other outcomes
    def rule(j_ext, j_burst, b_ext, b_burst, ext_changed=True, burst_changed=True):
        table = dict(ext=dict(J=j_ext, B=b_ext, changed=ext_changed), burst=dict(J=j_burst, B=b_burst, changed=burst_changed))
        ext, burst = table['ext'], table['burst']
        if burst['changed'] and burst['J'] <= (1+V.TOLERANCE)*ext['J'] and burst['B'] < ext['B']:
            return 'burst'
        return 'ext' if ext['changed'] else 'main'
    assert rule(5.8, 5.85, 9.0, 8.5) == 'burst' and rule(5.8, 5.9, 9.0, 8.5) == 'ext'
    assert rule(6.0, 6.0, 9.0, 9.0, False, False) == 'main' and rule(6.0, 5.9, 9.0, 8.0, False, True) == 'burst'
    saved = {v: V.selection(base, v, seed) for v in V.VARIANTS}
    for variant, same in (('ext', True), ('burst', True)):
        s_ = dict(saved[variant])
        s_['ProposedLower'] = dict(s_['ProposedLower'], selected=3, changed=False, dev_cost=6.0)
        C.write_json(C.seed_dir(V.variant_root(base, variant), seed)/'v26_selection.json', s_)
    (V.burstdev_dir(base, 'ext', seed)/'same_as_main.json').write_text('{}')
    (V.burstdev_dir(base, 'burst', seed)/'same_as_main.json').write_text('{}')
    assert V.decide(base, seeds, smoke=True)['adopted'] == 'main'
    for variant in V.VARIANTS:
        C.write_json(C.seed_dir(V.variant_root(base, variant), seed)/'v26_selection.json', saved[variant])
        (V.burstdev_dir(base, variant, seed)/'same_as_main.json').unlink()
    decision = V.decide(base, seeds, smoke=True)
    assert decision['adopted'] == 'ext'

    summary = R.build(base, main, seeds, smoke=True)
    text = summary.read_text(encoding='utf-8')
    assert 'v2.6' in text and 'NoStage3' in text and '零样本扫参' in text, text
    out = base/'results'
    d = V._ORIGINAL_DESIGN(smoke=True)
    assert len(C.read_json(out/'v26_records_test.json')) == len(d['test'])
    assert len(C.read_json(out/'v26_records_burst.json')) == len(d['sweep_scenarios'])
    assert len(C.read_json(out/'v26_records_zeroshot.json')) == len(points)*len(d['test'])
    assert len(C.read_json(out/'v26_records_ablation.json')) == len(d['test'])
    assert len(C.read_json(out/'v26_records_burstdev.json')) == 3*len(d['dev'])
    assert len(C.read_json(out/'v26_train_logs.json')) == 2*2*3
    assert tree_hash(main) == snapshot, 'the main root was modified'
    print(text)


def test_runner():
    tmp = Path(tempfile.mkdtemp(prefix='p2v26run_'))
    fake = tmp/'fake.py'
    fake.write_text('import sys\nprint(" ".join(sys.argv[1:]))\nsys.exit(1 if "--seed=13" in sys.argv else 0)\n')
    real, V.RUN = V.RUN, fake
    try:
        jobs = [V.Job('a', ['train', '--seed=1']), V.Job('b', ['dev', '--seed=1'], ['a']),
                V.Job('c', ['train', '--seed=13']), V.Job('d', ['dev', '--seed=13'], ['c'])]
        polls = []
        failed = V.run_jobs(jobs, tmp/'out', tmp/'main', workers=2, smoke=True, poll=polls.append, poll_seconds=0)
        assert failed == ['c', 'd'], failed
        assert (tmp/'out'/'_jobs'/'b.done').exists() and not (tmp/'out'/'_jobs'/'c.done').exists()
        log = (tmp/'out'/'_jobs'/'a.log').read_text()
        assert f'--main={tmp/"main"}' in log and '--smoke' in log, log
        fake.write_text('import sys\nsys.exit(0)\n')
        assert V.run_jobs(jobs, tmp/'out', tmp/'main', workers=2, smoke=True) == []
        assert polls and polls[-1]['total'] == 4
    finally:
        V.RUN = real
    print('runner OK')


FAKE_RUN = r"""
import importlib, sys, types
sys.path.insert(0, {src!r}); sys.path.insert(0, {repo!r})
try:
    importlib.import_module('torch')
except ImportError:
    stub = types.ModuleType('hrl_v2.hybrid_simplex')
    stub.SUPPORTS = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2))
    sys.modules['hrl_v2.hybrid_simplex'] = stub
from p2 import common as C, evaluate
from v26 import core as V
import run_v26
arg = dict(a[2:].split('=', 1) for a in sys.argv[2:] if '=' in a)
command, s = sys.argv[1], V.sizes(True)
if command == 'train':
    rd = C.run_dir(V.variant_root(arg['out-base'], arg['variant']), int(arg['seed']), arg['stage'])
    (rd/'ckpt').mkdir(exist_ok=True)
    for n in range(s['base']+1, s['total']+1):
        C.append_jsonl(rd/'train.jsonl', dict(stage=arg['stage'], seed=int(arg['seed']), episode=n, cost_per_s=6.5,
                                              actor_relative_entropy=-2.5))
        if n in s['new']:
            (rd/'ckpt'/('ep%04d.pt' % n)).write_bytes(b'ckpt')
elif command == 'dev':
    c = int(arg['candidate'])
    score = 5.0 + c/10 + (0.0 if arg['variant'] == 'ext' else 0.01)
    C.write_json(C.seed_dir(V.variant_root(arg['out-base'], arg['variant']), int(arg['seed']))/'dev'/arg['label']/str(c)/
                 'summary.json', {{C.SELECTION_METRIC: score}})
else:
    V.policy_paths = lambda *a, **k: ('CA', None, None)
    evaluate.candidate_paths = lambda *a, **k: ('CA', None, None)
    sys.exit(run_v26.main(sys.argv[1:]))
"""


def test_go():
    """The whole `go` command on a fake main root, with a stand-in for the jobs that need networks."""
    from v26 import stage
    tmp = Path(tempfile.mkdtemp(prefix='p2v26go_'))
    main, base = tmp/'main', tmp/'out'
    seed = V._ORIGINAL_DESIGN(smoke=True)['seeds'][0]
    fake_main(main, seed, dict(ProposedLower={0: 6.2, 2: 6.4, 3: 6.0}, HoverSAC={0: 10.3, 2: 10.2, 3: 10.4}))
    snapshot = tree_hash(main)
    fake = tmp/'fake_run.py'
    fake.write_text(FAKE_RUN.format(src=str(REPO/'src'), repo=str(REPO)), encoding='utf-8')
    real, V.RUN = V.RUN, fake
    try:
        code = stage.Go(main, base, 3, (seed,), smoke=True, publish=False).run()
        assert code == 0, (base/V.STATUS_PAGE).read_text(encoding='utf-8')
        page = (base/V.STATUS_PAGE).read_text(encoding='utf-8')
        assert '完成：采用 ext' in page and 'v2.6 的选择' in page, page
        state = C.read_json(base/'_v26'/'state.json')
        assert all(v['status'] in ('passed', 'skipped') for v in state['steps'].values()), state
        decision = C.read_json(base/'results'/'v26_decision.json')
        assert decision['adopted'] == 'ext' and decision['table']['ext']['per_seed'][str(seed)]['selected'] == 5
        assert (base/'results'/'v26_summary.md').exists() and len(C.read_json(base/'results'/'v26_records_test.json')) == 4
        done = len(list((base/'_jobs').glob('*.done')))
        assert stage.Go(main, base, 3, (seed,), smoke=True, publish=False).run() == 0      # resume: nothing to redo
        assert len(list((base/'_jobs').glob('*.done'))) == done
        names = [rel for _, rel, _ in stage.V26Publisher(base, remote='none').files()]
        assert V.STATUS_PAGE in names and 'DECISION.md' in names and 'results/v26_summary.md' in names, names
        # a failing job stops the run and is named on the status page
        (base/'_jobs'/f'ext_s{seed}_test_Proposed.done').unlink()
        fake.write_text('import sys\nsys.exit(1 if sys.argv[1] == "test" else 0)\n')
        assert stage.Go(main, base, 3, (seed,), smoke=True, publish=False).run() == 1
        page = (base/V.STATUS_PAGE).read_text(encoding='utf-8')
        assert '已停止' in page and f'ext_s{seed}_test_Proposed' in page, page
    finally:
        V.RUN = real
    assert tree_hash(main) == snapshot, 'the main root was modified'
    print('go OK:', done, 'jobs')


def main():
    test_design_and_plans()
    test_burst_plan()
    test_training_adapter()
    test_runner()
    test_flow()
    test_go()
    print('OK')


if __name__ == '__main__':
    main()
