"""Frozen dev evaluation of a single synchronous PPO/SAC checkpoint pair."""
from j_common import *
import multiprocessing
import psutil
import torch
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from hrl_v2.checkpoint import identity, load, json_digest
from hrl_v2.runtime import Learner, configure, parameter_hash
from hrl_v2.upper_baselines import fixed_upper_policy
from hrl_v2.policy import Policy
from hrl_v2.collector import collect_metrics
from hrl_v2.physics_adapter import PhysicsAdapter
from hrl_v2.randomness import Streams
from hrl_v2.metrics import PooledMetrics
from hrl_v2.run import read_config
from hrl_v2.evaluate import gate_manifest
from hrl_v2.timing import ClockConfig

REF = WT / 'outputs/lite_v2_4_0/baselines/summary.json'
CANDIDATES = tuple(range(220, 401, 20))
_CACHE = {}
_REF_CACHE = None


def put_new(path, value):
    """Exclusive creation: never replace a complete or partial result."""
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def whole_metrics_equal(left, right, label):
    equal(left, right, label)
    # Also distinguish scalar encodings such as -0.0 from +0.0.
    equal(json_digest(left), json_digest(right), label + '_serialized_bits')


def memory_guard():
    rss = psutil.Process().memory_info().rss
    guard_resources()  # Registered global available RAM condition; record per-process RSS only.
    return rss


def distribution(values):
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    check(a.size > 0 and np.isfinite(a).all(), 'finite_motion_samples')
    return dict(count=int(a.size), mean_m=float(a.mean()),
                p50_m=float(np.percentile(a, 50)), p90_m=float(np.percentile(a, 90)),
                p95_m=float(np.percentile(a, 95)), p99_m=float(np.percentile(a, 99)),
                max_m=float(a.max()), stationary_fraction=float(np.mean(a <= 1e-9)))


class MotionProbe:
    """Read executed coordinates after each physical tick; never call policy/report."""
    def __init__(self, adapter, expected_seconds=900):
        self.start = np.array(adapter.env.physics.uav_coords, dtype=np.float64, copy=True)
        self.previous = self.start.copy()
        self.interval = np.zeros(len(self.start), dtype=np.float64)
        self.total = np.zeros(len(self.start), dtype=np.float64)
        self.period_ticks = round(5. / adapter.env.dt)
        check(self.period_ticks * adapter.env.dt == 5., 'motion_sample_grid_5s')
        self.samples = []
        self.expected_intervals = int(expected_seconds // 5)

    def __call__(self, adapter, reading):
        current = np.array(adapter.env.physics.uav_coords, dtype=np.float64, copy=True)
        executed = np.linalg.norm(current - self.previous, axis=1)
        self.interval += executed
        self.total += executed
        self.previous = current
        if adapter.env.step_index % self.period_ticks == 0:
            self.samples.append(dict(time_s=float(adapter.env.time_s),
                executed_distance_m=self.interval.tolist(),
                distance_from_start_m=np.linalg.norm(current - self.start, axis=1).tolist()))
            self.interval.fill(0.)
            memory_guard()

    def result(self):
        check(len(self.samples) == self.expected_intervals, 'motion_expected_five_second_intervals')
        return dict(interval_seconds=5, stationary_threshold_m=1e-9,
            definition='executed path length summed over physical ticks in each 5s interval; distance from reset coordinates at interval end',
            initial_positions_m=self.start.tolist(), final_positions_m=self.previous.tolist(),
            executed_total_m_per_uav=self.total.tolist(),
            executed_5s=distribution([s['executed_distance_m'] for s in self.samples]),
            distance_from_start=distribution([s['distance_from_start_m'] for s in self.samples]),
            samples_5s=self.samples)


def frozen_reference():
    global _REF_CACHE
    if _REF_CACHE is not None:
        equal(sha(REF), _REF_CACHE['summary_sha256'], 'frozen_baseline_summary_unchanged')
        return _REF_CACHE
    ref = read(REF)
    manifest = gate_manifest(20)
    equal(ref['manifest'], manifest, 'frozen_reference_dev20_manifest')
    windows = {}
    for sc in manifest['scenarios']:
        for stream in (1, 2):
            path = REF.parent / f'hover_equal_{sc["scenario"]:02d}_{stream}.json'
            r = read(path)
            equal(r['identity']['scenario'], sc, 'reference_scenario')
            equal(r['identity']['stream'], stream, 'reference_stream')
            equal(r['metrics']['seconds'], 900, 'reference_900s')
            keys = ('event_sha256', 'canonical_event_sha256', 'realization_sha256')
            hashes = {k:r['metrics'][k] for k in keys if k in r['metrics']}
            check('event_sha256' in hashes, 'reference_event_hash_present')
            windows[f'{sc["scenario"]:02d}_{stream}'] = dict(path=str(path), sha256=sha(path), hashes=hashes)
    _REF_CACHE = dict(summary_path=str(REF), summary_sha256=sha(REF), windows=windows)
    return _REF_CACHE


def _initialize_worker():
    set_below_normal()
    assert_imports()
    memory_guard()


def _parameters(learner):
    return {f'{owner}.{name}':parameter_hash(getattr(getattr(learner, owner), name))
            for owner, names in (('sac', ('actor','q1','q2','target1','target2')),
                                 ('ppo', ('actor','value'))) for name in names}


def window(task):
    started = time.perf_counter()
    c = contract()
    binding = c['identity']
    assert_imports()
    equal(identity(WT, CONFIG), binding, 'worker_source_identity_before')
    check(task['seed'] in c['seed_order'] and task['split'] == 'dev', 'authorized_seed_dev_only')
    check(task['episode'] in CANDIDATES and task['stream'] in (1, 2), 'authorized_checkpoint_stream')
    check(1 <= task['scenario']['scenario'] <= 20, 'dev_scenarios_1_to_20')
    config = read_config(CONFIG, WT)
    device = configure()
    check(torch.get_num_threads() == 10, 'evaluation_original_ten_threads')
    timing = ClockConfig.from_config(config)
    check(timing.rollout_seconds == 900, 'evaluation_900s')
    memory_guard()
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        checkpoint = task['checkpoint']
        if checkpoint not in _CACHE:
            _CACHE.clear()
            gc.collect()
            torch.cuda.empty_cache()
            saved = load(checkpoint, binding)
            equal(saved['seed'], task['seed'], 'paired_checkpoint_seed')
            equal(saved['completed_episode'], task['episode'], 'paired_checkpoint_episode')
            equal(saved['kind'],'joint_sync_pair_nonresumable','paired_checkpoint_type')
            equal(saved['protocol_contract_sha256'],sha(CPIL/'contract.json'),'paired_checkpoint_protocol')
            equal(saved['resumable'],False,'paired_nonresumable_checkpoint')
            learner = Learner(task['seed'], device, config=config)
            learner.load_state_dict(saved['learner'])
            del saved
            gc.collect()
            _CACHE[checkpoint] = learner
        learner = _CACHE[checkpoint]
        before = _parameters(learner)
        counts_before = {owner:dict(counts=dict(getattr(learner, owner).counts),
                                   version=getattr(learner, owner).version) for owner in ('sac','ppo')}
        sc = task['scenario']
        scene = read(SCENE)
        adapter = PhysicsAdapter(scene, timing=timing)
        adapter.reset(**{k:v for k,v in sc.items() if k.endswith('_seed')})
        streams = Streams(task['seed'], episode=sc['scenario'], split='dev', policy_stream=task['stream'])
        if task['route'] == 'cvx':
            policy = fixed_upper_policy(learner.ppo, scene, streams, 'cvx', stochastic=False, mode='moving')
        else:
            policy = Policy(learner.sac.actor, learner.ppo.actor, None, streams, mode='moving',
                            stochastic=False, upper_stochastic=False, deployment=scene['deployment'])
        probe = MotionProbe(adapter)
        metrics = collect_metrics(adapter, policy, identity=task['identity'], on_tick=probe,
                                  reward_scale=config['objective']['reward_scale'])
        after = _parameters(learner)
        equal(before, after, 'eval_all_parameters_unchanged')
        equal(counts_before, {owner:dict(counts=dict(getattr(learner, owner).counts),
               version=getattr(learner, owner).version) for owner in ('sac','ppo')}, 'eval_update_counts_unchanged')
        equal(metrics['seconds'], 900, 'actual_evaluation_900s')
        hashes = {k:metrics[k] for k in task['reference']['hashes']}
        equal(hashes, task['reference']['hashes'], 'paired_frozen_external_hashes')
        diagnostics = {}
        if task['route'] == 'cvx':
            records = policy.lower_override.rule.records
            diagnostics = dict(solver_calls=len(records), fallback_calls=sum(r.get('fallback',False) for r in records))
        equal(identity(WT, CONFIG), binding, 'worker_source_identity_after')
        put_new(task['output'], normalize(dict(identity=task['identity'], metrics=metrics,
            diagnostics=diagnostics, moving_distance=probe.result(), learning_updates=0, replay_writes=0,
            parameter_hash_check=dict(before=before, after=after, passed=True),
            reference_external_hash_check=dict(passed=True, reference=task['reference']),
            memory_rss_bytes=memory_guard(), wall_seconds=time.perf_counter()-started)))
    return dict(scenario=sc['scenario'], stream=task['stream'], path=task['output'])


def _pool(tasks):
    pool = ProcessPoolExecutor(max_workers=6, initializer=_initialize_worker,
                               mp_context=multiprocessing.get_context('spawn'))
    futures = set()
    try:
        futures = {pool.submit(window, task) for task in tasks}
        while futures:
            done, futures = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
            for process in list(pool._processes.values()):
                if process.is_alive():
                    try:
                        rss = psutil.Process(process.pid).memory_info().rss
                    except psutil.NoSuchProcess:
                        continue
                    guard_resources()  # Available RAM guard; per-worker RSS is recorded, not a new abort threshold.
            for future in done:
                future.result()
    except BaseException:
        for future in futures:
            future.cancel()
        for process in list(pool._processes.values()):
            if process.is_alive():
                process.terminate()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown(wait=True)


def select_candidate(candidates):
    values = list(candidates.values()) if isinstance(candidates, dict) else list(candidates)
    check(sorted(int(r['episode']) for r in values) == list(CANDIDATES), 'exactly_ten_dev_candidates')
    check(all(r.get('route') is None for r in values), 'selection_original_pair_only')
    return min(values, key=lambda r:(r['mean_s'], int(r['episode'])))


def group(seed, episode, checkpoint, route=None):
    """Evaluate only one simultaneous pair; cvx is restricted to its selected upper."""
    started = time.perf_counter()
    c = contract()
    assert_imports()
    equal(identity(WT, CONFIG), c['identity'], 'group_source_identity_before')
    check(seed in c['seed_order'], 'group_authorized_seed')
    check(episode in CANDIDATES and route in (None, 'cvx'), 'group_authorized_candidate_route')
    checkpoint = Path(checkpoint).resolve()
    check(checkpoint.is_relative_to((CPIL/f'seed_{seed}').resolve()), 'new_pair_checkpoint_inside_seed')
    sidecar = read(checkpoint.with_suffix('.json'))
    equal(sidecar['completed_episode'], episode, 'checkpoint_sidecar_episode')
    equal(sidecar['identity'], c['identity'], 'checkpoint_sidecar_identity')
    checkpoint_sha = sha(checkpoint)
    equal(sidecar['sha256'], checkpoint_sha, 'checkpoint_sidecar_sha')
    if route == 'cvx':
        selected = select_candidate([read(CPIL/f'seed_{seed}/evaluations/episode-{n:04d}/summary.json') for n in CANDIDATES])
        equal(selected['episode'], episode, 'cvx_selected_episode_only')
        equal(selected['checkpoint_sha256'], checkpoint_sha, 'cvx_selected_pair_only')
    root = CPIL / f'seed_{seed}/evaluations/episode-{episode:04d}'
    if route == 'cvx':
        root = root/'cvx'
    root = output_path(root)
    check(not root.exists(), 'new_evaluation_directory', str(root))
    root.mkdir(parents=True)
    reference = frozen_reference()
    manifest = gate_manifest(20)
    tasks = []
    for sc in manifest['scenarios']:
        for stream in (1, 2):
            key = f'{sc["scenario"]:02d}_{stream}'
            ident = dict(protocol='v2_4_0_joint_sync_pilot_dev', seed=seed, episode=episode,
                manifest=json_digest(manifest), scenario=sc, stream=stream, split='dev',
                upper_policy='deterministic', lower_policy='deterministic', route=route,
                rollout_seconds=900, reward_scale=100, checkpoint_pair_sha256=checkpoint_sha)
            tasks.append(dict(seed=seed, episode=episode, checkpoint=str(checkpoint), route=route,
                scenario=sc, stream=stream, split='dev', identity=ident, reference=reference['windows'][key],
                output=str(root/f'windows/{key}.json')))
    _pool(tasks)
    pooled = PooledMetrics()
    per, files, events, motion_per = {}, [], {}, {}
    executed, displacement, totals = [], [], []
    for sid in range(1, 21):
        one = PooledMetrics()
        previous, previous_motion = None, None
        for stream in (1, 2):
            path = root/f'windows/{sid:02d}_{stream}.json'
            r = read(path)
            if previous is not None:
                whole_metrics_equal(r['metrics'], previous, f'whole_metrics_streams_{seed}_{episode}_{sid}')
                whole_metrics_equal(r['moving_distance'], previous_motion, f'motion_streams_{seed}_{episode}_{sid}')
            previous, previous_motion = r['metrics'], r['moving_distance']
            one.add(r['metrics'])
            pooled.add(r['metrics'])
            events[str(sid)] = r['metrics']['event_sha256']
            files.append(dict(scenario=sid, stream=stream, path=str(path), sha256=sha(path)))
            for sample in r['moving_distance']['samples_5s']:
                executed.extend(sample['executed_distance_m'])
                displacement.extend(sample['distance_from_start_m'])
            totals.extend(r['moving_distance']['executed_total_m_per_uav'])
        per[str(sid)] = one.result()
        motion_per[str(sid)] = {k:v for k,v in previous_motion.items() if k != 'samples_5s'}
    total = pooled.result()
    equal(sha(checkpoint), checkpoint_sha, 'evaluation_checkpoint_file_unchanged')
    equal(sha(REF), reference['summary_sha256'], 'reference_summary_unchanged_after')
    equal(identity(WT, CONFIG), c['identity'], 'group_source_identity_after')
    summary = dict(seed=seed, episode=episode, group=f'episode-{episode:04d}', route=route,
        mean_s=avg([r['truncated_delay_s'] for r in per.values()]), per_scenario=per, pooled=total,
        unfinished_per_window=total['unfinished_parents']/total['episodes'],
        unfinished_high_per_window=total['priority'][0]['unfinished']/total['episodes'],
        unfinished_low_per_window=total['priority'][1]['unfinished']/total['episodes'],
        energy_j_per_window={k:v/total['episodes'] for k,v in total['energy_j'].items()},
        cost_per_s=total['cost_per_s'], stage_delay_s=total['stage_delay_s'],
        checkpoint_path=str(checkpoint), checkpoint_sha256=checkpoint_sha,
        stream_consistency=dict(passed=True, scenarios=20, comparison='entire metrics including serialized scalar bits'),
        external_events=events, external_reference=dict(path=str(REF), sha256=reference['summary_sha256']),
        moving_distance=dict(executed_5s=distribution(executed), distance_from_start=distribution(displacement),
            total_per_uav_window=distribution(totals), per_scenario=motion_per),
        window_files=files, learning_updates=0, replay_writes=0,
        parameter_hash_check_passed=True, source_identity_unchanged=True,
        summary_path=str(root/'summary.json'), wall_seconds=time.perf_counter()-started)
    put_new(root/'summary.json', summary)
    print('DEV_GROUP_PASS', seed, episode, route, summary['mean_s'], summary['wall_seconds'], flush=True)
    return summary


def technical_smoke(checkpoint, seed=4104, seconds=10, output=None):
    """Three short technical rollouts only; never a candidate, baseline, or gate."""
    from dataclasses import replace
    c = contract()
    assert_imports()
    set_below_normal()
    check(seed in c['seed_order'] and seconds in (5, 10), 'technical_smoke_scope')
    checkpoint = Path(checkpoint).resolve()
    check(checkpoint.is_relative_to(CPIL.resolve()), 'technical_smoke_new_checkpoint_only')
    before_file = sha(checkpoint)
    binding = c['identity']
    equal(identity(WT,CONFIG),binding,'technical_source_before')
    config = read_config(CONFIG,WT)
    scene = read(SCENE)
    device = configure()
    check(torch.get_num_threads()==10,'technical_original_threads')
    timing = replace(ClockConfig.from_config(config),rollout_seconds=seconds)
    scenario = gate_manifest(20)['scenarios'][0]
    path = output_path(output or CPIL/'technical_eval_smoke.json')
    check(not path.exists(),'technical_smoke_new_result',str(path))
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        saved = load(checkpoint,binding)
        equal(saved['seed'],seed,'technical_checkpoint_seed')
        learner = Learner(seed,device,config=config)
        learner.load_state_dict(saved['learner'])
        del saved
        gc.collect()
        before = _parameters(learner)
        records = []
        for stream,hook in ((1,True),(2,True),(1,False)):
            adapter = PhysicsAdapter(scene,timing=timing)
            adapter.reset(**{k:v for k,v in scenario.items() if k.endswith('_seed')})
            streams = Streams(seed,episode=scenario['scenario'],split='dev',policy_stream=stream)
            policy = Policy(learner.sac.actor,learner.ppo.actor,None,streams,mode='moving',
                stochastic=False,upper_stochastic=False,deployment=scene['deployment'])
            probe = MotionProbe(adapter,expected_seconds=seconds) if hook else None
            item = dict(purpose='technical_consistency_only_not_model_selection',
                        scenario=scenario,stream=stream,seconds=seconds)
            metrics = collect_metrics(adapter,policy,identity=item,on_tick=probe,
                                      reward_scale=config['objective']['reward_scale'])
            records.append(dict(stream=stream,hook=hook,metrics=metrics,
                                motion=probe.result() if probe else None))
        whole_metrics_equal(records[0]['metrics'],records[1]['metrics'],'technical_two_streams_complete_metrics')
        whole_metrics_equal(records[0]['metrics'],records[2]['metrics'],'technical_motion_hook_preserves_complete_metrics')
        whole_metrics_equal(records[0]['motion'],records[1]['motion'],'technical_two_streams_motion')
        equal(records[0]['metrics']['event_sha256'],records[1]['metrics']['event_sha256'],'technical_two_streams_external_events')
        after = _parameters(learner)
        equal(before,after,'technical_parameters_unchanged')
    equal(sha(checkpoint),before_file,'technical_checkpoint_unchanged')
    equal(identity(WT,CONFIG),binding,'technical_source_after')
    result = dict(status='passed',purpose='technical_consistency_only_not_model_selection',
        seed=seed,scenario=1,rollout_seconds=seconds,rollouts=3,checkpoint_path=str(checkpoint),
        checkpoint_sha256=before_file,deterministic_streams_exact=True,
        motion_hook_preserves_whole_metrics=True,event_hashes_equal=True,
        event_sha256=records[0]['metrics']['event_sha256'],
        whole_metrics_sha256=json_digest(records[0]['metrics']),
        movement=records[0]['motion'],parameter_hash_check=dict(before=before,after=after,passed=True),
        learning_updates=0,replay_writes=0,compared_to_frozen_900s_baselines=False,
        eligible_for_selection=False,source_identity_unchanged=True)
    put_new(path,result)
    return result
