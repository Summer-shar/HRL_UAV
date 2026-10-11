"""Output-local five-episode synchronous joint pilot; no source modifications.

Production entry is run_seed.  run_joint_batch exposes the same collection and
update path for an explicitly marked short technical smoke, never a resume path.
"""
from j_common import *
import copy
from dataclasses import replace
import torch
from hrl_v2 import SCHEMA
from hrl_v2.checkpoint import identity, load, save, runtime_record, json_digest
from hrl_v2.runtime import Learner, configure, parameter_hash
from hrl_v2.run import read_config, append, collection_policy
from hrl_v2.randomness import Streams, derive
from hrl_v2.schedule import phase
from hrl_v2.physics_adapter import PhysicsAdapter
from hrl_v2.collector import collect
from hrl_v2.policy import Policy
from hrl_v2.telemetry import compact, resources
from hrl_v2.replay import Replay
from hrl_v2.timing import ClockConfig

EXPECTED200 = dict(triggers=18000, skipped=90, events=17910,
                   critic_fits=35820, actor_updates=8910, target_updates=17910)
EXPECTED_FINAL = dict(triggers=36000, skipped=90, events=35910,
                     critic_fits=71820, actor_updates=17010, target_updates=35910)
EXPECTED_JOINT = {k: EXPECTED_FINAL[k] - EXPECTED200[k] for k in EXPECTED_FINAL}
FROZEN_ORDINALS = frozenset((1, 2)) | frozenset(
    n for start in range(301, 400, 10) for n in (start, start + 1))
CANDIDATES = tuple(range(220, 401, 20))
OLD_LOWER_PHASES = tuple(phase(n) for n in range(401, 601) if phase(n).layer == 'lower')
PROTOCOL_TYPE = 'five_episode_synchronous_joint_D_mixed_base_plus_two_SAC_only'
NETWORKS = (('sac', ('actor', 'q1', 'q2', 'target1', 'target2')),
            ('ppo', ('actor', 'value')))


def source_phase(n):
    check(isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= 400,
          'valid_SAC_ordinal', n)
    return phase(n) if n <= 300 else OLD_LOWER_PHASES[n - 301]


def ordinal_plan(n):
    p = source_phase(n)
    c = contract()
    alpha = next(float(row['alpha']) for row in c['alpha_schedule'] if row['episodes'][0] <= n <= row['episodes'][1])
    equal(sorted(FROZEN_ORDINALS), c['actor_frozen_ordinals'], 'JC_actor_frozen_table')
    equal(p.alpha, alpha, 'source_alpha_mapping')
    equal(p.actor_enabled, n not in FROZEN_ORDINALS, 'source_actor_freeze_mapping')
    return dict(global_episode=n, sac_ordinal=n, layer='lower' if n <= 200 else 'both',
                alpha=alpha, actor_enabled=n not in FROZEN_ORDINALS,
                source_phase=dict(p.__dict__))


def budget_counts(last=400):
    counts = {key: 0 for key in EXPECTED_FINAL}
    for n in range(1, last + 1):
        actor_enabled = ordinal_plan(n)['actor_enabled']
        for _ in range(90):
            counts['triggers'] += 1
            if n == 1:
                counts['skipped'] += 1
                continue
            counts['events'] += 1
            counts['critic_fits'] += 2
            counts['target_updates'] += 1
            if actor_enabled and counts['events'] % 2 == 0:
                counts['actor_updates'] += 1
    return counts


def binding():
    return contract()['identity']


def source_guard():
    if contract().get('lean_runtime',{}).get('enabled'):return
    equal(identity(WT, CONFIG), binding(), 'training_source_config_scene_identity')


def rng_state():
    return dict(cpu=torch.get_rng_state().clone(),
                cuda=[x.clone() for x in torch.cuda.get_rng_state_all()])


def rng_assert(before, label):
    equal(torch.get_rng_state(), before['cpu'], 'RNG_CPU_unchanged_' + label)
    equal(torch.cuda.get_rng_state_all(), before['cuda'], 'RNG_CUDA_unchanged_' + label)


def hashes(learner):
    return {f'{family}_{name}': parameter_hash(getattr(getattr(learner, family), name))
            for family, names in NETWORKS for name in names}


def replay_summary(replay):
    return dict(upper_version=replay.upper_version, writes=replay.writes,
                flushes=replay.flushes,
                episodes=[dict(global_episode=e.identity['global_episode'],
                               upper_version=e.upper_version, lower_version=e.lower_version)
                          for e in replay.episodes])


def delta(after, before):
    return {key: value - before[key] for key, value in after.items()}


def clone_streams(streams, state=None):
    state = copy.deepcopy(streams.state_dict() if state is None else state)
    cloned = Streams(**state['identity'], policy_stream=state['policy_stream'])
    cloned.load_state_dict(state)
    return cloned


def external_seeds(seed, n):
    p = source_phase(n)
    return {key + '_seed': derive(seed, p.stage, p.cycle, p.block_episode, purpose=key)
            for key in ('arrival', 'layout', 'load')}


class Coverage:
    def __init__(self, adapter):
        self.start = adapter.env.physics.uav_coords.copy()
        self.centers = np.asarray(adapter.env._scene.realization['cluster_centers_m'], dtype=float)
        self.samples = [self.start.copy()]

    def tick(self, adapter, reading):
        if adapter.env.step_index % adapter.timing.physical_per_upper == 0:
            self.samples.append(adapter.env.physics.uav_coords.copy())
            guard_resources()

    def result(self):
        x = np.asarray(self.samples)
        return dict(samples=len(x), distance_start=np.linalg.norm(x-self.start, axis=-1).tolist(),
                    distance_cluster_center=np.linalg.norm(x-self.centers, axis=-1).tolist(),
                    pair_distance=np.stack([np.linalg.norm(x[:, i]-x[:, j], axis=-1)
                                            for i, j in ((0, 1), (0, 2), (1, 2))], axis=-1).tolist(),
                    definitions=dict(position='initial and every completed physical macro boundary; no RNG draws',
                                     cluster_center='causal initial realization cluster center for associated region',
                                     pair_order=[[0, 1], [0, 2], [1, 2]]))


def validate_behavior_episode(episode, upper_version, lower_version):
    check(episode.complete and bool(episode.high), 'joint_complete_episode')
    equal(episode.upper_version, upper_version, 'joint_true_upper_version')
    equal(episode.lower_version, lower_version, 'joint_true_lower_version')
    required = ('pre_tanh', 'mean', 'logstd', 'logprob', 'value', 'next_value',
                'reward_learning', 'terminated', 'truncated')
    for high in episode.high:
        check(all(key in high for key in required), 'joint_complete_high_fields')
        check(all(np.isfinite(np.asarray(high[key], dtype=float)).all()
                  for key in required[:7]), 'joint_finite_high_fields')
    check(all(not h['terminated'] for h in episode.high)
          and sum(bool(h['truncated']) for h in episode.high) == 1
          and bool(episode.high[-1]['truncated']), 'joint_high_boundary')


def add_preserving_versions(replay, episode, upper_version, lower_version):
    """Only move the container label; original episode labels remain immutable."""
    validate_behavior_episode(episode, upper_version, lower_version)
    before = [(id(e), e.upper_version, e.lower_version) for e in replay.episodes]
    flushes = replay.flushes
    replay.upper_version = upper_version
    replay.add(episode)  # original Replay.add and original Replay.sample are unchanged
    equal(replay.flushes, flushes, 'joint_no_replay_flush')
    expected = (before + [(id(episode), upper_version, lower_version)])[-55:]
    equal([(id(e), e.upper_version, e.lower_version) for e in replay.episodes],
          expected, 'joint_preserved_true_episode_versions')


def run_joint_batch(learner, replay, seed, numbers, config, scene, clocks=None, *,
                    technical_only=False, events_per_ordinal=90,
                    on_collected=None, on_learning=None, on_ordinal=None):
    """Collect five frozen-policy episodes, PPO once, then five SAC ordinals.

    The technical flag permits short windows/events for an isolated smoke only.
    run_seed never accepts or forwards such an override.  Each context owns an
    independent post-collection Streams instance and saved state; PPO receives
    a state-identical clone of the last episode's stream.
    """
    numbers = list(numbers)
    check(len(numbers) == 5 and numbers == list(range(numbers[0], numbers[0]+5))
          and 201 <= numbers[0] <= 396 and (numbers[0]-201) % 5 == 0,
          'joint_five_consecutive_ordinals', numbers)
    clocks = ClockConfig.from_config(config) if clocks is None else clocks
    if not technical_only:
        equal(clocks, ClockConfig.from_config(read_config(CONFIG, WT)), 'joint_full_production_clocks')
        equal(config, read_config(CONFIG, WT), 'joint_full_production_config')
        equal(events_per_ordinal, 90, 'joint_full_production_events')
        equal(clocks.lower_steps // 20, 90, 'joint_production_collector_trigger_budget')
    check(isinstance(events_per_ordinal, int) and events_per_ordinal > 0,
          'joint_positive_technical_event_budget')
    guard_resources()
    frozen = hashes(learner)
    counts_before = dict(learner.sac.counts)
    ppo_before = dict(learner.ppo.counts)
    upper = f'ppo-{learner.ppo.version}-stochastic'
    lower = learner.sac.version
    flush_before = replay.flushes
    contexts = []
    for n in numbers:
        before_rng = rng_state()
        p = source_phase(n)
        streams = Streams(seed, p.stage, p.cycle, p.block_episode)
        external = external_seeds(seed, n)
        adapter = PhysicsAdapter(scene, timing=clocks)
        adapter.reset(**external)
        policy = Policy(learner.sac.actor, learner.ppo.actor, learner.ppo.value,
                        streams, mode='moving', stochastic=True, upper_stochastic=True)
        coverage = Coverage(adapter)
        rng_assert(before_rng, 'joint_pre_collection')
        started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        episode = collect(adapter, policy, upper_version=upper, lower_version=lower,
                          identity=dict(global_episode=n, sac_ordinal=n, external=external,
                                        source_phase=dict(p.__dict__), behavior_source='ppo'),
                          on_event=None, on_tick=coverage.tick,
                          reward_scale=config['objective']['reward_scale'],
                          gamma_lower=config['objective']['gamma_micro'])
        audit_rng = rng_state()
        validate_behavior_episode(episode, upper, lower)
        equal(hashes(learner), frozen, 'joint_frozen_collection_networks')
        equal(learner.sac.counts, counts_before, 'joint_no_SAC_during_collection')
        equal(learner.ppo.counts, ppo_before, 'joint_no_PPO_during_collection')
        equal(learner.sac.version, lower, 'joint_live_lower_fixed')
        equal(f'ppo-{learner.ppo.version}-stochastic', upper, 'joint_live_upper_fixed')
        available = len(replay.episodes)
        episode.identity['lower_versions_by_transition'] = [lower] * len(episode.rewards)
        add_preserving_versions(replay, episode, upper, lower)
        maximum = float(np.linalg.norm(np.asarray([h['displacement'] for h in episode.high]), axis=-1).max())
        check(maximum <= 20 + 1e-6, 'joint_motion_cap', maximum)
        context = dict(ordinal=n, plan=ordinal_plan(n), episode=episode, streams=streams,
                       collection_stream_state=copy.deepcopy(streams.state_dict()),
                       external=external, coverage=coverage.result(), max_actual_step_m=maximum,
                       collection_hashes=frozen, replay_completed_before=available,
                       replay_completed_after=len(replay.episodes),
                       collection_seconds=time.perf_counter()-started,
                       actor_relative_entropy=policy.entropy_sum/policy.entropy_count if policy.entropy_count else None,
                       entropy_samples=policy.entropy_count)
        contexts.append(context)
        if on_collected is not None:
            on_collected(context)
        rng_assert(audit_rng, 'joint_collection_audits')
        del adapter, policy, coverage
    from d_train import collect_extras, replay_mix
    extras = collect_extras(learner, replay, seed, numbers, config, scene, clocks, frozen, lower, on_collected)
    check(len({id(c['streams']) for c in contexts}) == 5, 'joint_five_independent_stream_instances')
    collection_hashes_after = hashes(learner)
    equal(collection_hashes_after, frozen, 'joint_hashes_before_PPO')
    batch = [c['episode'] for c in contexts]
    equal([e.identity['global_episode'] for e in batch], numbers, 'PPO_exact_five_normal_episode_IDs')
    check(not any(e.identity.get('extra_only_SAC', False) for e in batch), 'random_extras_never_in_PPO')
    ppo_streams = clone_streams(contexts[-1]['streams'], contexts[-1]['collection_stream_state'])
    equal(ppo_streams.state_dict(), contexts[-1]['collection_stream_state'], 'joint_PPO_last_episode_stream_state')
    guard_resources()
    update_started = time.perf_counter()
    ppo_logs = learner.ppo.update(batch, ppo_streams)  # retain original version guards
    ppo_seconds = time.perf_counter()-update_started
    equal(learner.sac.counts, counts_before, 'joint_PPO_no_SAC_events')
    equal({k:v for k,v in hashes(learner).items() if k.startswith('sac_')},
          {k:v for k,v in frozen.items() if k.startswith('sac_')}, 'joint_PPO_SAC_networks_unchanged')
    equal(learner.ppo.counts['batches']-ppo_before['batches'], 1, 'joint_one_original_PPO_batch')
    if on_learning is not None:
        on_learning(dict(kind='ppo', global_episode=numbers[-1], batch_episodes=numbers,
                         ppo=ppo_logs, ppo_seconds=ppo_seconds))
    hashes_after_ppo = hashes(learner)
    for context in contexts:
        n = context['ordinal']
        validate_behavior_episode(context['episode'], upper, lower)
        equal(context['streams'].state_dict(), context['collection_stream_state'],
              'joint_SAC_uses_own_post_collection_stream')
        before = dict(learner.sac.counts)
        started = time.perf_counter()
        residuals = []
        for local_event in range(1, events_per_ordinal+1):
            guard_resources()
            log = learner.sac.event(replay, context['plan']['alpha'],
                                    context['plan']['actor_enabled'], context['streams'])
            check(not log.get('skipped'), 'joint_no_SAC_skips')
            residuals.extend(v for fit in log.get('critics', []) for v in fit['residual_mse'])
            if on_learning is not None:
                on_learning(dict(kind='sac', global_episode=n, sac_ordinal=n,
                                 batch_end_episode=numbers[-1], local_event=local_event,
                                 event=learner.sac.counts['triggers'], alpha=context['plan']['alpha'], **log))
        context.update(sac_delta=delta(learner.sac.counts, before), sac_counts=dict(learner.sac.counts),
                       ppo_counts=dict(learner.ppo.counts), sac_seconds=time.perf_counter()-started,
                       critic_residual_mse=avg(residuals) if residuals else None,
                       ppo_seconds=ppo_seconds if n == numbers[-1] else 0.)
        equal(context['sac_delta']['triggers'], events_per_ordinal, 'joint_SAC_ordinal_trigger_budget')
        equal(context['sac_delta']['skipped'], 0, 'joint_SAC_ordinal_skip_budget')
        if on_ordinal is not None:
            on_ordinal(context)
    equal(replay.flushes, flush_before, 'joint_whole_batch_no_flush')
    equal({k:v for k,v in hashes(learner).items() if k.startswith('ppo_')},
          {k:v for k,v in hashes_after_ppo.items() if k.startswith('ppo_')}, 'joint_SAC_does_not_change_PPO')
    for context in contexts:
        validate_behavior_episode(context['episode'], upper, lower)
    audit = dict(first_episode=numbers[0], last_episode=numbers[-1], technical_only=technical_only,
                 behavior_upper_version=upper, behavior_lower_version=lower,
                 collection_hashes_before=frozen, collection_hashes_after=collection_hashes_after,
                 hashes_after_PPO=hashes_after_ppo, hashes_after_SAC=hashes(learner),
                 SAC_delta=delta(learner.sac.counts, counts_before), PPO_batches_delta=1,
                 replay_flushes_delta=replay.flushes-flush_before, ppo_seconds=ppo_seconds,
                 stream_source_phases=[c['plan']['source_phase'] for c in contexts],
                 normal_episodes=numbers, extra_episodes=[x['ordinal'] for x in extras],
                 normal_returns=[float(sum(x['episode'].rewards)) for x in contexts],
                 extra_returns=[float(sum(x['episode'].rewards)) for x in extras],
                 extra_lower_versions=[x['episode'].lower_version for x in extras],
                 extra_excluded_from_PPO=not any(e.identity.get('extra_only_SAC',False) for e in batch),
                 replay_mix=replay_mix(replay))
    return dict(audit=audit, contexts=contexts, extras=extras, ppo_logs=ppo_logs)


def compare_s200(seed, learner):
    before = rng_state()
    if seed == 4104:
        path = PRIOR / 'seed_4104/step1_attempt1/checkpoints/A-S200h.pt'
        reference = load(path, binding())
        equal(reference['completed_episode'], 200, 'S200_CPIL_completed_episode')
    else:
        path = old_run(seed) / 'snapshots/episode-0200.pt'
        check(path.exists(), 'S200_original_snapshot_present', str(path))
        side = read(path.with_suffix('.json'))
        equal(sha(path), side['sha256'], 'S200_original_snapshot_digest')
        equal(side['identity'], binding(), 'S200_original_snapshot_side_identity')
        reference = torch.load(path, map_location='cpu', weights_only=False)
        equal(reference['identity'], binding(), 'S200_original_snapshot_identity')
        equal(reference['episode'], 200, 'S200_original_snapshot_episode')
        equal(reference['kind'], 'network_snapshot_not_resumable', 'S200_original_snapshot_kind')
        equal(reference['resumable'], False, 'S200_original_snapshot_nonresumable')
    live = learner.state_dict()
    for family, names in NETWORKS:
        for name in names:
            equal(live[family][name], reference['learner'][family][name],
                  f'S200_bitwise_network_{seed}_{family}_{name}')
    equal(live, reference['learner'], f'S200_complete_learner_bitwise_{seed}')
    record = dict(status='PASS', reference=str(path), reference_sha256=sha(path),
                  all_seven_networks_bitwise_equal=True, complete_learner_state_bitwise_equal=True,
                  includes_optimizer_counts_versions_discounts=True, reference_loaded_into_learner=False)
    del reference, live
    gc.collect()
    rng_assert(before, 'S200_reference_comparison')
    guard_resources()
    return record


def save_candidate(path, learner, replay, seed, n):
    before = rng_state()
    source_guard()
    guard_resources()
    path = output_path(path)
    check(not path.exists() and not path.with_suffix('.json').exists(),
          'new_joint_checkpoint', str(path))
    payload = dict(schema=SCHEMA, safe_episode_boundary=True, identity=binding(),
                   completed_episode=n, pending_evaluation=None, learner=learner.state_dict(),
                   global_cpu_rng=before['cpu'], global_cuda_rng=before['cuda'],
                   runtime=runtime_record(), seed=seed, resumable=False,
                   kind='joint_sync_pair_nonresumable',
                   protocol_type=PROTOCOL_TYPE, protocol_contract_sha256=sha(CPIL/'contract.json'),
                   replay_metadata=replay_summary(replay), replay_included=False,
                   nonresumable_reason='Complete paired network/optimizer state; replay and pending streams intentionally omitted.')
    save(path, payload)
    verified = load(path, binding())
    equal(verified['learner'], learner.state_dict(), 'joint_checkpoint_complete_network_optimizer_readback')
    equal(verified['kind'], 'joint_sync_pair_nonresumable', 'joint_checkpoint_kind')
    equal(verified['resumable'], False, 'joint_checkpoint_nonresumable')
    check('replay' not in verified, 'joint_checkpoint_omits_replay')
    equal(verified['global_cpu_rng'], before['cpu'], 'joint_checkpoint_saved_CPU_RNG')
    equal(verified['global_cuda_rng'], before['cuda'], 'joint_checkpoint_saved_CUDA_RNG')
    del verified, payload
    rng_assert(before, 'joint_checkpoint_save_readback')
    guard_resources()
    return str(path)


def position_summary(run, width=25):
    files = sorted((run/'positions').glob('episode-*.json'))
    result = []
    for start in range(0, len(files), width):
        rows_ = [read(p) for p in files[start:start+width]]
        row = dict(first=rows_[0]['global_episode'], last=rows_[-1]['global_episode'],
                   episodes=len(rows_), position_sampling=rows_[0]['coverage']['definitions'])
        for key in ('distance_start', 'distance_cluster_center', 'pair_distance'):
            values = np.asarray([v for r in rows_ for v in r['coverage'][key]])
            row[key] = [dict(p10=float(np.quantile(v, .1)), p50=float(np.quantile(v, .5)),
                             p90=float(np.quantile(v, .9)), max=float(np.max(v))) for v in values.T]
        row['max_actual_step_m'] = max(r['max_actual_step_m'] for r in rows_)
        result.append(row)
    put(run/'position_25_bins.json', result)
    return result


def protocol_implementation():
    return dict(type=PROTOCOL_TYPE, fresh_single_learner=True,
                learner_checkpoint_loads=0, total_episodes=400, first_200='original hover collection_policy and online SAC',
                first_200_audit='Full JSON-normalized episode metrics and SAC deltas; S200 complete learner including all networks/optimizers/counts/versions/discounts compared exactly to the original reference.',
                joint_episodes=[201, 400], batch_size=5, PPO_batches=40,
                joint_order=['five fixed both-stochastic collections', 'original PPO.update', 'five SAC ordinals of 90 triggers'],
                joint_external_scenarios='original SAC ordinal mapping; does not reproduce historical PPO scenarios',
                source_mapping='n<=300: phase(n); n=301..400: lower phases in original phase(401)..phase(600)',
                frozen_SAC_actor_ordinals=sorted(FROZEN_ORDINALS),
                alpha_ranges=[[1, 100, .005], [101, 200, .0025], [201, 350, .0005], [351, 400, 0.]],
                streams='Each episode retains its independent post-collection Streams instance and copied state. PPO.update receives a state-identical independent clone of the last episode stream; SAC uses each original episode stream.',
                replay='original Replay.add/sample, FIFO 55; current container upper label only; historical episode versions unchanged',
                joint_replay_flushes=0, joint_SAC_skips=0,
                candidates=list(CANDIDATES), checkpoint_kind='joint_sync_pair_nonresumable',
                checkpoint_has_replay=False, resumable=False,
                source_identity_preserved=True, checkpoint_schema_unchanged=True)


def run_seed(seed, on_candidate=None):
    """Fresh 400-episode production run; callback signature is (n, checkpoint)."""
    check(isinstance(seed, int) and not isinstance(seed, bool), 'integer_seed')
    check(seed in contract()['seed_order'], 'seed_authorized_by_frozen_contract', seed)
    assert_imports()
    source_guard()
    guard_resources()
    set_below_normal()
    config = read_config(CONFIG, WT)
    scene = read(SCENE)
    clocks = ClockConfig.from_config(config)
    equal(clocks.lower_steps//20, 90, 'production_ninety_events_per_episode')
    equal(config['ppo']['batch_episodes'], 5, 'production_five_episodes_per_PPO_batch')
    equal(budget_counts(200), EXPECTED200, 'planned_first_200_budget')
    equal(budget_counts(), EXPECTED_FINAL, 'planned_full_budget')
    run = output_path(CPIL/f'seed_{seed}/train')
    check(not run.exists(), 'fresh_training_directory_required', str(run))
    run.mkdir(parents=True)
    manifest_source = {row['global_episode']: row for row in read(old_run(seed)/'training_manifest.json')}
    manifest = []
    for n in range(1, 401):
        p = source_phase(n)
        external = external_seeds(seed, n)
        equal(external, {k:v for k,v in manifest_source[p.global_episode].items() if k.endswith('_seed')},
              'joint_original_SAC_external_seed_mapping')
        manifest.append(dict(ordinal_plan(n), external=external))
    put(run/'training_manifest.json', manifest)
    put(run/'protocol_implementation.json', protocol_implementation())
    learner = Learner(seed, configure(), config=config)
    replay = Replay()
    initial_ppo = copy.deepcopy(learner.state_dict()['ppo'])
    equal(replay_summary(replay), dict(upper_version=None, writes=0, flushes=0, episodes=[]), 'fresh_empty_replay')
    equal(learner.sac.counts, {k:0 for k in EXPECTED_FINAL}, 'fresh_SAC_counts')
    equal(learner.ppo.counts['batches'], 0, 'fresh_PPO_count')
    started = time.perf_counter()
    saved, batches, skipped = [], [], []
    evaluations = []
    s200 = None
    completed = 0

    def progress(n, *, collected=None, state='training'):
        record = dict(status=state, stage=state, seed=seed, completed_episode=n,
                      completed_episodes=n, total_episodes=400,
                      collected_episode=n if collected is None else collected,
                      global_episode=n, sac=dict(learner.sac.counts),
                      ppo=dict(learner.ppo.counts), replay_completed=len(replay.episodes),
                      replay_flushes=replay.flushes, elapsed_seconds=time.perf_counter()-started,
                      time=now())
        put(run/'training_progress.json', record)
        put(run/'progress.json', record)
        root_record=dict(record, status='running', seed_training_status=state)
        if state=='complete':root_record['stage']='seed_training_complete_waiting_diagnostics'
        put(CPIL/'progress.json', root_record)

    def learning(record):
        append(output_path(run/'learning.jsonl'), dict(seed=seed, **record))

    def candidate(n):
        checkpoint_started = time.perf_counter()
        path = save_candidate(run/f'checkpoints/episode-{n:04d}.pt', learner, replay, seed, n)
        saved.append(path)
        checkpoint_seconds = time.perf_counter()-checkpoint_started
        evaluation_seconds = 0.
        if on_candidate is not None:
            before = rng_state()
            fixed_hashes = hashes(learner)
            progress(n, state='development_evaluation')
            evaluation_started = time.perf_counter()
            evaluation = on_candidate(n, path)
            evaluation_seconds = time.perf_counter()-evaluation_started
            rng_assert(before, 'isolated_candidate_evaluation')
            equal(hashes(learner), fixed_hashes, 'candidate_callback_cannot_mutate_live_learner')
            evaluations.append(dict(episode=n, checkpoint=path, result=evaluation,
                                    evaluation_seconds=evaluation_seconds))
            source_guard()
            guard_resources()
        return checkpoint_seconds, evaluation_seconds

    def collection_record(context):
        n = context['ordinal']
        e = context['episode']
        put(run/f'positions/episode-{n:04d}.json',
            dict(global_episode=n, coverage=context['coverage'], max_actual_step_m=context['max_actual_step_m']))
        record = dict(seed=seed, global_episode=n, phase=context['plan'],
                      metrics=dict(compact(e.metrics), event_sha256=e.metrics['event_sha256']),
                      metrics_storage='compact only; collection and in-memory episode unchanged',
                      storage_bytes=e.storage_bytes, upper_version=e.upper_version,
                      lower_version=e.lower_version, external_seeds=context['external'],
                      collection_hashes=context.get('collection_hashes'),
                      collection_complete=True, updates_complete=False)
        put(run/f'episodes/episode-{n:04d}.json', record)
        progress(completed, collected=n, state='collecting_frozen_joint_batch' if n > 200 else 'training')

    def finalize(context):
        nonlocal completed
        n, e = context['ordinal'], context['episode']
        audit_rng = rng_state()
        checkpoint_seconds, evaluation_seconds = candidate(n) if n in CANDIDATES else (0., 0.)
        row = dict(seed=seed, global_episode=n, sac_ordinal=n, layer=context['plan']['layer'],
                   source_phase=context['plan']['source_phase'], alpha=context['plan']['alpha'],
                   actor_enabled=context['plan']['actor_enabled'],
                   reward_physical=float(sum(e.rewards)),
                   reward_learning_macro_sum=float(sum(h['reward_learning'] for h in e.high)),
                   collection_seconds=context['collection_seconds'],
                   sac_seconds=context.get('sac_seconds', 0.), ppo_seconds=context.get('ppo_seconds', 0.),
                   work_seconds=context['collection_seconds']+context.get('sac_seconds', 0.)+context.get('ppo_seconds', 0.),
                   checkpoint_seconds=checkpoint_seconds, evaluation_seconds=evaluation_seconds,
                   critic_residual_mse=context.get('critic_residual_mse'),
                   actor_relative_entropy=context['actor_relative_entropy'], entropy_samples=context['entropy_samples'],
                   entropy_scope='executed lower samples plus real bootstrap successor',
                   sac_delta=context['sac_delta'], sac=context['sac_counts'], ppo=context['ppo_counts'],
                   resources=resources(), event_sha256=e.metrics['event_sha256'],
                   behavior_upper_version=e.upper_version, behavior_lower_version=e.lower_version,
                   replay_completed_before=context['replay_completed_before'],
                   replay_completed_after=context['replay_completed_after'], replay_flushes=replay.flushes,
                   **compact(e.metrics))
        row['wall_seconds'] = row['work_seconds']+checkpoint_seconds+evaluation_seconds
        episode_path = run/f'episodes/episode-{n:04d}.json'
        record = read(episode_path)
        record.update(updates_complete=True, train_metric=row)
        put(episode_path, record)
        append(output_path(run/'train_metrics.jsonl'), row)
        if row['sac_delta']['skipped']:
            skipped.append(n)
        completed = n
        progress(n)
        print('JOINT_TRAIN', seed, n, 400, row['layer'], row['truncated_delay_s'],
              round(row['wall_seconds'], 3), flush=True)
        rng_assert(audit_rng, 'episode_output_checkpoint_evaluation')

    try:
        for n in range(1, 201):
            source_guard()
            guard_resources()
            before_rng = rng_state()
            p = phase(n)
            external = external_seeds(seed, n)
            streams = Streams(seed, p.stage, p.cycle, p.block_episode)
            adapter = PhysicsAdapter(scene, timing=clocks)
            adapter.reset(**external)
            replay.set_upper_version('hover-0')  # original first-200 path, including initial empty flush
            available = len(replay.episodes)
            policy = collection_policy(learner, streams, p)  # do not supply value or alter original RNG calls
            rng_assert(before_rng, 'hover_pre_episode')
            before_counts = dict(learner.sac.counts)
            residuals = []
            coverage = Coverage(adapter)
            episode_started = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            def event():
                guard_resources()
                log = learner.sac.event(replay, p.alpha, p.actor_enabled, streams)
                residuals.extend(v for fit in log.get('critics', []) for v in fit['residual_mse'])
                learning(dict(kind='sac', global_episode=n, sac_ordinal=n,
                              physical_tick=adapter.env.step_index, event=learner.sac.counts['triggers'],
                              alpha=p.alpha, **log))

            lower_at_start = learner.sac.version
            e = collect(adapter, policy, upper_version='hover-0', lower_version=lower_at_start,
                        identity=dict(global_episode=n, external=external),
                        on_event=event, on_tick=coverage.tick,
                        reward_scale=config['objective']['reward_scale'], gamma_lower=config['objective']['gamma_micro'])
            replay.add(e)
            audit_rng = rng_state()
            sac_delta = delta(learner.sac.counts, before_counts)
            historical = read(old_run(seed)/f'episodes/episode-{n:04d}.json')
            equal(e.metrics['event_sha256'], historical['metrics']['event_sha256'], 'hover_exact_event_digest')
            # Historical JSON stores numpy scalar values as native JSON numbers.
            # Normalize only this audit copy; the replay episode remains untouched.
            equal(normalize(e.metrics), historical['metrics'], 'hover_complete_metrics_bitwise')
            equal(json_digest(normalize(e.metrics)),json_digest(historical['metrics']),'hover_exact_metric_serialized_bits')
            equal(sac_delta, historical['train_metric']['sac_delta'], 'hover_exact_SAC_delta')
            del historical
            maximum = float(np.linalg.norm(np.asarray([h['displacement'] for h in e.high]), axis=-1).max())
            check(maximum <= 20+1e-6, 'hover_motion_cap', maximum)
            context = dict(ordinal=n, plan=ordinal_plan(n), episode=e, external=external,
                           coverage=coverage.result(), max_actual_step_m=maximum,
                           replay_completed_before=available, replay_completed_after=len(replay.episodes),
                           collection_seconds=time.perf_counter()-episode_started,
                           actor_relative_entropy=policy.entropy_sum/policy.entropy_count if policy.entropy_count else None,
                           entropy_samples=policy.entropy_count, sac_delta=sac_delta,
                           sac_counts=dict(learner.sac.counts), ppo_counts=dict(learner.ppo.counts),
                           critic_residual_mse=avg(residuals) if residuals else None)
            if n == 200:
                equal(learner.sac.counts, EXPECTED200, 'S200_exact_budget')
                equal(learner.state_dict()['ppo'], initial_ppo, 'first_200_complete_PPO_unchanged')
                s200 = compare_s200(seed, learner)
                put(run/'s200_bitwise_comparison.json', s200)
                equal(len(replay.episodes), 55, 'S200_replay55')
                equal(replay.flushes, 1, 'S200_only_initial_empty_flush')
            collection_record(context)
            finalize(context)
            rng_assert(audit_rng, 'hover_post_episode')
            del e, adapter, policy, coverage, context
            if n % 25 == 0:
                position_summary(run)
        del initial_ppo
        joint_flush_start = replay.flushes
        for first in range(201, 401, 5):
            source_guard()
            guard_resources()
            batch = run_joint_batch(learner, replay, seed, range(first, first+5), config, scene, clocks,
                                    on_collected=collection_record, on_learning=learning, on_ordinal=finalize)
            batches.append(batch['audit'])
            put(run/f'batches/batch-{first:04d}-{first+4:04d}.json', batch['audit'])
            del batch
            if (first+4) % 25 == 0:
                position_summary(run)
        source_guard()
        guard_resources()
        equal(learner.sac.counts, EXPECTED_FINAL, 'joint_final_exact_SAC_budget')
        equal(learner.ppo.counts['batches'], 40, 'joint_final_40_PPO_batches')
        equal(replay.flushes-joint_flush_start, 0, 'joint_region_no_replay_flushes')
        equal(skipped, [1], 'joint_only_first_empty_episode_skips')
        equal(len(saved), 11, 'joint_exactly_eleven_candidates')
        equal([e.identity['global_episode'] for e in replay.episodes], list(range(346, 401)), 'joint_final_FIFO55')
        result = dict(status='passed', seed=seed, run_dir=str(run),
                      train_metrics_path=str(run/'train_metrics.jsonl'), learning_path=str(run/'learning.jsonl'),
                      training_progress_path=str(run/'training_progress.json'),
                      episodes=400, SAC_counts=dict(learner.sac.counts), SAC_delta=dict(learner.sac.counts),
                      PPO_counts=dict(learner.ppo.counts), PPO_batches=40,
                      joint_SAC_delta=delta(learner.sac.counts, EXPECTED200),
                      joint_replay_flushes_delta=replay.flushes-joint_flush_start,
                      replay_flushes_total=replay.flushes, final_replay=replay_summary(replay),
                      skipped_episodes=skipped, checkpoints=saved, candidate_evaluations=evaluations,
                      s200_comparison=s200, batch_audits=batches, protocol=protocol_implementation(),
                      wall_seconds=time.perf_counter()-started, position_bins=position_summary(run))
        equal(result['joint_SAC_delta'], EXPECTED_JOINT, 'joint_region_exact_SAC_budget')
        put(run/'result.json', result)
        progress(400, state='complete')
        return result
    finally:
        source_guard()
        del learner, replay
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    raise RuntimeError('Use j_main.py; this module never launches standalone training.')
