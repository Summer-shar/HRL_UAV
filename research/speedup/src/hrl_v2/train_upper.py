"""Independent NC/TOO/TFOO training; no-argument execution is a dry run."""
import argparse
import json
from pathlib import Path
import time
import uuid

import torch

from . import SCHEMA
from .checkpoint import (atomic_json, digest, identity, json_digest, runtime_record,
                         runtime_signature, save)
from .collector import collect
from .evaluate import gate_manifest, manifest
from .paths import source_root
from .physics_adapter import PhysicsAdapter
from .randomness import derive, Streams
from .run import append, read_config, require_research_protocol
from .archive import (checkpoint_interval, should_checkpoint, publish_checkpoint,
                      supersede_episodes, recover_metric, save_network_snapshot)
from .runtime import configure
from .telemetry import compact, resources
from .timing import ClockConfig
from .upper_baselines import (UpperLearner, baseline_tag, fixed_upper_policy,
                              load_typed, upper_phases)


def training_manifest(seed, phases=None):
    phases = upper_phases() if phases is None else phases
    return [dict(episode=i, proposed_global_episode=p.global_episode, stage=p.stage,
                 cycle=p.cycle, block_episode=p.block_episode,
                 **{k+'_seed': derive(seed, p.stage, p.cycle, p.block_episode, purpose=k)
                    for k in ('arrival', 'layout', 'load')}) for i, p in enumerate(phases, 1)]


def expected_budget(config, phases=None):
    phases = upper_phases() if phases is None else phases
    clocks = ClockConfig.from_config(config)
    if config['ppo']['batch_episodes'] != 5 or len(phases) % 5:
        raise ValueError('Upper budget must contain complete five-episode PPO batches')
    return dict(episodes=len(phases), ppo_batches=len(phases)//5,
                physical_steps=len(phases)*clocks.physical_steps,
                upper_transitions=len(phases)*clocks.upper_steps,
                sac_forwards=0, sac_updates=0, sac_replay_writes=0)


def collect_upper(ppo, scene, config, seed, p, route):
    """Same physical collector/GAE rewards as Proposed, without SAC access."""
    streams = Streams(seed, p.stage, p.cycle, p.block_episode)
    adapter = PhysicsAdapter(scene, timing=ClockConfig.from_config(config))
    external = {k+'_seed': derive(seed, p.stage, p.cycle, p.block_episode, purpose=k)
                for k in ('arrival', 'layout', 'load')}
    adapter.reset(**external)
    policy = fixed_upper_policy(ppo, scene, streams, route, stochastic=True, with_value=True)
    episode = collect(adapter, policy, upper_version=f'ppo-{ppo.version}-stochastic',
                      lower_version='fixed-'+route,
                      identity=dict(proposed_global_episode=p.global_episode, external=external),
                      reward_scale=config['objective']['reward_scale'],
                      gamma_lower=config['objective']['gamma_micro'])
    return episode, streams


def _run(learner, scene, config, seed, root, output, baseline, binding, *, phases=None, saved=None, config_path=None, checkpoint_every=10):
    """Small schedules are injectable in tests only; the CLI fixes 200 episodes."""
    checkpoint_interval(checkpoint_every)
    root, output = Path(root), Path(output)
    phases = upper_phases() if phases is None else phases
    budget = expected_budget(config, phases)
    tag = baseline_tag(baseline)
    relative = str(output.relative_to(root)).replace('\\', '/')
    completed, batch = 0, []
    attempt_id = uuid.uuid4().hex
    last_streams = None
    last_train_metric = None
    if saved is not None:
        from .upper_baselines import require_checkpoint_type
        require_checkpoint_type(saved, baseline)
        if saved['output_relative'] != relative or saved['budget'] != budget or saved['seed'] != seed:
            raise ValueError('Upper resume directory, seed or budget mismatch')
        learner.load_state_dict(saved['learner'])
        completed, batch = saved['completed_episode'], saved['ppo_batch']
        last_streams = saved['streams']
        last_train_metric = saved.get('last_train_metric')
        if not 0 <= completed <= len(phases) or len(batch) != completed % 5:
            raise ValueError('Upper resume boundary mismatch')
        torch.set_rng_state(saved['global_cpu_rng'])
        if torch.cuda.is_available(): torch.cuda.set_rng_state_all(saved['global_cuda_rng'])
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output/'contract.json', dict(**tag, identity=binding, config=config,
            seed=seed, budget=budget, runtime=runtime_record(), checkpoint_every=checkpoint_every, seed_mapping='Proposed upper phases only'))
        atomic_json(output/'training_manifest.json', training_manifest(seed, phases))
        atomic_json(output/'development_manifest.json', gate_manifest(20))
        atomic_json(output/'test_manifest.json', manifest('test'))
    last_checkpoint_episode=completed
    def persist(label):
        nonlocal last_checkpoint_episode
        state = dict(schema=SCHEMA, safe_episode_boundary=True, **tag, identity=binding, seed=seed,
            completed_episode=completed, pending_evaluation=None, budget=budget, output_relative=relative,
            learner=learner.state_dict(), ppo_batch=batch, streams=last_streams,
            global_cpu_rng=torch.get_rng_state(), global_cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            config=config, scene=scene, runtime=runtime_record(), last_train_metric=last_train_metric)
        if config_path is not None and identity(root, config_path) != binding:
            raise RuntimeError('Source/config changed during upper training')
        path = output/'checkpoints'/f'episode-{completed:04d}-{label}.pt'
        if path.exists(): path = path.with_name(path.stem+'-'+uuid.uuid4().hex[:8]+'.pt')
        save(path, state)
        publish_checkpoint(root,output,path,completed,attempt_id)
        last_checkpoint_episode=completed
        # Keep the same two rolling completed boundaries as Proposed, confined
        # to this dedicated run; never touch another run's checkpoints.
        rolling = sorted((output/'checkpoints').glob('episode-*-complete*.pt'),key=lambda p:p.stat().st_mtime)
        for old in rolling[:-2]:
            if old.parent.resolve() != (output/'checkpoints').resolve(): raise RuntimeError('Checkpoint path escaped run')
            old.with_suffix('.json').unlink(); old.unlink()
    if saved is not None: supersede_episodes(output,completed)
    if saved is None: persist('initial')
    append(output/'attempts.jsonl',dict(attempt_id=attempt_id,resumed=saved is not None,
        restored_completed_episode=completed,checkpoint_every=checkpoint_every,
        discarded_after_episode=completed if saved is not None else None))
    if saved is not None: recover_metric(output,completed)
    dev_nodes=set(config['schedule']['development_global_episodes'])|{300,400,600}
    anchors=tuple(i for i,p in enumerate(phases,1) if p.global_episode in dev_nodes)
    for number in range(completed+1, len(phases)+1):
        if config_path is not None and identity(root, config_path) != binding:
            raise RuntimeError('Source/config changed during upper training')
        p = phases[number-1]
        started = time.perf_counter()
        episode, streams = collect_upper(learner.ppo, scene, config, seed, p, tag['lower_override'])
        batch.append(episode)
        ppo_start = time.perf_counter()
        if len(batch) == 5:
            logs = learner.ppo.update(batch, streams)
            batch.clear()
            append(output/'learning.jsonl', dict(attempt_id=attempt_id,episode=number, proposed_global_episode=p.global_episode, ppo=logs))
        ppo_seconds = time.perf_counter()-ppo_start
        row = dict(attempt_id=attempt_id,episode=number, proposed_global_episode=p.global_episode, baseline_type=baseline,
                   layer='upper', reward_physical=float(sum(episode.rewards)),
                   reward_learning_macro_sum=float(sum(h['reward_learning'] for h in episode.high)),
                   ppo_batches=learner.ppo.counts['batches'], ppo_seconds=ppo_seconds,
                   work_seconds=time.perf_counter()-started, checkpoint_seconds=None, snapshot_seconds=None,
                   wall_seconds=None, sac_forwards=0, sac_updates=0,
                   sac_replay_writes=0, **compact(episode.metrics))
        completed=number
        last_streams=streams.state_dict()
        last_train_metric=row
        atomic_json(output/'episodes'/f'episode-{number:04d}.json',dict(attempt_id=attempt_id,
            phase=p.__dict__,metrics=episode.metrics,wall_seconds=row['work_seconds'],
            storage_bytes=episode.storage_bytes,ppo=dict(learner.ppo.counts),train_metric=row))
        checkpoint_start = time.perf_counter()
        if should_checkpoint(completed,checkpoint_every,final=len(phases),boundaries=anchors):
            persist('anchor' if completed in anchors or completed==len(phases) else 'complete')
        checkpoint_seconds=time.perf_counter()-checkpoint_start
        snapshot_start=time.perf_counter()
        if completed%25==0: save_network_snapshot(output,learner,completed,binding)
        snapshot_seconds=time.perf_counter()-snapshot_start
        row.update(checkpoint_seconds=checkpoint_seconds,snapshot_seconds=snapshot_seconds,
                   wall_seconds=time.perf_counter()-started, resources=resources())
        append(output/'train_metrics.jsonl', row)
        atomic_json(output/'progress.json', dict(status='complete' if number==len(phases) else 'running',
            **tag, completed_episode=number, last_checkpoint_episode=last_checkpoint_episode, ppo=dict(learner.ppo.counts), budget=budget))
        print(f'[{baseline} {number}/{len(phases)}] reward={row["reward_physical"]:.6f} '
              f'delay={row["truncated_delay_s"]:.6f}s PPO_batches={row["ppo_batches"]} '
              f'wall={row["wall_seconds"]:.2f}s', flush=True)
    if batch or learner.ppo.counts['batches'] != budget['ppo_batches']:
        raise RuntimeError('Upper PPO update budget mismatch')
    result = dict(status='complete_performance_not_evaluated', **tag, actual_counts=budget,
                  ppo=dict(learner.ppo.counts), completed_episode=len(phases))
    atomic_json(output/'result.json', result)
    return result


def train(root, config_path, seed, output, baseline, resume=None, checkpoint_every=10):
    checkpoint_interval(checkpoint_every)
    root, output = Path(root).resolve(), Path(output).resolve()
    config = read_config(config_path, root)
    require_research_protocol(config)
    baseline_tag(baseline)
    binding = identity(root, config_path)
    budget = expected_budget(config)
    if (budget['episodes'], budget['ppo_batches'], budget['physical_steps']) != (200, 40, 1800000):
        raise ValueError('Production upper baseline budget is 200 x 900 seconds / 40 PPO batches')
    validation = json.loads((root/'validation/validation.json').read_text(encoding='utf-8'))
    if validation.get('status') != 'passed' or validation.get('identity') != binding:
        raise ValueError('Matching validation is required before upper baseline training')
    saved = load_typed(resume, binding, baseline) if resume else None
    if saved:
        seed = saved['seed']
        latest = json.loads((output/'latest.json').read_text(encoding='utf-8'))
        if latest['sha256'] != digest(resume): raise ValueError('Resume requires this run latest checkpoint')
    elif output.exists():
        raise FileExistsError('Fresh upper baselines never overwrite a run directory')
    device = configure()
    if runtime_signature(validation.get('runtime', {})) != runtime_signature(runtime_record()):
        raise ValueError('Upper training runtime differs from validation')
    if saved and runtime_signature(saved.get('runtime', {})) != runtime_signature(runtime_record()):
        raise ValueError('Upper resume runtime mismatch')
    learner = UpperLearner(seed, device, config)
    scene = json.loads((source_root()/'hrl_v2/configs/physical_scene.json').read_text(encoding='utf-8'))
    return _run(learner, scene, config, seed, root, output, baseline, binding, saved=saved, config_path=config_path, checkpoint_every=checkpoint_every)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path.cwd()/'outputs')
    parser.add_argument('--config', type=Path, default=source_root()/'hrl_v2/configs/v2_4_0_lite.json')
    parser.add_argument('--baseline', choices=('NC','TOO','TFOO'), required=True,
                        help='NC=local, TOO=equal, TFOO=cvx fixed routing')
    parser.add_argument('--seed', type=int, choices=range(4101,4106))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--checkpoint-every', type=int, default=10)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--fresh', action='store_true')
    group.add_argument('--resume', type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    rooted = lambda p: p if p.is_absolute() else root/p
    config_path = rooted(args.config)
    config = read_config(config_path, root)
    binding = identity(root, config_path)
    if not args.fresh and args.resume is None:
        print(json.dumps(dict(status='dry_run', **baseline_tag(args.baseline),
                              budget=expected_budget(config), physical_steps=0, updates=0)))
        return
    if args.fresh and args.seed is None: parser.error('--fresh requires --seed')
    if args.resume:
        if args.seed is not None or args.output is not None: parser.error('--resume reads seed and directory from checkpoint')
        saved = load_typed(rooted(args.resume), binding, args.baseline)
        output = root/saved['output_relative']
    else:
        output = rooted(args.output) if args.output else root/'upper_baselines'/json_digest(binding)[:12]/args.baseline/f'seed_{args.seed}'
    result = train(root, config_path, args.seed, output, args.baseline, rooted(args.resume) if args.resume else None,checkpoint_every=args.checkpoint_every)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__': main()
