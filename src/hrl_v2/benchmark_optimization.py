"""Bounded production-path throughput measurements; never starts research training.

One scripted finite episode supplies immutable reports and complete 1800-frame
histories. SAC keeps 128 samples / four episode identities. PPO invokes update
on all 900 rows and stops after one complete 150-row optimizer step; its full
six-epoch budget is an explicitly labelled extrapolation. Run variants in
separate processes, with no concurrent GPU work.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib
import importlib.util
import json
import pickle
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "hrl_v2/_reference/hrl_v2"


def verify_baseline():
    manifest=json.loads((BASELINE.parent/"manifest.json").read_text(encoding="utf-8"))
    if manifest["commit"]!="137d2f1": raise RuntimeError("Wrong reference baseline")
    for name,digest in manifest["sha256"].items():
        if hashlib.sha256((BASELINE.parent/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError("Frozen baseline source changed: "+name)
    return manifest["sha256"]


def package(variant):
    if variant == "baseline":
        name = "v2_benchmark_baseline"
        spec = importlib.util.spec_from_file_location(name, BASELINE / "__init__.py", submodule_search_locations=[str(BASELINE)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    else:
        name = "hrl_v2"
    return lambda suffix: importlib.import_module(name + "." + suffix)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timing(fn, repeats=3, warmup=1):
    for _ in range(warmup):
        fn()
    values = []
    result = None
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        result = fn()
        sync()
        values.append(time.perf_counter() - start)
    return dict(seconds=statistics.median(values), samples_seconds=values,
                repeats=repeats, warmup=warmup), result


class Scripted:
    def upper(self, report, history):
        return .2 * np.sin(np.arange(6) + report.tick / 50), {}

    def lower(self, report, history):
        from hrl_v2.hybrid_simplex import SUPPORTS
        values = np.asarray([[.3, .5, .2], [.6, .4, 0], [0, 1, 0], [.4, 0, .6], [0, .3, .7], [1, 0, 0]], dtype=np.float64)
        actions = np.roll(values, report.tick // 5, axis=0).reshape(2, 3, 3)
        return actions, np.asarray([SUPPORTS.index(tuple(np.flatnonzero(row))) for row in actions.reshape(6, 3)])


def make_fixture(path):
    from hrl_v2.physics_adapter import PhysicsAdapter
    from hrl_v2.collector import collect
    scene = json.loads((ROOT / "hrl_v2/configs/physical_scene.json").read_text())
    adapter = PhysicsAdapter(scene)
    adapter.reset(arrival_seed=1, layout_seed=2, load_seed=3)
    start = time.perf_counter()
    episode = collect(adapter, Scripted(), upper_version="benchmark-script", lower_version=0,
                      identity=dict(bounded_fixture=True, arrival_seed=1, layout_seed=2, load_seed=3))
    elapsed = time.perf_counter() - start
    payload = dict(episode=episode, seconds=elapsed, physics_steps=9000, learning_updates=0,
                   description="One complete scripted physical trajectory, no learned policy, no research episode")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(payload, protocol=5))
    return payload


def episodes_for(mod, fixture, count):
    result = []
    for index in range(count):
        source = fixture["episodes"][index] if "episodes" in fixture else fixture["episode"]
        episode = mod("replay").Episode(source.upper_version, source.lower_version,
                                        dict(source.identity, replica=index))
        # Bytes can be shared; resolve must still produce a different raw
        # history object for every episode identity in the optimizer step.
        for key in ("reports", "observations", "successors", "actions", "supports", "rewards", "terminated", "severe", "high"):
            setattr(episode, key, copy.deepcopy(getattr(source, key)))
        episode.reports = [memoryview(value).tobytes() for value in source.reports]
        episode.history = memoryview(source.history).tobytes()
        episode.complete = True
        episode.metrics = copy.deepcopy(source.metrics)
        result.append(episode)
    return result


def fixed_samples(mod, episodes, seed=71027):
    rng = np.random.default_rng(seed)
    result = []
    for episode in episodes[:4]:
        probability = mod("replay").probabilities(episode.severe)
        for index in rng.choice(1800, size=32, replace=True, p=probability):
            result.append(mod("replay").Sample(episode, int(index), float((1 / 1800) / probability[index])))
    return result


def measure(args):
    baseline_hashes = verify_baseline()
    mod = package(args.variant)
    device = mod("runtime").configure()
    fixture = pickle.loads(args.fixture.read_bytes())
    episode = fixture["episode"]
    history = mod("replay").unpack(episode.history)
    reports = [episode.report(i) for i in episode.observations]
    sorted_indices = sorted(range(1800), key=lambda i: len(reports[i].fragments))
    selected = dict(ordinary=sorted_indices[900], congested=sorted_indices[-1])
    record = dict(variant=args.variant, baseline_commit="137d2f1", baseline_source_hashes=baseline_hashes,
                  runtime=mod("checkpoint").runtime_record(), measurement_unix=time.time(),
                  fixture=dict(physical_steps=9000, history_frames=len(history), episode_storage_bytes=episode.storage_bytes,
                               scripted_collection_seconds=fixture["seconds"],
                               independent_scripted_episodes=len(fixture.get("episodes",[])),
                               replicas_are_not_independent_scenarios="episodes" not in fixture),
                  fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
                  measured_source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted((BASELINE if args.variant == "baseline" else ROOT / "hrl_v2").glob("*.py"))},
                  research_training_episodes=0, distribution_backend="cpu/float64 unchanged", measurements={})
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def write(name, value):
        record["measurements"][name] = value
        mod("checkpoint").atomic_json(args.output, record)
        print(json.dumps(dict(measurement=name, value=value), ensure_ascii=False), flush=True)

    learner = mod("runtime").Learner(4101, device)
    streams = lambda: mod("randomness").Streams(4101)
    episodes = episodes_for(mod, fixture, 5)
    samples = fixed_samples(mod, episodes)
    states = mod("replay").resolve(samples)
    sections = set(args.sections.split(","))
    enabled = lambda name: "all" in sections or name in sections

    if enabled("components"):
        t, _ = timing(lambda: mod("replay").resolve(samples))
        write("resolve_128", t)
        t, _ = timing(lambda: mod("replay").unpack(episode.history))
        write("decompress_full_history", t)
        t, _ = timing(lambda: [mod("features").build(reports[selected[k]]) for k in selected])
        write("fixed_graph_features_two_reports", t)
        histories = [mod("replay").unpack(e.history) for e in episodes[:4]]
        from dataclasses import replace
        long_states = [(replace(reports[-1], history_cursor=1800), h) for h in histories]
        encoder = learner.sac.actor.encoder

        def history_grad():
            encoder.zero_grad(set_to_none=True)
            result = encoder.histories(long_states)
            sum(x.square().mean() for x in result).backward()
        t, _ = timing(history_grad)
        write("four_full_1800_history_forward_backward", t)
        t, _ = timing(lambda: mod("features").arrival_array(history, 1800, device))
        write("history_normalize_and_h2d", t)
        cpu = torch.randn(1800, 39, 6, dtype=torch.float32)
        t, _ = timing(lambda: cpu.to(device))
        write("history_h2d_only", t)
        gpu_logits = torch.randn(16, 6, 7, device=device)
        t, _ = timing(lambda: gpu_logits.to("cpu", dtype=torch.float64))
        write("distribution_logits_d2h16", t)
        logits = torch.zeros(16, 6, 7, dtype=torch.float64)
        kappa = torch.full((16, 6, 7, 3), 2., dtype=torch.float64)
        t, _ = timing(lambda: mod("hybrid_simplex").sample(logits, kappa, streams(), 200), repeats=5)
        write("cpu_float64_distribution_sample16", t)

        def distribution_backward():
            l = logits.detach().requires_grad_()
            k = kappa.detach().requires_grad_()
            value = mod("hybrid_simplex").surrogate(l, k, lambda a: a.square().sum((-1, -2)), .005, streams())
            value.backward()
        t, _ = timing(distribution_backward, repeats=5)
        write("cpu_float64_surrogate16_without_network", t)
        with torch.no_grad():
            hidden = encoder.histories(states[:16])
            with patch.object(encoder, "histories", return_value=hidden):
                t, _ = timing(lambda: learner.sac.actor.batch(states[:16]))
            write("graph_and_actor16_history_precomputed", t)
        if hasattr(mod("batching"), "prepare"):
            from hrl_v2.prepared import PreparedBatch
            t, _ = timing(lambda: mod("batching").prepare(PreparedBatch(list(states[:16])), device))
            write("graph_index_device_prepare16_cold", t)
            prepared = PreparedBatch(list(states[:16]))
            mod("batching").prepare(prepared, device)
            t, _ = timing(lambda: mod("batching").prepare(prepared, device))
            write("graph_index_device_prepare16_reuse", t)

    if enabled("inference"):
        for label, index in selected.items():
            begin = max(0, min(index - 8, 1784))
            chosen = list(range(begin, begin + 16))
            def infer_window(value=False, upper=False):
                data = [episode.report(row["report"]) for row in episode.high] if upper else reports
                first = max(0, min(index // 10 - 8, 164)) if upper else begin
                chosen = list(range(first, first + 16))
                policy = mod("policy").Policy(learner.sac.actor, learner.ppo.actor,
                    learner.ppo.value if value else None, streams())
                # A single policy holds the same episode identity and immutable
                # append-only history; prime just before the measured window.
                if first:
                    (policy.upper if upper else policy.lower)(data[first - 1], history)
                sync()
                start = time.perf_counter()
                for i in chosen:
                    (policy.upper if upper else policy.lower)(data[i], history)
                sync()
                return (time.perf_counter() - start) / len(chosen)
            for purpose, kwargs in (("lower", {}), ("upper", dict(upper=True)), ("upper_with_value", dict(upper=True, value=True))):
                times = [infer_window(**kwargs) for _ in range(3)]
                write(f"{label}_{purpose}_inference", dict(seconds=statistics.median(times), samples_seconds=times,
                    frames_per_sample=16, representative_tick=reports[index].tick,
                    parents=len(reports[index].parents), fragments=len(reports[index].fragments),
                    history_cursor=reports[index].history_cursor, incremental_cache_primed=True))
        rebuild_indices = list(range(40, 1800, 40))
        rebuild_times, warm_times = [], []
        for index in rebuild_indices:
            policy = mod("policy").Policy(learner.sac.actor, learner.ppo.actor, None, streams())
            t, _ = timing(lambda: policy.lower(reports[index], history), repeats=1, warmup=0)
            rebuild_times.append(t["seconds"])
            t, _ = timing(lambda: policy.lower(reports[index + 1], history), repeats=1, warmup=0)
            warm_times.append(t["seconds"])
        write("lower_post_update_rebuild", dict(seconds=float(np.mean(rebuild_times)), samples_seconds=rebuild_times,
            warm_next_mean_seconds=float(np.mean(warm_times)), cursors=[reports[i].history_cursor for i in rebuild_indices],
            note="44 interior actor-update boundaries per eligible episode; terminal update has no following inference"))
        start_policy = mod("policy").Policy(learner.sac.actor, learner.ppo.actor, None, streams())
        t, _ = timing(lambda: start_policy.lower(reports[0], history), repeats=1, warmup=0)
        write("lower_episode_start", t)

    if enabled("sac"):
        for microbatch in args.microbatches:
            if args.variant == "baseline" and microbatch != 16:
                continue
            temporary = mod("runtime").Learner(4101, device)
            if hasattr(temporary.sac, "microbatch_size"):
                temporary.sac.microbatch_size = microbatch
            elif microbatch != 16:
                raise RuntimeError("Production SAC does not expose microbatch_size")
            torch.cuda.reset_peak_memory_stats()
            # Production resamples every fit. Repeating one batch accidentally
            # benchmarks a permanently hot baseline Graph LRU instead.
            critic_batches=iter([fixed_samples(mod,episodes,71027+i) for i in range(4)])
            t, logs = timing(lambda: temporary.sac._critic_fit(next(critic_batches), .005, streams()), repeats=3, warmup=1)
            write(f"critic128_micro{microbatch}", dict(t, logs=logs, effective_batch=128,
                  episode_identities=4, microbatch=microbatch, peak_allocated_bytes=torch.cuda.max_memory_allocated()))
            torch.cuda.reset_peak_memory_stats()
            actor_batches=iter([fixed_samples(mod,episodes,71127+i) for i in range(4)])
            t, logs = timing(lambda: temporary.sac._actor_fit(next(actor_batches), .005, streams()), repeats=3, warmup=1)
            write(f"actor128_micro{microbatch}", dict(t, logs=logs, effective_batch=128,
                  episode_identities=4, microbatch=microbatch, peak_allocated_bytes=torch.cuda.max_memory_allocated()))
            learner.sac = temporary.sac
            del temporary
            torch.cuda.empty_cache()

    if enabled("ppo"):
        for e in episodes:
            policy = mod("policy").Policy(learner.sac.actor, learner.ppo.actor, learner.ppo.value, streams())
            e_history=mod("replay").unpack(e.history)
            source_rows=[]
            for row in e.high:
                _,aux=policy.upper(e.report(row["report"]),e_history)
                source_rows.append(dict(row,**aux))
            e.high=source_rows
        temporary = mod("runtime").Learner(4101, device)
        starts = []
        ended = []
        first_batch = temporary.ppo.actor.batch
        first_minibatch = getattr(temporary.ppo, "_minibatch", None)
        original_step = temporary.ppo.value_optimizer.step
        class CompletedMinibatch(Exception):
            pass
        def batch(*a, **kw):
            if first_minibatch is None and not starts:
                sync()
                starts.append(time.perf_counter())
            return first_batch(*a, **kw)
        def minibatch(*a, **kw):
            sync()
            starts.append(time.perf_counter())
            return first_minibatch(*a, **kw)
        def stop_after_step(*a, **kw):
            result = original_step(*a, **kw)
            sync()
            ended.append(time.perf_counter())
            raise CompletedMinibatch()
        sync()
        before = time.perf_counter()
        with contextlib.ExitStack() as scope:
            scope.enter_context(patch.object(temporary.ppo.actor, "batch", side_effect=batch))
            scope.enter_context(patch.object(temporary.ppo.value_optimizer, "step", side_effect=stop_after_step))
            if first_minibatch is not None:
                scope.enter_context(patch.object(temporary.ppo, "_minibatch", side_effect=minibatch))
            try:
                temporary.ppo.update(episodes, streams())
            except CompletedMinibatch:
                pass
        if len(ended) != 1 or not starts:
            raise RuntimeError("Expected exactly one completed PPO value optimizer step")
        write("ppo150_actual_first_minibatch", dict(seconds=ended[0] - starts[0],
            update_setup_seconds=starts[0] - before, effective_batch=900, optimizer_minibatch=150,
            epochs_in_protocol=6, stopped_after="one complete actor and value optimizer step",
            caveat="Only a temporary model; production PPO loop and original minibatch denominator are unchanged"))
        rows = [(e, h, float(G)) for e in episodes for h, G in zip(e.high, mod("ppo").returns([x["reward"] for x in e.high]))]
        histories = {id(e): mod("replay").unpack(e.history) for e in episodes}
        kl_states = [(e.report(h["report"]), histories[id(e)]) for e, h, _ in rows]
        if hasattr(temporary.ppo, "_kl"):
            kl_states = mod("prepared").PreparedBatch(kl_states)
        def kl900():
            if hasattr(temporary.ppo, "_kl"):
                return temporary.ppo._kl(kl_states, rows)
            result = []
            with torch.no_grad():
                for start in range(0, 900, 16):
                    chosen = rows[start:start + 16]
                    states = [(e.report(h["report"]), histories[id(e)]) for e, h, _ in chosen]
                    mean, std = temporary.ppo.actor.batch(states)
                    old_mean = mean.new_tensor(np.asarray([h["mean"] for _, h, _ in chosen]))
                    old_std = std.new_tensor(np.asarray([h["logstd"] for _, h, _ in chosen]))
                    result.extend(mod("ppo").gaussian_kl(old_mean, old_std, mean, std).cpu().tolist())
            return float(np.mean(result))
        t, kl = timing(kl900, repeats=1, warmup=0)
        write("ppo_kl900", dict(t, kl=kl, production_method=hasattr(temporary.ppo, "_kl"),
            caveat="Production shared KL method when available; baseline reproduces exact original traversal and expression."))
        learner.ppo = temporary.ppo
        del temporary
        torch.cuda.empty_cache()
        if args.full_ppo:
            # Execute the real complete update, including its original KL
            # rollback/early stopping, on a disposable model only.
            full=mod("runtime").Learner(4101,device)
            t,logs=timing(lambda: full.ppo.update(episodes,streams()),repeats=1,warmup=0)
            write("ppo900_complete_batch",dict(t,counts=dict(full.ppo.counts),epoch_logs=logs,
                effective_batch=900,optimizer_minibatch=150,epochs=6,
                caveat="Five scripted trajectories with synthetic current-policy auxiliary values; computation fixture, not an on-policy research batch"))
            del full
            torch.cuda.empty_cache()

    if enabled("io"):
        replay = mod("replay").Replay()
        replay.set_upper_version(episodes[0].upper_version)
        for e in episodes[:4]:
            replay.add(e)
        t, _ = timing(lambda: replay.sample(streams().numpy), repeats=5)
        write("replay_sample128", t)
        t, _ = timing(lambda: [mod("replay").pack(reports[i]) for i in np.linspace(0, 1799, 32, dtype=int)])
        write("report_compression32", t)
        with tempfile.TemporaryDirectory(prefix="bounded-checkpoint-", dir=args.output.parent) as directory:
            path = Path(directory) / "checkpoint.pt"
            binding = dict(benchmark=True, baseline="137d2f1", variant=args.variant)
            payload = dict(schema=importlib.import_module(mod("checkpoint").__package__).SCHEMA,
                safe_episode_boundary=True, completed_episode=1, pending_evaluation=None, identity=binding,
                learner=learner.state_dict(), replay=replay, ppo_batch=[], streams=streams().state_dict())
            optimizer_slots = {name: len(optimizer.state) for name, optimizer in (
                ("sac_actor", learner.sac.actor_optimizer), ("sac_critic", learner.sac.critic_optimizer),
                ("ppo_actor", learner.ppo.actor_optimizer), ("ppo_value", learner.ppo.value_optimizer))}
            empty_payload = dict(payload, replay=mod("replay").Replay())
            t, _ = timing(lambda: mod("checkpoint").save(path, empty_payload), repeats=1, warmup=0)
            write("checkpoint_empty_replay_save", dict(t, bytes=path.stat().st_size, optimizer_parameter_slots=optimizer_slots))
            t, _ = timing(lambda: mod("checkpoint").save(path, payload), repeats=1, warmup=0)
            write("checkpoint_four_episodes_save", dict(t, bytes=path.stat().st_size, includes_fsync_and_hash=True))
            t, _ = timing(lambda: mod("checkpoint").load(path, binding), repeats=1, warmup=0)
            write("checkpoint_four_episodes_load", t)
            t, _ = timing(lambda: mod("checkpoint").atomic_json(Path(directory) / "metrics.json", episode.metrics))
            write("episode_metrics_json", t)
            t, _ = timing(lambda: mod("run").append(Path(directory) / "learning.jsonl", dict(episode=1, example_metrics=[dict(loss=1., norm=.1)] * 3)))
            write("learning_jsonl_append", t)
        # Complete evaluation/collection with deterministic scripted actions
        # isolates physical/report/serialization cost from neural inference.
        scene = json.loads((ROOT / "hrl_v2/configs/physical_scene.json").read_text())
        for store in (True, False):
            if not store and not hasattr(mod("collector"), "collect_metrics"):
                continue
            adapter = mod("physics_adapter").PhysicsAdapter(scene)
            adapter.reset(arrival_seed=1, layout_seed=2, load_seed=3)
            kwargs = dict(upper_version="benchmark", lower_version=0, identity={})
            fn = (lambda: mod("collector").collect(adapter, Scripted(), **kwargs)) if store else (
                lambda: mod("collector").collect_metrics(adapter, Scripted(), identity={}))
            t, result = timing(fn, repeats=1, warmup=0)
            write("scripted_collection_9000" if store else "streaming_evaluation_9000", dict(t, physics_steps=9000,
                includes="physics, reports, policy interface, metrics and requested episode serialization; no neural work",
                cost_sum=(result.metrics if store else result)["cost_sum"]))

    record["status"]="complete"
    mod("checkpoint").atomic_json(args.output,record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--make-fixture", action="store_true")
    parser.add_argument("--fixture", type=Path, default=Path.cwd() / "data/fixtures/independent_fixture.pkl")
    parser.add_argument("--variant", choices=("baseline", "optimized"), default="optimized")
    parser.add_argument("--output", type=Path, default=Path.cwd() / "outputs/benchmark.json")
    parser.add_argument("--sections", default="all")
    parser.add_argument("--microbatches", type=int, nargs="+", default=[16])
    parser.add_argument("--full-ppo",action="store_true")
    args = parser.parse_args()
    if args.make_fixture:
        value = make_fixture(args.fixture)
        print(json.dumps({k: v for k, v in value.items() if k != "episode"}, ensure_ascii=False))
    else:
        measure(args)


if __name__ == "__main__":
    main()
