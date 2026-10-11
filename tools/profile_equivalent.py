"""Fixed-workload measurements, never an entry point for research training.

Generate once (18,000 physical steps), then run before/after against the same
artifact and initial_state.pt. Run does NOT advance the physical environment.
Examples (from the repository root):
  .venv/Scripts/python tools/profile_equivalent.py generate --output outputs/perf_fixture
  .venv/Scripts/python tools/profile_equivalent.py run --fixtures outputs/perf_fixture --output outputs/perf_baseline --label baseline
  .venv/Scripts/python tools/profile_equivalent.py run --fixtures outputs/perf_fixture --output outputs/perf_current --label current --initial-state outputs/perf_baseline/initial_state.pt

Four replay objects and five PPO objects duplicate TWO physical trajectories;
they are explicitly structural fixtures, not independent 900-second scenarios.
Evaluation replay times inference on fixed reports, NOT closed-loop evaluation.
"""
import argparse
import copy
from contextlib import contextmanager
from dataclasses import asdict
import fnmatch
import functools
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import statistics
import sys
import threading
import time

SCHEMA = "fixed_performance_workloads_1"
ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                   allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            result.update(block)
    return result.hexdigest()


def source_identity():
    from hrl_v2.paths import source_root
    base = source_root()
    return {str(p.relative_to(base)).replace("\\", "/"): sha(p)
            for package in ("hrl_v2", "hrl_uav")
            for p in sorted((base / package).glob("*.py"))}


def read_settings():
    from hrl_v2.paths import default_config, source_root
    from hrl_v2.run import read_config
    config = read_config(default_config(), source_root())
    scene = json.loads((source_root() / "hrl_v2/configs/physical_scene.json").read_text(encoding="utf-8"))
    return config, scene


class Script:
    """Fixed hover and local/equal routing; no model calls during generation."""
    def __init__(self, route):
        self.route = route

    def upper(self, report, history):
        import numpy as np
        return np.zeros(6), {}

    def lower(self, report, history):
        import numpy as np
        from hrl_v2.hybrid_simplex import SUPPORTS
        action = (np.tile(np.eye(3), (2, 1, 1)) if self.route == "local"
                  else np.full((2, 3, 3), 1 / 3))
        support = [SUPPORTS.index(tuple(np.flatnonzero(row > 0)))
                   for row in action.reshape(6, 3)]
        return action.astype(np.float64), np.asarray(support)


def graph_description(report):
    from hrl_v2.features import build
    graph = build(report)
    lengths = {kind: [len(q) for q in queues] for kind, queues in graph.queues.items()}
    return dict(tick=report.tick, history_cursor=report.history_cursor,
                nodes=len(graph.neighbors), edges=sum(map(len, graph.edges.values())),
                parents=len(report.parents), fragments=len(report.fragments),
                queue_lengths=lengths, queued_items=sum(sum(v) for v in lengths.values()))


def generate(args):
    import numpy as np
    from hrl_uav.industrial_hotspot_env import HotspotIndustrialEnv
    from hrl_uav.industrial_hotspot_scene import HotspotIndustrialScene
    from hrl_v2.collector import collect
    from hrl_v2.physics_adapter import PhysicsAdapter
    from hrl_v2.randomness import derive
    from hrl_v2.timing import ClockConfig
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("Generation output must be empty; do not silently repeat physical work")
    out.mkdir(parents=True, exist_ok=True)
    config, scene = read_settings()
    timing = ClockConfig.from_scene(scene, rollout_seconds=900)
    if timing.physical_steps * 2 != 18000 or timing.lower_steps != 1800:
        raise RuntimeError("This fixture requires the unchanged 0.1/0.5/5-second physical clocks")
    budget = dict(limit=18000, reserved_steps=18000, physical_steps=0,
                  status="running", cases=[])
    write_json(out / "physical_budget.json", budget)
    episodes, tapes, trajectories = [], [], []
    original = HotspotIndustrialEnv.micro_step
    component_times = {}
    wrappers = []
    def cpu_timed(owner, name, label):
        function = getattr(owner, name)
        @functools.wraps(function)
        def measured(*a, **kw):
            start = time.perf_counter()
            try:
                return function(*a, **kw)
            finally:
                row = component_times.setdefault(label, dict(seconds=0., calls=0))
                row["seconds"] += time.perf_counter() - start
                row["calls"] += 1
        wrappers.append((owner, name, function))
        setattr(owner, name, measured)
    active = None

    def counted(env, route):
        if active is None or budget["physical_steps"] >= budget["reserved_steps"]:
            raise RuntimeError("Unreserved physical advance")
        before = env.step_index
        start = time.perf_counter()
        try:
            return original(env, route)
        finally:
            advance = env.step_index - before
            row = component_times.setdefault("physical.micro_step", dict(seconds=0., calls=0))
            row["seconds"] += time.perf_counter() - start
            row["calls"] += 1
            budget["physical_steps"] += advance
            active["physical_steps"] += advance
            if budget["physical_steps"] % 1000 == 0:
                write_json(out / "physical_budget.json", budget)

    HotspotIndustrialEnv.micro_step = counted
    for name in ("step", "prepare", "report"):
        cpu_timed(PhysicsAdapter, name, "adapter." + name)
    try:
        for number, route in enumerate(("local", "equal"), 1):
            seeds = {k + "_seed": derive(args.seed, 71, 0, number, purpose=k)
                     for k in ("arrival", "layout", "load")}
            # Materialize exogenous events before selecting or executing controls.
            external = HotspotIndustrialScene(scene, seeds["arrival_seed"],
                                               seeds["layout_seed"], seeds["load_seed"])
            tape = [external.frame(i) for i in range(timing.lower_steps + 1)]
            tapes.append(tape)
            adapter = PhysicsAdapter(scene, timing=timing)
            adapter.reset(**seeds)
            actual_frame = adapter.env._scene.frame
            seen = []

            def fixed_frame(index):
                if index != len(seen) or index >= len(tape):
                    raise AssertionError("External tape consumed out of order")
                # Advance the scene's audit digests too; equality ensures the
                # fixed tape is exactly its configured exogenous process.
                if actual_frame(index) != tape[index]:
                    raise AssertionError("Exogenous tape and configured generator differ")
                seen.append(index)
                return tape[index]

            adapter.env._scene.frame = fixed_frame
            active = dict(number=number, route=route, seeds=seeds,
                          reserved_steps=timing.physical_steps, physical_steps=0)
            budget["cases"].append(active)
            write_json(out / "physical_budget.json", budget)
            component_times.clear()
            collection_start = time.perf_counter()
            episode = collect(adapter, Script(route), upper_version="profile-script",
                              lower_version=0, identity=dict(profile=number, script=route, seeds=seeds),
                              reward_scale=config["objective"]["reward_scale"],
                              gamma_lower=config["objective"]["gamma_micro"])
            collection_seconds = time.perf_counter() - collection_start
            if active["physical_steps"] != timing.physical_steps or len(seen) != 1801:
                raise AssertionError("Physical or arrival-frame count mismatch")
            episodes.append(episode)
            # Queue counts are read directly; graph expansion is deferred until
            # selected samples, avoiding 3,600 unnecessary feature builds.
            loads = [sum(len(q.items) for q in episode.report(i).queues)
                     for i in episode.observations]
            trajectories.append(dict(active, lower_rows=len(episode.rewards), upper_rows=len(episode.high),
                history_frames=len(adapter.arrivals), queued_items=loads,
                collection_wall_s=collection_seconds, component_times=copy.deepcopy(component_times),
                timing_note="Single CPU scripted collection, no learned inference or updates. Inclusive nested timers; collection includes episode compression/seal and periodic budget JSON writes. Not a warmed three-repeat estimate.",
                queue_percentiles={str(q): float(np.percentile(loads, q)) for q in (0, 25, 50, 75, 90, 95, 100)},
                tape_sha256=hashlib.sha256(pickle.dumps(tape, protocol=5)).hexdigest()))
            active = None
            print(json.dumps(dict(generated=number, physical_steps=budget["physical_steps"])), flush=True)
        artifact = dict(schema=SCHEMA, seed=args.seed, config=config, scene=scene,
                        episodes=episodes, tapes=tapes, trajectories=trajectories)
        with (out / "workloads.pkl").open("wb") as stream:
            pickle.dump(artifact, stream, protocol=5)
        budget["status"] = "complete"
        write_json(out / "manifest.json", dict(schema=SCHEMA, seed=args.seed,
            artifact_sha256=sha(out / "workloads.pkl"), source=source_identity(),
            physical_steps=budget["physical_steps"], trajectories=trajectories,
            limitations=["Two independent physical trajectories only.",
                "Four replay objects and five PPO objects reuse these trajectories.",
                "Normal/backlog mean empirical queue-count percentile bands, not configured arrival regimes."]))
    except BaseException:
        budget["status"] = "failed"
        raise
    finally:
        HotspotIndustrialEnv.micro_step = original
        for owner, name, function in reversed(wrappers):
            setattr(owner, name, function)
        write_json(out / "physical_budget.json", budget)


def repair_rewards(args):
    """Re-seal reward fields algebraically; never construct a physical world."""
    import numpy as np
    from hrl_v2.replay import unpack
    directory = args.fixtures.resolve()
    path = directory / "workloads.pkl"
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    before = sha(path)
    if before != manifest["artifact_sha256"]:
        raise ValueError("Cannot repair an artifact with an unverified digest")
    if (directory / "ppo_behavior.pkl").exists():
        raise RuntimeError("Repair must precede profiling and fixed PPO behavior creation")
    with path.open("rb") as stream:
        artifact = pickle.load(stream)
    gamma = artifact["config"]["objective"]["gamma_micro"]
    changed = []
    for index, episode in enumerate(artifact["episodes"]):
        old_gamma = episode.gamma_lower
        span = len(episode.rewards) // len(episode.high)
        discounts = gamma ** np.arange(span, dtype=np.float64)
        for j, row in enumerate(episode.high):
            lower = np.asarray(episode.rewards[j*span:(j+1)*span], dtype=np.float64)
            row["reward_physical"] = float(lower.sum())
            row["reward_learning"] = float(discounts @ lower)
        episode.gamma_lower = gamma
        episode.seal(unpack(episode.history), episode.metrics)
        changed.append(dict(episode=index, from_gamma=old_gamma, to_gamma=gamma,
                            resealed=True, physical_steps=0))
    backup = directory / "workloads.before_reward_repair.pkl"
    if backup.exists():
        raise RuntimeError("Reward repair backup already exists; do not silently repeat migration")
    # Preserve the original bytes and their original manifest before replacement.
    import shutil
    shutil.copyfile(path, backup)
    shutil.copyfile(manifest_path, directory / "manifest.before_reward_repair.json")
    temporary = directory / "workloads.repaired.tmp"
    with temporary.open("wb") as stream:
        pickle.dump(artifact, stream, protocol=5)
    os.replace(temporary, path)
    manifest["artifact_sha256"] = sha(path)
    manifest["reward_repair"] = dict(before_sha256=before, after_sha256=sha(path),
        episodes=changed, physical_steps=0, method="Recompute both upper reward fields from preserved lower rewards and re-run Episode.seal")
    write_json(manifest_path, manifest)
    write_json(args.output / "reward_repair.json", manifest["reward_repair"])
    print(json.dumps(manifest["reward_repair"]), flush=True)


def collection_overhead(args):
    """One stored and one streamed 60-second scripted run: 1,200 steps."""
    from hrl_uav.industrial_hotspot_env import HotspotIndustrialEnv
    from hrl_v2.collector import collect, collect_metrics
    from hrl_v2.physics_adapter import PhysicsAdapter
    from hrl_v2.randomness import derive
    from hrl_v2.timing import ClockConfig
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("Overhead measurement output must be empty")
    out.mkdir(parents=True, exist_ok=True)
    config, scene = read_settings()
    timing = ClockConfig.from_scene(scene, rollout_seconds=60)
    expected = 2 * timing.physical_steps
    if expected != 1200:
        raise ValueError("Unexpected physical clocks")
    budget = dict(reserved_steps=expected, physical_steps=0, status="running", cases=[])
    write_json(out / "physical_budget.json", budget)
    original = HotspotIndustrialEnv.micro_step
    active = None
    def counted(env, route):
        if active is None or budget["physical_steps"] >= expected:
            raise RuntimeError("Unreserved physical advance")
        before = env.step_index
        start = time.perf_counter()
        try:
            return original(env, route)
        finally:
            active["physical_seconds"] += time.perf_counter() - start
            advance = env.step_index - before
            active["physical_steps"] += advance
            budget["physical_steps"] += advance
    HotspotIndustrialEnv.micro_step = counted
    try:
        for mode in ("stored", "streamed"):
            adapter = PhysicsAdapter(scene, timing=timing)
            adapter.reset(**{k + "_seed": derive(args.seed, 73, 0, 1, purpose=k)
                             for k in ("arrival", "layout", "load")})
            active = dict(mode=mode, seconds=60, physical_steps=0, physical_seconds=0., components={})
            budget["cases"].append(active)
            # Per-instance report/prepare/step timing; all labels are inclusive.
            for name in ("step", "prepare", "report"):
                original_method = getattr(adapter, name)
                def measured(*a, _method=original_method, _name=name, **kw):
                    start = time.perf_counter()
                    try:
                        return _method(*a, **kw)
                    finally:
                        row = active["components"].setdefault(_name, dict(seconds=0., calls=0))
                        row["seconds"] += time.perf_counter() - start
                        row["calls"] += 1
                setattr(adapter, name, measured)
            start = time.perf_counter()
            if mode == "stored":
                episode = collect(adapter, Script("equal"), upper_version="profile-script", lower_version=0,
                    identity=dict(profile="overhead"), reward_scale=config["objective"]["reward_scale"],
                    gamma_lower=config["objective"]["gamma_micro"])
                metrics = episode.metrics
                storage = episode.storage_bytes
            else:
                metrics = collect_metrics(adapter, Script("equal"), identity=dict(profile="overhead"),
                                          reward_scale=config["objective"]["reward_scale"])
                storage = 0
            active.update(wall_s=time.perf_counter()-start, storage_bytes=storage,
                          cost_sum=metrics["cost_sum"], lower_decisions=120, upper_decisions=12)
            if active["physical_steps"] != timing.physical_steps:
                raise AssertionError("Physical count mismatch")
            write_json(out / "physical_budget.json", budget)
        if budget["cases"][0]["cost_sum"] != budget["cases"][1]["cost_sum"]:
            raise AssertionError("Stored and streamed scripted paths diverged")
        budget["status"] = "complete"
        write_json(out / "collection_overhead.json", dict(schema=SCHEMA, source=source_identity(),
            physical_steps=budget["physical_steps"], cases=budget["cases"],
            limitation="Single 60-second CPU script per mode; same external seeds and routing. This is not a 900-second measurement, and extrapolation can miss queue/history growth. Inclusive component timers must not be added."))
    except BaseException:
        budget["status"] = "failed"
        raise
    finally:
        HotspotIndustrialEnv.micro_step = original
        write_json(out / "physical_budget.json", budget)


def samples_for(artifact, count, prefix, load):
    import numpy as np
    from hrl_v2.replay import Sample, probabilities
    episodes = artifact["episodes"]
    # Copy objects, not new trajectories. Replay/history grouping is by object
    # identity, so four objects exercise four independent encoder tapes.
    selected = [copy.deepcopy(episodes[i % 2]) for i in range(count)]
    rng = np.random.default_rng(artifact["seed"] + count * 10000 + prefix)
    samples, descriptions, selections = [], [], []
    for slot, episode in enumerate(selected):
        loads = np.asarray(artifact["trajectories"][slot % 2]["queued_items"][:prefix - 1])
        if len(loads) != prefix - 1:
            raise ValueError("Fixture cannot cover requested causal prefix")
        lo, hi = np.percentile(loads, (40, 60) if load == "normal" else (90, 100))
        pool = np.flatnonzero((loads >= lo) & (loads <= hi))
        if not len(pool):
            raise ValueError("Empty load-percentile stratum")
        ids = rng.choice(pool, size=128 // count, replace=True).tolist()
        # Explicit maximum-prefix anchor prevents a nominal h1800 case from
        # accidentally benchmarking only early observations in a low-load band.
        ids[-1] = prefix - 2
        q = probabilities(episode.severe)
        samples.extend(Sample(episode, int(i), float((1 / len(q)) / q[i])) for i in ids)
        unique = sorted(set(ids))
        descriptions.extend(graph_description(episode.report(episode.observations[i])) for i in unique)
        selections.append(dict(object=slot, physical_source=slot % 2, indices=ids,
                               queue_band=[float(lo), float(hi)], max_prefix_anchor=prefix - 2))
    return samples, dict(batch_size=128, episode_objects=count,
        independent_physical_trajectories=min(count, 2), synthetic=count > 2,
        max_successor_prefix=prefix, selection=load, selections=selections,
        report_graphs=descriptions,
        sampling_note="Fixed stratified stress sample plus one prefix anchor per object; not an unbiased replay estimate.")


def clone_cpu(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: clone_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_cpu(v) for v in value)
    return copy.deepcopy(value)


def rss_bytes():
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except ImportError:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                    (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
            value = Counters()
            value.cb = ctypes.sizeof(value)
            kernel = ctypes.windll.kernel32
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
            get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            get_memory.restype = wintypes.BOOL
            if not get_memory(kernel.GetCurrentProcess(), ctypes.byref(value), value.cb):
                raise OSError("GetProcessMemoryInfo failed")
            return int(value.WorkingSetSize)
        # Linux current RSS, rather than process-lifetime high-water mark.
        return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")


@contextmanager
def memory_sample():
    import torch
    stopped = threading.Event()
    state = dict(rss_before=rss_bytes(), rss_peak=0, polling_seconds=.01)
    state["rss_peak"] = state["rss_before"]
    def watch():
        while not stopped.wait(.01):
            state["rss_peak"] = max(state["rss_peak"], rss_bytes())
    thread = threading.Thread(target=watch, daemon=True)
    torch.cuda.reset_peak_memory_stats()
    thread.start()
    try:
        yield state
    finally:
        stopped.set()
        thread.join()
        state.update(rss_after=rss_bytes(), cuda_peak_allocated=torch.cuda.max_memory_allocated(),
                     cuda_peak_reserved=torch.cuda.max_memory_reserved())
        state["rss_peak"] = max(state["rss_peak"], state["rss_after"])


@contextmanager
def trace_scopes(learner):
    """Labels only in a short profiler run, never in wall-clock measurements."""
    import torch
    from hrl_v2 import batching, prepared, replay, sac, ppo, networks, features
    edits = []
    def wrap(owner, name, label):
        original = getattr(owner, name)
        @functools.wraps(original)
        def measured(*args, **kwargs):
            with torch.profiler.record_function(label):
                return original(*args, **kwargs)
        edits.append((owner, name, original))
        setattr(owner, name, measured)
    try:
        for owner, name, label in (
            (replay, "unpack", "data/replay_unpack"),
            (ppo, "unpack", "data/ppo_unpack"),
            (prepared, "build", "data/fixed_graph_features"),
            (batching, "prepare", "data/batch_graph_and_transfer"),
            (batching, "prepare_attention", "data/attention_indices"),
            (prepared.PreparedBatch, "arrivals", "data/arrival_normalize_and_transfer"),
            (prepared.HistoryTape, "__init__", "history/prefix_forward"),
            (prepared.HistoryTape, "backward", "history/full_prefix_backward"),
            (prepared.IncrementalHistory, "get", "history/inference_cache"),
            (networks.TypedAttention, "forward", "network/typed_attention"),
            (sac, "shared_action_features", "sac/action_cast"),
            (sac, "sample", "sac/hybrid_sampling"),
            (sac, "surrogate", "sac/actor_surrogate"),
            (sac.SAC, "_critic_fit", "sac/critic_fit"),
            (sac.SAC, "_actor_fit", "sac/actor_fit"),
            (ppo.PPO, "_minibatch", "ppo/minibatch"),
            (ppo.PPO, "_kl", "ppo/exact_epoch_kl"),
            (torch.autograd, "backward", "autograd/backward"),
            (torch.optim.Adam, "step", "optimizer/adam_step"),
        ):
            wrap(owner, name, label)
        for prefix, network in network_items(learner):
            wrap(network.encoder.arrival, "forward", prefix + "/arrival_gru")
            wrap(network, "batch" if hasattr(network, "batch") else "encode_batch", prefix + "/batch_forward")
            if hasattr(network, "value_features"):
                wrap(network, "value_features", prefix + "/q_value_head")
            for name, queue in network.encoder.queues.items():
                wrap(queue, "forward", prefix + "/" + name + "_gru")
        if hasattr(learner.sac, "_critic_targets"):
            wrap(learner.sac, "_critic_targets", "sac/target_construction")
        yield
    finally:
        for owner, name, original in reversed(edits):
            setattr(owner, name, original)


def network_items(learner):
    return [("sac/" + name, getattr(learner.sac, name)) for name in
            ("actor", "q1", "q2", "target1", "target2")] + [
            ("ppo/" + name, getattr(learner.ppo, name)) for name in ("actor", "value")]


def make_ppo_fixture(artifact, learner):
    """Populate valid old-policy statistics, while labelling off-policy reports.

    The state/reward paths were generated by a hover script. Old distribution
    statistics are recomputed for this fixed initial network; this is a timed
    update workload, NOT an on-policy training batch or learning experiment.
    """
    import numpy as np
    import torch
    from hrl_v2.replay import unpack
    from hrl_v2.prepared import PreparedBatch, history_scope
    from hrl_v2.ppo import logprob
    result = []
    for source in artifact["episodes"]:
        episode = copy.deepcopy(source)
        history = unpack(episode.history)
        states = PreparedBatch([(episode.report(h[key]), history)
                                for h in episode.high for key in ("report", "successor")])
        means, stds, values = [], [], []
        with torch.no_grad(), history_scope(learner.ppo.actor.encoder, states), history_scope(learner.ppo.value.encoder, states):
            for start in range(0, len(states), 16):
                current = states[start:start + 16]
                mean, std = learner.ppo.actor.batch(current)
                value = learner.ppo.value.batch(current)
                means.extend(mean.cpu().numpy()); stds.extend(std.cpu().numpy()); values.extend(value.cpu().numpy())
        for i, row in enumerate(episode.high):
            # Script's true hover request has z=0. No actions are resimulated.
            z = np.zeros(6)
            row.update(pre_tanh=z, mean=means[2*i], logstd=stds[2*i],
                logprob=float(logprob(torch.as_tensor(z), torch.as_tensor(means[2*i]), torch.as_tensor(stds[2*i]))),
                value=float(values[2*i]), next_value=float(values[2*i+1]))
        result.append(episode)
    batch = [copy.deepcopy(result[i]) for i in (0, 1, 0, 1, 0)]
    if sum(len(e.high) for e in batch) != 900:
        raise ValueError("Formal-size PPO fixture must have 900 macro rows")
    return batch


def initialize_optimizer_state(artifact, learner):
    """Bounded calibration updates; no simulation and no research run."""
    import numpy as np
    import torch
    from hrl_v2.randomness import Streams
    from hrl_v2.replay import unpack
    from hrl_v2.prepared import PreparedBatch, history_scope
    from hrl_v2.ppo import logprob
    streams = Streams(artifact["seed"], 74, 0, 1)
    samples, description = samples_for(artifact, 2, 120, "normal")
    start = time.perf_counter()
    learner.sac._critic_fit(samples, .0005, streams)
    learner.sac._actor_fit(samples, .0005, streams)
    states, following, rows = [], [], []
    for episode in artifact["episodes"]:
        high = copy.deepcopy(episode.high[11])
        history = unpack(episode.history)
        states.append((episode.report(high["report"]), history))
        following.append((episode.report(high["successor"]), history))
        rows.append((episode, high))
    states, following = PreparedBatch(states), PreparedBatch(following)
    with torch.no_grad(), history_scope(learner.ppo.actor.encoder, states):
        means, stds = learner.ppo.actor.batch(states)
    with torch.no_grad(), history_scope(learner.ppo.value.encoder, following):
        next_values = learner.ppo.value.batch(following).cpu().tolist()
    prepared_rows = []
    for index, (episode, high) in enumerate(rows):
        z = torch.zeros_like(means[index])
        high.update(pre_tanh=z.cpu().numpy(), mean=means[index].cpu().numpy(),
                    logstd=stds[index].cpu().numpy(),
                    logprob=logprob(z, means[index], stds[index]).item())
        target = high["reward_learning"] + learner.ppo.gamma * next_values[index]
        prepared_rows.append((episode, high, float(target)))
    learner.ppo._minibatch(states, prepared_rows, np.asarray([-1., 1.]), True)
    torch.cuda.synchronize()
    return dict(kind="bounded_optimizer_initialization_not_research_training",
        physical_steps=0, seconds=time.perf_counter()-start,
        operations=dict(critic_fits=1, actor_fits=1, ppo_actor_minibatches=1, ppo_value_minibatches=1),
        ppo_rows=2, sac_workload=description,
        note="Optimizer slots are created by actual gradients. Parameters unused by these samples may still have lazy slots.")


def measure(name, setup, operation, repeats, warmup):
    import numpy as np
    import torch
    for _ in range(warmup):
        setup()
        operation()
        torch.cuda.synchronize()
    records = []
    for _ in range(repeats):
        setup()
        torch.cuda.synchronize()
        with memory_sample() as memory:
            start = time.perf_counter()
            result = operation()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
        records.append(dict(seconds=seconds, memory=memory))
        del result
    times = [x["seconds"] for x in records]
    return dict(name=name, warmup=warmup, repeats=repeats, samples=records,
                median_s=statistics.median(times), p95_s=float(np.percentile(times, 95)),
                mean_s=statistics.mean(times), min_s=min(times), max_s=max(times),
                tail_estimate_reliable=False,
                tail_note="P95 is descriptive interpolation of a small sample, not a reliable tail estimate")


def selected(name, patterns):
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def run(args):
    import numpy as np
    import torch
    from hrl_v2 import SCHEMA as CHECKPOINT_SCHEMA
    from hrl_v2.checkpoint import save, load, identity, runtime_record, atomic_json
    from hrl_v2.paths import default_config, source_root
    from hrl_v2.runtime import configure, Learner
    from hrl_v2.randomness import Streams
    from hrl_v2.replay import Replay, unpack
    from hrl_v2.policy import Policy
    from hrl_v2.prepared import fixed_graph
    from hrl_v2.metrics import PooledMetrics
    # Any accidental closed-loop path in run must fail before consuming budget.
    from hrl_uav.industrial_hotspot_env import HotspotIndustrialEnv
    def forbidden_physical_step(*a, **kw):
        raise RuntimeError("run is a zero-physical-step workload; use the budgeted generation commands")
    HotspotIndustrialEnv.micro_step = forbidden_physical_step
    fixture_path = args.fixtures.resolve() / "workloads.pkl"
    manifest = json.loads((args.fixtures / "manifest.json").read_text(encoding="utf-8"))
    if sha(fixture_path) != manifest["artifact_sha256"]:
        raise ValueError("Fixture byte digest differs from its manifest")
    with fixture_path.open("rb") as stream:
        artifact = pickle.load(stream)  # Locally generated, trusted artifact only.
    if artifact["schema"] != SCHEMA:
        raise ValueError("Unknown fixture schema")
    config, scene = read_settings()
    if config != artifact["config"] or scene != artifact["scene"]:
        raise ValueError("Baseline/current configuration and physics must match the fixture exactly")
    if any(e.gamma_lower != config["objective"]["gamma_micro"] for e in artifact["episodes"]):
        raise ValueError("Fixture upper learning rewards require repair-rewards before profiling")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    metadata_path = out / "metadata.json"
    if metadata_path.exists() and not args.resume:
        raise RuntimeError("Run output exists; choose a new directory or explicitly --resume")
    device = configure()
    learner = Learner(artifact["seed"], device, config=config)
    if hasattr(learner.sac, "critic_microbatch_size"):
        learner.sac.critic_microbatch_size = args.critic_microbatch
    elif args.critic_microbatch != 16:
        raise ValueError("This source has no independent critic microbatch control; baseline supports only 16")
    initial_path = args.initial_state or (out / "initial_state.pt")
    if initial_path.exists():
        initial = torch.load(initial_path, map_location="cpu", weights_only=False)
        learner.load_state_dict(copy.deepcopy(initial))
    elif args.initial_state is not None:
        raise FileNotFoundError(initial_path)
    else:
        initialization_record = (initialize_optimizer_state(artifact, learner) if args.initialize_state
                                 else dict(kind="fresh_uninitialized_adam", physical_steps=0))
        initial = clone_cpu(learner.state_dict())
        torch.save(initial, initial_path)
        write_json(initial_path.with_suffix(".initialization.json"), initialization_record)
    binding = identity(source_root(), default_config())
    metadata = dict(schema=SCHEMA, label=args.label, artifact_sha256=sha(fixture_path),
        measurement_script_sha256=sha(Path(__file__)),
        initial_state_sha256=sha(initial_path), initial_state_path=str(initial_path.resolve()),
        initial_state_preparation=(json.loads(initial_path.with_suffix(".initialization.json").read_text(encoding="utf-8"))
            if initial_path.with_suffix(".initialization.json").exists() else dict(kind="external_state_provenance_unspecified")),
        microbatches=dict(critic=args.critic_microbatch, target=learner.sac.microbatch_size,
                          actor=learner.sac.microbatch_size, ppo=learner.ppo.microbatch_size),
        runtime=runtime_record(), platform=platform.platform(), source=source_identity(),
        physical_steps_this_run=0, fixture_physical_steps=manifest["physical_steps"],
        deterministic=torch.are_deterministic_algorithms_enabled(),
        cpu_threads=torch.get_num_threads(), config=config, command=sys.argv,
        warmup=args.warmup, repeats=args.repeats, initial_counts={k:initial[k]["counts"] for k in ("sac", "ppo")},
        initial_optimizer_slots={f"{k}.{n}":len(initial[k][n]["state"]) for k,n in
            (("sac","actor_optimizer"),("sac","critic_optimizer"),("ppo","actor_optimizer"),("ppo","value_optimizer"))},
        methodology="Restore exact weights/optimizer/counters/RNG before every repetition; setup is excluded. Wall-clock runs have no cProfile or record_function wrappers; CUDA synchronization only at operation boundaries. RSS peak is sampled at 10 ms, so narrow peaks can be missed. Graph preparation includes Python building and transfers; identify actual copies using aten::_to_copy/copy_ and CUDA memcpy events, not the entire preparation scope.",
        limitations=manifest["limitations"] + [
            "Fresh initial state has lazy Adam slots; timings include their first allocation unless an initialized state is supplied.",
            "PPO is a synthetic update workload on scripted states, not an on-policy rollout.",
            "Evaluation replay does not execute chosen actions and cannot measure closed-loop cost or physics time.",
            "Checkpoint uses two actual replay episodes; no claim about a full 55-episode buffer."])
    if args.resume:
        old = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in ("source", "artifact_sha256", "initial_state_sha256", "label", "warmup", "repeats", "measurement_script_sha256", "microbatches"):
            if old[key] != metadata[key]:
                raise ValueError("Resume contract changed: " + key)
    write_json(metadata_path, metadata)
    results_path = out / "times.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if args.resume and results_path.exists() else {}
    streams = None
    def reset():
        nonlocal streams
        # Runtime assigns the counts dictionaries directly. Copying protects
        # the immutable reference state from updates to learner counters.
        learner.load_state_dict(copy.deepcopy(initial))
        for _, network in network_items(learner):
            network.encoder.reset_inference()
            network.zero_grad(set_to_none=True)
        fixed_graph.cache_clear()
        streams = Streams(artifact["seed"], 72, 0, 1)
        torch.manual_seed(artifact["seed"])
        torch.cuda.manual_seed_all(artifact["seed"])

    def case(name, operation, description, setup=None):
        if not selected(name, args.cases):
            return
        prepare = setup or reset
        if name not in results:
            row = measure(name, prepare, operation, args.repeats, args.warmup)
            row["workload"] = description
            row["counts_after"] = dict(sac=dict(learner.sac.counts), ppo=dict(learner.ppo.counts))
            results[name] = row
            write_json(results_path, results)
            print(json.dumps(dict(case=name, median_s=row["median_s"])), flush=True)
        trace_path = out / (name + ".trace.json")
        if selected(name, args.trace_cases) and not trace_path.exists():
            prepare()
            torch.cuda.synchronize()
            with trace_scopes(learner):
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA], record_shapes=True,
                        profile_memory=True, with_stack=False) as profiler:
                    with torch.profiler.record_function(name):
                        operation()
                    torch.cuda.synchronize()
            profiler.export_chrome_trace(str(trace_path))
            averages = profiler.key_averages(group_by_input_shape=True)
            device_key = ("self_device_time_total" if len(averages) and
                          hasattr(averages[0], "self_device_time_total") else "self_cuda_time_total")
            for label, key in (("self_cpu_time_total", "self_cpu_time_total"),
                               ("self_cuda_time_total", device_key)):
                table = averages.table(sort_by=key, row_limit=80)
                (out / (name + "." + label + ".txt")).write_text(table, encoding="utf-8")
            results[name]["trace"] = str(trace_path)
            write_json(results_path, results)

    for prefix in (120, 600, 1800):
        for count in (1, 2, 4):
            for load_band in ("normal", "backlog"):
                stem = f"e{count}.h{prefix}.{load_band}"
                if not any(selected(k + "." + stem, args.cases) for k in ("critic", "actor")):
                    continue
                samples, description = samples_for(artifact, count, prefix, load_band)
                case("critic." + stem, lambda: learner.sac._critic_fit(samples, .0005, streams), description)
                case("actor." + stem, lambda: learner.sac._actor_fit(samples, .0005, streams), description)

    for prefix in (1, 120, 600, 1800):
        episode = artifact["episodes"][1]
        history = unpack(episode.history)
        current = episode.report(episode.observations[prefix - 1])
        previous = episode.report(episode.observations[prefix - 2])
        policy = None
        def inference_setup(cached=False):
            nonlocal policy
            reset()
            policy = Policy(learner.sac.actor, learner.ppo.actor, learner.ppo.value, streams)
            if cached:
                before = copy.deepcopy(streams.state_dict())
                policy.lower(previous, history)
                streams.load_state_dict(before)
        for mode in (("cached", "rebuild") if prefix > 1 else ("rebuild",)):
            case(f"inference.lower.h{prefix}.{mode}", lambda: policy.lower(current, history),
                 dict(report=graph_description(current), history_prefix=prefix,
                      cache_mode=mode, physical_source=1),
                 setup=lambda cached=(mode == "cached"): inference_setup(cached))

    for prefix in (0, 120, 600, 1800):
        episode = artifact["episodes"][1]
        history = unpack(episode.history)
        index = prefix // 10
        current = episode.report(episode.high[index]["report"] if index < 180
                                 else episode.high[-1]["successor"])
        previous = episode.report(episode.high[max(0, index - 1)]["report"])
        upper_policy = None
        def upper_setup(cached=False, with_value=False):
            nonlocal upper_policy
            reset()
            upper_policy = Policy(learner.sac.actor, learner.ppo.actor,
                learner.ppo.value if with_value else None, streams)
            if cached:
                before = copy.deepcopy(streams.state_dict())
                upper_policy.upper(previous, history)
                streams.load_state_dict(before)
        for with_value in (False, True):
            for mode in (("cached", "rebuild") if prefix else ("rebuild",)):
                kind = "upper_value" if with_value else "upper"
                case(f"inference.{kind}.h{prefix}.{mode}", lambda: upper_policy.upper(current, history),
                     dict(report=graph_description(current), history_prefix=prefix,
                          delta_history=10 if mode == "cached" else prefix,
                          includes_value=with_value, cache_mode=mode, physical_source=1),
                     setup=lambda cached=(mode == "cached"), value=with_value: upper_setup(cached, value))

    # Position/history caps and physical queue loads are distinct dimensions.
    # These are unmodified REAL reports in empirical load strata; report_cursor
    # is recorded and may be earlier than the nominal prefix cap.
    episode = artifact["episodes"][1]
    history = unpack(episode.history)
    for cap in (120, 600, 1800):
        loads = np.asarray(artifact["trajectories"][1]["queued_items"][:cap])
        for band in ("normal", "backlog"):
            lo, hi = np.percentile(loads, (40,60) if band == "normal" else (90,100))
            candidates = np.flatnonzero((loads >= lo) & (loads <= hi) & (np.arange(len(loads)) > 0))
            if not len(candidates):
                raise ValueError("No real inference report in requested queue stratum")
            index = int(candidates[-1])
            current = episode.report(episode.observations[index])
            previous = episode.report(episode.observations[index-1])
            stratum_policy = None
            def stratum_setup(cached=False):
                nonlocal stratum_policy
                reset()
                stratum_policy = Policy(learner.sac.actor, learner.ppo.actor, None, streams)
                if cached:
                    before = copy.deepcopy(streams.state_dict())
                    stratum_policy.lower(previous, history)
                    streams.load_state_dict(before)
            for mode in ("cached", "rebuild"):
                case(f"inference.lower.h{cap}.{band}.{mode}",
                    lambda: stratum_policy.lower(current, history),
                    dict(report=graph_description(current), requested_prefix_cap=cap,
                         actual_history_prefix=current.history_cursor, physical_source=1,
                         load_band=band, queue_count=int(loads[index]),
                         queue_band=[float(lo),float(hi)],
                         empirical_queue_percentile=float(100*np.mean(loads <= loads[index])),
                         cache_mode=mode, synthetic=False),
                    setup=lambda cached=(mode == "cached"): stratum_setup(cached))

    if selected("inference.lower.h1800.after_actor_update", args.cases):
        update_samples, update_description = samples_for(artifact, 2, 120, "normal")
        episode = artifact["episodes"][1]
        history = unpack(episode.history)
        current = episode.report(episode.observations[1799])
        changed_policy = None
        changed_description = dict(report=graph_description(current), physical_source=1,
            setup_update=update_description, setup_excluded=True,
            mechanism="Prime long cache, perform actual actor optimizer step on fixed h120/e2 samples, infer same long report")
        def changed_setup():
            nonlocal changed_policy
            reset()
            changed_policy = Policy(learner.sac.actor, learner.ppo.actor, None, streams)
            random_before = copy.deepcopy(streams.state_dict())
            changed_policy.lower(current, history)
            streams.load_state_dict(random_before)
            encoder = learner.sac.actor.encoder
            before = tuple(p._version for p in encoder.parameters())
            learner.sac._actor_fit(update_samples, .0005, streams)
            after = tuple(p._version for p in encoder.parameters())
            if before == after or encoder._incremental_history.cursor != current.history_cursor:
                raise AssertionError("Expected populated stale cache after actor update")
            changed_description.update(parameter_versions_changed=True,
                                       cache_cursor_before_inference=encoder._incremental_history.cursor)
        case("inference.lower.h1800.after_actor_update", lambda: changed_policy.lower(current, history),
             changed_description, setup=changed_setup)

    ppo_batch = None
    if selected("ppo.rows900", args.cases):
        behavior_path = args.fixtures / "ppo_behavior.pkl"
        behavior_identity = dict(initial_state_sha256=sha(initial_path), artifact_sha256=sha(fixture_path))
        if behavior_path.exists():
            with behavior_path.open("rb") as stream:
                saved_behavior = pickle.load(stream)
            if saved_behavior["identity"] != behavior_identity:
                raise ValueError("PPO fixed behavior artifact belongs to a different initial state")
            ppo_batch = saved_behavior["episodes"]
        else:
            reset()
            ppo_batch = make_ppo_fixture(artifact, learner)
            with behavior_path.open("wb") as stream:
                pickle.dump(dict(identity=behavior_identity, episodes=ppo_batch), stream, protocol=5)
        case("ppo.rows900", lambda: learner.ppo.update(ppo_batch, streams),
             dict(rows=900, episode_objects=5, physical_sources=[0,1,0,1,0], synthetic=True,
                  epochs=6, minibatch=150, max_value_steps=36,
                  behavior_statistics="Fixed initial policy on hover-script states; not on-policy data"))

    checkpoint_cases = ("checkpoint.save0", "checkpoint.save2", "checkpoint.load2", "checkpoint.digest2")
    if any(selected(n, args.cases) for n in checkpoint_cases):
        replay = Replay(); replay.set_upper_version("profile-script")
        for episode in artifact["episodes"]:
            replay.add(episode)
        checkpoint_path = out / "measured_checkpoint2.pt"
        empty_replay = Replay(); empty_replay.set_upper_version("profile-script")
        def payload(buffer):
            return dict(schema=CHECKPOINT_SCHEMA, safe_episode_boundary=True, completed_episode=2,
                pending_evaluation=None, identity=binding, seed=artifact["seed"],
                learner=learner.state_dict(), replay=buffer, ppo_batch=[],
                streams=streams.state_dict(), global_cpu_rng=torch.get_rng_state(),
                global_cuda_rng=torch.cuda.get_rng_state_all(), development_results=[],
                scene=scene, config=config, runtime=runtime_record())
        case("checkpoint.save0", lambda: save(out / "measured_checkpoint0.pt", payload(empty_replay)),
             dict(replay_episodes=0, ppo_objects=0, includes="state_dict, serialization, fsync, digest and atomic sidecar"))
        case("checkpoint.save2", lambda: save(checkpoint_path, payload(replay)),
             dict(replay_episodes=2, ppo_objects=0, includes="state_dict, serialization, fsync, digest and atomic sidecar"))
        if not checkpoint_path.exists():
            reset(); save(checkpoint_path, payload(replay))
        case("checkpoint.load2", lambda: load(checkpoint_path, binding),
             dict(bytes=checkpoint_path.stat().st_size, map_location="cpu",
                  includes="identity/digest validation and deserialization; excludes loading into GPU modules"))
        case("checkpoint.digest2", lambda: sha(checkpoint_path),
             dict(bytes=checkpoint_path.stat().st_size, includes="Sequential SHA256 of local checkpoint"))

    case("source.identity", lambda: identity(source_root(), default_config()),
         dict(includes="Full production source/config/scene hashing; filesystem cache is warmed"))
    replay = Replay(); replay.set_upper_version("profile-script")
    for episode in artifact["episodes"]:
        replay.add(episode)
    case("replay.sample128", lambda: replay.sample(streams.numpy),
         dict(batch_size=128, replay_episodes=2, draws=4, samples_per_draw=32))
    def polyak():
        with torch.no_grad():
            for target, current in ((learner.sac.target1, learner.sac.q1), (learner.sac.target2, learner.sac.q2)):
                for a, b in zip(target.parameters(), current.parameters()):
                    a.lerp_(b, .005)
    case("target.polyak", polyak, dict(target_networks=2, tau=.005))
    learning_row = dict(skipped=False, critics=[dict(residual_mean=[0.,0.], residual_mse=[1.,1.],
        gradient_norm=[1.,1.], target_mean=0., target_variance=1.) for _ in range(2)],
        actor=dict(objective=0., surrogate=0., gradient_norm=1.))
    def learning_jsonl():
        with (out / "measured_learning.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(learning_row, allow_nan=False) + "\n")
    case("io.learning_jsonl", learning_jsonl,
         dict(includes="One representative SAC event JSON serialization/open/append/close", synthetic=True))
    case("io.metrics_json", lambda: atomic_json(out / "measured_metrics.json", artifact["episodes"][0].metrics),
         dict(includes="Actual episode metrics JSON serialization/fsync/atomic replace"))

    if selected("evaluation.replay900s", args.cases):
        episode = artifact["episodes"][0]
        history = unpack(episode.history)
        observations = [episode.report(i) for i in episode.observations]
        upper = {episode.report(h["report"]).tick: episode.report(h["report"]) for h in episode.high}
        successor = episode.report(episode.high[-1]["successor"])
        upper[successor.tick] = successor
        eval_policy = None
        def evaluation_setup():
            nonlocal eval_policy
            reset()
            eval_policy = Policy(learner.sac.actor, learner.ppo.actor, None, streams, mode="moving")
        def evaluation_replay():
            for report in observations:
                if report.tick in upper:
                    eval_policy.upper(upper[report.tick], history)
                eval_policy.lower(report, history)
            eval_policy.upper(successor, history)
            pooled = PooledMetrics()
            pooled.add(episode.metrics)
            result = dict(kind="fixed_report_replay_not_closed_loop", original_script_metrics=pooled.result())
            atomic_json(out / "evaluation_replay_summary.json", result)
            return result
        case("evaluation.replay900s", evaluation_replay,
             dict(physical_steps=0, lower_calls=1800, upper_calls=181,
                  includes="Frozen stochastic inference on actual reports, metric pooling and atomic JSON",
                  excludes="Actual policy-dependent physics, scene resets and report construction",
                  physical_source=0), setup=evaluation_setup)
    write_json(out / "completion.json", dict(status="complete", cases=list(results),
        physical_steps_this_run=0, fixture_sha256=sha(fixture_path)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("generate", "run", "repair-rewards", "collection-overhead"))
    parser.add_argument("--source-dir", type=Path, default=ROOT / "src",
                        help="Directory containing hrl_v2 and hrl_uav packages; allows a frozen baseline checkout")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--initial-state", type=Path)
    parser.add_argument("--initialize-state", action="store_true",
                        help="Before saving a NEW initial state, run one critic/actor fit and a two-row PPO minibatch to initialize Adam; zero physical steps")
    parser.add_argument("--critic-microbatch", type=int, choices=(16,64,128), default=16,
                        help="Independent current-Q chunk size; target and actor remain 16. Legacy baseline only supports 16")
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--seed", type=int, default=4101)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cases", nargs="+", default=["*"], help="Case-name glob patterns")
    parser.add_argument("--trace-cases", nargs="*", default=["critic.e2.h1800.backlog"],
                        help="Independent short trace cases; pass with no values to disable")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.repeats < 3 or args.warmup < 1:
        parser.error("At least one warmup and three timed repetitions are required")
    if args.command in ("run", "repair-rewards") and args.fixtures is None:
        parser.error(args.command + " requires --fixtures")
    sys.path.insert(0, str(args.source_dir.resolve()))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    {"generate": generate, "run": run, "repair-rewards": repair_rewards,
     "collection-overhead": collection_overhead}[args.command](args)


if __name__ == "__main__":
    main()
