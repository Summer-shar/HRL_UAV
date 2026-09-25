"""Full stream-boundary checkpoints with verified atomic two-generation publication.

Importing this module does not import torch.  A checkpoint is a trusted local
training artifact; its hash establishes integrity, not untrusted-pickle safety.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import uuid


CHECKPOINT_SCHEMA = "industrial_deployment_checkpoint_v3"
INDEX_SCHEMA = "industrial_deployment_checkpoint_index_v3"
_GENERATION = re.compile(r"^checkpoint-[0-9]+-[a-f0-9]{32}\.pt$")


def capture_rng():
    import random
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    import random
    import numpy as np
    import torch
    if set(state) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("Incomplete global RNG state")
    if len(state["cuda"]) != torch.cuda.device_count():
        raise ValueError("Saved/current CUDA RNG device counts differ")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, payload):
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _verify_entry(directory, entry):
    if not isinstance(entry, dict) or set(entry) != {"filename", "sha256"}:
        raise ValueError("Malformed checkpoint index entry")
    filename = entry["filename"]
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("Checkpoint index cannot escape its directory")
    path = directory / filename
    if path.resolve().parent != directory.resolve():
        raise ValueError("Checkpoint target escapes its directory")
    if not path.is_file() or _sha(path) != entry["sha256"]:
        raise ValueError(f"Checkpoint integrity check failed: {filename}")
    return path


def make_checkpoint(agent, protocol, scene, identity, run_state):
    if run_state.get("at_stream_boundary") is not True:
        raise ValueError("Only completed independent stream boundaries can be checkpointed")
    if agent.stage not in ("pretrain", "joint"):
        raise ValueError("Invalid checkpoint training stage")
    if agent.stage == "pretrain" and (agent.ppo_updates or agent.macro_rollout):
        raise ValueError("Hover pretraining cannot contain PPO updates or samples")
    if set(identity) != {"method", "training_seed", "source_sha256"}:
        raise ValueError("Checkpoint identity requires method, training_seed, source_sha256")
    result = agent.checkpoint(agent.full_config, include_replay=True)
    result.update(checkpoint_schema=CHECKPOINT_SCHEMA, resumable=True,
                  protocol=copy.deepcopy(protocol), scene=copy.deepcopy(scene),
                  identity=copy.deepcopy(identity), run_state=copy.deepcopy(run_state),
                  global_rng=capture_rng(), guard_state=agent.guard_state())
    # state_dict tensors otherwise alias the live network while replay already owns copies.
    return copy.deepcopy(result)


def _same(left, right):
    import numpy as np
    import torch
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def restore_checkpoint(agent, saved, *, protocol, scene, identity=None, restore_random=True):
    """Restore exactly; any explicit budget extension is applied by the runner afterward."""
    import numpy as np
    import torch
    if saved.get("checkpoint_schema") != CHECKPOINT_SCHEMA or saved.get("resumable") is not True:
        raise ValueError("Not a resumable deployment V3 checkpoint")
    if saved.get("protocol") != protocol or saved.get("scene") != scene:
        raise ValueError("Checkpoint physical/protocol contract differs")
    if identity is not None and saved.get("identity") != identity:
        raise ValueError("Checkpoint method, seed or source hashes differ")
    if saved.get("config") != agent.full_config:
        raise ValueError("Checkpoint effective learning configuration differs")
    if (saved.get("micro_state_dim") != agent.micro_state_dim
            or saved.get("macro_state_dim") != agent.macro_state_dim
            or saved.get("observations") != agent.observation_spec):
        raise ValueError("Checkpoint observation fields/order/scales/version differ")
    if saved.get("stage") not in ("pretrain", "joint") or saved.get("run_state", {}).get("at_stream_boundary") is not True:
        raise ValueError("Checkpoint is not a completed stream boundary")
    training = saved.get("training_state", {})
    required_training = {"stage_initial_updates", "last_sac_stats", "last_ppo_stats", "macro_rollout"}
    if not required_training <= training.keys():
        raise ValueError("Checkpoint is missing pending PPO or stage state")
    if saved["stage"] == "pretrain" and (saved["updates"]["ppo"] or training["macro_rollout"]):
        raise ValueError("Pretraining checkpoint contains PPO learning state")
    replay = saved.get("replay")
    method = saved["identity"]["method"]
    if replay is None and method == "ppo_sac":
        raise ValueError("SAC continuation requires the original experience pool")
    if replay is not None:
        required_replay = {"transitions", "memory_counter", "capacity", "transition_width", "recent_window", "recent_ratio", "rng_state"}
        if not required_replay <= replay.keys():
            raise ValueError("Incomplete replay ring or sampling RNG")
        if (replay["capacity"] != agent.replay.capacity
                or replay["transition_width"] != agent.replay.mem.shape[1]
                or replay["recent_window"] != agent.replay.recent_window
                or replay["recent_ratio"] != agent.replay.recent_ratio):
            raise ValueError("Replay capacity, row width or sampling configuration changed")
        if not np.isfinite(np.asarray(replay["transitions"])).all():
            raise ValueError("Checkpoint replay contains non-finite values")
    if len(saved.get("optimizers", [])) != 5:
        raise ValueError("Checkpoint needs all five optimizer states")
    updates = saved["updates"]
    if not all(type(updates.get(name)) is int and updates[name] >= 0 for name in (
            "sac_critic", "sac_critic_gradients", "sac_target", "sac_actor", "ppo", "ppo_gradients")):
        raise ValueError("Incomplete or invalid update clocks")
    # Validate clock schema before mutating networks.
    if set(saved.get("guard_state", {})) != set(agent.guard_state()):
        raise ValueError("Incomplete guarded PPO counters")
    networks = dict(sac_actor=agent.sac_actor.action_net, sac_critic=agent.sac_critic.critic_v,
        sac_target=agent.sac_critic.target_critic_v, macro_actor=agent.macro_actor,
        macro_critic=agent.macro_critic)
    for name, network in networks.items():
        network.load_state_dict(saved[name], strict=True)
    with torch.no_grad():
        agent.entropy.log_alpha.copy_(saved["log_alpha"].to(agent.entropy.log_alpha.device))
    agent.entropy.alpha = agent.entropy.log_alpha.exp()
    optimizers = [agent.sac_actor.optimizer, agent.sac_critic.optimizer, agent.entropy.optimizer,
                  agent.ppo_actor_optimizer, agent.ppo_critic_optimizer]
    for optimizer, state in zip(optimizers, saved["optimizers"]):
        # Keep the loaded artifact immutable while the resumed optimizer updates
        # its step counters and moments in place.
        optimizer.load_state_dict(copy.deepcopy(state))
    agent.sac_updates = updates["sac_critic"]
    agent.sac_critic_gradient_steps = updates["sac_critic_gradients"]
    agent.sac_target_updates = updates["sac_target"]
    agent.sac_actor_updates = updates["sac_actor"]
    agent.ppo_updates = updates["ppo"]
    agent.ppo_gradient_steps = updates["ppo_gradients"]
    agent.stage = saved["stage"]
    agent.stage_initial_updates = training["stage_initial_updates"]
    agent.last_sac_stats = copy.deepcopy(training["last_sac_stats"])
    agent.last_ppo_stats = copy.deepcopy(training["last_ppo_stats"])
    agent.macro_rollout = copy.deepcopy(training["macro_rollout"])
    agent.restore_guard_state(saved["guard_state"])
    if replay is not None:
        agent.replay.load_state_dict(replay, memory_counter=replay["memory_counter"])
        if not _same(agent.replay.state_dict(), replay):
            raise AssertionError("Replay restoration changed ring data, counter or RNG")
    checks = all(_same(network.state_dict(), saved[name]) for name, network in networks.items())
    if not checks or not all(_same(opt.state_dict(), state) for opt, state in zip(optimizers, saved["optimizers"])):
        raise AssertionError("Network/optimizer restoration is not exact")
    if not _same(agent.macro_rollout, training["macro_rollout"]):
        raise AssertionError("Pending PPO batch restoration is not exact")
    if restore_random:
        restore_rng(saved["global_rng"])
        if not _same(capture_rng(), saved["global_rng"]):
            raise AssertionError("Global RNG restoration is not exact")
    return copy.deepcopy(saved["run_state"])


def save_checkpoint(directory, payload, *, persistent_name=None):
    """Publish a full immutable generation and keep at most latest + previous rolling files."""
    import torch
    directory = Path(directory).absolute()
    directory.mkdir(parents=True, exist_ok=True)
    directory = directory.resolve()
    if payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA or not payload.get("resumable"):
        raise ValueError("Only full V3 payloads can be published")
    if persistent_name is not None and persistent_name not in ("pretrain_final.pt", "final.pt"):
        raise ValueError("Persistent endpoint name must be pretrain_final.pt or final.pt")
    index_path = directory / "latest.json"
    old_latest = None
    if index_path.exists():
        old_index = json.loads(index_path.read_text(encoding="utf-8"))
        if old_index.get("schema") != INDEX_SCHEMA:
            raise ValueError("Unknown checkpoint index schema")
        _verify_entry(directory, old_index["latest"])
        old_latest = old_index["latest"]
    filename = f"checkpoint-{time.time_ns()}-{uuid.uuid4().hex}.pt"
    path, temporary = directory / filename, directory / ("." + filename + ".tmp")
    try:
        with temporary.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        digest = _sha(temporary)
        os.replace(temporary, path)
        entry = dict(filename=filename, sha256=digest)
        _atomic_json(path.with_name(path.name + ".sha256.json"), entry)
        _verify_entry(directory, entry)
        index = dict(schema=INDEX_SCHEMA, latest=entry, previous=old_latest)
        _atomic_json(index_path, index)
        persistent_path = None
        if persistent_name is not None:
            persistent_path = directory / persistent_name
            if persistent_path.exists():
                if _sha(persistent_path) != digest:
                    raise FileExistsError(f"Immutable endpoint already exists: {persistent_path}")
            else:
                persistent_temporary = directory / ("." + persistent_name + "." + uuid.uuid4().hex + ".tmp")
                try:
                    with path.open("rb") as source, persistent_temporary.open("xb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                        target.flush()
                        os.fsync(target.fileno())
                    if _sha(persistent_temporary) != digest:
                        raise IOError("Persistent checkpoint copy hash mismatch")
                    os.replace(persistent_temporary, persistent_path)
                finally:
                    if persistent_temporary.exists():
                        persistent_temporary.unlink()
            _atomic_json(persistent_path.with_name(persistent_path.name + ".sha256.json"),
                         dict(filename=persistent_name, sha256=digest))
        # Only our exact immutable-generation pattern is eligible for cleanup.
        # Resolve each file immediately before deletion; persistent endpoints are excluded.
        referenced = {entry["filename"]}
        if old_latest:
            referenced.add(old_latest["filename"])
        for candidate in directory.iterdir():
            if _GENERATION.fullmatch(candidate.name) and candidate.name not in referenced:
                if candidate.resolve().parent != directory:
                    raise ValueError("Refusing checkpoint cleanup outside its directory")
                candidate.unlink()
                sidecar = candidate.with_name(candidate.name + ".sha256.json")
                if sidecar.exists():
                    if sidecar.resolve().parent != directory:
                        raise ValueError("Refusing sidecar cleanup outside its directory")
                    sidecar.unlink()
        return dict(path=str(path), sha256=digest, index=str(index_path),
                    latest=entry, previous=old_latest,
                    persistent_path=str(persistent_path) if persistent_path else None)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path):
    """Load a trusted local .pt or latest.json after verifying its SHA256."""
    import torch
    path = Path(path).absolute()
    if path.suffix == ".json":
        index = json.loads(path.read_text(encoding="utf-8"))
        if index.get("schema") != INDEX_SCHEMA:
            raise ValueError("Unknown checkpoint index schema")
        path = _verify_entry(path.parent, index["latest"])
        digest = index["latest"]["sha256"]
    else:
        sidecar = path.with_name(path.name + ".sha256.json")
        if not sidecar.is_file():
            raise ValueError("Checkpoint integrity sidecar is missing")
        entry = json.loads(sidecar.read_text(encoding="utf-8"))
        if entry.get("filename") != path.name:
            raise ValueError("Integrity sidecar names another checkpoint")
        _verify_entry(path.parent, entry)
        digest = entry["sha256"]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
        raise ValueError("Checkpoint is not deployment V3")
    payload["_load_metadata"] = dict(path=str(path.resolve()), sha256=digest)
    return payload
