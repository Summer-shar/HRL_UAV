"""V4 identity and admission state over the existing complete agent serializer."""
import copy
import json
import os
from pathlib import Path
import re
import shutil
import time
import uuid

from .industrial_deployment_checkpoint import (capture_rng, restore_rng, _sha,
    _atomic_json, _verify_entry, make_checkpoint as _make_agent_checkpoint,
    restore_checkpoint as _restore_agent_checkpoint, CHECKPOINT_SCHEMA as AGENT_SCHEMA)

CHECKPOINT_SCHEMA = "industrial_hotspot_checkpoint_v4"
INDEX_SCHEMA = "industrial_hotspot_checkpoint_index_v4"
GENERATION = re.compile(r"^hotspot-[0-9]+-[a-f0-9]{32}\.pt$")
ENDPOINT = re.compile(r"^(?:candidate-(?:pretrain|joint)-[0-9]{4,}|pretrain_final|final|needs_review)\.pt$")


def _validate(payload):
    if payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA or payload.get("resumable") is not True:
        raise ValueError("A complete hotspot V4 checkpoint is required; older contracts cannot resume")
    if payload.get("protocol", {}).get("schema") != "industrial_hotspot_v4" or payload.get("scene", {}).get("schema") != "industrial_hotspot_scene_v4":
        raise ValueError("Hotspot physical and experiment identities are missing")
    state = payload.get("run_state", {})
    required = {"at_stream_boundary", "gate", "pending_evaluation", "stage", "episodes", "schedule", "startup_evidence"}
    if not required <= state.keys() or state["at_stream_boundary"] is not True:
        raise ValueError("Incomplete hotspot episode/admission continuation state")
    if state["stage"] != payload.get("stage"):
        raise ValueError("Runner and learner stages disagree")


def make_checkpoint(agent, protocol, scene, identity, run_state):
    # The agent architecture/observation serializer remains deployment V3; the
    # envelope, scene and experiment below are a distinct, strictly checked V4.
    result = _make_agent_checkpoint(agent, protocol, scene, identity, run_state)
    result["checkpoint_schema"] = CHECKPOINT_SCHEMA
    result["agent_serialization_schema"] = AGENT_SCHEMA
    _validate(result)
    return result


def restore_checkpoint(agent, saved, *, protocol, scene, identity=None, restore_random=True):
    _validate(saved)
    if saved.get("agent_serialization_schema") != AGENT_SCHEMA:
        raise ValueError("Unsupported underlying agent serialization")
    # Identity is checked before adapting only the serializer's version field.
    if saved["protocol"] != protocol or saved["scene"] != scene or (identity is not None and saved["identity"] != identity):
        raise ValueError("Hotspot protocol, physical scene, source, seed or method differs")
    adapted = dict(saved)
    adapted["checkpoint_schema"] = AGENT_SCHEMA
    return _restore_agent_checkpoint(agent, adapted, protocol=protocol, scene=scene,
        identity=identity, restore_random=restore_random)


def save_checkpoint(directory, payload, *, persistent_name=None):
    """Publish durable immutable data first, then one atomic two-generation index."""
    import torch
    _validate(payload)
    directory = Path(directory).absolute()
    directory.mkdir(parents=True, exist_ok=True)
    directory = directory.resolve()
    if persistent_name is not None and not ENDPOINT.fullmatch(persistent_name):
        raise ValueError("Unsupported hotspot endpoint name")
    index_path = directory / "latest.json"
    previous = None
    if index_path.exists():
        old = json.loads(index_path.read_text(encoding="utf-8"))
        if old.get("schema") != INDEX_SCHEMA:
            raise ValueError("Checkpoint directory contains another contract")
        _verify_entry(directory, old["latest"])
        previous = old["latest"]
    filename = f"hotspot-{time.time_ns()}-{uuid.uuid4().hex}.pt"
    path, temporary = directory / filename, directory / ("." + filename + ".tmp")
    try:
        with temporary.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        sha = _sha(temporary)
        os.replace(temporary, path)
        latest = dict(filename=filename, sha256=sha)
        _atomic_json(path.with_name(path.name + ".sha256.json"), latest)
        _verify_entry(directory, latest)
        persistent_path = None
        if persistent_name is not None:
            persistent_path = directory / persistent_name
            if persistent_path.exists():
                if _sha(persistent_path) != sha:
                    raise FileExistsError("Immutable checkpoint endpoint already exists")
            else:
                copy_path = directory / ("." + persistent_name + ".tmp")
                try:
                    with path.open("rb") as source, copy_path.open("xb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                        target.flush()
                        os.fsync(target.fileno())
                    if _sha(copy_path) != sha:
                        raise IOError("Checkpoint copy integrity failure")
                    os.replace(copy_path, persistent_path)
                finally:
                    if copy_path.exists():
                        copy_path.unlink()
            _atomic_json(persistent_path.with_name(persistent_path.name + ".sha256.json"),
                dict(filename=persistent_name, sha256=sha))
        index = dict(schema=INDEX_SCHEMA, latest=latest, previous=previous)
        _atomic_json(index_path, index)
        keep = {filename} | ({previous["filename"]} if previous else set())
        for item in directory.iterdir():
            if GENERATION.fullmatch(item.name) and item.name not in keep:
                if item.resolve().parent != directory:
                    raise ValueError("Refusing cleanup outside the checkpoint directory")
                item.unlink()
                sidecar = item.with_name(item.name + ".sha256.json")
                if sidecar.exists():
                    if sidecar.resolve().parent != directory:
                        raise ValueError("Refusing sidecar cleanup outside its directory")
                    sidecar.unlink()
        return dict(path=str(path), sha256=sha, index=str(index_path), latest=latest,
            previous=previous, persistent_path=str(persistent_path) if persistent_path else None)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path):
    """Load trusted local artifacts after integrity and V4-envelope checks."""
    import torch
    path = Path(path).absolute()
    if path.suffix == ".json":
        index = json.loads(path.read_text(encoding="utf-8"))
        if index.get("schema") != INDEX_SCHEMA:
            raise ValueError("Hotspot checkpoint index required")
        entry = index["latest"]
        path = _verify_entry(path.parent, entry)
    else:
        sidecar = path.with_name(path.name + ".sha256.json")
        if not sidecar.is_file():
            raise ValueError("Checkpoint SHA256 sidecar missing")
        entry = json.loads(sidecar.read_text(encoding="utf-8"))
        if entry.get("filename") != path.name:
            raise ValueError("Checkpoint sidecar names another file")
        _verify_entry(path.parent, entry)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _validate(payload)
    payload["_load_metadata"] = dict(path=str(path.resolve()), sha256=entry["sha256"])
    return payload
