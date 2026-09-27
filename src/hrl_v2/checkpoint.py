"""Atomic episode-boundary snapshots with code/config/scene identity checks."""
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import uuid
import torch
from . import SCHEMA


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def atomic_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    try:
        with temporary.open("w",encoding="utf-8",newline="\n") as f:
            json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False); f.write("\n")
            f.flush(); os.fsync(f.fileno())
        os.replace(temporary,path)
    finally:
        if temporary.exists(): temporary.unlink()


def identity(root,config_path):
    from .paths import source_root
    root=source_root()
    paths=sorted((root/"hrl_v2").glob("*.py"))
    paths += [root/"hrl_v2/design/reference_hybrid_simplex.py"]
    physical=("env.py","industrial_env.py","industrial_scene.py","industrial_hotspot_env.py",
              "industrial_hotspot_scene.py","industrial_deployment_env.py","industrial_deployment_scene.py",
              "industrial_observation.py","industrial_deployment_observation.py","industrial_baselines.py","xu_cvx_r.py","routing.py")
    paths += sorted((root/"hrl_uav").glob("*.py"))
    return dict(schema=SCHEMA,files={str(p.relative_to(root)).replace("\\","/"):digest(p) for p in paths},
                config_sha256=digest(config_path),scene_sha256=digest(root/"hrl_v2/configs/physical_scene.json"))


def runtime_record():
    import numpy
    return dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
                torch=torch.__version__,numpy=numpy.__version__,cuda=torch.version.cuda,
                device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                deterministic=torch.are_deterministic_algorithms_enabled())


def save(path,payload):
    if payload.get("schema")!=SCHEMA or not payload.get("safe_episode_boundary"):
        raise ValueError("Only V2 completed-episode safe boundaries can be saved")
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    try:
        with tmp.open("wb") as f:
            torch.save(payload,f); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if tmp.exists(): tmp.unlink()
    atomic_json(path.with_suffix(".json"),dict(schema=SCHEMA,sha256=digest(path),
        completed_episode=payload["completed_episode"],pending_evaluation=payload["pending_evaluation"],
        identity=payload["identity"]))


def load(path,expected_identity):
    path=Path(path)
    sidecar=json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if sidecar.get("schema")!=SCHEMA or sidecar.get("identity")!=expected_identity:
        raise ValueError("Checkpoint schema or source/config/scene identity mismatch")
    if digest(path)!=sidecar["sha256"]: raise ValueError("Checkpoint byte digest mismatch")
    value=torch.load(path,map_location="cpu",weights_only=False)
    if value.get("schema")!=SCHEMA or not value.get("safe_episode_boundary"):
        raise ValueError("Old or incomplete checkpoint is not a V2 resume boundary")
    if value["identity"]!=expected_identity: raise ValueError("Inner checkpoint identity mismatch")
    return value


def runtime_signature(record):
    """Numerical environment contract, independent of install/OS paths."""
    return {key:record.get(key) for key in ("python","torch","numpy","cuda","device","deterministic")}
