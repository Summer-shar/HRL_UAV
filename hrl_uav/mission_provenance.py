"""Immutable mission physics provenance, independent of later runner commits."""

import ast
import hashlib
from pathlib import Path
import subprocess

from .scene_prescreen import ROOT

PHYSICS_BASE_COMMIT = "689cd4603e0a926b60b73049246b02fd576f2408"
PHYSICS_FILES = (
    "hrl_uav/env.py", "hrl_uav/mdp.py", "hrl_uav/geometry.py",
    "hrl_uav/load_switching.py", "hrl_uav/routing.py", "hrl_uav/experiment.py",
    "hrl_uav/learning.py", "hrl_uav/agents/sac_agent.py", "hrl_uav/agents/ppo_agent.py",
    "configs/mission_rpgm_v1_smoke.json",
    "hrl_uav/bandwidth.py", "hrl_uav/scene_prescreen.py", "hrl_uav/ppo_frozen.py",
    "hrl_uav/trajectory_check.py", "hrl_uav/wireless_check.py", "hrl_uav/wireless_pilot.py",
)


def git_blob(commit, name):
    return subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)


def text_digest(data):
    # Git checkouts may use CRLF; record raw hashes separately, normalize only EOL.
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def physics_provenance(root=ROOT):
    result = {}
    for name in PHYSICS_FILES:
        expected = git_blob(PHYSICS_BASE_COMMIT, name)
        actual = (Path(root) / name).read_bytes()
        if text_digest(actual) != text_digest(expected):
            raise ValueError(f"Frozen mission physics changed: {name}")
        result[name] = dict(reference_sha256=text_digest(expected),
                            normalized_worktree_sha256=text_digest(actual),
                            raw_worktree_sha256=hashlib.sha256(actual).hexdigest())
    # The smoke runner may gain provenance metadata, but not new control semantics.
    name = "hrl_uav/mission_smoke.py"
    trees = [ast.parse(b.decode("utf-8-sig")) for b in
             (git_blob(PHYSICS_BASE_COMMIT, name), (Path(root) / name).read_bytes())]
    for definition in ("MissionPolicy", "MissionMeter", "mission_config"):
        nodes = [next(n for n in tree.body if getattr(n, "name", None) == definition) for tree in trees]
        dumps = [ast.dump(n, include_attributes=False) for n in nodes]
        if dumps[0] != dumps[1]:
            raise ValueError(f"Frozen mission controller/meter definition changed: {definition}")
        result[f"{name}::{definition}"] = dict(ast_sha256=hashlib.sha256(dumps[0].encode()).hexdigest())
    return result


def assert_committed_sources(names):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for name in names:
        if text_digest((ROOT / name).read_bytes()) != text_digest(git_blob(head, name)):
            raise ValueError(f"Pilot source must be committed before evaluation: {name}")
    return head
