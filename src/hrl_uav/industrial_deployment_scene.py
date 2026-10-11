"""Versioned deployment configuration around the unchanged industrial scene."""
import copy
import json
import math
from pathlib import Path

from .industrial_scene import validate_config

SCHEMA = "industrial_deployment_scene_v3"
PROFILE = "synchronous_linear_then_hold"


def validate_deployment_config(wrapper):
    """Return a validated copy, retaining both scene and deployment identities."""
    if not isinstance(wrapper, dict) or wrapper.get("schema") != SCHEMA:
        raise ValueError("Expected industrial_deployment_scene_v3 wrapper")
    if set(wrapper) - {"schema", "status", "base_scene", "deployment"}:
        raise ValueError("Unknown deployment scene key")
    c = copy.deepcopy(wrapper)
    c["base_scene"] = validate_config(c["base_scene"])
    d, base = c["deployment"], c["base_scene"]
    if set(d) != {"macro_period_s", "adjustment_s", "max_target_step_m", "profile"}:
        raise ValueError("Unexpected deployment fields")
    if d["profile"] != PROFILE:
        raise ValueError("Unsupported deployment profile")
    for name in ("macro_period_s", "adjustment_s", "max_target_step_m"):
        if isinstance(d[name], bool) or not math.isfinite(d[name]) or d[name] <= 0:
            raise ValueError("Deployment quantities must be finite and positive")
    h, m, dt, routing = (d["macro_period_s"], d["adjustment_s"],
                         base["time"]["physical_s"], base["time"]["routing_s"])
    if not math.isclose(h, base["time"]["motion_s"], rel_tol=0., abs_tol=1e-10):
        raise ValueError("Macro period must match base scene motion clock")
    if not 0 < m < h:
        raise ValueError("Deployment requires a positive adjustment and holding interval")
    for large, small in ((h, dt), (m, dt), (h, routing), (m, routing)):
        if not math.isclose(large/small, round(large/small), rel_tol=0., abs_tol=1e-9):
            raise ValueError("Deployment boundaries must align with physical and routing clocks")
    if d["max_target_step_m"] > base["motion"]["max_speed_mps"]*m + 1e-10:
        raise ValueError("Target radius is unreachable during the adjustment interval")
    return c


def load_config(path):
    return validate_deployment_config(json.loads(Path(path).read_text(encoding="utf-8-sig")))
