"""Sealed clustered layouts and complete business-profile permutations.

Generation metadata stays on the simulator side. Only the inherited public
terminal/task reports reach a controller.
"""
from dataclasses import asdict
from itertools import permutations
from pathlib import Path
import copy
import hashlib
import json
import math

import numpy as np

from .industrial_scene import Arrival, Terminal, TYPES
from .industrial_deployment_scene import validate_deployment_config

SCHEMA = "industrial_hotspot_scene_v4"
REALIZATION_SCHEMA = "industrial_hotspot_realization_v4"
PERMUTATIONS = tuple(permutations(range(3)))


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _seed(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _permutation(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("Expected a permutation of three business profiles")
    if any(type(v) is not int for v in value) or tuple(sorted(value)) != (0, 1, 2):
        raise ValueError("Expected a permutation of profiles 0, 1, 2")
    return list(value)


def validate_hotspot_config(wrapper):
    if not isinstance(wrapper, dict) or wrapper.get("schema") != SCHEMA:
        raise ValueError(f"Expected {SCHEMA}")
    allowed = {"schema", "status", "base_scene", "deployment", "geometry", "business_profiles"}
    if set(wrapper) - allowed or not allowed.difference({"status"}).issubset(wrapper):
        raise ValueError("Unexpected hotspot wrapper fields")
    c = copy.deepcopy(wrapper)
    old = validate_deployment_config(dict(schema="industrial_deployment_scene_v3",
        base_scene=c["base_scene"], deployment=c["deployment"]))
    c["base_scene"], c["deployment"] = old["base_scene"], old["deployment"]
    base, geometry, business = c["base_scene"], c["geometry"], c["business_profiles"]
    if geometry.get("random_uav_start") is not False:
        raise ValueError("UAV starts must remain fixed")
    if geometry.get("uav_initial_positions_m") != base["uav_initial_positions_m"]:
        raise ValueError("Geometry and physical UAV starts disagree")
    if (geometry.get("cluster_count_per_region"), geometry.get("background_count_per_region")) != (9, 4):
        raise ValueError("Each region requires nine cluster and four background terminals")
    radius = geometry.get("cluster_radius_m")
    if isinstance(radius, bool) or not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius <= 0:
        raise ValueError("Invalid cluster radius")
    bounds, regions = np.asarray(geometry.get("center_bounds_m"), float), np.asarray(base["regions_m"], float)
    if bounds.shape != (3, 2, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Invalid center bounds")
    if np.any(bounds[:, 0]-radius < regions[:, 0]) or np.any(bounds[:, 1]+radius > regions[:, 1]):
        raise ValueError("The entire cluster disk must remain in its region")
    if geometry.get("coordinate_clipping") is not False or geometry.get("cluster_sampling") != "uniform_disk_area" or geometry.get("background_sampling") != "uniform_region_reject_inside_cluster_disk":
        raise ValueError("Unsupported spatial sampling protocol")
    if business.get("mode") != "whole_profile_permutation" or business.get("physical_source_rule") != "user_id // 13":
        raise ValueError("Expected complete profiles with fixed physical source associations")
    if business.get("physical_uav_ids_permuted") is not False or business.get("forced_rotation") is not False or business.get("adjacent_repetition_allowed") is not True:
        raise ValueError("Load roles must be iid, with repetition and fixed physical UAV IDs")
    if business.get("sampling") != "iid uniform over all six permutations each training episode":
        raise ValueError("Unsupported load-role sampling")
    if business.get("profile_by_region_permutations") != [list(p) for p in PERMUTATIONS]:
        raise ValueError("All six profile permutations must be registered")
    catalog = business.get("catalog")
    if not isinstance(catalog, list) or len(catalog) != 3:
        raise ValueError("Expected three complete business profiles")
    for profile_id, profile in enumerate(catalog):
        if profile.get("profile_id") != profile_id:
            raise ValueError("Catalog order must be profile 0, 1, 2")
        counts = profile.get("device_counts_by_type")
        if counts != base["device_counts"][profile_id]:
            raise ValueError("Profile counts must preserve the registered template")
        quotas = np.asarray(profile.get("cluster_counts_by_type"))
        if quotas.shape != (3,) or not np.issubdtype(quotas.dtype, np.integer) or np.any(quotas < 0) or np.any(quotas > counts) or quotas.sum() != 9:
            raise ValueError("Invalid cluster type quotas")
        if profile.get("business_multiplier") != base["region_multipliers"][profile_id] or profile.get("activity_parameters") != base["activity"]:
            raise ValueError("A whole template must preserve its load and activity parameters")
        slots = profile.get("slots")
        expected_types = [kind for kind, count in zip(TYPES, counts) for _ in range(count)]
        if not isinstance(slots, list) or len(slots) != 13:
            raise ValueError("Each business profile requires thirteen slots")
        for row, kind in zip(slots, expected_types):
            if set(row) != {"task_type", "parameters"} or row["task_type"] != kind or row["parameters"] != base["task_types"][kind]:
                raise ValueError("Slot type, priority, bit and cycles definitions must move together")
    totals = np.asarray([p["device_counts_by_type"] for p in catalog]).sum(axis=0).tolist()
    if totals != [19, 10, 10] or business.get("global_device_counts_by_type") != totals:
        raise ValueError("The global device type totals must remain 19/10/10")
    _json(c)
    return c


def load_config(path):
    return validate_hotspot_config(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def generate_realization(wrapper, layout_seed, load_seed, profile_permutation=None):
    c = validate_hotspot_config(wrapper)
    layout_seed, load_seed = _seed(layout_seed, "layout_seed"), _seed(load_seed, "load_seed")
    override = None if profile_permutation is None else _permutation(profile_permutation)
    chosen = (list(PERMUTATIONS[int(np.random.default_rng(np.random.SeedSequence([load_seed, 5000])).integers(6))])
              if override is None else list(override))
    geometry, base = c["geometry"], c["base_scene"]
    radius = geometry["cluster_radius_m"]
    centers = [np.random.default_rng(np.random.SeedSequence([layout_seed, 5001, r])).uniform(*bounds).tolist()
               for r, bounds in enumerate(geometry["center_bounds_m"])]
    terminals = []
    for region, profile_id in enumerate(chosen):
        profile = c["business_profiles"]["catalog"][profile_id]
        center = np.asarray(centers[region])
        lo, hi = np.asarray(base["regions_m"][region])
        members = set()
        for kind_index, (kind, quota) in enumerate(zip(TYPES, profile["cluster_counts_by_type"])):
            indices = [s for s, row in enumerate(profile["slots"]) if row["task_type"] == kind]
            rng = np.random.default_rng(np.random.SeedSequence([layout_seed, 5002, profile_id, kind_index]))
            members.update(map(int, rng.choice(indices, size=quota, replace=False)))
        for slot, row in enumerate(profile["slots"]):
            member = slot in members
            purpose = 5003 if member else 5004
            rng = np.random.default_rng(np.random.SeedSequence([layout_seed, purpose, profile_id, slot]))
            if member:
                radial, theta = radius*math.sqrt(rng.random()), 2*math.pi*rng.random()
                position = center + radial*np.array([math.cos(theta), math.sin(theta)])
            else:
                for attempt in range(10000):
                    position = rng.uniform(lo, hi)
                    if np.linalg.norm(position-center) > radius:
                        break
                else:
                    raise RuntimeError("Background rejection sampler exceeded its fixed bound")
            terminals.append(dict(user_id=region*13+slot, source=region, profile_id=profile_id,
                slot=slot, task_type=row["task_type"], position_m=position.tolist(), is_cluster=member))
    record = dict(schema=REALIZATION_SCHEMA, config_sha256=_digest(c), layout_seed=layout_seed,
        load_seed=load_seed, profile_permutation_override=override, profile_by_region=chosen,
        cluster_centers_m=centers, uav_initial_positions_m=copy.deepcopy(base["uav_initial_positions_m"]),
        terminals=terminals)
    record["realization_sha256"] = _digest(record)
    return record


def validate_realization(wrapper, realization):
    if not isinstance(realization, dict) or realization.get("schema") != REALIZATION_SCHEMA:
        raise ValueError("Expected a sealed hotspot realization")
    try:
        rebuilt = generate_realization(wrapper, realization["layout_seed"], realization["load_seed"],
                                       realization["profile_permutation_override"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid realization metadata") from exc
    if _json(rebuilt) != _json(realization):
        raise ValueError("Sealed realization differs from deterministic reconstruction")
    return rebuilt


def effective_config(wrapper, realization):
    """Expose the actually assigned type counts and multipliers to service code."""
    c = copy.deepcopy(wrapper["base_scene"])
    catalog = wrapper["business_profiles"]["catalog"]
    c["device_counts"] = [copy.deepcopy(catalog[p]["device_counts_by_type"]) for p in realization["profile_by_region"]]
    c["region_multipliers"] = [catalog[p]["business_multiplier"] for p in realization["profile_by_region"]]
    return c


class HotspotIndustrialScene:
    """Task randomness is keyed by arrival seed, logical profile/slot and purpose."""
    def __init__(self, wrapper, seed, layout_seed=None, load_seed=None, realization=None):
        self.scene_wrapper = validate_hotspot_config(wrapper)
        self.seed = _seed(seed, "arrival_seed")
        if realization is None:
            layout_seed = self.seed if layout_seed is None else layout_seed
            load_seed = self.seed if load_seed is None else load_seed
            realization = generate_realization(self.scene_wrapper, layout_seed, load_seed)
        else:
            realization = validate_realization(self.scene_wrapper, realization)
            if layout_seed is not None and _seed(layout_seed, "layout_seed") != realization["layout_seed"]:
                raise ValueError("layout_seed disagrees with sealed realization")
            if load_seed is not None and _seed(load_seed, "load_seed") != realization["load_seed"]:
                raise ValueError("load_seed disagrees with sealed realization")
        self.realization = copy.deepcopy(realization)
        self.layout_seed, self.load_seed = realization["layout_seed"], realization["load_seed"]
        self.config = effective_config(self.scene_wrapper, realization)
        self.terminals = tuple(Terminal(t["user_id"], t["source"], t["task_type"], tuple(t["position_m"]))
                               for t in realization["terminals"])
        self._logical = tuple((t["profile_id"], t["slot"]) for t in realization["terminals"])
        def streams(purpose):
            return [np.random.default_rng(np.random.SeedSequence([self.seed, profile, slot, purpose]))
                    for profile, slot in self._logical]
        self._activity_rng, self._arrival_rng, self._mark_rng = streams(6001), streams(6002), streams(6003)
        self._profiles = [self.scene_wrapper["business_profiles"]["catalog"][p] for p, _ in self._logical]
        dt = self.config["time"]["routing_s"]
        self._switch_p = np.asarray([[-math.expm1(-dt/p["activity_parameters"][k])
                                     for k in ("normal_mean_s", "busy_mean_s")] for p in self._profiles])
        self._busy = np.array([r.random() < q[0]/q.sum() for r, q in zip(self._activity_rng, self._switch_p)])
        self._local_event_index = np.zeros(39, dtype=np.int64)
        self._frame, self._task_id = -1, 0
        self.event_digest, self.canonical_event_digest = hashlib.sha256(), hashlib.sha256()
        self.last_canonical_events = ()

    def frame(self, frame_index):
        if type(frame_index) is not int or frame_index != self._frame+1:
            raise ValueError("Exogenous frames must advance exactly once in physical order")
        dt = self.config["time"]["routing_s"]
        events, canonical = [], []
        for k, terminal in enumerate(self.terminals):
            if frame_index > 0 and self._activity_rng[k].random() < self._switch_p[k, int(self._busy[k])]:
                self._busy[k] = not self._busy[k]
            profile, (_, slot) = self._profiles[k], self._logical[k]
            params = profile["slots"][slot]["parameters"]
            activity = profile["activity_parameters"]["busy_multiplier" if self._busy[k] else "normal_multiplier"]
            rate = params["rate_per_s"]*profile["business_multiplier"]*activity
            for _ in range(self._arrival_rng[k].poisson(rate*dt)):
                priority = 0 if self._mark_rng[k].random() < params["high_probability"] else 1
                bits = float(params["input_bits"][priority])
                arrival = Arrival(self._task_id, terminal.user_id, terminal.source, terminal.task_type,
                                  priority, frame_index*dt, bits, bits*params["cycles_per_bit"][priority])
                self._task_id += 1
                events.append(arrival)
                self.event_digest.update((_json(asdict(arrival))+"\n").encode())
                canonical.append(dict(profile_id=profile["profile_id"], slot=slot,
                    local_event_index=int(self._local_event_index[k]), birth_s=arrival.birth_s,
                    priority=priority, input_bits=arrival.input_bits, compute_cycles=arrival.compute_cycles))
                self._local_event_index[k] += 1
        canonical.sort(key=lambda e: (e["profile_id"], e["slot"], e["local_event_index"]))
        self.canonical_event_digest.update((_json(dict(frame=frame_index, events=canonical))+"\n").encode())
        self.last_canonical_events = tuple(canonical)
        self._frame = frame_index
        return tuple(events)
