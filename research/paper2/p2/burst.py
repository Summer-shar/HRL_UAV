"""Controlled traffic burst: delay over time while one region is overloaded.

During [start_s, end_s) every terminal of the region with the largest expected
computation demand generates tasks at the busy intensity (the scene's
``busy_multiplier``, 3x the normal rate) whatever its hidden activity state.
Outside the window the scene is untouched.  The burst is exogenous: arrivals
do not depend on the policy, so all methods see identical task sequences in a
given scenario (paired comparison, as in the main test).

Only the *instance* method ``env._scene.frame`` of one evaluation is wrapped;
the v2.4.0 source is not modified.  The temporary change of the profile's
``normal_multiplier`` is restored after every frame.

Recorded per run, in bins of ``bin_s`` by task arrival time: the weighted
sojourn (completion - arrival, or window end - arrival if unfinished) and the
weights, overall and per priority.
"""
import numpy as np

BURST = dict(start_s=300.0, end_s=400.0, bin_s=10.0)
BURST_SMOKE = dict(start_s=10.0, end_s=30.0, bin_s=5.0)


def window(smoke=False):
    return dict(BURST_SMOKE if smoke else BURST)


def _region_demand(scene):
    """Expected Gcycle/s of each source region at the normal-state rate."""
    demand = np.zeros(3)
    for k, terminal in enumerate(scene.terminals):
        profile, (_, slot) = scene._profiles[k], scene._logical[k]
        p = profile["slots"][slot]["parameters"]
        hp = p["high_probability"]
        cycles = hp*p["input_bits"][0]*p["cycles_per_bit"][0] + (1-hp)*p["input_bits"][1]*p["cycles_per_bit"][1]
        demand[terminal.source] += p["rate_per_s"]*profile["business_multiplier"]*cycles/1e9
    return demand


def install(env, start_s, end_s, **_):
    """Wrap env._scene.frame so that the heaviest region is fully busy in the window."""
    scene = env._scene
    region = int(np.argmax(_region_demand(scene)))
    members = [k for k, t in enumerate(scene.terminals) if t.source == region]
    profiles = {id(scene._profiles[k]): scene._profiles[k] for k in members}
    if len(profiles) != 1:
        raise RuntimeError('a region is expected to carry exactly one business profile')
    profile = next(iter(profiles.values()))
    others = [k for k, t in enumerate(scene.terminals) if t.source != region]
    if any(scene._profiles[k] is profile for k in others):
        raise RuntimeError('the burst profile is shared with another region')
    act = profile["activity_parameters"]
    dt = scene.config["time"]["routing_s"]
    original = scene.frame

    def frame(index):
        if start_s <= index*dt < end_s:
            saved = act["normal_multiplier"]
            act["normal_multiplier"] = act["busy_multiplier"]
            try:
                return original(index)
            finally:
                act["normal_multiplier"] = saved
        return original(index)

    scene.frame = frame
    return dict(region=region, region_demand_gcycles=_region_demand(scene).tolist(),
                start_s=start_s, end_s=end_s, busy_multiplier=act["busy_multiplier"])


def record(env, bin_s, end_time_s, weights=(10.0, 1.0)):
    """Sojourn of every parent task binned by arrival time."""
    edges = np.arange(0.0, end_time_s + bin_s/2, bin_s)
    n = len(edges) - 1
    out = {k: np.zeros(n) for k in ('w_sojourn', 'w', 'high_sojourn', 'high_n', 'low_sojourn', 'low_n', 'unfinished')}
    for parent in env.parents.values():
        a = parent.arrival
        b = min(int(a.birth_s // bin_s), n - 1)
        finish = parent.finish_s if parent.finish_s is not None else end_time_s
        sojourn = finish - a.birth_s
        w = weights[a.priority]
        out['w_sojourn'][b] += w*sojourn
        out['w'][b] += w
        key = 'high' if a.priority == 0 else 'low'
        out[f'{key}_sojourn'][b] += sojourn
        out[f'{key}_n'][b] += 1
        out['unfinished'][b] += parent.finish_s is None
    return dict(bin_s=bin_s, edges=edges.tolist(), **{k: v.tolist() for k, v in out.items()})
