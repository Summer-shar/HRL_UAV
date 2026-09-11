"""Read-only diagnosis of stored PPO120 artifacts; never reconstruct training data."""

import csv
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrl_uav import trajectory_check as base
from hrl_uav import target_tracking_check as tracking


OUT = ROOT / "results/ppo120_macro_mechanism_offline"
WINDOWS = (0, 30, 60, 90, 120)
SEEDS = (33600, 33601)


def stats(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(mean=None, std=None, variance=None, p10=None, p50=None, p90=None)
    return dict(mean=float(x.mean()), std=float(x.std()), variance=float(x.var()),
                **{f"p{p}": float(np.percentile(x, p)) for p in (10, 50, 90)})


def save_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_geometry(trajectory, queues, cfg, window, seed):
    """Recover pre-action geometry from previous recorded cycle ends, omit cycle 1."""
    uavs, centroids, actions = (trajectory[k] for k in ("uav", "centroid", "macro_action"))
    if uavs.shape != (90, 3, 2) or actions.shape != (90, 6) or len(queues) != 90:
        raise ValueError("Unexpected saved evaluation layout")
    phase = 0
    rows, samples = [], []
    for k in range(1, 90):
        # phase_age is recorded after advance_macro_clock, before the NEXT decision.
        if float(queues[k-1]["phase_age"]) == 0:
            phase = (phase + 1) % len(cfg["phase_loads"])
        q, c = uavs[k-1], centroids[k-1]
        loads = np.asarray(cfg["phase_loads"][phase]) * cfg["environment"]["load_scale"]
        b, detail = tracking.tracking_action(q, c, loads, [[0, 0]] * 3, 60, .5)
        a_raw = np.asarray(actions[k], float)
        a = np.clip(a_raw.reshape(3, 2), -1, 1)
        a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1)
        a = a.ravel()
        basis, _ = tracking.cooperation_aligned_basis(c, q, loads)
        target = np.clip(c + 60 * np.einsum("nij,nj->ni", basis.astype(float), a.reshape(3, 2)), 0, 1000)
        delta = target - q
        distance = np.linalg.norm(delta, axis=1, keepdims=True)
        delta *= np.minimum(distance, 20) / np.maximum(distance, 1e-12)
        delta[distance[:, 0] <= .5] = 0
        # This readback invokes no environment safety function. Equality below establishes
        # that existing safety handling did not further change these recorded legs.
        np.testing.assert_allclose(q + delta, uavs[k], atol=1e-6, rtol=0)
        np.testing.assert_allclose(detail["original_target"], c, atol=1e-12)
        bn, an = np.linalg.norm(b), np.linalg.norm(a)
        cosine = float(np.dot(a, b) / (an * bn)) if an > 1e-12 and bn > 1e-12 else None
        row = dict(window=window, seed=seed, macro=k+1, phase_index=phase,
                   B_norm=float(bn), action_norm=float(an), distance_to_B=float(np.linalg.norm(a-b)),
                   raw_distance_to_B=float(np.linalg.norm(a_raw-b)),
                   angle_deg=float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))) if cosine is not None else None,
                   cosine=cosine, projection_length=float(np.dot(a, b)/bn) if bn > 1e-12 else None,
                   projection_gain=float(np.dot(a, b)/bn**2) if bn > 1e-12 else None,
                   closer_than_zero=bool(np.linalg.norm(a-b) < bn-1e-10),
                   dot=float(np.dot(a, b)), B_norm_squared=float(bn**2),
                   B_projection_triggered=int(np.sum(detail["projected"])),
                   action_projection_triggered=bool(np.max(np.abs(a-a_raw)) > 1e-10))
        for i in range(6):
            row[f"action_{i}"] = float(a[i])
            row[f"B_action_{i}"] = float(b[i])
        rows.append(row)
        samples.append((a, b))
    return rows, samples


def run():
    if OUT.exists():
        raise FileExistsError("Do not overwrite an existing offline diagnosis")
    cfg, source, _ = base.source_bundle()
    if base.git("rev-parse", "HEAD") != base.REVIEWED_HEAD:
        raise ValueError("Review changed HEAD first")
    prior = base.load_json(ROOT / "results/progressive_centroid_dev_recheck/manifest.json")
    for name, digest in prior["source_hashes_after"].items():
        if base.file_hash(ROOT / name) != digest:
            raise ValueError(f"Source differs from completed B recheck: {name}")
    manifest = base.load_json(base.HISTORY / "manifest.json")
    report = base.load_json(base.HISTORY / "report.json")
    inputs = {}

    def remember(path):
        path = Path(path)
        inputs[str(path.relative_to(ROOT))] = base.file_hash(path)

    for p in (base.DEFAULT_SOURCE, base.HISTORY / "manifest.json", base.HISTORY / "report.json",
              ROOT / "results/progressive_centroid_dev_recheck/manifest.json",
              ROOT / "results/progressive_centroid_dev_recheck/paired_summary.csv"):
        remember(p)
    endpoint = Path(manifest["final_checkpoint"])
    remember(endpoint)
    assert base.file_hash(endpoint) == manifest["final_checkpoint_sha256"]
    rows_checkpoints = []
    counters = {0: 0}
    endpoint_states = {}
    macro_cache_lengths = {}
    for w in WINDOWS:
        path = base.DEFAULT_SOURCE if w == 0 else base.HISTORY / "checkpoints" / f"window{w}.pt"
        remember(path)
        s = source if w == 0 else torch.load(path, map_location="cpu", weights_only=False)
        for key in ("sac_actor", "sac_critic", "sac_target", "log_alpha"):
            assert base.state_equal(s[key], source[key]), key
        assert s["updates"]["ppo"] == w // 10
        macro_cache_lengths[w] = len(s["training_state"]["macro_rollout"])
        if macro_cache_lengths[w] != 0:
            raise ValueError("Unexpected recoverable macro samples; inspect before proceeding")
        ps = s["training_state"]["last_ppo_stats"]
        sd = s["macro_actor"]["log_std"].double().exp().numpy()
        np.testing.assert_allclose(sd, sd[0], atol=1e-8)
        actor_steps = sorted({float(x["step"]) for x in s["optimizers"][3]["state"].values()})
        critic_steps = sorted({float(x["step"]) for x in s["optimizers"][4]["state"].values()})
        if w:
            assert actor_steps == [s["updates"]["ppo_gradients"]]
            assert critic_steps == [w]  # w/10 update events, ten critic epochs each.
        counters[w//10] = s["updates"]["ppo_gradients"]
        row = dict(window=w, updates=s["updates"]["ppo"], ppo_gradients=s["updates"]["ppo_gradients"],
                   std_checkpoint=float(sd[0]), macro_rollout_count=macro_cache_lengths[w],
                   last_update_actor_loss=ps.get("actor_loss"), last_update_critic_loss=ps.get("critic_loss"),
                   last_update_kl=ps.get("kl"), last_update_epochs=ps.get("epochs"),
                   last_update_pre_logprob_error=ps.get("pre_logprob_error"),
                   critic_optimizer_steps=critic_steps[0] if critic_steps else 0,
                   path=str(path.relative_to(ROOT)))
        rows_checkpoints.append(row)
        endpoint_states[w//10] = ps
        if w == 120:
            final = torch.load(endpoint, map_location="cpu", weights_only=False)
            for key in ("macro_actor", "macro_critic", "updates", "training_state"):
                assert base.state_equal(s[key], final[key]), key
            del final
        if w:
            del s
    del source

    training_path = base.HISTORY / "training.csv"
    remember(training_path)
    with training_path.open(encoding="utf-8-sig", newline="") as h:
        training = list(csv.DictReader(h))
    assert len(training) == 120 and all(None not in r for r in training)
    assert [int(r["window"]) for r in training] == list(range(1, 121))
    events = report["ppo_update_events"]
    assert len(events) == 12 and report["ppo_transitions_consumed"] == 3600
    batches = []
    previous_std = float(rows_checkpoints[0]["std_checkpoint"])
    # Bounds follow solely from actual checkpoint counters and a maximum of ten epochs.
    epoch_bounds = {}
    for end in (3, 6, 9, 12):
        known_last = int(endpoint_states[end]["epochs"])
        remaining = counters[end] - counters[end-3] - known_last
        for u in (end-2, end-1):
            epoch_bounds[u] = (max(1, remaining-10), min(10, remaining-1))
        epoch_bounds[end] = (known_last, known_last)
    for event in events:
        u = event["update"]
        known = endpoint_states.get(u, {})
        values = training[(u-1)*10:u*10]
        ws = stats([float(r["window_reward"]) for r in values])
        js = stats([float(r["window_J"]) for r in values])
        lo, hi = epoch_bounds[u]
        row = dict(update=u, transitions=event["transitions"], start_window=(u-1)*10+1,
                   end_window=u*10, std_before=previous_std, std_after=event["std_after"][0],
                   epochs_exact=lo if lo == hi else None, epochs_min=lo, epochs_max=hi,
                   epoch_source="checkpoint counter bounds, plus last_ppo_stats at 3/6/9/12",
                   ppo_gradients_exact=counters.get(u), actor_loss=known.get("actor_loss"),
                   critic_loss_last_pre_step=known.get("critic_loss"), kl=known.get("kl"),
                   advantage_mean=None, advantage_p10=None, advantage_p50=None, advantage_p90=None,
                   return_mean=None, return_variance=None, signed_value_error=None,
                   training_action_B_distance=None,
                   **{"window_reward_"+k: v for k,v in ws.items()},
                   **{"window_J_"+k: v for k,v in js.items()})
        batches.append(row)
        previous_std = event["std_after"][0]
    # All remaining exact counters are also implied when every prior epoch count is exact.
    for u in range(1, 13):
        if all(epoch_bounds[k][0] == epoch_bounds[k][1] for k in range(1, u+1)):
            batches[u-1]["ppo_gradients_exact"] = sum(epoch_bounds[k][0] for k in range(1, u+1))
        if u > 6 and u != 7:
            batches[u-1]["ppo_gradients_exact"] = 70 + (u-8)*10

    geometry_rows, geometry_summary = [], []
    for w in WINDOWS:
        pooled = []
        for seed in SEEDS:
            prefix = base.HISTORY / f"validation_ppo_w{w}_{seed}"
            trajectory_path, queues_path = Path(str(prefix)+"_trajectory.npz"), Path(str(prefix)+"_queues.csv")
            remember(trajectory_path)
            remember(queues_path)
            with queues_path.open(encoding="utf-8-sig") as h:
                queues = list(csv.DictReader(h))
            with np.load(trajectory_path) as t:
                local, _ = evaluate_geometry(t, queues, cfg, w, seed)
            geometry_rows.extend(local)
            pooled.extend(local)
        bnorm = np.array([r["B_norm"] for r in pooled])
        distances = np.array([r["distance_to_B"] for r in pooled])
        sd = next(r["std_checkpoint"] for r in rows_checkpoints if r["window"] == w)
        # This compares scheduled stochastic scale to B amplitude on deterministic
        # evaluation states, NOT to the missing stochastic training samples.
        b_component_rms = float(np.sqrt(np.mean(bnorm**2)/6))
        geometry_summary.append(dict(window=w, macro_samples=len(pooled),
            distance_to_B_mean=float(distances.mean()), B_norm_mean=float(bnorm.mean()),
            action_norm_mean=float(np.mean([r["action_norm"] for r in pooled])),
            distance_ratio_to_zero=float(distances.mean()/bnorm.mean()),
            closer_than_zero_fraction=float(np.mean([r["closer_than_zero"] for r in pooled])),
            pooled_projection_gain=float(sum(r["dot"] for r in pooled)/sum(r["B_norm_squared"] for r in pooled)),
            angle_deg_mean=stats([r["angle_deg"] if r["angle_deg"] is not None else np.nan for r in pooled])["mean"],
            B_component_rms=b_component_rms, std_checkpoint=sd,
            std_over_B_component_rms=sd/b_component_rms,
            B_target_offset_rms_per_component_m=60*b_component_rms,
            exploration_target_offset_std_per_component_m=60*sd,
            B_projection_count=sum(r["B_projection_triggered"] for r in pooled)))

    gamma = cfg["training"]["gamma_per_resource_frame"]
    reward_scale = sum(gamma**(i/5) for i in range(100)) / 20
    p = ROOT / "results/progressive_centroid_dev_recheck/paired_summary.csv"
    with p.open(encoding="utf-8-sig") as h:
        delta = next(r for r in csv.DictReader(h) if r["split"] == "all_development")
    signal = abs(float(delta["delta_J"])) * reward_scale
    analysis = dict(missing_training_macro_samples=3600, checkpoint_macro_cache_lengths=macro_cache_lengths,
                    training_rows=120, training_row_unit="300s window means, NOT macro samples",
                    training_critic_loss_column="inherited frozen SAC last loss, NOT PPO critic loss",
                    availability="raw macro state/action/logprob/reward/next_state/advantage/value/return absent",
                    evaluation_readback=dict(samples=890, per_checkpoint=178,
                        omitted_first_cycles=10, source="stored deterministic evaluation actions and previous cycle end geometry",
                        limitation="not training samples, no advantage correlation; each checkpoint has its own induced states"),
                    kl_stops=dict(update6=True, update7_or8="at least one, possibly both",
                        total_stopped_updates_min=2, total_stopped_updates_max=3,
                        actor_epochs=110, allowed_max=120, critic_epochs=120),
                    scale_check=dict(J_difference=abs(float(delta["delta_J"])),
                        constant_step_cost_to_macro_reward_factor=reward_scale,
                        illustrative_macro_reward_difference=signal,
                        window_reward_std_range=[min(r["window_reward_std"] for r in batches), max(r["window_reward_std"] for r in batches)],
                        window_reward_std_over_illustrative_signal_range=[min(r["window_reward_std"] for r in batches)/signal, max(r["window_reward_std"] for r in batches)/signal],
                        interpretation="constant per-physical-step improvement assumption only; actual B discounted reward and advantage noise cannot be recovered from average J"))
    OUT.mkdir(parents=True, exist_ok=False)
    save_csv(OUT / "checkpoint_stats.csv", rows_checkpoints)
    save_csv(OUT / "update_batches.csv", batches)
    save_csv(OUT / "evaluation_action_readback.csv", geometry_rows)
    save_csv(OUT / "evaluation_action_summary.csv", geometry_summary)
    base.write_json(OUT / "analysis.json", analysis)
    assert all(base.file_hash(ROOT / p) == h for p,h in inputs.items())
    base.write_json(OUT / "manifest.json", dict(actual_head=base.git("rev-parse", "HEAD"),
        worktree=base.git("status", "--short"), input_sha256=inputs,
        source_hashes=prior["source_hashes_after"], B_controller_sha256=base.file_hash(ROOT / "hrl_uav/target_tracking_check.py"),
        environment_calls=0, rollout_calls=0, policy_inference_calls=0, backward_calls=0,
        replay_sampling_calls=0, optimizer_updates=0, input_files_unchanged=True,
        windows=WINDOWS, scenes_read=SEEDS, final_test_scenes_run=[]))
    print(json.dumps(dict(checkpoints=rows_checkpoints, geometry=geometry_summary, analysis=analysis), indent=2))


if __name__ == "__main__":
    with patch.object(base.QueueMDP, "__init__", base.forbidden), patch.object(base, "rollout", base.forbidden), \
         patch("torch.autograd.backward", base.forbidden), patch("torch.autograd.grad", base.forbidden):
        run()
