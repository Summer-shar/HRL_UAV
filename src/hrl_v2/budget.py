"""Archive-only finite-window timing extrapolation, never current research.

The constants below describe the archived measurement protocol. Explicit
opt-in and matching historical fixture metadata are required. Continuing-
service benchmark records cannot be used in this old budget model.
"""
import argparse
import json
from pathlib import Path


def require_archived_measurement(measurement, *, archived_finite_window=False):
    if not archived_finite_window:
        raise ValueError("This budget is archive-only; explicitly opt in to archived finite-window extrapolation")
    fixture=measurement.get("fixture",{})
    if (measurement.get("schema") is not None or measurement.get("status")!="complete" or
            measurement.get("baseline_commit")!="137d2f1" or
            fixture.get("physical_steps")!=9000 or fixture.get("history_frames")!=1800 or
            "ppo_kl900" not in measurement.get("measurements",{})):
        raise ValueError("Budget accepts only archived finite-window measurements, never current continuing-service results")


def budget(measurement, microbatch=16, extra_io=None, *, archived_finite_window=False):
    require_archived_measurement(measurement,archived_finite_window=archived_finite_window)
    m = measurement["measurements"]
    t = lambda key: m[key]["seconds"]
    optimized = measurement["variant"] == "optimized"
    critic = t(f"critic128_micro{microbatch}")
    actor = t(f"actor128_micro{microbatch}")
    ppo = m["ppo150_actual_first_minibatch"]
    # No PPO early stopping is assumed: 36 actor/value minibatches and six
    # entire 900-row KL traversals. Setup is paid once per complete batch.
    ppo_batch = 36 * ppo["seconds"] + 6 * t("ppo_kl900") + ppo["update_setup_seconds"]
    replay_sample = t("replay_sample128")
    save0, save4 = m["checkpoint_empty_replay_save"], m["checkpoint_four_episodes_save"]
    episode_save = max(0., (save4["seconds"] - save0["seconds"]) / 4)
    episode_bytes = max(0., (save4["bytes"] - save0["bytes"]) / 4)
    # Exact occupancy from the unchanged schedule: 300 lower + 100 upper,
    # then ten cycles of 10 upper + 10 lower. Replay flushes on the first
    # lower episode after each upper block. The current upper batch has
    # 1,2,3,4,0 unfinished episodes after successive upper episodes.
    replay_slots = 0
    ppo_slots = 0
    saved_replay = 0
    saved_ppo = 0
    evaluated_slots = 0
    checkpoint_count = 1
    last_layer = None
    development = {300, 400, 440, 480, 520, 560, 600}
    for number in range(1, 601):
        if number <= 300:
            layer = "lower"
        elif number <= 400:
            layer = "upper"
        else:
            layer = "upper" if (number - 401) % 20 < 10 else "lower"
        if layer == "lower":
            if last_layer != "lower":
                replay_slots = 0
            replay_slots = min(55, replay_slots + 1)
        else:
            ppo_slots = (ppo_slots + 1) % 5
        copies = 2 if number in development else 1
        checkpoint_count += copies
        saved_replay += copies * replay_slots
        saved_ppo += copies * ppo_slots
        if number in development:
            evaluated_slots += replay_slots + ppo_slots
        last_layer = layer
    assert checkpoint_count == 608 and saved_replay == 22675 and saved_ppo == 400
    checkpoints_seconds = checkpoint_count * save0["seconds"] + (saved_replay + saved_ppo) * episode_save
    checkpoints_bytes = checkpoint_count * save0["bytes"] + (saved_replay + saved_ppo) * episode_bytes
    # save()'s digest is already in the measured save. persist() reads the
    # same file once more for latest.json; each development evaluation also
    # computes its checkpoint identity. These are additional reads only.
    extra_hash_bytes = checkpoints_bytes + 7 * save0["bytes"] + evaluated_slots * episode_bytes
    extra_hash_seconds = (extra_hash_bytes * extra_io["seconds"] / extra_io["file_bytes"]
                          if extra_io else 0.)
    full_ppo = m.get("ppo900_complete_batch")
    common = dict(
        critic_updates=70020 * critic,
        actor_updates=17010 * actor,
        ppo_batches_no_KL_early_stop=40 * ppo_batch,
        replay_sampling=87030 * replay_sample,
        scripted_physics_reports_compression_and_metrics=600 * t("scripted_collection_9000"),
        checkpoint_serialization_fsync_hash=checkpoints_seconds,
        additional_checkpoint_identity_hashes=extra_hash_seconds,
        learning_log_appends=36040 * t("learning_jsonl_append"),
        # An episode metrics payload is larger than progress/latest JSON. Use
        # it for all same-primitive writes as a conservative I/O sensitivity.
        other_json_writes= (600 + 600 + 608 + 156 + 7 + 5) * t("episode_metrics_json"),
    )
    rebuild = m["lower_post_update_rebuild"]
    lower_average = rebuild["warm_next_mean_seconds"]
    lower_cold = t("lower_episode_start")
    base_rebuild_count = 16632 if optimized else 0
    train_lower = ((1080000 - 600 - base_rebuild_count) * lower_average
                   + 600 * lower_cold + base_rebuild_count * rebuild["seconds"])
    upper_low = t("ordinary_upper_inference")
    upper_high = t("congested_upper_inference")
    value_low = max(0., t("ordinary_upper_with_value_inference") - upper_low)
    value_high = max(0., t("congested_upper_with_value_inference") - upper_high)
    # Upper inference samples two workload windows. Report a sensitivity
    # envelope, not a statistical confidence interval or a full-seed timing.
    value_calls = 36000 if optimized else 54000
    dev_physics = t("streaming_evaluation_9000") if optimized else t("scripted_collection_9000")
    envelopes = {}
    for label, upper, value in (("ordinary_upper_workload", upper_low, value_low),
                                ("congested_upper_workload", upper_high, value_high)):
        training = dict(common, lower_neural_sampling=train_lower,
                        upper_neural_sampling=54000 * upper, upper_value_sampling=value_calls * value)
        development_cost = dict(physics_reports_metrics=156 * dev_physics,
            lower_neural_sampling=(280800 - 156) * lower_average + 156 * lower_cold,
            upper_neural_sampling=12960 * upper)
        # Primary stochastic test for the fixed final checkpoint only.
        test_cost = dict(physics_reports_metrics=160 * dev_physics,
            lower_neural_sampling=(288000 - 160) * lower_average + 160 * lower_cold,
            upper_neural_sampling=14400 * upper,
            metrics_json=161 * t("episode_metrics_json"))
        train_total = sum(training.values())
        dev_total = sum(development_cost.values())
        test_total = sum(test_cost.values())
        envelopes[label] = dict(training_seconds=training, development_seconds=development_cost,
            optional_primary_test_seconds=test_cost, training_plus_development_hours=(train_total + dev_total) / 3600,
            optional_primary_test_hours=test_total / 3600,
            training_development_primary_test_hours=(train_total + dev_total + test_total) / 3600)
        subtotal=(train_total + dev_total) / 3600
        envelopes[label]["planning_hours_with_5_to_15_percent_reserve"]=[subtotal*1.05, subtotal*1.15]
        if full_ppo:
            measured=(train_total + dev_total - 40*ppo_batch + 40*full_ppo["seconds"])/3600
            envelopes[label]["hours_using_measured_fixture_PPO_KL_behavior"]=measured
    return dict(scope="archived_finite_window_extrapolation_only",applies_to_current_research=False,
        variant=measurement["variant"], sac_microbatch=microbatch,
        archived_protocol_counts=dict(episodes=600, physical_steps=5400000, sac_events=35010,
            critic_fits=70020, actor_updates=17010, ppo_batches=40, ppo_effective_batch=900,
            ppo_minibatch=150, maximum_ppo_actor_steps=1440, ppo_value_steps=1440,
            maximum_ppo_kl_scans=240, lower_sampling=1080000, upper_sampling=54000,
            upper_value_sampling=value_calls, actor_update_internal_cache_rebuilds=16632,
            development_episodes=156, development_moving=72, development_hover=84,
            optional_primary_test_episodes=160, checkpoints=608, checkpoint_replay_episode_records=22675,
            checkpoint_pending_ppo_episode_records=400, replay_samples=87030, learning_logs=36040),
        measured_coefficients=dict(critic128_seconds=critic, actor128_seconds=actor,
            ppo150_seconds=ppo["seconds"], ppo900_batch_seconds_extrapolated=ppo_batch,
            ppo900_complete_fixture=full_ppo,
            lower_average_seconds=lower_average, lower_rebuild_seconds=rebuild["seconds"],
            checkpoint_full55_seconds_extrapolated=save0["seconds"] + 55 * episode_save,
            checkpoint_full55_bytes_extrapolated=save0["bytes"] + 55 * episode_bytes,
            cumulative_checkpoint_write_bytes_extrapolated=checkpoints_bytes,
            additional_checkpoint_hash_reads=615,
            additional_checkpoint_hash_bytes_extrapolated=extra_hash_bytes,
            additional_hash_measured=extra_io is not None),
        envelopes=envelopes,
        limitations=[
            "All full-seed values are conditional extrapolations, not training results. SAC/PPO use five independent scripted physical trajectories; inference/component timing uses the first trajectory.",
            "PPO data use real scripted reports with synthetic current-policy auxiliary values; they are a computational fixture, not an on-policy research batch.",
            "SAC repeats perform bounded updates only on temporary learners; timing uses the median of three after one warmup.",
            "Ordinary/congested upper workload bounds are sensitivity cases, not probability or confidence bounds.",
            "Replay occupancy counts are exact for the schedule; checkpoint cost is affine extrapolation from 0/4 episodes to 55, including Adam states.",
            "Physical collection already includes compression/report generation; standalone component timings are not added again.",
            "The main PPO envelope assumes all 36 actor/value minibatches and six KL scans. The complete measured fixture also reports its actual KL behavior; its alternate budget does not change the research protocol. Later epoch snapshot overhead is included in the complete measurement, but not isolated in the no-early-stop extrapolation.",
            "Additional checkpoint identity hashing uses measured production read_bytes SHA256 throughput when supplied. Target Polyak updates, source-identity hashing, startup, garbage collection and filesystem latency variation use the explicit 5-15% planning reserve.",
            "Primary stochastic test is shown separately; optional secondary-checkpoint and deterministic-policy tests are excluded and each requires its own declared panel budget.",
            "No research batch reduction, history truncation, queue deletion, encoder sharing or distribution backend change is used.",
        ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra-io", type=Path)
    parser.add_argument("--archived-finite-window",action="store_true",
                        help="Explicitly reproduce only the archived finite-window timing model")
    args = parser.parse_args()
    before = json.loads(args.baseline.read_text())
    after = json.loads(args.optimized.read_text())
    if before.get("status") != "complete" or after.get("status") != "complete":
        raise ValueError("Only completed measurements can be extrapolated")
    extra_io=json.loads(args.extra_io.read_text()) if args.extra_io else None
    result = dict(baseline=budget(before, extra_io=extra_io,archived_finite_window=args.archived_finite_window),
                  optimized_micro16=budget(after, extra_io=extra_io,archived_finite_window=args.archived_finite_window),
                  optimized_micro32=budget(after,32,extra_io,archived_finite_window=args.archived_finite_window)
                    if "critic128_micro32" in after["measurements"] else None)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: None if v is None else {e: d["training_plus_development_hours"] for e, d in v["envelopes"].items()}
                      for k, v in result.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
