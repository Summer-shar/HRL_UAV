"""Bounded frozen moving/hovering evaluation for the device-observation pilot."""

from pathlib import Path
import time

import numpy as np
import torch

from industrial_device_pilot import experiment as e
from hrl_uav.industrial_deployment_training import DeploymentLearningControllers


EVALUATION_SCHEMA = "industrial_device_paired_evaluation_v1"
MAX_PHYSICAL_STEPS = 108000
MAX_WALL_SECONDS = 900


def evaluate(checkpoint, output):
    """Evaluate six complete paired 900 s scenarios without advancing learning.

    Each frame's requested physics steps are charged before execution, including
    a partially failed frame. An existing output directory is never reused, so a
    partial attempt cannot silently receive a fresh budget in this entry point.
    """
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows, attempted = [], 0
    before = None
    checkpoint = Path(checkpoint)
    e.atomic_json(output / "budget.json", dict(
        schema=EVALUATION_SCHEMA, maximum_physical_steps=MAX_PHYSICAL_STEPS,
        maximum_wall_seconds=MAX_WALL_SECONDS, trajectories=12,
        seconds_per_trajectory=900, learning_updates=0, replay_writes=0,
        failed_partial_frame_charged=True, automatic_retry=False,
    ))
    try:
        agent, scene, p, state = e.load_candidate(checkpoint)
        if p["purpose"] != "pilot":
            raise ValueError("Performance evaluation rejects functional-check checkpoints")
        panel = p["development_manifest"]
        if (panel["seconds"] != 900 or len(panel["scenarios"]) != 6
                or len({row["scenario_id"] for row in panel["scenarios"]}) != 6):
            raise ValueError("Evaluation requires the six sealed 900 s development scenarios")
        if state["completed_episodes"] not in p["budget"]["joint_development_check_episodes"]:
            raise ValueError("Checkpoint is not a registered development evaluation node")
        before = e.frozen_fingerprint(agent)
        counters_before = e.counters(agent)
        identity = e.source_identity(p)
        checkpoint_sha = e.file_sha(checkpoint)
        e.atomic_json(output / "manifest.json", dict(
            schema=EVALUATION_SCHEMA, status="running", episode=state["completed_episodes"],
            checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
            protocol_sha256=e.digest(p), identity=identity,
            frozen_fingerprint_before=before, counters_before=counters_before,
        ))

        def remaining_budget(frame_steps=0):
            if time.perf_counter() - started >= MAX_WALL_SECONDS:
                raise TimeoutError("Device evaluation wall-clock budget reached")
            if attempted + frame_steps > MAX_PHYSICAL_STEPS:
                raise TimeoutError("Device evaluation physical-step budget reached")

        with torch.no_grad():
            for scenario in panel["scenarios"]:
                for mode in ("hover", "moving"):
                    remaining_budget()
                    mdp = e.DeviceMDP(scene, p)
                    real = e.reset_scene(mdp, scenario, scenario["profile_permutation"])
                    controllers = DeploymentLearningControllers(
                        agent, mdp.encoder, deterministic=True, macro_enabled=mode == "moving")
                    curve = []
                    discounted_common = discounted_upper = 0.
                    macro_reward_sum = upper_discounted_return = 0.
                    for macro_index in range(180):
                        remaining_budget()
                        action, _ = controllers.macro(mdp.controller_information())
                        movement = mdp.start_macro(action)
                        macro_reward = 0.
                        for frame_index in range(mdp.frames_per_macro):
                            remaining_budget(mdp.env.frame_steps)
                            mdp.prepare_service()
                            start = mdp.env.step_index
                            # Reserve requested work before entering the physics routine.
                            attempted += mdp.env.frame_steps
                            result = mdp.step_frame(controllers.micro(mdp.controller_information()))
                            discounted_common += sum(
                                mdp.gamma_step ** (start + index) * row["cost"]
                                for index, row in enumerate(result["readings"]))
                            discounted_upper += sum(
                                mdp.upper_gamma_step ** (start + index) * row["cost"]
                                for index, row in enumerate(result["readings"]))
                            frame_reward = -sum(
                                mdp.upper_gamma_step ** index * row["cost"]
                                for index, row in enumerate(result["readings"]))
                            macro_reward += mdp.upper_gamma_frame ** frame_index * frame_reward
                        macro_reward_sum += macro_reward
                        upper_discounted_return += mdp.gamma_macro ** macro_index * macro_reward
                        curve.append(dict(
                            **e.snapshot(mdp), cost_sum=mdp.env.cost_sum,
                            holding_task_s=mdp.env.holding,
                            energy_j=sum(mdp.env.energy.values()),
                            raw_action=np.asarray(action).tolist(),
                            safety_scale=float(movement["safety_scale"]),
                            upper_reward=macro_reward,
                        ))
                    metrics = mdp.metrics()
                    scale_bits = max(1., mdp.env.generated_bits)
                    scale_cycles = max(1., mdp.env.generated_cycles)
                    if any(abs(value) > 1e-5 + 1e-9 * (
                            scale_cycles if key.endswith("cycles") else scale_bits)
                           for key, value in metrics["conservation"].items()):
                        raise AssertionError("Task conservation failed")
                    if (mdp.env.step_index != 9000 or metrics["seconds"] != 900
                            or not np.isclose(upper_discounted_return, -discounted_upper,
                                              rtol=1e-12, atol=1e-8)):
                        raise AssertionError("Trajectory duration or reward/cost accounting changed")
                    row = dict(
                        status="complete", technical_valid=True, frozen_verified=True,
                        metrics_frozen_before_priming=True,
                        scenario_id=scenario["scenario_id"], scenario_sha=e.digest(scenario),
                        realization_sha=real["realization_sha256"],
                        event_sha256=metrics["event_sha256"],
                        canonical_event_sha256=metrics["canonical_event_sha256"],
                        profile_permutation=real["profile_by_region"], mode=mode, metrics=metrics,
                        low_unfinished=metrics["priority"][1]["arrived"] - metrics["priority"][1]["completed"],
                        discounted_cost_common=discounted_common,
                        discounted_cost_upper=discounted_upper,
                        upper_reward_sum=macro_reward_sum,
                        upper_discounted_return=upper_discounted_return,
                        curve=curve,
                        upload_area_mbit_s_5s=float(5 * sum(point["upload_mbits"] for point in curve)),
                    )
                    rows.append(row)
                    e.atomic_json(output / (str(scenario["scenario_id"]) + "_" + mode + ".json"), row)
                    e.atomic_json(output / "progress.json", dict(
                        status="running", trajectories=len(rows), physical_steps=attempted))
        remaining_budget()
        if attempted != MAX_PHYSICAL_STEPS or len(rows) != 12:
            raise AssertionError("Incomplete paired evaluation budget")
        after = e.frozen_fingerprint(agent)
        if (before != after or e.source_identity(p) != identity
                or e.counters(agent) != counters_before or e.file_sha(checkpoint) != checkpoint_sha):
            raise AssertionError("Frozen evaluation changed learner, checkpoint, or source identity")
        moving = [row for row in rows if row["mode"] == "moving"]
        hover = [row for row in rows if row["mode"] == "hover"]
        pairs = e.paired_rows(moving, hover)
        if any(left["canonical_event_sha256"] != right["canonical_event_sha256"]
               for left, right in pairs):
            raise AssertionError("Canonical arrivals are not paired")
        moving_pooled, hover_pooled = e.pooled_metrics(moving), e.pooled_metrics(hover)
        cost_ratio = moving_pooled["mean_cost_per_s"] / hover_pooled["mean_cost_per_s"]
        delay_ratio = moving_pooled["weighted_delay_ms"] / hover_pooled["weighted_delay_ms"]
        upload_ratio = sum(row["upload_area_mbit_s_5s"] for row in moving) / max(
            1e-12, sum(row["upload_area_mbit_s_5s"] for row in hover))
        quality = p["quality_rule"]
        bad = (cost_ratio > quality["cost_ratio_bad"]
               and delay_ratio > quality["delay_ratio_bad"]
               and upload_ratio > quality["upload_area_ratio_bad"])
        report = dict(
            schema=EVALUATION_SCHEMA, protocol_schema=e.SCHEMA, status="complete",
            arm=p["arm"], episode=state["completed_episodes"], checkpoint=str(checkpoint),
            checkpoint_sha256=checkpoint_sha, protocol_sha256=e.digest(p), identity=identity,
            frozen_verified=True, frozen_fingerprint_before=before, frozen_fingerprint_after=after,
            paired_events_verified=True, physical_steps=attempted, learning_updates=0, replay_writes=0,
            counters_before=counters_before, counters_after=e.counters(agent),
            moving_pooled=moving_pooled, hover_pooled=hover_pooled,
            cost_ratio=cost_ratio, delay_ratio=delay_ratio, upload_area_ratio=upload_ratio,
            bad_performance_node=bool(bad), rows=rows, elapsed_seconds=time.perf_counter() - started,
        )
        e.atomic_json(output / "report.json", report)
        e.atomic_json(output / "manifest.json", dict(
            schema=EVALUATION_SCHEMA, status="complete", physical_steps=attempted,
            checkpoint_sha256=checkpoint_sha, protocol_sha256=e.digest(p), identity=identity,
            frozen_verified=True, paired_events_verified=True,
            report_sha256=e.file_sha(output / "report.json"),
        ))
        e.atomic_json(output / "progress.json", dict(status="complete", physical_steps=attempted))
        return report
    except BaseException as exc:
        e.atomic_json(output / "failure.json", dict(
            schema=EVALUATION_SCHEMA, status="failed", error=repr(exc),
            attempted_physical_steps=attempted, completed_trajectories=len(rows),
            elapsed_seconds=time.perf_counter() - started, learning_updates=0, replay_writes=0,
            frozen_fingerprint_before=before, automatic_retry=False,
        ))
        raise
