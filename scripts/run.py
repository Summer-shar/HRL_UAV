"""One entry point. Example: python scripts/run.py --mode smoke."""

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrl_uav.experiment import run, smoke_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, help="Mainline defaults to hrl_ppo_only_unlock.json; other modes use main.json")
    parser.add_argument("--mode", choices=("smoke","pilot","mainline","mainline-check","ready-state-check","ready-state-compare","sac-check","sac-compare","bandwidth-check","cpu-check","routing-check","scene-check"), default="smoke")
    parser.add_argument("--source", type=Path, default=ROOT/"results/sac_compare")
    parser.add_argument("--method", choices=("ppo","centroid"), default="ppo")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--critic-input", choices=("raw_score", "executed_route"))
    parser.add_argument("--micro-actor-lr", type=float, help="Override only the pretraining Actor learning rate")
    parser.add_argument("--pretrain-only", action="store_true", help="Bounded micro-only diagnostic; do not start joint training")
    parser.add_argument("--resume-from", type=Path, help="Continue a saved mainline checkpoint in its original stage; total budget comes from config")
    parser.add_argument("--joint-evaluation-mode", choices=("periodic", "final_only"),
                        help="Change joint-stage monitoring only; final evaluation and checkpoint saving remain enabled")
    args = parser.parse_args()
    if args.resume_from is not None and args.mode != "mainline":
        parser.error("--resume-from requires --mode mainline; add --pretrain-only only for a micro checkpoint")
    config_path = args.config or ROOT / "configs" / (
        "hrl_ppo_only_unlock.json" if args.mode in ("mainline", "mainline-check") else "main.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.mode in ("mainline", "mainline-check") and config["training"].get("sac_stage_policy") != "preserve":
        print("[legacy config] Stage-specific SAC rates/warmup are enabled; this is NOT PPO-only unlocking.", flush=True)
    if args.joint_evaluation_mode is not None:
        if args.mode not in ("mainline", "mainline-check"):
            parser.error("--joint-evaluation-mode requires mainline or mainline-check")
        config["qualification"]["joint_evaluation_mode"] = args.joint_evaluation_mode
    if args.micro_actor_lr is not None or args.pretrain_only:
        if args.mode not in ("mainline", "mainline-check"):
            parser.error("Micro training overrides require mainline or mainline-check")
    if args.micro_actor_lr is not None:
        if not math.isfinite(args.micro_actor_lr) or args.micro_actor_lr <= 0:
            parser.error("--micro-actor-lr must be finite and positive")
        config["training"]["pretrain_lr"][0] = args.micro_actor_lr
    if args.critic_input:
        if args.mode not in ("mainline", "mainline-check"):
            parser.error("--critic-input is only defined for the guarded mainline")
        config["training"]["critic_action_representation"] = args.critic_input
    if args.mode == "scene-check":
        from hrl_uav.bandwidth import run_scene_resource_check
        report = run_scene_resource_check(config, args.output or ROOT/"results/restored_load_resource_check")
        print(json.dumps({k: report[k] for k in ("status", "minutes", "budget", "means")}, indent=2, ensure_ascii=False))
        return
    if args.mode in ("ready-state-check", "ready-state-compare"):
        from hrl_uav.observation_ablation import run_observation_ablation
        report = run_observation_ablation(config, args.output or ROOT/"results"/args.mode.replace("-", "_"),
                                          smoke=args.mode == "ready-state-check")
        print(json.dumps({"status": report["status"], "integrity": report["integrity"],
                          "means": {k: v["mean"] for k, v in report["arms"].items()},
                          "ready_minus_control": report["ready_minus_control"]}, indent=2, ensure_ascii=False))
        return
    if args.mode in ("mainline", "mainline-check"):
        from hrl_uav.qualification import run_mainline
        suffix = "_routing_critic" if config["training"].get("critic_action_representation") == "executed_route" else ""
        report = run_mainline(config, args.output or ROOT/"results"/("mainline_check" if args.mode == "mainline-check" else "mainline_6mhz"+suffix),
                              smoke=args.mode == "mainline-check", pretrain_only=args.pretrain_only,
                              resume_from=args.resume_from)
        print(json.dumps({k: report[k] for k in ("status", "minutes", "mean", "joint_training_started")}, indent=2))
        return
    if args.mode == "routing-check":
        from hrl_uav.bandwidth import run_uniform_check
        reference = args.source.parent/"cpu_check"
        report = run_uniform_check(args.source, reference, args.output or reference/"uniform_control")
        print(json.dumps(report["paired_comparisons"],indent=2))
        return
    if args.mode in ("bandwidth-check", "cpu-check"):
        from hrl_uav.bandwidth import run_bandwidth
        report = run_bandwidth(args.source, args.output or ROOT/"results"/args.mode.replace("-", "_"), cpu_check=args.mode == "cpu-check")
        print(json.dumps({k: report[k] for k in ("status", "paired_mean_ms", "paired_t95_ci_ms", "minutes")}, indent=2))
        return
    if args.mode in ("sac-check", "sac-compare"):
        from hrl_uav.reference import run_reference
        output = args.output or ROOT/"results"/args.mode.replace("-", "_")
        report = run_reference(config, output, smoke=args.mode == "sac-check")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    if args.mode == "smoke":
        config = smoke_config(config)
    output = args.output or ROOT/"results"/f"{args.method}_{args.mode}"
    report = run(config, output, method=args.method, smoke=args.mode=="smoke")
    print(json.dumps({"status":report["status"],"minutes":report["minutes"],"stage_audits":report["stage_audits"]},indent=2))


if __name__ == "__main__":
    main()
