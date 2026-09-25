"""Deployment V3 CLI: no training without an explicit train/resume phase."""
import argparse
import json
from pathlib import Path
import sys

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))

from hrl_uav.industrial_deployment_experiment import (METHODS, aggregate_evaluations,
    dry_run, evaluate_job, json_write, load_protocol, resume_job, train_job)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("dry-run", "train", "resume", "evaluate", "summarize"), default="dry-run")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stage", choices=("pretrain", "joint", "both"), default="both")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--extend-stage-seconds", type=float, default=0)
    parser.add_argument("--training-run", type=Path)
    parser.add_argument("--evaluations", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.absolute()

    def rooted(path):
        return path if path is None or path.is_absolute() else root / path

    output, checkpoint = rooted(args.output), rooted(args.checkpoint)
    if args.phase == "resume":
        if checkpoint is None:
            parser.error("resume requires --checkpoint")
        if args.protocol or args.method or args.seed is not None:
            parser.error("resume obtains its sealed protocol/method/seed from the checkpoint")
        result = resume_job(checkpoint, output, root, extend_stage_seconds=args.extend_stage_seconds)
    else:
        if args.extend_stage_seconds:
            parser.error("--extend-stage-seconds is only valid for resume")
        protocol_path = rooted(args.protocol) if args.protocol else SOURCE_ROOT / "configs/industrial_deployment_v3.json"
        protocol, scene = load_protocol(protocol_path, root)
        if args.phase == "train":
            if args.method is None or args.seed is None:
                parser.error("train requires explicit --method and --seed")
            result = train_job(protocol, scene, args.method, args.seed, output, root,
                stage=args.stage, checkpoint=checkpoint)
        elif args.phase == "evaluate":
            if args.training_run is None:
                parser.error("evaluate requires --training-run")
            result = evaluate_job(protocol, scene, rooted(args.training_run), output, root)
        elif args.phase == "summarize":
            if not args.evaluations:
                parser.error("summarize requires explicit --evaluations JSON paths")
            reports = [json.loads(rooted(path).read_text(encoding="utf-8")) for path in args.evaluations]
            result = aggregate_evaluations(protocol, scene, reports)
            output.mkdir(parents=True, exist_ok=False)
            json_write(output / "comparison.json", result)
        else:
            result = dry_run(protocol, scene, output, root, method=args.method, seed=args.seed, stage=args.stage)
    if args.phase == "dry-run":
        brief = {key: result[key] for key in ("phase", "training_started", "torch_imported", "macro_dim",
            "micro_dim", "critic_input_dim", "replay_row_width", "deployment", "budget")}
    elif args.phase == "summarize":
        brief = {key: result[key] for key in ("complete_matrix", "matched_training_seeds", "missing_jobs")}
    else:
        brief = {key: result[key] for key in ("status", "method", "training_seed")}
        if "updates" in result:
            brief["updates"] = result["updates"]
    print(json.dumps(brief, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
