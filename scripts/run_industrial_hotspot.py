"""Hotspot V4 explicit phases; importing or dry-running never starts training."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))

from hrl_uav.industrial_hotspot_experiment import (METHODS, dry_run, load_protocol,
    resume_job, train_job, digest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("dry-run", "validate", "diagnose", "train", "resume", "evaluate", "worker-evaluate"), default="dry-run")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stage", choices=("pretrain", "joint", "both"), default="both")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--validation-record", type=Path)
    parser.add_argument("--diagnostic-record", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--budget-path", type=Path)
    parser.add_argument("--evaluation-kind", choices=("admission", "joint", "final"), default="final", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.absolute()

    def rooted(value):
        return value if value is None or value.is_absolute() else root / value

    output, checkpoint = rooted(args.output), rooted(args.checkpoint)
    if args.phase == "resume":
        if checkpoint is None or args.protocol is not None or args.method is not None or args.seed is not None:
            parser.error("resume requires --checkpoint and reads its sealed protocol/method/seed")
        result = resume_job(checkpoint, output, root)
    elif args.phase in ("evaluate", "worker-evaluate"):
        if checkpoint is None:
            parser.error("evaluation requires --checkpoint")
        from hrl_uav.industrial_hotspot_checkpoint import load_checkpoint
        from hrl_uav.industrial_hotspot_evaluation import candidate_evaluate, paired_evaluate
        saved = load_checkpoint(checkpoint)
        protocol = saved["protocol"]
        cache = rooted(args.cache_root) or root / "industrial_hotspot_records" / ("cache-" + digest(protocol)[:16])
        final = args.phase == "evaluate" or args.evaluation_kind == "final"
        budget = rooted(args.budget_path) or cache / ("final_budget.json" if final else "budget.json")
        if final:
            state = saved["run_state"]
            if saved["stage"] != "joint" or state["episodes"]["joint"] != protocol["budget"]["joint_episodes"] or state["pending_evaluation"] is not None:
                raise ValueError("Final testing requires the complete joint endpoint and all pending development checks committed")
            result = paired_evaluate(checkpoint, output, root, cache_root=cache, budget_path=budget, panel="final")
        elif args.evaluation_kind == "admission":
            result = candidate_evaluate(checkpoint, output, root, cache_root=cache, budget_path=budget)
        else:
            result = paired_evaluate(checkpoint, output, root, cache_root=cache, budget_path=budget)
    else:
        path = rooted(args.protocol) if args.protocol else SOURCE_ROOT / "configs/industrial_hotspot_v4.json"
        protocol, scene = load_protocol(path, root)
        if args.phase == "dry-run":
            result = dry_run(protocol, scene, output, root, method=args.method, seed=args.seed)
        elif args.phase == "validate":
            script = SOURCE_ROOT / "scripts/validate_industrial_hotspot.py"
            subprocess.run([sys.executable, "-B", str(script), "--project-root", str(root),
                "--protocol", str(path), "--output", str(output)], check=True)
            return
        elif args.phase == "diagnose":
            from hrl_uav.industrial_hotspot_evaluation import diagnose
            cache = rooted(args.cache_root) or root / "industrial_hotspot_records" / ("cache-" + digest(protocol)[:16])
            budget = rooted(args.budget_path) or cache / "budget.json"
            result = diagnose(protocol, scene, output, root, cache_root=cache, budget_path=budget)
        else:
            if args.method is None or args.seed is None:
                parser.error("train requires explicit --method and --seed")
            result = train_job(protocol, scene, args.method, args.seed, output, root, stage=args.stage,
                checkpoint=checkpoint, validation_record=rooted(args.validation_record), diagnostic_record=rooted(args.diagnostic_record))
    brief = {key: result[key] for key in ("status", "phase", "method", "training_seed", "passed", "quality_passed",
        "training_started", "torch_imported", "macro_dim", "micro_dim", "updates") if key in result}
    if args.phase == "dry-run":
        brief["budget"] = result["budget"]
    print(json.dumps(brief, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
