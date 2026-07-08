"""
论文实验批量运行脚本:
- 5 种方法 × 3 seeds = 15 个实验
- 统一场景: zone125_55_35, cpu=8GHz, priority_demand
- 每个实验 300 episodes, 每 50 ep 评估一次
- 结果放在 results/paper_batch/ 下

用法:
    python scripts/run_paper_experiments.py              # 全部跑
    python scripts/run_paper_experiments.py --method hrl # 只跑 HRL
    python scripts/run_paper_experiments.py --seed 1     # 只跑 seed=1
    python scripts/run_paper_experiments.py --dry-run    # 只打印命令不执行
"""
import argparse
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# ===== 实验配置 =====
SEEDS = [1, 42, 123]
EPISODES = 300
MACRO_STEPS = 30
MICRO_STEPS = 100
EVAL_INTERVAL = 50
EVAL_EPISODES = 5

# 统一场景参数
COMMON_ARGS = [
    "--zone_loads", "12.5,5.5,3.5",
    "--cpu_ghz", "8",
    "--cpu_alloc_mode", "priority_demand",
    "--load_scale", "1.0",
    "--traffic_profile", "normal",
    "--macro_steps", str(MACRO_STEPS),
    "--micro_steps", str(MICRO_STEPS),
    "--eval_interval", str(EVAL_INTERVAL),
    "--eval_episodes", str(EVAL_EPISODES),
    "--macro_load_state", "1",
    "--macro_flow_state", "1",
]


def get_experiments():
    """Define all experiment configurations."""
    experiments = []

    for seed in SEEDS:
        seed_str = str(seed)
        base_subdir = f"paper_batch/seed{seed}"

        # 1. HRL (MAPPO + SAC) - proposed method
        experiments.append({
            "name": f"hrl_s{seed}",
            "method": "hrl",
            "script": os.path.join(SCRIPT_DIR, "train_hrl.py"),
            "args": COMMON_ARGS + [
                "--episodes", str(EPISODES),
                "--seed", seed_str,
                "--results_subdir", f"{base_subdir}/hrl",
            ],
        })

        # 2. Fixed-position + SAC (ablation: trajectory has no effect)
        experiments.append({
            "name": f"fixed_pos_s{seed}",
            "method": "fixed_pos",
            "script": os.path.join(SCRIPT_DIR, "compare_fixed_pos.py"),
            "args": COMMON_ARGS + [
                "--episodes", str(EPISODES),
                "--seed", seed_str,
                "--results_subdir", f"{base_subdir}/fixed_pos",
            ],
        })

        # 3. No-coop (MAPPO trajectory + local-only processing)
        experiments.append({
            "name": f"no_coop_s{seed}",
            "method": "no_coop",
            "script": os.path.join(SCRIPT_DIR, "compare_no_coop.py"),
            "args": COMMON_ARGS + [
                "--episodes", str(EPISODES),
                "--seed", seed_str,
                "--results_subdir", f"{base_subdir}/no_coop",
            ],
        })

        # 4. Random (random trajectory + random offload, lower bound)
        experiments.append({
            "name": f"random_s{seed}",
            "method": "random",
            "script": os.path.join(SCRIPT_DIR, "compare_random.py"),
            "args": COMMON_ARGS + [
                "--episodes", str(EPISODES),
                "--seed", seed_str,
                "--results_subdir", f"{base_subdir}/random",
            ],
        })

        # 5. Centralized macro-only (MAPPO trajectory + heuristic micro)
        experiments.append({
            "name": f"macro_only_s{seed}",
            "method": "macro_only",
            "script": os.path.join(SCRIPT_DIR, "train_centralized_macro.py"),
            "args": COMMON_ARGS + [
                "--episodes", str(EPISODES),
                "--seed", seed_str,
                "--results_subdir", f"{base_subdir}/macro_only",
                "--train_load_min", "0.9",
                "--train_load_max", "1.1",
            ],
        })

    return experiments


def run_experiment(exp, python_cmd, dry_run=False):
    """Run a single experiment."""
    cmd = [python_cmd, exp["script"]] + exp["args"]
    cmd_str = " ".join(cmd)

    if dry_run:
        print(f"[DRY-RUN] {exp['name']}: {cmd_str}")
        return 0

    print(f"\n{'='*70}")
    print(f"[START] {exp['name']} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  CMD: {cmd_str}")
    print(f"{'='*70}")

    t0 = time.time()
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAIL(rc={result.returncode})"
    print(f"[{status}] {exp['name']} finished in {elapsed:.1f}s")

    # Save log
    log_dir = os.path.join(PROJECT_ROOT, "results", "paper_batch", "_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{exp['name']}.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# {exp['name']} | {status} | {elapsed:.1f}s\n")
        f.write(f"# CMD: {cmd_str}\n\n")
        f.write(result.stdout or "")

    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run paper experiments batch")
    parser.add_argument("--method", type=str, default="",
                        help="Only run experiments for this method (hrl/fixed_pos/no_coop/random/macro_only)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Only run experiments for this seed (0=all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--python", type=str, default=sys.executable,
                        help="Python interpreter to use")
    args = parser.parse_args()

    experiments = get_experiments()

    # Filter by method
    if args.method:
        experiments = [e for e in experiments if e["method"] == args.method]
    # Filter by seed
    if args.seed:
        experiments = [e for e in experiments if f"_s{args.seed}" in e["name"]]

    print(f"Paper Experiments Batch Runner")
    print(f"  Total experiments: {len(experiments)}")
    print(f"  Methods: {sorted(set(e['method'] for e in experiments))}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Episodes per run: {EPISODES}")
    print(f"  Scenario: zone125_55_35, cpu=8GHz, priority_demand")
    print()

    results = []
    for i, exp in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] Running: {exp['name']}")
        rc = run_experiment(exp, args.python, dry_run=args.dry_run)
        results.append((exp["name"], rc))

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY:")
    for name, rc in results:
        status = "✓" if rc == 0 else "✗"
        print(f"  {status} {name}")
    failed = sum(1 for _, rc in results if rc != 0)
    if failed:
        print(f"\n  {failed}/{len(results)} experiments FAILED")
    else:
        print(f"\n  All {len(results)} experiments completed successfully")


if __name__ == "__main__":
    main()
