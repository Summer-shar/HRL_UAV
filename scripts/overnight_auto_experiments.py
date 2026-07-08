import argparse
import csv
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = RESULTS_DIR / "_logs"
SUMMARY_PATH = RESULTS_DIR / "overnight_auto_summary.csv"


BASE_ARGS = [
    "--eval_interval",
    "50",
    "--eval_episodes",
    "2",
    "--trajectory_plot_interval",
    "100",
    "--cpu_ghz",
    "8",
    "--traffic_scale",
    "1.0",
    "--load_scale",
    "1.0",
    "--sac_batch_size",
    "256",
    "--sac_update_freq",
    "100",
    "--sac_gradient_steps",
    "1",
    "--ppo_update_episode_window",
    "20",
]


@dataclass
class Trial:
    name: str
    episodes: int
    args: list[str]
    source_trial: str = ""

    @property
    def result_dir(self) -> Path:
        return RESULTS_DIR / self.name

    @property
    def eval_csv(self) -> Path:
        return self.result_dir / "eval_central_macro.csv"


def parse_eval_csv(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        return {}

    def f(row: dict, key: str, default: float = 0.0) -> float:
        try:
            return float(row.get(key, default))
        except (TypeError, ValueError):
            return default

    best = min(rows, key=lambda row: f(row, "J", float("inf")))
    last = rows[-1]
    return {
        "best_episode": int(f(best, "episode")),
        "best_J": f(best, "J", float("inf")),
        "best_score": f(best, "score"),
        "last_episode": int(f(last, "episode")),
        "last_J": f(last, "J", float("inf")),
        "last_score": f(last, "score"),
        "last_u2g": f(last, "u2g"),
        "last_coop": f(last, "coop_ratio"),
        "last_low_ms": f(last, "task_delay_low_ms"),
        "last_high_ms": f(last, "task_delay_high_ms"),
        "last_energy": f(last, "energy"),
        "last_cpu_sat": f(last, "cpu_saturation"),
    }


def append_summary(trial: Trial, status: str, metrics: dict, elapsed_s: float) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = SUMMARY_PATH.exists()
    fields = [
        "time",
        "trial",
        "source_trial",
        "status",
        "episodes",
        "elapsed_min",
        "best_episode",
        "best_J",
        "last_episode",
        "last_J",
        "last_u2g",
        "last_coop",
        "last_high_ms",
        "last_low_ms",
        "last_energy",
        "last_cpu_sat",
    ]
    row = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "trial": trial.name,
        "source_trial": trial.source_trial,
        "status": status,
        "episodes": trial.episodes,
        "elapsed_min": round(elapsed_s / 60.0, 3),
    }
    row.update(metrics)
    with SUMMARY_PATH.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})


def run_trial(trial: Trial, force: bool = False) -> tuple[str, dict]:
    if trial.eval_csv.exists() and not force:
        metrics = parse_eval_csv(trial.eval_csv)
        return "skipped", metrics

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOG_DIR / f"{trial.name}.out.log"
    err_path = LOG_DIR / f"{trial.name}.err.log"
    cmd = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "scripts" / "train_centralized_macro.py"),
        "--episodes",
        str(trial.episodes),
        "--results_subdir",
        trial.name,
        *BASE_ARGS,
        *trial.args,
    ]
    start = time.time()
    print(f"\n[RUN] {trial.name}")
    print("      " + " ".join(cmd))
    with out_path.open("w", encoding="utf-8") as out_fp, err_path.open(
        "w", encoding="utf-8"
    ) as err_fp:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=out_fp,
            stderr=err_fp,
            text=True,
        )
    elapsed = time.time() - start
    metrics = parse_eval_csv(trial.eval_csv)
    status = "ok" if proc.returncode == 0 and metrics else f"failed:{proc.returncode}"
    append_summary(trial, status, metrics, elapsed)
    print(
        f"[DONE] {trial.name} status={status} "
        f"best_J={metrics.get('best_J')} last_J={metrics.get('last_J')} "
        f"elapsed={elapsed / 60.0:.1f}min"
    )
    return status, metrics


def make_pilot_trials(episodes: int) -> list[Trial]:
    return [
        Trial(
            name=f"auto_fixeddemand_base_p{episodes}",
            episodes=episodes,
            args=[
                "--cpu_alloc_mode",
                "fixed_demand",
                "--advisor_steps",
                "8",
            ],
        ),
        Trial(
            name=f"auto_prioritydemand_base_p{episodes}",
            episodes=episodes,
            args=[
                "--cpu_alloc_mode",
                "priority_demand",
                "--advisor_steps",
                "8",
            ],
        ),
        Trial(
            name=f"auto_prioritydemand_smallcomm_p{episodes}",
            episodes=episodes,
            args=[
                "--cpu_alloc_mode",
                "priority_demand",
                "--advisor_steps",
                "8",
                "--macro_comm_weight",
                "0.3",
            ],
        ),
        Trial(
            name=f"auto_prioritydemand_strongadvisor_p{episodes}",
            episodes=episodes,
            args=[
                "--cpu_alloc_mode",
                "priority_demand",
                "--advisor_steps",
                "16",
                "--advisor_final_weight",
                "0.2",
                "--advisor_decay_episodes",
                "300",
            ],
        ),
        Trial(
            name=f"auto_fixeddemand_noadvisor_p{episodes}",
            episodes=episodes,
            args=[
                "--cpu_alloc_mode",
                "fixed_demand",
                "--advisor_weight",
                "0.0",
                "--advisor_final_weight",
                "0.0",
                "--advisor_steps",
                "1",
            ],
        ),
    ]


def make_full_trial(source: Trial, episodes: int) -> Trial:
    suffix = source.name
    if "_p" in suffix:
        suffix = suffix.rsplit("_p", 1)[0]
    return Trial(
        name=f"{suffix}_best{episodes}",
        episodes=episodes,
        args=list(source.args),
        source_trial=source.name,
    )


def run_oracle_for_trial(trial: Trial) -> None:
    detail = trial.result_dir / "eval_central_macro_detail.npz"
    traj = trial.result_dir / "trajectory_snapshots" / "trajectory_final.npz"
    if not detail.exists() or not traj.exists():
        print(f"[ORACLE-SKIP] Missing detail or trajectory for {trial.name}")
        return
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "oracle_position_search.py"),
        "--result_dir",
        trial.name,
        "--detail_npz",
        str(detail),
        "--trajectory_npz",
        str(traj),
        "--random_starts",
        "128",
        "--seed",
        "5",
        "--min_separation_m",
        "80",
        "--output_dir",
        str(trial.result_dir / "oracle_positions_sep80"),
    ]
    print(f"[ORACLE] {trial.name}")
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot_episodes", type=int, default=150)
    parser.add_argument("--full_episodes", type=int, default=300)
    parser.add_argument("--target_j", type=float, default=0.52723)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    print(f"[INFO] Python: {sys.executable}")
    print(f"[INFO] Target best_J < {args.target_j:.5f}")
    best_trial = None
    best_metrics = None

    for trial in make_pilot_trials(args.pilot_episodes):
        status, metrics = run_trial(trial, force=args.force)
        if status.startswith("failed") or not metrics:
            continue
        if best_metrics is None or metrics["best_J"] < best_metrics["best_J"]:
            best_trial = trial
            best_metrics = metrics
        if metrics["best_J"] < args.target_j:
            print(f"[TARGET] Hit target in pilot: {trial.name}")
            run_oracle_for_trial(trial)
            return 0

    if best_trial is None:
        print("[FAIL] No successful pilot trial.")
        return 1

    print(
        f"[SELECT] {best_trial.name} best_J={best_metrics['best_J']:.6f}; "
        f"running full {args.full_episodes} episodes."
    )
    full_trial = make_full_trial(best_trial, args.full_episodes)
    status, metrics = run_trial(full_trial, force=args.force)
    if metrics:
        run_oracle_for_trial(full_trial)
    if metrics and metrics["best_J"] < args.target_j:
        print(f"[TARGET] Hit target in full run: {full_trial.name}")
        return 0
    print("[DONE] Finished configured overnight search without hitting target.")
    return 0 if not status.startswith("failed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
