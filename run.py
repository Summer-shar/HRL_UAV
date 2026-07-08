import os
import subprocess
import sys
import threading
import time
from pathlib import Path


# =========================
# 只需要改这里
# =========================

# 可选:
# train_sac          训练你的 Proposed HRL-SAC
# train_ppo          训练 PPO baseline
# train_no_coop      训练 No Cooperation baseline
# eval_random        评估 Random baseline
# eval_sca           评估 SCA baseline
# run_all_one_cpu    一次跑完一个 CPU 频率下的所有对比
# run_all_cpu_sweep  一次跑完 CPU_LIST 里所有频率的所有对比
MODE = "menu"

# Extra convenient modes:
# run_sac_cpu_sweep      train proposed SAC over CPU_LIST at LOAD_SCALE
# run_sac_load_sweep     train proposed SAC over LOAD_LIST at CPU_GHZ
# run_sac_grid           train proposed SAC over CPU_LIST x LOAD_LIST
# run_sac_traffic_sweep  legacy traffic_scale sweep; load stays fixed

# 单次实验 CPU 频率，单位 GHz
CPU_GHZ = 8

# CPU 扫描模式 (run_*_cpu_sweep) 会用这个列表
CPU_LIST = [8, 9, 10, 11, 12, 14]

# 批量模式要跑哪些算法。
# 训练类算法会比较耗时；Random 和 SCA 不需要训练，只做评估。
BATCH_MODES = [
    "train_sac",
    "train_ppo",
    "train_no_coop",
    "eval_random",
    "eval_sca",
]

# 批量模式的并行度。
# 1 = 完全顺序；2 或 3 = 同时开几个进程。
MAX_PARALLEL_JOBS = 3

# In parallel mode, child-process output is still saved to results/_logs,
# and this switch also mirrors it to the current Run window.
LIVE_CHILD_LOGS = True

# 如果只有一块 GPU，建议保持 False。
# False 时：SAC/PPO/NoCoop 训练最多只会同时跑一个，
# Random/SCA 这类评估可以和训练并行。
# 如果你有多块 GPU，或者确认显存够，可以改 True。
ALLOW_PARALLEL_TRAINING = False

# 是否让所有强化学习类对比共用同一个宏观 MAPPO。
# 如果还没有宏观 checkpoint，会先用 train_sac 跑出一个。
USE_SHARED_MACRO = False
AUTO_TRAIN_SHARED_MACRO = True

# 留空时默认查找 results/sac_cpu{CPU_GHZ}/checkpoints/best.pt，
# 如果 best.pt 不存在，再查找 final.pt。
SHARED_MACRO_CKPT = ""

EPISODES = 1500
MACRO_STEPS = 30
MICRO_STEPS = 100
EVAL_INTERVAL = 100
EVAL_EPISODES = 2
FIXED_BASELINE_EVAL_EPISODES = 30
TRAJECTORY_PLOT_INTERVAL = 100
SAC_BATCH_SIZE = 256
SAC_UPDATE_FREQ = 100
SAC_GRADIENT_STEPS = 1

TRAFFIC_PROFILE = "normal"
TRAFFIC_SCALE = 1.0
TRAFFIC_LIST = [0.8, 1.0, 1.2]
LOAD_SCALE = 1.0
LOAD_LIST = [0.8, 0.9, 1.0, 1.1, 1.2]
ZONE_LOADS = "12.5,5.5,3.5"
ARRIVAL_DT = 0.1
UPLINK_BANDWIDTH_MHZ = 20.0
U2U_BANDWIDTH_MHZ = 3.0
MICRO_DIAG_BIAS = 0.5
MICRO_ACTION_SCALE = 1.0
MICRO_STATE_MODE = "pressure"
MACRO_COMM_WEIGHT = 2.0
MACRO_COMM_FLOW_EMA = 0.8
MACRO_FLOW_STATE = 1
MACRO_LOAD_STATE = 1
MACRO_U2G_COST_WEIGHT = 2.0
MACRO_ZONE_INGRESS_WEIGHT = 0.5
MACRO_ACTION_MODE = "polar"
CPU_ALLOC_MODE = "fixed_demand"
KKT_V = 1.0
FIXED_CPU_UTIL = 0.5


ROOT = Path(__file__).resolve().parent
TRAINING_MODES = {"train_sac", "train_ppo", "train_no_coop"}
RL_MACRO_MODES = {"train_sac", "train_ppo", "train_no_coop"}


def _compact_float(value: float) -> str:
    return f"{float(value):g}"


def cpu_alloc_suffix() -> str:
    suffix = f"{CPU_ALLOC_MODE}_v{_compact_float(KKT_V)}"
    if CPU_ALLOC_MODE == "fixed_average":
        suffix += f"_u{_compact_float(FIXED_CPU_UTIL)}"
    return suffix


def zone_load_suffix() -> str:
    if not str(ZONE_LOADS).strip():
        return ""
    values = [
        _compact_float(float(item.strip()))
        for item in str(ZONE_LOADS).split(",")
        if item.strip()
    ]
    if not values:
        return ""
    return "zl" + "-".join(values)


def scenario_suffix() -> str:
    parts = [
        f"ul{_compact_float(UPLINK_BANDWIDTH_MHZ)}",
        f"arr{_compact_float(ARRIVAL_DT)}",
        cpu_alloc_suffix(),
    ]
    zl = zone_load_suffix()
    if zl:
        parts.append(zl)
    return "_".join(parts)


def result_name(
    prefix: str,
    cpu_ghz: float,
    traffic_scale: float = None,
    load_scale: float = None,
) -> str:
    cpu_text = f"{cpu_ghz:g}"
    if traffic_scale is None:
        traffic_scale = TRAFFIC_SCALE
    if load_scale is None:
        load_scale = LOAD_SCALE
    traffic_text = _compact_float(traffic_scale)
    load_text = _compact_float(load_scale)
    parts = [f"{prefix}_cpu{cpu_text}", f"ls{load_text}"]
    if abs(float(traffic_scale) - 1.0) > 1e-12:
        parts.append(f"ts{traffic_text}")
    parts.append(scenario_suffix())
    return "_".join(parts)


def base_args(cpu_ghz, traffic_scale=None, load_scale=None):
    if traffic_scale is None:
        traffic_scale = TRAFFIC_SCALE
    if load_scale is None:
        load_scale = LOAD_SCALE
    args = [
        "--cpu_ghz",
        str(cpu_ghz),
        "--macro_steps",
        str(MACRO_STEPS),
        "--micro_steps",
        str(MICRO_STEPS),
        "--eval_episodes",
        str(EVAL_EPISODES),
        "--sac_batch_size",
        str(SAC_BATCH_SIZE),
        "--sac_update_freq",
        str(SAC_UPDATE_FREQ),
        "--sac_gradient_steps",
        str(SAC_GRADIENT_STEPS),
        "--trajectory_plot_interval",
        str(TRAJECTORY_PLOT_INTERVAL),
        "--traffic_profile",
        TRAFFIC_PROFILE,
        "--traffic_scale",
        str(traffic_scale),
        "--load_scale",
        str(load_scale),
        "--arrival_dt",
        str(ARRIVAL_DT),
        "--uplink_bandwidth_mhz",
        str(UPLINK_BANDWIDTH_MHZ),
        "--u2u_bandwidth_mhz",
        str(U2U_BANDWIDTH_MHZ),
        "--cpu_alloc_mode",
        CPU_ALLOC_MODE,
        "--kkt_v",
        str(KKT_V),
        "--fixed_cpu_util",
        str(FIXED_CPU_UTIL),
        "--micro_diag_bias",
        str(MICRO_DIAG_BIAS),
        "--micro_action_scale",
        str(MICRO_ACTION_SCALE),
        "--micro_state_mode",
        MICRO_STATE_MODE,
        "--macro_comm_weight",
        str(MACRO_COMM_WEIGHT),
        "--macro_comm_flow_ema",
        str(MACRO_COMM_FLOW_EMA),
        "--macro_flow_state",
        str(MACRO_FLOW_STATE),
        "--macro_load_state",
        str(MACRO_LOAD_STATE),
        "--macro_u2g_cost_weight",
        str(MACRO_U2G_COST_WEIGHT),
        "--macro_zone_ingress_weight",
        str(MACRO_ZONE_INGRESS_WEIGHT),
        "--macro_action_mode",
        MACRO_ACTION_MODE,
    ]
    if str(ZONE_LOADS).strip():
        args.extend(["--zone_loads", ZONE_LOADS])
    return args


def default_macro_ckpt_candidates(cpu_ghz, traffic_scale=None, load_scale=None):
    if traffic_scale is None:
        traffic_scale = TRAFFIC_SCALE
    if load_scale is None:
        load_scale = LOAD_SCALE
    return [
        ROOT / "results" / result_name("sac", cpu_ghz, traffic_scale, load_scale) / "checkpoints" / "best.pt",
        ROOT / "results" / result_name("sac", cpu_ghz, traffic_scale, load_scale) / "checkpoints" / "final.pt",
    ]


def find_existing_macro_ckpt(cpu_ghz, traffic_scale=None, load_scale=None):
    if SHARED_MACRO_CKPT:
        path = Path(SHARED_MACRO_CKPT)
        if not path.is_absolute():
            path = ROOT / path
        return path if path.exists() else None

    for path in default_macro_ckpt_candidates(cpu_ghz, traffic_scale, load_scale):
        if path.exists():
            return path
    return None


def expected_macro_ckpt(cpu_ghz, traffic_scale=None, load_scale=None):
    if SHARED_MACRO_CKPT:
        path = Path(SHARED_MACRO_CKPT)
        if not path.is_absolute():
            path = ROOT / path
        return path
    return ROOT / "results" / result_name("sac", cpu_ghz, traffic_scale, load_scale) / "checkpoints" / "final.pt"


def add_macro_reuse_args(command, macro_ckpt, freeze_macro):
    if not macro_ckpt:
        return command

    command = [
        *command,
        "--macro_ckpt",
        str(macro_ckpt),
        "--macro_deterministic",
    ]
    if freeze_macro:
        command.append("--freeze_macro")
    return command


def build_command(
    mode,
    cpu_ghz,
    traffic_scale=None,
    load_scale=None,
    macro_ckpt=None,
    freeze_macro=False,
):
    py = sys.executable
    if traffic_scale is None:
        traffic_scale = TRAFFIC_SCALE
    if load_scale is None:
        load_scale = LOAD_SCALE

    if mode == "train_sac":
        return add_macro_reuse_args([
            py,
            str(ROOT / "scripts" / "train_hrl.py"),
            "--episodes",
            str(EPISODES),
            "--eval_interval",
            str(EVAL_INTERVAL),
            "--results_subdir",
            result_name("sac", cpu_ghz, traffic_scale, load_scale),
            *base_args(cpu_ghz, traffic_scale, load_scale),
        ], macro_ckpt, freeze_macro)

    if mode == "train_ppo":
        return add_macro_reuse_args([
            py,
            str(ROOT / "scripts" / "compare_micro_ppo.py"),
            "--episodes",
            str(EPISODES),
            "--eval_interval",
            str(EVAL_INTERVAL),
            "--results_subdir",
            result_name("micro_ppo", cpu_ghz, traffic_scale, load_scale),
            *base_args(cpu_ghz, traffic_scale, load_scale),
        ], macro_ckpt, freeze_macro)

    if mode == "train_no_coop":
        return add_macro_reuse_args([
            py,
            str(ROOT / "scripts" / "compare_no_coop.py"),
            "--episodes",
            str(EPISODES),
            "--eval_interval",
            str(EVAL_INTERVAL),
            "--results_subdir",
            result_name("no_coop", cpu_ghz, traffic_scale, load_scale),
            *base_args(cpu_ghz, traffic_scale, load_scale),
        ], macro_ckpt, freeze_macro)

    if mode == "eval_random":
        return [
            py,
            str(ROOT / "scripts" / "compare_random.py"),
            "--episodes",
            "1",
            "--eval_interval",
            "0",
            "--results_subdir",
            result_name("random", cpu_ghz, traffic_scale, load_scale),
            *base_args(cpu_ghz, traffic_scale, load_scale),
            "--eval_episodes",
            str(FIXED_BASELINE_EVAL_EPISODES),
        ]

    if mode == "eval_sca":
        return [
            py,
            str(ROOT / "scripts" / "compare_sca.py"),
            "--results_subdir",
            result_name("sca", cpu_ghz, traffic_scale, load_scale),
            *base_args(cpu_ghz, traffic_scale, load_scale),
            "--eval_episodes",
            str(FIXED_BASELINE_EVAL_EPISODES),
        ]

    if mode == "plot_quick":
        return [
            py,
            str(ROOT / "scripts" / "plot_quick_results.py"),
            "--smooth_window",
            "5",
        ]

    raise ValueError(f"Unknown MODE: {mode}")


def build_job(
    mode,
    cpu_ghz,
    traffic_scale=None,
    load_scale=None,
    macro_ckpt=None,
    freeze_macro=False,
    depends_on_path=None,
):
    if traffic_scale is None:
        traffic_scale = TRAFFIC_SCALE
    if load_scale is None:
        load_scale = LOAD_SCALE
    if mode == "plot_quick":
        label = mode
    else:
        label = result_name(mode, cpu_ghz, traffic_scale, load_scale)
    if freeze_macro:
        label += "_fixed_macro"
    return {
        "label": label,
        "mode": mode,
        "cpu_ghz": cpu_ghz,
        "traffic_scale": traffic_scale,
        "load_scale": load_scale,
        "is_training": mode in TRAINING_MODES,
        "depends_on_path": Path(depends_on_path) if depends_on_path else None,
        "command": build_command(
            mode,
            cpu_ghz,
            traffic_scale=traffic_scale,
            load_scale=load_scale,
            macro_ckpt=macro_ckpt,
            freeze_macro=freeze_macro,
        ),
    }


def build_jobs_for_modes(modes, cpu_ghz, traffic_scale=None, load_scale=None):
    if traffic_scale is None:
        traffic_scale = TRAFFIC_SCALE
    if load_scale is None:
        load_scale = LOAD_SCALE
    if not USE_SHARED_MACRO:
        return [
            build_job(mode, cpu_ghz, traffic_scale, load_scale)
            for mode in modes
        ]

    existing_macro_ckpt = find_existing_macro_ckpt(
        cpu_ghz,
        traffic_scale,
        load_scale,
    )
    expected_ckpt = expected_macro_ckpt(cpu_ghz, traffic_scale, load_scale)
    macro_ckpt_for_reuse = existing_macro_ckpt
    macro_will_be_trained = False
    jobs = []

    if macro_ckpt_for_reuse is None and AUTO_TRAIN_SHARED_MACRO:
        jobs.append(build_job("train_sac", cpu_ghz, traffic_scale, load_scale))
        macro_ckpt_for_reuse = expected_ckpt
        macro_will_be_trained = True

    for mode in modes:
        if macro_will_be_trained and mode == "train_sac":
            continue

        if mode in RL_MACRO_MODES and macro_ckpt_for_reuse is not None:
            jobs.append(
                build_job(
                    mode,
                    cpu_ghz,
                    traffic_scale,
                    load_scale,
                    macro_ckpt=macro_ckpt_for_reuse,
                    freeze_macro=True,
                    depends_on_path=macro_ckpt_for_reuse
                    if macro_will_be_trained
                    else None,
                )
            )
            continue

        jobs.append(build_job(mode, cpu_ghz, traffic_scale, load_scale))

    return jobs


def build_jobs_for_mode(mode, cpu_ghz, traffic_scale=None, load_scale=None):
    return build_jobs_for_modes([mode], cpu_ghz, traffic_scale, load_scale)


def dedupe_jobs(jobs):
    unique_jobs = []
    seen_labels = set()
    for job in jobs:
        label = job["label"]
        if label in seen_labels:
            continue
        seen_labels.add(label)
        unique_jobs.append(job)
    return unique_jobs


def build_all_cpu_sweep_jobs():
    jobs = []
    for cpu_ghz in CPU_LIST:
        jobs.extend(
            build_jobs_for_modes(BATCH_MODES, cpu_ghz, TRAFFIC_SCALE, LOAD_SCALE)
        )
    return jobs


def build_all_load_sweep_jobs():
    jobs = []
    for load_scale in LOAD_LIST:
        jobs.extend(
            build_jobs_for_modes(BATCH_MODES, CPU_GHZ, TRAFFIC_SCALE, load_scale)
        )
    return jobs


def build_all_traffic_sweep_jobs():
    jobs = []
    for traffic_scale in TRAFFIC_LIST:
        jobs.extend(
            build_jobs_for_modes(BATCH_MODES, CPU_GHZ, traffic_scale, LOAD_SCALE)
        )
    return jobs


def build_jobs():
    if MODE == "plot_quick":
        return [build_job("plot_quick", CPU_GHZ, TRAFFIC_SCALE, LOAD_SCALE)]

    if MODE == "run_sac_cpu_sweep":
        jobs = []
        for cpu_ghz in CPU_LIST:
            jobs.extend(
                build_jobs_for_mode("train_sac", cpu_ghz, TRAFFIC_SCALE, LOAD_SCALE)
            )
        return jobs

    if MODE == "run_sac_load_sweep":
        jobs = []
        for load_scale in LOAD_LIST:
            jobs.extend(
                build_jobs_for_mode("train_sac", CPU_GHZ, TRAFFIC_SCALE, load_scale)
            )
        return jobs

    if MODE == "run_sac_traffic_sweep":
        jobs = []
        for traffic_scale in TRAFFIC_LIST:
            jobs.extend(
                build_jobs_for_mode("train_sac", CPU_GHZ, traffic_scale, LOAD_SCALE)
            )
        return jobs

    if MODE == "run_sac_grid":
        jobs = []
        for load_scale in LOAD_LIST:
            for cpu_ghz in CPU_LIST:
                jobs.extend(
                    build_jobs_for_mode(
                        "train_sac",
                        cpu_ghz,
                        TRAFFIC_SCALE,
                        load_scale,
                    )
                )
        return jobs

    if MODE == "run_all_one_cpu":
        return build_jobs_for_modes(
            BATCH_MODES,
            CPU_GHZ,
            TRAFFIC_SCALE,
            LOAD_SCALE,
        )

    if MODE in {"run_all_cpu_sweep", "paper_cpu_sweep"}:
        return build_all_cpu_sweep_jobs()

    if MODE in {"run_all_load_sweep", "paper_load_sweep"}:
        return build_all_load_sweep_jobs()

    if MODE in {"run_all_traffic_sweep", "paper_traffic_sweep"}:
        return build_all_traffic_sweep_jobs()

    if MODE == "paper_fast":
        return dedupe_jobs([
            *build_all_cpu_sweep_jobs(),
            *build_all_load_sweep_jobs(),
        ])

    return build_jobs_for_mode(MODE, CPU_GHZ, TRAFFIC_SCALE, LOAD_SCALE)


def can_launch(job, active_jobs):
    depends_on_path = job.get("depends_on_path")
    if depends_on_path is not None and not depends_on_path.exists():
        return False

    if ALLOW_PARALLEL_TRAINING:
        return True

    if not job["is_training"]:
        return True

    return not any(active_item[1]["is_training"] for active_item in active_jobs)


def run_jobs_sequential(jobs):
    for idx, job in enumerate(jobs, start=1):
        command = job["command"]
        print(f"\n[{idx}/{len(jobs)}] Running {job['label']}:")
        print(" ".join(command))
        subprocess.run(command, cwd=ROOT, check=True)


def forward_process_output(process, job, log_file):
    for line in process.stdout:
        log_file.write(line)
        log_file.flush()
        if LIVE_CHILD_LOGS:
            text = line.rstrip()
            if text:
                print(f"[{job['label']}] {text}", flush=True)
    log_file.flush()


def run_jobs_parallel(jobs):
    log_dir = ROOT / "results" / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    pending = list(jobs)
    active = []
    completed = 0

    while pending or active:
        while pending and len(active) < MAX_PARALLEL_JOBS:
            launch_index = None
            for idx, job in enumerate(pending):
                if can_launch(job, active):
                    launch_index = idx
                    break

            if launch_index is None:
                break

            job = pending.pop(launch_index)
            log_path = log_dir / f"{job['label']}.log"
            log_file = open(log_path, "w", encoding="utf-8")
            print(f"\n[START] {job['label']}")
            print(" ".join(job["command"]))
            print(f"Log: {log_path}")
            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                job["command"],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            reader_thread = threading.Thread(
                target=forward_process_output,
                args=(process, job, log_file),
                daemon=True,
            )
            reader_thread.start()
            active.append((process, job, log_file, log_path, reader_thread))

        time.sleep(2.0)

        still_active = []
        for process, job, log_file, log_path, reader_thread in active:
            return_code = process.poll()
            if return_code is None:
                still_active.append((process, job, log_file, log_path, reader_thread))
                continue

            reader_thread.join(timeout=5.0)
            log_file.close()
            completed += 1
            if return_code != 0:
                print(f"\n[FAILED] {job['label']} | log: {log_path}")
                for active_process, _, active_log_file, _, active_reader in still_active:
                    active_process.terminate()
                    active_reader.join(timeout=2.0)
                    active_log_file.close()
                raise subprocess.CalledProcessError(return_code, job["command"])

            print(f"\n[DONE] {job['label']} ({completed}/{len(jobs)})")
            print(f"Log: {log_path}")

        active = still_active


MENU_OPTIONS = [
    ("1", "train_sac", "train proposed SAC once"),
    ("2", "run_sac_load_sweep", "train proposed over LOAD_LIST"),
    ("3", "run_sac_cpu_sweep", "train proposed over CPU_LIST"),
    ("4", "run_sac_grid", "train proposed over CPU_LIST x LOAD_LIST (very slow)"),
    ("5", "train_ppo", "train PPO baseline once"),
    ("6", "run_all_one_cpu", "run all algorithms at one CPU/load"),
    ("7", "run_all_cpu_sweep", "run all algorithms over CPU_LIST"),
    ("8", "eval_random", "evaluate random baseline"),
    ("9", "eval_sca", "evaluate SCA baseline"),
    ("10", "plot_quick", "plot quick result figures"),
    ("11", "run_all_load_sweep", "run all algorithms over LOAD_LIST"),
    ("12", "paper_cpu_sweep", "paper CPU sweep, fixed load"),
    ("13", "paper_load_sweep", "paper load sweep, fixed CPU"),
    ("14", "paper_fast", "paper fast: CPU sweep + load sweep, no grid"),
    ("15", "run_sac_traffic_sweep", "legacy train proposed over TRAFFIC_LIST"),
]


def _ask_float(prompt, default):
    while True:
        raw = input(f"{prompt} [{_compact_float(default)}]: ").strip()
        if not raw:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print("Please enter a number.")


def _ask_float_list(prompt, default):
    default_text = ",".join(_compact_float(value) for value in default)
    while True:
        raw = input(f"{prompt} [{default_text}]: ").strip()
        if not raw:
            return list(default)
        try:
            return [
                float(part.strip())
                for part in raw.replace("，", ",").split(",")
                if part.strip()
            ]
        except ValueError:
            print("Please enter a comma-separated number list, e.g. 8,12,16.")


def configure_from_menu():
    global MODE
    global CPU_GHZ, CPU_LIST
    global TRAFFIC_SCALE, TRAFFIC_LIST
    global LOAD_SCALE, LOAD_LIST

    if MODE != "menu":
        return False

    print("\nChoose what to run:")
    for key, _, label in MENU_OPTIONS:
        print(f"  {key}. {label}")

    choice = input("Select number or mode name [1]: ").strip() or "1"
    mode_by_key = {key: mode for key, mode, _ in MENU_OPTIONS}
    MODE = mode_by_key.get(choice, choice)

    cpu_single_modes = {
        "train_sac",
        "train_ppo",
        "train_no_coop",
        "eval_random",
        "eval_sca",
        "run_sac_load_sweep",
        "run_sac_traffic_sweep",
        "run_all_one_cpu",
        "run_all_load_sweep",
        "run_all_traffic_sweep",
        "paper_load_sweep",
        "paper_traffic_sweep",
        "paper_fast",
    }
    cpu_list_modes = {
        "run_sac_cpu_sweep",
        "run_sac_grid",
        "run_all_cpu_sweep",
        "paper_cpu_sweep",
        "paper_fast",
    }
    load_list_modes = {
        "run_sac_load_sweep",
        "run_all_load_sweep",
        "paper_load_sweep",
        "paper_fast",
        "run_sac_grid",
    }
    single_load_modes = {
        "train_sac",
        "train_ppo",
        "train_no_coop",
        "eval_random",
        "eval_sca",
        "run_sac_cpu_sweep",
        "run_all_one_cpu",
        "run_all_cpu_sweep",
        "paper_cpu_sweep",
        "paper_fast",
        "run_sac_traffic_sweep",
        "run_all_traffic_sweep",
        "paper_traffic_sweep",
    }
    traffic_list_modes = {
        "run_sac_traffic_sweep",
        "run_all_traffic_sweep",
        "paper_traffic_sweep",
    }
    single_traffic_modes = {
        "train_sac",
        "train_ppo",
        "train_no_coop",
        "eval_random",
        "eval_sca",
        "run_sac_cpu_sweep",
        "run_sac_load_sweep",
        "run_all_one_cpu",
        "run_all_cpu_sweep",
        "run_all_load_sweep",
        "paper_cpu_sweep",
        "paper_load_sweep",
        "paper_fast",
        "run_sac_grid",
    }
    if MODE in cpu_single_modes:
        CPU_GHZ = _ask_float("CPU GHz", CPU_GHZ)
    if MODE in cpu_list_modes:
        CPU_LIST = _ask_float_list("CPU list GHz", CPU_LIST)
    if MODE in single_load_modes:
        LOAD_SCALE = _ask_float("Load scale", LOAD_SCALE)
    if MODE in load_list_modes:
        LOAD_LIST = _ask_float_list("Load scale list", LOAD_LIST)
    if MODE in single_traffic_modes:
        TRAFFIC_SCALE = _ask_float("Traffic scale", TRAFFIC_SCALE)
    if MODE in traffic_list_modes:
        TRAFFIC_LIST = _ask_float_list("Traffic scale list", TRAFFIC_LIST)

    return True


def confirm_menu_jobs(jobs):
    print("\nPlanned jobs:")
    for idx, job in enumerate(jobs, start=1):
        print(f"  {idx}. {job['label']}")


def main():
    menu_used = configure_from_menu()
    jobs = build_jobs()
    if menu_used:
        confirm_menu_jobs(jobs)
    if len(jobs) == 1 or MAX_PARALLEL_JOBS <= 1:
        run_jobs_sequential(jobs)
    else:
        run_jobs_parallel(jobs)


if __name__ == "__main__":
    main()
