"""Six locked development comparisons of the explicit Xu-CVX-R specialization."""

from contextlib import ExitStack
from dataclasses import asdict
import gc
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import patch

import cvxpy
import numpy as np
import torch

from .agents.sac_agent import Memory
from .env import UAV_MEC_Env
from .experiment import append_csv, isolated_rng, write_json
from .mission_paper_comparison import (HISTORY, PPO_SEEDS, SCENES, evaluate_case,
    assert_saved_prefix, save_result, table_row)
from .mission_pilot import crn_records
from .mission_ppo_qualification import METRICS, read_json, restore_boundary, transaction, verify_transaction
from .mission_ppo_stage4a import MissionMacroPPO, Stage4Meter, collect_stream, stage_config, summarize
from .mission_provenance import physics_provenance
from .mission_smoke import ROOT, load_spec, load_source
from .priority_grid import file_hash
from .resumption import capture_rng, state_equal
from .scene_prescreen import assert_crn_equal
from .trajectory_check import ReadOnlySAC, forbidden, no_learning
from .xu_cvx_r import StaticModel, XuCVXRouting, SLICES, decode_observation, hypothetical_full_link_resources


PRIOR = ROOT / "results/mission_paper_comparison_20260914"
NEW_FILES = ("hrl_uav/xu_cvx_r.py", "hrl_uav/mission_xu_cvx_r.py",
             "scripts/run_mission_xu_cvx_r.py", "tests/test_xu_cvx_r.py", "docs/mission_xu_cvx_r.md")
PDF = Path("C:/Users/PC/Desktop/[11] UAV-Coop-MEC  Trajectory_Planning_and_Resource_Allocation_for_Multi-UAV_Cooperative_Computation.pdf")


def static_model(meter):
    env, reward = meter.env, meter.reward_config
    if (not env.continuous_mission or env.wireless_model != "psd_per_uav_budget"
            or env.B_u2u_total != 6e6 or env.cpu_alloc_mode != "priority_demand"
            or env.cpu_power_cap_w != 0 or meter.observation_layout != "split_v1"
            or meter.observe_routing_ready):
        raise ValueError("This bounded comparison requires the unchanged base mission model")
    return StaticModel(env.B_u2u_total, env.noise_psd_w_per_hz, env.u2u_tx_budget_w,
        env.u2u_reference_bandwidth_hz, env.u2u_reference_power_w, env.C, env.kappa,
        env.w_D_high, env.w_D_low, reward["delay_weight"], reward["energy_weight"],
        reward["holding_norm_task_s"], reward["energy_norm_j"])


def evaluate_cvx(policy, sac, scene, macros=90, macro_label=None):
    if scene not in SCENES or macros not in (2, 90):
        raise ValueError("Only the two declared development scenes and horizons are allowed")
    if macro_label is None:
        macro_label = "PPO120"
    meter = Stage4Meter(policy.config)
    actor = XuCVXRouting(static_model(meter))
    view = SimpleNamespace(sac=SimpleNamespace(sac_actor=actor), macro_action=policy.macro_action)
    before, rng = policy.ppo_state(), capture_rng()
    with ExitStack() as stack:
        stack.enter_context(isolated_rng())
        stack.enter_context(no_learning(sac))
        stack.enter_context(torch.no_grad())
        stack.enter_context(patch.object(Memory, "__init__", forbidden))
        stack.enter_context(patch.object(policy, "update_ppo", forbidden))
        stack.enter_context(patch.object(UAV_MEC_Env, "set_external_cpu_frequencies", forbidden))
        for optimizer in (policy.ppo_actor_optimizer, policy.ppo_critic_optimizer):
            stack.enter_context(patch.object(optimizer, "step", forbidden))
        summary, _, macros_rows, observations, windows, actions = collect_stream(
            policy.config, view, scene, train=False, macros=macros, meter_factory=lambda _: meter)
    if not state_equal(before, policy.ppo_state()) or not state_equal(rng, capture_rng()):
        raise AssertionError("Frozen PPO or evaluation RNG state changed")
    if dict(meter.micro_feature_slices) != {k: list(v) for k, v in SLICES.items()}:
        raise AssertionError("CVX input decoding does not match the real SAC layout")
    if meter.env._external_cpu_frequency_override is not None:
        raise AssertionError("CPU override unexpectedly active")
    if actor.calls != 20*macros or len(actor.records) != actor.calls:
        raise AssertionError("Solver decision count mismatch")
    np.testing.assert_allclose([a[0].reshape(2, 3, 3) for a in actions], [a[1] for a in actions], atol=1e-7, rtol=0)
    current = meter.micro_observation()
    predicted = hypothetical_full_link_resources(decode_observation(current)["eta"], actor.model)[-1]
    actual_hypothetical = meter.env.compute_u2u_rate_bps(active_link_mask=~np.eye(3, dtype=bool))/1e6
    np.testing.assert_allclose(predicted, actual_hypothetical, rtol=3e-6, atol=1e-6)
    row, tracking = summarize(summary, meter, policy, scene)
    row.update({k: v for k, v in tracking[0].items() if k not in ("policy", "seed", "uav")})
    row["u2u_tx_time_s"] = sum(r["tx_time_s"] for r in meter.link_readings)
    row["u2u_rate_time_weighted_mbps"] = row["tx_served_mbit"]/max(row["u2u_tx_time_s"], 1e-12)
    for key in METRICS:
        if not np.isfinite(row[key]):
            raise AssertionError(f"Missing/nonfinite actual metric {key}")
    for priority in ("high", "low"):
        if row[f"task_arrived_{priority}_count"] != row[f"task_completed_{priority}_count"]+row[f"task_unfinished_{priority}_count"]:
            raise AssertionError("Parent task conservation failed")
    times = np.asarray([r["solve_seconds"] for r in actor.records])
    row.update(condition="base", micro="Xu-CVX-R", macro=macro_label, ppo_seed=policy.seed,
               load_multiplier=1., bandwidth_mhz=6., reused=False, training=False,
               ppo_updates_during_evaluation=0, sac_updates_during_evaluation=0,
               solver_calls=actor.calls, solver_total_seconds=float(times.sum()),
               solver_mean_ms=float(times.mean()*1000), solver_p95_ms=float(np.percentile(times, 95)*1000),
               solver_max_ms=float(times.max()*1000), full_paper_sca_replication=False,
               solver_max_primal_residual=max(r["primal_residual"] for r in actor.records),
               solver_max_cleanup=max(r["max_cleanup"] for r in actor.records))
    checks = dict(**sac.assert_unchanged(), ppo_state_unchanged=True, global_rng_unchanged=True,
        replay_writes=0, replay_samples=0, ppo_update_increment=0, sac_update_increment=0,
        cpu_override_blocked=True, route_checked=True, parent_conservation=True,
        same_observation_layout=True, all_active_prediction_matches_original_formula=True)
    result = row, meter, tracking, macros_rows, windows, observations, actions, checks
    return result, actor


def save_cvx(directory, result, actor):
    save_result(directory, result)
    append_csv(directory / "solver.csv", actor.records)
    np.savez_compressed(directory / "solver_candidates.npz", objective=actor.candidate_costs)
    write_json(directory / "predictor.json", asdict(actor.model))


def summarize_results(output, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    rows = [{key: row.get(key) for key in fields} for row in rows]
    means, summary, pairs = [], [], []
    for method in ("SAC", "Xu-CVX-R"):
        for ppo_seed in PPO_SEEDS:
            group = [r for r in rows if r["micro"] == method and r["ppo_seed"] == ppo_seed]
            if len(group) != 2 or {r["seed"] for r in group} != set(SCENES):
                raise AssertionError("Incomplete seed/scene matrix")
            means.append(dict(micro=method, ppo_seed=ppo_seed,
                              **{k: float(np.mean([r[k] for r in group])) for k in METRICS}))
        group = [r for r in means if r["micro"] == method]
        summary.append(dict(micro=method, ppo_initializations=3, sac_initializations=1, scenes=2,
            **{k: float(np.mean([r[k] for r in group])) for k in METRICS},
            **{f"std_{k}": float(np.std([r[k] for r in group], ddof=1)) for k in METRICS}))
    for seed in PPO_SEEDS:
        for scene in SCENES:
            left = next(r for r in rows if (r["micro"], r["ppo_seed"], r["seed"]) == ("SAC", seed, scene))
            right = next(r for r in rows if (r["micro"], r["ppo_seed"], r["seed"]) == ("Xu-CVX-R", seed, scene))
            pairs.append(dict(ppo_seed=seed, seed=scene, difference="SAC minus Xu-CVX-R",
                              **{f"delta_{k}": left[k]-right[k] for k in METRICS}))
    append_csv(output / "per_seed.csv", rows)
    append_csv(output / "per_training_seed.csv", means)
    append_csv(output / "summary.csv", summary)
    append_csv(output / "paired_deltas.csv", pairs)
    columns = ("J", "J_holding", "J_energy", "task_delay_weighted_ms", "task_delay_high_ms",
               "task_delay_low_ms", "task_completion_high_rate", "task_completion_low_rate",
               "task_unfinished_high_count", "task_unfinished_low_count", "queue_total_mbit", "power_w")
    report = ["# Xu-CVX-R开发对照", "", "## 方法边界", "",
        "这是原论文分片完成时间公式的固定CPU、聚合观测路由适配版，不是原论文联合CPU的完整SCA，也不是QA-PSCA。",
        "固定CPU后，子问题为凸规划；没有SCA迭代。预测模型及其限制见docs/mission_xu_cvx_r.md。",
        "两种路由使用同一PPO120策略权重，但各自状态闭环演化，轨迹不强制相同。PPO原来搭配SAC训练，并未为CVX重新训练。",
        "三个PPO初始化共享SAC101，两个场景是已用开发集；不是三个独立完整方法训练或最终泛化评价。", "",
        "## 实测汇总", "", "|方法|"+"|".join(columns)+"|", "|---|"+"|".join("---" for _ in columns)+"|"]
    for row in summary:
        report.append("|"+row["micro"]+"|"+"|".join(f"{row[k]:.6f}" for k in columns)+"|")
    report += ["", "## 每个PPO初始化的两场景均值", "", "|PPO|方法|J|加权时延/ms|期末队列/Mbit|", "|---|---|---|---|---|"]
    for row in means:
        report.append(f"|{row['ppo_seed']}|{row['micro']}|{row['J']:.6f}|{row['task_delay_weighted_ms']:.2f}|{row['queue_total_mbit']:.3f}|")
    report += ["", "## 逐场景配对差：SAC减Xu-CVX-R", "", "|PPO|场景|ΔJ|Δ加权时延/ms|Δ低优完成率|Δ期末队列/Mbit|", "|---|---|---|---|---|---|"]
    for row in pairs:
        report.append(f"|{row['ppo_seed']}|{row['seed']}|{row['delta_J']:+.6f}|{row['delta_task_delay_weighted_ms']:+.2f}|{row['delta_task_completion_low_rate']:+.6f}|{row['delta_queue_total_mbit']:+.3f}|")
    report += ["", "## 解释限制", "",
        "- 物理J及全部能量来自真实连续900秒仿真，不用求解器预测代价替代。",
        "- 预测使用同一个68维观测和静态物理常量；不读逐任务详情、未来数据或额外routing_ready特征。",
        "- 固定全活跃链路速率仅用于预测，不冒充真实服务速率；真实资源量在links.csv。",
        "- 聚合批次、满频服务时间估计、当前队列等待估计均是近似，忽略新批次之间的争用和未来优先级到达；不能由结果判定原论文SCA优劣。",
        "- 最终论文若需要完整PPO-SCA竞争方法，仍需明确CPU控制范围并独立训练其宏观层，本轮不是该证据。",
        "- 没有调参、学习、replay写入/采样、CPU覆盖或模型修改；未运行保留场景。",
        "- 每次求解的状态、耗时、数值残差、支持集合及完整实际路由记录均保留；全部结果不按优劣删行。", ""]
    (output / "result.md").write_text("\n".join(report), encoding="utf-8")


def run(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory")
    physics = physics_provenance()
    previous = read_json(PRIOR / "manifest.json")
    qualification = read_json(HISTORY / "manifest.json")
    if previous["status"] != "COMPLETE_STOPPED" or not previous["source_unchanged"]:
        raise ValueError("Incomplete prior comparison")
    if qualification["status"] != "COMPLETE_3X120_STOPPED" or not qualification["source_unchanged"]:
        raise ValueError("Incomplete PPO qualification")
    lock = qualification["lock"]
    for name, digest in {**previous["sources_after"], **lock["sources"]}.items():
        if file_hash(ROOT/name) != digest:
            raise ValueError(f"Prior source changed: {name}")
    protected = {name: file_hash(ROOT/name) for name in dict.fromkeys((*previous["sources_after"], *NEW_FILES))}
    evidence = {str(p): file_hash(p) for p in (PRIOR/"manifest.json", HISTORY/"manifest.json", PDF)}
    spec = load_spec()
    saved, _ = load_source(spec)
    with isolated_rng():
        sac = ReadOnlySAC(saved)
        policies = {}
        for seed in PPO_SEEDS:
            cfg = stage_config(saved["config"], seed)
            if cfg != lock["configs"][str(seed)] or cfg != previous["configs"]["base"][str(seed)]:
                raise ValueError("Effective configuration mismatch")
            folder = HISTORY/f"seed{seed}/boundary040"
            verify_transaction(folder)
            verify_transaction(HISTORY/f"seed{seed}/monitor120")
            payload = torch.load(folder/"checkpoint.pt", map_location="cpu", weights_only=False)
            policy = MissionMacroPPO(cfg, seed, sac)
            if restore_boundary(policy, payload, lock) != 40:
                raise ValueError("Not the original window120 endpoint")
            policies[seed] = policy
            evidence[str(folder/"checkpoint.pt")] = file_hash(folder/"checkpoint.pt")
    del saved, payload
    gc.collect()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = dict(status="SMOKE_PENDING", actual_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        worktree=subprocess.check_output(["git", "status", "--short", "--branch"], cwd=ROOT, text=True),
        physics=physics, sources_before=protected, source_checkpoint=spec["source_checkpoint"],
        source_checkpoint_sha256=spec["checkpoint_sha256"], evidence=evidence,
        configs={str(s): policies[s].config for s in PPO_SEEDS},
        protocol=dict(baseline="Xu-CVX-R", full_paper_sca_replication=False, training=False,
            bandwidth_mhz=6, scenes=list(SCENES), ppo_seeds=list(PPO_SEEDS),
            endpoint=120, seconds=900, new_cases=6, reused_sac_cases=6,
            scene_status="reused development scenes", no_best_selection=True,
            no_retraining_of_macro_for_cvx=True),
        runtime=dict(cvxpy=cvxpy.__version__, numpy=np.__version__, torch=str(torch.__version__)),
        results_completed=0, training=False, reserved_scenes_executed=[])
    write_json(output/"manifest.json", manifest)
    rows = []
    try:
        reference = evaluate_case(policies[201], sac, "base", "SAC", 33600, macros=2)
        assert_saved_prefix(reference, HISTORY/"seed201/monitor120/33600")
        result, actor = evaluate_cvx(policies[201], sac, 33600, macros=2)
        assert_crn_equal(crn_records(reference[1]), crn_records(result[1]))
        with transaction(output/"smoke") as directory:
            save_cvx(directory, result, actor)
        del reference, result, actor
        manifest.update(status="RUNNING", smoke_passed=True)
        write_json(output/"manifest.json", manifest)
        print("20s prefix/freeze/CRN smoke passed; starting exactly six900s cases", flush=True)
        for seed in PPO_SEEDS:
            for scene in SCENES:
                source = PRIOR/f"cases/base_PPO120{seed}_SAC_{scene}"
                verify_transaction(source)
                original = read_json(source/"summary.json")
                crn = read_json(source/"crn.json")
                if original["duration_s"] != 900 or original["sac_update_increment"] or original["replay_writes"]:
                    raise ValueError("Incompatible SAC baseline")
                reuse = read_json(source/"reused.json")
                for name, digest in reuse["files"].items():
                    if file_hash(Path(name)) != digest:
                        raise ValueError(f"Original SAC evidence changed: {name}")
                evidence.update(reuse["files"])
                for filename in ("summary.json", "crn.json", "reused.json"):
                    evidence[str(source/filename)] = file_hash(source/filename)
                result, actor = evaluate_cvx(policies[seed], sac, scene)
                assert_crn_equal(crn, crn_records(result[1]))
                if len(crn) != 1800:
                    raise AssertionError("Incomplete CRN history")
                destination = output/f"cases/ppo{seed}_Xu-CVX-R_{scene}"
                with transaction(destination) as directory:
                    save_cvx(directory, result, actor)
                reuse_dir = output/f"cases/ppo{seed}_SAC_{scene}"
                with transaction(reuse_dir) as directory:
                    original["reused"] = True
                    write_json(directory/"summary.json", original)
                    write_json(directory/"crn.json", crn)
                    write_json(directory/"reused.json", dict(source=str(source), original_source=reuse))
                row = table_row(result[0], destination)
                row.update({k: v for k, v in result[0].items() if k.startswith("solver_")})
                rows.extend((table_row(original, reuse_dir), row))
                manifest.update(results_completed=len(rows)//2, elapsed_seconds=time.perf_counter()-started)
                write_json(output/"manifest.json", manifest)
                print(f"Completed {len(rows)//2}/6; PPO{seed} scene{scene}: J={row['J']:.6f}, weighted={row['task_delay_weighted_ms']:.2f}ms", flush=True)
                del result, actor
                gc.collect()
        summarize_results(output, rows)
        if any(file_hash(Path(name)) != digest for name, digest in evidence.items()):
            raise AssertionError("Source evidence changed during evaluation")
        after = {name: file_hash(ROOT/name) for name in protected}
        if after != protected or physics_provenance() != physics:
            raise AssertionError("Source/physics changed during evaluation")
        manifest.update(status="COMPLETE_STOPPED", source_unchanged=True, sources_after=after,
            evidence=evidence, crn_all_equal=True, frozen_sac=sac.assert_unchanged(),
            elapsed_seconds=time.perf_counter()-started, new_evaluations=6, reused_evaluations=6)
        write_json(output/"manifest.json", manifest)
    except BaseException as exc:
        manifest.update(status="FAILED_STOPPED", error=repr(exc), elapsed_seconds=time.perf_counter()-started)
        write_json(output/"manifest.json", manifest)
        raise
    return output


def finalize_completed_results(output):
    """Read-only recovery of six completed cases after the CSV schema error.

    Preserve the failed top-level attempt and all its transactions; write a
    separate analysis directory. Never call an environment or load a policy.
    """
    import ast
    output = Path(output).resolve()
    original = read_json(output/"manifest.json")
    if original["status"] != "FAILED_STOPPED" or original["results_completed"] != 6:
        raise ValueError("This recovery requires all six completed original evaluations")
    if "dict contains fields not in fieldnames" not in original.get("error", ""):
        raise ValueError("Unexpected failure; do not bypass it")
    before_source = output/"audit/mission_xu_cvx_r_before_csv_fix.py.txt"
    old_hash = original["sources_before"]["hrl_uav/mission_xu_cvx_r.py"]
    if file_hash(before_source) != old_hash:
        raise ValueError("Original evaluated source snapshot does not match its locked hash")
    def evaluation_ast(text):
        tree = ast.parse(text)
        tree.body = [n for n in tree.body if not isinstance(n, ast.FunctionDef)
                     or n.name not in ("summarize_results", "finalize_completed_results")]
        return ast.dump(tree, include_attributes=False)
    if evaluation_ast(before_source.read_text(encoding="utf-8")) != evaluation_ast(Path(__file__).read_text(encoding="utf-8")):
        raise ValueError("More than the offline aggregation/recovery code changed")
    for name, digest in original["sources_before"].items():
        if name != "hrl_uav/mission_xu_cvx_r.py" and file_hash(ROOT/name) != digest:
            raise ValueError(f"Locked source changed: {name}")
    if physics_provenance() != original["physics"]:
        raise ValueError("Physical implementation changed")
    for name, digest in original["evidence"].items():
        if file_hash(Path(name)) != digest:
            raise ValueError(f"Prior evidence changed: {name}")
    rows, case_hashes = [], {}
    for seed in PPO_SEEDS:
        for scene in SCENES:
            left = output/f"cases/ppo{seed}_SAC_{scene}"
            right = output/f"cases/ppo{seed}_Xu-CVX-R_{scene}"
            for directory in (left, right):
                verify_transaction(directory)
                row = read_json(directory/"summary.json")
                if row["duration_s"] != 900 or row["seed"] != scene or row["ppo_seed"] != seed:
                    raise ValueError("Case identity mismatch")
                if row["sac_update_increment"] or row["replay_writes"] or row["replay_samples"]:
                    raise ValueError("Learning occurred")
                np.testing.assert_allclose(row["J"], row["J_holding"]+row["J_energy"], rtol=1e-12, atol=1e-10)
                for priority in ("high", "low"):
                    if row[f"task_arrived_{priority}_count"] != row[f"task_completed_{priority}_count"]+row[f"task_unfinished_{priority}_count"]:
                        raise ValueError("Task count mismatch")
                record = table_row(row, directory)
                record.update({k: v for k, v in row.items() if k.startswith("solver_")})
                rows.append(record)
                case_hashes[str(directory/"complete.json")] = file_hash(directory/"complete.json")
            assert_crn_equal(read_json(left/"crn.json"), read_json(right/"crn.json"))
            reuse = read_json(left/"reused.json")["original_source"]
            for name, digest in reuse["files"].items():
                if file_hash(Path(name)) != digest:
                    raise ValueError(f"Original reused SAC result changed: {name}")
            frozen = read_json(right/"freeze_check.json")
            if not all(frozen[k] for k in ("ppo_state_unchanged", "global_rng_unchanged", "cpu_override_blocked", "parent_conservation")):
                raise ValueError("Freeze/integration check did not pass")
    analysis = output/"analysis"
    analysis.mkdir(exist_ok=False)
    summarize_results(analysis, rows)
    write_json(analysis/"manifest.json", dict(status="COMPLETE_OFFLINE_FINALIZATION", new_evaluations=0,
        completed_original_evaluations=6, reused_sac_evaluations=6, original_attempt_status=original["status"],
        original_manifest_sha256=file_hash(output/"manifest.json"), original_source_snapshot_sha256=old_hash,
        current_aggregation_source_sha256=file_hash(Path(__file__)), evaluation_ast_unchanged=True,
        all_other_locked_sources_unchanged=True, original_case_transactions_verified=12,
        case_manifest_hashes=case_hashes, crn_all_equal=True, original_results_not_overwritten=True,
        training=False, environments_run=False, reserved_scenes_executed=[]))
    return analysis
