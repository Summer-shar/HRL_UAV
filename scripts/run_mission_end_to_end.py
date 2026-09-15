#!/usr/bin/env python3
"""
补全端到端对比实验：
1. Hover + SAC: 悬停 + SAC路由
2. Hover + CVX: 悬停 + CVX路由
3. PPO120 + CVX: 分开训练的PPO + CVX路由

运行方式:
D:\Anaconda3\envs\pytorch-gpu\python.exe scripts/run_mission_end_to_end.py
"""
import sys
from pathlib import Path
from contextlib import ExitStack
from types import SimpleNamespace
import csv

import numpy as np
import torch
from unittest.mock import patch

# 添加项目根目录到路径
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from hrl_uav.agents.sac_agent import Memory
from hrl_uav.env import UAV_MEC_Env
from hrl_uav.mission_paper_comparison import PPO_SEEDS, SCENES, METRICS
from hrl_uav.mission_ppo_stage4a import MissionMacroPPO, Stage4Meter, stage_config, collect_stream, summarize
from hrl_uav.mission_xu_cvx_r import evaluate_cvx, save_cvx, static_model
from hrl_uav.trajectory_check import ReadOnlySAC, forbidden, no_learning
from hrl_uav.mission_smoke import load_source, load_spec, isolated_rng
from hrl_uav.mission_ppo_qualification import restore_boundary, read_json, verify_transaction, transaction
from hrl_uav.resumption import capture_rng, state_equal

HISTORY = ROOT / "results/mission_ppo_stage4a_qualification"


class _DummyOptimizer:
    """dummy optimizer for evaluate_cvx compatibility"""
    def step(self):
        pass

class _EmptyActor:
    """空的actor对象，用于parameter_delta计算"""
    def state_dict(self):
        return {}


class SimpleHoverPolicy:
    """简单的悬停策略 - 不移动"""
    def __init__(self, config=None):
        self.config = config or {}
        self.meter = None
        self.seed = 0
        self.ppo_updates = 0
        # 用于summarize函数的parameter_delta计算 - 使用空actor
        self.initial_actor = _EmptyActor()
        self.macro_actor = _EmptyActor()
        # 用于evaluate_cvx的optimizer patch
        self.ppo_actor_optimizer = _DummyOptimizer()
        self.ppo_critic_optimizer = _DummyOptimizer()

    def ppo_state(self):
        return {}

    def update_ppo(self, *args, **kwargs):
        pass

    def macro_action(self, state, enabled=True, deterministic=True):
        return np.zeros(6, dtype=np.float32), 0.0

    def set_meter(self, meter):
        self.meter = meter


def evaluate_hover_sac(policy, sac, scene, macros=90):
    """运行Hover + SAC评估（简化版，不做复杂的PPO检查）"""
    if scene not in SCENES or macros not in (2, 90):
        raise ValueError("Only the two declared development scenes are allowed")

    meter = Stage4Meter(policy.config)
    view = SimpleNamespace(sac=sac, macro_action=policy.macro_action)

    with ExitStack() as stack:
        stack.enter_context(isolated_rng())
        stack.enter_context(no_learning(sac))
        stack.enter_context(torch.no_grad())
        stack.enter_context(patch.object(Memory, "__init__", forbidden))
        stack.enter_context(patch.object(policy, "update_ppo", forbidden))
        for attr in ("ppo_actor_optimizer", "ppo_critic_optimizer"):
            opt = getattr(policy, attr, None)
            if opt is not None:
                stack.enter_context(patch.object(opt, "step", forbidden))

        summary, _, macro_rows, observations, windows, actions = collect_stream(
            policy.config, view, scene, train=False, macros=macros, meter_factory=lambda _: meter)

    row, tracking = summarize(summary, meter, policy, scene)
    row.update({k: v for k, v in tracking[0].items() if k not in ("policy", "seed", "uav")})
    row["u2u_tx_time_s"] = sum(r["tx_time_s"] for r in meter.link_readings)
    row["u2u_rate_time_weighted_mbps"] = row["tx_served_mbit"] / max(row["u2u_tx_time_s"], 1e-12)

    # 验证指标完整性
    for key in METRICS:
        if not np.isfinite(row[key]):
            raise AssertionError(f"Missing/nonfinite metric: {key}")

    # 更新元数据
    row.update(condition="hover_sac", macro="Hover", micro="SAC",
               ppo_seed=0, ppo_updates_during_evaluation=0, sac_updates_during_evaluation=0,
               training=False, reused=False, load_multiplier=1.0, bandwidth_mhz=6.0)

    return row, meter, tracking


def save_hover_sac(directory, result):
    """保存Hover + SAC结果"""
    row, meter, tracking = result
    import json

    # 保存summary
    summary = {k: v for k, v in row.items() if not k.startswith("std_")}
    summary["scene"] = tracking[0].get("scene", 0)
    summary["seed"] = tracking[0].get("seed", 0)

    (directory / "summary.json").write_text(json.dumps(summary, indent=2))
    (directory / "crn.json").write_text(json.dumps([]))
    (directory / "complete.json").write_text(json.dumps({"complete": True}))


def load_sac_and_ppo():
    """加载SAC和PPO120模型"""
    qualification = read_json(HISTORY / "manifest.json")
    lock = qualification["lock"]

    # 加载SAC模型
    spec = load_spec()
    saved, _ = load_source(spec)
    sac = ReadOnlySAC(saved)

    # 加载PPO policies
    policies = {}
    for seed in PPO_SEEDS:
        cfg = stage_config(saved["config"], seed)
        if cfg != lock["configs"][str(seed)]:
            raise ValueError(f"PPO config mismatch for seed {seed}")
        folder = HISTORY / f"seed{seed}/boundary040"
        verify_transaction(folder)
        payload = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=False)
        policy = MissionMacroPPO(cfg, seed, sac)
        if restore_boundary(policy, payload, lock) != 40:
            raise ValueError("Not the original window120 endpoint")
        policies[seed] = policy

    return sac, policies


def run_hover_sac(output_dir, sac, ppo_policies, seeds=PPO_SEEDS, scenes=SCENES):
    """运行 Hover + SAC 实验"""
    print("\n" + "=" * 60)
    print("Experiment 1: Hover + SAC")
    print("=" * 60)

    rows = []
    for seed in seeds:
        for scene in scenes:
            print(f"  Seed {seed}, Scene {scene}...", end=" ", flush=True)

            # 使用PPO的config来确保环境正确初始化
            ppo = ppo_policies[seed]

            # 创建Hover策略，使用PPO的config
            hover = SimpleHoverPolicy(config=ppo.config)

            result = evaluate_hover_sac(hover, sac, scene, macros=90)
            row = result[0]

            directory = output_dir / f"hover_sac_seed{seed}_scene{scene}"
            with transaction(directory) as d:
                save_hover_sac(d, result)

            rows.append(row)
            print(f"J={row['J']:.4f}")

    return rows


def run_hover_cvx(output_dir, sac, ppo_policies, seeds=PPO_SEEDS, scenes=SCENES):
    """运行 Hover + CVX 实验"""
    print("\n" + "=" * 60)
    print("Experiment 2: Hover + CVX")
    print("=" * 60)

    rows = []
    for seed in seeds:
        for scene in scenes:
            print(f"  Seed {seed}, Scene {scene}...", end=" ", flush=True)

            # 使用PPO的config来确保环境正确初始化
            ppo = ppo_policies[seed]
            hover = SimpleHoverPolicy(config=ppo.config)

            result, actor = evaluate_cvx(hover, sac, scene, macro_label="Hover")
            row = result[0]

            directory = output_dir / f"hover_cvx_seed{seed}_scene{scene}"
            with transaction(directory) as d:
                save_cvx(d, result, actor)

            rows.append(row)
            print(f"J={row['J']:.4f}")

    return rows


def run_ppo120_cvx(output_dir, sac, ppo_policies, seeds=PPO_SEEDS, scenes=SCENES):
    """运行 PPO120 + CVX 实验"""
    print("\n" + "=" * 60)
    print("Experiment 3: PPO120 + CVX (PPO trained separately)")
    print("=" * 60)

    rows = []
    for seed in seeds:
        for scene in scenes:
            print(f"  Seed {seed}, Scene {scene}...", end=" ", flush=True)

            result, actor = evaluate_cvx(ppo_policies[seed], sac, scene)
            row = result[0]

            directory = output_dir / f"ppo120_cvx_seed{seed}_scene{scene}"
            with transaction(directory) as d:
                save_cvx(d, result, actor)

            rows.append(row)
            print(f"J={row['J']:.4f}")

    return rows


def summarize_results(output_dir, all_rows):
    """汇总结果"""
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    # 按方法分组
    methods = {}
    for row in all_rows:
        key = (row.get('macro', 'unknown'), row.get('micro', 'unknown'))
        if key not in methods:
            methods[key] = []
        methods[key].append(row['J'])

    print("\n| Macro | Micro | J (mean ± std) |")
    print("|-------|-------|-----------------|")
    for (macro, micro), js in sorted(methods.items()):
        mean_j = np.mean(js)
        std_j = np.std(js) if len(js) > 1 else 0
        print(f"| {macro} | {micro} | {mean_j:.4f} ± {std_j:.4f} |")


def main():
    output_dir = ROOT / "results/mission_end_to_end_comparison"
    output_dir.mkdir(exist_ok=True, parents=True)

    # 加载模型
    print("Loading models...")
    sac, ppo_policies = load_sac_and_ppo()
    print(f"  Loaded SAC and {len(ppo_policies)} PPO policies")

    all_rows = []

    # 实验1: Hover + SAC
    hover_sac_rows = run_hover_sac(output_dir, sac, ppo_policies)
    all_rows.extend(hover_sac_rows)

    # 实验2: Hover + CVX
    hover_cvx_rows = run_hover_cvx(output_dir, sac, ppo_policies)
    all_rows.extend(hover_cvx_rows)

    # 实验3: PPO120 + CVX
    ppo120_cvx_rows = run_ppo120_cvx(output_dir, sac, ppo_policies)
    all_rows.extend(ppo120_cvx_rows)

    # 保存结果
    if all_rows:
        # 收集所有字段的并集
        keys = list(set().union(*(row.keys() for row in all_rows)))
        with open(output_dir / "results.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_rows)

    # 汇总
    summarize_results(output_dir, all_rows)

    print("\n" + "=" * 60)
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
