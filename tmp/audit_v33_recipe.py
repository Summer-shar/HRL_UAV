"""Resolve the archived no-teacher v33 recipe without launching training."""

import argparse
import hashlib
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tmp/v33_no_qa_sources"
sys.path[:0] = [str(SOURCE), str(SOURCE / "scripts")]
sys.argv = [str(SOURCE / "scripts/run_experiment.py"), "--config",
            str(SOURCE / "configs/tcom_v33_soft_energy_random_mobility.json")]


class NoSolverImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"compare_sca", "cvxpy", "clarabel", "compare_xu2024_sca"}:
            raise RuntimeError(f"Optimizer import forbidden in no-QA audit: {fullname}")
        return None


sys.meta_path.insert(0, NoSolverImports())

import numpy as np
import torch

import run_experiment as recipe
import train_centralized_macro as micro_trainer
import train_centralized_td3 as joint_trainer
import compare_common as common
import hrl_uav.agents.sac_agent as sac_module


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=serial) + "\n", encoding="utf-8")


def state_hash(actor):
    hashed = hashlib.sha256()
    for key, value in sorted(actor.action_net.state_dict().items()):
        hashed.update(key.encode())
        hashed.update(value.detach().cpu().numpy().tobytes())
    return hashed.hexdigest()


def parse(module, experiment):
    sys.argv = [experiment["script"], *experiment["args"]]
    return module.parse_args()


def main():
    output = ROOT / "results/v33_no_qa_recipe_audit"
    if (output / "report.json").exists():
        raise FileExistsError("Refusing to overwrite a completed audit")
    output.mkdir(exist_ok=True)
    main_hash_before = hashlib.sha256((ROOT / "configs/main.json").read_bytes()).hexdigest()
    stage = recipe.stage_config("formal")
    micro_exp = recipe.build_micro_pretrain("formal", 1, stage, "cpu")
    joint_exp = next(exp for exp in recipe.build_experiments("formal", [1], stage, "cpu")
                     if exp["method"] == "joint_td3_sac")
    micro_args = parse(micro_trainer, micro_exp)
    joint_args = parse(joint_trainer, joint_exp)
    assert micro_args.micro_solver == "sac"
    assert micro_args.freeze_macro and micro_args.frozen_macro_policy == "zero"
    assert not micro_args.micro_ckpt and not micro_args.micro_replay
    assert micro_args.advisor_weight == micro_args.advisor_final_weight == 0
    assert not joint_args.freeze_micro
    assert joint_args.micro_ckpt == recipe.micro_checkpoint_path("formal", 1)
    assert joint_args.micro_replay == recipe.micro_replay_path("formal", 1)
    assert joint_args.resume_micro_optimizer
    save(output / "resolved_micro_args.json", vars(micro_args))
    save(output / "resolved_joint_args.json", vars(joint_args))
    save(output / "archived_config.json", recipe.PROTOCOL)
    save(output / "dry_run_commands.json", [micro_exp, joint_exp])

    torch.set_num_threads(1)
    sac_module.device = torch.device("cpu")
    common.set_global_seeds(1)
    env = common.make_env_from_args(micro_args, record_trajectory=False)
    micro_trainer.configure_macro_objective(env, micro_args, shaping_weight=0.0)
    actor = sac_module.Actor(env.sac_obs_dim, env.micro_action_space.shape[0])
    actor.action_net.eval().requires_grad_(False)
    original_hash = state_hash(actor)
    policy = micro_trainer.sac_micro_policy(actor)
    observed_actions = []

    def observe(current, deterministic):
        action = policy(current, deterministic)
        assert action.shape == (18,) and np.all(np.isfinite(action))
        assert np.max(np.abs(action)) <= 1
        observed_actions.append(action.copy())
        return action

    eval_args = argparse.Namespace(**vars(micro_args))
    eval_args.eval_episodes = 1
    eval_args.eval_base_seed = 51001
    eval_args.macro_steps = 1
    with torch.inference_mode():
        metrics = common.evaluate_policy(
            env, eval_args,
            lambda current, local, deterministic: np.zeros((current.N, 2), np.float32),
            observe,
        )
    assert state_hash(actor) == original_hash
    assert len(observed_actions) == 18
    assert abs(env.env._current_time() - 10) < 1e-6
    assert np.isfinite(metrics["J"])

    fields = ("episodes", "sac_batch_size", "sac_update_freq", "sac_gradient_steps",
              "sac_policy_delay", "sac_actor_warmup_updates", "sac_actor_lr",
              "sac_critic_lr", "sac_alpha_lr", "sac_initial_alpha", "eval_episodes",
              "eval_base_seed", "micro_ckpt", "micro_replay", "resume_micro_optimizer")
    sources = ["configs/tcom_v33_soft_energy_random_mobility.json", "scripts/run_experiment.py",
               "scripts/train_centralized_macro.py", "scripts/train_centralized_td3.py",
               "scripts/compare_common.py", "hrl_uav/env.py", "hrl_uav/hrl_wrappers.py",
               "hrl_uav/routing.py", "hrl_uav/agents/sac_agent.py"]
    report = {
        "scope": "Archived recipe resolution and 10-second random-Actor inference contract only",
        "source_commit": subprocess.check_output(["git", "rev-parse", "ac1d108"], cwd=ROOT, text=True).strip(),
        "source_note": "Earliest repository snapshot containing the v33 record; no saved v33 checkpoint was recovered.",
        "source_sha256": {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() for name in sources},
        "training_performed": False, "historical_performance_rerun": False,
        "teacher_or_optimizer_called": False, "optimizer_imports_blocked": True,
        "micro_state_dim": int(env.sac_obs_dim), "micro_action_dim": int(env.micro_action_space.shape[0]),
        "micro_args": {key: getattr(micro_args, key, None) for key in fields},
        "joint_args": {key: getattr(joint_args, key, None) for key in fields},
        "capacity_audit": recipe.compute_capacity_audit(),
        "memory_capacity": micro_trainer.SAC_MEMORY_CAPACITY,
        "memory_warmup": micro_trainer.SAC_MEMORY_WARMUP,
        "micro_gamma": micro_trainer.SAC_GAMMA,
        "reward": {"delay_norm": env.J_delay_norm, "energy_norm": env.J_energy_norm,
                   "component_clip": env.J_cost_clip, "overflow_threshold_mbits": env.max_Q_threshold,
                   "overflow_weight": env.overflow_penalty_weight},
        "smoke": {"seconds": 10, "service_actor_calls": len(observed_actions),
                  "weights_unchanged": True, "finite_J": True},
        "main_config_unchanged": main_hash_before == hashlib.sha256((ROOT / "configs/main.json").read_bytes()).hexdigest(),
    }
    save(output / "report.json", report)
    print(json.dumps(report, indent=2, default=serial))


if __name__ == "__main__":
    main()
