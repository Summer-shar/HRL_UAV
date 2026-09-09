"""Pinned SB3 SAC comparison using the same physical rollout and raw actions.

SB3's optimizer/update implementation is unmodified. Only its input buffer and
network initialization are adapted. This is not an SB3-default benchmark.
"""

import copy
import hashlib
import inspect
import json
from pathlib import Path
import random
import time
from unittest.mock import patch

import gymnasium as gym
import numpy as np
from scipy import stats
import stable_baselines3
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.type_aliases import ReplayBufferSamples
import torch
from torch.nn import functional as F

from .experiment import append_csv, evaluate, isolated_rng, rollout, snapshot, write_json
from .learning import DirectHRL, tensor
from .mdp import QueueMDP


class SpaceOnlyEnv(gym.Env):
    """Spaces for SB3 construction; physical interaction uses QueueMDP.rollout."""

    def __init__(self, state_dim, action_dim):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (state_dim,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (action_dim,), np.float32)

    def reset(self, *, seed=None, options=None):
        raise RuntimeError("Use the shared QueueMDP collector, not SB3.learn()")

    def step(self, action):
        raise RuntimeError("Use the shared QueueMDP collector, not SB3.learn()")


class SB3ReplayView:
    """Private-RNG replay with physical discounts, including deployment gaps."""

    def __init__(self, memory, state_dim, action_dim):
        self.memory, self.n, self.a = memory, state_dim, action_dim

    def convert(self, batch):
        n, a = self.n, self.a
        return ReplayBufferSamples(
            observations=tensor(batch[:, :n]), actions=tensor(batch[:, n:n+a]),
            next_observations=tensor(batch[:, -n:]),
            dones=tensor(np.zeros((len(batch), 1), dtype=np.float32)),
            rewards=tensor(batch[:, n+a:n+a+1]),
            discounts=tensor(batch[:, n+a+1:n+a+2]),
        )

    def sample(self, batch_size, env=None):
        if env is not None:
            raise ValueError("Observation/reward normalization is not part of this comparison")
        return self.convert(self.memory.sample(batch_size))


class ActorView:
    def __init__(self, actor):
        self.action_net = actor
        self.optimizer = actor.optimizer

    def choose_action(self, state, deterministic=False):
        with torch.no_grad():
            action = self.action_net(tensor(state).unsqueeze(0), deterministic=deterministic)
        return action.cpu().numpy().ravel()

    def evaluate(self, states):
        action, logprob = self.action_net.action_log_prob(states)
        return action, logprob.reshape(-1, 1)


class UniformWarmup:
    def __init__(self, actor, agent, seed):
        self.actor, self.agent = actor, agent
        self.rng = np.random.default_rng(seed)

    def __getattr__(self, name):
        return getattr(self.actor, name)

    def choose_action(self, state, deterministic=False):
        if not deterministic and self.agent.replay.memory_counter < self.agent.cfg["replay_warmup"]:
            return self.rng.uniform(-1, 1, self.agent.action_dim).astype(np.float32)
        return self.actor.choose_action(state, deterministic)


def copy_layer(destination, source):
    destination.load_state_dict(source.state_dict())


def copy_networks(model, native):
    source = native.sac_actor.action_net
    for target, layer in zip(
        (model.actor.latent_pi[0], model.actor.latent_pi[2], model.actor.mu, model.actor.log_std),
        (source.in_to_y1, source.y1_to_y2, source.out, source.std_out),
    ):
        copy_layer(target, layer)
    for destination, source in ((model.critic, native.sac_critic.critic_v),
                                (model.critic_target, native.sac_critic.target_critic_v)):
        for layers, target in (
            ((source.in_to_y1, source.y1_to_y2, source.out), destination.qf0),
            ((source.q2_in_to_y1, source.q2_y1_to_y2, source.q2_out), destination.qf1),
        ):
            for index, layer in zip((0, 2, 4), layers):
                copy_layer(target[index], layer)


class SB3Agent(DirectHRL):
    def __init__(self, state_dim, action_dim, macro_dim, config, *, macro_state_dim=None):
        super().__init__(state_dim, action_dim, macro_dim, config, macro_state_dim=macro_state_dim)
        if self.critic_steps_per_update != 1:
            raise ValueError("The SB3 reference does not implement repeated native Critic fits")
        # Library construction consumes random numbers; it must not shift the
        # common initialization's subsequent stochastic action stream.
        with isolated_rng():
            self.model = SAC(
                "MlpPolicy", SpaceOnlyEnv(state_dim, action_dim),
                learning_rate=self.cfg["pretrain_lr"][0],
                buffer_size=1, batch_size=self.cfg["batch_size"],
                learning_starts=self.cfg["replay_warmup"], tau=self.cfg["tau"],
                gamma=self.cfg["gamma_per_resource_frame"],
                ent_coef=f"auto_{self.cfg['initial_alpha']}",
                policy_kwargs={"net_arch": [256, 256]},
                device=next(self.macro_actor.parameters()).device,
            )
            copy_networks(self.model, self)
        for group in self.model.critic.optimizer.param_groups:
            group["eps"] = 1e-5
        self.model.set_logger(configure(folder=None, format_strings=[]))
        self.model.replay_buffer = SB3ReplayView(self.replay, state_dim, action_dim)
        self.sac_actor = ActorView(self.model.actor)
        self.entropy.log_alpha = self.model.log_ent_coef
        self.entropy.optimizer = self.model.ent_coef_optimizer

    def update_sac(self):
        if self.replay.memory_counter < max(self.cfg["batch_size"], self.cfg["replay_warmup"]):
            return
        self.model.train(gradient_steps=1, batch_size=self.cfg["batch_size"])
        self.sac_updates += 1
        self.sac_critic_gradient_steps += 1
        self.sac_target_updates += 1
        self.sac_actor_updates += 1
        values = self.model.logger.name_to_value
        self.last_sac_stats = {
            "critic_loss": float(values["train/critic_loss"]),
            "actor_loss": float(values["train/actor_loss"]),
            "alpha": float(values["train/ent_coef"]),
        }
        if not all(np.isfinite(value) for value in self.last_sac_stats.values()):
            raise FloatingPointError("Non-finite SB3 update; do not modify the reward to continue")


def comparison_config(config, smoke=False):
    config = copy.deepcopy(config)
    reference, cfg = config["sac_reference"], config["training"]
    # The historical implementation audit fixes raw-score Q inputs in both arms.
    # It does not claim to validate the optional physical-routing feature map.
    cfg["critic_action_representation"] = "raw_score"
    cfg.update(
        pretrain_windows=reference["windows"], joint_windows=0,
        sac_update_frames=reference["update_frames"],
        sac_policy_delay=reference["policy_delay"],
        sac_actor_warmup_updates=reference["actor_warmup_updates"],
        pretrain_lr=reference["learning_rates"],
    )
    config["validation_seeds"] = reference["validation_seeds"]
    config["test_seeds"] = reference["test_seeds"]
    config["test_macros"] = reference["test_macros"]
    if smoke:
        cfg.update(pretrain_windows=1, stream_windows=1, window_macros=2,
                   batch_size=16, replay_warmup=16, replay_capacity=1000)
        config["test_seeds"] = config["test_seeds"][:1]
        config["validation_seeds"] = config["validation_seeds"][:1]
        config["test_macros"] = 6
    return config


def make_agent(config, implementation):
    seed = config["training"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    env = QueueMDP(config)
    state = env.reset(seed, config["training"]["window_macros"])
    constructor = SB3Agent if implementation == "sb3" else DirectHRL
    agent = constructor.from_mdp(env, config)
    if config["sac_reference"]["uniform_warmup"]:
        agent.sac_actor = UniformWarmup(agent.sac_actor, agent, seed + 732)
    return agent, state


def fixed_batch_audit(config):
    """Check actual upstream targets and analytic gradients, without training a policy."""
    with isolated_rng():
        native, state = make_agent(config, "native")
        reference, _ = make_agent(config, "sb3")
        rng = np.random.default_rng(777)
        n, a, count = state.size, native.action_dim, 32
        states = np.repeat(state[None, :], count, axis=0)
        states[:, :6] += rng.uniform(0, 2, (count, 6))
        next_states = states + rng.normal(0, 0.01, states.shape)
        actions = rng.uniform(-0.8, 0.8, (count, a))
        rewards = -rng.uniform(1, 100, (count, 1))
        discounts = np.resize([0.99, 0.99**3, 0.0], (count, 1))
        batch = np.concatenate((states, actions, rewards, discounts,
                                np.zeros((count, a)), next_states), axis=1).astype(np.float32)
        data = reference.model.replay_buffer.convert(batch)
        s, ns, act = data.observations, data.next_observations, data.actions
        torch.manual_seed(888)
        action_n, log_n = native.sac_actor.evaluate(s)
        torch.manual_seed(888)
        action_r, log_r = reference.sac_actor.evaluate(s)
        with torch.no_grad():
            qn = native.sac_critic.get_v(s, act)
            qr = reference.model.critic(s, act)
        q_error = max((left-right).abs().max().item() for left, right in zip(qn, qr))
        alpha = reference.model.log_ent_coef.exp().detach()
        torch.manual_seed(889)
        with torch.no_grad():
            native.sac_actor.evaluate(s)
            next_action, next_log = native.sac_actor.evaluate(ns)
            q1, q2 = native.sac_critic.target_critic_v(ns, next_action)
            target_n = data.rewards + data.discounts * (torch.minimum(q1, q2) - alpha * next_log)
        # Capture the target actually used inside the installed SB3.train(),
        # not a second copy of its formula. Its first sample is on current s.
        captured = []
        original_mse = F.mse_loss

        def capture_mse(prediction, target, *args, **kwargs):
            captured.append(target.detach().clone())
            return original_mse(prediction, target, *args, **kwargs)

        q_native = native.sac_critic.get_v(s, act)
        q_reference = reference.model.critic(s, act)
        loss_n = sum(F.mse_loss(q, target_n) for q in q_native)
        loss_r = 0.5 * sum(F.mse_loss(q, target_n) for q in q_reference)
        grad_n = torch.autograd.grad(loss_n, tuple(native.sac_critic.critic_v.parameters()))
        grad_r = torch.autograd.grad(loss_r, tuple(reference.model.critic.parameters()))
        grad_error = max((left-2*right).abs().max().item() for left, right in zip(grad_n, grad_r))
        # Log-probability smoothing differs (clamp vs +epsilon), so compare
        # gradients within tolerance rather than asserting bitwise identity.
        actor_n = (alpha*log_n-torch.minimum(*native.sac_critic.get_v(s, action_n))).mean()
        actor_r = (alpha*log_r-torch.minimum(*reference.model.critic(s, action_r))).mean()
        actor_grad_n = torch.autograd.grad(actor_n, tuple(native.sac_actor.action_net.parameters()))
        actor_grad_r = torch.autograd.grad(actor_r, tuple(reference.model.actor.parameters()))
        actor_grad_error = max((left-right).abs().max().item() for left, right in zip(actor_grad_n, actor_grad_r))
        temperature_n = native.entropy.temperature_loss(log_n)
        temperature_r = -(reference.model.log_ent_coef*(log_r-18).detach()).mean()
        temperature_error = abs(temperature_n.item()-temperature_r.item())
        torch.manual_seed(889)
        with patch.object(reference.model.replay_buffer, "sample", return_value=data), \
                patch("stable_baselines3.sac.sac.F.mse_loss", side_effect=capture_mse):
            reference.model.train(1, count)
        target_error = max((target-target_n).abs().max().item() for target in captured)
        result = {
            "network_q_max_error": q_error,
            "sampled_action_max_error": (action_n-action_r).abs().max().item(),
            "logprob_max_error": (log_n-log_r).abs().max().item(),
            "upstream_target_max_error": target_error,
            "critic_gradient_error_after_known_factor_two": grad_error,
            "actor_gradient_max_error": actor_grad_error,
            "temperature_loss_abs_error": temperature_error,
            "native_critic_loss": loss_n.item(), "sb3_critic_loss": loss_r.item(),
            "discounts_tested": [0.99, 0.99**3, 0.0],
            "tolerance": 0.001,
            "known_variants": [
                "Native critic loss is twice SB3 loss; gradients checked after factor correction.",
                "Native clips gradient norm to 1; upstream SB3 has no such clipping.",
                "Temperature/actor sampling order and squash epsilon differ.",
            ],
        }
        result["passed"] = bool(len(captured) == 2 and all(
            result[key] < result["tolerance"] for key in result if "error" in key
        ))
        return result


def mean_metrics(readings):
    keys = ("task_delay_weighted_ms", "task_delay_high_ms", "task_delay_low_ms",
            "task_completion_rate", "queue_total_mbit", "oldest_age_s", "J", "power_w")
    return {key: float(np.mean([row[key] for row in readings])) for key in keys}


def run_reference(config, output, smoke=False):
    if stable_baselines3.__version__ != "2.7.0":
        raise RuntimeError("This comparison is pinned to stable-baselines3==2.7.0")
    config = comparison_config(config, smoke)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    root = Path(__file__).resolve().parents[1]
    source = Path(inspect.getsourcefile(SAC))
    manifest = {
        "config": config, "scope": "matched-budget fresh micro pretraining; no macro learning",
        "sb3_version": stable_baselines3.__version__, "sb3_sac_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sb3_source": "https://github.com/DLR-RM/stable-baselines3/blob/v2.7.0/stable_baselines3/sac/sac.py",
        "source_hashes": {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                          for folder in ("hrl_uav", "configs", "scripts")
                          for p in (root/folder).rglob("*") if p.suffix in (".py", ".json")},
        "teacher": False, "old_checkpoint": None,
        "differences_from_previous_pilot": "Both arms have 1:1 updates, equal 3e-4 rates, no actor delay, private uniform warmup.",
        "interpretation": "Combined rebaseline, not an isolated test of update density versus the old pilot.",
    }
    write_json(output/"manifest.json", manifest)
    audit = fixed_batch_audit(config)
    write_json(output/"fixed_batch_audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError(f"Fixed-batch gate failed: {audit}")
    print(f"[fixed-batch PASS] target error={audit['upstream_target_max_error']:.8f}", flush=True)
    summaries = {}
    for implementation in ("native", "sb3"):
        arm = output/implementation
        arm.mkdir()
        agent, _ = make_agent(config, implementation)
        initial_actor = snapshot(agent.sac_actor.action_net)
        macro_before = snapshot(agent.macro_actor)
        started = time.perf_counter()
        cfg, reference = config["training"], config["sac_reference"]
        validation = evaluate(config, agent, config["validation_seeds"], cfg["window_macros"], arm, "initial", macro_enabled=False)
        append_csv(arm/"validation.csv", [{"window": 0, **row} for row in validation])
        completed, stopped = 0, False
        for offset in range(0, cfg["pretrain_windows"], cfg["stream_windows"]):
            if time.perf_counter()-started > reference["max_training_seconds_per_arm"]:
                stopped = True
                break
            windows = min(cfg["stream_windows"], cfg["pretrain_windows"]-offset)
            seed = reference["training_scene_seed_base"] + offset//cfg["stream_windows"]
            summary, rows, _, _, _ = rollout(config, agent, seed, windows*cfg["window_macros"],
                                             macro_enabled=False, train=True, window_offset=offset, complete_tail=True)
            append_csv(arm/"training.csv", rows)
            append_csv(arm/"boundaries.csv", [{"seed":seed, "windows":windows,
                       "pending_dropped":summary["final_pending_dropped"],
                       "boundary_discount":summary["boundary_discount_min"]}])
            completed = offset + windows
            if completed % reference["validation_every_windows"] == 0 or completed == cfg["pretrain_windows"]:
                readings = evaluate(config, agent, config["validation_seeds"], cfg["window_macros"], arm,
                                    f"validation_{completed}", macro_enabled=False)
                append_csv(arm/"validation.csv", [{"window": completed, **row} for row in readings])
                print(f"[{implementation} validation {completed}] {mean_metrics(readings)}", flush=True)
        if stopped:
            write_json(arm/"status.json", {"status":"BUDGET_STOP_INCOMPLETE", "windows":completed})
            write_json(output/"report.json", {"status":"BUDGET_STOP_INCOMPLETE", "arm":implementation})
            return {"status":"BUDGET_STOP_INCOMPLETE", "arm":implementation}
        if implementation == "sb3":
            agent.model.save(arm/"final.zip")
            torch.save(agent.replay.state_dict(), arm/"replay.pt")
        else:
            torch.save(agent.checkpoint(config), arm/"final.pt")
        readings = evaluate(config, agent, config["test_seeds"], config["test_macros"], arm, "final", macro_enabled=False)
        append_csv(arm/"final_evaluation.csv", readings)
        actor_delta = max((value.detach().cpu()-initial_actor[name]).abs().max().item()
                          for name,value in agent.sac_actor.action_net.state_dict().items())
        macro_delta = max((value.detach().cpu()-macro_before[name]).abs().max().item()
                          for name,value in agent.macro_actor.state_dict().items())
        growth = []
        for seed in config["test_seeds"]:
            import csv
            with (arm/f"final_{seed}_queues.csv").open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            late = rows[max(len(rows)*2//3, 1):]
            slope = np.polyfit([float(row["time_s"]) for row in late],
                               [float(row["queue_total_mbit"]) for row in late], 1)[0]
            growth.append(float(slope))
        summaries[implementation] = {
            "minutes":(time.perf_counter()-started)/60, "windows":completed,
            "transitions":agent.replay.memory_counter,
            "critic_updates":agent.sac_updates, "actor_updates":agent.sac_actor_updates,
            "actor_max_delta":actor_delta, "macro_max_delta":macro_delta,
            "mean":mean_metrics(readings), "final":readings, "late_queue_slopes_mbit_s":growth,
        }
        write_json(arm/"report.json", summaries[implementation])
        print(f"[{implementation} complete] {summaries[implementation]['mean']}", flush=True)
        del agent
    pairs = np.asarray([b["task_delay_weighted_ms"]-a["task_delay_weighted_ms"]
                        for a,b in zip(summaries["native"]["final"],summaries["sb3"]["final"])])
    ci = None if len(pairs)<2 else stats.t.interval(0.95, len(pairs)-1, loc=pairs.mean(), scale=stats.sem(pairs))
    crn = {}
    for seed in config["test_seeds"]:
        with np.load(output/"native"/f"final_{seed}_trajectory.npz") as a, \
                np.load(output/"sb3"/f"final_{seed}_trajectory.npz") as b:
            crn[str(seed)] = bool(np.array_equal(a["users"], b["users"]) and np.array_equal(a["uav"], b["uav"]))
    report = {
        "status":"SMOKE_ONLY" if smoke else "MICRO_COMPARISON_COMPLETE_NOT_FORMAL",
        "audit":audit, "arms":summaries,
        "sb3_minus_native_delay_ms":pairs.tolist(), "paired_mean_ms":float(pairs.mean()),
        "paired_ci95_ms": None if ci is None else list(map(float,ci)),
        "matched_update_budget":summaries["native"]["critic_updates"]==summaries["sb3"]["critic_updates"]
                                and summaries["native"]["actor_updates"]==summaries["sb3"]["actor_updates"],
        "crn_geometry_equal":crn, "training_seed_count":1,
        "evaluation_seed_count":len(config["test_seeds"]),
        "joint_training_started":False,
        "boundary":"Finite 900s queue slopes are diagnostics, not infinite-horizon stability proof.",
    }
    write_json(output/"report.json", report)
    return {key: report[key] for key in ("status","paired_mean_ms","paired_ci95_ms","matched_update_budget","joint_training_started")}
