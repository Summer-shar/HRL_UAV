"""Deployment PPO with a post-update KL guard; inherited SAC is unchanged."""
import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .learning import DirectHRL, gae, tensor


def batch_normal_kl(old_mean, old_std, mean, std):
    """Mean over states of KL(old || new), summed over all action coordinates."""
    old_mean, old_std, mean, std = [x.double() for x in (old_mean, old_std, mean, std)]
    return (torch.log(std / old_std)
            + (old_std.square() + (old_mean - mean).square()) / (2 * std.square())
            - .5).sum(-1).mean()


def explained_variance(target, predicted):
    target, predicted = np.asarray(target), np.asarray(predicted)
    variance = float(np.var(target))
    return None if variance <= 1e-12 else float(1 - np.var(target - predicted) / variance)


class DeploymentHRL(DirectHRL):
    """Separate entry point: the previous DirectHRL implementation is untouched."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.cfg.get("ppo_update_guard") != "post_step_batch_kl_v1":
            raise ValueError("Deployment PPO requires post_step_batch_kl_v1")
        if self.cfg["sac_stage_policy"] != "preserve":
            raise ValueError("Deployment stages must preserve SAC learning state")
        self.cfg.setdefault("ppo_guard_retries", 4)
        self.cfg.setdefault("ppo_std_backtracks", 12)
        self.cfg.setdefault("ppo_std_decay_updates", 360)
        if not 1 <= int(self.cfg["ppo_epochs"]) <= 10:
            raise ValueError("PPO permits 1 through 10 fitting epochs")
        if self.cfg["ppo_guard_retries"] != 4 or self.cfg["ppo_std_backtracks"] != 12:
            raise ValueError("The V3 guard uses four retries and twelve std candidates")
        if self.cfg["ppo_std_decay_updates"] <= 0:
            raise ValueError("Exploration decay horizon must be positive")
        if not 0 < self.cfg["ppo_target_kl"] <= .03:
            raise ValueError("The V3 post-step mean KL limit must be in (0, .03]")
        self.ppo_accepted_update_events = 0
        self.ppo_guard_rejected_steps = 0
        self.ppo_guard_attempted_steps = 0
        self.ppo_behavior_version = 0
        self.ppo_critic_gradient_steps = 0

    def guard_state(self):
        return {name: int(getattr(self, name)) for name in (
            "ppo_accepted_update_events", "ppo_guard_rejected_steps",
            "ppo_guard_attempted_steps", "ppo_behavior_version", "ppo_critic_gradient_steps")}

    def restore_guard_state(self, state):
        if set(state) != set(self.guard_state()):
            raise ValueError("Incomplete deployment PPO guard clocks")
        for key, value in state.items():
            if type(value) is not int or value < 0:
                raise ValueError("Invalid PPO guard clock")
            setattr(self, key, value)

    def update_ppo(self, progress=0.0, force=False):
        # progress intentionally does not drive exploration; extending a run cannot rebase it.
        if not self.macro_rollout or (len(self.macro_rollout) < self.cfg["ppo_update_macros"] and not force):
            return
        if self.stage != "joint":
            raise RuntimeError("PPO data or learning is forbidden during hover pretraining")
        states, actions, logprobs, rewards, next_states, continuations = map(np.asarray, zip(*self.macro_rollout))
        if any(not np.isfinite(value).all() for value in (states, actions, logprobs, rewards, next_states, continuations)):
            raise FloatingPointError("Non-finite PPO rollout")
        s, a, old_log = tensor(states), tensor(actions), tensor(logprobs)
        with torch.no_grad():
            values = self.macro_critic(s).ravel().cpu().numpy()
            next_values = self.macro_critic(tensor(next_states)).ravel().cpu().numpy()
            old_mean, old_std = [x.detach().clone() for x in self.macro_actor(s)]
            before = torch.distributions.Normal(old_mean, old_std).log_prob(a).sum(-1)
            pre_error = (before - old_log).abs().max().item()
        if not np.isfinite(pre_error) or pre_error > 1e-4:
            raise RuntimeError(f"PPO behavior log-prob mismatch: {pre_error}")
        advantages, returns_np = gae(rewards, values, next_values, continuations,
                                    self._macro_gamma, self.cfg["gae_lambda"])
        if not np.isfinite(advantages).all() or not np.isfinite(returns_np).all():
            raise FloatingPointError("Non-finite PPO advantage or return")
        adv = tensor((advantages - advantages.mean()) / (advantages.std() + 1e-8))
        returns = tensor(returns_np)
        base_lr, limit = self.cfg["ppo_actor_lr"], self.cfg["ppo_target_kl"]
        for group in self.ppo_actor_optimizer.param_groups:
            group["lr"] = base_lr
        accepted = rejected = attempted = 0
        candidate_lr = base_lr
        attempts, actor_loss_value = [], None
        std_before = self.macro_actor.log_std.detach().clone()
        for epoch in range(self.cfg["ppo_epochs"]):
            actor_snapshot = copy.deepcopy(self.macro_actor.state_dict())
            optimizer_snapshot = copy.deepcopy(self.ppo_actor_optimizer.state_dict())
            step_accepted = False
            for retry in range(self.cfg["ppo_guard_retries"] + 1):
                if retry:
                    self.macro_actor.load_state_dict(actor_snapshot)
                    # Optimizer loading may retain tensor aliases.  Every retry
                    # needs fresh moments so its step cannot mutate the rollback copy.
                    self.ppo_actor_optimizer.load_state_dict(copy.deepcopy(optimizer_snapshot))
                for group in self.ppo_actor_optimizer.param_groups:
                    group["lr"] = candidate_lr
                mean, std = self.macro_actor(s)
                logprob = torch.distributions.Normal(mean, std).log_prob(a).sum(-1)
                ratio = (logprob - old_log).exp()
                objective = torch.minimum(ratio * adv, ratio.clamp(
                    1-self.cfg["ppo_clip"], 1+self.cfg["ppo_clip"]) * adv)
                actor_loss = -objective.mean()
                attempted += 1
                gradient_norm = None
                candidate_kl = float("inf")
                finite_loss = bool(torch.isfinite(actor_loss))
                if finite_loss:
                    self.ppo_actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    gradient = nn.utils.clip_grad_norm_(self.macro_actor.parameters(), self.cfg["grad_clip"])
                    if bool(torch.isfinite(gradient)):
                        gradient_norm = float(gradient)
                        self.ppo_actor_optimizer.step()
                        with torch.no_grad():
                            new_mean, new_std = self.macro_actor(s)
                            candidate_kl = float(batch_normal_kl(old_mean, old_std, new_mean, new_std))
                passed = bool(np.isfinite(candidate_kl) and candidate_kl <= limit + 1e-12)
                attempts.append(dict(epoch=epoch, retry=retry, learning_rate=float(candidate_lr),
                    kl=float(candidate_kl) if np.isfinite(candidate_kl) else None,
                    gradient_norm=gradient_norm, accepted=passed,
                    actor_loss=float(actor_loss.detach()) if finite_loss else None))
                if passed:
                    accepted += 1
                    actor_loss_value = float(actor_loss.detach())
                    step_accepted = True
                    break
                rejected += 1
                self.macro_actor.load_state_dict(actor_snapshot)
                self.ppo_actor_optimizer.load_state_dict(copy.deepcopy(optimizer_snapshot))
                candidate_lr *= .5
            if not step_accepted:
                break
        # Learning-rate reduction is local to a behavior batch, not an implicit new config.
        for group in self.ppo_actor_optimizer.param_groups:
            group["lr"] = base_lr
        critic_losses = []
        for _ in range(self.cfg["ppo_epochs"]):
            critic_loss = F.mse_loss(self.macro_critic(s).ravel(), returns)
            if not bool(torch.isfinite(critic_loss)):
                raise FloatingPointError("Non-finite PPO value loss")
            self.ppo_critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            critic_gradient = nn.utils.clip_grad_norm_(self.macro_critic.parameters(), self.cfg["grad_clip"])
            if not bool(torch.isfinite(critic_gradient)):
                raise FloatingPointError("Non-finite PPO value gradient")
            self.ppo_critic_optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in self.macro_critic.parameters()):
                raise FloatingPointError("Non-finite PPO value parameters")
            self.ppo_critic_gradient_steps += 1
            critic_losses.append(float(critic_loss.detach()))
        std_checks = []
        std_target = float(std_before.exp().mean())
        if accepted:
            self.ppo_accepted_update_events += 1
            self.ppo_behavior_version += 1
            fraction = min(1., self.ppo_accepted_update_events / self.cfg["ppo_std_decay_updates"])
            log_target = ((1-fraction) * np.log(self.cfg["ppo_initial_std"])
                          + fraction * np.log(self.cfg["ppo_final_std"]))
            std_target = float(np.exp(log_target))
            for index in range(self.cfg["ppo_std_backtracks"]):
                candidate = std_before + (float(log_target) - std_before) / (2 ** index)
                with torch.no_grad():
                    self.macro_actor.log_std.copy_(candidate)
                    mean, std = self.macro_actor(s)
                    kl = float(batch_normal_kl(old_mean, old_std, mean, std))
                passed = bool(np.isfinite(kl) and kl <= limit + 1e-12)
                std_checks.append(dict(candidate=index, std=float(candidate.exp().mean()),
                    kl=kl if np.isfinite(kl) else None, accepted=passed))
                if passed:
                    break
                with torch.no_grad():
                    self.macro_actor.log_std.copy_(std_before)
        with torch.no_grad():
            mean, std = self.macro_actor(s)
            final_kl = float(batch_normal_kl(old_mean, old_std, mean, std))
            fitted = self.macro_critic(s).ravel().cpu().numpy()
        if not np.isfinite(final_kl) or final_kl > limit + 1e-12:
            raise AssertionError("Final deployed PPO policy exceeded its batch KL limit")
        self.ppo_updates += 1
        self.ppo_gradient_steps += accepted
        self.ppo_guard_attempted_steps += attempted
        self.ppo_guard_rejected_steps += rejected
        self.last_ppo_stats = dict(pre_logprob_error=pre_error, kl=final_kl, epochs=accepted,
            actor_loss=actor_loss_value, critic_loss=critic_losses[-1],
            critic_loss_first_fit=critic_losses[0], critic_gradient_steps=len(critic_losses),
            accepted_actor_steps=accepted, rejected_actor_steps=rejected, attempted_actor_steps=attempted,
            batch_size=len(states), behavior_version=self.ppo_behavior_version,
            accepted_update_events=self.ppo_accepted_update_events,
            std_before=float(std_before.exp().mean()), std_target=std_target,
            std_actual=float(self.macro_actor.log_std.detach().exp().mean()),
            advantage_mean=float(np.mean(advantages)), advantage_std=float(np.std(advantages)),
            return_mean=float(np.mean(returns_np)), return_std=float(np.std(returns_np)),
            value_mean_before=float(np.mean(values)), value_mean_after=float(np.mean(fitted)),
            explained_variance_before=explained_variance(returns_np, values),
            explained_variance_after=explained_variance(returns_np, fitted),
            attempts=attempts, std_checks=std_checks)
        self.macro_rollout.clear()


def build_deployment_agent(mdp, training_config):
    """Build fresh 85/86 networks with the exact physical discount and route contract."""
    training = copy.deepcopy(training_config)
    supplied = training.pop("macro_gamma", mdp.gamma_macro)
    if not np.isclose(supplied, mdp.gamma_macro):
        raise ValueError("Macro discount disagrees with physical duration")
    if "macro_gamma_frames" in training and training.pop("macro_gamma_frames") != mdp.frames_per_macro:
        raise ValueError("Legacy macro frame override is incompatible")
    if not np.isclose(training.get("gamma_per_resource_frame", mdp.gamma_frame), mdp.gamma_frame):
        raise ValueError("Routing discount disagrees with physical objective")
    if training.get("critic_action_representation", "executed_route") != "executed_route":
        raise ValueError("Deployment SAC uses executed-route evaluation")
    training["macro_gamma"] = mdp.gamma_macro
    training["gamma_per_resource_frame"] = mdp.gamma_frame
    training["critic_action_representation"] = "executed_route"
    spec = mdp.observation_spec()
    if (spec["macro"]["dim"], spec["micro"]["dim"], mdp.macro_action_dim, mdp.micro_action_dim) != (85, 86, 6, 18):
        raise ValueError("Deployment V3 requires observations 85/86 and actions 6/18")
    config = dict(schema="industrial_deployment_v3", environment=dict(num_uavs=3),
                  observation=dict(layout="split_v1"), training=training)
    agent = DeploymentHRL.from_mdp(mdp, config)
    agent.full_config = copy.deepcopy(config)
    return agent
