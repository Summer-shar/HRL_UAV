"""PPO and SAC updates for a single pure-RL, two-timescale training path."""

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .agents.ppo_agent import ActorNet as MacroActorNet, CriticNet as MacroCriticNet
from .agents.sac_agent import Actor, Critic, Entroy, Memory, device
from .agents.tqc import QuantileCritic, actor_value, quantile_loss, truncated_target


def tensor(value):
    return torch.as_tensor(np.asarray(value), dtype=torch.float32, device=device)


def gae(rewards, values, next_values, trace_continue, discount, trace_lambda):
    """Generalized Advantage Estimation with configurable lambda and discount."""
    output = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        delta = rewards[i] + discount * next_values[i] - values[i]
        running = delta + discount * trace_lambda * trace_continue[i] * running
        output[i] = running
    return output, output + values


# Default hyperparameters for PPO macro-learner
PPO_DEFAULTS = {
    "gae_lambda": 0.97,           # Phase2 fix: was missing (defaulted to 0)
    "macro_gamma_frames": 10,      # Phase2 fix: 0.99^10=0.904 effective horizon
    "macro_gamma": 0.95,           # Alternative: explicit discount factor (10-frame horizon)
}


def potential_distance_shaping(positions, centroids, scale=1.0):
    """
    Reward shaping based on centroid-relative distance (potential-based function).

    Args:
        positions: Array of UAV positions (n_uav, 2) or flattened (n_uav*2,)
        centroids: Array of centroid positions (n_uav, 2) or flattened (n_uav*2,)
        scale: Scaling factor for shaping reward

    Returns:
        Potential-based shaping reward (float)
        Closer to centroid = positive shaping, farther = negative shaping
    """
    pos = np.asarray(positions).reshape(-1, 2)
    cent = np.asarray(centroids).reshape(-1, 2)

    # Euclidean distance from each UAV to its centroid
    distances = np.linalg.norm(pos - cent, axis=1)
    mean_distance = distances.mean()

    # Potential: negative distance (closer = higher potential)
    potential = -mean_distance / scale

    return potential


def distance_shaping_reward(state, next_state, scale=50.0):
    """
    Compute shaping reward as difference in potentials.

    Args:
        state: Current state dict with 'positions' and 'centroids'
        next_state: Next state dict with 'positions' and 'centroids'
        scale: Distance scale for normalization

    Returns:
        Shaping reward: phi(s') - phi(s) (should be added to intrinsic reward)
    """
    if not all(k in state for k in ("positions", "centroids")):
        return 0.0

    phi_curr = potential_distance_shaping(state["positions"], state["centroids"], scale)
    phi_next = potential_distance_shaping(next_state["positions"], next_state["centroids"], scale)

    # Potential-based shaping: reward = phi(s') - phi(s)
    # If we get closer (phi_next > phi_curr), reward is positive
    return phi_next - phi_curr


class DirectHRL:
    def __init__(self, state_dim, micro_action_dim, macro_action_dim, config, *, macro_state_dim=None):
        self.cfg = config["training"]
        self.micro_algorithm = self.cfg.get("micro_algorithm", "sac")
        if self.micro_algorithm not in ("sac", "tqc"):
            raise ValueError(f"Unknown micro algorithm: {self.micro_algorithm}")
        self.critic_steps_per_update = self.cfg.get("sac_critic_steps_per_update", 1)
        if (type(self.critic_steps_per_update) is not int
                or self.critic_steps_per_update < 1):
            raise ValueError("sac_critic_steps_per_update must be a positive integer")
        # Missing policy retains historical checkpoint/config semantics.
        self.sac_stage_policy = self.cfg.get("sac_stage_policy", "legacy_stage_rates")
        if self.sac_stage_policy not in ("preserve", "legacy_stage_rates"):
            raise ValueError(f"Unknown SAC stage policy: {self.sac_stage_policy}")
        if self.sac_stage_policy == "preserve" and any(
                key in self.cfg for key in ("joint_lr", "joint_sac_actor_warmup_updates")):
            raise ValueError("Preserved SAC uses only pretrain_lr and one lifetime actor warmup; remove joint SAC overrides")
        self.observation_layout = config.get("observation", {}).get("layout", "shared_v1")
        if self.observation_layout == "split_v1" and macro_state_dim is None:
            raise ValueError("Split observations require an explicit macro_state_dim; use from_mdp()")
        self.micro_state_dim = state_dim
        self.macro_state_dim = state_dim if macro_state_dim is None else macro_state_dim
        self.state_dim, self.action_dim = state_dim, micro_action_dim
        self.sac_actor = Actor(state_dim, micro_action_dim)
        representation = self.cfg.get("critic_action_representation", "raw_score")
        if representation not in ("raw_score", "executed_route"):
            raise ValueError(f"Unknown critic action representation: {representation}")
        route_width = config["environment"]["num_uavs"] if representation == "executed_route" else 0
        self.sac_critic = Critic(state_dim, micro_action_dim, route_width)
        if self.micro_algorithm == "tqc":
            # Preserve the SAC initialization RNG budget so both arms start
            # with identical actors/PPO and identical subsequent random draws.
            with torch.random.fork_rng():
                self.sac_critic = QuantileCritic(
                    state_dim, micro_action_dim, route_width,
                    self.cfg.get("tqc_quantiles", 25), self.cfg.get("tqc_drop_per_critic", 2),
                )
        self.entropy = Entroy(micro_action_dim, self.cfg["initial_alpha"])
        nn.init.zeros_(self.sac_actor.action_net.out.weight)
        nn.init.zeros_(self.sac_actor.action_net.out.bias)
        nn.init.zeros_(self.sac_actor.action_net.std_out.weight)
        nn.init.constant_(self.sac_actor.action_net.std_out.bias, self.cfg["initial_sac_log_std"])
        # Replay includes both score and executed routing matrix; Q(s, score)
        # is trained on the score that actually generated this transition.
        self.replay = Memory(
            self.cfg["replay_capacity"], 2 * state_dim + 2 * micro_action_dim + 2,
            recent_ratio=0.0, rng=np.random.default_rng(self.cfg["seed"] + 731),
        )
        self.macro_actor = MacroActorNet(self.macro_state_dim, macro_action_dim).to(device)
        self.macro_critic = MacroCriticNet(self.macro_state_dim).to(device)
        nn.init.zeros_(self.macro_actor.out.weight)
        nn.init.zeros_(self.macro_actor.out.bias)
        self.macro_actor.log_std.requires_grad_(False)
        self.macro_actor.log_std.data.fill_(np.log(self.cfg["ppo_initial_std"]))
        self.ppo_actor_optimizer = torch.optim.Adam(self.macro_actor.parameters(), lr=self.cfg["ppo_actor_lr"], eps=1e-5)
        self.ppo_critic_optimizer = torch.optim.Adam(self.macro_critic.parameters(), lr=self.cfg["ppo_critic_lr"], eps=1e-5)
        # Ensure PPO macro-learner defaults are set
        self.cfg.setdefault("gae_lambda", PPO_DEFAULTS["gae_lambda"])
        # macro_gamma: either explicit value or computed from frames
        if "macro_gamma" in self.cfg:
            macro_gamma = self.cfg["macro_gamma"]
        elif "macro_gamma_frames" in self.cfg:
            macro_gamma = self.cfg["gamma_per_resource_frame"] ** self.cfg["macro_gamma_frames"]
        else:
            macro_gamma = PPO_DEFAULTS["macro_gamma"]  # 0.95 (10-frame equivalent)
        self._macro_gamma = macro_gamma
        self.macro_rollout = []
        self.sac_updates = self.sac_actor_updates = self.ppo_updates = self.ppo_gradient_steps = 0
        self.sac_critic_gradient_steps = self.sac_target_updates = 0
        self.stage = "pretrain"
        self.stage_initial_updates = 0
        self._set_sac_rates(self.cfg["pretrain_lr"])
        self.last_sac_stats = {}
        self.last_ppo_stats = {}

    @classmethod
    def from_mdp(cls, mdp, config):
        spec = mdp.observation_spec()
        agent = cls(spec["micro"]["dim"], mdp.micro_action_dim, mdp.macro_action_dim,
                    config, macro_state_dim=spec["macro"]["dim"])
        agent.observation_spec = spec
        return agent

    def _set_sac_rates(self, rates):
        for optimizer, rate in zip(
            [self.sac_actor.optimizer, self.sac_critic.optimizer, self.entropy.optimizer], rates,
        ):
            for group in optimizer.param_groups:
                group["lr"] = rate

    def configured_sac_rates(self):
        key = "pretrain_lr" if self.sac_stage_policy == "preserve" else f"{self.stage}_lr"
        return self.cfg[key]

    def sac_actor_warmup(self):
        if self.sac_stage_policy == "legacy_stage_rates" and self.stage == "joint":
            return self.cfg.get("joint_sac_actor_warmup_updates", self.cfg["sac_actor_warmup_updates"])
        return self.cfg["sac_actor_warmup_updates"]

    def set_stage(self, stage):
        if stage not in ("pretrain", "joint"):
            raise ValueError(f"Unknown training stage: {stage}")
        self.stage = stage
        if self.sac_stage_policy == "legacy_stage_rates":
            self.stage_initial_updates = self.sac_updates
            self._set_sac_rates(self.configured_sac_rates())
        # In preserve mode the stage only enables macro learning in rollout.

    def macro_action(self, state, enabled=True, deterministic=False):
        if np.asarray(state).shape != (self.macro_state_dim,):
            raise ValueError(f"PPO requires a {self.macro_state_dim}D macro observation")
        if not enabled:
            return np.zeros(self.macro_actor.out.out_features, dtype=np.float32), 0.0
        with torch.no_grad():
            mean, std = self.macro_actor(tensor(state).unsqueeze(0))
            dist = torch.distributions.Normal(mean, std)
            raw = mean if deterministic else dist.sample()
            logprob = dist.log_prob(raw).sum().item()
        # Store the un-clipped Gaussian score and its correct log density.
        # Physical clipping/projection takes place only in QueueMDP.start_macro.
        return raw.cpu().numpy().ravel(), logprob

    def store(self, state, action, route, reward, next_state, discount):
        if any(np.asarray(s).shape != (self.micro_state_dim,) for s in (state, next_state)):
            raise ValueError(f"SAC replay requires {self.micro_state_dim}D micro observations")
        self.replay.store_transition(
            state, action, reward, next_state,
            extras=np.r_[discount, np.asarray(route).ravel()],
        )

    def update_sac(self):
        if self.replay.memory_counter < max(self.cfg["batch_size"], self.cfg["replay_warmup"]):
            return
        batch = self.replay.sample(self.cfg["batch_size"])
        n, a = self.state_dim, self.action_dim
        state, action = tensor(batch[:, :n]), tensor(batch[:, n:n+a])
        reward = tensor(batch[:, n+a:n+a+1])
        discount = tensor(batch[:, n+a+1:n+a+2])
        next_state = tensor(batch[:, -n:])
        with torch.no_grad():
            next_action, logprob = self.sac_actor.evaluate(next_state)
            q1, q2 = self.sac_critic.target_critic_v(next_state, next_action)
            next_value = (truncated_target(q1, q2, self.sac_critic.drop_per_critic)
                          if self.micro_algorithm == "tqc" else torch.minimum(q1, q2))
            target = reward + discount * (next_value - self.entropy.log_alpha.exp() * logprob)
        # Repeat the fit, not sampling or target/actor clocks. No extra RNG draws.
        for fit in range(self.critic_steps_per_update):
            q1, q2 = self.sac_critic.get_v(state, action)
            loss = (quantile_loss(q1, q2, target) if self.micro_algorithm == "tqc"
                    else F.mse_loss(q1, target) + F.mse_loss(q2, target))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite SAC loss; stopping without altering reward")
            if fit == 0:
                first_loss = loss.item()
            critic_grad = self.sac_critic.learn(q1, q2, target, max_grad_norm=self.cfg["grad_clip"])
            self.sac_critic_gradient_steps += 1
        self.sac_updates += 1
        stats = {"critic_loss": first_loss, "critic_loss_last_fit": loss.item(),
                 "critic_grad": critic_grad, "target_std": target.std(unbiased=False).item()}
        actor_warmup = self.sac_actor_warmup()
        if (self.sac_updates - self.stage_initial_updates > actor_warmup
                and self.sac_updates % self.cfg["sac_policy_delay"] == 0):
            for parameter in self.sac_critic.critic_v.parameters():
                parameter.requires_grad_(False)
            sampled, logprob = self.sac_actor.evaluate(state)
            qa, qb = self.sac_critic.get_v(state, sampled)
            value = actor_value(qa, qb) if self.micro_algorithm == "tqc" else torch.minimum(qa, qb)
            actor_loss = (self.entropy.log_alpha.exp().detach() * logprob - value).mean()
            actor_grad = self.sac_actor.learn(actor_loss, max_grad_norm=self.cfg["grad_clip"])
            for parameter in self.sac_critic.critic_v.parameters():
                parameter.requires_grad_(True)
            self.entropy.learn(self.entropy.temperature_loss(logprob))
            self.entropy.alpha = self.entropy.log_alpha.exp()
            self.sac_actor_updates += 1
            stats.update(actor_loss=actor_loss.item(), actor_grad=actor_grad)
        with torch.no_grad():
            for target_parameter, parameter in zip(self.sac_critic.target_critic_v.parameters(), self.sac_critic.critic_v.parameters()):
                target_parameter.lerp_(parameter, self.cfg["tau"])
        self.sac_target_updates += 1
        stats["alpha"] = self.entropy.log_alpha.exp().item()
        self.last_sac_stats = stats

    def update_ppo(self, progress=0.0, force=False):
        if not self.macro_rollout or (len(self.macro_rollout) < self.cfg["ppo_update_macros"] and not force):
            return
        states, actions, logprobs, rewards, next_states, continuations = map(np.asarray, zip(*self.macro_rollout))
        s, a, old_log = tensor(states), tensor(actions), tensor(logprobs)
        with torch.no_grad():
            values = self.macro_critic(s).ravel().cpu().numpy()
            next_values = self.macro_critic(tensor(next_states)).ravel().cpu().numpy()
        advantages, returns = gae(rewards, values, next_values, continuations, self._macro_gamma, self.cfg["gae_lambda"])
        adv = tensor((advantages - advantages.mean()) / (advantages.std() + 1e-8))
        returns = tensor(returns)
        with torch.no_grad():
            m, sd = self.macro_actor(s)
            before = torch.distributions.Normal(m, sd).log_prob(a).sum(-1)
            pre_error = (before - old_log).abs().max().item()
        if pre_error > 1e-4:
            raise RuntimeError(f"PPO behavior log-prob mismatch: {pre_error}")
        epochs = 0
        kl = 0.0
        for epoch in range(self.cfg["ppo_epochs"]):
            m, sd = self.macro_actor(s)
            logprob = torch.distributions.Normal(m, sd).log_prob(a).sum(-1)
            logratio = logprob - old_log
            ratio = logratio.exp()
            kl = ((ratio - 1) - logratio).mean().item()
            if epoch > 0 and kl > self.cfg["ppo_target_kl"]:
                break
            objective = torch.minimum(ratio * adv, ratio.clamp(1-self.cfg["ppo_clip"], 1+self.cfg["ppo_clip"]) * adv)
            actor_loss = -objective.mean()
            self.ppo_actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.macro_actor.parameters(), self.cfg["grad_clip"])
            self.ppo_actor_optimizer.step()
            epochs += 1
        for _ in range(self.cfg["ppo_epochs"]):
            critic_loss = F.mse_loss(self.macro_critic(s).ravel(), returns)
            self.ppo_critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.macro_critic.parameters(), self.cfg["grad_clip"])
            self.ppo_critic_optimizer.step()
        self.ppo_updates += 1
        self.ppo_gradient_steps += epochs
        self.last_ppo_stats = {"pre_logprob_error": pre_error, "kl": kl, "epochs": epochs, "actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}
        self.macro_rollout.clear()
        # Change exploration only after all samples from the old std are used.
        if "ppo_std_decay_windows" in self.cfg:
            progress = min(1.0, max(0.0, progress * self.cfg["joint_windows"] / self.cfg["ppo_std_decay_windows"]))
        std = np.exp((1-progress) * np.log(self.cfg["ppo_initial_std"]) + progress * np.log(self.cfg["ppo_final_std"]))
        self.macro_actor.log_std.data.fill_(np.log(std))

    def checkpoint(self, config, include_replay=True):
        return {
            "schema": config["schema"], "config": copy.deepcopy(config), "stage": self.stage,
            "micro_state_dim": self.micro_state_dim, "macro_state_dim": self.macro_state_dim,
            "observations": getattr(self, "observation_spec", {"layout": self.observation_layout}),
            "sac_actor": self.sac_actor.action_net.state_dict(),
            "sac_critic": self.sac_critic.critic_v.state_dict(),
            "sac_target": self.sac_critic.target_critic_v.state_dict(),
            "log_alpha": self.entropy.log_alpha.detach(),
            "macro_actor": self.macro_actor.state_dict(), "macro_critic": self.macro_critic.state_dict(),
            "optimizers": [obj.state_dict() for obj in (self.sac_actor.optimizer, self.sac_critic.optimizer, self.entropy.optimizer, self.ppo_actor_optimizer, self.ppo_critic_optimizer)],
            # sac_critic retains its historical update-event clock for compatibility.
            "updates": {"sac_critic": self.sac_updates,
                        "sac_critic_gradients": self.sac_critic_gradient_steps,
                        "sac_target": self.sac_target_updates,
                        "sac_actor": self.sac_actor_updates, "ppo": self.ppo_updates,
                        "ppo_gradients": self.ppo_gradient_steps},
            "training_state": {"stage_initial_updates": self.stage_initial_updates,
                               "last_sac_stats": copy.deepcopy(self.last_sac_stats),
                               "last_ppo_stats": copy.deepcopy(self.last_ppo_stats),
                               "macro_rollout": copy.deepcopy(self.macro_rollout)},
            "replay": self.replay.state_dict() if include_replay else None,
        }
