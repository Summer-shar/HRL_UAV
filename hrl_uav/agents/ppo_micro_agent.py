import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_STD_MIN = -5.0
LOG_STD_MAX = 0.5
SQUASH_EPS = 1e-6


class ActorNet(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(
            torch.full((action_dim,), -1.0, dtype=torch.float32)
        )

        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.mean.weight.data.normal_(0, 0.1)

    def forward(self, inputs):
        hidden = F.relu(self.fc1(inputs))
        hidden = F.relu(self.fc2(hidden))
        mean = self.mean(hidden)
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp().expand_as(mean)
        return mean, std


class CriticNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.value = nn.Linear(256, 1)

        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.value.weight.data.normal_(0, 0.1)

    def forward(self, inputs):
        hidden = F.relu(self.fc1(inputs))
        hidden = F.relu(self.fc2(hidden))
        return self.value(hidden)


class MicroPPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        actor_lr=1e-4,
        critic_lr=5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.1,
        update_epochs=4,
        mini_batch_size=1024,
        entropy_coef=0.003,
        value_coef=0.5,
        max_grad_norm=0.5,
    ):
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_epsilon = float(clip_epsilon)
        self.update_epochs = int(update_epochs)
        self.mini_batch_size = int(mini_batch_size)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)

        self.actor = ActorNet(state_dim, action_dim).to(device)
        self.critic = CriticNet(state_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_lr,
            eps=1e-5,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_lr,
            eps=1e-5,
        )

    def _sample_action_and_logprob(self, mean, std, deterministic=False):
        dist = torch.distributions.Normal(mean, std)
        if deterministic:
            pre_tanh_action = mean
        else:
            pre_tanh_action = dist.rsample()

        action = torch.tanh(pre_tanh_action)
        log_prob = dist.log_prob(pre_tanh_action) - torch.log(
            torch.clamp(1.0 - action.pow(2), min=SQUASH_EPS)
        )
        return action, log_prob.sum(dim=-1), dist

    def _logprob_from_action(self, mean, std, action):
        action = torch.clamp(action, -1.0 + SQUASH_EPS, 1.0 - SQUASH_EPS)
        pre_tanh_action = torch.atanh(action)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(pre_tanh_action) - torch.log(
            torch.clamp(1.0 - action.pow(2), min=SQUASH_EPS)
        )
        return log_prob.sum(dim=-1), dist.entropy().sum(dim=-1)

    def choose_action(self, state, deterministic=False):
        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        with torch.no_grad():
            mean, std = self.actor(state_tensor)
            action, log_prob, _ = self._sample_action_and_logprob(
                mean,
                std,
                deterministic=deterministic,
            )
            value = self.critic(state_tensor).squeeze(-1)

        return (
            action.squeeze(0).cpu().numpy().astype(np.float32),
            float(log_prob.item()),
            float(value.item()),
        )

    def choose_action_deterministic(self, state):
        action, _, _ = self.choose_action(state, deterministic=True)
        return action

    def get_value(self, state):
        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        with torch.no_grad():
            value = self.critic(state_tensor).squeeze(-1)
        return float(value.item())

    def compute_gae(self, rewards, values, dones, next_value):
        rewards = np.asarray(rewards, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0

        values_ext = np.append(values, np.float32(next_value))
        for t in range(len(rewards) - 1, -1, -1):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values_ext[t + 1] * mask - values_ext[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns.astype(np.float32)

    def update(self, states, actions, old_log_probs, returns, advantages):
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        old_log_probs = np.asarray(old_log_probs, dtype=np.float32)
        returns = np.asarray(returns, dtype=np.float32)
        advantages = np.asarray(advantages, dtype=np.float32)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        num_updates = 0

        batch_size = states.shape[0]
        mini_batch_size = min(self.mini_batch_size, batch_size)

        for _ in range(self.update_epochs):
            indices = np.random.permutation(batch_size)
            for start in range(0, batch_size, mini_batch_size):
                batch_indices = indices[start:start + mini_batch_size]

                state_batch = torch.as_tensor(
                    states[batch_indices],
                    dtype=torch.float32,
                    device=device,
                )
                action_batch = torch.as_tensor(
                    actions[batch_indices],
                    dtype=torch.float32,
                    device=device,
                )
                old_log_prob_batch = torch.as_tensor(
                    old_log_probs[batch_indices],
                    dtype=torch.float32,
                    device=device,
                )
                return_batch = torch.as_tensor(
                    returns[batch_indices],
                    dtype=torch.float32,
                    device=device,
                )
                advantage_batch = torch.as_tensor(
                    advantages[batch_indices],
                    dtype=torch.float32,
                    device=device,
                )

                mean, std = self.actor(state_batch)
                new_log_probs, entropy = self._logprob_from_action(
                    mean,
                    std,
                    action_batch,
                )
                entropy = entropy.mean()

                ratio = torch.exp(new_log_probs - old_log_prob_batch)
                surrogate_1 = ratio * advantage_batch
                surrogate_2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon,
                ) * advantage_batch
                actor_loss = -torch.min(surrogate_1, surrogate_2).mean()
                actor_loss -= self.entropy_coef * entropy

                value_pred = self.critic(state_batch).squeeze(-1)
                critic_loss = F.mse_loss(value_pred, return_batch)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                (self.value_coef * critic_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                total_actor_loss += float(actor_loss.item())
                total_critic_loss += float(critic_loss.item())
                num_updates += 1

        if num_updates == 0:
            return 0.0, 0.0

        return total_actor_loss / num_updates, total_critic_loss / num_updates
