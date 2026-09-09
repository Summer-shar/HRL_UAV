import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..routing import project_rows_to_simplex_tensor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [HRL Abstracted SAC Agent] - Extracted from path planning

tau = 1e-2
q_lr = 3e-4
value_lr = 3e-3
policy_lr = 1e-3
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
SQUASH_EPS = 1e-6


class ActorNet(nn.Module):
    def __init__(self, inp, outp):
        super(ActorNet, self).__init__()
        self.in_to_y1 = nn.Linear(inp, 256)
        self.in_to_y1.weight.data.normal_(0, 0.1)
        self.y1_to_y2 = nn.Linear(256, 256)
        self.y1_to_y2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(256, outp)
        self.out.weight.data.normal_(0, 0.1)
        self.std_out = nn.Linear(256, outp)
        self.std_out.weight.data.normal_(0, 0.1)

    def forward(self, inputstate):
        inputstate = self.in_to_y1(inputstate)
        inputstate = F.relu(inputstate)
        inputstate = self.y1_to_y2(inputstate)
        inputstate = F.relu(inputstate)

        # Output the pre-squash mean; tanh is applied during sampling/evaluation.
        mean = self.out(inputstate)
        log_std = self.std_out(inputstate)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        return mean, std


class CriticNet(nn.Module):
    def __init__(self, input_dim, output_dim, route_width=0, quantiles=1):
        super(CriticNet, self).__init__()
        if type(quantiles) is not int or quantiles < 1:
            raise ValueError("quantiles must be a positive integer")
        self.route_width = int(route_width)
        if self.route_width and output_dim % self.route_width:
            raise ValueError("Routing critic input must contain complete simplex rows")
        self.in_to_y1 = nn.Linear(input_dim + output_dim, 256)
        self.in_to_y1.weight.data.normal_(0, 0.1)
        self.y1_to_y2 = nn.Linear(256, 256)
        self.y1_to_y2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(256, quantiles)
        self.out.weight.data.normal_(0, 0.1)

        self.q2_in_to_y1 = nn.Linear(input_dim + output_dim, 256)
        self.q2_in_to_y1.weight.data.normal_(0, 0.1)
        self.q2_y1_to_y2 = nn.Linear(256, 256)
        self.q2_y1_to_y2.weight.data.normal_(0, 0.1)
        self.q2_out = nn.Linear(256, quantiles)
        self.q2_out.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        if self.route_width:
            # All raw scores with the same executed routing share a Q value.
            # Autograd propagates through the projection to the original actor.
            a = project_rows_to_simplex_tensor(a.reshape(-1, self.route_width)).reshape_as(a)
        inputstate = torch.cat((s, a), dim=1)

        q1 = self.in_to_y1(inputstate)
        q1 = F.relu(q1)
        q1 = self.y1_to_y2(q1)
        q1 = F.relu(q1)
        q1 = self.out(q1)

        q2 = self.q2_in_to_y1(inputstate)
        q2 = F.relu(q2)
        q2 = self.q2_y1_to_y2(q2)
        q2 = F.relu(q2)
        q2 = self.q2_out(q2)
        return q1, q2


class Memory():
    def __init__(
        self,
        capacity,
        dims,
        recent_window=50000,
        recent_ratio=0.75,
        rng=None,
    ):
        self.capacity = int(capacity)
        self.mem = np.zeros((self.capacity, dims), dtype=np.float32)
        self.memory_counter = 0
        self.recent_window = max(1, min(int(recent_window), self.capacity))
        self.recent_ratio = float(np.clip(recent_ratio, 0.0, 1.0))
        self.rng = np.random if rng is None else rng

    def _integers(self, low, high, size):
        if isinstance(self.rng, np.random.Generator):
            return self.rng.integers(low, high, size=size)
        return self.rng.randint(low, high, size=size)

    def store_transition(self, s, a, r, s_, extras=None):
        extra_values = (
            np.asarray(extras, dtype=np.float32).reshape(-1)
            if extras is not None
            else np.empty(0, dtype=np.float32)
        )
        tran = np.hstack((s, a, [r], extra_values, s_))
        if tran.size != self.mem.shape[1]:
            raise ValueError(
                f"Transition width {tran.size} does not match replay width "
                f"{self.mem.shape[1]}."
            )
        index = self.memory_counter % self.capacity
        self.mem[index, :] = tran
        self.memory_counter += 1

    def sample(self, n):
        pool_size = min(self.memory_counter, self.capacity)
        if pool_size <= 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        if self.recent_ratio <= 0.0:
            sample_index = self._integers(0, pool_size, size=n)
            return self.mem[sample_index, :]

        start_global = max(0, self.memory_counter - pool_size)
        recent_available = min(pool_size, self.recent_window)
        recent_start_global = self.memory_counter - recent_available

        recent_count = min(int(round(n * self.recent_ratio)), n)
        historical_count = n - recent_count
        sample_parts = []

        if recent_count > 0:
            recent_global = self._integers(
                recent_start_global,
                self.memory_counter,
                size=recent_count,
            )
            sample_parts.append(recent_global % self.capacity)

        if historical_count > 0:
            if recent_start_global > start_global:
                historical_global = self._integers(
                    start_global,
                    recent_start_global,
                    size=historical_count,
                )
                sample_parts.append(historical_global % self.capacity)
            else:
                recent_global = self._integers(
                    recent_start_global,
                    self.memory_counter,
                    size=historical_count,
                )
                sample_parts.append(recent_global % self.capacity)

        sample_index = np.concatenate(sample_parts, axis=0)
        self.rng.shuffle(sample_index)
        new_mem = self.mem[sample_index, :]
        return new_mem

    def state_dict(self):
        pool_size = min(self.memory_counter, self.capacity)
        start_global = max(0, self.memory_counter - pool_size)
        indices = np.arange(
            start_global,
            self.memory_counter,
            dtype=np.int64,
        ) % self.capacity
        state = {
            "transitions": self.mem[indices, :].copy(),
            "memory_counter": int(self.memory_counter),
            "capacity": int(self.capacity),
            "transition_width": int(self.mem.shape[1]),
            "recent_window": int(self.recent_window),
            "recent_ratio": float(self.recent_ratio),
        }
        if isinstance(self.rng, np.random.Generator):
            state["rng_state"] = self.rng.bit_generator.state
        return state

    def load_state_dict(self, state, *, memory_counter=None):
        if not isinstance(state, dict) or "transitions" not in state:
            raise ValueError("Replay state must contain transitions.")
        transitions = np.asarray(state["transitions"], dtype=np.float32)
        if transitions.ndim != 2:
            raise ValueError("Replay transitions must be a 2-D array.")
        if transitions.shape[1] != self.mem.shape[1]:
            raise ValueError(
                "Replay transition width "
                f"{transitions.shape[1]} does not match {self.mem.shape[1]}."
            )
        retained = transitions[-self.capacity :]
        self.mem.fill(0.0)
        if memory_counter is None:
            self.mem[: retained.shape[0], :] = retained
            self.memory_counter = int(retained.shape[0])
        else:
            # Exact same-capacity continuation needs the original ring layout,
            # not just the retained transitions in chronological order.
            counter = int(memory_counter)
            if (counter < 0 or len(transitions) != min(counter, self.capacity)
                    or state.get("capacity", self.capacity) != self.capacity):
                raise ValueError("Replay counter/capacity does not match retained transitions")
            if (state["recent_window"] != self.recent_window
                    or state["recent_ratio"] != self.recent_ratio):
                raise ValueError("Replay sampling settings changed during continuation")
            indices = np.arange(counter - len(retained), counter, dtype=np.int64) % self.capacity
            self.mem[indices, :] = retained
            self.memory_counter = counter
        if (
            isinstance(self.rng, np.random.Generator)
            and "rng_state" in state
        ):
            self.rng.bit_generator.state = state["rng_state"]
        return self.memory_counter


class Actor():
    def __init__(self, state_dim, action_dim):
        self.action_net = ActorNet(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(
            self.action_net.parameters(),
            lr=policy_lr,
        )
        self.min_action = -1.0
        self.max_action = 1.0

    def _sample_action_and_logprob(self, mean, std, deterministic=False):
        dist = torch.distributions.Normal(mean, std)
        if deterministic:
            pre_tanh_action = mean
        else:
            pre_tanh_action = dist.rsample()

        action = torch.tanh(pre_tanh_action)
        squash_correction = torch.log(
            torch.clamp(1.0 - action.pow(2), min=SQUASH_EPS)
        )
        action_logprob = dist.log_prob(pre_tanh_action) - squash_correction
        return action, action_logprob.sum(dim=-1, keepdim=True)

    def choose_action(self, s, deterministic=False):
        inputstate = torch.as_tensor(
            s,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        with torch.no_grad():
            mean, std = self.action_net(inputstate)
            action, _ = self._sample_action_and_logprob(
                mean,
                std,
                deterministic=deterministic,
            )

        action = torch.clamp(action, self.min_action, self.max_action)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def evaluate(self, s):
        mean, std = self.action_net(s)
        action, action_logprob = self._sample_action_and_logprob(
            mean,
            std,
            deterministic=False,
        )
        action = torch.clamp(action, self.min_action, self.max_action)
        return action, action_logprob

    def learn(self, actor_loss, max_grad_norm=0.0):
        loss = actor_loss
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.action_net.parameters(),
            float(max_grad_norm) if max_grad_norm > 0.0 else float("inf"),
        )
        self.optimizer.step()
        return float(grad_norm)


class Entroy():
    def __init__(self, action_dim=32, initial_alpha=1.0):
        initial_alpha = float(initial_alpha)
        if not math.isfinite(initial_alpha) or initial_alpha <= 0.0:
            raise ValueError("initial_alpha must be finite and positive.")
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.full(
            (1,),
            math.log(initial_alpha),
            requires_grad=True,
            device=device,
        )
        self.alpha = self.log_alpha.exp()
        self.optimizer = torch.optim.Adam([self.log_alpha], lr=q_lr)

    def temperature_loss(self, log_prob):
        return -(
            self.log_alpha
            * (log_prob + self.target_entropy).detach()
        ).mean()

    def learn(self, entroy_loss):
        loss = entroy_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Critic():
    def __init__(self, state_dim, action_dim, route_width=0):
        self.critic_v = CriticNet(state_dim, action_dim, route_width).to(device)
        self.target_critic_v = CriticNet(state_dim, action_dim, route_width).to(device)
        self.target_critic_v.load_state_dict(self.critic_v.state_dict())
        self.optimizer = torch.optim.Adam(
            self.critic_v.parameters(),
            lr=value_lr,
            eps=1e-5,
        )
        self.lossfunc = nn.MSELoss()

    def soft_update(self):
        for target_param, param in zip(
            self.target_critic_v.parameters(),
            self.critic_v.parameters(),
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def get_v(self, s, a):
        return self.critic_v(s, a)

    def learn(self, current_q1, current_q2, target_q, max_grad_norm=0.0):
        loss = self.lossfunc(current_q1, target_q) + self.lossfunc(
            current_q2,
            target_q,
        )
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic_v.parameters(),
            float(max_grad_norm) if max_grad_norm > 0.0 else float("inf"),
        )
        self.optimizer.step()
        return float(grad_norm)
