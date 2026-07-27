import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, input_dim, output_dim):
        super(CriticNet, self).__init__()
        self.in_to_y1 = nn.Linear(input_dim + output_dim, 256)
        self.in_to_y1.weight.data.normal_(0, 0.1)
        self.y1_to_y2 = nn.Linear(256, 256)
        self.y1_to_y2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(256, 1)
        self.out.weight.data.normal_(0, 0.1)

        self.q2_in_to_y1 = nn.Linear(input_dim + output_dim, 256)
        self.q2_in_to_y1.weight.data.normal_(0, 0.1)
        self.q2_y1_to_y2 = nn.Linear(256, 256)
        self.q2_y1_to_y2.weight.data.normal_(0, 0.1)
        self.q2_out = nn.Linear(256, 1)
        self.q2_out.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
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
    def __init__(self, capacity, dims, recent_window=50000, recent_ratio=0.75):
        self.capacity = int(capacity)
        self.mem = np.zeros((self.capacity, dims), dtype=np.float32)
        self.memory_counter = 0
        self.recent_window = max(1, min(int(recent_window), self.capacity))
        self.recent_ratio = float(np.clip(recent_ratio, 0.0, 1.0))

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
            sample_index = np.random.randint(0, pool_size, size=n)
            return self.mem[sample_index, :]

        start_global = max(0, self.memory_counter - pool_size)
        recent_available = min(pool_size, self.recent_window)
        recent_start_global = self.memory_counter - recent_available

        recent_count = min(int(round(n * self.recent_ratio)), n)
        historical_count = n - recent_count
        sample_parts = []

        if recent_count > 0:
            recent_global = np.random.randint(
                recent_start_global,
                self.memory_counter,
                size=recent_count,
            )
            sample_parts.append(recent_global % self.capacity)

        if historical_count > 0:
            if recent_start_global > start_global:
                historical_global = np.random.randint(
                    start_global,
                    recent_start_global,
                    size=historical_count,
                )
                sample_parts.append(historical_global % self.capacity)
            else:
                recent_global = np.random.randint(
                    recent_start_global,
                    self.memory_counter,
                    size=historical_count,
                )
                sample_parts.append(recent_global % self.capacity)

        sample_index = np.concatenate(sample_parts, axis=0)
        np.random.shuffle(sample_index)
        new_mem = self.mem[sample_index, :]
        return new_mem


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

    def learn(self, actor_loss):
        loss = actor_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Entroy():
    def __init__(self, action_dim=32):
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(
            1,
            requires_grad=True,
            device=device,
        )
        self.alpha = self.log_alpha.exp()
        self.optimizer = torch.optim.Adam([self.log_alpha], lr=q_lr)

    def learn(self, entroy_loss):
        loss = entroy_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Critic():
    def __init__(self, state_dim, action_dim):
        self.critic_v = CriticNet(state_dim, action_dim).to(device)
        self.target_critic_v = CriticNet(state_dim, action_dim).to(device)
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

    def learn(self, current_q1, current_q2, target_q):
        loss = self.lossfunc(current_q1, target_q) + self.lossfunc(
            current_q2,
            target_q,
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
