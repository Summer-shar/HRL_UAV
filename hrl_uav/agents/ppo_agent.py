# -*- coding: utf-8 -*-
# [MAPPO UPGRADE] 宏观层升级为参数共享 MAPPO（CTDE）
# Actor:  输入单机局部观测，输出单机连续动作
# Critic: 输入全局拼接状态，输出全局状态价值

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== Hyperparameters ====================
GAMMA = 0.9
A_LR = 3e-4
C_LR = 1e-3
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
CLIP_EPSILON = 0.2


# ==================== Networks ====================
class ActorNet(nn.Module):
    """
    [MAPPO UPGRADE]
    参数共享 Actor：
    - 输入：单架无人机的 local observation
    - 输出：单架无人机的连续动作 [rho, theta]
    """

    def __init__(self, inp, outp):
        super(ActorNet, self).__init__()

        self.in_to_y1 = nn.Linear(inp, 256)
        self.in_to_y1.weight.data.normal_(0, 0.1)

        self.y1_to_y2 = nn.Linear(256, 256)
        self.y1_to_y2.weight.data.normal_(0, 0.1)

        self.out = nn.Linear(256, outp)
        self.out.weight.data.normal_(0, 0.1)

        # [MAPPO UPGRADE]
        # 使用全局可学习 log_std，避免早期状态相关大方差导致动作总打满边界。
        self.log_std = nn.Parameter(
            torch.full((outp,), -0.8, dtype=torch.float32)
        )

    def forward(self, inputstate):
        inputstate = F.relu(self.in_to_y1(inputstate))
        inputstate = F.relu(self.y1_to_y2(inputstate))

        mean = torch.tanh(self.out(inputstate))
        log_std = torch.clamp(self.log_std, -5.0, 1.0)
        std = log_std.exp().expand_as(mean)

        return mean, std


class CriticNet(nn.Module):
    """
    [MAPPO UPGRADE]
    中心化 Critic：
    - 输入：所有无人机局部状态拼接后的 global state
    - 输出：全局状态价值 V(s)
    """

    def __init__(self, inp, outp=1):
        super(CriticNet, self).__init__()

        self.in_to_y1 = nn.Linear(inp, 256)
        self.in_to_y1.weight.data.normal_(0, 0.1)

        self.y1_to_y2 = nn.Linear(256, 256)
        self.y1_to_y2.weight.data.normal_(0, 0.1)

        self.out = nn.Linear(256, outp)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, inputstate):
        inputstate = F.relu(self.in_to_y1(inputstate))
        inputstate = F.relu(self.y1_to_y2(inputstate))

        return self.out(inputstate)


# ==================== Agent Classes ====================
class Actor:
    """
    [MAPPO UPGRADE]
    参数共享 Actor 封装类。
    """

    def __init__(self, state_dim, action_dim):
        self.old_pi = ActorNet(state_dim, action_dim).to(device)
        self.new_pi = ActorNet(state_dim, action_dim).to(device)

        self.optimizer = torch.optim.Adam(
            self.new_pi.parameters(),
            lr=A_LR,
            eps=1e-5,
        )

        self.update_oldpi()

    def choose_action(self, s):
        """
        [MAPPO UPGRADE]
        输入:
            s: shape = (N, local_obs_dim) 或 (local_obs_dim,)
        输出:
            action:      shape = (N, action_dim) 或 (action_dim,)
            log_prob:    shape = (N, 1) 或 (1,)
        """
        inputstate = torch.as_tensor(
            s,
            dtype=torch.float32,
            device=device,
        )

        single_input = False
        if inputstate.dim() == 1:
            inputstate = inputstate.unsqueeze(0)
            single_input = True

        mean, std = self.old_pi(inputstate)
        dist = torch.distributions.Normal(mean, std)

        action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)

        action_logprob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        action_np = action.detach().cpu().numpy()
        logprob_np = action_logprob.detach().cpu().numpy()

        if single_input:
            return action_np[0], logprob_np[0]

        return action_np, logprob_np

    def choose_action_deterministic(self, s):
        """
        [MAPPO UPGRADE]
        评估时使用确定性动作（均值动作）。
        """
        inputstate = torch.as_tensor(
            s,
            dtype=torch.float32,
            device=device,
        )

        single_input = False
        if inputstate.dim() == 1:
            inputstate = inputstate.unsqueeze(0)
            single_input = True

        mean, _ = self.old_pi(inputstate)
        action = torch.clamp(mean, -1.0, 1.0)
        action_np = action.detach().cpu().numpy()

        if single_input:
            return action_np[0]

        return action_np

    def update_oldpi(self):
        """
        将 new_pi 的参数同步到 old_pi。
        """
        self.old_pi.load_state_dict(self.new_pi.state_dict())

    def learn(self, bs, ba, adv, bap):
        """
        [MAPPO UPGRADE]
        使用展平后的参数共享 batch 更新 Actor。

        输入:
            bs:  batch local states,  shape = (T * N, local_obs_dim)
            ba:  batch actions,       shape = (T * N, action_dim)
            adv: batch advantages,    shape = (T * N, 1)
            bap: batch old logprobs,  shape = (T * N, 1)
        """
        bs = torch.as_tensor(bs, dtype=torch.float32, device=device)
        ba = torch.as_tensor(ba, dtype=torch.float32, device=device)
        adv = torch.as_tensor(adv, dtype=torch.float32, device=device)
        bap = torch.as_tensor(bap, dtype=torch.float32, device=device)

        if adv.numel() > 1:
            adv_std = adv.std(unbiased=False)
            adv = (adv - adv.mean()) / (adv_std + 1e-8)
        else:
            adv = adv - adv.mean()

        for _ in range(A_UPDATE_STEPS):
            mean, std = self.new_pi(bs)
            dist_new = torch.distributions.Normal(mean, std)

            action_new_logprob = dist_new.log_prob(ba).sum(
                dim=-1,
                keepdim=True,
            )

            ratio = torch.exp(action_new_logprob - bap.detach())

            surr1 = ratio * adv
            surr2 = torch.clamp(
                ratio,
                1 - CLIP_EPSILON,
                1 + CLIP_EPSILON,
            ) * adv

            loss = -torch.min(surr1, surr2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.new_pi.parameters(), 0.5)
            self.optimizer.step()

        return loss.item()


class Critic:
    """
    [MAPPO UPGRADE]
    中心化 Critic 封装类。
    """

    def __init__(self, state_dim, value_dim=1):
        self.value_dim = value_dim
        self.critic_v = CriticNet(state_dim, value_dim).to(device)

        self.optimizer = torch.optim.Adam(
            self.critic_v.parameters(),
            lr=C_LR,
            eps=1e-5,
        )

        self.lossfunc = nn.MSELoss()

    def get_v(self, s):
        """
        输入 numpy 全局状态，返回 V(s) tensor。
        """
        inputstate = torch.as_tensor(
            s,
            dtype=torch.float32,
            device=device,
        )

        if inputstate.dim() == 1:
            inputstate = inputstate.unsqueeze(0)

        return self.critic_v(inputstate)

    def learn(self, bs, br):
        """
        [MAPPO UPGRADE]
        Critic 更新，返回:
            advantage (numpy)
            critic_loss (float)

        输入:
            bs: batch global states, shape = (T, global_obs_dim)
            br: batch discounted returns, shape = (T, 1)
        """
        bs_t = torch.as_tensor(
            bs,
            dtype=torch.float32,
            device=device,
        )
        reality_v = torch.as_tensor(
            br,
            dtype=torch.float32,
            device=device,
        )

        if reality_v.dim() == 1:
            reality_v = reality_v.unsqueeze(1)

        with torch.no_grad():
            current_v = self.critic_v(bs_t)
            advantage = (reality_v - current_v).cpu().numpy()

        for _ in range(C_UPDATE_STEPS):
            v = self.critic_v(bs_t)
            td_e = self.lossfunc(reality_v, v)

            self.optimizer.zero_grad()
            td_e.backward()
            nn.utils.clip_grad_norm_(self.critic_v.parameters(), 0.5)
            self.optimizer.step()

        return advantage, td_e.item()
