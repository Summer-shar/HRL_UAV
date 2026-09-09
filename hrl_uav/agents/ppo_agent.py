# -*- coding: utf-8 -*-
# [MAPPO UPGRADE] 宏观层升级为参数共享 MAPPO（CTDE）
# Actor:  输入单机局部观测，输出单机连续动作
# Critic: 输入全局拼接状态，输出全局状态价值

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
    - 输出：集中式宏观 waypoint 动作的高斯分布参数
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
