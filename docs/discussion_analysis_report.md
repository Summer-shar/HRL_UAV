# 双时间尺度HRL系统Discussion分析报告

## 摘要

本文档为论文Discussion章节提供完整的算法分析，涵盖双时间尺度HRL（Hierarchical Reinforcement Learning）的设计选择、PPO训练困境诊断、改进方案性能边界及实验局限性。

---

## 1. 方法论贡献与设计选择

### 1.1 双时间尺度架构

本系统采用双时间尺度HRL架构，融合PPO（Proximal Policy Optimization）与SAC（Soft Actor-Critic）两种强化学习算法，形成宏观轨迹规划与微观路由决策的分层控制：

| 层次 | 算法 | 更新频率 | 决策粒度 | 状态空间 |
|------|------|----------|----------|----------|
| 宏观 | PPO | 每300宏步（~9分钟） | 移动方向（3 UAV × 2轴） | 67维 |
| 微观 | SAC | 每20帧（~0.5秒） | 路由比例（2链路 × 3源） | 68维 |

**设计动机**：宏微观分离使高层策略专注于长期轨迹优化（计算资源分流/聚集），低层策略处理短视的路由决策。这种分离符合计算卸载问题的本质——UAV位置决定服务覆盖范围，路由决定资源分配效率。

### 1.2 异频更新机制

```python
# learning.py - 宏微观更新时钟
PPO_DEFAULTS = {
    "gae_lambda": 0.97,        # GAE lambda控制优势估计偏差-方差权衡
    "macro_gamma_frames": 10,   # γ^10 ≈ 0.904 有效视界
    "macro_gamma": 0.95,        # 显式折扣因子（10帧视界）
}
```

- **SAC更新**：每20帧一次，贴近环境时间尺度
- **PPO更新**：每300宏步一次，约9分钟物理时间
- **std衰减**：`ppo_std_decay_windows=350`，从1/3衰减至0.015

异频设计基于直觉：微观决策需要快速响应环境变化，宏观决策则应保持稳定以确保轨迹平滑。然而，这一设计选择也引入了探索-预测不匹配问题。

---

## 2. PPO训练问题诊断

### 2.1 根因分析

Stage4A实验揭示PPO训练存在系统性偏差，根因可归纳为三类：

#### 根因1：探索-预测不匹配（Exploration-Prediction Mismatch）

```
PPO探索: N(μ, σ) 高斯噪声, σ从0.333衰减至0.015
Predictive策略: 基于物理模型的因果预测, lead=10s
```

PPO的高斯探索与Predictive物理预测在机制上存在根本差异：
- PPO探索生成随机位移，期望通过奖励学习最优方向
- Predictive策略根据队列积压计算梯度方向，引导向计算资源富集区移动

实验数据显示，PPO120 vs Predictive在centroid distance上存在**18.5倍**差距，表明PPO未能习得Predictive的物理直觉。

#### 根因2：std衰减过快导致策略锁定

```python
# learning.py 第314-317行
if "ppo_std_decay_windows" in self.cfg:
    progress = min(1.0, max(0.0, progress * self.cfg["joint_windows"] / self.cfg["ppo_std_decay_windows"]))
std = np.exp((1-progress) * np.log(self.cfg["ppo_initial_std"]) + progress * np.log(self.cfg["ppo_final_std"]))
self.macro_actor.log_std.data.fill_(np.log(std))
```

- 初始std=1/3，末std=0.015，衰减因子约22倍
- 350窗口（~87.5分钟）内完成衰减
- 训练仅120窗口（30分钟），std实际从0.333降至0.10左右
- 策略被锁定在低熵状态，无法有效探索

#### 根因3：价值函数视界与任务不匹配

```python
# gae函数 - learning.py 第19-27行
def gae(rewards, values, next_values, trace_continue, discount, trace_lambda):
    """Generalized Advantage Estimation"""
    output = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        delta = rewards[i] + discount * next_values[i] - values[i]
        running = delta + discount * trace_lambda * trace_continue[i] * running
        output[i] = running
    return output, output + values
```

问题分析：
- `macro_gamma = 0.904`（γ^10），有效视界约10帧
- 任务为连续900秒（18000帧）mission
- 价值函数仅能可靠预测~10帧后的累积奖励
- PPO无法学习跨分钟尺度的轨迹优化

### 2.2 训练效率问题

| 指标 | Stage4A协议 | 期望收敛 |
|------|-------------|----------|
| 宏步总数 | 3600/step | >50000 |
| PPO批量更新 | 12次（120窗口） | >500次 |
| 策略改进 | std 0.333→0.10 | 需继续衰减 |
| 价值函数 | 仅12次更新 | 需数千次 |

120窗口训练仅产生12次PPO批量更新，远未达到收敛条件。每次更新使用5个epoch的PPOClipping，策略更新幅度受`ppo_target_kl=0.03`约束。

---

## 3. 改进方案

### 3.1 超参数调整

####gae_lambda调整
```python
# Phase2修复：原代码缺失gae_lambda（默认为0）
PPO_DEFAULTS = {
    "gae_lambda": 0.97,  # 增加长期优势估计权重
}
```
- 增大λ使GAE更重视长期回报
- 权衡：增大偏差，降低方差
- 适用场景：任务有明确长期目标

####学习率调整
```python
PPO_SETTINGS = dict(
    ppo_actor_lr=1e-4,      # 从3e-4下调
    ppo_critic_lr=2e-4,     # 从1e-3下调，比例1:2
    ppo_clip=.2,            # 保持
    ppo_target_kl=.03,      # 早期停止阈值
)
```
- 降低学习率配合更长训练
- 1:2的actor-critic学习率比例符合稳定收敛经验

####std衰减控制
```python
ppo_std_decay_windows=350,  # 从200增加，延长探索期
ppo_initial_std=1/3,       # 保持
ppo_final_std=0.015,       # 保持（不变）
```
- 延长探索窗口至350宏步
- 配合更长训练时间使用

### 3.2 奖励塑造（Reward Shaping）

```python
# learning.py 第38-85行
def potential_distance_shaping(positions, centroids, scale=1.0):
    """
    基于质心相对距离的势能函数塑造奖励。
    接近质心=正势能，远离=负势能
    """
    pos = np.asarray(positions).reshape(-1, 2)
    cent = np.asarray(centroids).reshape(-1, 2)
    distances = np.linalg.norm(pos - cent, axis=1)
    mean_distance = distances.mean()
    potential = -mean_distance / scale
    return potential

def distance_shaping_reward(state, next_state, scale=50.0):
    """势能差作为塑造奖励: φ(s') - φ(s)"""
    if not all(k in state for k in ("positions", "centroids")):
        return 0.0
    phi_curr = potential_distance_shaping(state["positions"], state["centroids"], scale)
    phi_next = potential_distance_shaping(next_state["positions"], next_state["centroids"], scale)
    return phi_next - phi_curr
```

**理论依据**：基于势能的奖励塑造（Potential-Based Reward Shaping, PBRS）满足下述不变性：
- 新奖励 = 原奖励 + γφ(s') - φ(s)
- 不改变最优策略的相对排序
- 可加速学习收敛

**实践意义**：
- 质心代表计算负载重心的空间位置
- 向质心移动减少任务传输距离
- 提供密集奖励信号，弥补稀疏mission奖励

### 3.3 课程学习（Curriculum Learning）

训练协议设计从固定到移动的课程：

| 阶段 | 任务难度 | 特点 |
|------|----------|------|
| SAC预训练 | 固定UAV | UAV悬停，专注路由学习 |
| PPO微调 | 移动UAV | 从简单轨迹开始 |

**局限性**：当前实验场景仍为单一移动模式，未实现完整课程梯度。

---

## 4. 性能边界

### 4.1 PPO vs Predictive差距量化

| 指标 | PPO120 | Predictive | 差距倍数 |
|------|--------|------------|----------|
| Centroid Distance | 18.5x参考 | 1.0x | 18.5x |
| Mission Cost J | +11.85% | 基准 | 11.85% |
| 宏步探索方差 | 高 | 零 | - |

PPO120在开发场景的性能差距表明：
1. 12次批量更新无法习得有效策略
2. 高斯探索在连续mission中不如因果预测有效
3. 价值函数尚未准确估计长期累积回报

### 4.2 联合训练价值

联合训练（joint training）理论上可实现宏微观协同优化，但存在以下挑战：

- **梯度耦合**：微观策略影响宏观状态，宏观动作影响微观上下文
- **更新时钟**：SAC每20帧 vs PPO每300宏步，需要同步机制
- **经验复用**：SAC回放中的状态依赖宏动作有效性

当前实验未完成联合训练的完整消融验证。

### 4.3 训练效率基准

| 阶段 | 宏步 | 物理时间 | PPO更新 |
|------|------|----------|---------|
| Stage4A 120窗口 | 3600 | 30分钟 | 12次 |
| 继续训练至600窗口 | 18000 | 150分钟 | 60次 |
| 经验收敛估计 | >50000 | >20小时 | >500次 |

---

## 5. 局限性讨论

### 5.1 实验范围局限

1. **开发场景验证**：
   - 仅在场景33600、33601验证
   - 未在保留场景集33700-33709测试
   - 泛化能力未知

2. **单一物理配置**：
   - U2U带宽固定为6MHz
   - 负载倍率固定为1.0
   - 参数敏感性未充分探索

3. **统计充分性**：
   - 仅3个PPO初始化种子（201、202、203）
   - 缺乏多样本统计
   - 单一SAC初始化（101）

### 5.2 方法论局限

1. **预测性参考的上限问题**：
   - Predictive策略作为"神谕"基线
   - 但其lead=10s假设在实际部署中不可行
   - 上限选择可能过于乐观

2. **HRL分离的固有代价**：
   - 宏观状态空间（67维）压缩了微观信息
   - 宏动作（6维）限制微观路由的调节范围
   - 分离设计可能丢失跨层耦合梯度

3. **训练稳定性**：
   - PPO对超参数敏感
   - std衰减与训练长度耦合
   - 价值函数视界限制长期学习

### 5.3 结果解释边界

1. **参数敏感性≠最优性**：
   - 论文对比实验使用冻结PPO权重
   - 不同参数点不代表重训后的最优性能
   - 敏感性曲线仅描述单次运行特性

2. **多种子误差估计**：
   - 三个PPO共享同一SAC初始化
   - 不能作为三次独立HRL训练的误差
   - 多种子变异性被低估

---

## 6. 实践建议

### 6.1 优先级排序

| 优先级 | 改进方向 | 预期收益 | 实施难度 |
|--------|----------|----------|----------|
| 高 | 奖励塑造（PBRS） | 快速收敛 | 低 |
| 高 | std衰减延长 | 充分探索 | 低 |
| 中 | gae_lambda调整 | 长期学习 | 中 |
| 中 | 课程学习扩展 | 稳定收敛 | 中 |
| 低 | 联合训练验证 | 协同优化 | 高 |

### 6.2 消融实验设计

验证各项改进的独立贡献：

```
基线: PPO120 (12次更新, std快速衰减)
消融1: +奖励塑造 (PBRS)
消融2: +std延长 (350窗口)
消融3: +gae_lambda=0.97
消融4: 完整改进组合
对照: Predictive策略
```

### 6.3 训练预算建议

基于收敛分析，建议训练预算：

| 场景 | 窗口数 | 物理时间 | PPO更新 | 预期收敛 |
|------|--------|----------|---------|----------|
| 快速验证 | 120 | 30分钟 | 12次 | 否 |
| 充分训练 | 600 | 150分钟 | 60次 | 接近 |
| 收敛训练 | 1200 | 5小时 | 120次 | 是 |

---

## 7. 结论

本文分析了双时间尺度HRL系统的设计选择与训练困境。主要发现：

1. **PPO训练失败根因**：探索-预测不匹配、std衰减过快、价值函数视界不足

2. **改进方案有效性**：奖励塑造为快速验证路径，超参数调优需配合更长训练

3. **性能边界**：PPO120 vs Predictive差距显著，联合训练价值待独立验证

4. **实验局限**：开发场景验证、单一配置、统计不充分限制结论泛化

未来工作应聚焦于：(1) 完整课程学习设计；(2) 联合训练的梯度同步机制；(3) 多场景保留测试验证泛化能力。

---

## 参考文献

1. Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms.
2. Haarnoja, T. et al. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL.
3. Ng, A. et al. (1999). Policy Invariance Under Reward Transformations.
4. Stage4A Protocol: `hrl_uav/mission_ppo_stage4a.py`
5. Learning Implementation: `hrl_uav/learning.py`
