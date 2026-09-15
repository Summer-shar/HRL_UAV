# Phase3 实验执行计划

## 当前状态总览

### 已有结果

| 实验 | 状态 | J | Centroid Distance | 说明 |
|------|------|---|-------------------|------|
| SAC预训练 | ✅ 完成 | ~0.55 | N/A | window2000 |
| PPO训练(原始) | ✅ 完成 | ~0.55 | 90m | window120, 性能不理想 |
| CVX/Predictive | ✅ 有数据 | 0.49 | 4.88m | Oracle基线 |

### 关键问题

1. **PPO centroid distance 90m vs Predictive 5m** - 宏观策略无法有效跟踪质心
2. **缺少联合训练对照** - 无法验证分层架构价值
3. **Phase2超参数修复** - `gae_lambda=0.97`, `macro_gamma_frames=10` 未应用

---

## 实验设计

### 对照实验矩阵

| ID | 实验组 | PPO训练 | SAC训练 | 预算 | 目标J | 预期时长 |
|----|--------|--------|--------|------|--------|----------|
| **E0** | Hover Baseline | - | - | 0 | ~0.70 | 0.5h |
| **E1** | 基准组(论文) | Fixed |grounds | Fixed | J=0.553 | - |
| **E2** | 改进PPO+固定SAC | **训练** | Fixed | 12 PPO更新 | <0.55 | 4h |
| **E3** | 改进PPO+训练SAC(联合) | **训练** | **训练** | 60 PPO + 600K SAC | <0.50 | 12h |
| **E4** | Predictive Oracle | Rule | Rule | 0 | 0.495 | 0.5h |
| **E5** | CVX-R | Solver | Solver | 0 | ~0.50 | 1h |

### 核心假设验证

| 假设 | 验证条件 | 成功标准 |
|------|----------|----------|
| H1: 改进PPO有效 | E2 vs 原始PPO | centroid < 30m |
| H2: 联合训练有价值 | E3 vs max(E1, E2) | J_E3 < J_others |
| H3: 分层优于单层 | E3 vs E5(CVX) | J_E3 ≤ J_E5 |

---

## 执行脚本清单

### 阶段1: 改进PPO训练 (Step 1-3)

#### Step 1: 验证Phase2超参数修复

**文件**: `hrl_uav/mission_ppo_qualification.py`

**修改内容**:
```python
# 在 PPO_SETTINGS 中添加 Phase2 修复
PPO_SETTINGS = dict(
    # ... 原有设置 ...
    gae_lambda=0.97,           # Phase2修复: 原来是0
    macro_gamma_frames=10,     # Phase2修复: 有效 horizon = 0.99^10 = 0.904
)
```

**验证命令**:
```powershell
python -m unittest tests.test_mission_ppo_qualification.RunnerTests -v
```

#### Step 2: 运行改进PPO Qualification

**脚本**: `scripts/run_mission_ppo_qualification.py`

**命令**:
```powershell
python scripts/run_mission_ppo_qualification.py `
    --output results/mission_ppo_qualification_improved `
    --run-authorized
```

**预期输出目录结构**:
```
results/mission_ppo_qualification_improved/
├── manifest.json
├── seed201/
│   ├── boundary000/
│   ├── boundary040/
│   └── monitor120/
├── seed202/
└── seed203/
```

**预期结果**:
- Window120: centroid_distance_mean < 30m
- J < 0.55

#### Step 3: 分析改进效果

**分析指标**:
```python
metrics = [
    "J",                    # 目标: 下降
    "business_error_m_mean",  # centroid distance proxy
    "J_holding",            # 时延分解
    "J_energy",             # 能耗分解
    "task_delay_weighted_ms"
]
```

**对比表**:
| 指标 | 原始PPO | 改进PPO | 变化率 |
|------|---------|---------|--------|
| J | 0.553 | ? | ? |
| Centroid | 90m | ? | ? |

---

### 阶段2: 奖励塑造验证 (Step 4-5)

#### Step 4: 实现势函数奖励

**文件**: `hrl_uav/learning.py`

**新增函数**:
```python
def potential_distance_shaping(positions, centroids, scale=50.0):
    """
    势函数奖励: 基于质心距离的塑造型奖励
    
    phi(s) = -mean_distance / scale
    reward = phi(s') - phi(s)  (potential-based)
    """
    pos = np.asarray(positions).reshape(-1, 2)
    cent = np.asarray(centroids).reshape(-1, 2)
    distances = np.linalg.norm(pos - cent, axis=1)
    return -distances.mean() / scale
```

**修改位置**: `mission_ppo_stage4a.py` 的 `collect_stream` 函数

**宏奖励修改**:
```python
# 在 compute macro reward 后添加势函数奖励
macro_reward = sum(gamma**i * r for i, r in enumerate(rewards)) / len(rewards)
macro_reward += lambda_shaping * distance_shaping_reward(prev_positions, curr_positions)
```

#### Step 5: 对比实验

**实验设计**:
| 实验 | 奖励塑造 λ | 预期效果 |
|------|------------|----------|
| A | λ=0 (无) | 基线性能 |
| B | λ=0.1 | 轻微改善 |
| C | λ=0.3 | 中等改善 |
| D | λ=1.0 | 过度干预风险 |

**命令**:
```powershell
# A: 无奖励塑造
python scripts/run_mission_ppo_qualification.py `
    --output results/ppo_no_shaping `
    --shaping_lambda 0.0

# C: λ=0.3
python scripts/run_mission_ppo_qualification.py `
    --output results/ppo_shaping_0.3 `
    --shaping_lambda 0.3
```

---

### 阶段3: 联合训练验证 (Step 6-8)

#### Step 6: 验证联合训练模块

**文件**: `hrl_uav/mission_joint_training.py`

**核心类**:
```python
class JointTrainingConfig:
    mode: TrainingMode = TrainingMode.JOINT
    ppo_update_interval: int = 10  # 1:10 更新比例
    total_micro_updates: int = 600_000
```

**单元测试**:
```powershell
python -m unittest tests.test_mission_joint_training -v
```

#### Step 7: 运行联合训练

**命令**:
```powershell
# E3: 联合训练 (主实验)
python scripts/run_mission_joint_training.py `
    --output results/joint_training_improved_ppo `
    --experiment joint_1_10 `
    --run-authorized

# E2: PPO-only 对照
python scripts/run_mission_joint_training.py `
    --output results/ppo_only_joint `
    --experiment ppo_only `
    --run-authorized
```

**预算匹配**:
```
联合训练: 600K micro updates = 60 PPO updates + 600K SAC updates
PPO-only:  60 PPO updates, SAC frozen
```

#### Step 8: 结果分析

**对照表**:
| 实验 | J | J变化 | Centroid | 说明 |
|------|---|-------|----------|------|
| 原始PPO | 0.553 | - | 90m | 基准 |
| 改进PPO (E2) | ? | vs 0.553 | <30m? | 单独训练 |
| 联合 (E3) | ? | vs E2 | <15m? | 联合训练 |

---

### 阶段4: 最终对照 (Step 9-10)

#### Step 9: 运行所有基线

```powershell
# Predictive Oracle
python scripts/run_mission_predictive.py `
    --output results/predictive_oracle

# CVX-R
python scripts/run_mission_xu_cvx_r.py `
    --output results/xu_cvx_r_improved_ppo
```

#### Step 10: 生成最终对比报告

**脚本**: `scripts/plot_mission_paper_comparison.py`

**报告内容**:
1. 主对比表 (J, 分解, 质心)
2. 统计显著性检验
3. 学习曲线
4. 消融分析

---

## 预期时间表

```
Week 1 (Day 1-5):
├── Day 1: Step 1-2 (PPO qualification 改进)
├── Day 2: Step 3 (分析 + Step 4 奖励塑造实现)
├── Day 3: Step 5 (奖励塑造对比)
├── Day 4: Step 6 (联合训练模块验证)
└── Day 5: Step 7 (联合训练运行)

Week 2 (Day 6-10):
├── Day 6-7: 联合训练继续 (12h)
├── Day 8: Step 8 (联合训练分析)
├── Day 9: Step 9 (基线运行)
└── Day 10: Step 10 (最终报告)
```

---

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| PPO改进效果不明显 | 中 | 调整学习率或增加训练窗口 |
| 联合训练不稳定 | 高 | 使用渐进式更新比例 |
| 奖励塑造过度干预 | 中 | 从小λ开始验证 |
| 预算超支 | 低 | 监控micro updates，设置early stopping |

---

## 验收标准

### 必须满足

- [ ] 改进PPO centroid < 30m (vs 原始90m)
- [ ] 联合训练J < 改进PPO J
- [ ] 所有实验可复现 (相同seed, 相同配置)

### 目标满足

- [ ] 联合训练J < Predictive J
- [ ] 奖励塑造λ=0.3有效改善跟踪
- [ ] 学习曲线单调收敛

---

## 执行命令汇总

### 阶段1: 改进PPO训练

```powershell
# Step 2: 运行改进PPO qualification
python scripts/run_mission_ppo_qualification.py `
    --output results/mission_ppo_qualification_improved `
    --run-authorized-qualification

# Step 2续: 断点续训 (如中断)
python scripts/run_mission_ppo_qualification.py `
    --output results/mission_ppo_qualification_improved `
    --run-authorized-qualification `
    --resume
```

### 阶段2: 联合训练

```powershell
# Step 7a: PPO-only 对照
python scripts/run_mission_joint_training.py `
    --output results/ppo_only_improved `
    --experiment ppo_only `
    --run-authorized

# Step 7b: 联合训练 (主实验)
python scripts/run_mission_joint_training.py `
    --output results/joint_improved_ppo `
    --experiment joint_1_10 `
    --run-authorized
```

### 阶段3: 基线对照

```powershell
# Predictive Oracle
python scripts/run_mission_predictive.py `
    --output results/predictive_oracle

# CVX-R
python scripts/run_mission_xu_cvx_r.py `
    --output results/xu_cvx_r_improved_ppo
```

### 阶段4: 结果分析

```powershell
# 生成对比报告
python scripts/plot_mission_paper_comparison.py `
    --results_dir results/ `
    --output results/comparison_report
```

---

## 脚本清单汇总

| 脚本 | 用途 | 预计时长 | 依赖 |
|------|------|----------|------|
| `run_mission_ppo_qualification.py` | 改进PPO qualification | 4h | SAC window2000 |
| `run_mission_joint_training.py` | 联合训练 | 12h | SAC window2000 |
| `run_mission_predictive.py` | Predictive基线 | 0.5h | - |
| `run_mission_xu_cvx_r.py` | CVX-R基线 | 1h | - |
| `plot_mission_paper_comparison.py` | 结果可视化 | 0.5h | 其他实验结果 |

**总计**: ~18h (可在2周内完成)

---

## 下一步行动

1. **立即执行**: 审查 `mission_ppo_qualification.py` 中的Phase2超参数修复
2. **Day 1**: 运行 `run_mission_ppo_qualification.py` 改进版
3. **Day 3**: 分析PPO改进效果，决定是否继续联合训练
4. **Day 4-6**: 运行联合训练
5. **Day 7-10**: 基线对照 + 最终报告
