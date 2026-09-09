# 多无人机协作计算：纯 PPO + Direct SAC

当前目录只有一条主线，不再用版本号拼接训练脚本。

**训练流程：微观 SAC 从零预训练 → PPO 和 SAC 按不同频率同步更新。**
没有 QA 教师、行为克隆、SCA 锚定或残差门控。宏观控制相对质心的自由二维 waypoint；微观独立输出双优先级路由。CPU 和带宽由公开的物理调度规则分配，不宣称它们也是 SAC 动作。

## 当前入口与结果（2026-09-07）

`--mode mainline` 和 `--mode mainline-check` 默认使用 `configs/hrl_ppo_only_unlock.json`。其他模式仍用 `configs/main.json`。显式传入历史配置仍按原配置运行，并在主线入口提示其是否采用旧的 SAC 阶段切换逻辑。

新入口为 **SAC 从零预训练 200 窗口，然后只开启 PPO，继续 600 窗口异频同步更新**。SAC 全程使用同一配方：Actor / Critic / 熵温度优化器学习率分别为 `3e-5 / 3e-4 / 3e-5`；权重、target、优化器完整状态、replay、采样 RNG、更新计数和 Actor 预热时钟在阶段边界全部保留。没有冻结 SAC，没有重建 Critic，没有教师。奖励、状态、动作、场景物理参数、SAC 更新频率和 PPO 超参数本次不变。

`sac_stage_policy=preserve` 禁止同时配置 `joint_lr` 或联合阶段 SAC 预热覆盖值，防止混用。运行时的 `joint_transition_audit.json` 核验完整优化器状态（包含学习率），不再只检查动量。历史配置缺少该字段时保留旧语义，供追溯和原样续训；不能将旧联合 checkpoint 换配置后冒充新方案的连续训练。

新入口的训练场景编号连续：预训练从 61000 开始，联合从 61067 接续，不再跳到 71000。这里的“连续”仅指场景编号，阶段边界仍结束当前 rollout 并重置环境；200 不是 3 的倍数，边界为两窗口短 stream。以后若做纯 SAC 与 HRL 的训练因果对照，必须匹配同样的场景、重置边界和预算，不能直接用历史整段 800 窗口代替。PPO 启用后动作采样和访问状态本就会不同，不能要求后续 SAC 参数逐位不变。

**已完成的是旧配方的 200+600 窗口训练，新入口尚无性能结果。** 旧配方联合阶段曾把 SAC Critic 学习率降至 `1e-4`、熵温度学习率降至 `1e-5`。三个重复使用的开发评估场景上：旧联合 PPO+SAC 为 **658.17 ms**；同 final SAC 强制质心为 **654.30 ms**；历史微观独训 800 窗口为 **632.34 ms**。前两者的配对差 **+3.87 ms** 是部署宏观消融；后两者差 **+21.95 ms** 混有训练场景、边界和 SAC 配方差异，不能证明它全由 PPO 导致，也不能证明改学习率就会恢复。

联合阶段采用最终评估模式，仍保存中间 checkpoint；预训练保留已有质量读数，是否开启 PPO 按固定 200 窗口和有效性检查执行，不以挑选 best 代替 final。这次只修正入口和验证，没有自动启动新一轮长训。

## 早期检查记录（不代表当前进度）

**早期状态：split_v1 的首次 60 窗口微观预训练未通过资格检查，当时没有启动 PPO 联训。此前 104/116 维状态对照也未通过。2026-09-06 恢复旧版负载分配、保留 6 MHz U2U 后，先完成零训练资源检查，再按用户授权运行该轮训练；结果见下文。历史 v64 的纯 SAC 曾有约 743 ms 的趋稳结果，新版失败不能否定历史结果。**

补充历史核验：更早的 **v33 也不使用 QA、教师或残差**，SAC 预训练 final 的归档值为 **616.625 ms**，联训 final 为 **631.050 ms**。其负载倍率为 **1.0**、微观状态 48 维、预训练 300 轮，与当前主配置不同。已解析历史入口并完成禁止求解器导入的前向检查；尚未找回原 v33 权重，**没有重训或宣称本次复现该时延**。旧收敛图及逐项对照见 [v33 无 QA 核验](docs/v33_no_qa_recovery.md)。

**当前接口更新：PPO 与 SAC 分别建模、共享同一物理环境。** `split_v1` 中宏观输入 67 维、微观输入 68 维，各自使用对应的网络和 transition；不再共用 104 维输入。先 SAC 预训练，再继承本次 SAC 权重、优化器与 replay 开放 PPO/SAC 异频联合更新。此次不改场景、奖励、动作或训练预算，不加入教师。旧共享观测保留供历史核验，不加载旧权重冒充新结果。候选状态定义、信息来源及部分可观测性边界见 [分层建模](docs/model.md#状态)。接口测试不等于性能或收敛通过。

接口核验已完成：70 项单元测试通过；`results/split_mdp_smoke` 为 **SMOKE_ONLY**，预训练 SAC Actor 更新 10 次且 PPO 不更新，联合阶段 SAC 再更新 18 次、PPO 更新 1 次。两阶段 replay 接续，未加载教师或旧权重。该 smoke 本身不是完整预算训练，也不是收敛成绩；后续完整预训练结果如下。

**split_v1 首次训练已完成（2026-09-06）：未通过，PPO 未启动。** 1 个训练初始化、60 个 SAC 预训练窗口，耗时 15.69 分钟；SAC Critic 更新 1571 次、Actor 更新 735 次，replay 32400 条。三场景各连续 900 s 测试的加权完成时延 **22028.39 ms**、完成率 **60.20%**、期末积压 **10136.07 Mbit**；固定均分只读对照为 **601.43 ms**。后者不是 SAC 成果。最后策略将低优任务集中到固定编号 UAV 1，其平均分配工作量 **20.26 Mbps**，实际计算约 **9.00 Mbps**，其他两架约 3.34 和 0.99 Mbps。当前是微观路由退化，不是 PPO 已经训练失败。结果、原始曲线与固定编号动作审计保存在 `results/split_mdp_mainline_seed1`，没有延长预算或挑选中间权重；v33 对照文档已同步更新到当前 67/68 维口径。

## 目录

| 目录 | 内容 |
|---|---|
| `hrl_uav/` | 物理环境、状态/奖励/动作接口、PPO/SAC 与训练循环 |
| `scripts/run.py` | 唯一训练入口 |
| `scripts/plot.py` | 读取新结果，输出中文 PNG |
| `configs/hrl_ppo_only_unlock.json` | 当前主线配置：SAC 配方不变，只开启 PPO |
| `configs/main.json` | 早期短检查配置，供其他诊断模式和历史复核使用 |
| `tests/` | 物理模型、路由、时间边界与真实更新测试 |
| `docs/` | 当前建模说明、清理清单与本轮结果 |
| `results/` | 只保存本轮新运行，不混入历史 checkpoint |

旧代码、论文包、结果和临时文件已按作者确认的外部备份前提删除。`.git` 和工具配置保留，没有提交或推送。清理路径清单在 `docs/cleanup_manifest.csv`。

## 运行

在项目根目录，用已安装 NumPy、PyTorch、SciPy 和 Matplotlib 的 Python：

```powershell
python -m unittest discover -s tests -v
python scripts/run.py --mode smoke
python scripts/run.py --mode mainline
python scripts/plot.py --result results/mainline_6mhz_routing_critic
```

本机 Python 为 `D:/Anaconda3/envs/pytorch-gpu/python.exe`。结果目录已存在时程序拒绝覆盖。新训练始终从零开始；预训练与联训之间保留本次运行自己的 SAC 权重、优化器及 replay。

### 开源 SAC 实现对照

项目 `.venv` 复用本机已安装的 PyTorch，但独立安装固定版本的 Stable-Baselines3 2.7.0 / Gymnasium 0.29.1，不修改原 Python 环境。

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe scripts/run.py --mode sac-check
.venv/Scripts/python.exe scripts/run.py --mode sac-compare
.venv/Scripts/python.exe scripts/plot.py --result results/sac_compare
```

两个新模式只做微观预训练的实现核验，不开放 PPO。对照协议、预算、实现差异及结果见 [SAC 实现对照](docs/sac_reference.md)。

`--method centroid` 可独立从零训练质心 + Direct SAC 对照；本轮不自动扩多种子或对比矩阵。

## 早期实验范围与读数

候选主场景：6 MHz U2U、每架 10 GHz、负载 [12.5,5.5,3.5] Mbps ×1.15 循环轮换，总期望负载 24.725 Mbps。本次仅恢复负载及轮换，奖励、动作和训练超参数不变；并非完整复现旧版训练配置。更不均衡负载和原 3 MHz 配置保留在旧实验 manifest 中，不覆盖旧结果。

```powershell
.venv/Scripts/python.exe scripts/run.py --mode scene-check
```

此命令不加载模型、不训练，只用固定均分路由比较 3/6 MHz 下的通信和计算积压。三场景各 900 s，保存逐链路资源账本到 `results/restored_load_resource_check`。不能将其结果称为 SAC 成绩或总体最优。新配置的训练尚未启动，需先完成旧版训练实现的回归核查。

检查已完成，三场景均值如下。两臂几何、实际路由与到达计数相同，只有 U2U 总带宽不同。

| 固定均分诊断 | 3 MHz | 6 MHz |
|---|---:|---:|
| 加权完成时延 / ms | 3203.65 | 601.43 |
| 低优完成时延 / ms | 16351.68 | 1135.97 |
| 时间平均上传积压 / Mbit | 2.98 | 2.98 |
| 时间平均转发积压 / Mbit | 158.56 | 0.91 |
| 时间平均计算积压 / Mbit | 26.24 | 9.64 |
| 期末总积压 / Mbit | 439.61 | 9.31 |
| 截止 900 s 完成率 | 97.281% | 99.948% |

3 MHz 下单条活跃链路平均分到 0.60 MHz、速率 3.90 Mbps；6 MHz 下分别为 1.53 MHz、9.94 Mbps。带宽增大后积压消退、同时活跃链路数减少，因此实际单链路增益不只是总带宽翻倍。上述证据表明通信限制了该固定路由的表现，不证明 3 MHz 下所有路由均不可行，也不证明 SAC 已经能达到 601 ms。该资源检查时 61 项单元测试通过，未启动训练；后续 split_v1 核验和训练状态见文首。

短 pilot：seed=1，60 个微观预训练窗口，资格检查通过才启动 60 个同步训练窗口。每窗口 300 s，每三个窗口连续运行 900 s，不在第 300/600 s 清空队列。使用本次运行自己的 Actor、Critic、优化器和 replay 衔接两阶段，不加载旧模型。微观未通过时只保存 `micro_pretrained.pt`，不生成冒充完整 HRL 的结果；联合完成才保存 `final.pt`。

`mainline` 每 6 个窗口做两场景各 900 s 的确定性验证。最后连续三次验证要求：每场景时延和 J 不比固定均分差超过 5%，完成率至少 99.5%，期末和时间平均积压不比均分多 10 Mbit，最后 300 s 队列斜率不超过 0.01 Mbit/s，三次时延及 J 的相对极差均不超过 5%。这是预先声明的工程放行标准，不是统计显著性检验或无限时域稳定性证明。均分只用于只读对照，不进入训练 replay、不提供教师标签、不接管 Actor 动作。

验证场景 33000/33001；最终只读测试场景 33500–33502；训练场景从 61000/71000 分别开始。测试不参与阶段放行。每阶段最多 20 分钟，超时只记为预算未完成，不追加轮次、不挑 best。`--mode mainline-check` 仅作短 smoke，显式跳过质量门槛验证两层代码更新，不构成训练成功。

同 final SAC 下强制质心仅为组件反事实，不冒充独立训练基线。成功运行、奖励曲线变平、没有丢任务，都不等于队列稳定或算法最优。须同时检查完成率、未完成数、队列增长、低优时延与固定种子评估。

模型定义和本次同时改变的建模因素见 `docs/model.md`。

### Critic 动作等价性对照

6 MHz 从零复测后，单独核验了“不同分数但相同执行路由”的 Q 值一致性。当前主配置显式设为 `critic_action_representation=executed_route`，将相同的可微单纯形投影放到双 Q 及 target Q 的输入端：`Q(s,z)=q(s,P(z))`。Actor 仍独立输出原 18 维分数，熵仍在分数空间计算，replay 仍保存真实分数及执行路由；没有教师、门控或额外奖励。`--critic-input raw_score` 保留原始输入消融；`sac-check/sac-compare` 固定两臂都使用原始分数输入，不混同两类审计。

```
.venv/Scripts/python.exe scripts/run.py --mode mainline --critic-input executed_route --output results/mainline_6mhz_routing_critic
```

对照已完成：3 个独立测试场景各连续 900 s，原输入加权时延 25814.39 ms，路由输入 10875.80 ms，固定均分 628.66 ms；两学习臂均不合格。两臂从零独立训练，初始参数、场景种子、预算、奖励和放行规则一致；旧分数 Critic 结果保留在 `results/mainline_6mhz`。默认保留的是物理等价性修正，不是“已经成功”的方法；本轮不继续追加其他超参数或架构搜索。
