# Claude / AI 阅读入口

这是为网页端代码阅读准备的精简交接快照，整理日期 2026-10-12。请先读本文，再读 [项目上下文](docs/PROJECT_CONTEXT.md)、[版本地图](docs/VERSION_MAP.md) 和 [历史报告索引](docs/history/INDEX.md)。不要从旧 README 推断当前状态。

## 用户的研究目标

研究 UAV 辅助移动边缘计算（UAV-MEC）的分层控制：上层 PPO 调整三架无人机的位置，下层 SAC 按任务优先级与源 UAV 动态拆分计算任务，联合降低优先级加权时延与物理能耗。需要可信的因果观测、容量约束、父任务完成语义、持续服务采样和公平的独立基线；最终应由配对、多场景、多种子评价判断策略效果，而不是靠某一条曲线或训练 loss。

## 必须区分的版本

- 根目录 `src/` 是 MAIN 的 v2.3.0 物理候选，源 commit `a9953dc36d9d65f99bcc35044cdfc250ca11bb49`。
- `research/speedup/src/` 是后续实验使用的 v2.4.0，源 commit `25d59ff9f7c3e27c060c63d462235a30ce28930a`。它加入终端 Gauss-Markov 移动、按每架 UAV 总预算分配 U2U 发射功率和存档优化。分析最新实验应优先读这份源码及 `v2_4_0_lite.json`。
- `research/experiments/outputs/scene_optimality_20260928/` 保留后续训练、评估与诊断驱动，以及选定实验合同。方案 C/D 和 curriculum 的行为还由这些驱动决定，不能只看核心 `run.py`。
- `docs/design_specification.md`、`docs/CONTINUING_SERVICE_VALIDATION.md` 和旧安装文档记录早期阶段。关于“固定终端”“train/resume 无条件禁用”“未训练”的文字是当时状态，不是后续实验的现状。具体物理参数以相应版本源码与配置为准。

## 阅读代码顺序

在选定的一份源码树内：`configs/physical_scene.json` 与 `configs/v2_4_0_lite.json`（或根目录 v2.3.0 配置）→ `hrl_uav/industrial_hotspot_env.py` / `industrial_env.py` → `hrl_v2/physics_adapter.py`、`timing.py` → `reports.py`、`lite_observation.py` → `collector.py`、`replay.py` → `hybrid_simplex.py`、`lite_networks.py`、`sac.py`、`ppo.py` → `metrics.py`、`evaluate.py`、`paper_evaluation.py`、`upper_baselines.py` → 实验合同与驱动。

## 不能混淆的语义

- 900 s 是采样窗口，不是真实任务截止；未完成任务记 censored，不等于失败。`terminated=False, truncated=True` 时应从真实 successor bootstrap。
- 下层动作形状 `2 × 3 × 3`，是 priority-class-level dynamic task splitting，不是为每个任务输出独立动作。精确零比例不创建 fragment；所有正比例 fragment 完成才算父任务完成。
- 上层观测在提交下一移动之前；下层观测在相应移动已提交及已发生到达准备之后。策略不能读未来到达、隐藏业务状态、RNG 或 rollout 倒计时。
- 物理 reward、学习用折扣 reward、截断时延、完成任务 p95 和 unfinished 是不同指标。比较需使用相同场景、种子、版本与统计口径。
- dev 选档与 test 必须分离；报告中的 G1 未通过不能改写成通过。最新报告包含 Proposed 不优于 TFOO 的负结果。
- 历史报告是运行时证据；本文未重跑训练或验证其全部数值。方案 D 的 10 月 11 日记录是运行中快照，不能推断 10 月 12 日已完成。

## 本快照边界

不包含模型权重、经验池、逐窗口轨迹、JSONL 训练日志、大型 profiling 原始数据、虚拟环境或旧 Git 历史。保留报告正文、图表、源码、配置与选定验证/合同记录。仅凭本快照可以理解和审查代码，不能直接续训或完整复现历史数值。实验驱动保留原路径和保护检查，有外部输入依赖；迁移机器时必须先审查路径和版本绑定，不能直接运行。

请回答时标明所依据的代码版本，区分源码事实、报告结论和推断；涉及优化建议时先指出证据及缺少的计时/验证项。
