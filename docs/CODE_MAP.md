# 代码地图

三个包都在 `src/` 下：`hrl_uav` 是仿真器，`hrl_v2` 是强化学习组件，`p2` 是论文实验流水线。`hrl_uav` 和 `hrl_v2` 沿用原仓库的包名，已训练的权重和主实验记录因此可以直接读取。

## 推荐阅读顺序

1. **配置**：`src/hrl_v2/configs/physical_scene.json`（场景、业务模板、资源、移动）→ `v2_4_0_lite.json`（时间尺度、奖励、网络与算法超参数）。
2. **场景生成**：`hrl_uav/industrial_scene.py`（任务类型、到达过程、校验）→ `industrial_hotspot_scene.py`（簇状布局、业务模板置换）。
3. **物理过程**：`hrl_uav/industrial_env.py`（`IndustrialPhysics` 链路速率与 U2U 资源分配，`GaussMarkovMobility` 终端移动，`IndustrialEnv` 任务拆分、上传 / U2U / CPU 排队、父任务完成语义、能耗）→ `industrial_hotspot_env.py`、`industrial_deployment_env.py`（上层位移的“调整后悬停”）。
4. **控制接口**：`hrl_v2/physics_adapter.py`（`PhysicsAdapter`：上层提交位移、下层逐 0.5 s 决策、0.1 s 物理步）→ `timing.py`（时钟）→ `reports.py`（因果报告，策略只能读它）。
5. **观测**：`hrl_v2/lite_observation.py`（上层 3K×17+66 维、下层 110 维的固定向量）。
6. **采样与经验**：`hrl_v2/collector.py`（连续服务窗口，900 s 截断时从真实后继状态 bootstrap）→ `replay.py`。
7. **算法**：`hrl_v2/hybrid_simplex.py`（混合支撑集卸载分布，可精确取零）→ `lite_networks.py` → `sac.py`、`ppo.py` → `runtime.py`（`Learner`：两层合在一起，按配置选 Lite 网络）→ `upper_baselines.py`（只有 PPO 上层的基线）。
8. **基线的下层规则**：`hrl_v2/baselines.py`（均分、本地）、`hrl_uav/xu_cvx_r.py`（TFOO 的 CVX 线性规划）。
9. **指标**：`hrl_v2/metrics.py`（加权截断时延、能耗、代价、未完成任务的截断计入）。
10. **流水线**：`src/p2/common.py` → `learn.py` → `evaluate.py` → `report.py` / `plots.py` → `pipeline.py` → `retrain.py` / `checks.py`。

## 仿真器 `src/hrl_uav/`

| 文件 | 内容 |
|---|---|
| `industrial_scene.py` | 任务类型、终端与到达过程、配置校验、期望负载 |
| `industrial_hotspot_scene.py` | 每区 K 个终端的簇状布局、业务模板与置换 |
| `industrial_env.py` | 物理与服务核心：链路、U2U 带宽与功率分配、终端移动、拆分 / 排队 / 完成、能耗 |
| `industrial_hotspot_env.py` | 场景重置适配 |
| `industrial_deployment_*.py` | 上层位移的调整与悬停、部署报告与观测契约 |
| `industrial_observation.py`、`mdp.py`、`routing.py` | 旧版报告编码与路由（被上面的模块引用） |
| `industrial_baselines.py`、`xu_cvx_r.py` | 下层基线规则，TFOO 的 CVX（CLARABEL）求解 |
| `env.py`、`geometry.py`、`load_switching.py` | 通用无线模型、坐标与负载切换工具 |

## 强化学习组件 `src/hrl_v2/`

| 文件 | 内容 |
|---|---|
| `physics_adapter.py`、`timing.py`、`reports.py` | 仿真接口、时钟、因果报告 |
| `lite_observation.py`、`lite_networks.py`、`lite_batch.py` | 固定向量观测与网络（论文使用的 Lite 版本，配置 `v2_4_0_lite.json`） |
| `features.py`、`networks.py`、`batching.py`、`prepared.py` | 图注意力版本的观测与网络（`runtime.py` 按配置选择，论文未使用） |
| `hybrid_simplex.py`、`design/reference_hybrid_simplex.py` | 混合支撑集分布及其数学参考实现 |
| `sac.py`、`ppo.py`、`runtime.py`、`upper_baselines.py` | 两层算法、`Learner`、PPO 基线 |
| `collector.py`、`replay.py`、`randomness.py`、`policy.py` | 采样、经验池、命名随机流、执行策略 |
| `metrics.py` | 物理代价与截断任务 |
| `baselines.py`、`baseline_protocol.py` | 下层固定规则、基线标签 |
| `checkpoint.py`、`paths.py`、`schedule.py` | 快照、路径、训练日程 |

## 实验流水线 `src/p2/`

| 文件 | 内容 |
|---|---|
| `common.py` | 实验设计常量（种子、算法、训练量、扫参点、试点）、随机数派生、场景缩放（CPU、负载、U2U、终端数：`users` 总负载固定 / `terminals` 每终端负载固定）、汇总（不需要 torch） |
| `learn.py` | M1 / M2 / M3 / HoverSAC / PPO 基线的训练与权重读写 |
| `evaluate.py` | 900 s 闭环评估、dev 选档、测试、固定策略扫参、突发实验、轨迹记录 |
| `report.py`、`plots.py` | 表格、配对检验、图 |
| `pipeline.py` | 带依赖的多进程任务调度（可续跑），主实验与扫参的任务图 |
| `retrain.py` | 扫参点目录的建立、完成判断、记录导出 |
| `checks.py` | 耗时估计（`timing`）、试点检查（`pilot-check`）、进度（`progress`） |
| `feasibility.py` | 两个运行阶段各一条命令：`feasibility`（检查、扫参冒烟、两个试点，`FEASIBILITY.md`）、`sweep`（全部点，`SWEEP.md`）与零样本扫参 `zeroshot`（`ZEROSHOT.md`） |
| `consistency.py` | 零样本扫参的逐位一致性检查（`zeroshot-check`）：标称点对主实验记录，新代码重算已有点对原记录 |
| `publish.py` | 把状态页、日志末尾和汇总表推送到 GitHub 分支 `paper2-runs` |
| `perf.py` | 逐位等价的仿真加速 |
| `burst.py` | 受控突发业务 |
| `positioning.py` | 轨迹的定位分析（论文未使用） |
| `revise.py` | 从已完成目录派生新微调配方目录（主实验结果的来历，见 `HISTORY.md`） |

`run.py` 是唯一入口，每个流水线任务都是一次 `python run.py <命令>` 子进程。

## 不能混淆的语义

- 900 s 是采样窗口，不是任务截止时间。窗口结束时未完成的任务记为截断（censored），不等于失败；此时 `terminated=False, truncated=True`，从真实后继状态 bootstrap。
- 下层动作形状为 2 × 3 × 3（优先级 × 源无人机 × 目的无人机），是按优先级类别的动态任务拆分，不是为每个任务单独输出动作。比例精确为零时不产生分片；所有正比例分片完成，父任务才算完成。
- 上层观测在提交下一次位移之前；下层观测在位移已提交、本时隙到达已就绪之后。策略读不到未来到达、隐藏业务状态、随机数或剩余时间。
- 物理代价、学习用的折扣奖励、截断时延、完成任务 p95、未完成率是不同指标。比较时必须用相同场景、种子、代码版本和统计口径。
- dev 选档（场景 1–8）与测试（场景 21–40）分开。
