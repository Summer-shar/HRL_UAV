# lite_v2_3_0 交付记录

基于干净提交 `4f8198ac1e0ce391b2860cfbbfa896f5412a73ce`，分支 `codex/v2-lite-paper-caliber`。仅本轮物理口径、脚本基线、测试与证据变更；未 push，未训练、未恢复 seed 4101，未修改任何既有 checkpoint 或旧输出。

## 物理合同与实现

U2U 总带宽 3 MHz，由当前有积压的有向链路等分；每条激活链路独立使用 1 W、固定噪声 1e-13 W。服务、报告、CVX 报告速率及 Greedy 候选速率均来自 `UAV_MEC_Env.allocate_u2u_resources`。旧 PSD×带宽、源功率均分分支保留，并做旧配置数值逐位回归。上行公式没有改变。

最大速度 20 m/s、最大单周期目标位移 20 m，调整段仍为 1 s、宏周期仍为 5 s。V2 实际调用 `IndustrialDeploymentEnv.start_motion`；没有调用旧基类 `IndustrialEnv.start_motion` 或 `_reference/physics_adapter.py`。速度特征仍除以场景中的 `max_speed_mps`。

每个宏周期 tick 0–9 计原始推进能耗，tick 10–49 不计；是否提交运动命令不影响此时间窗，悬停同样收费。CPU、上传、转发和队列服务顺序保持原样。

`cost = 0.9 * holding / 10 + 0.1 * (E_upload + E_cpu + E_u2u + 0.005 * E_propulsion) / 14`。所有 `energy_j` 仍是原始焦耳；`objective_energy_j` 单列加权目标能量。每窗口同时独立校验父任务 holding 和能量—成本恒等式。CVX 是下层路由代理，其推进项与路由无关，省略该常数不会改变 LP 最优路由。

论文出处：[Xu et al., Trajectory Planning and Resource Allocation for Multi-UAV Cooperative Computation, IEEE TCOM 72(7), 4305–4318 (2024)](https://doi.org/10.1109/TCOMM.2024.3361536)。按用户给定式 (13)、式 (5)–(6)、式 (20) 口径落地；等分带宽、每链路满功率及避免重复计 K 次推进权重的解释均写入 `parameter_sources.paper_11`，不宣称这些分配规则是论文明确规定的。

## 基线与配置

`hover_greedy` 对每个 (priority, source) 单独比较当前 CPU/在途 cycles 和传输 bits。假设候选链路激活，并保留其他当前激活链路，通过公共分配器求速率；高优只看高优 CPU/流入量，低优看全部。每行 one-hot，平局选最小目的编号，不读取未来到达，不将不同候选行的假设分配写回报告。

质心与诊断脚本的运动步长均从 `physical_scene.json` 的 deployment 经 Policy / baseline_reference 或 Script 构造参数传入。10 s 历史窗、20 帧、5 s 预测、质心权重不变；20 m 配置下距质心 8 m 实际走 8 m、距质心 30 m 实际走 20 m，10 m 配置下与归档函数输出逐位一致。

新 Lite 配置为 `src/hrl_v2/configs/v2_3_0_lite.json`，图路径对应配置为 `v2_3_0.json`；默认图入口转向后者以匹配新物理成本。旧 v2_1_0、v2_2_1_lite 文件保留未改。观测版本仍为 `lite_110_729_v2_2_1`，因为本轮没有修改观测或网络结构。场景 schema 为 `industrial_hotspot_scene_v5`；布局 realization schema 不变，因为布局生成规则未改。

旧配置在新物理场景下被 read_config 明确拒绝；真实 ep300 checkpoint 在 source/config/scene identity 检查时即被拒绝，未反序列化。见 `old_checkpoint_rejection.json`。未改 G1；`--include-greedy` 仅附加离线比较，默认 G1 四组参考与通过门槛保持原样。

## 测试

- 修改前：核心 137 项，136 通过、1 项原条件跳过；入口 17 项全部通过。
- 最终版本：核心 149 项，148 通过、1 项同条件跳过；入口 17 项全部通过。
- 跳过项为已弃用 critic 64/128 microbatch 候选的可选数值审计，本轮没有改变 microbatch。
- 当前环境未安装 pytest，未安装或升级依赖；执行仓库原有 unittest 验证入口及根目录 unittest discovery。完整测试内有临时模型的优化器单元测试，没有科研训练或旧模型学习更新。
- 日志：`old_validation/tests.txt`、`old_entry_tests.log`、`validation_final/tests.txt`、`entry_tests.log`；`validation_final/validation.json` status=passed 且 identity 与本次基线相同。

## 数值验收

单链路 0→1：20.426356372992 Mbps；全六链路约 [[0,3.404,3.245],[3.404,0,3.245],[3.245,3.245,0]] Mbps，每链路 1 W、1e-13 W 噪声。

参考函数完整带宽 512820.5128205128 Hz × 1.95e-19 W/Hz = 1e-13 W，最大谱效率误差 0。题面四舍五入带宽 512820.5 Hz 的乘积为 9.999999749999999e-14 W，与 1e-13 的相对误差仅 2.5e-8。

p(20)、p(10)、p(0) = 205.227453672, 152.493485164, 194.763445001 W；(20,10,0) m 示例周期能量 552.484383837 J。900 s 悬停推进能量 105172.260300743 J（按题面精度为 105172.3 J）。

验收 3：20 个 dev 场景、两条 policy stream 均通过。holding 及 upload/cpu/u2u 原始能耗与指定独立参考逐场景完全一致。场景算术平均截断时延 **0.599426642378 s**，四舍五入为 **0.5994 s**。

### 验收 3 逐场景对照

| 场景 | 参考 holding task·s | 新版 holding task·s | 相对误差 | 截断时延 s |
|---:|---:|---:|---:|---:|
| 1 | 40334.900000000 | 40334.900000000 | 0 | 0.581554854 |
| 2 | 45006.500000000 | 45006.500000000 | 0 | 0.640279121 |
| 3 | 44922.700000000 | 44922.700000000 | 0 | 0.633686928 |
| 4 | 43715.200000000 | 43715.200000000 | 0 | 0.624449333 |
| 5 | 38837.300000000 | 38837.300000000 | 0 | 0.569829509 |
| 6 | 41223.300000000 | 41223.300000000 | 0 | 0.588778119 |
| 7 | 41083.100000000 | 41083.100000000 | 0 | 0.577326063 |
| 8 | 47155.000000000 | 47155.000000000 | 0 | 0.660952568 |
| 9 | 46522.400000000 | 46522.400000000 | 0 | 0.640143103 |
| 10 | 41047.100000000 | 41047.100000000 | 0 | 0.576358505 |
| 11 | 41751.500000000 | 41751.500000000 | 0 | 0.599464450 |
| 12 | 45763.600000000 | 45763.600000000 | 0 | 0.644231094 |
| 13 | 45216.900000000 | 45216.900000000 | 0 | 0.626472422 |
| 14 | 43160.700000000 | 43160.700000000 | 0 | 0.610865473 |
| 15 | 38698.800000000 | 38698.800000000 | 0 | 0.558029676 |
| 16 | 38434.600000000 | 38434.600000000 | 0 | 0.555943530 |
| 17 | 39447.900000000 | 39447.900000000 | 0 | 0.566934939 |
| 18 | 40504.600000000 | 40504.600000000 | 0 | 0.593935217 |
| 19 | 38427.300000000 | 38427.300000000 | 0 | 0.549165404 |
| 20 | 41230.200000000 | 41230.200000000 | 0 | 0.590132539 |

完整逐场景能耗对照见 `acceptance_3.csv` / `.json`，参考文件 SHA256 为 `7d9250a694c4f6f7203be174309c876a7bc4f0d524e11a8b986b52f157d494f0`。

## 五组基线均值

每项均为 20 个场景的 900 s 窗口；两个 policy stream 各独立执行一次。主列为逐场景算术平均截断时延，另外列出原格式的 pooled 时延，避免不同分母混用。各项能耗均为未加权原始 J，cost 为每窗口原始物理成本。

| 基线 | 场景均值时延 s | pooled 时延 s | 上传 J | CPU J | U2U J | 推进 J | cost/900s | cost/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hover_local | 5.446247943 | 5.460338627 | 541.300031 | 204748.206825 | 0.000000 | 105172.260301 | 35986.515573 | 39.985017303 |
| hover_equal | 0.599426642 | 0.599746856 | 541.300031 | 198781.167832 | 2893.028861 | 105172.260301 | 5239.328757 | 5.821476397 |
| hover_cvx | 0.645202457 | 0.645519928 | 541.300031 | 205825.136566 | 1485.121508 | 105172.260301 | 5568.931389 | 6.187701543 |
| centroid_equal | 0.587915629 | 0.588200952 | 507.655347 | 199115.478371 | 2980.718954 | 93221.777304 | 5168.690875 | 5.742989862 |
| hover_greedy | 0.645850754 | 0.646152860 | 541.300031 | 216896.766425 | 1178.804451 | 105172.260301 | 5649.827430 | 6.277586033 |

完成 200 个窗口、1,800,000 个物理步，墙钟 771.3 s。CVX 40 个窗口共 72000 次路由调用，fallback 0 次。五组均调用现有 collector/metrics，全部原始窗口 JSON 留在 `baselines/`；总汇为 `baselines/summary.json`。

## 修改文件及理由

| 文件 | 修改理由 |
|---|---|
| `src/hrl_uav/env.py` | 公共 U2U 分配器新增固定噪声、逐链路满功率模式，同时保留旧 PSD/源功率均分分支。 |
| `src/hrl_uav/industrial_baselines.py` | 新增仅使用当前报告的 GreedyRouting，并说明 CVX 路由代理省略与路由无关的推进常数。 |
| `src/hrl_uav/industrial_env.py` | 读取 U2U 模式；按部署时间窗计原始推进焦耳，成本中只乘一次推进权重。 |
| `src/hrl_uav/industrial_hotspot_scene.py` | 新增 v5 场景 schema，兼容读取 v4 的旧模式配置。 |
| `src/hrl_uav/industrial_scene.py` | 校验新增 U2U 模式、固定噪声与部署段推进能耗定义。 |
| `src/hrl_v2/baseline_reference.py` | 复用单窗口执行函数；显式附加 hover_greedy，不改变 G1 的默认四组参考或门槛。 |
| `src/hrl_v2/baselines.py` | 质心步长由配置传入；GreedyRouting 经现有报告转换接口接入。 |
| `src/hrl_v2/configs/physical_scene.json` | 落地本轮列出的 U2U、20 m/s、20 m、部署段推进和 0.005 权重，登记论文口径及分配假设。 |
| `src/hrl_v2/configs/v2_3_0.json` | 增加与新物理成本绑定的图路径配置副本，算法数值保持旧版。 |
| `src/hrl_v2/configs/v2_3_0_lite.json` | 新增 Lite 版本配置，仅改变状态说明和明确的推进能耗定义/权重。 |
| `src/hrl_v2/control_diagnostics.py` | 诊断脚本的三处 10 m 步长改为从部署配置读取。 |
| `src/hrl_v2/control_mechanisms.py` | 独立复算成本时按原始推进焦耳乘一次 0.005。 |
| `src/hrl_v2/evaluate.py` | 向 Policy 传入场景部署配置，使质心评估使用当前步长。 |
| `src/hrl_v2/metrics.py` | 保持原始能耗统计，单列加权目标能量并校验其与物理成本一致。 |
| `src/hrl_v2/paths.py` | 默认图路径入口使用新物理口径的对应配置；旧配置文件保留。 |
| `src/hrl_v2/physics_adapter.py` | 明确校验旧 full 与新部署段加权推进两种目标合同。 |
| `src/hrl_v2/policy.py` | 接收部署配置并传入质心控制，不写死 20 m。 |
| `src/hrl_v2/run.py` | 同步固定场景 SHA，并拒绝能耗模式/推进权重与物理场景不匹配的配置。 |
| `src/hrl_v2/tests/test_lite.py` | 新 Lite 配置继续校验图网络和算法保护范围，只豁免已批准修改的物理文件。 |
| `src/hrl_v2/tests/test_physical_contract.py` | 容量、成本与运动边界断言匹配新物理合同。 |
| `src/hrl_v2/tests/test_reference_continuation.py` | 独立窗口成本复算加入推进权重，保留同轨迹连续服务验证。 |
| `src/hrl_v2/tests/test_reference_parents.py` | 实际工作能耗单测按新推进权重复算成本。 |
| `src/hrl_v2/tests/test_reference_physics.py` | 多链路容量及位置—速率的独立解析答案采用新噪声/功率。 |
| `src/hrl_v2/tests/test_paper_caliber.py` | 新增 12 项数值、因果、步长兼容、配置保护、身份拒绝及实际调用路径验收。 |
| `tools/run_paper_caliber.py` | 只运行固定脚本基线，保留原文件格式和种子，生成逐场景独立参考对照与均值。 |

证据文件全部位于 `outputs/lite_v2_3_0/`，不改变已存在的其它输出目录。提交只纳入上述任务文件与精简证据；完整窗口 JSON 留在本地，不把大型仿真输出或旧 checkpoint 塞入 Git。

## 复现（只测量，不训练）

```powershell
.venv/Scripts/python.exe -m hrl_v2.validate --config src/hrl_v2/configs/v2_3_0_lite.json --output outputs/lite_v2_3_0/validation_final --skip-benchmark
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe -u tools/run_paper_caliber.py --workers 6
```

基线工具遇到已有窗口时先验证完整 identity，匹配才复用；不覆盖不匹配的缓存。旧版本应检出旧 commit 复现。审核前不开始任何训练。
