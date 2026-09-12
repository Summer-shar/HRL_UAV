# Mission V1 第一阶段：固定10次900秒无学习开发pilot

## 授权与物理版本

物理提交固定为 `19976306281cf213f1e7f6973b999f2dcd39d593`，分支 `codex/mission_rpgm_v1`。
本轮只修复runner复现性元数据、新增观测日志/分析/测试，不修改env.py、mdp.py或物理配置。
该物理提交同时包含论文材料，不称为纯mission提交。

`mission_provenance.py` 从固定提交读取env/mdp、配置、动作映射、负载、学习、统计等16个核心文件。
比对源码时只统一Git CRLF/LF换行，同时记录原始工作树SHA256；不忽略空白或代码差异。
还比较原 `MissionPolicy`、`MissionMeter`、`mission_config` 的AST，防止借元数据修复改变控制定义。
后续提交可增加runner/test/analyzer，正式运行要求本轮依赖源码已提交且与实际HEAD一致。
不再要求HEAD等于旧smoke的修改前提交92d3214。

`configs/mission_rpgm_v1_smoke.json` 保持字节不变，其中base_commit仍表示历史smoke开始时的HEAD；
新manifest另记physics_base_commit和actual_head，不篡改历史smoke成绩或来源。

## 固定矩阵

仅33600、33601，每个场景依次执行Hover、Current centroid、Progressive 0.5、Predictive、Blind Patrol。
每条连续900秒，共10条；300/600秒不重置。没有按结果停止、改规则或追加微观比较。
保留集33700–33709以及33610–33612均不运行。

全部使用旧SAC101 window2000，确定性冻结零样本迁移：
`results/sac_only_wireless6_long_seed101/checkpoints/window2000.pt`，
SHA256 `be946d3a576918460690a16f1a57f2c6dc33dde13db86d360db5690858a76932`。
恢复时核对原manifest、完整配置、端点及SAC-only阶段，不建立PPO网络或经验池。
物理参数、20m位移、三路线、900秒迁移时长、连续服务、准稳态P(v)、负推进增量全部不变。

## 观测而不改控制

新增 `ObservedPolicy` 调用原 `MissionPolicy` 产生动作，仅在调用前计算未裁剪请求用于日志，
并断言其裁剪值与原动作完全一致。Blind Patrol仍只读宏观调用次数，不访问业务或位置。
没有用观察日志重新选动作，没有调用未来路线位置。

新增 `MissionPilotMeter` 继承已验收统计器；原物理步完全复用。
记录到业务质心与任务群中心的两种距离（0.1秒含初始点，逐机及三机等权的mean/median/p90/max）。
宏观日志区分未裁剪请求、20m裁剪、地图裁剪、安全投影与实际位移，分别记录计数/比例。
请求长度大于20+1e-9米计cap触发，动作圆盘边界另计饱和；恰好20米不误报为被裁剪。
分母为90个macro或270个UAV行，另给受影响UAV数及最小路径间距。

共同随机过程核对逐任务出生、逐物理步用户/群中心/业务质心、负载调度，
并记录每资源帧环境RNG完整状态及因果运动速度状态的累计指纹。
UAV运动、实际信道、队列、路由、完成结果不要求一致。

J为原物理步代价均值；驻留与能量分项闭合，完整平均功率含完整推进能量。
时延仅为完成父任务统计，必须并列高低优到达/完成/未完成数及完成率。
平均积压为0.5秒帧末均值，期末积压单列；后段斜率为600<t<=900秒的总队列OLS斜率。
U2G有效速率为实际工作量除以累计用户发送时间，U2U实际活跃速率复用原活跃链路-步统计。
发送时间是并发发送方时间之和，不是墙钟时延；吞吐按900秒。
原环境info将U2G发送时间转为float32，时长上界核验只放行一个float32 ULP的舍入误差，
不裁剪记录或修改实际物理服务。

## 测试范围与运行命令

仅选择 `tests.test_mission_rpgm_v1` 和 `tests.test_mission_pilot`。
前者七项已核查为固定规则/无网络替身；后者禁止autograd.backward/grad，
只有一个20秒真实冻结SAC推理测试，其他为静态元数据、人工动作或固定规则短测试。
不运行已有 `test_split_observation` 训练式测试。

```powershell
& 'D:\Anaconda3\envs\pytorch-gpu\python.exe' -m unittest tests.test_mission_rpgm_v1 tests.test_mission_pilot -v
# 通过并提交后，才运行下列固定矩阵；不可用于追加评价。
& 'D:\Anaconda3\envs\pytorch-gpu\python.exe' scripts/run_mission_rpgm_v1_pilot.py
```

测试覆盖固定物理指纹、旧模型字节回归、连续运动、20帧服务、任务流水滞后、共享步首几何、
带符号能量账、位移安全、CRN、冻结SAC、额外场景/规则/时长拒绝、观测日志不改输出。
实际执行上述定向命令：首轮13项中11通过、2失败，均为新增U2G发送时间上界断言未容纳
原float32导出值的舍入（0.1秒不能被float32精确表示）。仅将新统计器容差改为一个float32 ULP，
未修改物理服务或裁剪数据；复测13项全部通过，用时9.258秒。没有反向传播或优化器更新。
物理核心相对1997630的diff为空。独立60秒smoke未重跑，历史结果未覆盖。

## 输出与解释

新目录 `results/mission_rpgm_v1_pilot/` 必须不存在，已有目录不覆盖、不自动恢复或重跑。
逐场景目录保存queues、commands、uplink、links、uavs、routes、macro_metrics、
CRN、冻结核验、0.1秒轨迹NPZ与追踪误差，保留全部正负结果。
根目录保存manifest、per_seed、summary、paired_deltas、paired_means、trajectory_summary、result。
每场景四策略统一范围轨迹图与单列Blind诊断图，PNG 300dpi和原生矢量PDF；每100秒标记UAV位置。

Blind风险保持预声明：delta_J<0、delta_holding>=0、delta_energy<0，使用1e-10数值容差。
完整10条后统一报告，不为该风险更改模型或提前终止某臂。
分别比较Tracking-Hover、Progressive-Current、Predictive-Current、Predictive-Progressive，
报告逐场景绝对差与相对差；两个开发场景不作为独立训练重复或显著性证据。

没有新增硬QoS、奖励、加减速/转向代价或轨迹约束。
是否有足够线索进入第二阶段微观比较须在本轮完成后由用户裁决，不能自动运行。
