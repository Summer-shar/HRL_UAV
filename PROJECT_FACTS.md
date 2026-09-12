# PROJECT_FACTS

## 2026-09-09 追加核验：候选模式 rollout 集成测试

- 【基准与范围】本轮 HEAD 为 `95633816bdbc7e7a48718259b8c14e3b7b5886fb`，相关文件修改前无差异。只新增一项测试和记录，未修改生产代码、配置、论文或历史结果，未提交或推送。
- 【运行核实】`tests/test_deployment_compute.py::DeploymentComputeTests.test_rollout_completes_deployment_boundaries_without_tail_metric_leak` 通过。显式开启候选模式，真实 `rollout(..., train=True, complete_tail=True)` 使用固定动作替身、内存 store 和无学习更新替身；不实例化网络、不反向传播、不加载权重。`train=True` 只用于走真实经验写入分支，不是训练实验。
- 【断言范围】人工父任务分别在 10.1 s 和 20.2 s 完成；36 条服务经验包含两条恰好补齐一次的边界经验，后继为 11/21 s 的实际队列与未完成任务数，回报含两个部署帧、折扣为 gamma^3。尾部环境推进至 21 s，窗口行仍对应 10/20 s，summary 保持 20 s 的完成、时延、代价和功率统计；部署期不增加路由决策或学习更新。
- 【实际测试】新增单测首次运行 1 项通过（0.137 s）；随后只运行部署专项文件，11 项全部通过（0.936 s，不含解释器导入）。这是短人工集成检查，不是性能、稳定性或收敛证据；详情见 `docs/deployment_compute_mode.md` 的追加集成测试记录。

## 2026-09-09 候选扩展补充：部署期 CPU 服务

- 【源码静态核实】本轮基准 HEAD 为 `4850092f4d198e53c4826754c12a232cee5df569`；修改前 `hrl_uav/`、配置、脚本、测试及本事实文件相对 HEAD 无未提交差异。下文较早的 HEAD、文件指纹和原模型成绩保留为历史快照，不代表扩展后的工作树。
- 【已实现，默认未启用】新增 `environment.compute_during_deployment`，默认 `false`；缺失选项的旧配置保持部署全停服。开启时仅服务已进入 UAV 计算队列的分片，上传、U2U 和路由仍暂停；要求 `pause_service_during_flight=true`。
- 【源码静态核实】只复用原 CPU 分配、高低优服务、能耗与父任务完成逻辑；奖励定义、观测维数、动作、通信参数、时间步、到达、移动和训练时钟不变。计算能量通过原 `total_energy` 进入奖励和功率账本。
- 【统计接口】候选部署步保留 `service_paused=true`，另报 `compute_service_paused=false`。统计器新增 `compute_steps`，避免丢弃部署期 CPU 服务量；CPU 利用率分母为 CPU 可服务步，通信分母不变。
- 【运行核实：小场景】缺省模式与上述基准源码逐步对照 200 个物理步（20 s），原始状态、全部返回统计与宏观输出完全一致；不是新性能结果。人工 0.6 Mbit 高优＋0.8 Mbit 低优算例，两步服务量分别为 1.0、0.4 Mbit，计算能量分别为 10、0.64 J，完成时间分别为 0.1、0.2 s。
- 【单元测试运行核实】25 项定向测试通过，含 10 项新增部署 CPU 测试；覆盖缺省/显式关闭兼容、通信不推进、CPU 优先级与上限、完成统计、服务守恒、代价/能量累计、时钟及边界折扣。未运行全量测试或性能评估。
- 【范围】未改任何现有配置 JSON、论文或历史结果，未运行训练/900 s 评估、未加载旧权重。实现与测试阶段未提交或推送；本次本地提交由作者后续单独授权，不包含推送。本扩展不证明性能改善；详细开关和测试边界见 `docs/deployment_compute_mode.md`。

## 版本信息

- 整理日期：2026-09-09；本次证据读取截至本地约10:57。根目录：C:/Users/PC/Desktop/HRL_UAV。
- 作用：记录当前源码、默认入口、指定已完成运行与稿件副本的关系，不代表用户已把最近实验选为投稿主结果。以后按受影响部分更新，不从旧聊天补齐。
- 【用户明确确认】小层主方法简称SAC，不加Direct等前缀；两层使用PPO与SAC。中文统一动作网络、评价网络、双评价网络、目标评价网络。名称简化不改变动作分数、投影、执行路由、评价网络输入，原代码标识符和历史归档字符串不修改。
- 【源码静态核实】git HEAD：be8556ee911c5541deeb19fde76add0e719d4601。限定相关文件的diff显示，至少hrl_uav/agents/ppo_agent.py和hrl_uav/agents/sac_agent.py存在未提交修改；因此HEAD不单独代表当前工作树。未做全仓库状态扫描，另记文末文件SHA256。
- 【源码＋运行记录核实】本次核对的14个源码/默认配置文件，其SHA256均与R2.source_hashes一致；不代表整个仓库或全部依赖已经审计。
- 【源码静态核实】通用入口scripts/run.py；显式mainline模式调用qualification.run_mainline，默认C0；显式--config可替代。未指定模式时是smoke，不是mainline。
- 【运行记录＋源码核实】最近指定续训命令为：python scripts/run_convergence_watch.py --source results/sac_tqc_continue1600to1800_seed1/sac/final.pt --output results/sac_continue1800to2000_seed1。这里只记录命令，本轮没有执行。
- 【运行记录核实】R0状态COMPLETE、current_global_window=2000，completed_at=2026-09-09T00:44:37.338951+00:00；R3为JOINT_PILOT_COMPLETE_NOT_FORMAL。累计200预训练窗口＋1800联合窗口，本次只增加最后200个联合窗口。
- 【有效配置核实】续训从来源checkpoint同目录manifest.config继承配置，再增加joint_windows；C1与R2.config逐项一致，不是直接使用C0。本轮未加载或重新哈希大型checkpoint，其来源哈希仅据归档记录。
- 【待确认】最终投稿采用哪个结果端点、哪份Overleaf工程，作者未在本轮指定；本文件不自行冻结投稿主版本。

### 资料定位

相对路径以仓库根目录为起点。

- C0：configs/hrl_ppo_only_unlock.json，通用mainline默认配置。
- C1：results/sac_continue1800to2000_seed1/config.json，指定2000窗口续训的配置快照。
- R0：results/sac_continue1800to2000_seed1/status.json。
- R1：results/sac_continue1800to2000_seed1/protocol.json。
- R2：results/sac_continue1800to2000_seed1/at2000/manifest.json，含配置、观测切片、源哈希和恢复来源。
- R3：results/sac_continue1800to2000_seed1/at2000/report.json，含结果、逐场景数据和更新计数。
- R4：results/sac_continue1800to2000_seed1/at2000/resume_audit.json，恢复检查及优化器状态记录。
- R5：results/sac_continue1800to2000_seed1/result.md，简短结果摘要，以R3数值为准。
- D1：README.md，只作为入口与历史差异定位；其中“尚无结果”不是最新状态。
- D2：docs/current_objective_contract.md、docs/current_objective_audit1200_result.md，明确限定为1200窗口的目标/诊断说明，用于核对指标口径，不当作2000端点成绩。
- D3：docs/sac1800_macro_action_readback.md，明确限定为1800窗口的动作分析，不外推到2000策略行为。
- D4：docs/sac_continuation_2000.md，最近续训协议及来源。
- P0：paper_current_20260907/overleaf/main.tex及sections/01_introduction.tex；本轮也定向定位sections/04_algorithm.tex。仓库引言与本聊天最新文本不同。
- P1：C:/Users/PC/Desktop/论文交付/前文章节增量_20260908/sections/03_problem_mdp.tex、04_algorithm.tex、05_evaluation_protocol.tex。
- P2：C:/Users/PC/Desktop/论文交付/引言增量_20260909/sections/01_introduction.tex，现有本地交付副本，仍不是后来聊天中的普通编号、SAC名称版本。
- P3：C:/Users/PC/Desktop/论文交付/修订稿_20260907_文献排版/overleaf/communication-journal.sty，仅定位引用及缩进定义，不代表网页端已经同步。
- 【用户明确确认】题目“基于分层强化学习的多无人机轨迹规划与协作卸载算法”；普通编号贡献首行缩进两个汉字。当前论文唯一主文件路径待确认，本轮不回改上述副本。

## 证据等级

- 用户明确确认：作者明确提出的名称、目标或约束；写作要求不等于实现证据。
- 源码静态核实：读取实际调用路径、函数及相关参数，或静态推算维数；不是本轮运行测试。
- 有效配置核实：入口覆盖关系与指定运行的config/manifest交叉核对，不只看默认值。
- 运行日志核实：读取已有机器生成status、manifest、report或audit；未重新运行、未独立重放checkpoint。
- 论文当前记载：所读稿件的表述，不能反过来证明实现或性能。
- 待确认：证据不足、版本冲突或需作者选择，不自行填补。
- 计划修改，尚未实现：建议或待同步内容，不写成已实现、已验证成果。

## 场景与服务流程

物理项以C1为运行快照。本轮确认C1.environment及cpu_ghz与C0一致，再结合QueueMDP构造时的覆盖和环境有效默认值核对。

- 【有效配置＋源码静态核实】3架UAV、39用户，每13人固定归属一架源UAV；区域[0,1000]m×[0,1000]m，UAV高度50m。用户移动不触发最近UAV重关联。来源：mdp.py:55；env.py:111、347–395、591–607。
- 【有效配置＋源码静态核实】reference_point_group分支即RPGM，群组参考运动叠加局部随机速度；0.5s更新运动量，0.1s积分位置。群组均速参数0.25m/s，局部速度上限0.6m/s。user_zone_radius=0是不施加该半径裁剪，不是用户位置重合。来源：env.py:1566–1584、1747–1862。
- 【有效配置＋源码静态核实】三相基准负载[12.5,5.5,3.5]、[3.5,12.5,5.5]、[5.5,3.5,12.5] Mbit/s，统一乘1.15，总期望24.725Mbit/s；相位循环，每相随机驻留整数8–12个大周期，切换仅影响后续到达，不清队列。来源：mdp.py:79–105；load_switching.py:74–139；env.py:1384–1526。
- 【源码静态核实】traffic_scale=0.5先影响基础类型到达率，随后按区域目标负载重标定，不能把24.725再乘0.5。CPU默认16GHz被cpu_ghz=10覆盖，U2U默认3MHz被6MHz覆盖。来源：mdp.py:55–58；env.py:_configure_industrial_device_profiles。
- 【源码静态核实】任务数按Poisson(用户到达率×到达间隔)生成；大小由设备类别和优先级决定：传感高/低优0.35/0.90Mbit，控制0.55/0.35Mbit，视觉1.20/1.40Mbit，并非连续随机大小。来源：env.py:29–47、1979–2018。
- 【源码静态核实】父任务出生即登记源UAV、优先级、大小和出生时间；用户上传队列FIFO。完整上传后，后续物理步取出上传完成任务，按当前帧固定路由比例只拆一次。来源：env.py:939、1064–1172、2636–2655。
- 【源码静态核实】本地分片进入源UAV计算队列；远端分片先传输，完整传完才进入目的计算队列。已经拆分的分片不重新路由，既有计算队列任务不再次迁移。来源：env.py:_route_arrived_parent_tasks、_consume_tx_queue。
- 【源码静态核实】一个服务物理步按既定顺序路由、上传、计算、U2U转发；计算先于转发，因此本步传完的远端分片最早下一步计算，本步刚上传完成任务也不能越过已经执行的路由阶段。来源：env.py:2636–2713。
- 【源码静态核实】父任务全部有效分片完成才完成，按物理步末登记时刻；端到端时延为完成减出生，不以单个分片完成代替。当前任务链未单独模拟结果回传队列/代价，论文需说明相应忽略假设。来源：env.py:_mark_chunk_completed:1042、_record_completed_parent_task:1007及上述处理链。
- 【有效配置＋源码静态核实】大周期10s、物理步0.1s、资源帧/到达间隔0.5s；前1s部署暂停、后9s服务，即2部署帧＋18服务帧。即使零位移也暂停1s；任务继续到达、用户继续移动，上传/转发/计算不推进。来源：env.py:166–168、2156–2196、2621–2623、3038–3044。
- 【论文当前记载＋源码静态核实】P1描述集中式控制，假设当前/历史状态报告同步准确，忽略控制信令丢失、时延和额外开销。代码直接读取环境记录、期望负载和位置计算链路，不能称期望负载是在线学习预测值或未来信息。实体控制器位置、现实测量误差属于未单独模拟的部署假设。
- 【待确认】投稿最终是否继续采用固定关联、暂停协议和结果回传忽略。本轮未确认或实现替代模型。

## 通信与计算

- 【有效配置＋源码静态核实】U2G在本项目是用户到UAV上行：20MHz总带宽固定分给39用户，按距离损耗和Shannon式计算速率。用户发射功率0.1W、参考增益10^-6、等效噪声功率10^-13W；并非按活跃用户重分带宽。来源：env.py:577–610及构造函数有效默认值。
- 【有效配置＋源码静态核实】U2U总6MHz在有非空转发队列的活跃有向链路间均分，对角不参与；链路高优先服务、余量给低优，UAV传输功率参数1W。来源：env.py:612–631、2690–2713。
- 【有效配置＋源码静态核实】CPU每架上限10GHz，1000cycles/bit；按本步高优需求先分频率，剩余给低优。cpu_power_cap_w=0表示关闭这项功率截断，不是功耗为零。来源：mdp.py:55–58；env.py:2198、2505–2517、2563–2566。
- 【源码静态核实】CPU能量κ(f_H+f_L)^3δ，κ=10^-28；上行/U2U分别计服务能量。部署使用kinetic_lower_bound附加项0.5mv²，不是完整加减速动力学仿真。来源：env.py:1540–1553、2680–2687、3046–3055。
- 【源码静态核实】RL控制部署和任务路由；CPU、带宽份额、用户关联及优先服务顺序是固定规则，不是额外学习动作。
- 【源码静态核实】奖励E=服务能量＋相对全悬停基线的部署推进增量按100物理步均摊；报告power_w则使用完整推进能量。来源：mdp.py:251–253、264–271、278–292。不得混写两种账本或添加硬总功率约束。

## 优化问题

- 【D2目标记载＋源码静态核实】关注优先级加权端到端时延、任务积压和既定能耗代价；不额外奖励本地率、转发量、CPU均衡或位移。P1将其写为受限聚合观测策略类的折扣物理代价最小化，不是全地图任意航迹的精确最优化。
- 【有效配置＋源码静态核实】每0.1s步代价c=0.9H/10+0.1E/14。H为服务结束后仍未完成父任务的加权任务秒，高/低权重10/1；E为奖励能量、单位J。H不是Mbit队列长度或本步完成任务均值。来源：mdp.py:37–43、256–276；env.py:750–777、2907。
- 【源码静态核实＋论文当前记载】控制量为部署与路由，约束包括区域边界、每次位移≤20m、整条同步直线部署路径UAV间距≥5m、执行路由非负且每源每优先级行和为1。来源：mdp.py:228–250；env.py:2943–2989；P1问题表述。
- 【有效配置＋源码静态核实】能耗为软代价，不设额外840W或其他硬总功率约束；CPU和带宽确定性规则不是优化问题的两个独立学习变量。
- 【源码静态核实】SAC带分数空间熵正则，soft Q不同于确定性策略纯物理回报。completed-only时延、未折扣平均J和训练折扣回报不能宣称逐样本严格等价。
- 【源码静态核实】观测是完整任务状态的因果聚合，不是已证明的充分马尔可夫状态；联训时另一层策略变化，不能视作两个独立平稳MDP。

## 大时间尺度

- 【源码静态＋运行记录核实】split_v1宏观观测67维，mdp.py:141–221与R2.observations.macro一致。顺序：计算队列6、按目的汇总转发6、上传6、未完成父任务6、最老年龄6、当前负载3、CPU能力3、U2U谱效率9、源群平均上行速率3、UAV坐标6、业务质心偏移6、历史质心速度6、相位年龄1。
- 【源码静态核实】压力按指定尺度作有符号log(1+|x|/scale)压缩，几何/能力/时钟按各自尺度处理。精确展平顺序及归一化见mdp.py:_observation_parts/_observation，不任意重排后复用权重。
- 【源码静态核实】动作网络67→256→256→6，输出均值；保存原始高斯动作及行为对数概率。log_std不参与梯度训练，而按进度调度。评价网络67→256→256→1估计状态价值V。来源：ppo_agent.py:32–85；learning.py:74–82、127–139、246–251。
- 【源码静态核实】每UAV二维分数限幅[-1,1]后映射单位圆，按负载关系构造局部正交基，在业务质心周围60m生成目标；继而处理地图边界、最大20m位移、0.5m死区及联合安全。坐标旋转不强制支援某节点，零分数不等于始终不移动。来源：mdp.py:236；geometry.py:6–23；env.py:2943–2989。
- 【有效配置＋源码静态核实】10s部署一次；周期回报为20个资源帧回报折扣和/20，后继折扣γ^20，γ=0.99；GAE λ=0.95，重置处截断跨流优势递推但保留下一状态价值。来源：experiment.py:118–125；learning.py:19–25、208。
- 【有效配置＋源码静态核实】300个大周期样本触发PPO更新；动作最多10轮且KL>0.03可提前停止，评价10轮；学习率3×10^-4、10^-3，clip=0.2；探索标准差在前200联合窗口由1/3衰减至1/30。来源：learning.py:201–251、C1.training。
- 【未来改动需核对】learning.py:208使用固定20帧宏观折扣，当前时序匹配；改变周期/帧长时需同步检查，不能宣称任意时间参数均自动适配。

## 小时间尺度

- 【源码静态＋运行记录核实】split_v1、routing_ready=false，68维；mdp.py:18–28、141–221与R2.observations.micro一致。顺序：计算队列6、完整转发18、上传6、未完成父任务6、最老年龄6、当前到达6、当前负载3、CPU能力3、U2U谱效率9、源群平均上行速率3、相位年龄1、周期内时钟1；额外routing_ready特征未启用。
- 【源码静态核实】动作网络68→256→256，分别输出18维均值/对数标准差；重参数化高斯经tanh得到18维[-1,1]分数，对数概率包含tanh修正。来源：sac_agent.py:24–47、243–282。
- 【源码静态核实】分数重排为(优先级2,源3,目的3)，每行欧氏单纯形投影。执行路由18个数、6个行和约束，独立自由度12；不能写成动作网络输出12维潜在动作。来源：mdp.py:228–234；routing.py:103–127。
- 【有效配置＋源码静态核实】executed_route使双评价网络/目标评价网络内部先把分数投影为路由再拼接68维观测，各分支(68+18)→256→256→1。相同观测与执行路由产生相同评价输入，不意味着消除全部价值误差。来源：learning.py:53–58；sac_agent.py:50–90。
- 【源码静态核实】熵在18维tanh分数空间，目标熵−18，不是12自由度单纯形路由熵。来源：sac_agent.py:301；learning.py:update_sac。
- 【有效配置＋源码静态核实】服务期0.5s决策一次；帧奖励按γ^(1/5)累计5个物理步负代价；普通后继折扣γ=0.99。
- 【源码静态核实】周期末经验暂存，经过下周期2部署帧后，回报r_last+γr_deploy,1+γ²r_deploy,2，折扣γ³；后继取部署结束且准备好下一服务帧到达后的真实观测，不用reset空队列。来源：experiment.py:57–61、80–115；mdp.py:224。
- 【有效配置＋源码静态核实】replay≥1000、batch128，每条流累计20服务帧触发一次更新事件。C1每事件只取一次批、构造一次目标，同批同目标拟合评价网络2次。前100事件不更动作网络，之后每2事件更动作网络及温度；目标评价网络每事件软更新1次，τ=0.01。来源：learning.py:149–198；experiment.py:113–115。
- 【有效配置＋运行记录核实】动作/评价/温度学习率3×10^-5、3×10^-4、3×10^-5。两次拟合不等于两倍环境采样，也不使动作、温度和目标网络时钟翻倍。

## 训练流程

- 【有效配置＋源码静态核实】C1累计200预训练＋1800联合；每窗口30周期=300s，每常规采样流3窗口=900s，尾部不足时缩短。预训练大层零分数跟踪质心；联训开启PPO，SAC继续更新。来源：qualification.py:348–350、C1。
- 【源码静态＋运行记录核实】preserve阶段切换保留SAC网络、目标、温度、优化器状态及学习率、replay与采样随机状态、累计更新/等待计数，不重启动作等待。来源：learning.py:118–125；qualification.py:288–311；R4。
- 【源码静态核实】每rollout重新构建/reset环境，300/600s日志窗口边界不清队列；流之间与阶段之间可重置。流末重置前执行真实下一部署补全SAC经验，不计入已结束窗口性能；PPO保留真实后继价值但截断跨重置GAE。不能说2000窗口全部无reset。来源：experiment.py:64–168。
- 【运行记录核实】原预训练质量passed=false，但fixed_budget政策、validation_integrity=true、allowed=true，因此已经联训不等于预训练已达质量门槛或收敛。来源：R2/R3/R4。
- 【运行记录核实】本次1800→2000；R4.start_window=1600是已有联合窗口数，加200预训练才是1800全局窗口。R4.exact_restore_checks记录两层网络、目标、优化器、replay、待更新PPO样本、计数/RNG均保留，本轮未重载checkpoint实测。
- 【运行记录核实】新增200联合窗口、108000经验；评价更新事件5400、评价梯度10800、目标更新5400、动作更新2700、PPO更新事件20。累计评价事件53951、评价梯度107902、目标53951、动作26925、PPO事件180。replay累计写入1080000不等于容量100000。PPO梯度轮数不能由事件数直接乘10。来源：R3.stage_audits/updates_total、R4。
- 【有效配置＋运行记录核实】初始化seed=1；终点评估34500–34504，监测33000/33001；最近续训场景从61602接续。场景编号连续不是环境队列永不重置。
- 【源码＋运行记录核实】固定2000末次权重，不挑最低点。此次是看过前期结果后的探索性续训，不等同于预先设计的等预算独立训练。来源：R1、R2.selection、resumption.py:151。

## 实验状态

### 已完成并可定位

【运行记录核实，未重跑】R0/R3确认2000端点完成。以下为5个复用场景各900s指标的算术平均，不是5个独立训练初始化，也不是合并所有父任务后重算的均值。来源：R3；聚合见qualification.py:44–48。

| 指标 | PPO＋SAC | 质心＋同一SAC | 质心＋固定均分诊断 |
|---|---:|---:|---:|
| 加权完成时延/ms | 598.759 | 597.179 | 609.247 |
| 高优完成时延/ms | 431.882 | 430.132 | 487.486 |
| 低优完成时延/ms | 1380.561 | 1379.822 | 1179.617 |
| 平均物理代价J | 0.692330 | 0.686208 | 0.696871 |
| 完成率 | 0.999677 | 0.999632 | 0.999752 |
| 平均期末积压/Mbit | 4.580 | 5.467 | 4.429 |
| 完整平均功率/W | 832.290 | 825.995 | 823.510 |

- 同SAC质心列由R3.matched_centroid逐场景数值算术平均；未运行环境。
- queue_total_mbit为期末上传＋计算＋转发积压，不是时间平均。PPO＋SAC时间平均计算11.860748Mbit、转发0.801118Mbit，见R3.mean。
- 【运行记录核实】质心＋同SAC为部署组件配对重放，均分是不训练的诊断，都不是独立训练竞争方法；见R3.baseline_scope。
- 【运行记录核实】R3记录配对用户/到达一致，UAV轨迹不同是部署对照预期；本轮未读取NPZ独立核验。

### 当前可以支持

- 【运行记录核实】两层确实完成训练更新，取得上述配置和场景内的有限时间指标。
- 【由R3比较】相对均分诊断，加权/高优完成时延及J较低；低优时延、功率、期末积压并非更好，完成率也未提高。
- 【由R3比较】相对同SAC质心部署，PPO＋SAC加权时延高约1.580ms，J及功率更高；不能据此称PPO部署已带来性能提升。
- 上述只限一个训练初始化及复用场景，不直接等于论文创新验证充分。

### 尚未完成或证据待确认

- 【待确认】最终投稿冻结端点、配置、等预算独立训练基线、独立训练重复与未用于开发的评估集，本次材料不足以确认已全部完成。
- 【待确认】边界处理、执行路由评价、部署参数化的独立消融及控制器推理开销，需指定记录；本轮不遍历历史寻找。
- 【待确认】实际信令误差、替代暂停协议、完整任务阶段时延因果分解，不从期末队列和几何均值猜测。
- 【运行协议记载】TQC未参加此次1800→2000续训；未读取其独立报告，不给TQC成绩，也不做2000对1800的等预算排名。
- 【不能支持】普遍SOTA优势、全局最优、无限时域队列稳定、统计显著性、多训练初始化的收敛/鲁棒性保证、PPO正向作用已得到因果验证。soft Q不能直接当短期确定性分叉回报真值。

## 代码索引

行号按本次SHA256定位；代码变化后同步更新。大文件按相关函数阅读，不声称全仓库审计。

| 功能 | 文件 | 函数/类及定位 | 证据等级 |
|---|---|---|---|
| 入口及覆盖 | scripts/run.py | main，31–68 | 源码静态 |
| 最近续训 | scripts/run_convergence_watch.py | continuation_config，56；main，123–185 | 源码静态＋R1/R2 |
| 环境/CPU覆盖 | hrl_uav/mdp.py | QueueMDP.__init__，55–77 | 源码静态＋C1 |
| 环境步进 | hrl_uav/env.py | micro_step；macro_step；服务顺序2636–2713、暂停3038–3044 | 源码静态 |
| 任务产生 | hrl_uav/env.py | _generate_traffic，1979；_create_parent_task，939 | 源码静态 |
| 到达率标定 | hrl_uav/env.py | _get_zone_target_load_mbps，1429；_configure_industrial_device_profiles，1440 | 源码静态 |
| 负载轮换 | hrl_uav/load_switching.py；hrl_uav/mdp.py | build_load_switch_schedule，74；reset/advance_macro_clock，79/95 | 源码静态 |
| 用户运动 | hrl_uav/env.py | _update_reference_point_group_mobility，1747；_integrate_user_positions，1862 | 源码静态 |
| 上行服务 | hrl_uav/env.py | compute_u2g_rate_bps，577；_serve_uplink_task_queues，1069 | 源码静态 |
| 拆分及完成 | hrl_uav/env.py | _route_arrived_parent_tasks，1172；_serve_compute_task_queues，1271；_mark_chunk_completed，1042 | 源码静态 |
| U2U服务 | hrl_uav/env.py | allocate_u2u_link_bandwidth_hz，612；_consume_tx_queue，1294 | 源码静态 |
| CPU服务 | hrl_uav/env.py | _queue_to_frequency_need，2198；_allocate_cpu_frequencies，2505 | 源码静态 |
| 观测 | hrl_uav/mdp.py | task_aggregates，107；_observation_parts，141；_observation，184；observation_spec，216 | 源码静态＋R2 |
| 部署动作 | hrl_uav/mdp.py；hrl_uav/geometry.py | start_macro，236；cooperation_aligned_basis，6 | 源码静态 |
| 路径安全 | hrl_uav/env.py | _project_joint_motion_to_safe_separation，2943 | 源码静态 |
| 路由映射 | hrl_uav/mdp.py；hrl_uav/routing.py | route，228；project_rows_to_simplex，103 | 源码静态 |
| 奖励 | hrl_uav/mdp.py；hrl_uav/env.py | holding_energy_cost，37；step_frame，256；get_alive_task_wait_info，750 | 源码静态 |
| 跨周期处理 | hrl_uav/experiment.py | boundary_reward，57；rollout，64–168 | 源码静态 |
| PPO网络/更新 | hrl_uav/agents/ppo_agent.py；hrl_uav/learning.py | Actor/Critic；DirectHRL.update_ppo，201；gae，19 | 源码静态 |
| SAC网络/更新 | hrl_uav/agents/sac_agent.py；hrl_uav/learning.py | Actor，24；Critic，50；DirectHRL.update_sac，149 | 源码静态 |
| 阶段保留 | hrl_uav/learning.py；hrl_uav/qualification.py | set_stage，118；run_mainline中288–311 | 源码静态＋R4 |
| 续训恢复 | hrl_uav/resumption.py | validate_resume_config，14；load_source，72；restore_agent，171 | 源码静态＋R4 |
| 评价统计 | hrl_uav/qualification.py；hrl_uav/mdp.py；hrl_uav/env.py | metered_evaluate，21；means，44；metrics，278；get_task_delay_stats，779 | 源码静态＋R3 |

## 本轮与上一确认版本的变化

### 本轮实际实施

- 只新建根目录AGENTS.md与本文件，不修改模型、算法、配置、论文、图或既有实验记录。
- 从C1/R2/R3/R4确认指定2000窗口端点；不把旧聊天的1200/1800等成绩写成该端点。
- 记录SAC及动作/评价网络命名要求，历史文件字符串仍保留。

### 仅计划、尚未实施

- 按作者最新要求同步对应论文section、图注和算法框的旧译名、Direct SAC、普通编号贡献及引用句式；后续须限定章节范围。
- 没有在本轮确认新的奖励、动作、网络或场景方案；讨论中改法不能写入当前已实现模型。
- 如果选择2000端点投稿，需要同步预算、评估集合、评价拟合次数和结论边界，而不是仅替换成绩数字。

### 版本冲突与影响

1. C0为200＋600窗口、每事件默认拟合1次、评估33500–33502；C1为200＋1800、每事件2次、评估34500–34504。都是真实文件，但不能拼成一个实验。
2. README“新入口尚无性能结果”等段落早于R0完成记录，仅作历史描述，不作为当前进度。
3. P1/04_algorithm.tex:193仍写200预训练＋800联合，R3是200＋1800；影响训练、预算、实验与结论。本轮不自行选2000为投稿版。
4. P0、P1、P2和后来聊天中的section不同；P0的xu2024/hao2024与增量稿长引用键不一致，替换要匹配工程文献库。Overleaf当前内容未读取，不能称已经同步。
5. P1算法标题/图注及P2仍有Direct SAC、演员、评论家旧词，P2还有enumerate贡献格式；待按最新要求改，本轮未改。
6. ppo_agent.py头部/类注释仍有MAPPO、单机局部观测旧说法，实际learning.py使用单套67维观测、6维联合动作、标量状态评价的集中式PPO；不能依据注释称当前是MAPPO。
7. 67/68观测、18维分数与12自由度、固定20帧宏观折扣、奖励增量能量与报告完整功率须成套核对；以后更改时序、动作、奖励会影响模型、MDP/SMDP、算法框、图和实验可比性。
8. 旧结果只对其自身配置、训练经历、评估集有效；未判定所有旧结果失效，也不自动纳入正式对比。D2/D3仅保留1200/1800范围。
9. 样文清单是写作参照，本轮未重读PDF，也未确认另外两篇样文的本地位置；不宣称完成逐篇比对和版式检查。

## 实际读取范围与未执行操作

- 源码/配置：文末前14项；沿任务链、动作/奖励/边界及训练调用关系读取，env.py等大文件按函数定位。
- 运行材料：仅定向读取R0–R5、C1；report按需要提取字段，未加载权重、完整训练日志、轨迹NPZ或经验池。
- 文档/稿件：D1–D4、P0/P1/P2及P3相关定位。算法文件按相关段落核查，不声称逐式完成全文审稿。
- 检查：文本读取、JSON配置一致性、静态维数推算、限定git查询、SHA256比较、已有逐场景指标均值计算。
- 未执行训练、仿真、重新评估、单元测试、论文编译、图生成、依赖安装、全仓库/历史归档扫描、打包、删除、移动、提交或推送。只写两个新说明文件。

## 文件指纹

以下SHA256为本次读取时快照，前14项与R2.source_hashes一致，后5项固定本次运行证据。源码读取不等于重新执行验证。

| 文件 | SHA256 |
|---|---|
| scripts/run.py | 32df703c7cedae061442ade9522bbe19d65c3c93230407211e0a3f2e7a69a294 |
| scripts/run_convergence_watch.py | e5e1602e5a3f1af9e24512ba9bec0214fb3d83497749f9f692c5b0802e2b591f |
| hrl_uav/env.py | d5574bf66bb085cb8fb724882d2cccd50701244e73b5626c15cfeba7028ed05e |
| hrl_uav/mdp.py | 7c4873f82e392d9203e81c1a8929bd3d6618ebf21a1c24f10cee3a18cfaccf98 |
| hrl_uav/geometry.py | 93a32babbb42d6df2676b274a57673afa4e6896e717f56988369726d254241af |
| hrl_uav/load_switching.py | cdef8c0270304164e0e7e070c89bb2fd49ea0914e3452d3ea9ef44b2380892b4 |
| hrl_uav/routing.py | 5e45b645b8eb97bb5b1ee6fe8f943631adfd235d37c304d3cf6f8b171c56ddde |
| hrl_uav/learning.py | ef350d543600b18f01b3549e56bf563a5cc8da6edff1cc0d9325d7675f42fa11 |
| hrl_uav/experiment.py | 4b8485e42ec65fcef0f2a591fa761fd347cbf5b6e7968fef5c21263f189bbe7a |
| hrl_uav/qualification.py | 4a6879d8f7d7e67a164e70ce5be8767f879e0108726f9434838e34cd967e48cd |
| hrl_uav/resumption.py | 7f8e9a61b43dc67d1ef5814b645c685e9b03167959e03cdfe05395567b5b50e9 |
| hrl_uav/agents/ppo_agent.py | d9193fd36fb47c283974daae6b0256c806c1ab9a21610f72baa017ec5e7889e6 |
| hrl_uav/agents/sac_agent.py | e0dd62ffa39d3045f568ab5c4712b04ed1cf4acd0e8396b35f6a0a12941147a8 |
| configs/hrl_ppo_only_unlock.json | 1ee1f10d4185bcaee5300dfc45ff4fdc974645f62d508e40d1afadd4ef8cdb60 |
| results/sac_continue1800to2000_seed1/config.json | 099cbea903a4c40ea6307804f7beb6ca19a24e994f2a582b12cfa53e11eab449 |
| results/sac_continue1800to2000_seed1/status.json | 57e756a868be5f2d0b13fe40342c4a571c028646ca84cc20e4b70482d5ebd492 |
| results/sac_continue1800to2000_seed1/at2000/manifest.json | d0ea07c1cac65db30f48a644c0021f0c12430fa7878d91cf5370300feb635698 |
| results/sac_continue1800to2000_seed1/at2000/report.json | 99e1fc5ec2657e51165805c1775e2cfabbcd91f9918a79bd0a7a56218be1405e |
| results/sac_continue1800to2000_seed1/at2000/resume_audit.json | cd1e1e1642d2993ce98b142ba8557025603dba9142bf3dc1683e635e08c971a9 |

## 2026-09-09 追加：候选无线模式功能扩展

- 【源码静态核实】本次基准/实际 HEAD 为 `07a727d8189be4f73c0218fca47603d8cbf6956b`，分支 `codex/u2u-model`；上述历史源码行号和指纹保留为原版本快照，不适用于本次新增行的定位。
- 【已实现】`environment.wireless_model` 缺省或 `legacy_fixed_noise` 保持旧定义；显式 `psd_per_uav_budget` 使用固定等效噪声谱密度与每架 UAV 的 U2U 总射频发射预算。新模式仅支持活跃有向链路均分总带宽，拒绝自定义固定链路带宽矩阵。
- 【参数来源】`b_ref=20e6/39 Hz`，`N_eff=1e-13/b_ref=1.95e-19 W/Hz`，是旧上行参考条件的等效换算，不是实测或理想热噪声参数；运行中不随用户数、带宽和活跃数重标定。默认每机 U2U 预算1 W、参考功率1 W，独立于旧 `p_u2u` 和整机功率账。
- 【模型语义】物理步按高低优转发积压分配一次带宽与每源功率，高优先服务、低优使用剩余容量；发送能量来自同一份实际功率和实际发送时长。部署期间通信仍暂停，`compute_during_deployment` 的独立默认值和 CPU 规则不变。
- 【观测边界】9维 U2U 特征明确为固定参考质量，宏观/微观仍为67/68维，不增加带宽或 routing_ready 输入。实际服务速率逐步计算，不使用位置缓存的参考量替代。不声称实现了跨带宽自适应，旧权重不是已训练的新模式策略。
- 【功能运行验证】新增 `tests/test_wireless_model.py` 7项测试，连同既有部署CPU、通信和带宽短测试共31项通过；包含与基准源码逐值旧模式回归（CPU开/关）、独立链路算例和3周期固定规则rollout的实际资源/能量/奖励/CRN检查。未训练、未加载权重、未反向传播，未运行900s评估或参数扫描。
- 【保留范围】三个未提交的六臂预筛选源码/测试及两个预筛选结果目录原样保留；configs与这两个结果目录共169个文件逐项指纹未变。本轮未修改论文、既有JSON或历史结果，未提交推送。
- 【不能支持】硬件并发能力、真实接收机参数、长期队列稳定、候选性能更好及新模式下的学习结果均未验证。历史588.47 ms、601.43 ms仍属于旧无线模式，不能移植为本候选成绩。
- 定义、代码差异、指纹、人工算例、实际测试命令及后续评估条件见 `docs/wireless_model_psd_per_uav_budget.md`。此次只追加记录，不否定或改写旧模型。
- 【历史措辞差异，未改实现】原“路由、上传、计算、U2U”概述未严格对应 micro_step 的语句顺序：实际先取此前上传完成任务，推进本步上传和运动，再路由此前取出的任务，随后计算和U2U；本步刚上传完成的任务仍须等待后续路由。本次保留原记录并单列差异，不以无线扩展为由重排服务流程。

## 2026-09-12 追加：continuous-motion mission V1 短程实现

- 【版本】基准与HEAD为 `92d3214788ee94de82ae3c0186fdd3ac1566e765`，新分支 `codex/mission_rpgm_v1`；以下为未提交候选实现，不覆盖前文指定版本事实。协议见 `docs/mission_rpgm_v1.md`。
- 【已实现】`environment.motion_protocol` 缺省/`legacy_fly_hover` 走旧协议；显式 `mission_rpgm_v1` 使用三组预定900秒直线参考路线、10秒世界坐标直接位移、全周期匀速移动与连续通信、0.5秒路由和0.1秒物理步。每架位移不超过20米，仍经过原路径安全限制，零动作表示悬停。
- 【任务时序】仅新模式在步首几何下执行上传、路由此前上传完成任务、CPU、U2U，再更新用户/UAV位置；本步上传完成与U2U完成各保留一个物理步的原任务流水滞后。旧模式服务/运动语句分支保留。
- 【能量】沿用原P(v)，完整推进能量为10P(v)，目标增量为10[P(v)-P(0)]，允许为负；新模式取消每macro启停附加项，明确忽略加减速及转向能耗。无线、CPU、奖励、观测维数及学习实现未改。
- 【功能验证】新增专项7项，连同已有部署11项、无线7项共25项相关测试通过；包括与修改前源码逐步字节级回归、任务时序、能量、动作安全、随机流、20次SAC决策及无网络替身的真实rollout边界守恒。旧行为兼容结论限于已测路径。
- 【执行范围偏差】一次扩大测试调用共34项通过，但额外包含的旧 `tests/test_split_observation.py` 测试内部执行临时SAC/PPO网络更新，违反本轮无网络训练要求。未加载/保存/修改SAC101，已向用户报告且未再运行该文件；本轮整体不能表述为绝无反向传播。详细测试名与命令见协议文档。
- 【独立smoke】`results/mission_rpgm_v1_smoke/`：场景33600，Hover/Current/Progressive/Predictive/Blind Patrol各60秒；SAC101 window2000哈希为 `be946d3a576918460690a16f1a57f2c6dc33dde13db86d360db5690858a76932`。该smoke使用冻结确定性推理，学习、replay写入和采样为0，网络/优化器/梯度/行为状态核验不变，外生过程配对一致。
- 【风险信号】Blind Patrol相对Hover的J变化为-0.00093722，其中驻留项+0.009855、能量项-0.01079222；三架推进能量减少922.468 J/60秒。盲巡航服务更差却因能量抵消小幅胜过Hover，但仍差于三种追踪规则。只能确认这一短程现象，不能断言PPO将学会巡航或新模型性能更好。
- 【停止】未运行900秒、33700–33709或正式训练；没有根据结果调参数、约束或奖励，没有提交推送。轨迹图只画真实0–60秒，历史时延成绩未改写；等待下一轮授权。
