# HRL_UAV V2：从零训练的实现规格 1.0

日期：2026-09-26。状态：**新设计，尚未完成环境接入、训练或性能验收**。

本文件中的“必须”是实现合同，不是已有实验结论。数学参考模块已做14项CPU单元测试；不能把这些测试说成完整系统通过验证。数值超参数是预先选定的第一版，不声称已调优。

## 0. 范围、依据与不能伪造的承诺

### 0.1 最终目的

在不降低原负载、不放宽物理约束、不改变任务分片含义的条件下，构造一个每次都能从随机初始化启动的PPO上层＋SAC下层流程；用独立训练种子、未用于调参的场景和预先定义的指标验收。旧S250、S330、P300、Anchor检查点不作为初始化、教师或先验。

“解决旧问题”分为两种：

* **可通过构造和测试消除的缺陷**：丢弃当前可报告的FIFO/父子关联、下层投影平坦区、分数熵与分配熵混称、跨PPO版本回放混用、宏微奖励不一致、混淆终止/截断、种子未真正传入初始化、用开发结果冒充独立测试。
* **必须实测的要求**：低时延、低成本、拥堵响应、学习不明显回退、跨种子可靠、部署优于悬停。有限神经网络和有限随机训练不能在实现前证明对所有状态、所有随机种子达到最优，也不能保证Q永远准确。未通过这些验收就是未完成目标，不得把局部补丁改名为成功。

### 0.2 保留的任务；明确修改的训练目标语义

**主任务冻结为原900秒评估窗口内的有限时域服务优化。** 900秒物理过程和原未折扣cost/s不变；改变的是学习端：从原0.99折扣且900秒bootstrap，改为有限900秒、gamma=1、900秒真实终止。所有控制器均知道剩余预算。这是显式的V2设计决定，不是对旧实现“本来就是有限时域”的误述。

因此V2只主张解决该900秒任务，不能据此宣称无限期队列稳定。未来1800秒/持续部署属于独立泛化测试，不得用900秒模型的结果代替。没有凭空添加截止后成本，没有把未完成任务算成零时延。

### 0.3 来源边界

旧实现依据用户上传的 `anchor340_text_evidence_complete.zip`、`HANDOVER_20260925.md`、16次评价复核、anchor交付台账。物理文件字节身份见 `physical_source_binding.json`。

轻量ZIP没有携带完整原物理scene JSON。本地绑定路径和已有证据中的SHA已写入该文件。实施者只需从本地现成scene导出并核对，禁止从本文举例重构物理参数、禁止选一个相似默认场景。完成此绑定后，scene JSON必须随新实验代码一同交付。

---

## 1. 理想目标与正式验收

### 1.1 决策对象

上层决定三架UAV的二维移动请求；下层决定两种优先级、三个来源分别向三个目的节点的连续分片比例。CPU与带宽调度仍是原物理系统的固定规则，不改成新的学习动作。

### 1.2 主要目标

令物理步长为 \(\delta=0.1\) 秒，\(K=9000\)。在物理步开始时，\(N_p(k)\)为优先级p尚未完成的父任务数；\(w_H=10,w_L=1\)。

\[
H_k=\delta\,[10N_H(k)+N_L(k)],\qquad
c_k=0.09H_k+E_k/140.
\]

其中\(E_k\)是原模型CPU、推进/悬停、U2U和上传能耗之和，单位J。

\[
J(\theta,\phi)=\mathbb E_{\xi,\epsilon}\left[\frac1{900}\sum_{k=0}^{8999}c_k\right].
\]

\(\xi\)为外部场景和任务流；\(\epsilon\)为明确登记的策略随机流。目标是最小化J，不是最大化动作熵、不只是最小化已完成任务平均时延。

### 1.3 目标分层

1. 硬验收：物理守恒、动作约束、无未来信息、零旧学习状态依赖、可恢复、评价零更新全部通过。
2. 首轮性能目标：在固定开发协议中，运动主方法优于同一SAC的悬停控制，并具有与预登记强基线竞争的成绩。
3. 最终工程目标：五个独立训练种子全部报告；每个种子在独立面板上成本不劣于预登记最强基线及同一SAC悬停；跨种子平均成本和pooled时延各改善至少3%。这是一项**预设验收目标**，不是有效性定理。
4. 服务底线：高优未完成率相对对应基线最多增加0.0001，低优最多增加0.0005；任何违反都报告为取舍或未通过，不以成本下降掩盖。
5. 绝对时延600 ms只作为挑战目标，不能承诺、不能通过调低负载达到。

五个种子的结果只支持“这五次”，不支持“所有未来种子必然成功”。若无法满足预设目标，版本仍是候选，不重新包装为已解决全部问题。

---

## 2. 保留什么、重写什么

### 2.1 只读保留

* `IndustrialPhysics`调用的原无线信道、带宽/功率分配、推进能耗和路径安全函数。
* `IndustrialEnv`的上传、分片、CPU、转发、完成判定、守恒及物理能耗。
* `IndustrialDeploymentEnv.start_motion`的1秒调整＋4秒驻留及安全映射。
* `HotspotIndustrialEnv/Scene`的原场景布局与任务生成过程。
* 原CVX/Local/Random规则作为命名明确的基线；CVX仍是代理优化规则，不冒称全局最优。
* 旧实验、旧检查点、C盘兼容入口及D盘大文件迁移结构。

不编辑上述源文件；新适配层只能调用它们和导出额外的因果报告。

### 2.2 独立重写

* 报告schema与新编码器；原86/523维不再是新版接口约束。
* 上层actor/value、下层actor/twin-Q及目标Q。
* 下层动作分布、概率密度、梯度估计；不继承旧raw-tanh-simplex actor。
* 新collector、gamma/terminal逻辑、回放、阶段调度和checkpoint格式。
* 新seed管理、完整训练CLI、统一测试入口和指标聚合。

不是“全仓库推倒重来”；是**保留物理内核，控制器与训练系统重新编写**。不得继承`DirectHRL.update_sac`后不停打补丁；其代码仅作旧版本参照。

### 2.3 新文件布局

```text
hrl_v2/
  reports.py               # immutable schema、因果报告检查
  physics_adapter.py       # 调用只读原物理核；原时序；不含学习
  features.py              # 固定尺度转换、图、队列及历史构造
  networks.py              # typed graph encoder、PPO与SAC头
  hybrid_simplex.py        # 7支持集＋Dirichlet；动作与密度
  sac.py                   # critic与逐行枚举actor梯度
  ppo.py                   # gamma=1, lambda=1, clip PPO
  replay.py                # 完整episode、上层版本隔离、分层IS
  collector.py             # 宏微动作时序、实际后继、终止
  schedule.py              # 唯一训练状态机
  randomness.py            # 分流seed、状态保存
  checkpoint.py            # 完整新格式，不读取旧模型
  evaluate.py              # stochastic主评价＋deterministic补充
  metrics.py               # cost、pooled、删失与分片账本
  baselines.py             # 因果适配；不得偷看未来
  run.py
configs/v2_1_0.json
configs/physical_scene.json
tests/v2/...
runs/v2/<protocol_sha12>/<method>/seed_<id>/...
```

这些是待实现的文件，不声称附件已提供整个程序。

---

## 3. 时间、任务与物理时序

### 3.1 固定时钟

| 量 | 值 |
|---|---:|
| 物理步 | 0.1秒 |
| 下层决策 | 0.5秒＝5物理步 |
| 上层决策 | 5秒＝10下层周期 |
| 每宏步运动段 | 1秒 |
| 每宏步驻留段 | 4秒 |
| 每episode | 900秒＝180宏步＝1800下层周期 |
| 最大单次目标位移 | 保留原10米及原安全映射 |

不要把控制周期写成网络更新周期。

### 3.2 父任务与分片

父任务j由\((m_j,s_j,p_j,b_j,D_j,C_j)\)标识：设备、固定所属UAV、优先级、出生时刻、输入bits、计算cycles。数值任务ID只作关联键，不进入网络特征。

路由行\(a_{p,s,d}\)应用于本周期原逻辑实际取出的ready父任务。\(a_{p,s,d}>0\)则产生目标d的分片：

\[
D_{jd}=a_{p_j,s_j,d}D_j,\qquad C_{jd}=a_{p_j,s_j,d}C_j.
\]

\(a=0\)不产生该分片。父任务完成时刻为所有实际创建分片完成时刻的最大值。每个父任务只分片一次，不在后续周期反复切割、迁移或复制。

### 3.3 必须保持的0.1秒执行顺序

直接复用当前`micro_step`，顺序不可“优化”为另一套：

1. 用步开始时的所有活跃父任务计算驻留成本。
2. 取出步开始时ready队列；本步新上传完成的任务仍按旧实现等待后续步被取出。
3. 上传服务。
4. 用当前路由比例分配先前取出的ready任务。
5. CPU服务。
6. U2U转发服务；本步刚转发完成的分片按旧实现最早下一物理步参与CPU。
7. 推进能耗和位置积分；更新时钟、完成事件及公共报告。

每个0.5秒路由应用五个物理步；期间有新上传完成父任务，不能用决策瞬间ready量×比例冒充整周期实际分配cycles。

### 3.4 物理容量不被“修复”为动作屏蔽

对每个UAV i，保留：
\[
f_{i,H}=\min(F_i,Q_{i,H}/\delta),\qquad
f_{i,L}=\min(F_i-f_{i,H},Q_{i,L}/\delta).
\]

同优FIFO、转发高优先等均不变。拥堵节点仍可合法排队；不得凭空把“很差的动作”定义成违反硬约束。V2不乘一个任意“容量权重”后重新归一化动作。

---

## 4. 信息边界：完整报告，不泄漏模拟器秘密

### 4.1 控制器唯一输入

`ReportV2`是不可变数据对象。actor/critic只接收该对象和已记录的历史，不能获得env、scene生成器或ledger对象引用。

报告由适配层从**此刻可被中央调度器报告的运行状态**构造。假设同步零延迟控制遥测、其开销未计入物理成本，与原研究的理想化遥测口径一致；同时记录报告字节量，不能声称已经验证真实通信部署开销。

“当前链路速率”指原物理分配器对当前活跃队列集合给出的即时分配值，inactive取0；all-active另标为原模型假设值，不能混称实际观测速率。原scene中的旧学习discount字段保留作来源身份，但V2学习端只使用本规格第9节，无权改变实际物理服务。

新增的信息必须同样向所有基线开放。未经任务模型允许的额外数据不能悄悄仅供V2使用。

### 4.2 原始schema（不得top-k截断）

* 时钟：physical_tick、routing_index、macro_index、remaining_ticks、phase、当前已提交target/velocity、到下一高层决策剩余时间。
* 39份终端报告：位置、固定source、当前上传FIFO中每个活跃父任务、每0.5秒本次已经发生的高/低优到达bits/cycles/count、报告时刻和有效性。
* 3份UAV报告：位置、速度、已提交target、CPU参数、各优先级CPU FIFO、当前ready工作、近5秒实际CPU服务量。
* 每条有向U2U链路、每优先级FIFO：分片次序、剩余bits、对应后续cycles、实际当前链路速率、原模型all-active假设速率及二者标签。预测的all-active速率不得冒充真实吞吐量。
* 活跃父任务表：终端键、source、priority、出生时刻、完整bits/cycles、upload剩余量、已发生的upload/route状态、已完成子片数、未完成子片键。
* 活跃分片表：父键、source、destination、priority、原始分配比例、完整bits/cycles、剩余transfer/compute、当前阶段、进入当前阶段时刻。
* 原始队列顺序：upload×39、全局ready×1、CPU×2×3、U2U×2×6，共58条队列；空队列也保留类型和owner。

不把最终完成时刻、最终尾片所在节点、未来到达、隐藏业务模板/忙闲状态、生成器强度或真实随机状态暴露给策略。已发生的完成确认可以报告；“当前仅剩一个未完成分片”是合法当前事实，不是未来尾片标签。

### 4.3 尚不能消除的部分可观测性

完整服务报告移除了已知的FIFO/分片关联缺失，但未来到达和隐藏业务变化仍未知。理论策略作用于合法历史\(\mathcal H_k\)；实现使用报告图和因果到达历史编码近似它。不得宣称有限维神经编码已证明是充分马尔可夫状态。

这是不可用“偷偷读取真实忙闲模式”根除的限制。需要在论文中承认，并由性能验收检验实际影响。

---

## 5. 特征与历史编码（实现者不自行猜归一化）

### 5.1 统一转换

* 所有非负计数和工作量：\(L_s(x)=\log(1+x/s)\)，x<0或非有限值直接报错，不裁剪伪装。
* bits尺度\(10^6\)，cycles尺度\(10^9\)，rate尺度分别\(10^6\)bit/s、\(10^9\)cycles/s。
* 年龄/时间长度用\(L_{10}(x)\)；计数/queue rank用\(L_1(x)\)。
* 坐标除原map_size；有符号相对坐标同样除map_size；速度除原最大速度。
* 当前已提交target除map_size；remaining episode除900，remaining macro除5，remaining adjustment除1。
* 比例及布尔值不做log。stage/priority用one-hot。
* 缺失时间用数值0＋独立valid bit；不把缺失伪装成真实0。

### 5.2 因果到达记忆

每终端每路由帧输入6维：高优和低优各自new_bits、new_cycles、new_count，使用上述变换。共享参数、单层GRU，hidden=16。

\[
h_{m,k}=\operatorname{GRU}_{\rho}(x^{arr}_{m,k},h_{m,k-1}),\quad h_{m,-1}=0.
\]

上层在本宏步新到达尚未注入时，只能使用此前已发生的历史；下层在prepare_frame后包含本帧新到达。历史cursor必须保存在报告中。

回放保存原始到达序列，不把旧网络的hidden当作当前hidden。每次优化对本batch涉及的episode用**当前对应网络参数**重算完整前缀，并对到达GRU做完整BPTT。它只处理39×6的到达信号，不对整个历史任务图反复做图编码。target-Q使用target自己的编码参数。

如果计算成本不满足预算，必须作为新版本改动报告；禁止静默将完整前缀换成零初始化的短历史，或截掉队列。允许数学等价的batch/vectorization与梯度累积。

### 5.3 节点特征

不向网络传入数值ID、seed、scenario_id、checkpoint编号、业务模板名。ID只用于建图和稳定排序。

* Terminal（27维）：xy(2)、所属链路当前rate(1)、高/低upload bits/cycles/count/oldest age(8)、arrival-GRU hidden(16)。
* Parent（14维）：priority(2)、D(1)、C(1)、remaining upload(1)、age(1)、stage[ground/ready/routed](3)、routed bit(1)、unfinished child count(1)、completed child count(1)、time since routing(1)、route-time-valid(1)。
* Fragment（16维）：priority(2)、stage[transfer/cpu](2)、原分配比例(1)、原D(1)、原C(1)、remaining transfer(1)、remaining compute(1)、stage age(1)、parent age(1)、queue rank(1)、同优前方工作(1)、高优前方工作(1)、其他未完成siblings数(1)、唯一未完成child bit(1)。CPU前方工作以cycles计，transfer以bits计；阶段位区分单位。
* UAV（24维）：xy(2)、velocity(2)、target(2)、remaining macro(1)、remaining adjustment(1)、CPU F(1)、高/低CPU积压秒数(2)、高/低在途compute/F(2)、高/低近5秒实际CPU服务量(2)、高/低ready bits(2)、cycles(2)、count(2)、phase one-hot(3)，顺序[uncommitted, adjustment, hold]；按已提交宏动作时钟判断，不按速度是否为0判断。

Fragment的高优前方工作：对low用当前对应资源的全部high queue工作，对high为0；同优前方工作按当前真实FIFO求和，不包含该分片自身。它不是对未来低优等待的精确预测。

---

## 6. 网络：队列次序＋父子图＋几何关系

所有actor/value/Q有独立编码器参数，避免某层或某个损失暗中改变被冻结策略。下层两个Q及target也分别有独立编码器。运行中不得共享可训练的全局normalizer。

### 6.1 初始嵌入与队列编码

每种节点独立`Linear(input,64)→ReLU→Linear(64,64)`。

四种资源分别一个GRU(64,64)：upload、ready、transfer、cpu。对58条队列按真实head→tail次序运行，初始hidden=0；保存**每个队列项**的hidden，不仅保存末项。将item原64维与queue-context64维拼接，线性投影回64，LayerNorm(eps=1e-5)。同类型队列共享GRU参数，不同类型不共享。

空队列向owner提供零context和显式count=0。父任务ground/ready进入对应队列，已路由父任务不伪造queue位置；分片只进入它当前所属transfer或CPU队列。

### 6.2 图边

使用有类型双向边：terminal↔source-UAV；parent↔terminal；parent↔source-UAV；parent↔active-child；child↔source-UAV；child↔destination-UAV；queue predecessor/successor；三UAV之间的6条有向链路。重复关系保留独立类型。无数值边特征的关系使用长度0的边向量和可训练类型偏置；Q投影按层/head共享，K/V/边投影按关系类型独立。反向边使用独立关系类型。

UAV链路边特征9维：current rate、all-active rate、currently-active bit、high/low transfer bits(2)、high/low future cycles(2)、current power、distance/map_size。

Terminal↔source-UAV边特征：相对xy、距离、当前rate；仅原固定关联链路提供rate，不伪造改关联后的吞吐量。其他UAV关系通过UAV-UAV消息传递；不能把注意力解释成关联切换。

### 6.3 三层有类型注意力

宽64，4个head，每head16维，无dropout。每层每head：
\[
e_{uv}=\frac{(W_Qh_v)^\top(W_K^{\tau}h_u+W_E^{\tau}x_{uv})}{\sqrt{16}}+b_{\tau},
\quad \alpha_{uv}=\operatorname{softmax}_{u\in\mathcal N(v)}e_{uv},
\]
\[
\tilde h_v=\operatorname{LN}\left(h_v+W_O\operatorname{concat}_{heads}\sum_u\alpha_{uv}W_V^{\tau}h_u\right),
\]
\[
h_v^+=\operatorname{LN}\{\tilde h_v+W_2\operatorname{ReLU}(W_1\tilde h_v+b_1)+b_2\}.
\]

FF宽128；self-edge始终存在，避免空softmax。每层有独立参数，不做全体任务间N²无掩码注意力。入边按destination和关系类型稳定排序；可按入度上取2的幂分桶后padding/mask，不能删边。优先用gather、masked softmax和按邻居维求和实现，避免无控制的原子scatter累加；确定性模式报不支持时必须修正实现而不是关闭确定性标志。

### 6.4 UAV读出

每UAV以自身64维为query，分别对它的terminal、source-parent、destination-fragment做单头注意力读出（query/key/value均64）；三类空集合输出0。拼接自身和三类读出共256维，再拼三类log count(3)、elapsed/remaining episode(2)，共261维，经`MLP(261,256,128)`得\(v_i\)。

额外几何需求注意力：每UAV只对所属13终端query128，terminal key/value由64投影到128。score为点积除sqrt(128)，加一个`MLP(6,32,1)`偏置；其6维输入固定为[relative_x/map_size, relative_y/map_size, distance/map_size, L_1e9(high upload cycles), L_1e9(low upload cycles), L_1e6(total upload bits)]。context与v_i拼接后`MLP(256,128,128)`。

三UAV再做两层128宽4-head self-attention，FF256、残差LayerNorm、无dropout。全局g由`MLP([mean(v),max(v),elapsed,remaining],256,128)`得到，输入258维。

压缩发生在可训练编码之后；原始报告和完整次序仍可复核。不能声称这种有限宽网络永不产生表示混淆。

### 6.5 下层heads

r=(priority,source)，\(g_r=MLP([g,v_{source},Emb(priority)_{16},ready\ bits/cycles/count],256,128)\)，输入275。

对每支持集M：support logit网络输入`[g_r, sum(v_j,j∈M), mean(v_j,j∉M), |M|/3, source∈M]`（386维），MLP(386,128,1)。空补集均值=0。

对j∈M：concentration raw网络输入`[g_r,v_j,mean(v_l,l∈M),|M|/3,j==source]`（386维），MLP(386,128,1)。两种head分别共享跨r/M/j参数。kappa=1+softplus(raw)。初始logit输出层权重/偏置=0；concentration输出层权重=0，bias=log(exp(1)-1)，即初始kappa=2。

Q1/Q2各将`[v0,v1,v2,g]`(512)与执行比例18、support indicator18拼接（548），MLP(548,256,256,1)，两个Q独立。indicator不对连续比例求导，支持集变化由离散梯度处理。

### 6.6 上层heads

每UAV的v_i通过共享128→64→2 mean head和logstd head。mean末层初始全0。logstd=-2.5+2.5*tanh(x)，初值ln(0.15)，范围(-5,0)。logstd末层权重为0，bias=atanh((ln(0.15)+2.5)/2.5)。

\(z_i=\mu_i+\sigma_i\epsilon_i, u_i=\tanh z_i\)。三UAV的u组成6维请求，送入**不变的**`start_motion`。PPO计算这6维请求的概率，不把安全缩放后的位移硬塞回原高斯logprob。

Value encoder独立，以全局g接MLP(128,128,1)。所有非特殊线性层用orthogonal gain=sqrt(2)，bias=0；actor末层按上文初始化，Q/V末层gain=1。所有初始化之前设置各网络独立seed。每个GRU按PyTorch的reset/update/new门顺序，将input-to-hidden矩阵分成三块，各块Xavier-uniform(gain=1)；hidden-to-hidden分成三块，各块orthogonal(gain=1)；所有GRU bias为0。priority embedding用Normal(0,0.02)。其他MLP中只有最后一层无ReLU；LayerNorm初值weight=1,bias=0。

---

## 7. 下层动作：保留连续分片，移除欧氏投影训练路径

### 7.1 固定支持集顺序

0:{0}, 1:{1}, 2:{2}, 3:{0,1}, 4:{0,2}, 5:{1,2}, 6:{0,1,2}。

每路由行r选择：
\[
M_r\sim\operatorname{Cat}(q_r),\qquad
p_{r,M_r}\sim\operatorname{Dirichlet}(\kappa_{r,M_r})\quad(|M_r|>1).
\]

\[
a_{r,j}=\begin{cases}p_{r,j},&j\in M_r\\0,&j\notin M_r.\end{cases}
\]

单点集直接输出对应one-hot，连续log-density为0。只在选中的节点之间分配。六行条件独立，但都基于全局图；Q评价完整六行动作及其耦合。

这覆盖三角形内部、三条边和三个顶点，不把动作离散成10%网格。它不强迫每个任务创建三个分片，也没有“先全正再阈值归零”。

无ready时仍必须生成六行：本周期可能有新上传完成任务，旧物理逻辑也会应用该周期路由；不能只凭初始ready=0 mask掉一行。

### 7.2 为什么不是普通softmax替换

连续部分没有Euclidean projection，因此没有由该projection顶点区域造成的整行局部平坦区；更改“哪些节点参与”由categorical support负责。连续Dirichlet也可能有数值困难或小梯度，离散概率也可能集中；这些不被谎称为数学上彻底不存在。

精确0来自未选择的支持，不依赖连续变量趋近无穷。选中节点可以有小正比例，这是合法控制空间的一部分；是否值得分给它由策略学习和性能测试决定。

### 7.3 浮点合同

分布、log-density及环境执行数组用float64；神经嵌入用float32。分布计算和抽样固定在CPU，actor参数通过保留autograd的`.to(device="cpu", dtype=torch.float64)`传入；Q头所需动作按第7.3节转换。不得通过numpy转换切断训练梯度。传入Q时记录float64→float32误差，要求最大绝对误差≤1e-7且不能把支持内正比例变成0。若数值不满足，停下报告，不静默阈值、重采样或修复。

`micro_step`要求每行和在1e-10内为1；Dirichlet样本直接传入，不重投影、不再次归一化。Q、replay、环境都使用同一执行比例与support记录。

---

## 8. 混合分布密度与正则：不是S330 Anchor

不同维数的面不能随便混用一个“18维Gaussian entropy”。定义固定参考概率测度：
\[
\nu_r(M,p)=\frac17\,\operatorname{Dirichlet}_{|M|}(p;\mathbf1),\quad
\nu=\prod_r\nu_r.
\]

支持M的大小k，均匀Dirichlet密度为\(\Gamma(k)\)。定义：
\[
\ell_r=\log q_r(M)+\log7+\log\operatorname{Dir}(p;\kappa_{r,M})-\log\Gamma(k),
\quad \ell=\sum_r\ell_r.
\]

k=1时continuous项=0。这是执行动作分布相对明确参考测度的log-density；其期望是KL(π||ν)。ν是固定均匀参考测度，不是旧模型、教师或“好锚点”。

\[
\log\operatorname{Dir}(p;\kappa)=\log\Gamma(\sum_j\kappa_j)-\sum_j\log\Gamma(\kappa_j)+\sum_j(\kappa_j-1)\log p_j.
\]

PyTorch用`Dirichlet.rsample()`和`log_prob()`；categorical不能用argmax＋普通反传冒充可导。参考代码给出了逐行枚举的具体梯度实现。

温度不自适应为旧target_entropy=-18；使用预先固定退火：

| 下层训练阶段 | alpha（已缩放奖励单位） |
|---|---:|
| 下层预训练1—100轮 | .005 |
| 101—200轮 | .0025 |
| 201—300轮 | .0005 |
| 交替cycle1—5下层块 | .0005 |
| cycle6—10下层块 | **0** |

alpha是全局阶段配置，不随state变化；预训练的三个固定区间内保持不变。最后50个下层训练episode不含熵正则，目标回到原物理成本。早期有显式探索正则，不能声称整个训练过程一直精确优化无正则目标。

---

## 9. 奖励、Bellman与episode边界

每路由帧：
\[
r_k=-\frac1{100}\sum_{i=0}^{4}c_{5k+i}.
\]

每宏步：
\[
R_n=\sum_{k=10n}^{10n+9}r_k.
\]

没有帧内0.99折扣；没有重复折扣；没有另加queue、几何、route、tail奖励。PPO不接收SAC entropy或loss。

\[
\sum_{n=0}^{179}R_n=\sum_{k=0}^{1799}r_k=-900\,J_{episode}/100.
\]

该恒等式是必须逐episode通过的单元/集成测试。

令d_k=1仅当服务已完成原900秒；代码中称`terminated=True`。技术中断、OOM或人为停止不作为成功终止样本。终点不再注入900秒处下一帧任务，不调用旧collector的终点prime，不产生跨reset转移。

lower Q：
\[
V^\pi(\mathcal H)=\mathbb E_{A\sim\pi}[Q^\pi(\mathcal H,A)-\alpha\ell(A|\mathcal H)],
\]
\[
Q^\pi(\mathcal H_k,A_k)=\mathbb E[r_k+(1-d_k)V^\pi(\mathcal H_{k+1})].
\]

上层未来行为是固定于当前下层学习块的随机PPO策略。下一状态必须包含真实已提交的下一宏动作（若跨宏边界），不能只换位置不换macro phase。

有限时域的真实递推合法，不等于神经网络近似有收敛保证。目标函数从旧discounted continuing改为finite undiscounted必须出现在Methods及消融配置差异里。

---

## 10. SAC的更新公式与精确梯度路径

### 10.1 critic

批次权重w见第12节，下一动作由当前actor采样，目标Q不求导：
\[
y=r+(1-d)\{\min(Q_{\bar\psi_1},Q_{\bar\psi_2})(\mathcal H',A')-\alpha\ell_\theta(A'|\mathcal H')\}.
\]

\[
L_Q=\frac1B\sum_b w_b\sum_{j=1}^{2}(Q_{\psi_j}(\mathcal H_b,A_b)-\operatorname{sg}(y_b))^2.
\]

终点直接令y=r，不先算终点无效策略再用0相乘。每事件2次独立抽batch与target噪声的critic拟合；actor每2事件一次；target每事件一次\(\bar\psi\leftarrow .995\bar\psi+.005\psi\)。

### 10.2 真正的actor目标

\[
L_\pi=\mathbb E_{\mathcal H\sim D,A\sim\pi_\theta}
[\alpha\ell_\theta(A|\mathcal H)-\min_jQ_{\psi_j}(\mathcal H,A)].
\]

Q参数冻结，但Q对执行比例的梯度保留。actor、两个Q编码器不共享参数。

### 10.3 六行×七支持集的可实现梯度

对每个r，其他五行动作从当前策略采样并detach，记\(A_{-r}^{sg}\)。枚举M=0..6，各采一个Dirichlet重参数样本p（k1无连续采样）：
\[
\widetilde L_r=\sum_Mq_r(M)
\left[\alpha\ell_r(M,p)-Q_{\min}(\mathcal H,A_{-r}^{sg},a_r(M,p))\right].
\]

优化surrogate：
\[
\widetilde L=\frac1B\sum_b w_b\sum_{r=1}^{6}\widetilde L_r.
\]

**对r求和，不除以6。** q_r的梯度显式包含支持集选择；连续参数通过rsample和Q反传。其他行正则项对q_r的期望梯度为0，因此可省略。其他行在这个r项里detach，其各自参数梯度由自己的r项给出。在条件独立策略与常规可交换微分条件下，该surrogate的期望梯度等于Lπ梯度。

surrogate数值重复包含6次Q，**不是可报告的完整actor objective值**。日志另用完整一次动作样本计算Lπ。不能用surrogate与旧actor_loss绝对值比较“变好了”。

每state42种candidate Q头调用；冻结的Q状态图编码每state只算一次，不能把完整图网络重复42次。无需枚举7^6个支持组合。条件动作采样噪声与环境采样噪声分流。

### 10.4 优化器

actor lr=1e-4，critic lr=3e-4；Adam betas(.9,.999),eps1e-8；全参数L2梯度裁剪1。不裁剪物理reward，不裁剪Q值，不按评估表现修改LR。batch128，允许16为microbatch累积完整均值后只step一次。

这些参数是设计初值，不声称已解决Q误估。每个Q分别记录残差、目标均值/方差、gradient norm和实际step；两个头的分歧不能被min掩盖。

---

## 11. PPO更新：完整宏时域回报，不缩成短视局部奖励

上层batch由同一下层策略版本的5个完整episode构成：900个宏转移。gamma=1，lambda=1。因此：
\[
G_n=\sum_{j=n}^{179}R_j,\qquad \hat A_n=G_n-V_{old}(\mathcal H_n^H).
\]

每episode独立反向累计，不跨reset。整900样本batch做一次advantage中心化/标准差归一化，eps1e-8。

\[
L_{PPO}=-\mathbb E\left[\min(\rho_n\hat A_n,
\operatorname{clip}(\rho_n,.85,1.15)\hat A_n)\right],
\quad \rho_n=e^{\log\pi_\theta(u_n|\mathcal H_n)-\log\pi_{old}(u_n|\mathcal H_n)}.
\]

\(L_V=\frac12\operatorname{mean}(V-G)^2\)。actor/value分别Adam，lr1e-4/3e-4；6 epochs；minibatch150；grad norm1；entropy coefficient0。PPO的随机探索通过策略标准差存在，但没有额外几何或熵reward。

保存raw tanh-request及pre-tanh sample，logprob采用稳定tanh修正。均值/方差变化后的logprob用同一已采raw request求值，不用实际安全位移替代。

每epoch末对全900状态计算old→new的6维Gaussian解析KL（tanh为双射，KL等价），平均>0.02则恢复actor与actor optimizer到该epoch开始，停止剩余actor epochs；value仍完成原定6epochs。被拒绝更新及回滚次数单列。guard只限制近端变化，不提供单调性能保证。

Monte Carlo优势方差可能较大；这是取消折扣短时域偏差的明确取舍，不在运行中悄悄改回lambda=.95或gamma=.99。

---

## 12. 回放：上层版本隔离＋明确的罕见状态采样

### 12.1 存储单元

存完整结束的episode，包括原始报告/关联/队列顺序、原始到达序列、执行A/support、r/d、真实next-report、upper_policy_version及阶段身份。不保存或训练旧tanh分数。最多55个完整episode=99000转移。

逻辑schema保存完整信息，物理存储可用唯一实体表＋时间差分＋无损数组压缩，要求解码后报告逐字段一致；不把100000份重复Python对象直接堆在内存作为唯一实现。任何数据量/队列截断不属于无损优化。

正在收集的episode先暂存，完成所有本轮更新触发点之后才加入池。episode中途失败不伪装成正常终点、不加入完整池。

### 12.2 严格的上层条件

PPO参数/探索规则/hover-vs-moving模式发生变化后，进入下层块时清空旧replay；当前池只能有一个upper policy version。Q和actor网络参数可以继续继承本次训练所得状态，清空的是旧条件数据，不是抹去全部已学知识。

上层块不写SAC replay、不更新SAC/target/温度。PPO batch也不得跨SAC版本。

### 12.3 分层与重要性权重

每batch等概率有放回选4个episode，各抽32转移。episode长度统一1800，因此与均匀episode内抽样相组合的目标是均匀转移分布。

在被选episode中，C表示“low ready总cycles>0且任意目的节点low CPU/F≥5秒”的转移集合。n=1800,n_C=|C|：
\[
q_i=0.9/n+0.1\,\mathbf1(i\in C)/n_C,
\qquad w_i=(1/n)/q_i.
\]

n_C=0时q_i=1/n,w_i=1。按混合方式采样，权重同时用于critic和actor的state平均，不自归一化。这样增加罕见样本出现频率，但不把“20%拥堵权重”偷偷写成另一个目标。

该采样不创造没访问过的状态，也不保证critic一定学对。记录真实n_C、抽样比例、权重范围、按层残差。

---

## 13. 唯一训练状态机：600轮从零，没有好检查点救援

### 13.1 初始化

五个预登记master seed：4101—4105，4101是开发pilot。每个seed初始化全部网络、targets、Adam、缓存、计数、RNG。target初值复制各自在线Q；optimizer state为空；没有load-old-checkpoint参数。

各阶段连续继承**这一次**训练的最终学习状态。所有阶段切换用固定预算末端，不用开发集挑出的best模型，不根据结果手工回退到某轮。

### 13.2 阶段A：300轮下层预训练

上层执行恒0请求、无PPO采样或更新。每episode900秒、路由正常。当前上层版本='hover-0'。

前2轮只允许critic准备，不更新actor；因只使用完整episode池，第1轮无可用池，不做学习，第2轮可更新critic。从第3轮开始正常actor日程。

每20个下层决策周期一次事件；每事件2critic fits，actor隔2事件。事件计数持续记录，缺数据的触发计为skipped，不冒充更新。

### 13.3 阶段B：100轮上层单训

冻结A300的整个下层actor/critic/target/optimizer；固定下层仍按当前分布随机采样执行。PPO从本次未训练的初始化开始，每5episode批次学习；只训练PPO。

不自动把SAC改成确定性版本。B100末尾进入交替阶段，不声明“PPO此时必定有效”。

### 13.4 阶段C：10个交替cycle，各20轮

每cycle顺序固定为：

* U块10轮：冻结SAC，每5轮收集900宏步并完成PPO更新，两批；不写SAC replay。
* L块10轮：冻结该U块末尾PPO及其探索设置；清空上层版本不同的旧replay；前2轮不更新SAC actor，critic按可用池更新；其余8轮正常更新下层。

第6—10 cycle的alpha=0。最终是L块结束，所以最终SAC至少已在最终PPO条件下适应，而不是最后又更新一次PPO后未经适应就停。

总数：300+100+10*(10+10)=600episode，5400000物理步/seed。这个预算没有包括验证与测试。它不是20分钟运行承诺。

在无技术失败、固定触发边界定义下：SAC参与学习的episode400，PPO-only episode200。预期SAC有35010个有效critic事件（每个新上层版本首个完整episode的90事件跳过，共11次上下文建立）；70020个critic fits；17010个actor updates。PPO有40个900宏步batch。实现必须逐计数核对；若collector不同，先修正实现到此时序，不能事后篡改预期数。

### 13.5 开发评价

固定global episode节点300、400、440、480、520、560、600；300仅验证hover SAC，其他做moving/hover。6开发场景×2策略随机流。评价不写回放，不改变任何训练RNG，评估后按暂停处继续。

主方法最终报告固定600轮。可另报开发集best（400/440/480/520/560/600中cost最低、并列较早），但不可用其替代末轮、不可用它重启训练或当Anchor。

---

## 14. Collector的明确伪代码

```python
reset_scene_and_history()
for n in range(180):
    # 上层观察早于本宏步新到达；保持原物理事件顺序
    high_report = export_report(pre_arrival=True)
    u, high_logprob = upper_policy(high_report, observed_history)
    start_motion_unchanged(u)
    R = 0
    for j in range(10):
        prepare_frame_once()      # 只注入此时已经发生的外生到达
        report = export_report(pre_arrival=False)
        A, support, ell = lower_policy(report, observed_arrivals)
        readings = [env.micro_step(A) for _ in range(5)]
        r = -sum(x['cost'] for x in readings) / 100
        R += r
        # 保存pending；next report须等下一次真实macro commit/prepare
        stage_pending_low(report, A, support, r)
        if original_900s_reached:
            finish_pending_low(done=True, next_report=terminal_report_without_new_arrivals)
        else:
            # 下次循环/宏边界产生真实后继后，再封存该transition
            pass
    store_high_transition(R, done=(n == 179))
# 此处不得再_prime_macro或prepare900秒的新任务
freeze_episode_metrics()
commit_completed_episode_to_replay_only_if_lower_training_block()
```

实现要对pending下一状态有唯一处理者，不能省略宏边界上的新PPO动作、不能重复prepare、不能用reset初始报告收尾。下层学习事件在当前episode可获得的已经封存transition后触发，但它采样的池只包含此前结束episode；末轮把新episode加入池的动作发生在最后一个触发点之后，保证第13节计数。

---

## 15. 推理与评价策略：不再把两种策略混称为一个

### 15.1 主评价

采用冻结后的随机PPO和随机hybrid-SAC，与训练中的策略分布族一致；alpha只属于训练目标，不直接修改推理动作。每场景4组明确策略seed；同一比较中的外部任务事件完全相同，策略噪声按语义分流。

同样的seed、平台、实现应可重新产生相同随机执行轨迹。随机执行不是不可复现。

### 15.2 确定性补充

PPO取tanh(mean)。SAC每行取logit最大支持，支持内用Dirichlet mean κ/sumκ；平分时按固定支持序号最小。它不是完整混合分布的均值，不叫“同一随机策略的真实期望动作”。

与旧确定性641/704/710等成绩比较，必须在同一面板、同一确定性口径下进行；不能拿新版4随机流平均和旧单次确定性值直接算优越性。

### 15.3 空和异常

无任务时仍按上述策略运行，悬停能耗不删去。solver failure使用原可见fallback并计数；新RL数值异常不回退到某个隐形启发式掩盖，只记录技术失败。

---

## 16. 指标与父任务删失

对测试集合\(\mathcal E\)：
\[
\widehat J=\frac{\sum_e\sum_kc_{e,k}}{\sum_e T_e},\qquad
\widehat D_{pool}=1000\frac{10\sum_{e,j\in H,done}(T_j-b_j)+\sum_{e,j\in L,done}(T_j-b_j)}{10\sum_e N^{done}_{e,H}+\sum_e N^{done}_{e,L}}.
\]

同时输出每优先级arrived/completed/unfinished、completion rate、completed delay分布、截止未完成年龄分布；未完成任务的真正最终时延记为censored，不能填0或推测。

对初始无积压、在窗口内出生的任务，按原离散网格驻留定义检查：
\[
\sum_kH_k=\sum_jw_{p_j}\,[\min(T_j,T)-b_j].
\]

右侧对未完成取T，完成时刻采用原量化完成规则；若浮点网格存在已知舍入，使用整数tick计时。恒等式确保未完成父任务已在主成本里产生驻留代价。

5秒记录的队列面积只能标注采样近似；新版服务账本同时累加每0.1秒真实队列量，保留两种不同口径。实际分配cycles从fragment-created事件累加，不能由初始ready比例反推。

最后分片归因仅用于离线结果账本，不可进入决策报告。所有图表分train/dev/test、moving/hover、stochastic/deterministic和final/dev-best。

---

## 17. 从零可复现、场景划分及基线

### 17.1 Seed派生

使用`numpy.random.SeedSequence([2026092602, split_code, master_seed_component, stage, cycle, episode_or_scenario, purpose, stream_id])`生成32/64位seed。split_code: train10/dev20/test30；purpose: layout1/load2/arrival3/upper_sample4/lower_support5/lower_ratio6/replay7/init_actor8/init_Q1_9/init_Q2_10/init_upper11/init_value12。非采样重复的stream_id固定为0；开发策略流为1—2，测试策略流为1—4。stage编号A=1,B=2,C=3；C上层块cycle_field=2*c-1、下层块=2*c，其他阶段cycle_field=0；块内episode从1起，初始化的stage/cycle/episode均为0。

训练和初始化的master_seed_component使用实际训练seed。开发/测试的外部layout/load/arrival使用master_seed_component=0、stage=cycle=0及固定scenario编号，因此五个模型共享同一面板；该面板不随训练seed改变。策略采样流仍使用对应训练seed和固定stream_id。每个算法对相同master seed使用同一外部训练清单，不用算法名称改外生seed。

PyTorch seed取派生uint64模(2^63-1)。禁止Python内置hash派生seed。若原scene内部还做子流拆分，保存其输入种子和内部调用原样，不改发生器。

Dirichlet.rsample不接受独立generator参数：在CPU `torch.random.fork_rng(devices=[])` 中设置专属lower_ratio流状态，抽样后保存更新后的该流状态，退出时恢复全局CPU RNG。categorical使用独立lower_support流；可用torch.multinomial(generator=...)或同样的fork_rng隔离。上层Gaussian噪声用专属upper_sample generator。所有流状态纳入checkpoint，不每次重新设成同一seed。参考sample_joint为简明验证使用全局RNG；生产实现必须按这里分流，不能直接共享训练与环境的global RNG。actor更新的support/ratio抽样使用这两个purpose加stream_id=100，target用200，在线行动用0，评价用规定1—4；数据shuffle用replay purpose和stream_id=300，上层minibatchshuffle用400。

配对算法共用外部layout/load/arrival清单；网络初始化与动作/回放流独立。评价独立进程或完整隔离并还原所有训练RNG。初始参数hash必须验证同seed同版本一致、不同seed不同，不是只改文件名。

### 17.2 正式测试面板

20个同分布场景，先封存manifest再看任何模型成绩；排除与已知训练/开发的完整seed tuple重复。不能按是否拥堵、是否有利于V2筛选。5训练seed×20场景×4策略seed×2模式=800次900秒评价，7200000物理步，仅此方法。额外基线预算单列，不能隐藏。

### 17.3 必需比较层次

1. **部署净收益控制**：同一最终SAC，分别使用最终PPO、Hover、因果需求质心。它们是固定模型控制器替换，不冒称重新训练过的独立方法。
2. **主基线**：PPO+原CVX、PPO+Local、PPO+原Random，各训练自己的PPO；同一信息可用性、同一600episode环境交互上限，原规则实现和fallback冻结。不能简单把V2的PPO搭上CVX再当成充分训练的PPO+CVX主基线。
3. **强下层基线**：Hover+从零SAC、因果需求质心+从零SAC，明确各自训练预算；旧S250成绩只作历史参考，不能替代本次独立训练。
4. **新版关键消融**：完整V2 vs 同表示/同动作/同目标的同时更新训练；完整V2 vs 移除队列与父子图（旧聚合语义）；完整V2 vs旧投影动作/原raw熵定义。每次消融只切一个设计开关，所有边界改变显式记录。旧投影对照保持相同finite900目标与回报尺度，仅将下层分布切回tanh Gaussian+projection，并用同一alpha日程下的对应raw密度；明确它同时改变动作分布及其自然概率测度，不能把差异只归因于Jacobian。

先完成完整V2的一个600轮pilot，再锁定方法跑独立训练；不用等待所有消融才允许实现新版。主基线与消融也不能在论文提交时被省略成没有执行过的表格。

因果需求质心规则固定：对所属终端m，\(w_m=\sum_p w_p[C^{upload}_{m,p}+5\,\widehat\lambda^{cycles,10s}_{m,p}]\)；目标为按w_m加权位置中心，w和为0则保持当前；移动请求限制到原10米并经同一安全映射。只能用已观察10秒实际到达估计率，不使用生成器真实lambda。

### 17.4 统计单位

先给每训练seed的所有成绩，再报seed均值/标准差、成功数、配对差。policy-seed和场景不是独立训练seed。置信区间用10,000次分层配对bootstrap：先有放回抽训练seed，再抽同一套场景索引；同一次抽样中保留各方法配对、保留场景的4策略流整体。5训练seed仍有较大统计不确定性，不能声称普遍保证。

---

## 18. 计算预算与开发顺序

固定执行设备：物理环境与分布抽样CPU；神经网络训练/推理cuda:0，CPU线程10。使用用户原Python/PyTorch/CUDA版本，不自动升级；关闭TF32，启用torch deterministic algorithms，CUBLAS_WORKSPACE_CONFIG=:4096:8。设备不存在或算子不能满足确定性则在接口阶段停止并给出具体原因，不静默切换设备后冒称同一协议。数学参考的14项CPU测试不代表这些CUDA集成条件已经满足。

一套V2完整训练600episode=540万物理步；5seed=2700万物理步，不含评价。更复杂编码器的实际墙钟时间尚未测得，不沿用旧代码13分钟/20分钟估计。

开发交付顺序固定，不再先查旧模型因果：

1. 绑定原物理scene及源码；实现ReportV2/adapter/新collector，用已知动作轨迹做一致性测试。
2. 实现图与到达历史编码器、hybrid动作；通过动作/密度/信息测试。
3. 实现gamma1 SAC/PPO及状态机；完成短的非科研接口烟测和实际吞吐/内存测量。
4. 跑seed4101完整600轮候选；同一面板给出主成本与部署净收益。
5. 满足首轮目标后锁定配置，执行4102—4105与封存测试、主基线和关键消融。若根据4101结果修改了任何方法/超参数，旧4101仅作开发记录，最终锁定版须重新从零执行4101—4105；不能把不同版本的五次结果拼成五seed。

若pilot失败，可以依据新系统结果做下一版本，但必须改schema、写差异、不继续把不断调参称为已封存的最终测试。一次设计不保证成功；失败也必须交付代码、模型、记录和明确结果。

---

## 19. 完整checkpoint与运行目录

新checkpoint保存：全部网络/targets/Adam；SAC/PPO版本号与阶段counter；alpha日程位置；完成episode回放及到达历史；PPO尚未消费batch；所有RNG流；物理scene及SHA；报告/动作schema；代码文件SHA；训练与开发manifest；pending transition；运行是否在安全边界；累计预算。

支持任意物理tick恢复需要保存完整env状态；第一版承诺的resume边界限定为**episode结束、没有pending转移、优化器步完成的原子checkpoint**。中途断电只从最近完成episode重启并登记丢弃工作，不能称为逐物理步精确resume。

开发节点eval快照和完成后rolling均有明确`pending_evaluation`；恢复先核对，禁止重复评价。采用临时文件写入＋fsync＋os.replace。所有路径相对project root，root由唯一CLI参数解析；继续支持用户原C入口，不按移动后的D镜像另建虚假root。

接口命令合同（待实现后才能运行）：

```text
python -m hrl_v2.run --config configs/v2_1_0.json --seed 4101 --fresh
python -m hrl_v2.run --resume <new_v2_completed_episode_checkpoint>
python -m hrl_v2.evaluate --checkpoint <new_v2_checkpoint> --manifest <sealed_test.json> --mode paired --policy stochastic
```

`--fresh`和`--resume`互斥；旧schema checkpoint一律拒绝。run不会在import时启动训练，不自动安装依赖、不自动升级torch。

---

## 20. 必须完成的测试，不允许用训练成绩替代

### T01 物理不变

用相同外部事件和预先录制的合法上层请求/执行比例，旧内核与V2 adapter分别运行同一900秒；逐步cost/energy/position/queue/conservation与父任务完成记录一致。只允许报告与学习端差异，不能有服务时序差异。

### T02 因果信息

两实例保持至时刻t全部合法观测相同，但替换未来arrival流/隐藏模板名称；ReportV2和actor输入必须完全相同。控制器不得有env/scene/完整ledger引用。

### T03 队列及父子关联保真

构造总cycles相同但FIFO次序不同的两个状态，原始报告/队列图必须不同；构造其他任务量相同但siblings关系不同的状态，图必须不同。神经输出不要求手工规定方向，也不把不同输出当作有效性的证明。

### T04 动作空间与密度

七种支持的exact-zero、row-sum、positive-interior、k1 log-density、k2/k3 uniform normalization、重算logprob一致性、连续梯度、categorical支持梯度。附件14项数学参考测试覆盖其中一部分，但仍需本地接入测试。

### T05 actor梯度

在独立小规模玩具critic上，逐行枚举surrogate梯度与可枚举联合策略的精确期望梯度/足量Monte Carlo一致到预设容差；检查其他行detach及没有除6。必须测试非可加Q，不能仅用线性可分离toy就声称验证任意耦合。

### T06 回报/边界

逐帧、宏步、episode reward与物理cost恒等式；d只在900秒为1；终点无新任务/新高层动作；终点target=r；不能跨reset拼next state。

### T07 单层冻结

U块SAC全部参数/targets/Adam/replay不变；L块PPO参数/Adam/探索不变。不同网络不得共享可训练encoder。

### T08 版本与采样

SAC pool不能含其他PPO版本；PPO batch只有一个SAC版本；分层重要性权重在合成有限数组上恢复uniform估计。

### T09 随机性和恢复

独立初始化hash、分流RNG、评价前后训练RNG、episode边界完整恢复；同平台同环境同seed重复短任务完全一致。跨PyTorch/CUDA版本不承诺bitwise复现。

### T10 结果诚实性

重新从原始父任务账本合池；未完成值标删失；train/dev/test和stochastic/deterministic分表；全部seed、失败、fallback与终点公开。

### T11 工程预算

典型/拥堵状态的报告大小、推理延迟、batch显存/内存；无静默top-k和episode筛除。若真实执行要求0.5秒内决策，报告目标机器上p95/p99，不用服务器平均值替代。未达时延目标先进行数学等价实现优化，模型删减必须升版本。

---

## 21. 全部旧问题如何结账

| 旧问题/限制 | V2的明确处理 | 可否在设计时宣布根除 |
|---|---|---|
| 控制器无稳定整体净收益 | 固定训练流程、同SAC悬停和强基线验收 | 否，必须实测 |
| 继续学习回退 | 更完整表示、版本隔离、交替训练、固定末端报告 | 否，不承诺单调改进 |
| 部分PPO轨迹变差 | UAV需求几何注意力＋整episode宏回报 | 否，实际位置收益验收 |
| 拥堵时序路由错误 | 全队列/关联图、历史、稀有状态采样 | 否，行为和服务结果验收 |
| 父任务被慢分片拖住 | 保留真实完成语义，显式报告siblings及队列次序 | 信息缺失可消除，拖尾物理现象不能删 |
| 聚合FIFO信息缺失 | 全item队列编码及图边，无top-k | 原始接口缺失可消除 |
| 隐藏业务/未来未知 | 因果历史GRU，不偷看真实mode | 不能从根本消除未知未来 |
| 下层投影平坦区 | 7-support＋Dirichlet直接比例，逐行枚举离散梯度 | 该特定projection机制可移除 |
| raw熵≠执行分配熵 | 定义执行混合分布相对ν的密度 | 概念/实现混淆可消除 |
| 两层同batch内共同变化 | U/L块完整冻结及版本断言 | 指定块内非平稳性可消除，跨块适应仍需学 |
| 旧PPO版本回放 | 切换版本即清池，仅当前上层 | 数据混用可消除；已学权重影响不谎称清零 |
| 关键状态较少 | 0.1混合抽样＋精确IS＋覆盖日志 | 抽样合同清楚，不保证所有稀有状态都访问 |
| Q排序分歧/误差 | 独立Q、当前上下文、终点/指标校准 | 不能保证Q永远正确 |
| 训练/物理评价时域不同 | finite900、gamma1、alpha末阶段0、剩余预算输入 | 指定目标的定义差异可消除 |
| 随机/确定性评价混淆 | sampled主策略、确定性另表 | 口径混淆可消除，性能差异仍可能存在 |
| 未完成任务被平均时延隐藏 | cost主指标＋删失＋完成率底线 | 错误统计可消除，不补造最终时延 |
| 手工挑S330修复 | fresh入口、阶段用固定末端、不加载旧权重 | 旧检查点依赖可消除 |
| 单seed伪装稳定 | 五seed、真实失败、配对测试 | 证据范围可明确，不能保证所有未来seed |
| 历史源码/路径混用 | 物理manifest、新namespace、相对路径、原子checkpoint | 指定工程混淆可测试消除 |
| 诊断无限扩展 | 门禁后实现完整pilot；不再查旧根因作为前置 | 工作流程约束 |

---

## 22. 论文措辞与实施约束

本方案是对现有系统提出的**新完整候选设计**，不是文献中的原样现成算法。图注意力、混合SAC、PPO和回放采样分别有方法基础，但组合在此UAV问题上的收益尚未证明。不能宣称这些组件本身都是原创。

保留既有坏结果和Anchor补充结果，但不以它们替代V2的从零实验。所有架构、gamma、alpha、初始化、报告额外开销假设和训练预算应在方法/附录写明。

### 主要方法来源（只用于方法依据，不为性能背书）

[SAC] Haarnoja et al., Soft Actor-Critic Algorithms and Applications. https://arxiv.org/abs/1812.05905
[PPO] Schulman et al., Proximal Policy Optimization Algorithms. https://arxiv.org/abs/1707.06347
[GAE] Schulman et al., High-Dimensional Continuous Control Using Generalized Advantage Estimation. https://arxiv.org/abs/1506.02438
[Hybrid] Delalleau et al., Discrete and Continuous Action Representation for Practical RL in Video Games. https://arxiv.org/abs/1912.11077
[Time] Pardo et al., Time Limits in Reinforcement Learning. https://arxiv.org/abs/1712.00378
[GAT] Veličković et al., Graph Attention Networks. https://arxiv.org/abs/1710.10903
[Memory] Hausknecht and Stone, Deep Recurrent Q-Learning for Partially Observable MDPs. https://arxiv.org/abs/1507.06527
[HRL] Nachum et al., Data-Efficient Hierarchical Reinforcement Learning. https://arxiv.org/abs/1805.08296
[Eval] Agarwal et al., Deep Reinforcement Learning at the Edge of the Statistical Precipice. https://arxiv.org/abs/2108.13264
[API] PyTorch 2.7 distributions. https://docs.pytorch.org/docs/2.7/distributions.html
[Repro] PyTorch 2.7 reproducibility. https://docs.pytorch.org/docs/2.7/notes/randomness.html

原代码依据与scene绑定见 `physical_source_binding.json`。标量配置见 `config_v2.json`。下层概率/梯度及时间目标的可执行参考见 `reference_hybrid_simplex.py`。它不是已经完成的UAV训练程序。
