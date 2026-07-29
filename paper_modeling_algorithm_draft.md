# 基于多时间尺度约束分层强化学习的多无人机移动边缘计算联合部署与任务调度

> **稿件状态说明**：本文档为《通信学报》风格的“系统建模—问题定义—算法设计—复杂度—仿真方案”主体初稿。全部方法描述以当前代码为依据。由于 `results/` 尚无正式多随机种子实验结果，所有定量结论均保留为“待实验填写”，未虚构任何性能数据。引言、相关工作和真实参考文献将在模型与实验闭环确认后补充。

## 主要符号

| 符号 | 含义 | 维度或单位 |
|---|---|---|
| \(\mathcal N=\{1,\ldots,N\}\) | 无人机集合 | \(N=3\) |
| \(\mathcal K=\{1,\ldots,K\}\) | 地面用户集合 | \(K=39\) |
| \(\mathcal P=\{\mathrm H,\mathrm L\}\) | 高、低优先级任务集合 | 2类 |
| \(e\) | 训练回合索引 | — |
| \(t\) | 宏观决策周期索引 | — |
| \(\tau\) | 宏观周期内资源帧索引 | — |
| \(l\) | 资源帧内物理积分步索引 | — |
| \(\Delta_{\mathrm M}\) | 宏观周期长度 | 10 s |
| \(\Delta_{\mathrm d}\) | 周期开始时的部署阶段长度 | 1 s |
| \(\Delta_{\mathrm f}\) | 资源帧长度 | 0.5 s |
| \(\delta\) | 物理积分步长 | 0.1 s |
| \(T_{\mathrm f}\) | 每个宏观周期包含的资源帧数 | \(\Delta_{\mathrm M}/\Delta_{\mathrm f}=20\) |
| \(\mathbf q_n^t\) | 周期 \(t\) 开始时无人机 \(n\) 的水平坐标 | m |
| \(\Delta\mathbf q_n^t\) | 宏观层输出的无人机位移 | m |
| \(D_{\max}\) | 单周期最大部署距离 | 20 m |
| \(\mathbf u_k^{t,\tau,l}\) | 用户 \(k\) 的水平坐标 | m |
| \(H\) | 无人机固定飞行高度 | 50 m |
| \(A_{k,p}^{t,\tau}\) | 用户 \(k\) 在资源帧内产生的 \(p\) 类任务量 | Mbit |
| \(Q_{n,p}^{\mathrm{loc}}\) | 无人机 \(n\) 的本地计算队列 | Mbit |
| \(Q_{n,j,p}^{\mathrm{tx}}\) | 从无人机 \(n\) 转发至 \(j\) 的传输队列 | Mbit |
| \(\rho_{n,j,p}^{t,\tau}\) | 源无人机 \(n\) 的 \(p\) 类业务分配给 \(j\) 的比例 | \([0,1]\) |
| \(R_{k,n}^{\mathrm{UG}}\) | 用户 \(k\) 至无人机 \(n\) 的上行速率 | Mbit/s |
| \(R_{n,j}^{\mathrm{UU}}\) | 无人机 \(n\) 至 \(j\) 的机间速率 | Mbit/s |
| \(f_{n,p}^{t,\tau,l}\) | 无人机 \(n\) 分配给 \(p\) 类任务的CPU频率 | cycle/s |
| \(F_n^{\max}\) | 无人机 \(n\) 的最大CPU频率 | cycle/s |
| \(C\) | 每比特任务所需CPU周期数 | cycle/bit |
| \(E_{n}^{t,\tau,l}\) | 无人机单物理步电池能耗 | J |
| \(P_n^{\max}\) | 无人机长期平均功率预算 | W |
| \(C_{\mathrm D}^{t,\tau,l}\) | 加权积压压力，即时延代理 | 无量纲归一值 |
| \(g_n^{t,\tau,l}\) | 第 \(n\) 架无人机的归一化功率约束代价 | 无量纲 |
| \(\lambda_n\) | 第 \(n\) 架无人机功率约束的拉格朗日乘子 | 非负 |
| \(\mathbf s_{\mathrm m}\)、\(\mathbf a_{\mathrm m}\) | 微观层状态、动作 | — |
| \(\mathbf s_{\mathrm M}\)、\(\mathbf a_{\mathrm M}\) | 宏观层状态、动作 | — |
| \(\gamma_{\mathrm m}\)、\(\gamma_{\mathrm M}\) | 微观层、宏观层折扣因子 | \((0,1)\) |

# 1 系统模型

## 1.1 网络架构与多时间尺度

考虑一个覆盖 \(1000\,\mathrm m\times1000\,\mathrm m\) 区域的多无人机移动边缘计算系统。系统包含 \(N\) 架固定高度飞行的无人机和 \(K\) 个独立移动的地面用户。每架无人机既是用户接入节点，也是具有有限计算能力和电池功率预算的边缘计算节点。用户按照预先确定的服务区域归属于源无人机；任务上传至源无人机后，可以在源无人机本地执行，也可以经无人机间链路转发至其他无人机执行。

系统在三个嵌套时间粒度上演化：

1. 物理积分步长为 \(\delta=0.1\) s，用于更新用户位置、队列服务和能耗；
2. 微观资源帧长度为 \(\Delta_{\mathrm f}=0.5\) s，同一个任务路由动作在连续5个物理步内保持不变；
3. 宏观周期长度为 \(\Delta_{\mathrm M}=10\) s，每个周期开始时先执行长度为 \(\Delta_{\mathrm d}=1\) s的无人机部署，随后进入9 s服务阶段。

因此，一个宏观周期包含20个资源帧，其中前2帧属于部署阶段、后18帧属于服务阶段。正式协议设每个训练回合包含30个宏观周期。部署阶段暂停通信与计算服务；正式配置允许外生任务继续到达，从而使移动决策的机会成本通过队列积压进入后续回报。

![图1 多无人机MEC系统模型](figures/system_model.pdf)

**图1绘制要求**：用三架无人机、三个用户区域和若干异构终端表示系统；实线表示用户—无人机上行链路，虚线表示无人机间转发链路；在无人机旁标注CPU和功率预算；以两种颜色区分高、低优先级队列；在图下方增加“宏观部署—微观路由—物理服务”的时间轴。该图不显示神经网络内部结构。

## 1.2 用户分布、移动与任务到达

用户集合按照服务区域划分为互不重叠的子集 \(\{\mathcal K_n\}_{n\in\mathcal N}\)，满足

\[
\mathcal K_n\cap\mathcal K_j=\varnothing,\quad n\ne j,\qquad
\bigcup_{n\in\mathcal N}\mathcal K_n=\mathcal K.
\tag{1}
\]

用户采用Gauss-Markov移动模型。设用户 \(k\) 在资源帧边界的速度和航向分别为 \(v_k\) 和 \(\theta_k\)，则

\[
v_k^{+}
=
\operatorname{clip}\!\left[
\alpha_v v_k+(1-\alpha_v)\bar v+
\sqrt{1-\alpha_v^2}\,\varepsilon_{v,k},
0,v_{\max}
\right],
\tag{2}
\]

\[
\theta_k^{+}
=
\operatorname{wrap}\!\left[
\alpha_\theta\theta_k+(1-\alpha_\theta)\bar\theta+
\sqrt{1-\alpha_\theta^2}\,\varepsilon_{\theta,k}
\right],
\tag{3}
\]

其中，\(\bar v=0.5\) m/s，\(\bar\theta=\pi/4\)，\(\alpha_v=\alpha_\theta=0.5\)，速度与航向噪声标准差均为0.1，且 \(v_{\max}=3\) m/s。用户坐标以物理步长更新：

\[
\mathbf u_k^{+}
=
\Pi_{\mathcal A}
\left[
\mathbf u_k+
\delta v_k
\begin{bmatrix}
\cos\theta_k\\ \sin\theta_k
\end{bmatrix}
\right],
\tag{4}
\]

其中 \(\Pi_{\mathcal A}\) 表示区域边界反射映射。正式配置关闭人工圆形服务区限制，因此用户仅在整个方形区域边界发生反射。

系统包含传感器、控制器和视觉终端三种设备类型，每类设备具有不同的高、低优先级到达率缩放系数和任务大小。任务到达过程由代码明确实现为泊松过程：

\[
N_{k,p}^{t,\tau}
\sim
\operatorname{Poisson}
\left(\lambda_{k,p}\Delta_{\mathrm f}\right),
\tag{5}
\]

\[
A_{k,p}^{t,\tau}
=N_{k,p}^{t,\tau}L_{k,p},
\tag{6}
\]

其中 \(L_{k,p}\) 为单任务大小，单位为Mbit。各区域内用户到达率经过统一缩放，使区域期望业务负载与配置值一致。正式实验三个区域的目标负载为 \(12.5\)、\(5.5\) 和 \(3.5\) Mbit/s，再乘以全局负载系数1.15。式(5)只在0.5 s资源帧边界采样，避免把同一业务到达重复计入5个物理步。

## 1.3 无人机运动与部署能耗

无人机固定高度为 \(H\)，只优化水平坐标。宏观层输出归一化二维动作 \(\widetilde{\mathbf a}_n^t\in[-1,1]^2\)，首先投影到单位圆盘：

\[
\widehat{\mathbf a}_n^t
=
\frac{\widetilde{\mathbf a}_n^t}
{\max\{1,\|\widetilde{\mathbf a}_n^t\|_2\}},
\tag{7}
\]

再映射为物理位移：

\[
\Delta\mathbf q_n^t
=D_{\max}\widehat{\mathbf a}_n^t,\qquad
\mathbf q_n^{t+}
=\Pi_{\mathcal A}
\left(\mathbf q_n^t+\Delta\mathbf q_n^t\right).
\tag{8}
\]

这里 \(D_{\max}=20\) m。无人机在部署时间 \(\Delta_{\mathrm d}\) 内完成该位移，因此部署速度为

\[
v_n^t=\frac{\|\Delta\mathbf q_n^t\|_2}{\Delta_{\mathrm d}}.
\tag{9}
\]

飞行/悬停功率使用代码中的U形旋翼功率曲线：

\[
P_{\mathrm{fly}}(v)
=
U_1\!\left(1+\frac{3v^2}{v_{\mathrm{tip}}^2}\right)
+U_2\sqrt{\sqrt{U_3+\frac{v^4}{4}}-\frac{v^2}{2}}
+U_4v^3+U_5,
\tag{10}
\]

其中 \(U_1=78\)、\(U_2=22\)、\(U_3=265\)、\(U_4=0.0092\)、\(U_5=28\)、\(v_{\mathrm{tip}}=110\)。正式协议还采用加速瞬态能量下界：

\[
E_{n,\mathrm{tr}}^t
=\frac12m_{\mathrm U}(v_n^t)^2.
\tag{11}
\]

式(11)只表示从悬停加速到巡航速度所需动能的保守下界，不包含有限加速时间和减速损耗，因而不应解释为完整飞行动力学模型。

## 1.4 用户—无人机上行模型

设用户 \(k\in\mathcal K_n\) 与源无人机 \(n\) 的三维距离为

\[
d_{k,n}^{\mathrm{UG}}
=
\sqrt{\|\mathbf u_k-\mathbf q_n\|_2^2+H^2}.
\tag{12}
\]

根据环境实现，参考信道增益为 \(\beta_0=10^{-6}\)，上行发射功率为 \(p_{\mathrm{UG}}=0.1\) W，总带宽为 \(B_{\mathrm{UG}}=20\) MHz。设代码采用的距离衰落函数为 \(h_{k,n}^{\mathrm{UG}}\)，则上行速率统一写为

\[
R_{k,n}^{\mathrm{UG}}
=
B_{k,n}^{\mathrm{UG}}
\log_2\!\left(
1+\frac{p_{\mathrm{UG}}h_{k,n}^{\mathrm{UG}}}{\sigma^2}
\right).
\tag{13}
\]

其中 \(B_{k,n}^{\mathrm{UG}}\) 是用户共享上行带宽后的有效带宽。论文终稿必须将 `env.py` 中的具体带宽共享和路径损耗表达式逐项代入式(13)，不能额外引入代码中不存在的干扰项。用户上传能量属于地面终端能量：

\[
E_{k,n}^{\mathrm{UG}}
=p_{\mathrm{UG}}T_{k,n}^{\mathrm{UG}},
\tag{14}
\]

但逐无人机电池功率约束只统计无人机侧电池能量，不能把式(14)混入无人机电池预算。

## 1.5 无人机间链路与任务路由

无人机 \(n\) 与 \(j\) 的距离为

\[
d_{n,j}^{\mathrm{UU}}
=\|\mathbf q_n-\mathbf q_j\|_2.
\tag{15}
\]

设机间带宽为 \(B_{\mathrm{UU}}=3\) MHz、发射功率为 \(p_{\mathrm{UU}}=1\) W，则机间速率写为

\[
R_{n,j}^{\mathrm{UU}}
=
B_{n,j}^{\mathrm{UU}}
\log_2\!\left(
1+\frac{p_{\mathrm{UU}}h_{n,j}^{\mathrm{UU}}}{\sigma^2}
\right),\quad n\ne j.
\tag{16}
\]

微观层分别为高、低优先级任务生成两套 \(N\times N\) 路由矩阵。Actor原始动作维数为 \(2N^2\)，记为 \(z_{n,j,p}\)。动作首先按比例缩放，并对本地处理对应的对角元素加入偏置 \(b_{\mathrm{diag}}\)，随后逐行Softmax：

\[
\rho_{n,j,p}
=
\frac{\exp(\eta z_{n,j,p}+b_{\mathrm{diag}}\mathbb I[n=j])}
{\sum_{\ell\in\mathcal N}
\exp(\eta z_{n,\ell,p}+b_{\mathrm{diag}}\mathbb I[n=\ell])}.
\tag{17}
\]

正式实验设置 \(b_{\mathrm{diag}}=0\)，即不人为偏向本地计算；保留该符号只是为了完整描述动作映射接口，不能把对角偏置作为正式方法的组成部分。

由式(17)自动保证

\[
\rho_{n,j,p}\ge0,\qquad
\sum_{j\in\mathcal N}\rho_{n,j,p}=1.
\tag{18}
\]

当 \(j=n\) 时，任务保留在源无人机本地计算；当 \(j\ne n\) 时，任务先进入 \(Q_{n,j,p}^{\mathrm{tx}}\)，经机间链路传输后进入目标无人机 \(j\) 的计算队列。两类优先级使用独立路由动作，避免以单一比例强迫具有不同截止期和排队权重的任务采取相同路径。

## 1.6 双优先级队列与服务过程

系统维护本地计算队列 \(Q_{n,p}^{\mathrm{loc}}\) 和机间传输队列 \(Q_{n,j,p}^{\mathrm{tx}}\)。对每个到达源无人机 \(n\) 的任务量 \(A_{n,p}\)，按式(17)拆分：

\[
A_{n\rightarrow j,p}
=\rho_{n,j,p}A_{n,p}.
\tag{19}
\]

本地部分直接加入 \(Q_{n,p}^{\mathrm{loc}}\)，转发部分加入 \(Q_{n,j,p}^{\mathrm{tx}}\)。用 \(S_{n,p}^{\mathrm{cmp}}\) 表示本地计算服务量，\(S_{n,j,p}^{\mathrm{tx}}\) 表示机间传输服务量，则队列递推为

\[
Q_{n,j,p}^{\mathrm{tx},+}
=
\left[Q_{n,j,p}^{\mathrm{tx}}
-S_{n,j,p}^{\mathrm{tx}}\right]^+
+A_{n\rightarrow j,p},
\quad j\ne n,
\tag{20}
\]

\[
Q_{j,p}^{\mathrm{loc},+}
=
\left[Q_{j,p}^{\mathrm{loc}}
-S_{j,p}^{\mathrm{cmp}}\right]^+
+A_{j\rightarrow j,p}
+\sum_{n\ne j}S_{n,j,p}^{\mathrm{tx}}.
\tag{21}
\]

环境使用任务块跟踪器维护父任务的出生时间、分块、转发和完成时刻，因此可以统计已完成任务的真实端到端时延；但是训练目标主要采用积压压力代理，而不是直接以每个已完成任务的平均时延作为即时奖励。论文后文始终将训练量称为“时延代理”或“加权积压代价”，只有在实验指标来自任务跟踪器时才称为“真实完成时延”。

高优先级任务的权重为 \(w_{\mathrm H}=10\)，低优先级任务为 \(w_{\mathrm L}=1\)。高优先级队列在机间传输和CPU服务时具有抢占式优先级，但两类任务的实际服务量仍由链路容量和CPU频率共同限制。

## 1.7 计算资源分配与计算能耗

每架无人机最大CPU频率为 \(F_n^{\max}=10\) GHz（正式运行参数覆盖环境默认值），每比特任务需要 \(C=1000\) 个CPU周期。若物理步内为两类任务分配频率 \(f_{n,\mathrm H}\) 和 \(f_{n,\mathrm L}\)，则

\[
f_{n,\mathrm H}\ge0,\quad
f_{n,\mathrm L}\ge0,\quad
f_{n,\mathrm H}+f_{n,\mathrm L}\le F_n^{\max}.
\tag{22}
\]

相应计算服务量为

\[
S_{n,p}^{\mathrm{cmp}}
=
\min\!\left\{
Q_{n,p}^{\mathrm{loc}},
\frac{f_{n,p}\delta}{C\cdot10^6}
\right\}\quad\text{Mbit}.
\tag{23}
\]

正式配置采用 `priced_maxweight` CPU分配模式。对无人机 \(n\)，其局部频率分配求解

\[
\max_{f_{n,\mathrm H},f_{n,\mathrm L}}
\sum_{p\in\mathcal P}
w_p\frac{Q_{n,p}^{\mathrm{loc}}}{Q_n^{\mathrm{ref}}}
\frac{f_{n,p}}{F_n^{\max}}
-
\frac{\lambda_n}{P_n^{\max}}
\kappa(f_{n,\mathrm H}+f_{n,\mathrm L})^3,
\tag{24a}
\]

并满足式(22)及基于当前队列需求得到的频率上界。实现先按边际队列压力排序；对给定总频率，优先向边际收益更高的任务类分配，再比较分段驻点和边界候选。由于每架无人机只有高、低两类任务，该局部求解不需要通用凸优化器。CPU动态能耗为

\[
E_{n,\mathrm{cmp}}
=
\kappa\delta
\left(f_{n,\mathrm H}+f_{n,\mathrm L}\right)^3.
\tag{24b}
\]

其中 \(\kappa=10^{-28}\)。式(24b)对应代码中的 `total_cubic` 模型，高、低优先级频率先求和再取三次方。

## 1.8 统一代价与逐无人机功率约束

定义第 \(n\) 架无人机的加权积压压力

\[
\Psi_n
=
w_{\mathrm H}
\left(
Q_{n,\mathrm H}^{\mathrm{loc}}
+\sum_j Q_{n,j,\mathrm H}^{\mathrm{tx}}
\right)
+w_{\mathrm L}
\left(
Q_{n,\mathrm L}^{\mathrm{loc}}
+\sum_j Q_{n,j,\mathrm L}^{\mathrm{tx}}
\right).
\tag{25}
\]

系统即时延代理采用归一化、裁剪后的压力：

\[
C_{\mathrm D}
=
\operatorname{clip}
\left(
\frac{1}{N}
\sum_{n=1}^{N}
\frac{\Psi_n}{\Psi_{\mathrm{norm}}},
0,C_{\max}
\right),
\tag{26}
\]

其中 \(\Psi_{\mathrm{norm}}=8\)，\(C_{\max}=5\)。设无人机 \(n\) 在物理步内的完整电池能量为

\[
E_n
=E_{n,\mathrm{cmp}}+E_{n,\mathrm{UU}}
+E_{n,\mathrm{fly}}+E_{n,\mathrm{tr}},
\tag{27}
\]

其中部署能量按宏观周期分摊至对应物理步。定义归一化约束比

\[
g_n
=
\frac{E_n}{P_n^{\max}\delta}.
\tag{28}
\]

正式目标为在逐无人机长期平均功率约束下最小时延代理：

\[
\begin{aligned}
\mathbf P0:\quad
\min_{\{\mathbf q^t,\boldsymbol\rho^{t,\tau}\}}
&\quad
\limsup_{T\to\infty}
\frac1T
\mathbb E\!\left[
\sum_{\xi=0}^{T-1}C_{\mathrm D}^{\xi}
\right]\\
\mathrm{s.t.}\quad
&\mathrm{C1}:\
\mathbf q_n^t\in\mathcal A,\quad\forall n,t,\\
&\mathrm{C2}:\
\|\Delta\mathbf q_n^t\|_2\le D_{\max},\quad\forall n,t,\\
&\mathrm{C3}:\
\rho_{n,j,p}^{t,\tau}\ge0,\quad\forall n,j,p,t,\tau,\\
&\mathrm{C4}:\
\sum_j\rho_{n,j,p}^{t,\tau}=1,\quad\forall n,p,t,\tau,\\
&\mathrm{C5}:\
f_{n,\mathrm H}+f_{n,\mathrm L}\le F_n^{\max},\\
&\mathrm{C6}:\
\limsup_{T\to\infty}
\frac{1}{T\delta}
\mathbb E\!\left[\sum_{\xi=0}^{T-1}E_n^\xi\right]
\le P_n^{\max},\quad\forall n,\\
&\mathrm{C7}:\
\limsup_{T\to\infty}
\frac{\mathbb E[Q_{n,p}^{\mathrm{loc},T}]}{T}=0,\\
&\mathrm{C8}:\
\limsup_{T\to\infty}
\frac{\mathbb E[Q_{n,j,p}^{\mathrm{tx},T}]}{T}=0.
\end{aligned}
\tag{29}
\]

正式配置 \(P_n^{\max}=280\) W。C7—C8表达均值速率稳定目标，但当前算法没有提供严格稳定性证明，须通过负载可行性审计、队列轨迹和长时评估作经验验证。

# 2 双时间尺度问题分解

## 2.1 分解动机与耦合关系

问题P0同时包含慢变的位置控制和快变的任务路由。若在每个0.1 s物理步联合优化 \(2N\) 维位置和 \(2N^2\) 维路由，不仅动作维数高，而且会产生不现实的频繁飞行；若只优化位置而固定路由，则无法利用不同无人机的瞬时空闲计算能力。为此，将P0分解为宏观部署子问题和微观路由子问题。

该分解不是两个独立问题：宏观位置决定用户上行距离、机间拓扑、部署能耗和服务暂停代价；微观路由决定每架无人机的计算负载、机间流量、队列积压和电池能耗；一个宏观周期内真实产生的微观代价再聚合为宏观学习信号。

![图2 双时间尺度执行时序](figures/two_timescale_timeline.pdf)

## 2.2 微观约束马尔可夫决策过程

给定宏观位置 \(\mathbf q^t\)，微观层在每个资源帧选择两类路由矩阵。代码的 `pressure` 状态至少包含归一化本地队列、传输队列压力、链路/拓扑特征、CPU利用率、业务到达以及能量对偶状态。统一记为

\[
\mathbf s_{\mathrm m}^{t,\tau}
=
\mathcal F_{\mathrm m}
\left(
\mathbf Q^{t,\tau},
\mathbf A^{t,\tau},
\mathbf R^{\mathrm{UU},t},
\mathbf u^{t,\tau},
\mathbf q^t,
\boldsymbol\lambda
\right).
\tag{30}
\]

微观动作为

\[
\mathbf a_{\mathrm m}^{t,\tau}
=
\operatorname{vec}
\left(
\mathbf Z_{\mathrm H}^{t,\tau},
\mathbf Z_{\mathrm L}^{t,\tau}
\right)
\in[-1,1]^{2N^2},
\tag{31}
\]

再通过式(17)映射为可行路由矩阵。正式约束目标下的单步原始代价为

\[
c_{\mathrm m}^{t,\tau,l}=C_{\mathrm D}^{t,\tau,l},
\tag{32}
\]

拉格朗日代价为

\[
\mathcal L_{\mathrm m}^{t,\tau,l}
=
C_{\mathrm D}^{t,\tau,l}
+\sum_{n=1}^{N}\lambda_n
\left(g_n^{t,\tau,l}-1\right).
\tag{33}
\]

实现中的常数项可能被省略，因为 \(-\sum_n\lambda_n\) 不影响给定乘子下的动作排序。微观奖励为负代价并乘以缩放系数 \(\zeta_{\mathrm m}=10\)：

\[
r_{\mathrm m}^{t,\tau,l}
=-\zeta_{\mathrm m}\mathcal L_{\mathrm m}^{t,\tau,l}.
\tag{34}
\]

一个资源帧内动作保持不变，帧奖励为

\[
R_{\mathrm f}^{t,\tau}
=
\sum_{l=0}^{4}r_{\mathrm m}^{t,\tau,l}.
\tag{35}
\]

## 2.3 跨部署边界的SMDP转移

服务阶段最后一个微观动作的下一个可决策状态并非紧邻物理步，而是在下一宏观周期的1 s部署结束后出现。若简单把最后一帧与下一服务状态相连，会遗漏部署期间队列积压和折扣时长。代码因此构造SMDP边界转移：

\[
\widetilde R_{\mathrm f}
=
R_{\mathrm f}^{\mathrm{last}}
+\sum_{b=1}^{2}
\gamma_{\mathrm m}^{b}
R_{\mathrm{hold}}^{b},
\tag{36}
\]

\[
\widetilde\gamma_{\mathrm f}
=
\gamma_{\mathrm m}^{1+2}
\tag{37}
\]

其中2对应部署阶段包含的两个0.5 s资源帧。式(36)中的部署奖励是服务暂停时由实际队列积压产生的holding reward，不是人为固定惩罚。该处理保证微观经验回放中的奖励、下一个可决策状态和持续时间一致。

## 2.4 宏观半马尔可夫决策过程

宏观状态由每架无人机的局部摘要拼接而成，包括相对业务质心的位置、分优先级队列、到达量、机间流量、CPU利用率、用户上行距离、历史宏观流量以及功率对偶变量等。记局部状态为 \(\mathbf o_n^t\)，全局状态为

\[
\mathbf s_{\mathrm M}^{t}
=
\operatorname{concat}
\left(\mathbf o_1^t,\ldots,\mathbf o_N^t\right).
\tag{38}
\]

宏观动作为所有无人机的联合二维位移：

\[
\mathbf a_{\mathrm M}^t
=
\operatorname{vec}
\left(
\widehat{\mathbf a}_1^t,\ldots,\widehat{\mathbf a}_N^t
\right)\in\mathbb R^{2N}.
\tag{39}
\]

严格全局奖励模式使用本周期100个物理步实际微观奖励的均值：

\[
R_{\mathrm M}^t
=
\frac1{100}
\sum_{l=0}^{99}r_{\mathrm m}^{t,l}
+F_{\mathrm{shape}}^t
-C_{\mathrm{move}}^t
-C_{\mathrm{tie}}^t.
\tag{40}
\]

正式配置中势函数塑形、移动惩罚和飞行能量tiebreak权重均应以运行参数为准；若为0，式(40)退化为纯微观全局信号。宏观经验回放在约束模式下保存原始时延代价奖励和逐无人机能量约束比，TD3的时延Critic与能耗Critic分别学习，避免把一个随对偶变量变化的标量奖励当作固定目标。

# 3 基于TD3-SAC的约束分层学习算法

## 3.1 总体架构

算法包含三个阶段：

1. 固定宏观悬停策略，预训练微观SAC，使路由策略先获得基本的负载迁移能力；
2. 加载SAC参数和经验回放，初始化宏观TD3；
3. 在相同环境中继续更新SAC和TD3，并按回合更新逐无人机能量乘子。

该设计缓解了从随机宏观位置和随机微观路由同时开始学习产生的双重非平稳性。联合阶段不冻结SAC，因而宏观策略面对的是持续改进的微观执行器；为控制由此产生的分布漂移，训练脚本保留微观经验回放并以固定评估场景定期选择检查点。

![图3 TD3-SAC约束分层联合学习框架](figures/hrl_framework.pdf)

## 3.2 微观SAC任务路由

### 3.2.1 随机策略

SAC Actor由两层256神经元ReLU网络构成，分别输出均值和对数标准差：

\[
(\boldsymbol\mu_\phi,\log\boldsymbol\sigma_\phi)
=\pi_\phi(\mathbf s_{\mathrm m}),
\quad
\log\boldsymbol\sigma_\phi\in[-20,2].
\tag{41}
\]

采用重参数化采样和双曲正切压缩：

\[
\mathbf z
=\boldsymbol\mu_\phi+
\boldsymbol\sigma_\phi\odot\boldsymbol\epsilon,
\quad
\boldsymbol\epsilon\sim\mathcal N(\mathbf0,\mathbf I),
\tag{42}
\]

\[
\mathbf a_{\mathrm m}=\tanh(\mathbf z).
\tag{43}
\]

对数概率包含Tanh雅可比修正：

\[
\log\pi_\phi(\mathbf a|\mathbf s)
=
\log\mathcal N(\mathbf z;\boldsymbol\mu_\phi,\boldsymbol\sigma_\phi)
-\sum_i\log(1-a_i^2+\varepsilon).
\tag{44}
\]

### 3.2.2 双Q网络和目标值

两个Q网络共享相同结构但参数独立，每个网络输入 \((\mathbf s,\mathbf a)\)，经两层256神经元ReLU输出标量。对经验
\((\mathbf s,\mathbf a,r,\mathbf s',\Gamma)\)，其中 \(\Gamma\) 是普通帧或SMDP边界对应的实际自举折扣，目标为

\[
y
=
r+\Gamma
\left[
\min_{i=1,2}Q_{\bar\theta_i}(\mathbf s',\mathbf a')
-\alpha\log\pi_\phi(\mathbf a'|\mathbf s')
\right].
\tag{45}
\]

Critic损失为

\[
L_Q(\theta_1,\theta_2)
=
\sum_{i=1}^{2}
\mathbb E_{\mathcal D_{\mathrm m}}
\left[
\left(Q_{\theta_i}(\mathbf s,\mathbf a)-y\right)^2
\right].
\tag{46}
\]

Actor损失为

\[
L_\pi(\phi)
=
\mathbb E_{\mathcal D_{\mathrm m}}
\left[
\alpha\log\pi_\phi(\mathbf a|\mathbf s)
-\min_iQ_{\theta_i}(\mathbf s,\mathbf a)
\right].
\tag{47}
\]

温度参数采用自动调整：

\[
L_\alpha
=
-\mathbb E
\left[
\log\alpha
\left(\log\pi_\phi(\mathbf a|\mathbf s)+\mathcal H_{\mathrm{tar}}\right)
\right],
\quad
\mathcal H_{\mathrm{tar}}=-2N^2.
\tag{48}
\]

目标Critic进行软更新：

\[
\bar\theta_i
\leftarrow
(1-\tau_{\mathrm S})\bar\theta_i+\tau_{\mathrm S}\theta_i,
\quad\tau_{\mathrm S}=10^{-2}.
\tag{49}
\]

## 3.3 宏观TD3部署控制

TD3 Actor使用两层256神经元ReLU网络和Tanh输出，再按式(7)逐无人机投影到单位圆盘。TD3同时维护：

- 两个时延Q网络 \(Q_{\theta_1}^{\mathrm D},Q_{\theta_2}^{\mathrm D}\)；
- 两个能耗Q网络 \(\mathbf Q_{\psi_1}^{\mathrm E},\mathbf Q_{\psi_2}^{\mathrm E}\)，每个输出 \(N\) 维逐无人机约束回报。

目标动作加入截断高斯噪声：

\[
\widetilde{\mathbf a}'
=
\Pi_{\mathbb D^N}
\left[
\mu_{\bar\phi}(\mathbf s')
+\operatorname{clip}(\boldsymbol\epsilon,-c,c)
\right],
\tag{50}
\]

其中 \(\Pi_{\mathbb D^N}\) 表示对每架无人机的二维动作分别投影到单位圆盘。时延Critic目标为

\[
y^{\mathrm D}
=
r_{\mathrm D}
+(1-d)\gamma_{\mathrm M}
\min_{i=1,2}Q_{\bar\theta_i}^{\mathrm D}
(\mathbf s',\widetilde{\mathbf a}').
\tag{51}
\]

能耗是约束成本而非奖励，采用双Critic较大值形成保守目标：

\[
\mathbf y^{\mathrm E}
=
\mathbf c^{\mathrm E}
+(1-d)\gamma_{\mathrm M}
\max_{i=1,2}
\mathbf Q_{\bar\psi_i}^{\mathrm E}
(\mathbf s',\widetilde{\mathbf a}').
\tag{52}
\]

两个时延Critic和两个能耗Critic分别最小化均方误差。每隔 \(d_{\mathrm{policy}}\) 次Critic更新一次Actor：

\[
L_{\mathrm{TD3}}(\phi)
=
\mathbb E
\left[
-Q_{\theta_1}^{\mathrm D}
(\mathbf s,\mu_\phi(\mathbf s))
+\sum_{n=1}^{N}\lambda_n
\max_{i=1,2}Q_{\psi_i,n}^{\mathrm E}
(\mathbf s,\mu_\phi(\mathbf s))
\right].
\tag{53}
\]

式(53)表明宏观Actor不是简单最大化混合奖励，而是在学习时延价值的同时显式抑制保守估计的逐无人机能耗价值。

## 3.4 逐无人机原始—对偶更新

设第 \(e\) 个回合测得无人机 \(n\) 的平均电池功率为 \(\bar P_n^e\)。为减少单回合波动，使用最近 \(W_{\lambda}=5\) 个回合的滑动均值：

\[
\widehat P_n^e
=
\frac{1}{|\mathcal W_e|}
\sum_{j\in\mathcal W_e}\bar P_n^j.
\tag{54}
\]

归一化约束违背为

\[
\nu_n^e
=
\frac{\widehat P_n^e-P_n^{\max}}
{\max(P_n^{\max},\varepsilon)}.
\tag{55}
\]

乘子更新为

\[
\lambda_n^{e+1}
=
\operatorname{clip}
\left(
\lambda_n^e+\eta_\lambda\nu_n^e,
0,\lambda_{\max}
\right),
\tag{56}
\]

其中 \(\eta_\lambda=0.02\)，\(\lambda_{\max}=20\)。更新后的乘子同步进入微观状态、微观拉格朗日代价、CPU能量价格和宏观Actor损失。每架无人机使用独立乘子，避免总功率满足预算但个别无人机长期过载的情况。

式(56)属于带函数逼近和有限窗口估计的经验原始—对偶方法。本文不声称其保证全局收敛；约束有效性通过逐无人机功率轨迹、违背率、乘子轨迹和长时固定策略评估验证。

## 3.5 算法伪代码

### 算法1 微观SAC预训练

```text
输入：环境E，固定悬停宏观策略，预训练回合数Epre，
      SAC参数φ、θ1、θ2、α，逐无人机乘子λ，经验池Dm
输出：预训练SAC、目标Critic、温度参数、经验池和乘子

1  初始化SAC网络、目标网络、温度参数和经验池Dm
2  初始化λn≥0, n=1,...,N
3  for e = 1,...,Epre do
4      重置场景、队列、用户位置和统计量
5      for t = 1,...,TM do
6          执行全零宏观位移，进入部署阶段
7          逐帧执行部署；累计holding reward和功率
8          for τ = 1,...,Tservice do
9              在资源帧边界更新用户移动和泊松到达
10             观察微观状态sm
11             从Tanh-Gaussian策略采样动作am
12             用式(17)生成高、低优先级路由矩阵
13             保持am执行5个物理步，获得帧奖励Rf
14             若为普通服务帧，存储(sm,am,Rf,sm',γm)
15             若跨越下一部署边界，按式(36)—(37)构造SMDP转移
16             当|Dm|达到预热阈值后，更新双Q、Actor和α
17             软更新目标Critic
18         end for
19     end for
20     统计每架无人机回合平均功率
21     按式(54)—(56)更新λ并同步CPU能量价格
22 end for
23 保存SAC参数、优化器、温度参数、经验池和λ
```

### 算法2 TD3-SAC联合训练

```text
输入：预训练SAC及经验池，宏观TD3，总回合数E，
      宏观经验池DM，功率预算Pmax
输出：联合训练后的TD3、SAC、乘子和评估检查点

1  加载预训练SAC、优化器、经验池和λ
2  初始化TD3 Actor、目标Actor、时延双Critic、能耗双Critic及目标网络
3  for e = 1,...,E do
4      按训练协议采样初始位置、业务负载和随机种子
5      获得宏观全局状态sM
6      for t = 1,...,TM do
7          若处于TD3预热阶段，在单位圆盘内采样联合位移动作
8          否则执行μ(sM)并叠加当前回合探索噪声
9          投影动作并映射为最大20 m的物理位移
10         执行1 s部署，累计飞行能耗、瞬态能耗和holding cost
11         若存在上一周期末待定SAC转移，组成跨部署SMDP转移并存入Dm
12         for τ = 1,...,Tservice do
13             观察sm并由SAC生成2N²维动作
14             执行一个0.5 s资源帧，累计实际队列和能量代价
15             保存普通帧转移；最后一帧暂存至下一部署完成
16             当经验池充分时继续更新SAC
17         end for
18         由本周期100个物理步的真实微观信号计算RM
19         取得下一宏观状态sM'和逐无人机能量约束比g
20         向DM存储(sM,aM,rD,g,sM',d)
21         当DM达到预热阈值后，更新时延Critic和能耗Critic
22         每隔dpolicy次Critic更新，按式(53)更新TD3 Actor
23         软更新TD3目标网络
24         sM ← sM'
25     end for
26     统计回合功率，并按式(54)—(56)更新逐无人机乘子
27     按固定评估场景周期性评估，保存候选和最优检查点
28 end for
```

### 算法3 在线分层执行

```text
输入：训练完成的TD3 Actor μ、SAC Actor π、当前系统状态
输出：无人机部署和任务路由决策

1  在宏观周期开始时构造sM
2  计算确定性动作aM=μ(sM)，逐无人机投影并映射为物理位移
3  执行部署；部署阶段不调用微观策略
4  在每个服务资源帧边界构造sm
5  使用SAC均值动作am=tanh(μφ(sm))
6  按式(17)生成满足非负性和行和约束的两套路由矩阵
7  保持该路由动作0.5 s并更新队列、通信、计算和能量状态
8  重复步骤4—7直至宏观周期结束，返回步骤1
```

## 3.6 可行性、稳定性与收敛边界

**动作可行性**：宏观动作的单位圆盘投影保证单周期位移不超过 \(D_{\max}\)；微观动作的逐行Softmax保证式(18)。CPU频率由局部KKT分配器投影至可行频率集合。因此，神经网络输出不会直接破坏三类硬约束。

**有界学习信号**：动作经Tanh限制，归一化代价被裁剪，对偶变量限制在 \([0,\lambda_{\max}]\)，梯度范数限制为10。这些机制提高数值稳定性，但不等价于数学意义上的策略收敛证明。

**时间一致性**：普通SAC经验对应一个0.5 s资源帧；跨部署经验按实际持续帧数调整奖励与自举折扣；宏观经验对应完整10 s周期。测试 `test_resource_frame_smdp.py` 和 `test_hierarchical_feedback.py` 用于验证持续时间与奖励重构。

**队列稳定性边界**：若长期到达率超过系统总计算容量，任何策略都无法稳定队列。因此正式实验前必须执行容量审计，确认

\[
\sum_n\bar A_n
<
\sum_n\frac{F_n^{\max}}{C},
\tag{57}
\]

并同时报告局部容量缺口，因为跨无人机路由的价值恰来自“局部过载、全局可行”的场景。

**不能声称的结论**：在非线性函数逼近、联合更新、经验回放和移动用户共同存在时，当前实现不能证明P0的全局最优性，也不能仅凭训练回报平稳就证明长期队列稳定或功率约束严格满足。论文只能在明确实验时域、置信区间和统计误差下声称经验改进。

## 3.7 时间和空间复杂度

设全连接网络层宽序列为 \(\mathbf d=(d_0,d_1,\ldots,d_L)\)，一次前向传播乘加复杂度为

\[
\mathcal C(\mathbf d)
=
O\!\left(\sum_{\ell=0}^{L-1}d_\ell d_{\ell+1}\right).
\tag{58}
\]

令微观状态维数为 \(d_{\mathrm m}\)，微观动作维数

\[
a_{\mathrm m}=2N^2,
\tag{59}
\]

宏观状态维数为 \(d_{\mathrm M}\)，宏观动作维数

\[
a_{\mathrm M}=2N.
\tag{60}
\]

SAC Actor单样本推理复杂度为

\[
\mathcal C_{\mathrm{SA}}
=O(d_{\mathrm m}h_{\mathrm S}+h_{\mathrm S}^2+h_{\mathrm S}a_{\mathrm m}),
\quad h_{\mathrm S}=256.
\tag{61}
\]

双Critic单批更新复杂度为

\[
\mathcal C_{\mathrm{SQ}}
=
O\!\left(
B[2(d_{\mathrm m}+a_{\mathrm m})h_{\mathrm S}
+4h_{\mathrm S}^2]
\right).
\tag{62}
\]

一次完整SAC更新还包括Actor、目标Critic和温度更新，数量级为

\[
\mathcal C_{\mathrm{SAC}}
=
O\!\left(
B[
4(d_{\mathrm m}+a_{\mathrm m})h_{\mathrm S}
+6h_{\mathrm S}^2
+2h_{\mathrm S}a_{\mathrm m}]
\right).
\tag{63}
\]

TD3 Actor推理复杂度为

\[
\mathcal C_{\mathrm{TA}}
=O(d_{\mathrm M}h_{\mathrm T}+h_{\mathrm T}^2+h_{\mathrm T}a_{\mathrm M}),
\quad h_{\mathrm T}=256.
\tag{64}
\]

TD3包含时延双Critic和能耗双Critic，单批Critic更新复杂度为

\[
\mathcal C_{\mathrm{TQ}}
=
O\!\left(
B[
4(d_{\mathrm M}+a_{\mathrm M})h_{\mathrm T}
+8h_{\mathrm T}^2
+2h_{\mathrm T}(1+N)]
\right).
\tag{65}
\]

设每回合宏观周期数为 \(T_{\mathrm M}\)，每周期服务资源帧数为 \(T_{\mathrm S}=18\)，训练回合数为 \(E\)，每个服务帧执行一次SAC更新、每个宏观周期执行 \(U_{\mathrm T}\) 次TD3更新，则训练网络复杂度为

\[
\mathcal C_{\mathrm{train}}
=
O\!\left[
E T_{\mathrm M}
\left(
T_{\mathrm S}\mathcal C_{\mathrm{SAC}}
+U_{\mathrm T}\mathcal C_{\mathrm{TQ}}
+\frac{U_{\mathrm T}}{d_{\mathrm{policy}}}\mathcal C_{\mathrm{TA}}
\right)
\right].
\tag{66}
\]

在线执行不进行反向传播，每10 s执行一次TD3 Actor、每0.5 s服务帧执行一次SAC Actor：

\[
\mathcal C_{\mathrm{online/macro}}
=
\mathcal C_{\mathrm{TA}}
+T_{\mathrm S}\mathcal C_{\mathrm{SA}}
+O(T_{\mathrm S}N^2),
\tag{67}
\]

其中最后一项是两套路由矩阵的Softmax。

若微观经验池容量为 \(M_{\mathrm m}\)，宏观经验池容量为 \(M_{\mathrm M}\)，空间复杂度为

\[
\mathcal S
=
O\!\left[
M_{\mathrm m}(2d_{\mathrm m}+2N^2+c_{\mathrm m})
+M_{\mathrm M}(2d_{\mathrm M}+2N+N+c_{\mathrm M})
\right],
\tag{68}
\]

其中常数项包含奖励、折扣、终止标志等。正式实现 \(M_{\mathrm m}=500000\)，\(M_{\mathrm M}=12000\)。

SCA基线对 \(2N^2\) 个路由logit进行最多 \(I_{\mathrm{SCA}}\) 次逐次凸近似。若通用内点法求解具有 \(v=O(N^2)\) 个变量的子问题，保守复杂度可写为

\[
\mathcal C_{\mathrm{SCA}}
=O(I_{\mathrm{SCA}}v^3)=O(I_{\mathrm{SCA}}N^6),
\tag{69}
\]

但实际脚本还利用解析梯度、投影和小规模结构，终稿应同时报告实测求解时间，避免仅用最坏阶数夸大差异。

# 4 仿真设计与结果呈现规范

## 4.1 正式参数

| 参数 | 数值 |
|---|---:|
| 区域大小 | \(1000\,\mathrm m\times1000\,\mathrm m\) |
| 无人机数 \(N\) | 3 |
| 用户数 \(K\) | 39 |
| 固定高度 \(H\) | 50 m |
| 区域基础负载 | 12.5、5.5、3.5 Mbit/s |
| 全局负载系数 | 1.15 |
| 无人机CPU频率 | 10 GHz |
| 每比特CPU周期 \(C\) | 1000 cycle/bit |
| 上行带宽 | 20 MHz |
| 机间带宽 | 3 MHz |
| 用户上行功率 | 0.1 W |
| 无人机间发射功率 | 1 W |
| 单机功率预算 | 280 W |
| 宏观周期 | 10 s |
| 部署时间 | 1 s |
| 资源帧 | 0.5 s |
| 物理步长 | 0.1 s |
| 最大单周期位移 | 20 m |
| 每回合宏观周期 | 30 |
| SAC预训练回合 | 200 |
| 联合训练回合 | 600 |
| 正式随机种子 | 1、42、123 |
| TD3经验池 | 12000 |
| SAC经验池 | 500000 |
| 对偶学习率 | 0.02 |
| 对偶更新窗口 | 5回合 |
| 对偶上限 | 20 |

表中未列出的TD3折扣率、目标更新系数、策略噪声、批大小和实际网络学习率必须从最终运行命令及保存的scenario metadata自动回填，不能只引用脚本默认值。

## 4.2 对比算法与公平性

建议正式保留以下可执行对比：

1. **Joint TD3-SAC**：本文完整方法；
2. **Hover-SAC**：无人机保持初始位置，仅由SAC优化路由；
3. **Centroid-SAC**：宏观层使用业务质心追踪，微观层使用相同SAC；
4. **TD3-Local**：宏观TD3优化位置，所有任务本地处理，用于验证协作路由价值；
5. **TD3-SCA**：宏观TD3配合模型驱动SCA路由，用于比较学习型和在线优化型调度；
6. **PPO-SAC**：使用仓库现有宏观PPO训练器，作为宏观算法对比；
7. **Position Oracle**：只作为位置性能上界或诊断工具，不与在线方法进行不加说明的公平性排名。

所有方法使用相同用户轨迹、任务到达随机种子、初始无人机位置、功率预算和评估时长。比较学习算法时应报告训练预算和网络参数量；比较SCA时必须报告在线求解时间；Oracle必须说明额外使用的未来信息或搜索预算。

## 4.3 必须完成的实验

### 4.3.1 收敛性

- SAC预训练平均帧奖励、时延代理和逐无人机功率；
- 联合阶段TD3宏观原始奖励、时延Critic损失、能耗Critic损失；
- 微观SAC在联合阶段是否继续改善；
- 三个乘子 \(\lambda_n\) 的轨迹；
- 宏观奖励重构误差，应接近数值零；
- SMDP边界持续时间、折扣指数和组合奖励审计。

![图4 SAC预训练收敛性](figures/sac_pretrain_convergence.pdf)

![图5 TD3-SAC联合训练收敛性](figures/joint_convergence.pdf)

### 4.3.2 总体性能

主表至少报告：加权积压压力、平均队列、已完成任务平均/95分位时延、高优先级截止期违背率、总电池能量、逐无人机平均功率、功率预算违背率、吞吐量和在线推理时间。每项给出3个随机种子的均值、标准差和95%置信区间。

| 方法 | 时延代理 | 高优先级P95时延 | 平均队列 | 吞吐量 | 最大单机功率 | 违背率 |
|---|---:|---:|---:|---:|---:|---:|
| Hover-SAC | 【待实验】 | 【待实验】 | 【待实验】 | 【待实验】 | 【待实验】 | 【待实验】 |
| Centroid-SAC | 【待实验】 | 【待实验】 | 【待实验】 | 【待实验】 | 【待实验】 | 【待实验】 |
| TD3-Local | 【待实验】 | 【待实验】 | 【待实验】 | 【待实验】 | 【待实验】 | 【待实验】 |
| TD3-SCA | 【待实验】 | 【待实验】 | 【待实验】 | 【待实验】 | 【待实验】 | 【待实验】 |
| PPO-SAC | 【待实验】 | 【待实验】 | 【待实验】 | 【待实验】 | 【待实验】 | 【待实验】 |
| Joint TD3-SAC | 【待实验】 | 【待实验】 | 【待实验】 | 【待实验】 | 【待实验】 | 【待实验】 |

### 4.3.3 负载、预算与规模敏感性

- 负载系数：检验从轻载到接近容量边界时的性能；
- 单机功率预算：检验约束—时延权衡；
- 用户数：检验状态维度和业务密度变化；
- 无人机数：检验 \(2N^2\) 动作增长对性能和推理时间的影响；
- 最大部署距离：检验机动性与飞行能耗权衡；
- 宏观周期长度：检验位置更新频率；
- 资源帧长度：检验路由响应速度和决策开销。

任何超出训练维度的 \(N\) 或 \(K\) 变化都需要重新训练相应网络，不能直接把固定输入维度模型称为零样本泛化。

### 4.3.4 鲁棒性

设计突发负载实验：在指定连续宏观周期把某一区域负载提高至基准的1.5、2.0或3.0倍，随后恢复。报告队列峰值、恢复时间、截止期违背率和乘子响应。还应交换热点区域，验证策略是否只记忆固定地理位置。

### 4.3.5 消融实验

| 消融 | 删除机制 | 验证问题 |
|---|---|---|
| w/o pretrain | SAC不预训练 | 预训练是否降低联合非平稳性 |
| frozen SAC | 联合阶段冻结SAC | 联合更新是否必要 |
| w/o SMDP boundary | 忽略部署holding与持续时间 | 边界反馈是否造成价值偏差 |
| system dual | 3个乘子改为单一总功率乘子 | 逐无人机约束是否避免个体过载 |
| w/o dual | \(\lambda_n=0\) | 约束机制是否真正控制功率 |
| w/o transient | 去除式(11) | 瞬态能量对部署策略的影响 |
| hover | 去除宏观位置控制 | 空间部署收益 |
| local only | 去除机间协作 | 微观路由收益 |

## 4.4 图表解释模板

每张图必须采用以下说明，防止出现“图有趋势、正文无论证”的问题：

> **作图目的**：说明该图验证哪一条方法声明。  
> **横轴与单位**：明确训练回合、时间、负载或预算。  
> **纵轴与单位**：区分真实时延、时延代理、能量和功率。  
> **统计方法**：均值曲线、阴影标准差/置信区间、随机种子。  
> **控制变量**：列出除横轴变量外保持不变的配置。  
> **数据来源**：CSV/NPZ文件和绘图脚本。  
> **异常解释**：同时检查队列、吞吐量和约束，不能只选择有利指标。  

## 4.5 在线运行时间

在同一硬件上分别预热100次并测量1000次：

- TD3 Actor单次推理；
- SAC Actor单次推理；
- 路由Softmax映射；
- SCA单次求解；
- 完整0.5 s资源帧决策。

报告均值、标准差、中位数和P95。推理计时必须在需要时调用CUDA同步，且把环境仿真时间与策略推理时间分开。只有当P95决策时间显著小于0.5 s时，才可声称满足在线资源帧决策要求。

# 5 结束语

本文面向具有异构业务、动态用户和逐无人机功率预算的多无人机移动边缘计算系统，建立了无人机周期部署与双优先级任务路由的联合优化模型。针对位置与路由在决策频率、动作维度和物理影响上的差异，将原问题分解为宏观部署与微观任务调度两个相互耦合的决策层。微观层使用SAC生成高、低优先级两套连续路由矩阵，并通过Softmax保证路由可行性；宏观层使用带独立能耗双Critic的TD3优化多无人机联合位移；跨部署边界的微观经验按照SMDP持续时间组合holding reward和自举折扣；逐无人机原始—对偶更新则用于抑制长期功率违背。

当前模型和算法主体已经形成闭环，但最终结论必须由正式多随机种子实验支撑。待完成主对比、消融、突发负载、约束轨迹和在线时延测量后，方可回填“性能提升比例”和“实时性”等定量结论。

# 附录A 代码—公式映射

| 论文机制 | 公式 | 实现位置 | 一致性说明 |
|---|---|---|---|
| 泊松到达 | (5)—(6) | `env.py::_generate_traffic` | 一致 |
| Gauss-Markov移动 | (2)—(4) | `env.py::_update_gauss_markov_mobility`、`_integrate_user_positions` | 一致 |
| 飞行功率 | (10) | `env.py::_compute_tcom_flying_power` | 一致 |
| 部署瞬态能量 | (11) | `env.py::_compute_deployment_transient_energy` | 一致 |
| 双优先级路由动作 | (17)—(18) | `hrl_wrappers.py::micro_step` | 动作为 \(2N^2\) |
| CPU频率分配 | (22)—(24b) | `env.py::_priced_maxweight_frequency_pair`、`_allocate_cpu_frequencies` | 正式模式为 `priced_maxweight`，能耗为总频率立方 |
| 资源帧动作保持 | (35) | `hrl_wrappers.py::step_resource_frame` | 5个0.1 s物理步 |
| SMDP边界转移 | (36)—(37) | `train_centralized_macro.py::compose_smdp_boundary_transition` | 包含2个部署帧 |
| 宏观严格全局奖励 | (40) | `hrl_wrappers.py::get_macro_reward` | 使用实际微观奖励均值 |
| SAC Actor/双Critic | (41)—(49) | `agents/sac_agent.py`、`train_centralized_macro.py::update_sac` | 一致 |
| TD3 Actor和时延双Critic | (50)—(53) | `scripts/train_centralized_td3.py` | 一致 |
| TD3能耗双Critic | (52)—(53) | `scripts/train_centralized_td3.py::TD3Agent` | 输出维数为 \(N\) |
| 逐无人机对偶更新 | (54)—(56) | `scripts/train_centralized_td3.py::update_energy_dual` | 5回合窗口 |
| 宏观动作维数 | (39)、(60) | `train_centralized_td3.py` | \(2N\) |
| 微观动作维数 | (31)、(59) | `hrl_wrappers.py` | \(2N^2\) |

# 附录B 建模一致性审计

1. **TD3与MAPPO注释冲突**：`hrl_wrappers.py` 顶部保留历史“MAPPO UPGRADE”注释，但正式主方法由 `scripts/train_centralized_td3.py` 实现集中式TD3。本文以实际训练入口为准，PPO仅作为对比算法。
2. **环境默认CPU与正式配置不同**：环境类默认16 GHz，正式协议通过运行参数覆盖为10 GHz。论文参数表使用正式配置，并要求scenario metadata确认。
3. **时延概念**：训练主目标是加权积压压力，不是直接的平均完成时延。真实时延只能从任务跟踪器统计并作为评估指标。
4. **能量边界**：逐无人机预算统计无人机电池能量；地面用户上传能量单独报告，不并入无人机功率约束。
5. **部署期间到达**：正式脚本启用部署期间外生到达，但通信与计算暂停。若最终运行参数改变，式(36)和实验说明必须同步修改。
6. **CPU能耗公式**：正式 `priced_maxweight` 模式使用 \(\kappa(f_{\mathrm H}+f_{\mathrm L})^3\)；若后续更换CPU分配模式，必须同步修改式(24a)—(24b)，不能沿用当前模型。
7. **归一化常数**：\(\Psi_{\mathrm{norm}}=8\)、能量归一化等源自smoke校准，不属于物理常数；实验需进行敏感性分析或说明固定依据。
8. **约束保证**：对偶更新提供经验约束控制，不构成有限时间严格保证。必须报告每架无人机而非仅系统平均功率。
9. **复杂度**：微观动作随 \(N^2\) 增长；用户数通过状态构造、业务生成和环境仿真影响开销。不得声称复杂度与用户数无关。
10. **实验空缺**：当前无正式结果，所有性能数字、显著性和实时性声明均待实验回填。

# 附录C 投稿前不可省略的验证清单

- [ ] 主方法三个随机种子全部完成且可复现；
- [ ] 每个检查点使用固定独立评估种子；
- [ ] 队列容量审计表明全局负载可行；
- [ ] 宏观奖励重构最大误差接近浮点误差；
- [ ] SMDP边界持续时间和折扣指数与物理时间一致；
- [ ] 每架无人机平均功率、P95功率和违背率均已报告；
- [ ] 真实时延与时延代理明确区分；
- [ ] 所有基线使用相同任务轨迹和预算；
- [ ] SCA报告在线求解时间；
- [ ] Oracle明确额外信息和搜索预算；
- [ ] 消融分别支撑预训练、SMDP反馈和逐机对偶三项机制；
- [ ] 公式单位完成bit/Mbit、Hz/MHz、J/W和s的逐项审计；
- [ ] 所有图可从原始CSV/NPZ一键复现；
- [ ] 没有用单次最好结果替代均值与置信区间；
- [ ] 没有在无统计检验时使用“显著提升”；
- [ ] 引言中的每条贡献均能对应公式、算法和实验；
- [ ] 参考文献均真实可检索且与具体技术声明对应。
