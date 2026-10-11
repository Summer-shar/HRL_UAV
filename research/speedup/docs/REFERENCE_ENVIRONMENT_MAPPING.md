# 参考环境与本地实现对应表

仅静态读取指定版本环境源码；本轮测试结果另见 [REFERENCE_ENVIRONMENT_VALIDATION.md](REFERENCE_ENVIRONMENT_VALIDATION.md)。

固定版本：[LyDROO d1084ab](https://github.com/revenol/LyDROO/tree/d1084ab1038a49eb8c97b4a9342c456920728eaf)、[UAV-DDPG 67e4675](https://github.com/fangvv/UAV-DDPG/tree/67e46755fa484b51424c923cfd3ddc4217676bc9)、[FANET a267b0c](https://github.com/raoulraft/multi-agent-fanet-offload/tree/a267b0c1ce060fa6ac383f8043c55d92c2297669)。

| 参考源码与做法 | 本地位置 | 保留 / 修复 / 不采用 |
|---|---|---|
| LyDROO [LyDROO.py:140–148](https://github.com/revenol/LyDROO/blob/d1084ab1038a49eb8c97b4a9342c456920728eaf/LyDROO.py#L140-L148)：Q 按到达和服务跨槽递推。 | [industrial_env.py](../src/hrl_uav/industrial_env.py):116–126、240–342，_admit 与三类服务函数。 | **保留**持久队列；本地保留完整 parent/fragment，不以 aggregate Q 替换身份。 |
| LyDROO [ResourceAllocation.py:64–71](https://github.com/revenol/LyDROO/blob/d1084ab1038a49eb8c97b4a9342c456920728eaf/ResourceAllocation.py#L64-L71)、[144–160](https://github.com/revenol/LyDROO/blob/d1084ab1038a49eb8c97b4a9342c456920728eaf/ResourceAllocation.py#L144-L160)：CPU 不超需求/上限，offload 不超 Q 和共享时间。 | [industrial_env.py](../src/hrl_uav/industrial_env.py):240–342，_upload_step/_compute_step/_transfer_step；[env.py](../src/hrl_uav/env.py):708、733–734、795–806，共享带宽/源功率。 | **保留** min(remaining,budget) 扣账；**不采用**固定 6 位服务量舍入。tiny-positive 残差须真实服务，当前代码已修复。 |
| UAV-DDPG [UAV_env.py:184–193](https://github.com/fangvv/UAV-DDPG/blob/67e46755fa484b51424c923cfd3ddc4217676bc9/DDPG/UAV_env.py#L184-L193)：距离/高度影响信道增益与通信速率。 | [industrial_env.py](../src/hrl_uav/industrial_env.py):14；[env.py](../src/hrl_uav/env.py):705–714，compute_u2g_rate_bps。 | **保留**当前位置决定当前服务容量；使用本地多 UAV 共享资源参数。 |
| UAV-DDPG [UAV_env.py:195–199](https://github.com/fangvv/UAV-DDPG/blob/67e46755fa484b51424c923cfd3ddc4217676bc9/DDPG/UAV_env.py#L195-L199)：并行本地/边缘分支完成时间取 max。 | [industrial_env.py](../src/hrl_uav/industrial_env.py):224–237、266–274，_route/_complete。 | **保留**等最慢正分片完成；**不采用**闭式无队列时延。本地先上传再多 UAV 分片，没有终端本地计算分支。 |
| FANET [drone.py:98–157](https://github.com/raoulraft/multi-agent-fanet-offload/blob/a267b0c1ce060fa6ac383f8043c55d92c2297669/drone.py#L98-L157)：同一 Packet 从 offload FIFO 进入目的 processing FIFO，处理后完成。 | [industrial_env.py](../src/hrl_uav/industrial_env.py):116–126、229–235、266–274、277–342，_admit/_route/_complete/服务函数。 | **保留** parent 身份、forward→目的 CPU、FIFO。转发不算完成；两优先级和连续 fragments 是本地扩展。 |
| FANET [uav_env.py:232–247](https://github.com/raoulraft/multi-agent-fanet-offload/blob/a267b0c1ce060fa6ac383f8043c55d92c2297669/uav_env.py#L232-L247)：实际 reward 是当期已完成 delay 均值的负值。 | [industrial_env.py](../src/hrl_uav/industrial_env.py):354、372，micro_step；[metrics.py](../src/hrl_v2/metrics.py):63–99，finish。 | **不采用** completed-only reward。保留父级 H=dt(10N_H+N_L)、c=0.09H+E/140；等待尾 fragment 仍计一次 holding，未完成记 censored。 |
| UAV-DDPG [UAV_env.py:164–173](https://github.com/fangvv/UAV-DDPG/blob/67e46755fa484b51424c923cfd3ddc4217676bc9/DDPG/UAV_env.py#L164-L173) 替换任务；FANET [uav_env.py:265–274](https://github.com/raoulraft/multi-agent-fanet-offload/blob/a267b0c1ce060fa6ac383f8043c55d92c2297669/uav_env.py#L265-L274) 按 max_time 返回 done。 | [collector.py](../src/hrl_v2/collector.py):100–118，_collect 与 _continuation。 | **不采用**控制步清任务或截止即终止。保留真实队列、successor；正常窗口 terminated=False、truncated=True。 |

不采用：二元卸载、虚拟 energy queue、概率整任务卸载、最短队列目的选择、满队列 drop、旧式 done、简式飞行能耗；这些参考做法不替代本地连续分片和物理成本。
