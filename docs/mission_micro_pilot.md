# Mission V1 第二阶段：固定Predictive下的微观规则比较

## 范围与来源

第一阶段运行提交 `ef1a0a8178f7e1733c07d5cb1496573b6468955a` 已推送至 `codex/mission_rpgm_v1`，不包含results。
物理版本继续固定 `19976306281cf213f1e7f6973b999f2dcd39d593`；本轮不改旧源码、配置、物理模型或算法。
新增独立runner、定向测试和本说明，运行manifest记录未提交文件指纹，并保存新源码副本，不把未提交新增入口冒充ef1提交内容。

仅33600/33601，各900秒，宏观始终复用第一阶段Predictive：当前业务质心加10秒历史速度，原位移和安全映射。
高低优本地比例：Local=(1,1)，Uniform=(1/3,1/3)，Fixed-A=(2/3,1/3)，Fixed-B=(1/2,1/3)。非本地份额在另外两机间均分。
Fixed-A/B定义核对SAC101原checkpoint配置中的pilot.fixed_rules，不重新搜索比例。

两条SAC结果从 `results/mission_rpgm_v1_pilot/predictive_centroid_33600` 和 `predictive_centroid_33601` 复用。
必须匹配运行提交、物理来源、全部有效配置、确定性/冻结标志、源码指纹、900秒时域和原始CSV/JSON数值；否则停止，不自动补跑。
SAC101来源SHA256固定 `be946d3a576918460690a16f1a57f2c6dc33dde13db86d360db5690858a76932`，只读校验，不在新8条固定规则评价中加载成推理网络。

## 执行

```powershell
& 'D:\Anaconda3\envs\pytorch-gpu\python.exe' -m unittest tests.test_mission_micro_pilot -v
& 'D:\Anaconda3\envs\pytorch-gpu\python.exe' scripts/run_mission_micro_pilot.py
```

只运行新增定向测试，禁止神经网络构造和反向传播。包含一个20秒固定Uniform的真实rollout，配对原SAC900秒的前缀。
正式入口先四种规则各20秒功能检查，再新增恰好8条900秒评价；短检查不作性能结果。复用第一阶段evaluate_one、MeteredMDP派生统计器、原路由类和物理step，不构建第二套队列模型。
每个实际执行路由核对目标矩阵；Local U2U工作量、发送时间、发送能量必须为0。
无PPO/SAC网络，无optimizer更新，无replay读写，无训练或微调。300/600秒不重置，不执行complete_tail。

任务出生、用户/群中心/业务质心、负载与环境RNG逐步配对；因固定Predictive不依赖路由状态，另外逐值核对宏观动作和UAV轨迹。
不强制回放SAC路由或队列，各微观规则接受自己行为造成的真实结果。

实际定向测试：首次5项中4通过、1失败，原因是短20秒与原900秒的完整日程哈希包含不同数组长度。
只修复新入口的短程配对核验：独立核对900秒原日程哈希、20秒日程是其严格前缀，随后比较实际逐帧负载及全部其他外生指纹。
完整900秒评价仍要求原完整日程哈希逐帧一致。没有修改环境日程生成器或放宽正式评价配对。
复测5项全部通过（0.384秒测试执行时间）；无网络构造、无反向传播。CSV增量写入另测原始表头顺序，避免复用行和新增行字段排列差异。

## 输出与边界

新目录 `results/mission_rpgm_v1_micro_pilot/`，不覆盖历史。保存manifest、10行per_seed、5行summary、逐场景及均值配对差、路由和当前角色统计、详细报告。
新增臂保存0.5秒队列、宏观指标、U2U链路、CPU服务、实际路由、命令及真实轨迹。旧SAC原文件不复制覆盖，保留来源路径、哈希和reused=true。
J、驻留/能量项、完整功率、完成任务时延、未完成数/完成率、平均/期末积压、后300秒斜率、吞吐和资源统计沿用第一阶段口径。
反例和负结果全部保留，不按低completed-only时延挑结果；两开发场景不算两次训练，不声称泛化或完整HRL。
完成8条后停止，不自动轨迹搜索、PPO训练、SAC微调、其他场景或微观规则扫描。禁止33610–33612和33700–33709。
