> 归档说明：以下内容是当前仓库自带的先前工程记录，不定义本轮持续服务合同，也不代表本轮测试或训练结果。当前说明见 [场景与控制合同](design_specification.md) 和 [本轮验证](CONTINUING_SERVICE_VALIDATION.md)。

# 来源与本轮改动

算法基线：137d2f1。吞吐优化来源：dbf2b977b35a14b5cb560d649cf3c22f976e9584，原仓库 https://github.com/Summer-shar/HRL_UAV ，分支codex/report_ledger_v1。

本轮改为src布局、标准wheel/sdist安装、通用CLI、包内资源寻址和独立输出工作区。配置仅移除本机Python绝对路径；训练验证和检查点继续记录实际数值环境。发布记录不能替代目标机器的本地验证。

系统模型、优化问题、MDP/SMDP、状态、动作、奖励、网络独立性、有效更新预算、图特征、物理实现及论文贡献没有改动。训练流程仅调整启动、路径和验证记录位置；未新增研究训练或性能结论。

旧本机运行副本保留在 `.local_backup/standalone_v1/`，不进入Git、wheel、sdist或源码发布ZIP。旧fixture保留在本机 `data/fixtures/` 并被Git忽略。原证据中的本机路径及“未提交”叙述是当时状态，不是本项目依赖。

默认协议配置、软件数值环境及源码仍以哈希和验证记录约束。不同GPU、软件版本或跨平台需要目标机器重新验证；本轮不能证明所有受声明依赖范围覆盖的后端数值一致。
