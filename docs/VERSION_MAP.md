# 代码、实验和文档版本地图

| 阅读位置 | 原始来源 | 用途 |
|---|---|---|
| 根目录 `src/`、`tests/`、`tools/` | MAIN commit `a9953dc36d9d65f99bcc35044cdfc250ca11bb49` | v2.3.0 核心代码和早期工程验证 |
| `research/speedup/` | speedup commit `25d59ff9f7c3e27c060c63d462235a30ce28930a` | 最新 v2.4.0 实验核心代码、测试、配置 |
| `research/experiments/outputs/scene_optimality_20260928/` | MAIN 本地历史实验脚本及选定 JSON 合同/报告 | 三阶段、多种子、curriculum、方案 C/D 驱动 |
| `docs/history/outputs/` | MAIN 的 outputs 下报告及图表 | 历史研究结论、诊断、失败/重试记录 |
| `docs/` 和 `research/speedup/docs/` 的既有说明 | 各源树的受版本控制文档 | 按标题/日期解释的历史说明，不保证反映最新实验 |
| `src/hrl_v2/_reference/` | 包内归档源码 | 限定工程对照，不是最新模型定义 |

根目录与 speedup 是两份独立 Python 项目，不能把一份源码与另一份配置混装。默认配置分别由各自 `hrl_v2/paths.py` 指向 v2.3.0 / v2.4.0 full 配置；研究 Lite 路径需显式选择对应 `*_lite.json`。`pyproject.toml` 的包版本 2.1.0 与物理候选版本不是同一个编号。

`run.require_research_protocol` 在当前代码中检查 research 确认标志及严格折扣，训练还检查本地 validation identity、runtime 和 frozen baseline 等输入。早期 README 的“无条件禁用”已经过时。配置通过技术检查也不是自动授权启动训练。

后续实验驱动引用原机器路径、冻结输入、专门的输出保护和源代码身份；保留它们是为了代码理解，迁移机器需另行适配。原始输出前缀 `D:/HRL_UAV_V2/outputs/...` 的报告现在位于 `docs/history/outputs/...`；对应脚本位于 `research/experiments/outputs/...`。图表链接已转换成 GitHub 可读相对路径。缺失原始数据/权重的路径明确保留为文字，不伪装成可用链接。

本快照来自工作目录实际文件，未恢复用户已删除的旧 validation 文件。完整原始源 commit、文件校验和及链接处理记录见根目录 `HANDOFF_MANIFEST.json`。这个分支从单一无父提交创建，因此上传时不会携带旧 Git 历史；原仓库其他分支及其数据仍保持原状。
