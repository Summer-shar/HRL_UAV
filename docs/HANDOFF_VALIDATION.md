# 交接快照检查

日期：2026-10-12。

- MAIN 与 speedup 两份源码分别通过配置/包装/性能工具测试，各 17 项。
- 导出 Python 文件全部通过 AST 语法解析；源码与配置相对原文件字节校验一致。
- 保留 58 份 outputs 历史 Markdown 报告及图表；既有 docs 报告也保留。
- 未包含 .pt/.pth/.pkl/.npz/.npy/.csv/.jsonl/.zip、密钥文件或虚拟环境；大型原始 profiling JSON 排除。
- 修复图表的 Windows 本地路径，并把未包含的历史附件改为明确文字；交接入口链接全部存在。排除记录见 EXCLUDED_ARTIFACT_LINKS.json。
- 未启动训练；未重跑历史实验、完整 CUDA 实现套件或报告中的所有统计值。
- 原 MAIN 与 speedup 工作目录的修改状态保持不变；快照使用独立 Git 仓库和无父提交。

已有源码/报告中的空白和行尾保留，不作为新增算法修改。源文件校验需要保留字节，因此快照禁用 Git 自动行尾转换。
