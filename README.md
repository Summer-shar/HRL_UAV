# UAV-MEC 分层强化学习：代码与研究交接

这是 2026-10-12 整理的代码阅读快照：上层 PPO 控制 UAV 移动，下层 SAC 按优先级和源 UAV 动态拆分任务，研究优先级加权时延与能耗的联合优化。包含核心代码、最新实验代码、历史报告和图表，排除训练数据、模型权重和旧 Git 历史。

**给 Claude / AI：先读 [CLAUDE.md](CLAUDE.md)，再读 [项目目标与上下文](docs/PROJECT_CONTEXT.md)、[版本地图](docs/VERSION_MAP.md) 和 [历史报告索引](docs/history/INDEX.md)。**

根目录 `src/` 是 v2.3.0；最新 v2.4.0 核心代码在 [research/speedup](research/speedup)，后续实验驱动在 [research/experiments](research/experiments)。历史评价含负结果；不能从下面保留的早期说明推断当前训练状态。

## 网页端 Claude 建议提示词

> 请先阅读 CLAUDE.md、docs/PROJECT_CONTEXT.md、docs/VERSION_MAP.md 和 docs/history/INDEX.md，再分析 research/speedup/src 以及相应实验驱动。先说明项目目标、上下层算法、物理环境、观测/动作/奖励、训练与评价协议、版本演进、已知负结果和当前瓶颈。区分源码事实、历史报告和推断；注明分析的版本。需要修改前请列出所依据的文件。

私有仓库需要在 Claude 端授权 GitHub 访问，或上传源码包。请明确选择 `codex/claude-handoff-20261012` 分支；原仓库默认分支保留原状。

## 安装与只读检查

Python 支持范围以各项目 `pyproject.toml` 为准（>=3.9,<3.13）。在独立环境安装适合设备的 PyTorch 后，可安装根目录项目；分析 v2.4.0 则从 `research/speedup` 安装。以下命令不启动研究训练：

```powershell
python -m pip install -e ".[dev,baselines]"
python -m hrl_v2 check
python -m hrl_v2 dry-run
python -m unittest discover -s tests -v
```

完整实现验证有设备要求，CPU 检查不代表 CUDA 数值验收。本快照未包含续训所需 checkpoint/replay 或历史基线逐场景输入，不能直接恢复旧实验。请勿直接运行保留的 train_4101.py 或历史实验驱动；先核对源码、配置、路径和冻结协议。

## 历史文档

[早期首页原文](docs/archive/ROOT_README_before_handoff.md)保留作历史记录，其中训练状态和物理参数不能替代当前版本的源码与配置。
