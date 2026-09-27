# HRL UAV V2

可安装的 Python 研究项目：使用分层 PPO 与 SAC 控制无人机运动和任务卸载。当前版本 **2.1.0**，基于已验证的吞吐优化提交 `dbf2b97`；尚未完成600轮科研训练或策略性能验收。

## 安装

建议使用 Python 3.11 的独立环境（项目支持3.9–3.12）。先按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 为显卡和系统安装合适的 CUDA 版 PyTorch，再安装本项目。完整训练和GPU验证需要NVIDIA CUDA；不会自动降为CPU或降低精度。

```powershell
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux: source .venv/bin/activate
python -m pip install --upgrade pip
# 先执行 PyTorch 官方页面给出的适合本机的安装命令
python -m pip install -e ".[dev,baselines]"
hrl-v2 check
```

普通使用可执行 `python -m pip install .`；`dev` 是开发工具，`baselines` 提供CVX对照依赖。不同PyTorch/CUDA版本需要重新验证；安装兼容范围不等于数值后端已逐版验证。历史精确环境见 `requirements/reference-environment.txt`，不会强制别人的解释器安装位置。

## 使用

所有输出都写入工作目录，默认当前目录的 `outputs/`。`--workspace` 放在子命令之前。无参数命令只检查，不启动训练。

```powershell
hrl-v2 --workspace ./outputs check
hrl-v2 --workspace ./outputs dry-run
hrl-v2 --workspace ./outputs validate
hrl-v2 --workspace ./outputs validate-cuda
```

`validate` 执行51项有界实现测试，含临时模型优化器步骤；`validate-cuda` 执行4项CUDA基线对照。首次训练必须有当前源码、配置和数值环境匹配的本地验证记录，发布包不预置“在你电脑上已验证”的标志。

下面命令**会启动完整600轮训练**，需使用者明确执行；本次工程整理没有执行：

```powershell
hrl-v2 --workspace ./outputs train --seed 4101
```

训练种子为4101–4105。默认运行目录是 `outputs/v2_runs/<protocol_hash>/ppo_sac/seed_<id>/`。研究有效batch、更新次数、完整历史、独立编码器和CPU float64分布不变，microbatch默认16。

从该运行 `latest.json` 读取相对路径恢复：

```powershell
hrl-v2 --workspace ./outputs resume "v2_runs/<hash>/ppo_sac/seed_4101/checkpoints/<latest>.pt"
hrl-v2 --workspace ./outputs evaluate --checkpoint "<检查点路径>" --manifest "<test_manifest.json路径>" --output evaluations/final --mode paired --policy stochastic
```

不附带训练好的权重。正式测试仍要求完整600轮末端检查点。移动工作目录可恢复相同版本/数值环境的检查点；算法、软件版本或设备变化不会被静默忽略。

## 项目目录

```text
src/hrl_v2/             V2算法、CLI、内置固定配置与有界验证
src/hrl_uav/            必要的物理实现，保持原内容
src/hrl_v2/_reference/  137d2f1只读参考源码及哈希，无需Git历史
tests/                 安装、资源和协议可移植性测试（CPU）
docs/                  安装、配置、来源、历史吞吐证据
examples/              调用示例
requirements/          开发依赖与历史精确环境
data/fixtures/         本地生成的有界fixture，Git忽略
outputs/               验证、训练、检查点和评价，Git忽略
pyproject.toml         包依赖、构建配置及hrl-v2入口
```

默认配置位于 `src/hrl_v2/configs/`，随wheel分发；配置位置由安装包解析，不依赖当前目录。科学协议已固定，修改算法配置需要显式新版本和重新验证，详见 [配置说明](docs/CONFIGURATION.md)。

## 开发与发布

```powershell
python -m unittest discover -s tests -v
python -m build
# 或 pip wheel . --no-deps --wheel-dir dist
```

GitHub CPU工作流只运行安装/协议/CLI检查，不声称完成GPU或研究训练。完整验证由有CUDA的本机执行。源码发布无需大体积fixture、运行记录或本机环境；这些内容由 `.gitignore` 排除。许可证尚未指定，不擅自添加第三方授权声明。

其他说明：[安装与GPU](docs/INSTALLATION.md)、[工程基准](docs/BENCHMARKS.md)、[版本来源与变更](docs/PROVENANCE.md)。
