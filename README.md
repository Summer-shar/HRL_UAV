# UAV-MEC 分层控制：持续服务环境

本项目模拟 3 架 UAV 为 39 个固定地面终端持续提供通信和计算服务。本轮交付只确认场景、任务状态转移、因果观测、物理成本和 rollout 截断语义；没有启动正式强化学习训练，也没有策略性能结论。

任务经过上传、等待分片、U2U 转发或本地 CPU、父任务完成。队列跨控制周期和采样窗口保留；只有全部正比例 fragment 完成，父任务才完成。上层控制 UAV 移动，下层实施 **priority-class-level dynamic task splitting**。

默认 physical step 为 0.1 s，下层周期 0.5 s，上层周期 5 s。默认 900 s 只表示一次数据收集窗口：最后一步为 `terminated=False, truncated=True`，系统继续运行，任务价值不会在截止时归零。

## 当前入口

使用已安装本项目及依赖的 Python 环境：

```powershell
python -m hrl_v2 check
python -m hrl_v2 dry-run
python -m hrl_v2 --workspace ./outputs validate
python -m unittest discover -s tests -v
```

`check` 和 `dry-run` 不训练；`validate` 执行有界实现测试，其中学习接口测试会对临时模型做有限优化器更新，这不构成研究训练。完整算法测试的设备要求由测试和 `runtime.configure` 决定，不能把 CPU 配置检查当成 CUDA 验证。

**`train` 和 `resume` 当前无条件 fail-fast。** 需要研究者确认 continuing-service 的 discount、训练组织方式、环境持久化和检查点协议后，才能另外实施训练入口。仅修改 JSON 中的确认标志不会解除禁用。

可以使用脚本动作生成测试轨迹，详见 [示例](examples/README.md)。这不需要训练好的模型或任何其他项目的 checkpoint。

## 代码与说明

| 路径 | 当前职责 |
|---|---|
| `src/hrl_uav/` | 热点终端场景、真实队列、上传/U2U/CPU、运动和能耗 |
| `src/hrl_v2/timing.py` | 四种时间尺度、整除校验、步数推导 |
| `src/hrl_v2/physics_adapter.py` | 物理服务入口、不可变因果报告、已观察到达 |
| `src/hrl_v2/collector.py` | 连续采样窗口、真实 successor、跨窗口 continuation |
| `src/hrl_v2/reports.py` / `features.py` | 任务、fragment、队列和真实 motion phase；无 rollout 倒计时 |
| `src/hrl_v2/metrics.py` | cost/s、完成时延、censored、队列与能耗 |
| `src/hrl_v2/tests/`、`tests/` | 物理、信息、截断、学习公式和配置回归测试 |
| `src/hrl_v2/configs/` | 包内场景与当前配置 |
| `outputs/`、`data/fixtures/` | 本地输出；不作为当前实现正确性的预先证明 |

当前建模定义见 [设计说明](docs/design_specification.md)，可配置时钟和保留参数见 [配置说明](docs/CONFIGURATION.md)，修改范围、测试结果及待研究问题见 [本轮验证交付](docs/CONTINUING_SERVICE_VALIDATION.md)。

`src/hrl_v2/_reference/`、`docs/evidence/` 中除本轮 `continuing_service/` 外的历史材料，以及其他保留的来源、安装、吞吐文档属于归档或特定工程用途材料，不是本轮 continuing-service 模型、训练授权或性能结论的依据；其中与当前文档冲突的有限时域和训练描述不再适用。

项目支持范围以 `pyproject.toml` 为准。已有网络结构、encoder、attention、optimizer、学习率、batch、gradient clipping、alpha 和 baseline 本轮未重新设计。默认配置仍保留尚未确认的研究参数；通过环境测试不代表这些研究参数已经确定。
