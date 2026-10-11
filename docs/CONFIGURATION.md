# Continuing-service 配置

本轮的配置依据是当前目录内的代码与 JSON。`src/hrl_v2/configs/physical_scene.json` 保存物理参数，`src/hrl_v2/configs/v2_1_0.json` 保存控制、采样和保留的学习参数。文件名为了现有资源路径保持不变；当前 schema 为 `hrl_uav_continuing_service_1_0`。

## 四种独立时间尺度

```json
{
  "task": {
    "physical_dt": 0.1,
    "lower_interval": 0.5,
    "upper_interval": 5,
    "rollout_seconds": 900
  }
}
```

| 配置 | 含义 |
|---|---|
| `physical_dt` | 每次真实物理服务推进的秒数 |
| `lower_interval` | 分片比例更新周期 |
| `upper_interval` | 移动请求更新周期 |
| `rollout_seconds` | 人工轨迹分段长度，不是任务 deadline |

`ClockConfig` 验证正数、有限性及整除关系，统一推导 physical/lower/upper 步数和每层包含的下层步数。rollout 必须包含整数个 upper interval；upper 必须包含整数个 lower interval；lower 必须包含整数个 physical step。

| rollout | physical steps | lower decisions | upper intervals |
|---|---:|---:|---:|
| 900 s | 9000 | 1800 | 180 |
| 60 s | 600 | 120 | 12 |
| 5 s | 50 | 10 | 1 |

collector 为最后 lower transition 构造下一次真实 successor 时，还会在截止时刻提交下一 upper 动作并准备该时刻到达。因此可观察到额外一次边界动作和 arrival frame；它们属于下一窗口，没有额外推进 physical time，也不算入当前窗口的 cost 和完成/未完成指标。

物理场景中的 `physical_s`、`routing_s`、deployment `macro_period_s` 必须与控制时钟一致，入口会检查。物理 JSON 中保留的 `report_window_s` 不是 collector 的截止计时源。只改采样长度时，仅修改 `task.rollout_seconds`；不需要更改物理 JSON 或 Python 源码。

## 60 秒配置示例

```python
import json
from pathlib import Path
from hrl_v2.paths import default_config

config = json.loads(default_config().read_text(encoding="utf-8"))
config["task"]["rollout_seconds"] = 60
Path("continuing_60s.json").write_text(
    json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
```

```powershell
python -m hrl_v2 dry-run --config (Resolve-Path ./continuing_60s.json).Path
python -m hrl_v2 fixture --config continuing_60s.json --output data/fixtures/continuing_60s.pkl
```

dry-run 的相对配置路径以 workspace 为基准（默认 outputs），因此这里传入绝对路径；fixture 的相对配置路径以当前工作目录为基准。

fixture 生成 5 条独立初始化的脚本轨迹，`learning_updates=0`；每条轨迹内部的截止是截断。用同一 adapter 连续调用 `collect`，可以继续处理之前的队列，示例见 [examples](../examples/README.md)。

## 成本与学习参数

物理成本保持 `0.09 * holding_task_s + energy_j / 140`；高/低优先级权重保持 `[10, 1]`，两层 reward 都按固定 `reward_scale=100` 缩放。物理参数和权重本轮未调优。

SAC 的 `objective.gamma_micro`、PPO 的 `ppo.gamma` 和 `ppo.gae_lambda` 保留已有数值 1.0，明确标记为**未确认研究参数**。`objective.gamma_macro` 也保留为配置元数据；当前 PPO 实际读取 `ppo.gamma`。物理 JSON 中的 `gamma_per_routing_frame` 不参与单步物理 cost 计算，不能把它混当成 SAC/PPO 的已确认 discount。公式测试可显式传入测试 discount，不改变科研配置。

网络、optimizer、batch、alpha 和 schedule 数值留在配置中供后续研究者审查；它们的存在不代表已确定训练方案。本轮没有选定预训练、curriculum、联合或交替训练安排。

`run.require_research_protocol` 对正式 train/resume 无条件报错，且在创建训练输出、加载恢复 checkpoint 或执行学习之前拒绝。不能通过翻转一个配置标志解除这项当前实现限制。

## 配置身份与边界

代码和配置身份由内容哈希记录，不依赖安装目录。环境检查会核对物理 JSON 和控制时钟；改变物理参数应另行审查，不能沿用本轮验证结论。

现有 Local、Random、CVX 和 centroid baseline 未改动。仅改变 rollout 长度不影响其物理周期解释；若以后改变 lower/upper 控制周期，必须重新审查 baseline 内固定历史窗口、需求预测时长和动作位移假设。

归档 `_reference/` 与 `docs/evidence/` 中其他历史材料的参数及运行结果不构成本轮当前配置的批准或验证。
