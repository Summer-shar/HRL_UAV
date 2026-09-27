# 有界验证与连续采样示例

下面示例只使用当前项目的场景和脚本动作，没有训练，也不需要 checkpoint。

```powershell
python -m hrl_v2 check
python -m hrl_v2 dry-run
python -m unittest hrl_v2.tests.test_physical_contract hrl_v2.tests.test_observation_metrics hrl_v2.tests.test_rollout_semantics -v
python -m unittest discover -s tests -v
```

完整有界验证入口为 `python -m hrl_v2 --workspace ./outputs validate`。其中学习接口测试可对临时模型执行有限更新；该入口不执行正式研究训练。

以下代码在同一个世界中收集两个 5 s 窗口。第二次 `collect` 直接消费已保存的下一 upper 动作及其 lower successor，任务、队列、生成器和真实时间继续前进：

```python
import json
import numpy as np
from hrl_v2.paths import default_config
from hrl_v2.timing import ClockConfig
from hrl_v2.physics_adapter import PhysicsAdapter
from hrl_v2.collector import collect

class LocalHover:
    def upper(self, report, observed_arrivals):
        return np.zeros(6), {}

    def lower(self, report, observed_arrivals):
        # priority x source x destination；每个 source 全部本地计算。
        routes = np.tile(np.eye(3), (2, 1, 1)).astype(np.float64)
        # 单元素支持 {0}, {1}, {2} 对应 support ID 0, 1, 2。
        return routes, np.array([0, 1, 2, 0, 1, 2])

scene = json.loads(default_config().with_name("physical_scene.json").read_text())
timing = ClockConfig(.1, .5, 5., 5.)
world = PhysicsAdapter(scene, timing=timing)
world.reset(arrival_seed=1, layout_seed=2, load_seed=3)
policy = LocalHover()
first = collect(world, policy, upper_version="script", lower_version=0,
                identity={"window": 1})
second = collect(world, policy, upper_version="script", lower_version=0,
                 identity={"window": 2})

assert first.terminated[-1] is False and first.truncated[-1] is True
assert first.report(first.successors[-1]) == second.report(second.observations[0])
assert world.env.time_s == 10.
print(first.metrics["cost_per_s"], second.metrics["unfinished_parents"])
```

每个窗口的 metrics 在该截止时刻的下一批到达注入之前计算。`first.successors[-1]` 则是已经提交下一 upper 动作、已准备新到达的同世界下层决策状态，两者边界用途不同。不要用 `reset()` 后的新场景替换 successor。

将 rollout 改为 60 s 的文件配置示例见 [配置说明](../docs/CONFIGURATION.md)。`train` 和 `resume` 当前会直接报错；这套环境验证没有授权或启动正式训练。
