> 归档说明：以下内容是当前仓库自带的先前工程记录，不定义本轮持续服务合同，也不代表本轮测试或训练结果。当前说明见 [场景与控制合同](design_specification.md) 和 [本轮验证](CONTINUING_SERVICE_VALIDATION.md)。

# 有界工程基准

原吞吐测量见 `docs/evidence/throughput/throughput_report.md`，其路径是历史来源，不是运行依赖。普通用户无需下载或携带约92MB的本机fixture。

需要复测时显式生成5条完整脚本物理轨迹（45000步、0学习更新）：

```sh
hrl-v2 fixture --output data/fixtures/independent_fixture.pkl
hrl-v2 benchmark --variant baseline --fixture data/fixtures/independent_fixture.pkl --output outputs/baseline.json --sections all --microbatches 16 --full-ppo
hrl-v2 benchmark --variant optimized --fixture data/fixtures/independent_fixture.pkl --output outputs/optimized.json --sections all --microbatches 16 32 --full-ppo
```

两版串行运行，避免GPU争用。基准在临时模型上执行，不是训练任务，不改变科研更新预算。历史普通/拥堵窗口还有冷图缓存补测，见原报告；基准重复窗口中的固定图缓存可能热命中，不应单独当作连续在线吞吐保证。

`hrl-v2 budget --baseline ... --optimized ... --extra-io ... --output outputs/budget.json` 根据完成的计时JSON计算完整seed条件预算。该预算不是600轮墙钟实测。默认训练参数不根据基准自动切换。
