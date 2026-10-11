# 正确性基线与等价工程加速

本轮延续当前本地工作区，包含此前未提交的折扣、训练技术检查和版本检查修改。物理场景、任务/分片、FIFO/优先级、服务能力、holding 与能耗定义不变。本轮不启动科研训练。

## 正确性契约

- 下层 reward 仍为本下层步原始物理成本的负值除以固定 reward_scale。
- 每个宏步同时保存 `reward_physical = sum(r_j)` 和 `reward_learning = sum(gamma_lower**j * r_j)`；PPO 的 GAE 只读取 `reward_learning`。指标、成本、holding、能耗和物理 reward 恒等式只使用未折扣量。
- 当前候选参数保留 `gamma_lower=0.99`（0.5 秒）、`gamma_upper=0.99**10`（5 秒）、`lambda=0.95`；不宣称最优或已通过性能验收。对应折扣半衰期约 34.48 秒。
- 每个训练窗口可显式 reset 到独立场景；普通采样结束是 truncated。bootstrap 后继由 reset 前同一轨迹上的合法上层承诺/到达/下层观测产生，不能替换为新场景的初态。
- PPO 检查采样时保存的行为版本：缺失/混合上层版本、混合下层版本均拒绝；该检查不以更新时的当前版本替代采样记录。
- `check` 与 `dry-run` 只报告技术就绪状态。配置确认项不代表用户授权；二者均报告 `training_authorized=false`、`training_started=false`。正式入口仍需显式调用及匹配当前代码、配置、运行时的验证记录。
- 数据 schema 更新到 `hrl_uav_continuing_service_1_1`，新字段与源码/配置哈希共同防止旧目标下的 checkpoint 或验证记录静默混入。

## 测量与交付约定

正确性提交作为性能比较基线；生产研究代码、`tools/` 测量程序和 `docs/evidence/` 实测记录分开提交。原始大型 fixture/trace 位于 `outputs/`，以清单与 SHA256 定位。

使用临时初始化模型和固定样本，不使用科研训练结果。计时与 profiler trace 分开；保留预热次数、原始重复时间、输入规模和内存峰值。少量重复只报告样本与中位数，不据此声称可靠 P95。嵌套分项不能直接相加。

critic 有效 batch 固定 128；actor 与 target 随机采样分块保持 16。工程候选只改变 critic Q 计算分块，保持样本对应、完整 batch 归一化、历史梯度累计、联合裁剪、Adam 与 Polyak 更新规则。已实现的多图合批、PreparedBatch 与 HistoryTape 直接复用。

完整 seed 预算按实际 schedule 逐窗口计数，包括回放为空跳过、actor warmup、阶段切换清池、PPO、开发评价、checkpoint 和日志；外推与实测分开标记，未测部分明确保留不确定性。
