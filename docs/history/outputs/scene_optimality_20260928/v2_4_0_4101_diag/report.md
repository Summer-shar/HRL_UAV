# D1/D2 执行检查报告（已停止）

| 检查项 | 结果 |
|---|---|
| HEAD | 25d59ff9f7c3e27c060c63d462235a30ce28930a |
| 执行前、执行后 src 状态 | 均为空 |
| 执行前 OUT | 不存在 |
| 执行前相关 Python 进程 | 未发现 |
| hrl_v2 导入路径 | D:/HRL_UAV_V2_speedup/src/hrl_v2/__init__.py |
| configure() | cuda:0 |
| contract identity | 相等 |
| 三份 manifest 摘要 | 相等 |
| 400、600 pending 摘要与评估 identity | 均相等 |
| 400、600 snapshot 与 pending 的 SAC/PPO actor 哈希 | 均相等 |
| 执行退出码 | 1 |
| 本次执行墙钟秒 | 66.8835826 |
| D1 | 未执行 |
| D2 汇总、配对表和图 | 未生成 |
| 执行后诊断及训练相关 Python 进程 | 未发现 |

## 停止点

诊断脚本 diag_4101.py 第 90 行抛出异常：

```text
return dict(n=n,mode=mode,sid=sid,check=check,per_scenario=PooledMetrics().add(metrics).result())
AttributeError: 'NoneType' object has no attribute 'result'
```

完整 traceback 保存在 run_log.txt 和 failure.json。

windows/ 中已保存 N400_hover_01 至 N400_hover_13 的 JSON 和 NPZ，共 26 个文件。本轮未完成 120 个窗口及全部 80 个重放核对，不形成诊断结论。

未修改源码、配置、训练目录或 checkpoint；未训练，未 kill 进程。未自动修复或重跑，已有诊断产物保留。
