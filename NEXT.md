# 下一步说明（第 3 号，2026-10-20 凌晨）

**给用户电脑上的 Claude。** 读完照做，做完向用户用一两句话汇报，然后结束，不要等实验跑完。

## 规则

- 不修改任何代码文件；不安装或升级任何包；
- `D:\p2_runs_v2`、`D:\p2_runs`、`D:\p2_sweeps_v3` 只读；不删除任何文件；
- 新窗口启动后不要等待它结束。

## 背景

- 可行性阶段已经结束（CPU 9 GHz 通过，U2U 1.5 MHz 未通过）。用户已决定改做零样本泛化扫参（方案 v4）：用主实验在标称参数下训练好的模型，直接在全部 16 个参数点上测试，不再训练。
- 这一步只启动一个“值守程序”（`tools/watch.py`）。它每 3 分钟查看 GitHub 上的 `RUN_REQUEST.json`，云端 Claude 写好代码并测试后会放上请求，值守程序随即拉取代码、开始运行。用户可以直接去睡觉。
- 值守程序只会运行固定的几条实验命令，参数都写在下面的启动命令里。

## 步骤

1. **确认没有实验在运行**：确认没有正在运行 `run.py` 的 python 进程。
2. **更新代码**：
   - `git -C D:\p2_code pull --ff-only`
   - 检查：`git -C D:\p2_code merge-base --is-ancestor c50e086 HEAD` 的退出码应为 0。
3. **确认防睡眠**：`powercfg /change standby-timeout-ac 0`。
4. **用新窗口启动值守程序**（窗口保持打开，不要关）：
   ```powershell
   Start-Process powershell -ArgumentList '-NoExit','-Command',"Set-Location D:\p2_code; & 'D:\HRL_UAV_V2\.venv\Scripts\python.exe' tools\watch.py --main D:\p2_runs_v2 --zeroshot-out D:\p2_zeroshot_v4 --sweep-out D:\p2_sweeps_v3"
   ```
5. **确认在运行**：约 1 分钟后，新窗口里应出现 `watching origin/paper2-release:RUN_REQUEST.json`，后面跟着 `waiting: ...` 或 `started zeroshot ...`。状态文件在 `D:\p2_watch\state.json`。
6. **汇报并结束**：用一两句话告诉用户“值守程序已启动，正在等待/已开始零样本扫参”，然后结束。
