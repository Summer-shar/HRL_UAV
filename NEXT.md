# 下一步说明（第 4 号，2026-10-20 13:10）

**给用户电脑上的 Claude。** 读完照做，做完向用户用一两句话汇报，然后结束，不要等实验跑完。

## 规则

- 不修改任何代码文件；不安装或升级任何包；
- `D:\p2_runs_v2`、`D:\p2_runs`、`D:\p2_sweeps_v3` 只读；不删除任何文件（`D:\p2_zeroshot_v4` 里的已有结果也不删）；
- 新窗口启动后不要等待它结束。

## 背景

- 第一次零样本扫参（请求 v4-1）已于 10 月 20 日早上完成，16 个点全部完成。
- 之后修订了用户数扫参的设计：改为 `terminals`（每区终端数 7、9、11、15，每个终端的业务量不变，终端越多总负载越大）。同时加入逐位一致性检查（标称点对主实验，新代码重算 4 个已有点对原记录），汇总表增加每个种子的数值。代码 commit `4764624`，运行请求 `RUN_REQUEST.json` 的编号为 `v4-2`（commit `6d86acf`）。
- 如果值守程序（`tools/watch.py`）还开着，它会在 3 分钟内自动拉取代码并运行 `run.py zeroshot`：管道冒烟 → 一致性检查（约 30 分钟）→ 4 个新点（约 1.5 小时；已完成的 12 个点自动跳过）→ 汇总。状态推送到 GitHub 分支 `paper2-runs` 的 `p2_zeroshot_v4/ZEROSHOT.md`。

## 步骤

1. **先看是否已经在运行**：查看 `D:\p2_watch\state.json`。
   - 如果其中 `started` 里已有 `v4-2`，或者有 python 进程正在运行 `run.py zeroshot`：说明已经自动开始了，**什么都不要做**，直接跳到第 6 步汇报。
   - 如果有 python 进程正在运行 `tools\watch.py`（值守程序开着），但 `state.json` 里还没有 `v4-2`：等 3–5 分钟再看一次 `state.json`；出现 `v4-2` 后跳到第 6 步。
2. **只有在值守程序没有运行时才继续**：确认没有正在运行 `run.py` 或 `tools\watch.py` 的 python 进程。
3. **更新代码**：
   - `git -C D:\p2_code pull --ff-only`
   - 检查：`git -C D:\p2_code merge-base --is-ancestor 6d86acf HEAD` 的退出码应为 0。
4. **确认防睡眠**：`powercfg /change standby-timeout-ac 0`。
5. **用新窗口启动值守程序**（窗口保持打开，不要关）：
   ```powershell
   Start-Process powershell -ArgumentList '-NoExit','-Command',"Set-Location D:\p2_code; & 'D:\HRL_UAV_V2\.venv\Scripts\python.exe' tools\watch.py --main D:\p2_runs_v2 --zeroshot-out D:\p2_zeroshot_v4 --sweep-out D:\p2_sweeps_v3"
   ```
   约 1 分钟后，新窗口里应出现 `watching origin/paper2-release:RUN_REQUEST.json`，随后出现 `started zeroshot for request v4-2`。
6. **汇报并结束**：用一两句话告诉用户“v4-2 已开始（或：值守程序已启动，v4-2 已开始）”，然后结束。
