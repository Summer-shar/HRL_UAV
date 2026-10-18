# 下一步说明（第 2 号，2026-10-18）

**给用户电脑上的 Claude。** 读完照做，做完向用户用一两句话汇报，然后结束，不要等实验跑完。

## 规则

- 不修改任何代码文件；不安装或升级任何包；
- `D:\p2_runs_v2`、`D:\p2_runs` 只读；不删除任何文件；
- 同一个输出目录同时只能有一个实验命令在运行；
- 实验命令一律用新窗口启动，启动后不要等待它结束。

## 背景

第 1 号说明启动的运行在第 2 步失败，GitHub 推送也失败了。原因是代码里有一处只在 Python 3.10 以上可用的写法，而本机环境是 Python 3.9。已在 commit `47bfdf4` 修复，并在 Python 3.9.25 下测试过。原样重跑同一条命令即可。

## 步骤

1. **关掉旧窗口**：关闭第 1 号说明打开的那个 PowerShell 窗口（里面的程序已经停止）。确认没有正在运行 `run.py` 的 python 进程。
2. **更新代码**：
   - `git -C D:\p2_code pull --ff-only`
   - 检查：`git -C D:\p2_code merge-base --is-ancestor 47bfdf4 HEAD` 的退出码应为 0。
3. **用新窗口重新启动**（Python 与上次相同；`D:\p2_sweeps_v3` 保留不动）：
   ```powershell
   Start-Process powershell -ArgumentList '-NoExit','-Command',"Set-Location D:\p2_code; & 'D:\HRL_UAV_V2\.venv\Scripts\python.exe' run.py feasibility --main D:\p2_runs_v2 --out-base D:\p2_sweeps_v3"
   ```
4. **确认状态已推送**：约 3 分钟后运行：
   ```powershell
   git -C D:\p2_code fetch origin paper2-runs
   git -C D:\p2_code log origin/paper2-runs -3 --format="%h %ci %s"
   ```
   应看到以 `p2_sweeps_v3: feasibility:` 开头的提交。如果没有，打开 `D:\p2_sweeps_v3\FEASIBILITY.md`，把“GitHub 推送”那一行和步骤表告诉用户。
5. **汇报并结束**：用一两句话告诉用户“已重新启动，GitHub 上已看到状态”（或看到的错误），然后结束。

## 之后

- 整个可行性阶段推测约半天到一天，期间不需要任何操作。
- 云端 Claude 从本分支读取进度。有新的说明时，这个文件会更新（编号加一）。
