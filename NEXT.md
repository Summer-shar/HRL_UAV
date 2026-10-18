# 下一步说明（第 1 号，2026-10-18）

**给用户电脑上的 Claude。** 读完照做，做完向用户用一两句话汇报，然后结束，不要等实验跑完。

## 规则

- 不修改任何代码文件；不安装或升级任何包；
- `D:\p2_runs_v2`、`D:\p2_runs` 只读；不删除任何文件；
- 同一个输出目录同时只能有一个实验命令在运行；
- 实验命令一律用新窗口启动，启动后不要等待它结束。

## 背景

- 上一次运行在 verify 处停下，是比对脚本的误报，已经在代码里修好。
- 现在代码通过 git 获取，状态由实验程序自动推送到本分支的 `p2_sweeps_v3/`，不再需要下载 zip 或上传文件。
- `D:\HRL_UAV_paper2`（以前从 zip 解压的）不再使用，保留不动即可。

## 步骤

1. **获取代码**：代码放在 `D:\p2_code`，至少要是 commit `38f88e2`。
   - 如果 `D:\p2_code` 不存在：
     `git clone -b paper2-release https://github.com/Summer-shar/HRL_UAV.git D:\p2_code`
   - 如果已存在：`git -C D:\p2_code pull --ff-only`
   - 检查：`git -C D:\p2_code merge-base --is-ancestor 38f88e2 HEAD` 的退出码应为 0。
2. **Python 环境**：用上次运行 verify 的同一个环境（Python 3.9.25，torch 2.7.0.dev20250227+cu128）。找到它的 `python.exe` 完整路径，下面记作 `<PY>`。不要安装任何东西。
3. **确认没有其他实验在运行**：确认没有正在运行 `run.py` 的 python 进程。如果 `D:\p2_sweeps_v3` 已存在，保留它，命令会续跑。
4. **防止睡眠**：`powercfg /change standby-timeout-ac 0`（插电时不睡眠）。
5. **启动可行性阶段**（新窗口，保持打开）：
   ```powershell
   Start-Process powershell -ArgumentList '-NoExit','-Command',"Set-Location D:\p2_code; & '<PY>' run.py feasibility --main D:\p2_runs_v2 --out-base D:\p2_sweeps_v3"
   ```
6. **确认状态已推送**：约 3 分钟后运行：
   ```powershell
   git -C D:\p2_code fetch origin paper2-runs
   git -C D:\p2_code log origin/paper2-runs -3 --format="%h %ci %s"
   ```
   应看到以 `p2_sweeps_v3: feasibility:` 开头的提交。如果没有，打开 `D:\p2_sweeps_v3\FEASIBILITY.md`，把其中“GitHub 推送”那一行和表格告诉用户。多半是 GitHub 登录问题，用户在该电脑上手动 `git push` 一次即可保存登录。
7. **汇报并结束**：向用户说明已启动、用的 `<PY>`、新窗口已打开、GitHub 上已看到状态。然后结束。

## 之后

- 整个可行性阶段推测约半天到一天，期间不需要任何操作。
- 云端 Claude 会从本分支读取进度。有新的说明时，这个文件会更新（编号加一）。
