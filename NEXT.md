# 下一步说明（第 5 号，2026-11-11 23:45）

**给用户电脑上的 Claude。** 读完照做，做完向用户用两三句话汇报，然后结束。只等第 1 步（冒烟，10–20 分钟）结束，不要等整个实验跑完。

## 规则

- 不修改任何代码文件；不安装或升级任何包；
- `D:\p2_runs_v2`、`D:\p2_runs`、`D:\p2_sweeps_v3`、`D:\p2_zeroshot_v4` 只读，不删除、不改动其中任何文件；
- **不动现有代码目录 `D:\p2_code`**（不 pull、不切换分支）。新版本放在新目录 `D:\p2_code_v26`；
- 新窗口启动后只等冒烟一步，不等整个实验结束。

## 背景

- 新版本 v2.6 在 GitHub 分支 `paper2-v26`（commit `a39827c` 或之后）。它只新增文件（`run_v26.py`、`src/v26/`、`tests/test_v26.py`、`docs/V26.md`），原有代码逐字节不变。
- 内容：把主实验第三阶段（SAC 微调）从第 100 个窗口的快照继续训练到 300 个窗口，分两个版本（ext：照原样继续；burst：训练窗口中加入区域突发），在验证场景上选择，再对选中的版本做测试、突发实验和零样本扫参；另外评估“无第三阶段”的消融。设计与选择规则见该分支的 `docs/V26.md`。
- 主实验目录只读，所有新文件写到 `D:\p2_v26`。状态自动推送到本分支的 `p2_v26/`（状态页 `p2_v26/V26.md`）。
- 预计总耗时约 5 小时（估计值）。

## 步骤

1. **确认没有别的实验在跑**：查看是否有 python 进程正在运行 `run.py` 或 `run_v26.py`。
   - 有 `run_v26.py go` 在跑：说明已经开始，什么都不要做，跳到第 8 步汇报。
   - 有 `run.py` 的训练或评估在跑：不要启动，向用户汇报后结束。
   - 只有 `tools\watch.py`（值守程序，空闲）在跑：不用管它，继续。
2. **检查主实验的快照是否齐全**。下面 6 个文件都应存在，并记下它们的大小之和 S：
   `D:\p2_runs_v2\seed_4101\train\M3\state.pt`、`...\seed_4101\train\HoverSAC\state.pt`，以及种子 4102、4103 的同名文件。
   - 任何一个不存在：不要启动，把缺少的文件名告诉用户后结束。
3. **检查磁盘空间**：D 盘可用空间应大于 2 × S + 10 GB（快照会各复制两份）。不够就告诉用户后结束。
4. **取得新版本代码**（如果 `D:\p2_code_v26` 已存在，改为 `git -C D:\p2_code_v26 pull --ff-only`）：
   ```powershell
   $url = git -C D:\p2_code remote get-url origin
   git clone --branch paper2-v26 --single-branch $url D:\p2_code_v26
   git -C D:\p2_code_v26 merge-base --is-ancestor a39827c HEAD     # 退出码应为 0
   ```
5. **运行不需要 torch 的自检**（1–2 分钟），最后一行应为 `OK`：
   ```powershell
   Set-Location D:\p2_code_v26
   & 'D:\HRL_UAV_V2\.venv\Scripts\python.exe' tests\test_v26.py
   ```
   - 最后一行不是 `OK`：不要启动，把最后 20 行输出告诉用户后结束。
6. **确认防睡眠**：`powercfg /change standby-timeout-ac 0`。
7. **用新窗口启动**（窗口保持打开，不要关）：
   ```powershell
   Start-Process powershell -ArgumentList '-NoExit','-Command',"Set-Location D:\p2_code_v26; & 'D:\HRL_UAV_V2\.venv\Scripts\python.exe' run_v26.py go --main D:\p2_runs_v2 --out-base D:\p2_v26 --zeroshot-main D:\p2_zeroshot_v4"
   ```
   然后等第 1 步“管道冒烟”结束（通常 10–20 分钟）：每隔几分钟查看一次 `D:\p2_v26\V26.md`，最多等 40 分钟。
   - 第 1 步显示“通过”、后面的步骤开始运行：正常，进入第 8 步。
   - 状态页写着“已停止”：把状态页里“停止原因”一节和 `D:\p2_v26\_v26\smoke.log` 的最后 40 行告诉用户，然后结束（不要自己改代码）。
   - 40 分钟后第 1 步仍是“运行中”：照实告诉用户，然后结束。
8. **汇报并结束**：告诉用户“v2.6 已启动，状态页在 `D:\p2_v26\V26.md`，也会推送到 GitHub 分支 paper2-runs 的 `p2_v26/V26.md`”，然后结束。

## 中断后怎么办

电脑重启或窗口被关掉后，原样再执行第 7 步的命令即可：已完成的任务会跳过，训练从最近的快照（每 10 个窗口一次）继续。
