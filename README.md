# paper2-runs：运行状态与下一步说明

这个分支不放代码（代码在 `paper2-release` 分支），只用来在用户电脑和云端 Claude 之间传递信息，不需要手动下载或上传文件。

- `NEXT.md`：云端 Claude 写给用户电脑的下一步说明。用户电脑上的 Claude 读取它并照做：
  `git -C D:\p2_code fetch origin paper2-runs`，然后 `git -C D:\p2_code show origin/paper2-runs:NEXT.md`。
- `p2_sweeps_v3/`：用户电脑运行 `python run.py feasibility` / `python run.py sweep` 时自动推送的状态，每一步开始与结束各推一次，长步骤运行中每 30 分钟一次。内容：
  - `FEASIBILITY.md` / `SWEEP.md`：状态页；
  - `_feasibility/`、`_sweep/`：步骤日志末尾、verify 结果；
  - `results/`：汇总表；
  - `unfinished_job_logs/`：未完成任务日志的末尾；
  - `PUBLISHED.txt`：最后一次推送的时间。

  不含权重、轨迹和完整日志。

- `p2_v26/`：v2.6（分支 `paper2-v26`，`python run_v26.py go`）自动推送的状态：`V26.md` 状态页、`DECISION.md`、`results/` 汇总表与合并后的评估记录。

推送由代码包的 `src/p2/publish.py` 完成，使用用户电脑上已有的 GitHub 登录。推送失败不影响实验。
