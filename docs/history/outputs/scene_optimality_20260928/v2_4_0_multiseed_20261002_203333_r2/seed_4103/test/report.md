# v2.4.0 seed 4103 test 面板

协议SHA256：6797ca6538e663389845bf90c63743540feabf8beeab225df9bdd5aed194106a

检查：{"P1": "PASS", "P2": "PASS", "P3": "PASS", "P5": "PASS", "P6": "PASS", "protocol_hash": "PASS", "protocol_hash_recheck": "PASS"}

|候选|参考|候选均值s|参考均值s|相对差%|胜场|Wilcoxon p|
|---|---|---|---|---|---|---|
| Proposed | TFOO | 7.560720106158246 | 0.7689560310519243 | 883.24478914812 | 4 | 0.0003223419189453125 |
| Proposed | NC | 7.560720106158246 | 5.084511501483164 | 48.70101294790592 | 13 | 0.956329345703125 |
| Proposed | CA | 7.560720106158246 | 1.31684462025671 | 474.15430718654835 | 13 | 0.6215133666992188 |
| Proposed | TOO | 7.560720106158246 | 0.7942971347287995 | 851.87553568851 | 5 | 0.0005855560302734375 |
| Proposed | Abl_TwoStage | 7.560720106158246 | 0.7750896578501758 | 875.4639388595386 | 4 | 0.001209259033203125 |
| Proposed | Abl_HoverSAC | 7.560720106158246 | 1.235241590669648 | 512.0843212589236 | 13 | 0.6215133666992188 |
| Proposed | Abl_UpperCVX | 7.560720106158246 | 0.8043778278669771 | 839.94635906456 | 9 | 0.082550048828125 |
| Abl_TwoStage | TFOO | 0.7750896578501758 | 0.7689560310519243 | 0.7976563744302334 | 12 | 0.8408222198486328 |

## 实测耗时

{
  "preflight_seconds": 11.9279987,
  "test_evaluation_seconds": 2574.955809,
  "test_per_group_mean_window_wall_seconds": {
    "Proposed": 53.631219895,
    "NC": 30.120619396249992,
    "CA": 15.113938897499981,
    "TOO": 15.02150512625,
    "TFOO": 19.84267946750001,
    "Abl_TwoStage": 19.20855689000002,
    "Abl_HoverSAC": 19.40680109124999,
    "Abl_UpperCVX": 19.69223444124999
  },
  "first_test_window_started_at": "2026-10-03T10:55:48.364249+08:00",
  "summary_seconds": 110.15138709999974,
  "total_wall_seconds_before_report_write": 2697.4522294
}

## 最终Git状态

{
  "D:\\HRL_UAV_V2": {
    "status": "",
    "branch": "codex/v2-lite-paper-caliber",
    "head": "a9953dc36d9d65f99bcc35044cdfc250ca11bb49"
  },
  "D:\\HRL_UAV_V2_speedup": {
    "status": "",
    "branch": "codex/v2-lite-speedup",
    "head": "25d59ff9f7c3e27c060c63d462235a30ce28930a"
  }
}

脚本错误与修正：无。
