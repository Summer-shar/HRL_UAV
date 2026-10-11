# v2.4.0 seed 4105 test 面板

协议SHA256：6797ca6538e663389845bf90c63743540feabf8beeab225df9bdd5aed194106a

检查：{"P1": "PASS", "P2": "PASS", "P3": "PASS", "P5": "PASS", "P6": "PASS", "protocol_hash": "PASS", "protocol_hash_recheck": "PASS"}

|候选|参考|候选均值s|参考均值s|相对差%|胜场|Wilcoxon p|
|---|---|---|---|---|---|---|
| Proposed | TFOO | 0.8073471688580799 | 0.7961983272040823 | 1.400259366676604 | 12 | 0.8983173370361328 |
| Proposed | NC | 0.8073471688580799 | 5.007063263641167 | -83.87583446926587 | 20 | 1.9073486328125e-06 |
| Proposed | CA | 0.8073471688580799 | 1.31684462025671 | -38.690779729145795 | 20 | 1.9073486328125e-06 |
| Proposed | TOO | 0.8073471688580799 | 1.0014474514512854 | -19.381973793224773 | 20 | 1.9073486328125e-06 |
| Proposed | Abl_TwoStage | 0.8073471688580799 | 0.8716827215805208 | -7.380615805460589 | 20 | 1.9073486328125e-06 |
| Proposed | Abl_HoverSAC | 0.8073471688580799 | 1.2513105941195313 | -35.47987424927387 | 20 | 1.9073486328125e-06 |
| Proposed | Abl_UpperCVX | 0.8073471688580799 | 0.9265413440972143 | -12.864420567791568 | 20 | 1.9073486328125e-06 |
| Abl_TwoStage | TFOO | 0.8716827215805208 | 0.7961983272040823 | 9.480601980351855 | 3 | 0.000133514404296875 |

## 实测耗时

{
  "preflight_seconds": 11.8823901,
  "test_evaluation_seconds": 2015.3655829,
  "test_per_group_mean_window_wall_seconds": {
    "Proposed": 18.349694834999998,
    "NC": 29.035025399999995,
    "CA": 14.622631959999996,
    "TOO": 14.299291442499998,
    "TFOO": 18.465747685000004,
    "Abl_TwoStage": 18.399990464999984,
    "Abl_HoverSAC": 18.205086648749994,
    "Abl_UpperCVX": 18.659017619999993
  },
  "first_test_window_started_at": "2026-10-04T02:00:07.111280+08:00",
  "summary_seconds": 115.20755410000015,
  "total_wall_seconds_before_report_write": 2142.8370989
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
