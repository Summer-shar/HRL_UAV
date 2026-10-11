# v2.4.0 seed 4102 test 面板

协议SHA256：6797ca6538e663389845bf90c63743540feabf8beeab225df9bdd5aed194106a

检查：{"P1": "PASS", "P2": "PASS", "P3": "PASS", "P5": "PASS", "P6": "PASS", "protocol_hash": "PASS", "protocol_hash_recheck": "PASS"}

|候选|参考|候选均值s|参考均值s|相对差%|胜场|Wilcoxon p|
|---|---|---|---|---|---|---|
| Proposed | TFOO | 0.9134134471794786 | 0.8411420729393928 | 8.592053181638093 | 6 | 0.029575347900390625 |
| Proposed | NC | 0.9134134471794786 | 4.905229492895519 | -81.3787826134859 | 20 | 1.9073486328125e-06 |
| Proposed | CA | 0.9134134471794786 | 1.31684462025671 | -30.636201634676162 | 20 | 1.9073486328125e-06 |
| Proposed | TOO | 0.9134134471794786 | 0.8794294239018041 | 3.8643263864081456 | 7 | 0.1893482208251953 |
| Proposed | Abl_TwoStage | 0.9134134471794786 | 2.77649325035597 | -67.10190283868434 | 14 | 0.021484375 |
| Proposed | Abl_HoverSAC | 0.9134134471794786 | 3.0504680880761623 | -70.0566135817031 | 20 | 1.9073486328125e-06 |
| Proposed | Abl_UpperCVX | 0.9134134471794786 | 0.9180327977356274 | -0.5031792510619137 | 12 | 0.5705966949462891 |
| Abl_TwoStage | TFOO | 2.77649325035597 | 0.8411420729393928 | 230.08612215216417 | 2 | 1.9073486328125e-05 |

## 实测耗时

{
  "preflight_seconds": 11.837323399999999,
  "test_evaluation_seconds": 2365.6294945,
  "test_per_group_mean_window_wall_seconds": {
    "Proposed": 18.54479862375,
    "NC": 30.376488068750007,
    "CA": 15.092720420000003,
    "TOO": 15.193811563750012,
    "TFOO": 19.955066034999994,
    "Abl_TwoStage": 28.736133682500007,
    "Abl_HoverSAC": 28.484030335,
    "Abl_UpperCVX": 19.968967039999978
  },
  "first_test_window_started_at": "2026-10-03T04:21:44.136516+08:00",
  "summary_seconds": 109.97402510000029,
  "total_wall_seconds_before_report_write": 2487.8247291
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
