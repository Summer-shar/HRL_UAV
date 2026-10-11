# v2.4.0 seed 4104 test 面板

协议SHA256：6797ca6538e663389845bf90c63743540feabf8beeab225df9bdd5aed194106a

检查：{"P1": "PASS", "P2": "PASS", "P3": "PASS", "P5": "PASS", "P6": "PASS", "protocol_hash": "PASS", "protocol_hash_recheck": "PASS"}

|候选|参考|候选均值s|参考均值s|相对差%|胜场|Wilcoxon p|
|---|---|---|---|---|---|---|
| Proposed | TFOO | 0.7980702277519404 | 0.7669025218483826 | 4.064102674801173 | 7 | 0.15364646911621094 |
| Proposed | NC | 0.7980702277519404 | 4.887902043776931 | -83.67254047637861 | 20 | 1.9073486328125e-06 |
| Proposed | CA | 0.7980702277519404 | 1.31684462025671 | -39.39526232059466 | 20 | 1.9073486328125e-06 |
| Proposed | TOO | 0.7980702277519404 | 0.9169712063825664 | -12.966707984178482 | 17 | 0.0003948211669921875 |
| Proposed | Abl_TwoStage | 0.7980702277519404 | 0.8585110290750431 | -7.040189266784546 | 14 | 0.00830841064453125 |
| Proposed | Abl_HoverSAC | 0.7980702277519404 | 1.2699493961938457 | -37.15732058743208 | 20 | 1.9073486328125e-06 |
| Proposed | Abl_UpperCVX | 0.7980702277519404 | 0.8910106157722988 | -10.430895701483955 | 20 | 1.9073486328125e-06 |
| Abl_TwoStage | TFOO | 0.8585110290750431 | 0.7669025218483826 | 11.94526091867143 | 2 | 2.6702880859375e-05 |

## 实测耗时

{
  "preflight_seconds": 12.1521915,
  "test_evaluation_seconds": 1927.5222447,
  "test_per_group_mean_window_wall_seconds": {
    "Proposed": 16.50928928375,
    "NC": 27.9481353625,
    "CA": 13.704999998749997,
    "TOO": 13.879962683749998,
    "TFOO": 17.998059760000007,
    "Abl_TwoStage": 17.647873577500004,
    "Abl_HoverSAC": 17.76705100250001,
    "Abl_UpperCVX": 18.184733485000002
  },
  "first_test_window_started_at": "2026-10-03T19:09:12.955502+08:00",
  "summary_seconds": 119.41082529999994,
  "total_wall_seconds_before_report_write": 2059.585603
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
