# 方案 D seed 4104 数值报告

结束：SMOKE REPORT ONLY; NOT PLAN D RESULTS
selected：{}
last：{}

|组|平均延迟 s|相对 TFOO 改善 %|胜场 /20|Wilcoxon p|配对 t p|符号 p|
|---|---:|---:|---:|---:|---:|---:|
|K-TFOO-4104|0.762375885110|0.000000|0|1.0|None|1.0|

JC 参考值：
{
  "TFOO": {
    "mean_s": 0.7623758851102915,
    "source": "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_jointC_diag_20261010_attempt2\\groups\\K-TFOO-4104\\summary.json"
  },
  "JC_selected": {
    "mean_s": 0.8026326301204698,
    "source": "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_jointC_diag_20261010_attempt2\\groups\\K-JC-4104\\summary.json"
  },
  "JC_upper_cvx400": {
    "mean_s": 0.8378586073489526,
    "source": "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_jointC_diag_20261010_attempt2\\groups\\B-4104-400\\summary.json"
  },
  "JC_upper_cvx_curve": [
    {
      "episode": 220,
      "mean_s": 1.1883902764529133,
      "source": "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_jointC_diag_20261010_attempt2\\groups\\B-4104-220\\summary.json"
    },
    {
      "episode": 240,
      "mean_s": 1.1458402103115028,
      "source": "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_jointC_diag_20261010_attempt2\\groups\\B-4104-240\\summary.json"
    },
    {
      "episode": 260,
      "mean_s": 1.0910611349379513,
      "source": "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_jointC_diag_20261010_attempt2\\groups\\B-4104-260\\summary.json"
    },
    {
      "episode": 280,
      "mean_s": 1.0069584773234417,
      "source": "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_jointC_diag_20261010_attempt2\\groups\\B-4104-280\\summary.json"
    },
    {
      "episode": 300,
      "mean_s": 0.91250756368325,
      "source": "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_jointC_diag_20261010_attempt2\\groups\\B-4104-300\\summary.json"
    },
    {
      "episode": 320,
      "mean_s": 0.8428030777044471,
      "source": "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_jointC_diag_20261010_attempt2\\groups\\B-4104-320\\summary.json"
    },
    {
      "episode": 340,
      "mean_s": 0.8187704873314706,
      "source": "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_jointC_diag_20261010_attempt2\\groups\\B-4104-340\\summary.json"
    },
    {
      "episode": 360,
      "mean_s": 0.8400876259107688,
      "source": "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_jointC_diag_20261010_attempt2\\groups\\B-4104-360\\summary.json"
    },
    {
      "episode": 380,
      "mean_s": 0.8588471522685506,
      "source": "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_jointC_diag_20261010_attempt2\\groups\\B-4104-380\\summary.json"
    },
    {
      "episode": 400,
      "mean_s": 0.8378586073489526,
      "source": "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_jointC_diag_20261010_attempt2\\groups\\B-4104-400\\summary.json"
    }
  ],
  "JC_episode1": {
    "event_sha256": "5e11bd0ecdd46141b870bcad93d75ab7d413ad59e0251ace70d4ae030e341375",
    "sac": {
      "triggers": 90,
      "skipped": 90,
      "events": 0,
      "critic_fits": 0,
      "actor_updates": 0,
      "target_updates": 0
    },
    "sac_delta": {
      "triggers": 90,
      "skipped": 90,
      "events": 0,
      "critic_fits": 0,
      "actor_updates": 0,
      "target_updates": 0
    },
    "critic_residual_mse": null,
    "reward_physical": -102.91121691896974,
    "reward_learning_macro_sum": -98.40137072340731
  }
}

PPO 更新 KL 与接受 epoch 数：
[]

每批正常/附加回报和经验占比：
[]

训练硬检查：
{
  "SAC_counts": {
    "triggers": 36000,
    "skipped": 90,
    "events": 35910,
    "critic_fits": 71820,
    "actor_updates": 17010,
    "target_updates": 35910
  },
  "base_SAC_counts": {
    "triggers": 18000,
    "skipped": 90,
    "events": 17910,
    "critic_fits": 35820,
    "actor_updates": 8910,
    "target_updates": 17910
  },
  "joint_SAC_delta": {
    "triggers": 18000,
    "skipped": 0,
    "events": 18000,
    "critic_fits": 36000,
    "actor_updates": 8100,
    "target_updates": 18000
  },
  "PPO_counts": {
    "batches": 40
  },
  "replay_flushes": 0,
  "hardchecks_passed": true
}

回报低于 −300：{'count': 0, 'episodes': []}

逐场景、分段驻留、未完成任务、cost、距离、队列、每批回报及 PPO 原日志详见 report.json。

## K-TFOO-4104

{
  "per_scenario": {
    "1": {
      "cost_per_s": 6.259570004317766,
      "cost_sum": 11267.226007771978,
      "seconds": 1800.0,
      "truncated_delay_s": 0.666226912928761,
      "weighted_completed_delay_s": 0.6656130019341393,
      "stage_delay_s": {
        "upload": 0.31842997668807854,
        "ready": 0.048168924648513974,
        "u2u": 0.09578557319376134,
        "cpu": 0.2038424383984277
      },
      "pooled_delay_ms": 665.6130019341393,
      "priority": [
        {
          "arrived": 10814,
          "carry_in": 0,
          "served_population": 10814,
          "completed": 10806,
          "unfinished": 8,
          "delay_sum_s": 5689.2000000001635,
          "residence_sum_s": 5694.2000000001635,
          "unfinished_rate": 0.000739781764379508,
          "completion_rate": 0.9992602182356205,
          "mean_delay_ms": 526.4852859522639,
          "truncated_delay_s": 0.5265581653412394,
          "completed_p95_s": 0.9,
          "stage_residence_sum_s": {
            "upload": 2892.26776369617,
            "ready": 508.13223630416354,
            "u2u": 989.5999999999892,
            "cpu": 1304.19999999994
          },
          "oldest_unfinished_age_s": 1.0
        },
        {
          "arrived": 30574,
          "carry_in": 0,
          "served_population": 30574,
          "completed": 30504,
          "unfinished": 70,
          "delay_sum_s": 35338.000000000444,
          "residence_sum_s": 35473.000000000444,
          "unfinished_rate": 0.002289527049126709,
          "completion_rate": 0.9977104729508733,
          "mean_delay_ms": 1158.4710201940875,
          "truncated_delay_s": 1.1602341859096108,
          "completed_p95_s": 2.4000000000000004,
          "stage_residence_sum_s": {
            "upload": 15248.018149348423,
            "ready": 1600.381850652332,
            "u2u": 3390.7999999995172,
            "cpu": 15233.8000000001
          },
          "oldest_unfinished_age_s": 6.0
        }
      ],
      "episodes": 2,
      "windows": 2,
      "completed_parents": 41310,
      "unfinished_parents": 78,
      "unfinished_rate": 0.0018846042331110468,
      "oldest_unfinished_age_s": 6.0,
      "holding_task_s": 92415.00000000016,
      "energy_j": {
        "upload": 1269.0851144064982,
        "u2u": 1766.3820247260874,
        "cpu": 408915.3853987931,
        "propulsion": 206357.71003048285
      },
      "average_power_w": 343.5047569824492,
      "average_unfinished_parents": 22.870666666666676,
      "queue_integrals_physical_grid": {
        "upload_bits": 15362982324.57074,
        "ready_cycles": 4482972000000.0,
        "cpu_cycles": 13414525223461.895,
        "transfer_bits": 1226994204.1989317
      },
      "mean_cutoff_backlog": {
        "upload_bits": 25036512.430359744,
        "ready_cycles": 3520000000.0,
        "transfer_bits": 0.0,
        "cpu_cycles": 10644365517.01957,
        "upload_parents": 21.0,
        "ready_parents": 4.0,
        "transfer_fragments": 0.0,
        "cpu_fragments": 23.0
      },
      "censoring": "unfinished_at_each_cutoff; carry_in_can_recur_in_adjacent_windows",
      "rate_denominator": "sum_of_window_service_populations"
    },
    "2": {
      "cost_per_s": 6.614900179640068,
      "cost_sum": 11906.820323352122,
      "seconds": 1800.0,
      "truncated_delay_s": 0.7009673931599613,
      "weighted_completed_delay_s": 0.7003489183531189,
      "stage_delay_s": {
        "upload": 0.31337668737338786,
        "ready": 0.05150267300904892,
        "u2u": 0.09378165367324635,
        "cpu": 0.24230637910430522
      },
      "pooled_delay_ms": 700.3489183531188,
      "priority": [
        {
          "arrived": 10946,
          "carry_in": 0,
          "served_population": 10946,
          "completed": 10944,
          "unfinished": 2,
          "delay_sum_s": 5758.800000000151,
          "residence_sum_s": 5759.800000000151,
          "unfinished_rate": 0.0001827151470856934,
          "completion_rate": 0.9998172848529143,
          "mean_delay_ms": 526.206140350891,
          "truncated_delay_s": 0.5262013520921023,
          "completed_p95_s": 0.9,
          "stage_residence_sum_s": {
            "upload": 2890.4510074516247,
            "ready": 567.5489925487359,
            "u2u": 1008.5999999999974,
            "cpu": 1293.1999999999275
          },
          "oldest_unfinished_age_s": 0.5
        },
        {
          "arrived": 31124,
          "carry_in": 0,
          "served_population": 31124,
          "completed": 30994,
          "unfinished": 130,
          "delay_sum_s": 40764.80000000038,
          "residence_sum_s": 40946.80000000038,
          "unfinished_rate": 0.004176841023004755,
          "completion_rate": 0.9958231589769952,
          "mean_delay_ms": 1315.248112537923,
          "truncated_delay_s": 1.3156021076982516,
          "completed_p95_s": 2.9000000000000004,
          "stage_residence_sum_s": {
            "upload": 15151.238143184115,
            "ready": 1564.9618568167743,
            "u2u": 3098.1999999996915,
            "cpu": 21132.40000000037
          },
          "oldest_unfinished_age_s": 3.5
        }
      ],
      "episodes": 2,
      "windows": 2,
      "completed_parents": 41938,
      "unfinished_parents": 132,
      "unfinished_rate": 0.003137627763251723,
      "oldest_unfinished_age_s": 3.5,
      "holding_task_s": 98544.79999999999,
      "energy_j": {
        "upload": 1237.4942837516896,
        "u2u": 1687.2413930589735,
        "cpu": 421335.42186338064,
        "propulsion": 206041.54582186794
      },
      "average_power_w": 350.16761297892185,
      "average_unfinished_parents": 25.948111111111043,
      "queue_integrals_physical_grid": {
        "upload_bits": 15610703654.818607,
        "ready_cycles": 4571886000000.0,
        "cpu_cycles": 20853450312756.19,
        "transfer_bits": 897860730.3845328
      },
      "mean_cutoff_backlog": {
        "upload_bits": 7286171.336101768,
        "ready_cycles": 4260000000.0,
        "transfer_bits": 96335.567740903,
        "cpu_cycles": 52107249217.97338,
        "upload_parents": 12.0,
        "ready_parents": 4.0,
        "transfer_fragments": 1.0,
        "cpu_fragments": 98.0
      },
      "censoring": "unfinished_at_each_cutoff; carry_in_can_recur_in_adjacent_windows",
      "rate_denominator": "sum_of_window_service_populations"
    },
    "3": {
      "cost_per_s": 7.137572517675198,
      "cost_sum": 12847.630531815357,
      "seconds": 1800.0,
      "truncated_delay_s": 0.7820146421971788,
      "weighted_completed_delay_s": 0.7819387985546613,
      "stage_delay_s": {
        "upload": 0.41352335723813094,
        "ready": 0.04991770015987744,
        "u2u": 0.09535342991352215,
        "cpu": 0.2232201548856696
      },
      "pooled_delay_ms": 781.9387985546613,
      "priority": [
        {
          "arrived": 11154,
          "carry_in": 0,
          "served_population": 11154,
          "completed": 11148,
          "unfinished": 6,
          "delay_sum_s": 6723.400000000091,
          "residence_sum_s": 6726.400000000091,
          "unfinished_rate": 0.0005379236148466917,
          "completion_rate": 0.9994620763851533,
          "mean_delay_ms": 603.103695730184,
          "truncated_delay_s": 0.6030482338174727,
          "completed_p95_s": 1.1,
          "stage_residence_sum_s": {
            "upload": 3797.5033549469636,
            "ready": 556.4966450533968,
            "u2u": 1032.3999999999414,
            "cpu": 1339.9999999999177
          },
          "oldest_unfinished_age_s": 0.5
        },
        {
          "arrived": 30242,
          "carry_in": 0,
          "served_population": 30242,
          "completed": 30216,
          "unfinished": 26,
          "delay_sum_s": 43563.60000000039,
          "residence_sum_s": 43611.60000000039,
          "unfinished_rate": 0.0008597314992394683,
          "completion_rate": 0.9991402685007605,
          "mean_delay_ms": 1441.7394757744373,
          "truncated_delay_s": 1.4420871635473973,
          "completed_p95_s": 4.0,
          "stage_residence_sum_s": {
            "upload": 20655.135086467046,
            "ready": 1512.4649135337747,
            "u2u": 3195.399999999583,
            "cpu": 18248.600000000828
          },
          "oldest_unfinished_age_s": 6.0
        }
      ],
      "episodes": 2,
      "windows": 2,
      "completed_parents": 41364,
      "unfinished_parents": 32,
      "unfinished_rate": 0.0007730215479756498,
      "oldest_unfinished_age_s": 6.0,
      "holding_task_s": 110875.6000000004,
      "energy_j": {
        "upload": 1469.4761918232891,
        "u2u": 1620.3606824467151,
        "cpu": 397511.58030876983,
        "propulsion": 206859.4542218614
      },
      "average_power_w": 337.4782618916118,
      "average_unfinished_parents": 27.965555555555532,
      "queue_integrals_physical_grid": {
        "upload_bits": 22239893951.247948,
        "ready_cycles": 4392870000000.0,
        "cpu_cycles": 17305824175025.2,
        "transfer_bits": 866331425.0527115
      },
      "mean_cutoff_backlog": {
        "upload_bits": 4995067.592020104,
        "ready_cycles": 1990000000.0,
        "transfer_bits": 362424.9886142267,
        "cpu_cycles": 4027264945.188319,
        "upload_parents": 5.0,
        "ready_parents": 3.0,
        "transfer_fragments": 2.0,
        "cpu_fragments": 13.0
      },
      "censoring": "unfinished_at_each_cutoff; carry_in_can_recur_in_adjacent_windows",
      "rate_denominator": "sum_of_window_service_populations"
    },
    "4": {
      "cost_per_s": 6.7935373117599855,
      "cost_sum": 12228.367161167975,
      "seconds": 1800.0,
      "truncated_delay_s": 0.7345984629888852,
      "weighted_completed_delay_s": 0.7341588898575273,
      "stage_delay_s": {
        "upload": 0.3762958867383565,
        "ready": 0.04901480091701687,
        "u2u": 0.09225209267776706,
        "cpu": 0.2170356826557659
      },
      "pooled_delay_ms": 734.1588898575274,
      "priority": [
        {
          "arrived": 10964,
          "carry_in": 0,
          "served_population": 10964,
          "completed": 10962,
          "unfinished": 2,
          "delay_sum_s": 6254.800000000114,
          "residence_sum_s": 6260.800000000114,
          "unfinished_rate": 0.00018241517694272163,
          "completion_rate": 0.9998175848230573,
          "mean_delay_ms": 570.5893085203534,
          "truncated_delay_s": 0.5710324699015062,
          "completed_p95_s": 1.1,
          "stage_residence_sum_s": {
            "upload": 3415.7175710876013,
            "ready": 532.4824289127451,
            "u2u": 1005.9999999999764,
            "cpu": 1306.5999999999053
          },
          "oldest_unfinished_age_s": 3.0
        },
        {
          "arrived": 30372,
          "carry_in": 0,
          "served_population": 30372,
          "completed": 30326,
          "unfinished": 46,
          "delay_sum_s": 40194.60000000038,
          "residence_sum_s": 40244.60000000038,
          "unfinished_rate": 0.0015145528776504674,
          "completion_rate": 0.9984854471223495,
          "mean_delay_ms": 1325.4171338125825,
          "truncated_delay_s": 1.3250559726063604,
          "completed_p95_s": 3.4000000000000004,
          "stage_residence_sum_s": {
            "upload": 18528.76398313476,
            "ready": 1537.8360168659146,
            "u2u": 2856.3999999997577,
            "cpu": 17321.60000000004
          },
          "oldest_unfinished_age_s": 3.0
        }
      ],
      "episodes": 2,
      "windows": 2,
      "completed_parents": 41288,
      "unfinished_parents": 48,
      "unfinished_rate": 0.0011612154054577125,
      "oldest_unfinished_age_s": 3.0,
      "holding_task_s": 102852.5999999998,
      "energy_j": {
        "upload": 1377.706094761784,
        "u2u": 1575.7193906095868,
        "cpu": 412041.65561346285,
        "propulsion": 206712.29293648526
      },
      "average_power_w": 345.39298557517753,
      "average_unfinished_parents": 25.83633333333337,
      "queue_integrals_physical_grid": {
        "upload_bits": 19752516126.014168,
        "ready_cycles": 4509004000000.0,
        "cpu_cycles": 16276634093087.936,
        "transfer_bits": 711509335.0577517
      },
      "mean_cutoff_backlog": {
        "upload_bits": 15743018.456187727,
        "ready_cycles": 720000000.0,
        "transfer_bits": 3575440.9712168593,
        "cpu_cycles": 2457823756.4066753,
        "upload_parents": 16.0,
        "ready_parents": 1.0,
        "transfer_fragments": 10.0,
        "cpu_fragments": 6.0
      },
      "censoring": "unfinished_at_each_cutoff; carry_in_can_recur_in_adjacent_windows",
      "rate_denominator": "sum_of_window_service_populations"
    },
    "5": {
      "cost_per_s": 6.654885449312808,
      "cost_sum": 11978.793808763054,
      "seconds": 1800.0,
      "truncated_delay_s": 0.7381096308468786,
      "weighted_completed_delay_s": 0.7381603288797643,
      "stage_delay_s": {
        "upload": 0.4059859236411519,
        "ready": 0.05117705540694391,
        "u2u": 0.09546628323258162,
        "cpu": 0.18548036856622435
      },
      "pooled_delay_ms": 738.1603288797643,
      "priority": [
        {
          "arrived": 10582,
          "carry_in": 0,
          "served_population": 10582,
          "completed": 10574,
          "unfinished": 8,
          "delay_sum_s": 6136.000000000128,
          "residence_sum_s": 6141.000000000128,
          "unfinished_rate": 0.000756000756000756,
          "completion_rate": 0.9992439992439992,
          "mean_delay_ms": 580.2912804993501,
          "truncated_delay_s": 0.5803250803250924,
          "completed_p95_s": 1.1,
          "stage_residence_sum_s": {
            "upload": 3380.9918098709927,
            "ready": 549.6081901293319,
            "u2u": 1000.3999999999986,
            "cpu": 1209.9999999999186
          },
          "oldest_unfinished_age_s": 1.0
        },
        {
          "arrived": 30492,
          "carry_in": 0,
          "served_population": 30492,
          "completed": 30480,
          "unfinished": 12,
          "delay_sum_s": 39192.20000000021,
          "residence_sum_s": 39203.20000000021,
          "unfinished_rate": 0.0003935458480913026,
          "completion_rate": 0.9996064541519087,
          "mean_delay_ms": 1285.83333333334,
          "truncated_delay_s": 1.2856880493244198,
          "completed_p95_s": 3.2,
          "stage_residence_sum_s": {
            "upload": 21530.83512466277,
            "ready": 1479.9648753380188,
            "u2u": 3009.1999999996815,
            "cpu": 13183.19999999999
          },
          "oldest_unfinished_age_s": 2.0
        }
      ],
      "episodes": 2,
      "windows": 2,
      "completed_parents": 41054,
      "unfinished_parents": 20,
      "unfinished_rate": 0.00048692603593514147,
      "oldest_unfinished_age_s": 2.0,
      "holding_task_s": 100613.19999999972,
      "energy_j": {
        "upload": 1374.0732999958616,
        "u2u": 1592.2229851302934,
        "cpu": 405303.4326180113,
        "propulsion": 207016.8647364421
      },
      "average_power_w": 341.825885355322,
      "average_unfinished_parents": 25.191222222222187,
      "queue_integrals_physical_grid": {
        "upload_bits": 23936061297.522297,
        "ready_cycles": 4438114000000.0,
        "cpu_cycles": 10730678723271.447,
        "transfer_bits": 680801074.407349
      },
      "mean_cutoff_backlog": {
        "upload_bits": 3768717.674878645,
        "ready_cycles": 2100000000.0,
        "transfer_bits": 0.0,
        "cpu_cycles": 1356993632.4891243,
        "upload_parents": 8.0,
        "ready_parents": 1.0,
        "transfer_fragments": 0.0,
        "cpu_fragments": 3.0
      },
      "censoring": "unfinished_at_each_cutoff; carry_in_can_recur_in_adjacent_windows",
      "rate_denominator": "sum_of_window_service_populations"
    },
    "6": {
      "cost_per_s": 7.097119784854277,
      "cost_sum": 12774.815612737699,
      "seconds": 1800.0,
      "truncated_delay_s": 0.7692665857316281,
      "weighted_completed_delay_s": 0.7681911106663487,
      "stage_delay_s": {
        "upload": 0.38450189053432066,
        "ready": 0.04996786594646394,
        "u2u": 0.10504748982360483,
        "cpu": 0.22974933942726244
      },
      "pooled_delay_ms": 768.1911106663487,
      "priority": [
        {
          "arrived": 10890,
          "carry_in": 0,
          "served_population": 10890,
          "completed": 10880,
          "unfinished": 10,
          "delay_sum_s": 6298.200000000078,
          "residence_sum_s": 6318.200000000078,
          "unfinished_rate": 0.0009182736455463728,
          "completion_rate": 0.9990817263544536,
          "mean_delay_ms": 578.8786764705955,
          "truncated_delay_s": 0.5801836547291165,
          "completed_p95_s": 1.1,
          "stage_residence_sum_s": {
            "upload": 3469.9652696987914,
            "ready": 541.2347303015611,
            "u2u": 1040.399999999974,
            "cpu": 1266.5999999999306
          },
          "oldest_unfinished_age_s": 8.0
        },
        {
          "arrived": 31130,
          "carry_in": 0,
          "served_population": 31130,
          "completed": 31098,
          "unfinished": 32,
          "delay_sum_s": 44486.400000000074,
          "residence_sum_s": 44538.400000000074,
          "unfinished_rate": 0.0010279473176999679,
          "completion_rate": 0.9989720526823,
          "mean_delay_ms": 1430.5228632066394,
          "truncated_delay_s": 1.4307227754577603,
          "completed_p95_s": 3.4000000000000004,
          "stage_residence_sum_s": {
            "upload": 19142.147034533013,
            "ready": 1584.6529654677354,
            "u2u": 4305.799999999644,
            "cpu": 19505.800000000254
          },
          "oldest_unfinished_age_s": 7.0
        }
      ],
      "episodes": 2,
      "windows": 2,
      "completed_parents": 41978,
      "unfinished_parents": 42,
      "unfinished_rate": 0.000999524036173251,
      "oldest_unfinished_age_s": 8.0,
      "holding_task_s": 107720.39999999988,
      "energy_j": {
        "upload": 1385.15526925723,
        "u2u": 1843.1585871512125,
        "cpu": 426937.28391704324,
        "propulsion": 206309.601965892
      },
      "average_power_w": 353.5973331885243,
      "average_unfinished_parents": 28.25366666666667,
      "queue_integrals_physical_grid": {
        "upload_bits": 20503034074.955925,
        "ready_cycles": 4587784000000.0,
        "cpu_cycles": 18986496589514.87,
        "transfer_bits": 1865610594.535097
      },
      "mean_cutoff_backlog": {
        "upload_bits": 15196936.539053505,
        "ready_cycles": 0.0,
        "transfer_bits": 0.0,
        "cpu_cycles": 2477206688.8482485,
        "upload_parents": 14.0,
        "ready_parents": 0.0,
        "transfer_fragments": 0.0,
        "cpu_fragments": 13.0
      },
      "censoring": "unfinished_at_each_cutoff; carry_in_can_recur_in_adjacent_windows",
      "rate_denominator": "sum_of_window_service_populations"
    },
    "7": {
      "cost_per_s": 9.855248182554007,
      "cost_sum": 17739.446728597213,
      "seconds": 1800.0,
      "truncated_delay_s": 1.159810851449532,
      "weighted_completed_delay_s": 1.1262473374617492,
      "stage_delay_s": {
        "upload": 0.8357233687720516,
        "ready": 0.04860231524029081,
        "u2u": 0.09855819901349636,
        "cpu": 0.1769269684237099
      },
      "pooled_delay_ms": 1126.247337461749,
      "priority": [
        {
          "arrived": 11226,
          "carry_in": 0,
          "served_population": 11226,
          "completed": 11192,
          "unfinished": 34,
          "delay_sum_s": 9742.600000000139,
          "residence_sum_s": 10090.600000000139,
          "unfinished_rate": 0.0030286834135043648,
          "completion_rate": 0.9969713165864956,
          "mean_delay_ms": 870.4967834167386,
          "truncated_delay_s": 0.8988597897737519,
          "completed_p95_s": 1.6,
          "stage_residence_sum_s": {
            "upload": 7191.636730743053,
            "ready": 538.7632692573284,
            "u2u": 1077.1999999999075,
            "cpu": 1282.9999999999286
          },
          "oldest_unfinished_age_s": 31.5
        },
        {
          "arrived": 30062,
          "carry_in": 0,
          "served_population": 30062,
          "completed": 29862,
          "unfinished": 200,
          "delay_sum_s": 62255.60000000034,
          "residence_sum_s": 64160.60000000034,
          "unfinished_rate": 0.006652917304237909,
          "completion_rate": 0.9933470826957621,
          "mean_delay_ms": 2084.7766392070303,
          "truncated_delay_s": 2.134275829951445,
          "completed_p95_s": 8.200000000000001,
          "stage_residence_sum_s": {
            "upload": 47025.45398294539,
            "ready": 1529.5460170553843,
            "u2u": 3254.9999999997526,
            "cpu": 12350.599999999951
          },
          "oldest_unfinished_age_s": 31.0
        }
      ],
      "episodes": 2,
      "windows": 2,
      "completed_parents": 41054,
      "unfinished_parents": 234,
      "unfinished_rate": 0.005667506297229219,
      "oldest_unfinished_age_s": 31.5,
      "holding_task_s": 165066.6000000003,
      "energy_j": {
        "upload": 1618.1506310293057,
        "u2u": 1697.7272580197605,
        "cpu": 399332.2385690799,
        "propulsion": 207053.10909651866
      },
      "average_power_w": 338.72290308591533,
      "average_unfinished_parents": 41.250666666666675,
      "queue_integrals_physical_grid": {
        "upload_bits": 61839306562.08342,
        "ready_cycles": 4412616000000.0,
        "cpu_cycles": 9965004842129.562,
        "transfer_bits": 903247746.9575799
      },
      "mean_cutoff_backlog": {
        "upload_bits": 138367657.66686562,
        "ready_cycles": 2520000000.0,
        "transfer_bits": 789199.2426680935,
        "cpu_cycles": 3491677558.8856087,
        "upload_parents": 109.0,
        "ready_parents": 2.0,
        "transfer_fragments": 2.0,
        "cpu_fragments": 11.0
      },
      "censoring": "unfinished_at_each_cutoff; carry_in_can_recur_in_adjacent_windows",
      "rate_denominator": "sum_of_window_service_populations"
    },
    "8": {
      "cost_per_s": 7.500519269523808,
      "cost_sum": 13500.934685142855,
      "seconds": 1800.0,
      "truncated_delay_s": 0.8129485310607748,
      "weighted_completed_delay_s": 0.8104979584964558,
      "stage_delay_s": {
        "upload": 0.4008555119937314,
        "ready": 0.05032328370039954,
        "u2u": 0.09695559542497886,
        "cpu": 0.2648141399416929
      },
      "pooled_delay_ms": 810.4979584964558,
      "priority": [
        {
          "arrived": 11208,
          "carry_in": 0,
          "served_population": 11208,
          "completed": 11202,
          "unfinished": 6,
          "delay_sum_s": 6576.400000000145,
          "residence_sum_s": 6584.400000000145,
          "unfinished_rate": 0.0005353319057815846,
          "completion_rate": 0.9994646680942184,
          "mean_delay_ms": 587.0737368327215,
          "truncated_delay_s": 0.5874732334047239,
          "completed_p95_s": 1.1,
          "stage_residence_sum_s": {
            "upload": 3646.167565174339,
            "ready": 564.83243482601,
            "u2u": 1015.7999999999831,
            "cpu": 1357.5999999999187
          },
          "oldest_unfinished_age_s": 2.0
        },
        {
          "arrived": 30608,
          "carry_in": 0,
          "served_population": 30608,
          "completed": 30522,
          "unfinished": 86,
          "delay_sum_s": 49766.00000000036,
          "residence_sum_s": 50154.00000000036,
          "unfinished_rate": 0.002809722948248824,
          "completion_rate": 0.9971902770517512,
          "mean_delay_ms": 1630.4960356464308,
          "truncated_delay_s": 1.6385912179822386,
          "completed_p95_s": 4.5,
          "stage_residence_sum_s": {
            "upload": 20735.59564361815,
            "ready": 1532.2043563825105,
            "u2u": 3676.3999999995535,
            "cpu": 24209.80000000109
          },
          "oldest_unfinished_age_s": 19.0
        }
      ],
      "episodes": 2,
      "windows": 2,
      "completed_parents": 41724,
      "unfinished_parents": 92,
      "unfinished_rate": 0.002200114788597666,
      "oldest_unfinished_age_s": 19.0,
      "holding_task_s": 115997.99999999984,
      "energy_j": {
        "upload": 1421.5958317805487,
        "u2u": 1686.782873509434,
        "cpu": 424415.844197743,
        "propulsion": 206366.60339402346
      },
      "average_power_w": 352.1615701650313,
      "average_unfinished_parents": 31.521333333333605,
      "queue_integrals_physical_grid": {
        "upload_bits": 22726918063.878723,
        "ready_cycles": 4580744000000.0,
        "cpu_cycles": 24684123239396.094,
        "transfer_bits": 1540735376.5303118
      },
      "mean_cutoff_backlog": {
        "upload_bits": 34286844.66805139,
        "ready_cycles": 720000000.0,
        "transfer_bits": 701472.6980400252,
        "cpu_cycles": 14915375895.192465,
        "upload_parents": 31.0,
        "ready_parents": 1.0,
        "transfer_fragments": 2.0,
        "cpu_fragments": 22.0
      },
      "censoring": "unfinished_at_each_cutoff; carry_in_can_recur_in_adjacent_windows",
      "rate_denominator": "sum_of_window_service_populations"
    },
    "9": {
      "cost_per_s": 7.553692248013918,
      "cost_sum": 13596.646046425052,
      "seconds": 1800.0,
      "truncated_delay_s": 0.8082504299965568,
      "weighted_completed_delay_s": 0.8074564603841187,
      "stage_delay_s": {
        "upload": 0.4096081208467107,
        "ready": 0.05053773398648129,
        "u2u": 0.09703199174406113,
        "cpu": 0.2510725834193287
      },
      "pooled_delay_ms": 807.4564603841186,
      "priority": [
        {
          "arrived": 11402,
          "carry_in": 0,
          "served_population": 11402,
          "completed": 11398,
          "unfinished": 4,
          "delay_sum_s": 6750.400000000091,
          "residence_sum_s": 6761.400000000091,
          "unfinished_rate": 0.00035081564637782847,
          "completion_rate": 0.9996491843536222,
          "mean_delay_ms": 592.2442533777935,
          "truncated_delay_s": 0.5930012278547703,
          "completed_p95_s": 1.0,
          "stage_residence_sum_s": {
            "upload": 3756.2397189603894,
            "ready": 580.3602810399789,
            "u2u": 1087.9999999999695,
            "cpu": 1336.7999999998779
          },
          "oldest_unfinished_age_s": 5.0
        },
        {
          "arrived": 31330,
          "carry_in": 0,
          "served_population": 31330,
          "completed": 31290,
          "unfinished": 40,
          "delay_sum_s": 49795.200000000004,
          "residence_sum_s": 49865.200000000004,
          "unfinished_rate": 0.0012767315671879987,
          "completion_rate": 0.998723268432812,
          "mean_delay_ms": 1591.4093959731547,
          "truncated_delay_s": 1.591611873603575,
          "completed_p95_s": 5.6000000000000005,
          "stage_residence_sum_s": {
            "upload": 21974.143175465513,
            "ready": 1542.0568245352663,
            "u2u": 3223.599999999591,
            "cpu": 23125.400000000653
          },
          "oldest_unfinished_age_s": 6.5
        }
      ],
      "episodes": 2,
      "windows": 2,
      "completed_parents": 42688,
      "unfinished_parents": 44,
      "unfinished_rate": 0.001029673312739867,
      "oldest_unfinished_age_s": 6.5,
      "holding_task_s": 117479.19999999953,
      "energy_j": {
        "upload": 1470.38001745271,
        "u2u": 1642.8564443652524,
        "cpu": 419147.9030712412,
        "propulsion": 206277.39329047664
      },
      "average_power_w": 349.18807379085325,
      "average_unfinished_parents": 31.45922222222229,
      "queue_integrals_physical_grid": {
        "upload_bits": 24072065692.11597,
        "ready_cycles": 4555034000000.0,
        "cpu_cycles": 23806924530239.965,
        "transfer_bits": 856474044.8213267
      },
      "mean_cutoff_backlog": {
        "upload_bits": 15294855.807610268,
        "ready_cycles": 4200000000.0,
        "transfer_bits": 0.0,
        "cpu_cycles": 1400647359.6490428,
        "upload_parents": 16.0,
        "ready_parents": 2.0,
        "transfer_fragments": 0.0,
        "cpu_fragments": 5.0
      },
      "censoring": "unfinished_at_each_cutoff; carry_in_can_recur_in_adjacent_windows",
      "rate_denominator": "sum_of_window_service_populations"
    },
    "10": {
      "cost_per_s": 6.51807588372491,
      "cost_sum": 11732.536590704838,
      "seconds": 1800.0,
      "truncated_delay_s": 0.6819104720716697,
      "weighted_completed_delay_s": 0.6818026040423183,
      "stage_delay_s": {
        "upload": 0.3028520825681131,
        "ready": 0.04703137386147041,
        "u2u": 0.09983009913224056,
        "cpu": 0.2321969165098682
      },
      "pooled_delay_ms": 681.8026040423182,
      "priority": [
        {
          "arrived": 11146,
          "carry_in": 0,
          "served_population": 11146,
          "completed": 11146,
          "unfinished": 0,
          "delay_sum_s": 5730.000000000156,
          "residence_sum_s": 5730.000000000156,
          "unfinished_rate": 0.0,
          "completion_rate": 1.0,
          "mean_delay_ms": 514.0857706800787,
          "truncated_delay_s": 0.5140857706800787,
          "completed_p95_s": 0.9,
          "stage_residence_sum_s": {
            "upload": 2858.359861457313,
            "ready": 502.640138543024,
            "u2u": 1009.6000000000233,
            "cpu": 1359.3999999999062
          },
          "oldest_unfinished_age_s": 0.0
        },
        {
          "arrived": 30976,
          "carry_in": 0,
          "served_population": 30976,
          "completed": 30934,
          "unfinished": 42,
          "delay_sum_s": 39784.600000000304,
          "residence_sum_s": 39828.600000000304,
          "unfinished_rate": 0.0013558884297520662,
          "completion_rate": 0.9986441115702479,
          "mean_delay_ms": 1286.1123682679352,
          "truncated_delay_s": 1.2857889979338941,
          "completed_p95_s": 2.9000000000000004,
          "stage_residence_sum_s": {
            "upload": 14553.440618098624,
            "ready": 1672.5593819021592,
            "u2u": 4123.399999999582,
            "cpu": 19479.200000000525
          },
          "oldest_unfinished_age_s": 3.5
        }
      ],
      "episodes": 2,
      "windows": 2,
      "completed_parents": 42080,
      "unfinished_parents": 42,
      "unfinished_rate": 0.0009971036512986088,
      "oldest_unfinished_age_s": 3.5,
      "holding_task_s": 97128.60000000034,
      "energy_j": {
        "upload": 1211.7043201740637,
        "u2u": 1818.5486359489846,
        "cpu": 414670.8345356722,
        "propulsion": 206735.04137568423
      },
      "average_power_w": 346.90896048193304,
      "average_unfinished_parents": 25.310333333333315,
      "queue_integrals_physical_grid": {
        "upload_bits": 14935781909.599203,
        "ready_cycles": 4506700000000.0,
        "cpu_cycles": 18014946889751.574,
        "transfer_bits": 1852334710.5898283
      },
      "mean_cutoff_backlog": {
        "upload_bits": 2790271.4277135655,
        "ready_cycles": 4920000000.0,
        "transfer_bits": 1077351.0803011153,
        "cpu_cycles": 13931419700.017887,
        "upload_parents": 5.0,
        "ready_parents": 3.0,
        "transfer_fragments": 3.0,
        "cpu_fragments": 20.0
      },
      "censoring": "unfinished_at_each_cutoff; carry_in_can_recur_in_adjacent_windows",
      "rate_denominator": "sum_of_window_service_populations"
    },
    "11": {
      "cost_per_s": 6.821122565368952,
      "cost_sum": 12278.020617664113,
      "seconds": 1800.0,
      "truncated_delay_s": 0.7449589363657253,
      "weighted_completed_delay_s": 0.7380609147280229,
      "stage_delay_s": {
        "upload": 0.37255577066394113,
        "ready": 0.05312048708936246,
        "u2u": 0.09592953135768112,
        "cpu": 0.22335314725476227
      },
      "pooled_delay_ms": 738.060914728023,
      "priority": [
        {
          "arrived": 10858,
          "carry_in": 0,
          "served_population": 10858,
          "completed": 10838,
          "unfinished": 20,
          "delay_sum_s": 6069.400000000105,
          "residence_sum_s": 6162.400000000105,
          "unfinished_rate": 0.001841959845275373,
          "completion_rate": 0.9981580401547246,
          "mean_delay_ms": 560.0110721535435,
          "truncated_delay_s": 0.5675446675262577,
          "completed_p95_s": 1.0,
          "stage_residence_sum_s": {
            "upload": 3304.0371887957504,
            "ready": 581.9628112045876,
            "u2u": 976.3999999999781,
            "cpu": 1299.9999999999147
          },
          "oldest_unfinished_age_s": 9.0
        },
        {
          "arrived": 30716,
          "carry_in": 0,
          "served_population": 30716,
          "completed": 30634,
          "unfinished": 82,
          "delay_sum_s": 41906.800000000345,
          "residence_sum_s": 42145.800000000345,
          "unfinished_rate": 0.0026696184399010287,
          "completion_rate": 0.997330381560099,
          "mean_delay_ms": 1367.9832865443736,
          "truncated_delay_s": 1.3721122541997768,
          "completed_p95_s": 3.3000000000000003,
          "stage_residence_sum_s": {
            "upload": 18855.156742446838,
            "ready": 1579.8432575539575,
            "u2u": 3598.5999999997675,
            "cpu": 18112.20000000022
          },
          "oldest_unfinished_age_s": 10.0
        }
      ],
      "episodes": 2,
      "windows": 2,
      "completed_parents": 41472,
      "unfinished_parents": 102,
      "unfinished_rate": 0.0024534564872275943,
      "oldest_unfinished_age_s": 10.0,
      "holding_task_s": 103769.80000000008,
      "energy_j": {
        "upload": 1391.305058020467,
        "u2u": 1712.6904350106176,
        "cpu": 407285.98974105215,
        "propulsion": 206684.24777877924
      },
      "average_power_w": 342.81901834047915,
      "average_unfinished_parents": 26.83788888888894,
      "queue_integrals_physical_grid": {
        "upload_bits": 19877100518.833336,
        "ready_cycles": 4463448000000.0,
        "cpu_cycles": 16944525835087.592,
        "transfer_bits": 1226769732.206618
      },
      "mean_cutoff_backlog": {
        "upload_bits": 46460869.32904711,
        "ready_cycles": 0.0,
        "transfer_bits": 37344.05412860622,
        "cpu_cycles": 9637890996.011412,
        "upload_parents": 38.0,
        "ready_parents": 0.0,
        "transfer_fragments": 1.0,
        "cpu_fragments": 23.0
      },
      "censoring": "unfinished_at_each_cutoff; carry_in_can_recur_in_adjacent_windows",
      "rate_denominator": "sum_of_window_service_populations"
    },
    "12": {
      "cost_per_s": 7.034492572893238,
      "cost_sum": 12662.086631207829,
      "seconds": 1800.0,
      "truncated_delay_s": 0.7561898192465802,
      "weighted_completed_delay_s": 0.7562980593189381,
      "stage_delay_s": {
        "upload": 0.32773446538558487,
        "ready": 0.05250231596474571,
        "u2u": 0.10490173996282964,
        "cpu": 0.2710512979334452
      },
      "pooled_delay_ms": 756.2980593189382,
      "priority": [
        {
          "arrived": 11152,
          "carry_in": 0,
          "served_population": 11152,
          "completed": 11148,
          "unfinished": 4,
          "delay_sum_s": 6021.00000000009,
          "residence_sum_s": 6023.00000000009,
          "unfinished_rate": 0.0003586800573888092,
          "completion_rate": 0.9996413199426112,
          "mean_delay_ms": 540.0968783638402,
          "truncated_delay_s": 0.5400824964132075,
          "completed_p95_s": 0.9,
          "stage_residence_sum_s": {
            "upload": 3057.2532360534906,
            "ready": 590.1467639468507,
            "u2u": 1027.799999999981,
            "cpu": 1347.7999999999174
          },
          "oldest_unfinished_age_s": 0.5
        },
        {
          "arrived": 30552,
          "carry_in": 0,
          "served_population": 30552,
          "completed": 30532,
          "unfinished": 20,
          "delay_sum_s": 47193.40000000015,
          "residence_sum_s": 47203.40000000015,
          "unfinished_rate": 0.0006546216286986122,
          "completion_rate": 0.9993453783713014,
          "mean_delay_ms": 1545.7028691209273,
          "truncated_delay_s": 1.5450183294056083,
          "completed_p95_s": 4.0,
          "stage_residence_sum_s": {
            "upload": 15989.358605725909,
            "ready": 1557.6413942748463,
            "u2u": 4625.599999999322,
            "cpu": 25030.80000000125
          },
          "oldest_unfinished_age_s": 0.5
        }
      ],
      "episodes": 2,
      "windows": 2,
      "completed_parents": 41680,
      "unfinished_parents": 24,
      "unfinished_rate": 0.0005754843660080568,
      "oldest_unfinished_age_s": 0.5,
      "holding_task_s": 107433.40000000015,
      "energy_j": {
        "upload": 1287.0498696202262,
        "u2u": 1745.1106438820982,
        "cpu": 414967.35186562774,
        "propulsion": 206355.1979938999
      },
      "average_power_w": 346.8637279850166,
      "average_unfinished_parents": 29.570222222222135,
      "queue_integrals_physical_grid": {
        "upload_bits": 16602811770.493734,
        "ready_cycles": 4511752000000.0,
        "cpu_cycles": 25477001234728.805,
        "transfer_bits": 2428492949.1108747
      },
      "mean_cutoff_backlog": {
        "upload_bits": 2383767.912297903,
        "ready_cycles": 7740000000.0,
        "transfer_bits": 0.0,
        "cpu_cycles": 395876265.33899367,
        "upload_parents": 5.0,
        "ready_parents": 5.0,
        "transfer_fragments": 0.0,
        "cpu_fragments": 4.0
      },
      "censoring": "unfinished_at_each_cutoff; carry_in_can_recur_in_adjacent_windows",
      "rate_denominator": "sum_of_window_service_populations"
    },
    "13": {
      "cost_per_s": 7.391638486625799,
      "cost_sum": 13304.949275926438,
      "seconds": 1800.0,
      "truncated_delay_s": 0.785789101791428,
      "weighted_completed_delay_s": 0.7854475387793107,
      "stage_delay_s": {
        "upload": 0.41167817731057466,
        "ready": 0.04860418410652816,
        "u2u": 0.09069094032724906,
        "cpu": 0.23481580004709932
      },
      "pooled_delay_ms": 785.4475387793107,
      "priority": [
        {
          "arrived": 11280,
          "carry_in": 0,
          "served_population": 11280,
          "completed": 11280,
          "unfinished": 0,
          "delay_sum_s": 6543.200000000124,
          "residence_sum_s": 6543.200000000124,
          "unfinished_rate": 0.0,
          "completion_rate": 1.0,
          "mean_delay_ms": 580.0709219858267,
          "truncated_delay_s": 0.5800709219858267,
          "completed_p95_s": 1.0,
          "stage_residence_sum_s": {
            "upload": 3675.3453661502776,
            "ready": 541.4546338500903,
            "u2u": 966.2000000000068,
            "cpu": 1360.199999999902
          },
          "oldest_unfinished_age_s": 0.0
        },
        {
          "arrived": 31554,
          "carry_in": 0,
          "served_population": 31554,
          "completed": 31478,
          "unfinished": 76,
          "delay_sum_s": 47890.80000000014,
          "residence_sum_s": 47999.80000000014,
          "unfinished_rate": 0.0024085694365215187,
          "completion_rate": 0.9975914305634784,
          "mean_delay_ms": 1521.4054260118223,
          "truncated_delay_s": 1.5211954110413939,
          "completed_p95_s": 3.5,
          "stage_residence_sum_s": {
            "upload": 22673.93794598792,
            "ready": 1601.6620540128633,
            "u2u": 3429.5999999996448,
            "cpu": 20294.599999999955
          },
          "oldest_unfinished_age_s": 3.5
        }
      ],
      "episodes": 2,
      "windows": 2,
      "completed_parents": 42758,
      "unfinished_parents": 76,
      "unfinished_rate": 0.0017742914507167203,
      "oldest_unfinished_age_s": 3.5,
      "holding_task_s": 113431.7999999998,
      "energy_j": {
        "upload": 1323.0208082583777,
        "u2u": 1750.9737888619413,
        "cpu": 429346.40855391754,
        "propulsion": 206363.0957321703
      },
      "average_power_w": 354.8797216017823,
      "average_unfinished_parents": 30.301666666666694,
      "queue_integrals_physical_grid": {
        "upload_bits": 26433054063.472904,
        "ready_cycles": 4620174000000.0,
        "cpu_cycles": 19171695281805.445,
        "transfer_bits": 1165402117.8787973
      },
      "mean_cutoff_backlog": {
        "upload_bits": 24063024.424873136,
        "ready_cycles": 4200000000.0,
        "transfer_bits": 0.0,
        "cpu_cycles": 15339076381.933258,
        "upload_parents": 23.0,
        "ready_parents": 2.0,
        "transfer_fragments": 0.0,
        "cpu_fragments": 32.0
      },
      "censoring": "unfinished_at_each_cutoff; carry_in_can_recur_in_adjacent_windows",
      "rate_denominator": "sum_of_window_service_populations"
    },
    "14": {
      "cost_per_s": 7.073000152561909,
      "cost_sum": 12731.400274611437,
      "seconds": 1800.0,
      "truncated_delay_s": 0.7659783454815663,
      "weighted_completed_delay_s": 0.7657167946066988,
      "stage_delay_s": {
        "upload": 0.3638774105188415,
        "ready": 0.04947054787054501,
        "u2u": 0.09750336140400154,
        "cpu": 0.25512702568819756
      },
      "pooled_delay_ms": 765.7167946066988,
      "priority": [
        {
          "arrived": 11098,
          "carry_in": 0,
          "served_population": 11098,
          "completed": 11094,
          "unfinished": 4,
          "delay_sum_s": 6242.800000000108,
          "residence_sum_s": 6245.800000000108,
          "unfinished_rate": 0.0003604253018561903,
          "completion_rate": 0.9996395746981438,
          "mean_delay_ms": 562.7185866234098,
          "truncated_delay_s": 0.5627860875833581,
          "completed_p95_s": 1.0,
          "stage_residence_sum_s": {
            "upload": 3339.1172482363027,
            "ready": 547.8827517640428,
            "u2u": 1012.7999999999743,
            "cpu": 1345.9999999999006
          },
          "oldest_unfinished_age_s": 1.0
        },
        {
          "arrived": 30330,
          "carry_in": 0,
          "served_population": 30330,
          "completed": 30272,
          "unfinished": 58,
          "delay_sum_s": 45700.40000000008,
          "residence_sum_s": 45782.40000000008,
          "unfinished_rate": 0.0019122980547312892,
          "completion_rate": 0.9980877019452687,
          "mean_delay_ms": 1509.6590909090937,
          "truncated_delay_s": 1.5094757665677574,
          "completed_p95_s": 4.0,
          "stage_residence_sum_s": {
            "upload": 18028.344398054458,
            "ready": 1511.855601946287,
            "u2u": 3650.199999999715,
            "cpu": 22592.000000000193
          },
          "oldest_unfinished_age_s": 3.5
        }
      ],
      "episodes": 2,
      "windows": 2,
      "completed_parents": 41366,
      "unfinished_parents": 62,
      "unfinished_rate": 0.0014965723665154001,
      "oldest_unfinished_age_s": 3.5,
      "holding_task_s": 108240.40000000014,
      "energy_j": {
        "upload": 1346.950261458095,
        "u2u": 1726.8462878742412,
        "cpu": 414461.02310710045,
        "propulsion": 206435.75783329576
      },
      "average_power_w": 346.650320827627,
      "average_unfinished_parents": 28.904555555555593,
      "queue_integrals_physical_grid": {
        "upload_bits": 19228984996.480537,
        "ready_cycles": 4513594000000.0,
        "cpu_cycles": 23132915064037.844,
        "transfer_bits": 1428357381.275963
      },
      "mean_cutoff_backlog": {
        "upload_bits": 15064163.929772211,
        "ready_cycles": 0.0,
        "transfer_bits": 1787301.4179032508,
        "cpu_cycles": 12230677129.901112,
        "upload_parents": 16.0,
        "ready_parents": 0.0,
        "transfer_fragments": 2.0,
        "cpu_fragments": 21.0
      },
      "censoring": "unfinished_at_each_cutoff; carry_in_can_recur_in_adjacent_windows",
      "rate_denominator": "sum_of_window_service_populations"
    },
    "15": {
      "cost_per_s": 6.62752462458639,
      "cost_sum": 11929.544324255501,
      "seconds": 1800.0,
      "truncated_delay_s": 0.7274005393012157,
      "weighted_completed_delay_s": 0.724873990843585,
      "stage_delay_s": {
        "upload": 0.33607755820256263,
        "ready": 0.05081626867312525,
        "u2u": 0.10253356212778406,
        "cpu": 0.23797315029776958
      },
      "pooled_delay_ms": 724.8739908435849,
      "priority": [
        {
          "arrived": 10882,
          "carry_in": 0,
          "served_population": 10882,
          "completed": 10866,
          "unfinished": 16,
          "delay_sum_s": 5860.800000000096,
          "residence_sum_s": 5890.800000000096,
          "unfinished_rate": 0.0014703179562580407,
          "completion_rate": 0.9985296820437419,
          "mean_delay_ms": 539.3705135284462,
          "truncated_delay_s": 0.5413343135453129,
          "completed_p95_s": 1.0,
          "stage_residence_sum_s": {
            "upload": 3060.3014119997943,
            "ready": 547.0985880005444,
            "u2u": 1012.1999999999921,
            "cpu": 1271.19999999991
          },
          "oldest_unfinished_age_s": 10.0
        },
        {
          "arrived": 29878,
          "carry_in": 0,
          "served_population": 29878,
          "completed": 29822,
          "unfinished": 56,
          "delay_sum_s": 41774.000000000364,
          "residence_sum_s": 41981.000000000364,
          "unfinished_rate": 0.0018742887743490194,
          "completion_rate": 0.998125711225651,
          "mean_delay_ms": 1400.777949165058,
          "truncated_delay_s": 1.405080661356194,
          "completed_p95_s": 3.8000000000000003,
          "stage_residence_sum_s": {
            "upload": 16010.27104758109,
            "ready": 1577.1289524196811,
            "u2u": 4099.199999999472,
            "cpu": 20294.400000000947
          },
          "oldest_unfinished_age_s": 11.5
        }
      ],
      "episodes": 2,
      "windows": 2,
      "completed_parents": 40688,
      "unfinished_parents": 72,
      "unfinished_rate": 0.0017664376840039254,
      "oldest_unfinished_age_s": 11.5,
      "holding_task_s": 100889.00000000001,
      "energy_j": {
        "upload": 1224.2824110884058,
        "u2u": 1637.9387442579637,
        "cpu": 395041.39209498133,
        "propulsion": 206238.4290887897
      },
      "average_power_w": 335.6344679661763,
      "average_unfinished_parents": 26.59544444444442,
      "queue_integrals_physical_grid": {
        "upload_bits": 17075319386.787102,
        "ready_cycles": 4351762000000.0,
        "cpu_cycles": 19694227565562.797,
        "transfer_bits": 1823594844.0218942
      },
      "mean_cutoff_backlog": {
        "upload_bits": 30967785.403615747,
        "ready_cycles": 550000000.0,
        "transfer_bits": 1459085.7221948602,
        "cpu_cycles": 2963057295.4633827,
        "upload_parents": 27.0,
        "ready_parents": 1.0,
        "transfer_fragments": 3.0,
        "cpu_fragments": 10.0
      },
      "censoring": "unfinished_at_each_cutoff; carry_in_can_recur_in_adjacent_windows",
      "rate_denominator": "sum_of_window_service_populations"
    },
    "16": {
      "cost_per_s": 6.762099129260392,
      "cost_sum": 12171.778432668705,
      "seconds": 1800.0,
      "truncated_delay_s": 0.7550337026643934,
      "weighted_completed_delay_s": 0.7548696130358153,
      "stage_delay_s": {
        "upload": 0.38384359007394503,
        "ready": 0.049921272367140584,
        "u2u": 0.10605490786009458,
        "cpu": 0.21521393236323097
      },
      "pooled_delay_ms": 754.8696130358153,
      "priority": [
        {
          "arrived": 10872,
          "carry_in": 0,
          "served_population": 10872,
          "completed": 10870,
          "unfinished": 2,
          "delay_sum_s": 6272.000000000124,
          "residence_sum_s": 6275.000000000124,
          "unfinished_rate": 0.0001839587932303164,
          "completion_rate": 0.9998160412067697,
          "mean_delay_ms": 577.0009199632128,
          "truncated_delay_s": 0.5771707137601291,
          "completed_p95_s": 1.1,
          "stage_residence_sum_s": {
            "upload": 3428.2676975056906,
            "ready": 535.732302494647,
            "u2u": 1051.7999999999995,
            "cpu": 1259.199999999904
          },
          "oldest_unfinished_age_s": 1.5
        },
        {
          "arrived": 29548,
          "carry_in": 0,
          "served_population": 29548,
          "completed": 29504,
          "unfinished": 44,
          "delay_sum_s": 41606.00000000058,
          "residence_sum_s": 41647.00000000058,
          "unfinished_rate": 0.0014891024773250304,
          "completion_rate": 0.998510897522675,
          "mean_delay_ms": 1410.1816702820154,
          "truncated_delay_s": 1.4094693380262822,
          "completed_p95_s": 4.3,
          "stage_residence_sum_s": {
            "upload": 18790.608537287324,
            "ready": 1545.1914627133237,
            "u2u": 4145.999999999563,
            "cpu": 17165.20000000018
          },
          "oldest_unfinished_age_s": 2.0
        }
      ],
      "episodes": 2,
      "windows": 2,
      "completed_parents": 40374,
      "unfinished_parents": 46,
      "unfinished_rate": 0.0011380504700643245,
      "oldest_unfinished_age_s": 2.0,
      "holding_task_s": 104397.00000000035,
      "energy_j": {
        "upload": 1312.6999714081853,
        "u2u": 1676.8258720104452,
        "cpu": 384624.0974794477,
        "propulsion": 206631.45015047025
      },
      "average_power_w": 330.13615192963147,
      "average_unfinished_parents": 26.623333333333346,
      "queue_integrals_physical_grid": {
        "upload_bits": 20560394776.98896,
        "ready_cycles": 4281872000000.0,
        "cpu_cycles": 16132070895756.557,
        "transfer_bits": 1767275249.2999742
      },
      "mean_cutoff_backlog": {
        "upload_bits": 18271997.332337525,
        "ready_cycles": 720000000.0,
        "transfer_bits": 119314.98770830093,
        "cpu_cycles": 4888568038.493255,
        "upload_parents": 17.0,
        "ready_parents": 1.0,
        "transfer_fragments": 2.0,
        "cpu_fragments": 9.0
      },
      "censoring": "unfinished_at_each_cutoff; carry_in_can_recur_in_adjacent_windows",
      "rate_denominator": "sum_of_window_service_populations"
    },
    "17": {
      "cost_per_s": 6.974688541905509,
      "cost_sum": 12554.439375429916,
      "seconds": 1800.0,
      "truncated_delay_s": 0.7737112142682613,
      "weighted_completed_delay_s": 0.7668422719290751,
      "stage_delay_s": {
        "upload": 0.42268577421216547,
        "ready": 0.049765038509728465,
        "u2u": 0.10106494589039629,
        "cpu": 0.20019545565599337
      },
      "pooled_delay_ms": 766.8422719290751,
      "priority": [
        {
          "arrived": 10922,
          "carry_in": 0,
          "served_population": 10922,
          "completed": 10910,
          "unfinished": 12,
          "delay_sum_s": 6632.0000000000955,
          "residence_sum_s": 6692.0000000000955,
          "unfinished_rate": 0.0010986998718183483,
          "completion_rate": 0.9989013001281817,
          "mean_delay_ms": 607.8826764436385,
          "truncated_delay_s": 0.612708295184041,
          "completed_p95_s": 1.1,
          "stage_residence_sum_s": {
            "upload": 3846.264242866394,
            "ready": 545.935757133963,
            "u2u": 1041.9999999999698,
            "cpu": 1257.7999999999067
          },
          "oldest_unfinished_age_s": 14.0
        },
        {
          "arrived": 29942,
          "carry_in": 0,
          "served_population": 29942,
          "completed": 29848,
          "unfinished": 94,
          "delay_sum_s": 40231.20000000016,
          "residence_sum_s": 40751.20000000016,
          "unfinished_rate": 0.0031394028455013026,
          "completion_rate": 0.9968605971544987,
          "mean_delay_ms": 1347.86920396677,
          "truncated_delay_s": 1.3610046089105656,
          "completed_p95_s": 3.2,
          "stage_residence_sum_s": {
            "upload": 20359.155282249434,
            "ready": 1466.0447177512035,
            "u2u": 3644.399999999631,
            "cpu": 15281.60000000028
          },
          "oldest_unfinished_age_s": 14.5
        }
      ],
      "episodes": 2,
      "windows": 2,
      "completed_parents": 40758,
      "unfinished_parents": 106,
      "unfinished_rate": 0.0025939702427564605,
      "oldest_unfinished_age_s": 14.5,
      "holding_task_s": 107671.19999999978,
      "energy_j": {
        "upload": 1381.013176898333,
        "u2u": 1718.1366400834831,
        "cpu": 396829.8797352332,
        "propulsion": 207072.60159537665
      },
      "average_power_w": 337.2231284153287,
      "average_unfinished_parents": 26.357333333333408,
      "queue_integrals_physical_grid": {
        "upload_bits": 22554386910.011753,
        "ready_cycles": 4384208000000.0,
        "cpu_cycles": 13510769298901.385,
        "transfer_bits": 1212770195.3516896
      },
      "mean_cutoff_backlog": {
        "upload_bits": 48749123.750181325,
        "ready_cycles": 1440000000.0,
        "transfer_bits": 1163346.7183682993,
        "cpu_cycles": 7915520270.0203705,
        "upload_parents": 42.0,
        "ready_parents": 2.0,
        "transfer_fragments": 3.0,
        "cpu_fragments": 12.0
      },
      "censoring": "unfinished_at_each_cutoff; carry_in_can_recur_in_adjacent_windows",
      "rate_denominator": "sum_of_window_service_populations"
    },
    "18": {
      "cost_per_s": 6.389456472292029,
      "cost_sum": 11501.021650125651,
      "seconds": 1800.0,
      "truncated_delay_s": 0.6976494567209688,
      "weighted_completed_delay_s": 0.6975034470619378,
      "stage_delay_s": {
        "upload": 0.3083790469396532,
        "ready": 0.049220260947821004,
        "u2u": 0.09923603677580924,
        "cpu": 0.24081411205770736
      },
      "pooled_delay_ms": 697.5034470619378,
      "priority": [
        {
          "arrived": 10614,
          "carry_in": 0,
          "served_population": 10614,
          "completed": 10612,
          "unfinished": 2,
          "delay_sum_s": 5534.200000000098,
          "residence_sum_s": 5537.200000000098,
          "unfinished_rate": 0.0001884303749764462,
          "completion_rate": 0.9998115696250236,
          "mean_delay_ms": 521.5039577836503,
          "truncated_delay_s": 0.5216883361597981,
          "completed_p95_s": 0.9,
          "stage_residence_sum_s": {
            "upload": 2780.121641673274,
            "ready": 519.8783583270614,
            "u2u": 987.4000000000067,
            "cpu": 1249.799999999903
          },
          "oldest_unfinished_age_s": 1.5
        },
        {
          "arrived": 30254,
          "carry_in": 0,
          "served_population": 30254,
          "completed": 30228,
          "unfinished": 26,
          "delay_sum_s": 39761.20000000011,
          "residence_sum_s": 39783.20000000011,
          "unfinished_rate": 0.0008593904938189992,
          "completion_rate": 0.999140609506181,
          "mean_delay_ms": 1315.3764721450348,
          "truncated_delay_s": 1.3149732266807732,
          "completed_p95_s": 3.1,
          "stage_residence_sum_s": {
            "upload": 14259.835311554318,
            "ready": 1514.5646884464838,
            "u2u": 3661.199999999658,
            "cpu": 20347.599999999908
          },
          "oldest_unfinished_age_s": 2.5
        }
      ],
      "episodes": 2,
      "windows": 2,
      "completed_parents": 40840,
      "unfinished_parents": 28,
      "unfinished_rate": 0.0006851326221004208,
      "oldest_unfinished_age_s": 2.5,
      "holding_task_s": 95155.19999999982,
      "energy_j": {
        "upload": 1177.5015494822871,
        "u2u": 1757.047325530513,
        "cpu": 407221.4048975906,
        "propulsion": 206311.44899905528
      },
      "average_power_w": 342.4818904286992,
      "average_unfinished_parents": 25.177999999999994,
      "queue_integrals_physical_grid": {
        "upload_bits": 14746593669.8095,
        "ready_cycles": 4453480000000.0,
        "cpu_cycles": 20130226511624.94,
        "transfer_bits": 1319789860.8879917
      },
      "mean_cutoff_backlog": {
        "upload_bits": 9714554.151822584,
        "ready_cycles": 7740000000.0,
        "transfer_bits": 0.0,
        "cpu_cycles": 125095801.31740238,
        "upload_parents": 8.0,
        "ready_parents": 5.0,
        "transfer_fragments": 0.0,
        "cpu_fragments": 1.0
      },
      "censoring": "unfinished_at_each_cutoff; carry_in_can_recur_in_adjacent_windows",
      "rate_denominator": "sum_of_window_service_populations"
    },
    "19": {
      "cost_per_s": 6.525010551127228,
      "cost_sum": 11745.01899202901,
      "seconds": 1800.0,
      "truncated_delay_s": 0.7076942864492548,
      "weighted_completed_delay_s": 0.7042488662537102,
      "stage_delay_s": {
        "upload": 0.35750445241334994,
        "ready": 0.04942812254309305,
        "u2u": 0.09790779432360323,
        "cpu": 0.20285391716922493
      },
      "pooled_delay_ms": 704.2488662537102,
      "priority": [
        {
          "arrived": 11032,
          "carry_in": 0,
          "served_population": 11032,
          "completed": 11026,
          "unfinished": 6,
          "delay_sum_s": 5954.0000000001,
          "residence_sum_s": 5987.0000000001,
          "unfinished_rate": 0.0005438723712835388,
          "completion_rate": 0.9994561276287165,
          "mean_delay_ms": 539.9963722111463,
          "truncated_delay_s": 0.5426939811457668,
          "completed_p95_s": 0.9,
          "stage_residence_sum_s": {
            "upload": 3138.833450217634,
            "ready": 539.7665497827154,
            "u2u": 1003.0000000000065,
            "cpu": 1305.3999999998675
          },
          "oldest_unfinished_age_s": 8.5
        },
        {
          "arrived": 29628,
          "carry_in": 0,
          "served_population": 29628,
          "completed": 29542,
          "unfinished": 86,
          "delay_sum_s": 38915.400000000176,
          "residence_sum_s": 39170.400000000176,
          "unfinished_rate": 0.0029026596462805453,
          "completion_rate": 0.9970973403537194,
          "mean_delay_ms": 1317.2906370591083,
          "truncated_delay_s": 1.3220737140542789,
          "completed_p95_s": 3.0,
          "stage_residence_sum_s": {
            "upload": 18643.698604167166,
            "ready": 1519.7013958336313,
            "u2u": 3671.9999999995603,
            "cpu": 15335.000000000015
          },
          "oldest_unfinished_age_s": 11.0
        }
      ],
      "episodes": 2,
      "windows": 2,
      "completed_parents": 40568,
      "unfinished_parents": 92,
      "unfinished_rate": 0.002262666010821446,
      "oldest_unfinished_age_s": 11.0,
      "holding_task_s": 99040.40000000031,
      "energy_j": {
        "upload": 1209.3195298279613,
        "u2u": 1708.33919307547,
        "cpu": 392442.4053248461,
        "propulsion": 206710.96726307107
      },
      "average_power_w": 334.4839062837892,
      "average_unfinished_parents": 25.087444444444493,
      "queue_integrals_physical_grid": {
        "upload_bits": 21012318840.205925,
        "ready_cycles": 4330774000000.0,
        "cpu_cycles": 13979745595135.064,
        "transfer_bits": 1416704869.1682434
      },
      "mean_cutoff_backlog": {
        "upload_bits": 30738226.235081233,
        "ready_cycles": 2820000000.0,
        "transfer_bits": 1576964.6732674257,
        "cpu_cycles": 15371726013.282816,
        "upload_parents": 26.0,
        "ready_parents": 2.0,
        "transfer_fragments": 2.0,
        "cpu_fragments": 33.0
      },
      "censoring": "unfinished_at_each_cutoff; carry_in_can_recur_in_adjacent_windows",
      "rate_denominator": "sum_of_window_service_populations"
    },
    "20": {
      "cost_per_s": 6.401700870363224,
      "cost_sum": 11523.061566653803,
      "seconds": 1800.0,
      "truncated_delay_s": 0.6790083874846117,
      "weighted_completed_delay_s": 0.6790632741018756,
      "stage_delay_s": {
        "upload": 0.34921276062691486,
        "ready": 0.04801049532021474,
        "u2u": 0.09186442618726925,
        "cpu": 0.18992070535023756
      },
      "pooled_delay_ms": 679.0632741018757,
      "priority": [
        {
          "arrived": 10940,
          "carry_in": 0,
          "served_population": 10940,
          "completed": 10936,
          "unfinished": 4,
          "delay_sum_s": 6043.400000000126,
          "residence_sum_s": 6045.400000000126,
          "unfinished_rate": 0.00036563071297989033,
          "completion_rate": 0.9996343692870201,
          "mean_delay_ms": 552.6152158010357,
          "truncated_delay_s": 0.5525959780621688,
          "completed_p95_s": 1.0,
          "stage_residence_sum_s": {
            "upload": 3187.6067144226163,
            "ready": 522.3932855777509,
            "u2u": 1010.1999999999704,
            "cpu": 1325.19999999992
          },
          "oldest_unfinished_age_s": 0.5
        },
        {
          "arrived": 30332,
          "carry_in": 0,
          "served_population": 30332,
          "completed": 30318,
          "unfinished": 14,
          "delay_sum_s": 34416.20000000053,
          "residence_sum_s": 34425.20000000053,
          "unfinished_rate": 0.0004615587498351576,
          "completion_rate": 0.9995384412501649,
          "mean_delay_ms": 1135.1738241308967,
          "truncated_delay_s": 1.1349465910589651,
          "completed_p95_s": 2.4000000000000004,
          "stage_residence_sum_s": {
            "upload": 16920.1303236939,
            "ready": 1484.669676306736,
            "u2u": 2734.3999999998023,
            "cpu": 13286.000000000198
          },
          "oldest_unfinished_age_s": 1.0
        }
      ],
      "episodes": 2,
      "windows": 2,
      "completed_parents": 41254,
      "unfinished_parents": 18,
      "unfinished_rate": 0.0004361310331459585,
      "oldest_unfinished_age_s": 1.0,
      "holding_task_s": 94879.19999999976,
      "energy_j": {
        "upload": 1313.342332911272,
        "u2u": 1640.850195736584,
        "cpu": 413763.3358411673,
        "propulsion": 206634.19234364168
      },
      "average_power_w": 346.30651150747605,
      "average_unfinished_parents": 22.483666666666664,
      "queue_integrals_physical_grid": {
        "upload_bits": 17823729401.49898,
        "ready_cycles": 4530752000000.0,
        "cpu_cycles": 11253256447808.89,
        "transfer_bits": 546870511.8314208
      },
      "mean_cutoff_backlog": {
        "upload_bits": 538204.7798676771,
        "ready_cycles": 6610000000.0,
        "transfer_bits": 0.0,
        "cpu_cycles": 589388932.0368022,
        "upload_parents": 2.0,
        "ready_parents": 6.0,
        "transfer_fragments": 0.0,
        "cpu_fragments": 1.0
      },
      "censoring": "unfinished_at_each_cutoff; carry_in_can_recur_in_adjacent_windows",
      "rate_denominator": "sum_of_window_service_populations"
    }
  },
  "pooled": {
    "cost_per_s": 6.99929273991807,
    "cost_sum": 251974.53863705054,
    "seconds": 36000.0,
    "truncated_delay_s": 0.7629930121902259,
    "weighted_completed_delay_s": 0.7599116977658041,
    "stage_delay_s": {
      "upload": 0.39019533841696874,
      "ready": 0.04985237175126434,
      "u2u": 0.09787397168996967,
      "cpu": 0.22507133033204527
    },
    "pooled_delay_ms": 759.9116977658041,
    "priority": [
      {
        "arrived": 219982,
        "carry_in": 0,
        "served_population": 219982,
        "completed": 219832,
        "unfinished": 150,
        "delay_sum_s": 126832.60000000232,
        "residence_sum_s": 127468.60000000232,
        "unfinished_rate": 0.0006818739715067596,
        "completion_rate": 0.9993181260284932,
        "mean_delay_ms": 576.9524000145672,
        "truncated_delay_s": 0.5794501368293875,
        "completed_p95_s": 1.0,
        "stage_residence_sum_s": {
          "upload": 70116.44885100846,
          "ready": 10914.351148998523,
          "u2u": 20357.79999999965,
          "cpu": 26079.999999998217
        },
        "oldest_unfinished_age_s": 31.5
      },
      {
        "arrived": 609644,
        "carry_in": 0,
        "served_population": 609644,
        "completed": 608404,
        "unfinished": 1240,
        "delay_sum_s": 864536.4000000055,
        "residence_sum_s": 868915.4000000055,
        "unfinished_rate": 0.002033973925766513,
        "completion_rate": 0.9979660260742335,
        "mean_delay_ms": 1420.9906575236282,
        "truncated_delay_s": 1.4252832800782187,
        "completed_p95_s": 3.6,
        "stage_residence_sum_s": {
          "upload": 395075.2677402062,
          "ready": 30914.932259808887,
          "u2u": 71395.39999999249,
          "cpu": 371529.80000000703
        },
        "oldest_unfinished_age_s": 31.0
      }
    ],
    "episodes": 40,
    "windows": 40,
    "completed_parents": 828236,
    "unfinished_parents": 1390,
    "unfinished_rate": 0.0016754537586816229,
    "oldest_unfinished_age_s": 31.5,
    "holding_task_s": 2143601.400000001,
    "energy_j": {
      "upload": 26801.306023406578,
      "u2u": 34005.75940128966,
      "cpu": 8185594.86873416,
      "propulsion": 4131167.005648284
    },
    "average_power_w": 343.82135943908725,
    "average_unfinished_parents": 27.677333333333358,
    "queue_integrals_physical_grid": {
      "upload_bits": 436893957991.3898,
      "ready_cycles": 89479540000000.0,
      "cpu_cycles": 353465042349084.06,
      "transfer_bits": 25737926953.568893
    },
    "mean_cutoff_backlog": {
      "upload_bits": 24485888.542386927,
      "ready_cycles": 2838500000.0,
      "transfer_bits": 637279.1061075983,
      "cpu_cycles": 8813345069.773457,
      "upload_parents": 22.05,
      "ready_parents": 2.25,
      "transfer_fragments": 1.65,
      "cpu_fragments": 18.0
    },
    "censoring": "unfinished_at_each_cutoff; carry_in_can_recur_in_adjacent_windows",
    "rate_denominator": "sum_of_window_service_populations"
  },
  "observations": {
    "executed_5s": {
      "count": 21600,
      "mean_m": 1.4778380914590175,
      "p50_m": 1.4147563662648797,
      "p90_m": 2.510342611795868,
      "p95_m": 2.8283045708997268,
      "p99_m": 3.4301065257717474,
      "max_m": 5.5603862787149625,
      "stationary_fraction": 0.0
    },
    "distance_from_start": {
      "count": 21600,
      "mean_m": 117.4324737283551,
      "p50_m": 102.54935493068618,
      "p90_m": 239.41785036334477,
      "p95_m": 269.7108506790692,
      "p99_m": 317.28960252342205,
      "max_m": 357.3731894685377,
      "stationary_fraction": 0.0
    },
    "movement_ge_19_5m_fraction": 0.0,
    "nearest_uav_distance": {
      "count": 280800,
      "mean_m": 133.44761232330293,
      "p50_m": 116.76809744237553,
      "p90_m": 249.26508527906975,
      "p95_m": 293.2198467084484,
      "p99_m": 363.5400539016616,
      "max_m": 498.8468494679185,
      "stationary_fraction": 0.0
    },
    "upload_task_distance_mean_m": [
      143.2428406144633,
      145.37509967617848
    ],
    "upload_task_window_count": [
      224838.0,
      655022.0
    ],
    "birth_task_distance_mean_m": [
      141.35115182642863,
      139.86940960776158
    ],
    "birth_task_count": [
      219982.0,
      609644.0
    ],
    "assigned_cycles": [
      [
        [
          30318786515390.617,
          8078571485854.661,
          7431201998754.739
        ],
        [
          8724104411214.529,
          32059840717033.945,
          7776774871751.6455
        ],
        [
          6974374745838.883,
          6744279594006.368,
          29545065660154.695
        ]
      ],
      [
        [
          139041508938693.67,
          54539870551569.0,
          53458280509736.76
        ],
        [
          61542713966513.664,
          151962448103541.44,
          59704237929944.39
        ],
        [
          50924768515368.59,
          47071381520478.305,
          138897189964153.3
        ]
      ]
    ],
    "assigned_task_fractions": [
      [
        [
          49190.04413328042,
          12832.903563321683,
          11791.052303397682
        ],
        [
          13367.648306246234,
          50006.787089584395,
          11843.564604169285
        ],
        [
          11264.115556787849,
          10859.86437766025,
          48708.02006555191
        ]
      ],
      [
        [
          114247.0169776176,
          43469.746672021975,
          42703.236350360094
        ],
        [
          47532.161643151245,
          120280.32382195829,
          45599.51453488967
        ],
        [
          40896.237453326336,
          38007.239067358416,
          116054.52347931499
        ]
      ]
    ],
    "cpu_queue_mean": [
      [
        0.4652388888888973,
        0.46289444444445127,
        0.46869444444445246
      ],
      [
        5.3285999999999945,
        5.40858333333333,
        5.121872222222246
      ]
    ],
    "cpu_queue_max": [
      [
        7.0,
        7.0,
        6.0
      ],
      [
        103.0,
        100.0,
        114.0
      ]
    ]
  },
  "streams": {
    "passed": true,
    "scenarios": 20,
    "metrics": "exact Python float values; numeric repr recorded",
    "observations": "exact"
  }
}
