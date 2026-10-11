# 随机位置预训练 + 粗交替试点汇总

合同SHA256：`d7421efd446e64947c9b5a29d09509d7a6a15f41b8998a415a85cee444c03f82`。

seed 4104全部完成后才执行4105；3K分支已执行；test评估次数=0。

|seed|H|F|TFOO-dev|判据1|gap fraction|判据2|F/TFOOdev-1|胜场|
|---|---|---|---|---|---|---|---|---|
|4104|H_c4|F_c9|TFOO-200|True|1.040581776231|False|0.586029211232|2|
|4105|H_S300r|F_c6|TFOO-200|True|1.443359514420|True|-0.070934191422|16|

## 曲线末段

|曲线|4104|4105|
|---|---|---|
|new_stage1|0.056639845107; 仍在明显下降（可能没训够）|-0.011971664329; 基本平稳|
|original_stage1|-0.017519565098; 基本平稳|-0.016092243060; 基本平稳|
|D_a|0.164125062117; 仍在明显下降（可能没训够）|-0.025148908850; 基本平稳|
|U_cvx|0.072563137230; 仍在明显下降（可能没训够）|0.081892234644; 仍在明显下降（可能没训够）|
|TFOO_cvx|0.002494650364; 基本平稳|0.022232009450; 仍在明显下降（可能没训够）|
|c1_c5|-0.071178497278; 基本平稳|0.026275935479; 仍在明显下降（可能没训够）|
|c6_c10|-0.018818210316; 基本平稳|0.499787628635; 仍在明显下降（可能没训够）|
|v1_c1_c10|-0.032059988015; 基本平稳|0.000307708423; 基本平稳|
|K1_K5|-2.709913473815; 基本平稳|-1.514798888838; 基本平稳|

## 3K经验池对照

|seed|K最好|c1–c5最好|K均值|c1–c5均值|K1/start-1|c1/start-1|
|---|---|---|---|---|---|---|
|4104|1.410165628291|1.660294725805|2.860276843087|3.186864679006|-0.187855460541|0.763335583380|
|4105|0.947820712386|0.978614720703|1.379608640451|1.148235779998|0.002131140155|0.599623053087|

## 两seed按场景先均值后配对，n=20

```json
{
  "candidate_mean_s": 0.9831636957918087,
  "reference_mean_s": 0.7886816641346008,
  "relative_difference": 0.24659129342205244,
  "wins": 3,
  "losses": 17,
  "ties": 0,
  "wilcoxon_two_sided_p": 0.0002613067626953125,
  "paired_t_two_sided_p": 0.0019514766652848087,
  "sign_two_sided_p": 0.0025768280029296875
}
```

## 每步墙钟(s)

seed 4104:

```json
{
  "RP-E_E5": 144.8996742,
  "RP-E_TFOO-200": 142.9679602,
  "RP-1": 259.64729159999996,
  "RP-U": 137.3557988,
  "RP-L": 70.33749799999998,
  "dev_TFOO-100": 142.8418806,
  "dev_TFOO-180": 141.470078,
  "dev_TFOO-190": 141.6231161,
  "step1": 14662.360154,
  "dev_D-a_TFOO_S100r": 488.2657913000003,
  "dev_D-a_TFOO_S200r": 149.79314890000023,
  "dev_D-a_TFOO_S300r": 190.61125660000107,
  "dev_D-a_SD200_S300r": 187.4276635999995,
  "step2": 1463.2563566000008,
  "step3": 1616.9582938999993,
  "dev_H_S300r": 181.66392079999787,
  "dev_H_c1": 187.74817640000037,
  "dev_H_c2": 344.5054870999993,
  "dev_H_c3": 160.82327040000018,
  "dev_H_c4": 147.1813512000008,
  "dev_H_c5": 145.6637703000015,
  "step4": 1434.4879088000016,
  "step5": 1460.9886536999984,
  "dev_F_H": 149.3536525999989,
  "dev_F_c6": 147.64114859999972,
  "dev_F_c7": 141.73556109999845,
  "dev_F_c8": 142.26224549999824,
  "dev_F_c9": 142.78362750000088,
  "dev_F_c10": 143.50744080000004,
  "dev_D-b_U25_cvx": 142.8865654000001,
  "dev_D-b_U50_cvx": 141.68031580000024,
  "dev_D-b_U75_cvx": 141.2112240000024,
  "dev_D-b_U100_cvx": 140.83103779999874,
  "dev_D-b_U125_cvx": 139.5552839000011,
  "dev_D-b_U150_cvx": 139.20750500000213,
  "dev_D-b_U175_cvx": 138.3575325999991,
  "dev_D-b_U200_cvx": 138.22113420000096,
  "step3K": 1780.6444150999996,
  "dev_K1": 139.63087999999698,
  "dev_K2": 140.37405389999913,
  "dev_K3": 141.96092589999898,
  "dev_K4": 157.73371699999916,
  "dev_K5": 364.5728003000004
}
```

seed 4105:

```json
{
  "RP-E_E5": 140.13271330000134,
  "RP-E_TFOO-200": 139.25123150000218,
  "RP-1": 234.42763240000204,
  "RP-U": 135.79196530000263,
  "RP-L": 70.02782949999892,
  "dev_TFOO-100": 141.4210080999983,
  "dev_TFOO-180": 139.92353150000054,
  "dev_TFOO-190": 139.8784233999977,
  "step1": 9762.4590532,
  "dev_D-a_TFOO_S100r": 131.4180499999984,
  "dev_D-a_TFOO_S200r": 131.52223400000003,
  "dev_D-a_TFOO_S300r": 170.05761429999984,
  "dev_D-a_SD200_S300r": 167.65924620000078,
  "step2": 1365.4513414999983,
  "step3": 1467.6511852999993,
  "dev_H_S300r": 168.05431409999983,
  "dev_H_c1": 145.31993590000093,
  "dev_H_c2": 137.3544022999995,
  "dev_H_c3": 132.19258760000048,
  "dev_H_c4": 134.36964970000008,
  "dev_H_c5": 132.71518280000055,
  "step4": 1340.5286147000006,
  "step5": 1633.0868004999993,
  "dev_F_H": 166.22654989999864,
  "dev_F_c6": 134.3812781000015,
  "dev_F_c7": 419.763944899998,
  "dev_F_c8": 146.47413250000318,
  "dev_F_c9": 156.8662265999992,
  "dev_F_c10": 135.16553919999933,
  "dev_D-b_U25_cvx": 144.63777819999814,
  "dev_D-b_U50_cvx": 143.8072205999997,
  "dev_D-b_U75_cvx": 143.0875218000001,
  "dev_D-b_U100_cvx": 142.85469860000012,
  "dev_D-b_U125_cvx": 142.63367899999866,
  "dev_D-b_U150_cvx": 141.72195210000064,
  "dev_D-b_U175_cvx": 140.8455182999969,
  "dev_D-b_U200_cvx": 140.39915730000212,
  "step3K": 1477.6456671999986,
  "dev_K1": 134.72330110000257,
  "dev_K2": 133.58418110000275,
  "dev_K3": 134.65802429999894,
  "dev_K4": 135.8579568999994,
  "dev_K5": 189.22790600000008
}
```

## RP和全部硬检查

### seed 4104

```json
{
  "RP": {
    "RP-E": {
      "E5": {
        "status": "PASS",
        "mean_s": 0.893591794719829,
        "summary": "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\rp\\RP-E\\E5\\summary.json"
      },
      "TFOO-200": {
        "status": "PASS",
        "mean_s": 0.7623758851102915,
        "summary": "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\rp\\RP-E\\TFOO-200\\summary.json"
      }
    },
    "RP-1": {
      "status": "PASS",
      "episodes": 5,
      "SAC_delta": {
        "triggers": 450,
        "skipped": 90,
        "events": 360,
        "critic_fits": 720,
        "actor_updates": 135,
        "target_updates": 360
      },
      "PPO_batches": null,
      "wall_seconds": 259.64729159999996
    },
    "RP-U": {
      "status": "PASS",
      "episodes": 10,
      "SAC_delta": null,
      "PPO_batches": 2,
      "wall_seconds": 137.3557988
    },
    "RP-L": {
      "status": "PASS",
      "episodes": 3,
      "SAC_delta": {
        "triggers": 270,
        "skipped": 90,
        "events": 180,
        "critic_fits": 360,
        "actor_updates": 45,
        "target_updates": 180
      },
      "PPO_batches": null,
      "wall_seconds": 70.33749799999998
    }
  },
  "hard_checks": {
    "status": "PASS",
    "SAC_counts": {
      "step1": {
        "triggers": 27000,
        "skipped": 90,
        "events": 26910,
        "critic_fits": 53820,
        "actor_updates": 13410,
        "target_updates": 26910
      },
      "step2": null,
      "step3": {
        "triggers": 4500,
        "skipped": 90,
        "events": 4410,
        "critic_fits": 8820,
        "actor_updates": 1800,
        "target_updates": 4410
      },
      "step4": null,
      "step5": {
        "triggers": 4500,
        "skipped": 90,
        "events": 4410,
        "critic_fits": 8820,
        "actor_updates": 1800,
        "target_updates": 4410
      },
      "step3K": {
        "triggers": 4500,
        "skipped": 0,
        "events": 4500,
        "critic_fits": 9000,
        "actor_updates": 1800,
        "target_updates": 4500
      }
    },
    "flushes": {
      "step1": 1,
      "step2": null,
      "step3": 1,
      "step4": null,
      "step5": 1,
      "step3K": 0
    },
    "available_completed": {
      "step1": {
        "min": 0,
        "max": 55
      },
      "step3": {
        "min": 0,
        "max": 49
      },
      "step5": {
        "min": 0,
        "max": 49
      },
      "step3K": {
        "min": 55,
        "max": 55
      }
    },
    "frozen_PPO_all_lower": true,
    "frozen_SAC_all_upper": true,
    "PPO_batches_40": true,
    "all_motion_steps_within20m": true,
    "all_external_seeds_match": true,
    "all_deterministic_stream_metrics_equal": true,
    "contract_unchanged": true
  },
  "raw_hard_check_count": 2627
}
```

### seed 4105

```json
{
  "RP": {
    "RP-E": {
      "E5": {
        "status": "PASS",
        "mean_s": 0.9257924798277818,
        "summary": "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4105\\rp\\RP-E\\E5\\summary.json"
      },
      "TFOO-200": {
        "status": "PASS",
        "mean_s": 0.8149874431589103,
        "summary": "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4105\\rp\\RP-E\\TFOO-200\\summary.json"
      }
    },
    "RP-1": {
      "status": "PASS",
      "episodes": 5,
      "SAC_delta": {
        "triggers": 450,
        "skipped": 90,
        "events": 360,
        "critic_fits": 720,
        "actor_updates": 135,
        "target_updates": 360
      },
      "PPO_batches": null,
      "wall_seconds": 234.42763240000204
    },
    "RP-U": {
      "status": "PASS",
      "episodes": 10,
      "SAC_delta": null,
      "PPO_batches": 2,
      "wall_seconds": 135.79196530000263
    },
    "RP-L": {
      "status": "PASS",
      "episodes": 3,
      "SAC_delta": {
        "triggers": 270,
        "skipped": 90,
        "events": 180,
        "critic_fits": 360,
        "actor_updates": 45,
        "target_updates": 180
      },
      "PPO_batches": null,
      "wall_seconds": 70.02782949999892
    }
  },
  "hard_checks": {
    "status": "PASS",
    "SAC_counts": {
      "step1": {
        "triggers": 27000,
        "skipped": 90,
        "events": 26910,
        "critic_fits": 53820,
        "actor_updates": 13410,
        "target_updates": 26910
      },
      "step2": null,
      "step3": {
        "triggers": 4500,
        "skipped": 90,
        "events": 4410,
        "critic_fits": 8820,
        "actor_updates": 1800,
        "target_updates": 4410
      },
      "step4": null,
      "step5": {
        "triggers": 4500,
        "skipped": 90,
        "events": 4410,
        "critic_fits": 8820,
        "actor_updates": 1800,
        "target_updates": 4410
      },
      "step3K": {
        "triggers": 4500,
        "skipped": 0,
        "events": 4500,
        "critic_fits": 9000,
        "actor_updates": 1800,
        "target_updates": 4500
      }
    },
    "flushes": {
      "step1": 1,
      "step2": null,
      "step3": 1,
      "step4": null,
      "step5": 1,
      "step3K": 0
    },
    "available_completed": {
      "step1": {
        "min": 0,
        "max": 55
      },
      "step3": {
        "min": 0,
        "max": 49
      },
      "step5": {
        "min": 0,
        "max": 49
      },
      "step3K": {
        "min": 55,
        "max": 55
      }
    },
    "frozen_PPO_all_lower": true,
    "frozen_SAC_all_upper": true,
    "PPO_batches_40": true,
    "all_motion_steps_within20m": true,
    "all_external_seeds_match": true,
    "all_deterministic_stream_metrics_equal": true,
    "contract_unchanged": true
  },
  "raw_hard_check_count": 3020
}
```

## 最终Git与协议

```json
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
```

冻结协议SHA256：6797ca6538e663389845bf90c63743540feabf8beeab225df9bdd5aed194106a。

两个seed判据2都通过=False；未自动扩展。

## 脚本实现更正

```json
[
  {
    "stage": "step0",
    "error": "training_summary.json is at stage3 root, not runs/stage3",
    "attempt1": "step0_attempt1",
    "attempt2": "step0_attempt2",
    "diff": "step0_correction.diff",
    "result": "passed"
  },
  {
    "stage": "contract_creation",
    "error": "Python file parser rejects UTF-8 in a very long physical source line; in-memory compile of same UTF-8 text passes",
    "change": "encode string literals as ASCII unicode escapes; values unchanged",
    "diff": "contract_correction.diff",
    "previous_contract_created": false
  },
  {
    "stage": "seed_4105_step1",
    "error": "KeyError: sac",
    "substantive_check": false,
    "explanation": "paused_G1 progress.json omits sac; read the original identity-and-SHA-verified S300 checkpoint learner.sac.counts instead",
    "failed_attempt": "seed_4105/step1_attempt1",
    "new_attempt": "seed_4105/step1_attempt2",
    "failed_attempt_episode_count": 300,
    "failed_attempt_recorded_episode_wall_seconds_sum": 9732.226441399984,
    "original_S300_counts_readonly_verified_for_seeds": [
      4104,
      4105
    ],
    "training_or_check_conditions_changed": false,
    "completed_steps_reused": [
      "seed_4104 complete",
      "seed_4105 RP-E RP-1 RP-U RP-L",
      "seed_4105 TFOO-dev"
    ],
    "diffs": [
      "implementation_error_4105_step1/train_correction.diff",
      "implementation_error_4105_step1/main_retry.diff"
    ],
    "previous_failure": "implementation_error_4105_step1/previous_failure.json"
  }
]
```

## 只读保护终检

旧checkpoint哈希逐项一致：36/36；源码/config/物理场景identity未变；原冻结协议SHA256未变；两棵Git工作树干净。核对时间：2026-10-06T18:22:26.066822+08:00。
