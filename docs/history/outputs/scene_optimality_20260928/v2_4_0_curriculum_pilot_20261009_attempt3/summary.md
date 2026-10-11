# seed 4104 课程式预训练试点汇总

结束时间：2026-10-09T10:15:44.842221+08:00；合同SHA256：b76d59e8af1a50673322b77b110baff8a20b486659b4b2cc6c83e3bcd105edf7。

判据1通过=False；判据2通过=False；两项同时通过=False。

硬检查=PASS；实现修正记录如下。

|组|验证均值s|相对TFOO-200差|胜场/20|Wilcoxon双侧p|配对t双侧p|符号双侧p|
|---|---|---|---|---|---|---|
|TFOO-200|0.762375885110|0.000000000000|0|1.000000000000|1.000000000000|1.000000000000|
|E5|0.893591794720|0.172114454526|0|0.000001907349|0.000000803082|0.000001907349|
|A-F|0.812441233583|0.065670162778|8|0.032768249512|0.015760373609|0.503444671631|
|A-H|0.925054624037|0.213383899077|0|0.000001907349|0.000015201348|0.000001907349|
|A-U200+cvx|0.926094611423|0.214748039006|0|0.000001907349|0.000001382797|0.000001907349|
|A-U100+cvx|1.017733424977|0.334949655222|0|0.000001907349|0.000000166788|0.000001907349|
|DG-SAC300|0.793830529050|0.041258707882|8|0.388376235962|0.456721156769|0.503444671631|
|DG-v1c8|0.653153623155|-0.143265630627|18|0.000019073486|0.000028414755|0.000402450562|
|TFOO-200+A-S200h|0.793696546078|0.041082963902|6|0.015312194824|0.009610154478|0.115318298340|
|TFOO-200+A-S300c|0.830179700735|0.088937513566|9|0.069580078125|0.015730420651|0.823802947998|

各数字来源、逐场景表、三条训练曲线、检查与计数见：D:\HRL_UAV_V2\outputs\scene_optimality_20260928\v2_4_0_curriculum_pilot_20261009_attempt3\seed_4104\report.md；机器可读记录：D:\HRL_UAV_V2\outputs\scene_optimality_20260928\v2_4_0_curriculum_pilot_20261009_attempt3\seed_4104\report.json。

总墙钟=32061.379085 s；阶段计时之和（含复用旧RP，非本次总耗时）=32443.530932 s。

A-H=H_A-c2，A-F=F_A-c6。诊断组不参与选模型；未自动扩展到4105或其他阶段。

## 实现修正

```json
{
  "time": "2026-10-09T01:17:06.763050+08:00",
  "scope": "attempt3 path guard only; no numerical, RNG, training or evaluation changes",
  "user_authorization": "User explicitly authorized repairing and rerunning after attempt2 failure, with error notification.",
  "previous_failed_attempt": "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2",
  "previous_corrections_readonly_reference": {
    "path": "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\implementation_corrections.json",
    "sha256": "86cfcba9c770aa025f9da05c7e803c4fb27c0221d28bca8c4162da7e77730b03"
  },
  "original_failure": {
    "stage": "seed_4104_dev_DG-SAC300",
    "error": "RuntimeError",
    "message": "Write outside CPIL: \\\\?\\D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\evaluations\\DG-SAC300\\windows\\01_2.json",
    "time": "2026-10-08T18:20:32.249663+08:00",
    "substantive_check": false
  },
  "fix": "Resolve candidate and CPIL independently; normalize equivalent Windows extended DOS/UNC anchors only in comparison keys; retain original resolved candidate for long-path I/O.",
  "diff": "output_guard_correction.diff",
  "diff_sha256": "ebb64b465be3cf2548d359472080be18a4651693200fac9ee94a1efe93ecf574",
  "tests": {
    "path": "output_guard_tests.json",
    "status": "PASS",
    "count": 14,
    "real_directory_junction_escape_rejected": true,
    "UNC_tests": "isolated resolved-path representation tests; no network access"
  },
  "review": "output_guard_review.json",
  "authoring_notes": [
    "Initial in-memory authoring of test script had incorrectly escaped Windows literals and failed compile before any test file was written; replaced test literals with chr(92) constants. No experiment stage ran or was retried."
  ],
  "all_previous_files_readonly": true
}
```

## 最终只读和Git复核

```json
{
  "final_readonly_checks": {
    "status": "PASS",
    "inputs": {
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\cp_contract.json": {
        "expected": "4b077f6025dc0b0a296071c6c273dbd0bb6cc9af603272eb6adf27f17b2fe396",
        "actual": "4b077f6025dc0b0a296071c6c273dbd0bb6cc9af603272eb6adf27f17b2fe396",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\cp_contract.sha256": {
        "expected": "e830fb53002aa3e9d6da112f89c4cc00ae1762e3673b60db8b6c3c4c5f8297b9",
        "actual": "e830fb53002aa3e9d6da112f89c4cc00ae1762e3673b60db8b6c3c4c5f8297b9",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_4101_test\\protocol_v2_4_0_three_stage.json": {
        "expected": "6797ca6538e663389845bf90c63743540feabf8beeab225df9bdd5aed194106a",
        "actual": "6797ca6538e663389845bf90c63743540feabf8beeab225df9bdd5aed194106a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\pilot_contract.json": {
        "expected": "d7421efd446e64947c9b5a29d09509d7a6a15f41b8998a415a85cee444c03f82",
        "actual": "d7421efd446e64947c9b5a29d09509d7a6a15f41b8998a415a85cee444c03f82",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\contract.json": {
        "expected": "22bf7dd84995e990a7c5630676e568bda6272f50d638595dade7d541eb36f40b",
        "actual": "22bf7dd84995e990a7c5630676e568bda6272f50d638595dade7d541eb36f40b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\validation\\validation.json": {
        "expected": "ac9790a12b3fe2cda74c21703dfcc7506f70681a0b54905bab52886df9695049",
        "actual": "ac9790a12b3fe2cda74c21703dfcc7506f70681a0b54905bab52886df9695049",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0001.json": {
        "expected": "965f2dae8047dcfc955bbcf42c8d158e3623dc0b8d97c43bb7aeabefd4b4cf7a",
        "actual": "965f2dae8047dcfc955bbcf42c8d158e3623dc0b8d97c43bb7aeabefd4b4cf7a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0002.json": {
        "expected": "8782667a21e961fb6414cad1c5b5a2d1b62b572436ecf36f55ca37189c77836f",
        "actual": "8782667a21e961fb6414cad1c5b5a2d1b62b572436ecf36f55ca37189c77836f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0003.json": {
        "expected": "4b9487ebda355f588b451c08e42624a467265c6c72dba087df94f6015deb9e0a",
        "actual": "4b9487ebda355f588b451c08e42624a467265c6c72dba087df94f6015deb9e0a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0004.json": {
        "expected": "ea34b0018cf0cde772255204ed3b3507814dda47e7fab5a51871d112f051f3f1",
        "actual": "ea34b0018cf0cde772255204ed3b3507814dda47e7fab5a51871d112f051f3f1",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0005.json": {
        "expected": "4feb90b70aae3ac45bce71a963c6f3750a80e731fd1bb0c9d42a140a8f0bd3e5",
        "actual": "4feb90b70aae3ac45bce71a963c6f3750a80e731fd1bb0c9d42a140a8f0bd3e5",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0006.json": {
        "expected": "3076f79f19b3856b13f75bf01c799b7f8dedcfa824347f7b8055598f768bc1ad",
        "actual": "3076f79f19b3856b13f75bf01c799b7f8dedcfa824347f7b8055598f768bc1ad",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0007.json": {
        "expected": "b15f74194af7c2f1bda69ce42bf3d799864a46f363585d9d31a7129601930e1b",
        "actual": "b15f74194af7c2f1bda69ce42bf3d799864a46f363585d9d31a7129601930e1b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0008.json": {
        "expected": "04705d83dffdfdc56b54baf36644ae8b0ce9e171f5d92fed0ff07b91a2bcaf23",
        "actual": "04705d83dffdfdc56b54baf36644ae8b0ce9e171f5d92fed0ff07b91a2bcaf23",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0009.json": {
        "expected": "537e03cd3507aa44e7a7ac80e254d2db944124982b1851275dd5c79770949de7",
        "actual": "537e03cd3507aa44e7a7ac80e254d2db944124982b1851275dd5c79770949de7",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0010.json": {
        "expected": "5e4090044154b373d1ea225c8ceda05bb0952205ad3ac7e7813ea8519bb49089",
        "actual": "5e4090044154b373d1ea225c8ceda05bb0952205ad3ac7e7813ea8519bb49089",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0011.json": {
        "expected": "00854a60a21646110dfd35b25b3fd4fcf511a313b59092e746cd941913032b2f",
        "actual": "00854a60a21646110dfd35b25b3fd4fcf511a313b59092e746cd941913032b2f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0012.json": {
        "expected": "2e2f649e9a3798812e3d432796bde056002030b6f4588fb2f8ccc9dce16094fb",
        "actual": "2e2f649e9a3798812e3d432796bde056002030b6f4588fb2f8ccc9dce16094fb",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0013.json": {
        "expected": "33f7da1d1cbd50434a8f0d24f6d53a32e9a80ea937a53d9638a93db60780a9eb",
        "actual": "33f7da1d1cbd50434a8f0d24f6d53a32e9a80ea937a53d9638a93db60780a9eb",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0014.json": {
        "expected": "ba3f77b6e996d70b20f014ca6cd2f578f0dd246d07a4b90ad706b8dab2946a89",
        "actual": "ba3f77b6e996d70b20f014ca6cd2f578f0dd246d07a4b90ad706b8dab2946a89",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0015.json": {
        "expected": "f786633fc044f416f143dd626ad247215e70ceeb5243fa99a8d4c1648098da44",
        "actual": "f786633fc044f416f143dd626ad247215e70ceeb5243fa99a8d4c1648098da44",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0016.json": {
        "expected": "43611837a86267a82884163068195bef45fa2adb108bcb7e15b925cab1de9752",
        "actual": "43611837a86267a82884163068195bef45fa2adb108bcb7e15b925cab1de9752",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0017.json": {
        "expected": "731ac730e59dd9023eb3b2c5781b99b21a78a1e40b4ad8d8718a8dd121e01162",
        "actual": "731ac730e59dd9023eb3b2c5781b99b21a78a1e40b4ad8d8718a8dd121e01162",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0018.json": {
        "expected": "0ade97b160e9e49bd73819c21a272435ca8e9402cc268d8bd7e02d7141420e35",
        "actual": "0ade97b160e9e49bd73819c21a272435ca8e9402cc268d8bd7e02d7141420e35",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0019.json": {
        "expected": "ced5b1a70bcf44f3356580d46c15a0405e676880d777c74e58b8188bb0df352f",
        "actual": "ced5b1a70bcf44f3356580d46c15a0405e676880d777c74e58b8188bb0df352f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0020.json": {
        "expected": "d507a0144dfb1490bef0b40837180962a2f4cdbfab97630f59171fd17419a2ae",
        "actual": "d507a0144dfb1490bef0b40837180962a2f4cdbfab97630f59171fd17419a2ae",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0021.json": {
        "expected": "f6dc0254186fcf61409fb3d9bba4aca2a1b2dc2fd14cd3bab8f0ea98b13b4ea2",
        "actual": "f6dc0254186fcf61409fb3d9bba4aca2a1b2dc2fd14cd3bab8f0ea98b13b4ea2",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0022.json": {
        "expected": "2c404ac2d37807b7b676531c17356b4896d1edf416087299b8eee682addd11ab",
        "actual": "2c404ac2d37807b7b676531c17356b4896d1edf416087299b8eee682addd11ab",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0023.json": {
        "expected": "9d6e61c618e7040f150cb0e1bb998900bda9048be0cbdcef8d74953f56f9e6e6",
        "actual": "9d6e61c618e7040f150cb0e1bb998900bda9048be0cbdcef8d74953f56f9e6e6",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0024.json": {
        "expected": "eb7db92e76b25ff7d0be2d584fb399b0be0ea49d73d89c32114316d56a3f73b5",
        "actual": "eb7db92e76b25ff7d0be2d584fb399b0be0ea49d73d89c32114316d56a3f73b5",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0025.json": {
        "expected": "7e710583e3a15723de7fe1bec59e5040ec19ee077f1dc322d2f9b64a0b2faa8b",
        "actual": "7e710583e3a15723de7fe1bec59e5040ec19ee077f1dc322d2f9b64a0b2faa8b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0026.json": {
        "expected": "bee3c2562f61fbba63f1cf40fb3f8c97cba23a87aba42067dbd7216244831157",
        "actual": "bee3c2562f61fbba63f1cf40fb3f8c97cba23a87aba42067dbd7216244831157",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0027.json": {
        "expected": "be9793ad299b402b891bf1f1ca41ede4934d52fd88756f3ac9760f1c9bef9e88",
        "actual": "be9793ad299b402b891bf1f1ca41ede4934d52fd88756f3ac9760f1c9bef9e88",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0028.json": {
        "expected": "0477b384ab4465b76d3da67dd7c89906040c8d50bb66f817c2dc5b39176dcc86",
        "actual": "0477b384ab4465b76d3da67dd7c89906040c8d50bb66f817c2dc5b39176dcc86",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0029.json": {
        "expected": "723af004a753f4c8998b994c19b998e827d3d9d3c24fc17d09034fb402310a9a",
        "actual": "723af004a753f4c8998b994c19b998e827d3d9d3c24fc17d09034fb402310a9a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0030.json": {
        "expected": "e42b58343ced445f08e2998378440fd845951a625376eef9be919db3dfe84d88",
        "actual": "e42b58343ced445f08e2998378440fd845951a625376eef9be919db3dfe84d88",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0031.json": {
        "expected": "9625050015b20de8e687f5c3e57e2d27d8421e7ca35df3cd9a2fbc9d869db3f9",
        "actual": "9625050015b20de8e687f5c3e57e2d27d8421e7ca35df3cd9a2fbc9d869db3f9",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0032.json": {
        "expected": "7cae1bc4e1eb855a8b30a5d953dd9302d4fb46e18c0a4b2fd2d585db4e195bba",
        "actual": "7cae1bc4e1eb855a8b30a5d953dd9302d4fb46e18c0a4b2fd2d585db4e195bba",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0033.json": {
        "expected": "9f6d4accdea536af8e2436e15db8454bce2277256e2eaf54a8c846d8f9016bc5",
        "actual": "9f6d4accdea536af8e2436e15db8454bce2277256e2eaf54a8c846d8f9016bc5",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0034.json": {
        "expected": "6c76b9578c1b60f43dc32505d26feff50ff2834bab617003207c7e2a029e7bd6",
        "actual": "6c76b9578c1b60f43dc32505d26feff50ff2834bab617003207c7e2a029e7bd6",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0035.json": {
        "expected": "fcf1468af86658481bd0f92484bf1c74ce27d48258b10bda4be77bce3a64bf80",
        "actual": "fcf1468af86658481bd0f92484bf1c74ce27d48258b10bda4be77bce3a64bf80",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0036.json": {
        "expected": "a4bb22509cafc72a18ad166fbe207da9115e384ceab9fba6ea066993da2377ec",
        "actual": "a4bb22509cafc72a18ad166fbe207da9115e384ceab9fba6ea066993da2377ec",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0037.json": {
        "expected": "9e85eab389ebac45902ccacbe5b9caa4df90dc68e4427b9d36730520ce1593e0",
        "actual": "9e85eab389ebac45902ccacbe5b9caa4df90dc68e4427b9d36730520ce1593e0",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0038.json": {
        "expected": "55798d8faf1306645c16fb2303b1acf01a6fad536e048b953f3fff3b99a062a0",
        "actual": "55798d8faf1306645c16fb2303b1acf01a6fad536e048b953f3fff3b99a062a0",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0039.json": {
        "expected": "0349fd84769f5b4fbf911ea4c08d131f6b709a0fb1f3c76279d843e81dae668e",
        "actual": "0349fd84769f5b4fbf911ea4c08d131f6b709a0fb1f3c76279d843e81dae668e",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0040.json": {
        "expected": "bb99bdab5fccb36236a0d5a669fb4753bf5175b6117c4e8391a45ce161155ca4",
        "actual": "bb99bdab5fccb36236a0d5a669fb4753bf5175b6117c4e8391a45ce161155ca4",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0041.json": {
        "expected": "f76cf6b102c7aaa4292aeae71c5ce12c57f38d32398c43bac4719a8bbb509723",
        "actual": "f76cf6b102c7aaa4292aeae71c5ce12c57f38d32398c43bac4719a8bbb509723",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0042.json": {
        "expected": "3bf19ce051dfca88d1fd593d90f01a42f2e76f7b78fa0309653946fe93f1d982",
        "actual": "3bf19ce051dfca88d1fd593d90f01a42f2e76f7b78fa0309653946fe93f1d982",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0043.json": {
        "expected": "dbdde83e7a71115cf059e9b597c1da7235a9bcefede405fdfea30d5172a9e3bc",
        "actual": "dbdde83e7a71115cf059e9b597c1da7235a9bcefede405fdfea30d5172a9e3bc",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0044.json": {
        "expected": "bd7553fe7e3a0958896d6c5849595c49a6f14867b75ab398b174f064e62a2199",
        "actual": "bd7553fe7e3a0958896d6c5849595c49a6f14867b75ab398b174f064e62a2199",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0045.json": {
        "expected": "55ac39ccebd8a0cd662d3a2fb3acdd928904e42fdc3603f7de1a577fb11128a5",
        "actual": "55ac39ccebd8a0cd662d3a2fb3acdd928904e42fdc3603f7de1a577fb11128a5",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0046.json": {
        "expected": "dde7d9b5a248275411ebd2ec199fa0f39e35efcc6d3e38be1e035ad9ebe471f2",
        "actual": "dde7d9b5a248275411ebd2ec199fa0f39e35efcc6d3e38be1e035ad9ebe471f2",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0047.json": {
        "expected": "3f074fbe90f65b99d67d9148a75cecc46702091f086e7c681b7a6be5632bdc96",
        "actual": "3f074fbe90f65b99d67d9148a75cecc46702091f086e7c681b7a6be5632bdc96",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0048.json": {
        "expected": "2b9cadbea087a5c80d2d2b3883db17921b3076e28df8be53726802c6f1ad2865",
        "actual": "2b9cadbea087a5c80d2d2b3883db17921b3076e28df8be53726802c6f1ad2865",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0049.json": {
        "expected": "2cb2eb0c5010e02e9dbcdd5208bd6a09aed8a0a2f92ea372a51a6aa8198c6530",
        "actual": "2cb2eb0c5010e02e9dbcdd5208bd6a09aed8a0a2f92ea372a51a6aa8198c6530",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0050.json": {
        "expected": "6b85c5f46f1ad8f6a229b60f9e562d16be5e99afe0ee6f3f01e04d78fd2ab63c",
        "actual": "6b85c5f46f1ad8f6a229b60f9e562d16be5e99afe0ee6f3f01e04d78fd2ab63c",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0051.json": {
        "expected": "ea1e8977f744da0c3736d24b2431f31aa134032ceb76413dee9bc51875d2d8d7",
        "actual": "ea1e8977f744da0c3736d24b2431f31aa134032ceb76413dee9bc51875d2d8d7",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0052.json": {
        "expected": "047b783ca0bdc8eb10e45a9a476ff3054f758aa63be03eb2c746cce8b9e9638f",
        "actual": "047b783ca0bdc8eb10e45a9a476ff3054f758aa63be03eb2c746cce8b9e9638f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0053.json": {
        "expected": "78153230061e83fff6d5d97799cf600615469a3e6be068a05a583259a8838e1d",
        "actual": "78153230061e83fff6d5d97799cf600615469a3e6be068a05a583259a8838e1d",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0054.json": {
        "expected": "d311fc07c75c4c8c55c04d9ab18cddba251ff004ed1ab1488441bcd1a529c30d",
        "actual": "d311fc07c75c4c8c55c04d9ab18cddba251ff004ed1ab1488441bcd1a529c30d",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0055.json": {
        "expected": "0057f41527882c55a497af7e82f68994678fcbbd7954295a25ea0289d1871c1a",
        "actual": "0057f41527882c55a497af7e82f68994678fcbbd7954295a25ea0289d1871c1a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0056.json": {
        "expected": "323dfcd87821836e4dbc7123ac3c0eb2675a6fbf73758857fca5b78c6cf53dd2",
        "actual": "323dfcd87821836e4dbc7123ac3c0eb2675a6fbf73758857fca5b78c6cf53dd2",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0057.json": {
        "expected": "d0ce78321a9081ed3dd6915aefe75f9bee7465da0ad3e36ec067dc723798f55c",
        "actual": "d0ce78321a9081ed3dd6915aefe75f9bee7465da0ad3e36ec067dc723798f55c",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0058.json": {
        "expected": "ccfdc71ec3f49edd5b742d66471ecb0baa84b66408bcb580070834d68aeca509",
        "actual": "ccfdc71ec3f49edd5b742d66471ecb0baa84b66408bcb580070834d68aeca509",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0059.json": {
        "expected": "f10b63067c4684ef37e7fb8cd5d969eeed91af4ab6ac8fc4af388997d7356e4f",
        "actual": "f10b63067c4684ef37e7fb8cd5d969eeed91af4ab6ac8fc4af388997d7356e4f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0060.json": {
        "expected": "10ca111ea6356f6c914f65e664fe6d42f9867dc556e315a7de70afb43e7cad00",
        "actual": "10ca111ea6356f6c914f65e664fe6d42f9867dc556e315a7de70afb43e7cad00",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0061.json": {
        "expected": "6f2172025c46bc393357fa1c1302c78066c6a1791614e41fd4b0b7489283d1eb",
        "actual": "6f2172025c46bc393357fa1c1302c78066c6a1791614e41fd4b0b7489283d1eb",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0062.json": {
        "expected": "e73db9829afe29ba2b09b741c71f68bb4109604f335dabadaaa6458834ba2c43",
        "actual": "e73db9829afe29ba2b09b741c71f68bb4109604f335dabadaaa6458834ba2c43",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0063.json": {
        "expected": "0ab68bb6f0f04d9eff0611e586b7f2ee930ff8e5fd1b5cb78f5374f2943cad7f",
        "actual": "0ab68bb6f0f04d9eff0611e586b7f2ee930ff8e5fd1b5cb78f5374f2943cad7f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0064.json": {
        "expected": "8e60a0e94849b2d3b8660526ccf1d6eaf6c5a79285b233d5f5ab535bcf87cb20",
        "actual": "8e60a0e94849b2d3b8660526ccf1d6eaf6c5a79285b233d5f5ab535bcf87cb20",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0065.json": {
        "expected": "4df5eaa6730bc5575b9c0154a6fc409d1e3c0fb7f0e28dc7220ea8da50ffba01",
        "actual": "4df5eaa6730bc5575b9c0154a6fc409d1e3c0fb7f0e28dc7220ea8da50ffba01",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0066.json": {
        "expected": "3f21d599d25080990ee4f87d56fd24686bd0a422aeb6a71a81ed8df05b2a6eb4",
        "actual": "3f21d599d25080990ee4f87d56fd24686bd0a422aeb6a71a81ed8df05b2a6eb4",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0067.json": {
        "expected": "55ea38e1a66945582b9da3fb6cff23ed553079d6877599c0ece3b1c5d1a5cbc0",
        "actual": "55ea38e1a66945582b9da3fb6cff23ed553079d6877599c0ece3b1c5d1a5cbc0",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0068.json": {
        "expected": "392e48dee30dd480fb5ab27a0ab5d7c7bff09527a8eb0f6f4bb675d502b4e8cc",
        "actual": "392e48dee30dd480fb5ab27a0ab5d7c7bff09527a8eb0f6f4bb675d502b4e8cc",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0069.json": {
        "expected": "9094a8d516093a7325d2fea3ea2136fcfadc619cb798c63fceef4bb461504c5a",
        "actual": "9094a8d516093a7325d2fea3ea2136fcfadc619cb798c63fceef4bb461504c5a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0070.json": {
        "expected": "3eef318c9d4b8901d4c53fcfaecf6c4e0fa3a24f41fd15370b6fe8fb613877ea",
        "actual": "3eef318c9d4b8901d4c53fcfaecf6c4e0fa3a24f41fd15370b6fe8fb613877ea",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0071.json": {
        "expected": "5fd45c2718a50f5d729eb599e8961ca7c00a8eff782c8053a4fa39bed447806d",
        "actual": "5fd45c2718a50f5d729eb599e8961ca7c00a8eff782c8053a4fa39bed447806d",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0072.json": {
        "expected": "0df58b7a3d45fb794e5529b1b09a13d11cbbf9dbb3f04c0941d9ae40b03c1e4e",
        "actual": "0df58b7a3d45fb794e5529b1b09a13d11cbbf9dbb3f04c0941d9ae40b03c1e4e",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0073.json": {
        "expected": "f465fd34593716420fcd3a07d56b95f7a0629d3fbec1e0a340ef4419d4f9bbc9",
        "actual": "f465fd34593716420fcd3a07d56b95f7a0629d3fbec1e0a340ef4419d4f9bbc9",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0074.json": {
        "expected": "880a207aae8bc6682f4e0a7749972693c7320b3745dc4636430d3228f077429b",
        "actual": "880a207aae8bc6682f4e0a7749972693c7320b3745dc4636430d3228f077429b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0075.json": {
        "expected": "8d83b588510a4a597d50fe4e30621c555dc8ce2b2caee773977f910e621513fb",
        "actual": "8d83b588510a4a597d50fe4e30621c555dc8ce2b2caee773977f910e621513fb",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0076.json": {
        "expected": "c5799c057f649071afff22d75c8361c6e58e4243831c2a3b956f8b91ca14b010",
        "actual": "c5799c057f649071afff22d75c8361c6e58e4243831c2a3b956f8b91ca14b010",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0077.json": {
        "expected": "c0b81643808698495f8d1e14838b6b9ff0119aa2ebbd7f620b824e08f3cfe591",
        "actual": "c0b81643808698495f8d1e14838b6b9ff0119aa2ebbd7f620b824e08f3cfe591",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0078.json": {
        "expected": "7e4dbf126fb467666cd03b38fe6364631e02155d3d6e45eec986a54655215762",
        "actual": "7e4dbf126fb467666cd03b38fe6364631e02155d3d6e45eec986a54655215762",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0079.json": {
        "expected": "b5fe25d8d3690a24aeec96713482d662e220e71ab0df11c52fc09bf42761c715",
        "actual": "b5fe25d8d3690a24aeec96713482d662e220e71ab0df11c52fc09bf42761c715",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0080.json": {
        "expected": "9adbcaee33e7887bfa7ab847be8443c9a0f81673822487e6464be42b033037e4",
        "actual": "9adbcaee33e7887bfa7ab847be8443c9a0f81673822487e6464be42b033037e4",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0081.json": {
        "expected": "efddc400dbfb5df9be2c183c4b0fae20595a56eb405effd61eff58a69a3cdff5",
        "actual": "efddc400dbfb5df9be2c183c4b0fae20595a56eb405effd61eff58a69a3cdff5",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0082.json": {
        "expected": "b565c6f06385347bcc3014a05e5277aaa888f2cb7ee58369f477619059963af4",
        "actual": "b565c6f06385347bcc3014a05e5277aaa888f2cb7ee58369f477619059963af4",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0083.json": {
        "expected": "288e6c1d3e0daf0cc4c9d2d8551392fbf2c08c687c3ff8eca219cd59f19ac306",
        "actual": "288e6c1d3e0daf0cc4c9d2d8551392fbf2c08c687c3ff8eca219cd59f19ac306",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0084.json": {
        "expected": "97fdbe7b74cadc2e76634a5d9d84111b4e21719174a0e09631360af920d29046",
        "actual": "97fdbe7b74cadc2e76634a5d9d84111b4e21719174a0e09631360af920d29046",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0085.json": {
        "expected": "c8d479d8c3afa3512fe9034a3fcb2373166d01938cfef572d4cf3cce6f17ac5b",
        "actual": "c8d479d8c3afa3512fe9034a3fcb2373166d01938cfef572d4cf3cce6f17ac5b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0086.json": {
        "expected": "952d13299503e1be703b84b445c37fda6eb37c9d5b697264ded6e0d65225327c",
        "actual": "952d13299503e1be703b84b445c37fda6eb37c9d5b697264ded6e0d65225327c",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0087.json": {
        "expected": "3c939b0ab50110adb1dc887f0a1d7b58ab0053538da4b58e8a7254a51c66fb30",
        "actual": "3c939b0ab50110adb1dc887f0a1d7b58ab0053538da4b58e8a7254a51c66fb30",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0088.json": {
        "expected": "59eef6a3524675a2c92e4442715ae0f20de4d44a2485888928e9a2b9340776b6",
        "actual": "59eef6a3524675a2c92e4442715ae0f20de4d44a2485888928e9a2b9340776b6",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0089.json": {
        "expected": "4a327556970ad9afeed7db57e203b1c296505f78e9deca80b004554160760504",
        "actual": "4a327556970ad9afeed7db57e203b1c296505f78e9deca80b004554160760504",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0090.json": {
        "expected": "91ad5406afb2f16b0e1c663237690617813d431e7788295a1d68ed7c072a8f5e",
        "actual": "91ad5406afb2f16b0e1c663237690617813d431e7788295a1d68ed7c072a8f5e",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0091.json": {
        "expected": "cba3f325b968e66274611b10d69429efcaba15daa9f814a3d552cc7896a3b238",
        "actual": "cba3f325b968e66274611b10d69429efcaba15daa9f814a3d552cc7896a3b238",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0092.json": {
        "expected": "8afbd0babd3f383f37a93466d26848c6915d23db7674ce15dea9539d1bdb309e",
        "actual": "8afbd0babd3f383f37a93466d26848c6915d23db7674ce15dea9539d1bdb309e",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0093.json": {
        "expected": "989569974f1e92d72f4d62a733c1b93931280e5fcd8d63533b501ab24f42cde4",
        "actual": "989569974f1e92d72f4d62a733c1b93931280e5fcd8d63533b501ab24f42cde4",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0094.json": {
        "expected": "8323c08a3e56e1b3a34153564198c68a82bb11bc862bc1eaf7b4d264570f0b2d",
        "actual": "8323c08a3e56e1b3a34153564198c68a82bb11bc862bc1eaf7b4d264570f0b2d",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0095.json": {
        "expected": "cd0196215bea582dbab30f5ea7e676614d17bd23b100eae4e4821b4e214c21e1",
        "actual": "cd0196215bea582dbab30f5ea7e676614d17bd23b100eae4e4821b4e214c21e1",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0096.json": {
        "expected": "1636ff19eac588a90f5cd68218a0d88b66f3499aca18f85aef05d583d472d2b9",
        "actual": "1636ff19eac588a90f5cd68218a0d88b66f3499aca18f85aef05d583d472d2b9",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0097.json": {
        "expected": "0eff314225d9cd2bd3f34f36ead2e14fefd19feb7b65157744694613ec2a94ec",
        "actual": "0eff314225d9cd2bd3f34f36ead2e14fefd19feb7b65157744694613ec2a94ec",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0098.json": {
        "expected": "0b7ad959779196b87f3c91721dce5e7b7ad42260be91966bcca56d7cb9cedf2e",
        "actual": "0b7ad959779196b87f3c91721dce5e7b7ad42260be91966bcca56d7cb9cedf2e",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0099.json": {
        "expected": "c7c93bb52871f93fa89c28d6e7aa65e37e0e54a5e0d63208e8cbc6d9221ab5ce",
        "actual": "c7c93bb52871f93fa89c28d6e7aa65e37e0e54a5e0d63208e8cbc6d9221ab5ce",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0100.json": {
        "expected": "1a3a3eb64442df60cbcc338d183038466981c986be566b708e90bc0d82281bbb",
        "actual": "1a3a3eb64442df60cbcc338d183038466981c986be566b708e90bc0d82281bbb",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0101.json": {
        "expected": "8c5b0efdeadab35c2bfe6dbd1784e746bd6bf3581bc83f2e0d9a16da37f378a6",
        "actual": "8c5b0efdeadab35c2bfe6dbd1784e746bd6bf3581bc83f2e0d9a16da37f378a6",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0102.json": {
        "expected": "e510d8fba89a2a90cc0235d16c5817fffbd2e472a710461eab1ef77d7ad69217",
        "actual": "e510d8fba89a2a90cc0235d16c5817fffbd2e472a710461eab1ef77d7ad69217",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0103.json": {
        "expected": "a075d17bbdd168444c85bfb619fa74c257f754e8be3d334d7a3baf1242e4b7b9",
        "actual": "a075d17bbdd168444c85bfb619fa74c257f754e8be3d334d7a3baf1242e4b7b9",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0104.json": {
        "expected": "20b68f29edf9b997934b21ce807d93feef9de4023623477727bac3bfdbf8b4e9",
        "actual": "20b68f29edf9b997934b21ce807d93feef9de4023623477727bac3bfdbf8b4e9",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0105.json": {
        "expected": "5cc2646baec0aad75daea9dccc2823f6c1b0c79100b523232c9fe2ee70946457",
        "actual": "5cc2646baec0aad75daea9dccc2823f6c1b0c79100b523232c9fe2ee70946457",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0106.json": {
        "expected": "2d2cf632e67ac0a0b7c977004b459b9b9ca607d20f7f16e132111e67c2e05dc1",
        "actual": "2d2cf632e67ac0a0b7c977004b459b9b9ca607d20f7f16e132111e67c2e05dc1",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0107.json": {
        "expected": "ea33e52a12e6b070d5533a77a724a497d0e4264be786d76e482422ebc96e4ac2",
        "actual": "ea33e52a12e6b070d5533a77a724a497d0e4264be786d76e482422ebc96e4ac2",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0108.json": {
        "expected": "5f21fb80a8a11a46f7cec771b9f92b49a4e0ab750255e789b86e7b72e7cc8b0a",
        "actual": "5f21fb80a8a11a46f7cec771b9f92b49a4e0ab750255e789b86e7b72e7cc8b0a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0109.json": {
        "expected": "98af35215104fe1022c006565db900801d04aebf6eb8f79af6ed47464039f925",
        "actual": "98af35215104fe1022c006565db900801d04aebf6eb8f79af6ed47464039f925",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0110.json": {
        "expected": "33ec75534c934e7d20acb039df348ee3920f7e883e4efd2bafb513a9db161d76",
        "actual": "33ec75534c934e7d20acb039df348ee3920f7e883e4efd2bafb513a9db161d76",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0111.json": {
        "expected": "34be30b4aa6ea4bc5adfdc3cd9f374c5cf56cc96eaf6da62d985efafa50e35a7",
        "actual": "34be30b4aa6ea4bc5adfdc3cd9f374c5cf56cc96eaf6da62d985efafa50e35a7",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0112.json": {
        "expected": "3228870a3ce0463d1a2f901e306ee2ffe14753a0738a379aa6bac4200828727a",
        "actual": "3228870a3ce0463d1a2f901e306ee2ffe14753a0738a379aa6bac4200828727a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0113.json": {
        "expected": "ec640e93aafd742837de26bedb064588b07bb15f524c673d69144972983323b6",
        "actual": "ec640e93aafd742837de26bedb064588b07bb15f524c673d69144972983323b6",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0114.json": {
        "expected": "0595e802efa8be434eee8ae2d1d571df25c929ffb087aca926c1700baca99d4e",
        "actual": "0595e802efa8be434eee8ae2d1d571df25c929ffb087aca926c1700baca99d4e",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0115.json": {
        "expected": "b66c6b2995261dc7b1f5145316a1201315306b8e9f2b9487233af12e0f6bd446",
        "actual": "b66c6b2995261dc7b1f5145316a1201315306b8e9f2b9487233af12e0f6bd446",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0116.json": {
        "expected": "4aca2a7386fa0223d02325981610ab515f9633177cafcd9a1677b0a0c2a4eef9",
        "actual": "4aca2a7386fa0223d02325981610ab515f9633177cafcd9a1677b0a0c2a4eef9",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0117.json": {
        "expected": "d5e1e2724eb43878228b09110bb9a6a5e36d733980a6f6afb925f77f61b279df",
        "actual": "d5e1e2724eb43878228b09110bb9a6a5e36d733980a6f6afb925f77f61b279df",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0118.json": {
        "expected": "f85687dbf9253eeeae43238a3fc7f642f697254dbbe5fdfa78a7a4c3798b7b90",
        "actual": "f85687dbf9253eeeae43238a3fc7f642f697254dbbe5fdfa78a7a4c3798b7b90",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0119.json": {
        "expected": "f66e4afc483c16d51e36c295ce097bf1f4e09c892ead6799d6cb6ad29147262e",
        "actual": "f66e4afc483c16d51e36c295ce097bf1f4e09c892ead6799d6cb6ad29147262e",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0120.json": {
        "expected": "444802d2a9b5d95dd3274768cfbd3d049433163c4ec203b6ee2eae532d44f32c",
        "actual": "444802d2a9b5d95dd3274768cfbd3d049433163c4ec203b6ee2eae532d44f32c",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0121.json": {
        "expected": "c9b55410209ceb3e22936ef1da7a3c58a4946b11dbaf8ce0f1a5a12139d02660",
        "actual": "c9b55410209ceb3e22936ef1da7a3c58a4946b11dbaf8ce0f1a5a12139d02660",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0122.json": {
        "expected": "c23db36da6dce82edb1a8b8a902d7a6a12c45133c6bcc3401a455a732086f51e",
        "actual": "c23db36da6dce82edb1a8b8a902d7a6a12c45133c6bcc3401a455a732086f51e",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0123.json": {
        "expected": "b18caa849364160c1407b69521bbd4569e4f54b52ad4ac529d5ed42f9ca09c25",
        "actual": "b18caa849364160c1407b69521bbd4569e4f54b52ad4ac529d5ed42f9ca09c25",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0124.json": {
        "expected": "2e27e3d3dc309cf2e6fd01deb112be0eef230ccba3e5d30acf0f363ab98a5889",
        "actual": "2e27e3d3dc309cf2e6fd01deb112be0eef230ccba3e5d30acf0f363ab98a5889",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0125.json": {
        "expected": "db6cfd731f8b69c28517f58b8b38e9f8031a0692f03aa24c7a1e41a97faa0163",
        "actual": "db6cfd731f8b69c28517f58b8b38e9f8031a0692f03aa24c7a1e41a97faa0163",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0126.json": {
        "expected": "63eb766c1e00aec0a08201680832196a9d17f74e94e39e00c4b2d39c50d32c33",
        "actual": "63eb766c1e00aec0a08201680832196a9d17f74e94e39e00c4b2d39c50d32c33",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0127.json": {
        "expected": "3aadf9d4157ce870f5fb3d1036b4d28cf73c8d3a1417ca170eb6799717a36dd9",
        "actual": "3aadf9d4157ce870f5fb3d1036b4d28cf73c8d3a1417ca170eb6799717a36dd9",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0128.json": {
        "expected": "9ad35e7c872adbd92b7f32fd8f3322f44717f891b9f6db185d8cc0352d394261",
        "actual": "9ad35e7c872adbd92b7f32fd8f3322f44717f891b9f6db185d8cc0352d394261",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0129.json": {
        "expected": "1ab8d73f66db8cc8688f89c4470a7a6e3879da4d1e32c77e75ba65e85f720986",
        "actual": "1ab8d73f66db8cc8688f89c4470a7a6e3879da4d1e32c77e75ba65e85f720986",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0130.json": {
        "expected": "e657ec0415991c4d6a7700e6401d927aa890d5f3ff8fb793bf028d0090d72c6e",
        "actual": "e657ec0415991c4d6a7700e6401d927aa890d5f3ff8fb793bf028d0090d72c6e",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0131.json": {
        "expected": "9e2ad423c4a393a509bd947a740e4b8f508ffcc22eb25da93028880dd82cce19",
        "actual": "9e2ad423c4a393a509bd947a740e4b8f508ffcc22eb25da93028880dd82cce19",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0132.json": {
        "expected": "49765af859efb45decaffbb2acaf73ee72a4922c3bbf01fe02ac7b16fb8ab884",
        "actual": "49765af859efb45decaffbb2acaf73ee72a4922c3bbf01fe02ac7b16fb8ab884",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0133.json": {
        "expected": "bcc71ef2af36f291edf2d1131874d01cc03f5cfd946cd1ffafb0c6f04d84fdda",
        "actual": "bcc71ef2af36f291edf2d1131874d01cc03f5cfd946cd1ffafb0c6f04d84fdda",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0134.json": {
        "expected": "b3387ccb69763f478425ee5ccdfa29dce631fcf2c581a7082034160a03cfe3e2",
        "actual": "b3387ccb69763f478425ee5ccdfa29dce631fcf2c581a7082034160a03cfe3e2",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0135.json": {
        "expected": "d7ac9389bd9351fe41078eddf124472c34ae82efa36a281b912f4201a949a7b0",
        "actual": "d7ac9389bd9351fe41078eddf124472c34ae82efa36a281b912f4201a949a7b0",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0136.json": {
        "expected": "749c4dcc5b25923df6687ac1c901a9fa5a29e628790fd986d81ca011a9ee7c30",
        "actual": "749c4dcc5b25923df6687ac1c901a9fa5a29e628790fd986d81ca011a9ee7c30",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0137.json": {
        "expected": "4ce6e685fe62ce2084a1fcda48ca2b452f065a2974344ccbab45afc1bfaf2180",
        "actual": "4ce6e685fe62ce2084a1fcda48ca2b452f065a2974344ccbab45afc1bfaf2180",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0138.json": {
        "expected": "99d49edc79e6afa7113c980d8c93c94c3b84974aa8f4bcdc43612428fd1b8d31",
        "actual": "99d49edc79e6afa7113c980d8c93c94c3b84974aa8f4bcdc43612428fd1b8d31",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0139.json": {
        "expected": "d59519ded0528cbdf8c1ef72d5de6061f64a0fb6f431a15d45d48c814b67915b",
        "actual": "d59519ded0528cbdf8c1ef72d5de6061f64a0fb6f431a15d45d48c814b67915b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0140.json": {
        "expected": "eb50c261bb3769bc367d6e92cd29cbebaaf3f6154cb982343cc69c3f75bbdfcb",
        "actual": "eb50c261bb3769bc367d6e92cd29cbebaaf3f6154cb982343cc69c3f75bbdfcb",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0141.json": {
        "expected": "a134f32ef7f200b0d8b59f8be0e7cc47b7ca0be7b6c1f9e9dfd216297b4efaa6",
        "actual": "a134f32ef7f200b0d8b59f8be0e7cc47b7ca0be7b6c1f9e9dfd216297b4efaa6",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0142.json": {
        "expected": "ab20c79c6a20bcf8fd596bc7864e7364aa50d63ce6bdcf7aadd2a2e86d60e425",
        "actual": "ab20c79c6a20bcf8fd596bc7864e7364aa50d63ce6bdcf7aadd2a2e86d60e425",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0143.json": {
        "expected": "7b56e5fb698094c334eb8cd011615ca583fc22b4a2b34f441b7a3e523a772a6b",
        "actual": "7b56e5fb698094c334eb8cd011615ca583fc22b4a2b34f441b7a3e523a772a6b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0144.json": {
        "expected": "543d94f03b3590620602896fe3aae2bbff5c5306faf184724b940768abe7406c",
        "actual": "543d94f03b3590620602896fe3aae2bbff5c5306faf184724b940768abe7406c",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0145.json": {
        "expected": "7e4c83fbf41849e3337a02812b0a1a90f5e0291fcf1af47d486bbba4e0257a45",
        "actual": "7e4c83fbf41849e3337a02812b0a1a90f5e0291fcf1af47d486bbba4e0257a45",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0146.json": {
        "expected": "dae67b9ea910faf9f8849b9ddeca5be2bc06aa531ec98af2722961bfade60cf0",
        "actual": "dae67b9ea910faf9f8849b9ddeca5be2bc06aa531ec98af2722961bfade60cf0",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0147.json": {
        "expected": "43de59aad52f1039b4e46fb33de80a5453696c5b7e304943c5dd760ccea18a93",
        "actual": "43de59aad52f1039b4e46fb33de80a5453696c5b7e304943c5dd760ccea18a93",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0148.json": {
        "expected": "e3047c6a4268af6dbb67e35b996c243398489f65b5665d87a90de30f9aacd925",
        "actual": "e3047c6a4268af6dbb67e35b996c243398489f65b5665d87a90de30f9aacd925",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0149.json": {
        "expected": "68dcef6e4104274c2bf6c5314a002977a04e7f351d41e7b4da8153c3df62b4a6",
        "actual": "68dcef6e4104274c2bf6c5314a002977a04e7f351d41e7b4da8153c3df62b4a6",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0150.json": {
        "expected": "0b6bf6181a543358b8f17cc5514693eadc8bd1f3d05750aaab6cf0cc18ae4c62",
        "actual": "0b6bf6181a543358b8f17cc5514693eadc8bd1f3d05750aaab6cf0cc18ae4c62",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0151.json": {
        "expected": "d4682b39b92b378ce80d48d806b05478b0a1fdc95b5a99f862ef481c3b31355c",
        "actual": "d4682b39b92b378ce80d48d806b05478b0a1fdc95b5a99f862ef481c3b31355c",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0152.json": {
        "expected": "90582328bde5dc920c112d2323f905599ed64acfb0ef73ddf939edc319864a9f",
        "actual": "90582328bde5dc920c112d2323f905599ed64acfb0ef73ddf939edc319864a9f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0153.json": {
        "expected": "f4ac6fd6042bed817f8afd1c45ccf90844a541b9a833129d8a4ae397a3a5efeb",
        "actual": "f4ac6fd6042bed817f8afd1c45ccf90844a541b9a833129d8a4ae397a3a5efeb",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0154.json": {
        "expected": "e2d81d9eed32522171fbdddce93e07264d0ed07cfacfb264b9f841b9d2fff81f",
        "actual": "e2d81d9eed32522171fbdddce93e07264d0ed07cfacfb264b9f841b9d2fff81f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0155.json": {
        "expected": "6cf333bc68ebee54f84ff5e5d074e9a81737a4b8af64f76eb54ccc3035dd74b7",
        "actual": "6cf333bc68ebee54f84ff5e5d074e9a81737a4b8af64f76eb54ccc3035dd74b7",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0156.json": {
        "expected": "f01798e51c7cd8cd7834d97ce04b3bf8fb866f4791380f327451c819b2112c07",
        "actual": "f01798e51c7cd8cd7834d97ce04b3bf8fb866f4791380f327451c819b2112c07",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0157.json": {
        "expected": "9ce7022ad4c1527b56a12c02dff1ab725585c972084b89d2779d5c20edcbb659",
        "actual": "9ce7022ad4c1527b56a12c02dff1ab725585c972084b89d2779d5c20edcbb659",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0158.json": {
        "expected": "d2c4fe59dd8072061724da29107493c55c559166d78ed2902925ca8b3b018a93",
        "actual": "d2c4fe59dd8072061724da29107493c55c559166d78ed2902925ca8b3b018a93",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0159.json": {
        "expected": "ad57b56f15cc604b8f012ab43795b8af9618b0ead6895f42db89124c6263674c",
        "actual": "ad57b56f15cc604b8f012ab43795b8af9618b0ead6895f42db89124c6263674c",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0160.json": {
        "expected": "67f87eb63cadf9919e2a11928a656061cee164f95321099777d588a8bf2abefd",
        "actual": "67f87eb63cadf9919e2a11928a656061cee164f95321099777d588a8bf2abefd",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0161.json": {
        "expected": "571f27ce7e56a66ce118ededf1c67a96d95767ba0760e2da84785577cf3e75e3",
        "actual": "571f27ce7e56a66ce118ededf1c67a96d95767ba0760e2da84785577cf3e75e3",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0162.json": {
        "expected": "b760b01efc5d395d9955bd5e8c26fb351b17b01965c2bcf74c2658f939aeff26",
        "actual": "b760b01efc5d395d9955bd5e8c26fb351b17b01965c2bcf74c2658f939aeff26",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0163.json": {
        "expected": "2d971eb5ebe05ae57d6ab57ad64cfa2dadfe1a7610d47555ed55bca09a7debfb",
        "actual": "2d971eb5ebe05ae57d6ab57ad64cfa2dadfe1a7610d47555ed55bca09a7debfb",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0164.json": {
        "expected": "1e0d71ff753f56b5111a68fc117673b7ba48db963526c162ff16bcb31fd55d74",
        "actual": "1e0d71ff753f56b5111a68fc117673b7ba48db963526c162ff16bcb31fd55d74",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0165.json": {
        "expected": "70d3460852b05ea619fba69b86e58c3af34b0cbeb550dda7d2f8f52bc46bfb63",
        "actual": "70d3460852b05ea619fba69b86e58c3af34b0cbeb550dda7d2f8f52bc46bfb63",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0166.json": {
        "expected": "801cd747d8b1f04c2b6d8887352e4fa729ba64c56f254667f22857ae9f28579f",
        "actual": "801cd747d8b1f04c2b6d8887352e4fa729ba64c56f254667f22857ae9f28579f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0167.json": {
        "expected": "bf1d45724f8ee23e3c571432c0d83827df5b32ffbf5e005eb981895ee83a11ae",
        "actual": "bf1d45724f8ee23e3c571432c0d83827df5b32ffbf5e005eb981895ee83a11ae",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0168.json": {
        "expected": "bd0739b55570be2b64d13fac02b51179f60637dfd2218fda676d32c2f78ddb68",
        "actual": "bd0739b55570be2b64d13fac02b51179f60637dfd2218fda676d32c2f78ddb68",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0169.json": {
        "expected": "fd429420fdc719c239a885e69ff83118287097c6a6f21a1de9677e5018b445d0",
        "actual": "fd429420fdc719c239a885e69ff83118287097c6a6f21a1de9677e5018b445d0",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0170.json": {
        "expected": "a9f2aad5f37b49b7f184f91cdf981a34a96d3777dda847a026b710d0b7bd09de",
        "actual": "a9f2aad5f37b49b7f184f91cdf981a34a96d3777dda847a026b710d0b7bd09de",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0171.json": {
        "expected": "361850c9c373fb8e14c29bf197213b4f699507358f4d10f0ad2726d2c1d3b4e5",
        "actual": "361850c9c373fb8e14c29bf197213b4f699507358f4d10f0ad2726d2c1d3b4e5",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0172.json": {
        "expected": "78038f863b2c328160b01174a804c58ba849fc50281e0ff62d6c4fe4da5ba694",
        "actual": "78038f863b2c328160b01174a804c58ba849fc50281e0ff62d6c4fe4da5ba694",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0173.json": {
        "expected": "19576f60d1debec8a00c942c76aa3c7f8f58d92ed60cc55db58175b2682f7bf5",
        "actual": "19576f60d1debec8a00c942c76aa3c7f8f58d92ed60cc55db58175b2682f7bf5",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0174.json": {
        "expected": "6170847cce83e960829ec23baca50278716fc55735def8cc3f3c4da8ed48cbee",
        "actual": "6170847cce83e960829ec23baca50278716fc55735def8cc3f3c4da8ed48cbee",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0175.json": {
        "expected": "37a473f188edddfa54c264e1ca09cae79ce5a3546e60fff4c994eb266ddf1081",
        "actual": "37a473f188edddfa54c264e1ca09cae79ce5a3546e60fff4c994eb266ddf1081",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0176.json": {
        "expected": "96217cde7bcf1b6302fef517d6f52ec9c735b0941d5acbaa3f55b7c1074e68e0",
        "actual": "96217cde7bcf1b6302fef517d6f52ec9c735b0941d5acbaa3f55b7c1074e68e0",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0177.json": {
        "expected": "819dc6148cf7e4a75f5440308601c5ed939723bde4f9a0409b93206ec72d777f",
        "actual": "819dc6148cf7e4a75f5440308601c5ed939723bde4f9a0409b93206ec72d777f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0178.json": {
        "expected": "3514cd03294e3fbdaa7004fd41165be8c8e6986da3c601c897a95b8997bd8cf1",
        "actual": "3514cd03294e3fbdaa7004fd41165be8c8e6986da3c601c897a95b8997bd8cf1",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0179.json": {
        "expected": "c5c2f8b7dc8e623225a152f2db7cf6ff655926f92b83ad237f0ffd58b4cfab75",
        "actual": "c5c2f8b7dc8e623225a152f2db7cf6ff655926f92b83ad237f0ffd58b4cfab75",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0180.json": {
        "expected": "c697d1a5cefef6db0fd45bf8ffff5a51e4ee6745784fe09def3e9fec0c74bbbf",
        "actual": "c697d1a5cefef6db0fd45bf8ffff5a51e4ee6745784fe09def3e9fec0c74bbbf",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0181.json": {
        "expected": "a13d9d17a3d83520f3a9f9e94f3f0b0d0493a11215868a1476313647c00eb73e",
        "actual": "a13d9d17a3d83520f3a9f9e94f3f0b0d0493a11215868a1476313647c00eb73e",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0182.json": {
        "expected": "19b3bfa13e1510558da56f022cfb8f2d41b46a5a286faf3e95a8d6ef7e5db7f0",
        "actual": "19b3bfa13e1510558da56f022cfb8f2d41b46a5a286faf3e95a8d6ef7e5db7f0",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0183.json": {
        "expected": "2ba86bc43590aa17652c7538369d88330dc785f3b1971974d843a653dba95c2e",
        "actual": "2ba86bc43590aa17652c7538369d88330dc785f3b1971974d843a653dba95c2e",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0184.json": {
        "expected": "4a734ac290d5aeff9dca379cbb1a570da1a48fb3e86ab53357841af5a3f2d3e0",
        "actual": "4a734ac290d5aeff9dca379cbb1a570da1a48fb3e86ab53357841af5a3f2d3e0",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0185.json": {
        "expected": "cbb31fabbd8bed2c8cd8fb2ea758a3c981d90ba720352a6e3336f4e7c732b237",
        "actual": "cbb31fabbd8bed2c8cd8fb2ea758a3c981d90ba720352a6e3336f4e7c732b237",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0186.json": {
        "expected": "940bbdea834c7d2cb492b168f7b0ac30170794daf3401bf77a5c1ef7cca89265",
        "actual": "940bbdea834c7d2cb492b168f7b0ac30170794daf3401bf77a5c1ef7cca89265",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0187.json": {
        "expected": "777017a18344ed1e8acb1c58f9f5c4c5250a783b423cf3a69d5056b9dd236785",
        "actual": "777017a18344ed1e8acb1c58f9f5c4c5250a783b423cf3a69d5056b9dd236785",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0188.json": {
        "expected": "6fe0b7dbadc9ad8f64cc7e6c7235a0db374719b48bf3b861cceefcdf0adf8d01",
        "actual": "6fe0b7dbadc9ad8f64cc7e6c7235a0db374719b48bf3b861cceefcdf0adf8d01",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0189.json": {
        "expected": "d63a4f192221b0394f2768245f16001b986267ef12a1776f6e5c300af65180e1",
        "actual": "d63a4f192221b0394f2768245f16001b986267ef12a1776f6e5c300af65180e1",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0190.json": {
        "expected": "615d6877845c0a41c558b17f58e626c4c04cce750c4f2004b5b8afcb51e8d729",
        "actual": "615d6877845c0a41c558b17f58e626c4c04cce750c4f2004b5b8afcb51e8d729",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0191.json": {
        "expected": "230748bb02fd89602a1f250fe792b86d455c46fbd71b14df5bafa66b3574f9b5",
        "actual": "230748bb02fd89602a1f250fe792b86d455c46fbd71b14df5bafa66b3574f9b5",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0192.json": {
        "expected": "daeb19af96e5d9de48a2815e6e57b913c9c976ef83ce9cd317466082bf7167cd",
        "actual": "daeb19af96e5d9de48a2815e6e57b913c9c976ef83ce9cd317466082bf7167cd",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0193.json": {
        "expected": "be5a1d003d921b8cd13e53dc870411168967c7c0a0df5d2a40d869c053eeef14",
        "actual": "be5a1d003d921b8cd13e53dc870411168967c7c0a0df5d2a40d869c053eeef14",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0194.json": {
        "expected": "e3171156f9f23ec8924fc8ea12bae495f0e4e424b7d5954a6749dbb65b90983a",
        "actual": "e3171156f9f23ec8924fc8ea12bae495f0e4e424b7d5954a6749dbb65b90983a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0195.json": {
        "expected": "a1b7834054ce5da4fe757d64439330b30a8148b9386d185fed8076d391a8f383",
        "actual": "a1b7834054ce5da4fe757d64439330b30a8148b9386d185fed8076d391a8f383",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0196.json": {
        "expected": "e474f8d349524d3baf8e14a2081285f55d13573c41511a59bbb6c4f177a10187",
        "actual": "e474f8d349524d3baf8e14a2081285f55d13573c41511a59bbb6c4f177a10187",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0197.json": {
        "expected": "79a1ec49ec46fe8d9d152a72282d6f72bd4319ee9d592d922e5c1e9960c5de6d",
        "actual": "79a1ec49ec46fe8d9d152a72282d6f72bd4319ee9d592d922e5c1e9960c5de6d",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0198.json": {
        "expected": "11683cf047db44b1f8256a6cf43a6f09e0259fdf7476d4b0dffd106f65498402",
        "actual": "11683cf047db44b1f8256a6cf43a6f09e0259fdf7476d4b0dffd106f65498402",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0199.json": {
        "expected": "4a74c89923ae3f16e6a31c744f854a1eca842d68eb2add6e89f2c94fddf13729",
        "actual": "4a74c89923ae3f16e6a31c744f854a1eca842d68eb2add6e89f2c94fddf13729",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0200.json": {
        "expected": "2fc991f5024f0f280d4aa35fd6a7bbf942f83eebbb4e7192bb087f42f95f93ca",
        "actual": "2fc991f5024f0f280d4aa35fd6a7bbf942f83eebbb4e7192bb087f42f95f93ca",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0201.json": {
        "expected": "16e600a2d52184eb2dd6289610107fda42d5071afef845d1ad0becc903a310fa",
        "actual": "16e600a2d52184eb2dd6289610107fda42d5071afef845d1ad0becc903a310fa",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0202.json": {
        "expected": "6a49e6e6b7546d09e37e22815c1fc225454c5901902eb4756a2c483c1214f67b",
        "actual": "6a49e6e6b7546d09e37e22815c1fc225454c5901902eb4756a2c483c1214f67b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0203.json": {
        "expected": "307dcc5d09f74538a9991d47c2ff15e981b5815a16ce51b43fbe742a7241930f",
        "actual": "307dcc5d09f74538a9991d47c2ff15e981b5815a16ce51b43fbe742a7241930f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0204.json": {
        "expected": "9799ba8a2799e66102f39507b3e2e21550e04dc97c741c3fc953ea5bb85970e9",
        "actual": "9799ba8a2799e66102f39507b3e2e21550e04dc97c741c3fc953ea5bb85970e9",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0205.json": {
        "expected": "35fbf77813aac4e2c5ff019d0c046590d85ef11549eff0c8e4e7e334e911167a",
        "actual": "35fbf77813aac4e2c5ff019d0c046590d85ef11549eff0c8e4e7e334e911167a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0206.json": {
        "expected": "3d27da6c3a417c023590ec7da330b20f88a3aa3df01c046a7bc1d03e305454b8",
        "actual": "3d27da6c3a417c023590ec7da330b20f88a3aa3df01c046a7bc1d03e305454b8",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0207.json": {
        "expected": "e79add33e4e95db0fccfb7cba8d6f24545cb08214d3c4d71a9054b500d52a283",
        "actual": "e79add33e4e95db0fccfb7cba8d6f24545cb08214d3c4d71a9054b500d52a283",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0208.json": {
        "expected": "74a344267b0b6768af804a7739a8f0bfe7deb70ddc9dc3469667b65a478c7856",
        "actual": "74a344267b0b6768af804a7739a8f0bfe7deb70ddc9dc3469667b65a478c7856",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0209.json": {
        "expected": "19cdca947fb1bd0a3de74c91e103fc2d6ed3bcc01fa3f23728693a52ff747897",
        "actual": "19cdca947fb1bd0a3de74c91e103fc2d6ed3bcc01fa3f23728693a52ff747897",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0210.json": {
        "expected": "c8a799f1621b885e1308faabbc9e5c40196d091a03dbde75e5cd54bb6676dbd1",
        "actual": "c8a799f1621b885e1308faabbc9e5c40196d091a03dbde75e5cd54bb6676dbd1",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0211.json": {
        "expected": "1bd948c414109e3833098a7133df8b519ac2b2425583bcfd3dfdc077d05c9df8",
        "actual": "1bd948c414109e3833098a7133df8b519ac2b2425583bcfd3dfdc077d05c9df8",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0212.json": {
        "expected": "d7d366da7de5280425a5a43593b6caf35de1a89c4ad09412cd4be97f4749dc70",
        "actual": "d7d366da7de5280425a5a43593b6caf35de1a89c4ad09412cd4be97f4749dc70",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0213.json": {
        "expected": "550065c7ef1e9ac76499fb72017c265f408f80118ec95f79746fd7374c52dbb5",
        "actual": "550065c7ef1e9ac76499fb72017c265f408f80118ec95f79746fd7374c52dbb5",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0214.json": {
        "expected": "05c4ce4773bcee55c9dab02198d631eb92bbddb2b461d786d3d7bfa0536ba202",
        "actual": "05c4ce4773bcee55c9dab02198d631eb92bbddb2b461d786d3d7bfa0536ba202",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0215.json": {
        "expected": "a59d1d34bcbb21402558effa0d9686fd73f353af6b31139b032dfa96bf33fce4",
        "actual": "a59d1d34bcbb21402558effa0d9686fd73f353af6b31139b032dfa96bf33fce4",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0216.json": {
        "expected": "0582630b2096f8636dabd3a4c950e6fdb371a53b117f22ff0f28d3e27e71e74f",
        "actual": "0582630b2096f8636dabd3a4c950e6fdb371a53b117f22ff0f28d3e27e71e74f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0217.json": {
        "expected": "73e8fc1ce18c828cf847544376982e0b4458812a8f83d219b80b4ee2d98de798",
        "actual": "73e8fc1ce18c828cf847544376982e0b4458812a8f83d219b80b4ee2d98de798",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0218.json": {
        "expected": "354a19d21816e68a75ad87b33194825e7d3fe819ed0d460b5ae38e462bad0612",
        "actual": "354a19d21816e68a75ad87b33194825e7d3fe819ed0d460b5ae38e462bad0612",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0219.json": {
        "expected": "ea6a1b3f90a1e8f74b2cc7f4f3db11ec8dd967e61bea90ac527823911fc7d7ea",
        "actual": "ea6a1b3f90a1e8f74b2cc7f4f3db11ec8dd967e61bea90ac527823911fc7d7ea",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0220.json": {
        "expected": "7af9cab9e2179f6ed41ace7879b0a2beda5467e759a24b52f2117e7d8e779157",
        "actual": "7af9cab9e2179f6ed41ace7879b0a2beda5467e759a24b52f2117e7d8e779157",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0221.json": {
        "expected": "f97b0cdb96d3e7a742753f40b3f063648ece4a8fbcff5072927a7900155e1c64",
        "actual": "f97b0cdb96d3e7a742753f40b3f063648ece4a8fbcff5072927a7900155e1c64",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0222.json": {
        "expected": "189e677a17b1e630319807549d4998b3e5d5bd9443187e935e70e5aba2f4796d",
        "actual": "189e677a17b1e630319807549d4998b3e5d5bd9443187e935e70e5aba2f4796d",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0223.json": {
        "expected": "3174fa8f79024803beab813c22b0621fe4acb342308f8db1cbf60f48f02be711",
        "actual": "3174fa8f79024803beab813c22b0621fe4acb342308f8db1cbf60f48f02be711",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0224.json": {
        "expected": "940e71ba1e7598d8c1a1d3e806e1069dd30d096f576ea0627ec8c42d961198cb",
        "actual": "940e71ba1e7598d8c1a1d3e806e1069dd30d096f576ea0627ec8c42d961198cb",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0225.json": {
        "expected": "69f7a6f067edd436c432de33cbdcf3b178cb3502e043a2ded1dc6943c3c14995",
        "actual": "69f7a6f067edd436c432de33cbdcf3b178cb3502e043a2ded1dc6943c3c14995",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0226.json": {
        "expected": "463c83ef5e942c4c4a7af7d5fa3fe7b86939c4995e9dfb5c5e5c5cda6b7315a3",
        "actual": "463c83ef5e942c4c4a7af7d5fa3fe7b86939c4995e9dfb5c5e5c5cda6b7315a3",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0227.json": {
        "expected": "43731a71924b20703adcf20df25104126dbb282a4ce7c9c5198c03c4641fa4c3",
        "actual": "43731a71924b20703adcf20df25104126dbb282a4ce7c9c5198c03c4641fa4c3",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0228.json": {
        "expected": "1382fd58cc183ed6115fb29e4a1b6856e48bed69efd6c499e657abab0bf06470",
        "actual": "1382fd58cc183ed6115fb29e4a1b6856e48bed69efd6c499e657abab0bf06470",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0229.json": {
        "expected": "36012876d553aee5251845ae4a00458d0e69f2255db7450721fdec8cf63540e9",
        "actual": "36012876d553aee5251845ae4a00458d0e69f2255db7450721fdec8cf63540e9",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0230.json": {
        "expected": "d9e687474ff60908622d290cc02f374c86c08c55f86eb65ad8b435fbb2fe705d",
        "actual": "d9e687474ff60908622d290cc02f374c86c08c55f86eb65ad8b435fbb2fe705d",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0231.json": {
        "expected": "d1331fcd0c6c24a5888c3915f2fd976bf36981091003c7ca057045f2e05c5cb5",
        "actual": "d1331fcd0c6c24a5888c3915f2fd976bf36981091003c7ca057045f2e05c5cb5",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0232.json": {
        "expected": "5a2ea0761b15747dd3b77fd63a6414a6ff0f1b149b11a0d86b54cbcb060393ec",
        "actual": "5a2ea0761b15747dd3b77fd63a6414a6ff0f1b149b11a0d86b54cbcb060393ec",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0233.json": {
        "expected": "472b5fa9907b48b1b476ea6882a52f8a2ed94747ba1e4a81ca6047d738168f8d",
        "actual": "472b5fa9907b48b1b476ea6882a52f8a2ed94747ba1e4a81ca6047d738168f8d",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0234.json": {
        "expected": "a1f1f91fd8aa9083dd639f9727a527947fd004807d1b8efa5ba3113529ab7a5c",
        "actual": "a1f1f91fd8aa9083dd639f9727a527947fd004807d1b8efa5ba3113529ab7a5c",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0235.json": {
        "expected": "674593390757fb5a6a0839f8ac76769838e46959d027186504f9c6938be73e3a",
        "actual": "674593390757fb5a6a0839f8ac76769838e46959d027186504f9c6938be73e3a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0236.json": {
        "expected": "fd612a3dc0713ec2e9adb8ff332bcc4aeef9de868bb69ae4f649b9b386cf5983",
        "actual": "fd612a3dc0713ec2e9adb8ff332bcc4aeef9de868bb69ae4f649b9b386cf5983",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0237.json": {
        "expected": "3ec5792de09a7e25f29d9bd86aec59ea99a0d21475497cc6830334501ab9ca36",
        "actual": "3ec5792de09a7e25f29d9bd86aec59ea99a0d21475497cc6830334501ab9ca36",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0238.json": {
        "expected": "13ec0d652551e8ae8c03271bc54ea6127683d80a441763bfb12c0677cd97451d",
        "actual": "13ec0d652551e8ae8c03271bc54ea6127683d80a441763bfb12c0677cd97451d",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0239.json": {
        "expected": "bd853ec4f5fc41b3f1bb579fd2a8ef906841991141b993c20ea6d40f75a340a5",
        "actual": "bd853ec4f5fc41b3f1bb579fd2a8ef906841991141b993c20ea6d40f75a340a5",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0240.json": {
        "expected": "ec20b024e410f8b702add646e948f8072e1ae530ddb4225e28a6f2a71a0ae3f4",
        "actual": "ec20b024e410f8b702add646e948f8072e1ae530ddb4225e28a6f2a71a0ae3f4",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0241.json": {
        "expected": "ad3b04d69732009930bfd6ce633e062075f9d4fc7179b281b1a4de4c46c7446b",
        "actual": "ad3b04d69732009930bfd6ce633e062075f9d4fc7179b281b1a4de4c46c7446b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0242.json": {
        "expected": "a49a884937b666ecc1875f954680ca381f198ed7ef63118ac5240c87a178de11",
        "actual": "a49a884937b666ecc1875f954680ca381f198ed7ef63118ac5240c87a178de11",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0243.json": {
        "expected": "edb94aeab5947ec429524dc782a721e7649b9eed0f99c3b1e858d0541413ef7a",
        "actual": "edb94aeab5947ec429524dc782a721e7649b9eed0f99c3b1e858d0541413ef7a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0244.json": {
        "expected": "11a54bdc9eaeb966efda6a09e6ec67cb038a80413319c64c089bdd1326d59547",
        "actual": "11a54bdc9eaeb966efda6a09e6ec67cb038a80413319c64c089bdd1326d59547",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0245.json": {
        "expected": "6a0a8a7be33d41e7b66af5d11b4261c9935312c92cfce481534065458b89c9ca",
        "actual": "6a0a8a7be33d41e7b66af5d11b4261c9935312c92cfce481534065458b89c9ca",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0246.json": {
        "expected": "1caa7d085039e94193e96c77542e9548a9427739ee07df5474c3d1b054963cd2",
        "actual": "1caa7d085039e94193e96c77542e9548a9427739ee07df5474c3d1b054963cd2",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0247.json": {
        "expected": "8d2286c223b847bf9119cb764b897903e9350f2f7f9b3568128ea5673c2204a0",
        "actual": "8d2286c223b847bf9119cb764b897903e9350f2f7f9b3568128ea5673c2204a0",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0248.json": {
        "expected": "8f7e7662939545d9f0ab4ad1d88adae21dc6dfa95d4f7bdfe1d68f62f2a0eb4e",
        "actual": "8f7e7662939545d9f0ab4ad1d88adae21dc6dfa95d4f7bdfe1d68f62f2a0eb4e",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0249.json": {
        "expected": "32b559c99ffb5fbb394b24aac5fbfcb93f39f1e7be4036d86765ef4caa600f0b",
        "actual": "32b559c99ffb5fbb394b24aac5fbfcb93f39f1e7be4036d86765ef4caa600f0b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0250.json": {
        "expected": "c6ed28e180d9eeecb1b780b03f77078c02afc6b2482bdc996c150346de6cd234",
        "actual": "c6ed28e180d9eeecb1b780b03f77078c02afc6b2482bdc996c150346de6cd234",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0251.json": {
        "expected": "c107e2c694ebe7bafe24288ae93fc9dd4dc1bb3731488ccee7ac09c3cde0ecf6",
        "actual": "c107e2c694ebe7bafe24288ae93fc9dd4dc1bb3731488ccee7ac09c3cde0ecf6",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0252.json": {
        "expected": "68448cf8f786f80a1c0bf2aea0061cb2e4f2c7cd0d8a4f40c57782efa1d95b5f",
        "actual": "68448cf8f786f80a1c0bf2aea0061cb2e4f2c7cd0d8a4f40c57782efa1d95b5f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0253.json": {
        "expected": "9841a75e711de27fd58ca0a3ea0049e376f05bbd2c06ac0a27f5bd939b66b478",
        "actual": "9841a75e711de27fd58ca0a3ea0049e376f05bbd2c06ac0a27f5bd939b66b478",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0254.json": {
        "expected": "b4b0747cf1e4f27c2dfe970a7e069572c96acae6b38ee839d7d2fc9df6d3b411",
        "actual": "b4b0747cf1e4f27c2dfe970a7e069572c96acae6b38ee839d7d2fc9df6d3b411",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0255.json": {
        "expected": "0e971a25af8dcda47ca67faa54a5375369988a44002b82ca44e4c52c262c4a7f",
        "actual": "0e971a25af8dcda47ca67faa54a5375369988a44002b82ca44e4c52c262c4a7f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0256.json": {
        "expected": "0bb53854ea54428e8d440a01c362ddf0094338ce1dea05901d021068b0897e34",
        "actual": "0bb53854ea54428e8d440a01c362ddf0094338ce1dea05901d021068b0897e34",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0257.json": {
        "expected": "c2d6749b03266f52e795852443b657ee24cbe3996f5b4cb1911b689152ad5565",
        "actual": "c2d6749b03266f52e795852443b657ee24cbe3996f5b4cb1911b689152ad5565",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0258.json": {
        "expected": "9658fadade8e40e5396271d22e6f023b1eab537d6690756562f0838d74fca7bb",
        "actual": "9658fadade8e40e5396271d22e6f023b1eab537d6690756562f0838d74fca7bb",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0259.json": {
        "expected": "36eee68716caf25ffa970e92ac2ccedbbf24ef7ed407853dc5c5977f8bbcdf30",
        "actual": "36eee68716caf25ffa970e92ac2ccedbbf24ef7ed407853dc5c5977f8bbcdf30",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0260.json": {
        "expected": "39fead405b8b34d3b9f090f6de0dfd3649f841227e999716da3cd40da1a24e0f",
        "actual": "39fead405b8b34d3b9f090f6de0dfd3649f841227e999716da3cd40da1a24e0f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0261.json": {
        "expected": "030137e6f781e6755b0281eacc8eab7c917f82fe6ac1a35de1ff5c95e3786fcb",
        "actual": "030137e6f781e6755b0281eacc8eab7c917f82fe6ac1a35de1ff5c95e3786fcb",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0262.json": {
        "expected": "0394db6573f6e0cd50e1a5eeea187de198dbfd79f00fd0d3ecc7d7e5d317fc86",
        "actual": "0394db6573f6e0cd50e1a5eeea187de198dbfd79f00fd0d3ecc7d7e5d317fc86",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0263.json": {
        "expected": "13725839a860300d86832bed5d00eaac5093b5f512c1a1f3358ade456070544a",
        "actual": "13725839a860300d86832bed5d00eaac5093b5f512c1a1f3358ade456070544a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0264.json": {
        "expected": "f9b417c8bc7fab773c433a9d0bed107673b9b15a40b78110602fdc4ad2964e54",
        "actual": "f9b417c8bc7fab773c433a9d0bed107673b9b15a40b78110602fdc4ad2964e54",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0265.json": {
        "expected": "501b52ed8d8caee5c84100d9352b7ff25e94c99fec4582403e114f2a13ad8a37",
        "actual": "501b52ed8d8caee5c84100d9352b7ff25e94c99fec4582403e114f2a13ad8a37",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0266.json": {
        "expected": "adfa2f6e820edc5b12703e97193a6363ba753bdb7a51f71945bbb2c00772d46a",
        "actual": "adfa2f6e820edc5b12703e97193a6363ba753bdb7a51f71945bbb2c00772d46a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0267.json": {
        "expected": "3e656f6262b6822720697466e093d731924709f2dfb099760fd0407f03897c78",
        "actual": "3e656f6262b6822720697466e093d731924709f2dfb099760fd0407f03897c78",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0268.json": {
        "expected": "44188fa6ec0acf56d22228549eef1987f4db0f9804fcc9e134c85e976ca0a50e",
        "actual": "44188fa6ec0acf56d22228549eef1987f4db0f9804fcc9e134c85e976ca0a50e",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0269.json": {
        "expected": "ac35ee1b043331855ab48ae0833904603d7af1b8598c6b292203dbb272593e15",
        "actual": "ac35ee1b043331855ab48ae0833904603d7af1b8598c6b292203dbb272593e15",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0270.json": {
        "expected": "9fdd23e5c4fe24334360da087623ee2f9d6da8a84db93bd049085fe719d270ab",
        "actual": "9fdd23e5c4fe24334360da087623ee2f9d6da8a84db93bd049085fe719d270ab",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0271.json": {
        "expected": "83182bbdc1e189aee64dadf8339d426dbcf62dba983d30bfffa2624bc7636b00",
        "actual": "83182bbdc1e189aee64dadf8339d426dbcf62dba983d30bfffa2624bc7636b00",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0272.json": {
        "expected": "cfd4634e5d0247be54d726257637fd86b523ad7c3f3c9b254e300659729478cb",
        "actual": "cfd4634e5d0247be54d726257637fd86b523ad7c3f3c9b254e300659729478cb",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0273.json": {
        "expected": "7fd5a65a7a7da87fbf1ad955e9c4a246f798698c76fa6ffbf3c6af25fc8db0bc",
        "actual": "7fd5a65a7a7da87fbf1ad955e9c4a246f798698c76fa6ffbf3c6af25fc8db0bc",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0274.json": {
        "expected": "327c582ae6eefa77b43a3362640ebef9c8a9d1db8700d5b62f33d490882398cb",
        "actual": "327c582ae6eefa77b43a3362640ebef9c8a9d1db8700d5b62f33d490882398cb",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0275.json": {
        "expected": "5ca47fcb5a400b7aea791253e81025b1527924b43e3153d3a6e7fd759032dac6",
        "actual": "5ca47fcb5a400b7aea791253e81025b1527924b43e3153d3a6e7fd759032dac6",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0276.json": {
        "expected": "278637a5dcdbec7f237be0d27edb16039550ac88f4e99c433d95a47c774e089b",
        "actual": "278637a5dcdbec7f237be0d27edb16039550ac88f4e99c433d95a47c774e089b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0277.json": {
        "expected": "f1d9c4dbb4b743e5a4a9b5b00d5e38523a56bbfeabcd6e096d7e24b6d4157169",
        "actual": "f1d9c4dbb4b743e5a4a9b5b00d5e38523a56bbfeabcd6e096d7e24b6d4157169",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0278.json": {
        "expected": "b68e5507dd3f3c48220ff12425f671974e1d901b7dcfacc7e34e25563bb8c521",
        "actual": "b68e5507dd3f3c48220ff12425f671974e1d901b7dcfacc7e34e25563bb8c521",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0279.json": {
        "expected": "51a78944e0c34790f509c63bed525f1a725c1463a432459b43d881c86d290fad",
        "actual": "51a78944e0c34790f509c63bed525f1a725c1463a432459b43d881c86d290fad",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0280.json": {
        "expected": "3556f7f49f31f0e012f01d291fac7a00ceba08eeb75e567c036b275ec9f11907",
        "actual": "3556f7f49f31f0e012f01d291fac7a00ceba08eeb75e567c036b275ec9f11907",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0281.json": {
        "expected": "927283d7a9f320a76be2bb7977c54b83bfaf71249fc413d90e534cdb35841720",
        "actual": "927283d7a9f320a76be2bb7977c54b83bfaf71249fc413d90e534cdb35841720",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0282.json": {
        "expected": "32d6c29f9474eed545770f325b72f84097ce6560bcd378f1b6169a1ee275e095",
        "actual": "32d6c29f9474eed545770f325b72f84097ce6560bcd378f1b6169a1ee275e095",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0283.json": {
        "expected": "fb1f7c06a0f3b1d24584ae75a8019ea461e9e663c8e75232bcd94baf2cf9c2fe",
        "actual": "fb1f7c06a0f3b1d24584ae75a8019ea461e9e663c8e75232bcd94baf2cf9c2fe",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0284.json": {
        "expected": "37f7b1f6ef8a2296eb601b06039170474fa6b77b03129033726806a1dcbb9567",
        "actual": "37f7b1f6ef8a2296eb601b06039170474fa6b77b03129033726806a1dcbb9567",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0285.json": {
        "expected": "1ef5e812c342833ff4b8a34532621cd92091a9a00dabe6f5c8f8a9fe5e472e6b",
        "actual": "1ef5e812c342833ff4b8a34532621cd92091a9a00dabe6f5c8f8a9fe5e472e6b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0286.json": {
        "expected": "6de597625284ceb879e097027234008d75435d69e77320fad2d00da786d178ff",
        "actual": "6de597625284ceb879e097027234008d75435d69e77320fad2d00da786d178ff",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0287.json": {
        "expected": "1b932b281367d3f82b9c5c0021bd8d2f7ed3dc25dc71d56318fe1bdcfa22d06c",
        "actual": "1b932b281367d3f82b9c5c0021bd8d2f7ed3dc25dc71d56318fe1bdcfa22d06c",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0288.json": {
        "expected": "43519222225d4ad3d1a0c8911ecfd4b77038178eb217c9e3ff8a7b343a529966",
        "actual": "43519222225d4ad3d1a0c8911ecfd4b77038178eb217c9e3ff8a7b343a529966",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0289.json": {
        "expected": "9750949dfa8a284b5d3e87b3d6d7ecb6f3b06fc65e11b77ff84216d65ce80faf",
        "actual": "9750949dfa8a284b5d3e87b3d6d7ecb6f3b06fc65e11b77ff84216d65ce80faf",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0290.json": {
        "expected": "168bc9edab3fb3d0db7ab1bc8589015e05c4caf79828b407849a9600d22515ff",
        "actual": "168bc9edab3fb3d0db7ab1bc8589015e05c4caf79828b407849a9600d22515ff",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0291.json": {
        "expected": "740619efff169d1d35ed78299eb356dc86db1fc3978522ff579fd87ebe25059f",
        "actual": "740619efff169d1d35ed78299eb356dc86db1fc3978522ff579fd87ebe25059f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0292.json": {
        "expected": "bfce0b9d54d7baf1b28a2c0ed2a171697f2cfdfae14ec44a5d654837e16cd145",
        "actual": "bfce0b9d54d7baf1b28a2c0ed2a171697f2cfdfae14ec44a5d654837e16cd145",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0293.json": {
        "expected": "7fbc57966577af10387e548087fd9a57fcbbee9a42d591322e352241f52db84d",
        "actual": "7fbc57966577af10387e548087fd9a57fcbbee9a42d591322e352241f52db84d",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0294.json": {
        "expected": "e3dd19046266c69133725b3425314ac50d942c95428e249a6df07231c1e465ce",
        "actual": "e3dd19046266c69133725b3425314ac50d942c95428e249a6df07231c1e465ce",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0295.json": {
        "expected": "2e14b56fc6c7229d222bf8273086bc58e049c111ff7d18c214e3c7557759f737",
        "actual": "2e14b56fc6c7229d222bf8273086bc58e049c111ff7d18c214e3c7557759f737",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0296.json": {
        "expected": "dfc227db6a3d3636492600c8afbd6e95f3025249f60ff3cba181b17683569921",
        "actual": "dfc227db6a3d3636492600c8afbd6e95f3025249f60ff3cba181b17683569921",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0297.json": {
        "expected": "1c7eea8b9c2d061a1c9a1bb8b90d8346ad86271f2d499c3d559d352ef2ed97bc",
        "actual": "1c7eea8b9c2d061a1c9a1bb8b90d8346ad86271f2d499c3d559d352ef2ed97bc",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0298.json": {
        "expected": "1591116786c0e858ee291442b81fed0367889d1b460623966bbf92064e949c73",
        "actual": "1591116786c0e858ee291442b81fed0367889d1b460623966bbf92064e949c73",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0299.json": {
        "expected": "239e54e447ef60240546d6c2b52f4102b7799c094bca416c745fde75e7287cbc",
        "actual": "239e54e447ef60240546d6c2b52f4102b7799c094bca416c745fde75e7287cbc",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\episodes\\episode-0300.json": {
        "expected": "d41ddc6fee585002d42e0b3dd9a5e91aad52cc76c24857498fadd6268ca6f4a9",
        "actual": "d41ddc6fee585002d42e0b3dd9a5e91aad52cc76c24857498fadd6268ca6f4a9",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\training_manifest.json": {
        "expected": "f5daedd483eb4beae9ea9b558a543293cf7c492c560b275e16bd78db8c2feb00",
        "actual": "f5daedd483eb4beae9ea9b558a543293cf7c492c560b275e16bd78db8c2feb00",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\snapshots\\episode-0025.json": {
        "expected": "f5af55eef2166ea8465e2fa1a734ed43729d31d92b3a8273cc7f4c0f2ef07601",
        "actual": "f5af55eef2166ea8465e2fa1a734ed43729d31d92b3a8273cc7f4c0f2ef07601",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\snapshots\\episode-0025.pt": {
        "expected": "563b1026c2da5a1a5bf0f1e7c09d7a668ac2f37287053e694345bd66ac4e2df4",
        "actual": "563b1026c2da5a1a5bf0f1e7c09d7a668ac2f37287053e694345bd66ac4e2df4",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\snapshots\\episode-0050.json": {
        "expected": "79f34f134b892946959b21df63e6e7c0581a9396b507f417fb228bdbc33397c4",
        "actual": "79f34f134b892946959b21df63e6e7c0581a9396b507f417fb228bdbc33397c4",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\snapshots\\episode-0050.pt": {
        "expected": "c84474c3655103fe6efceb42aa8669e3421a8a6f2720827dbd3e0406932ca8d8",
        "actual": "c84474c3655103fe6efceb42aa8669e3421a8a6f2720827dbd3e0406932ca8d8",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\snapshots\\episode-0075.json": {
        "expected": "9d2f1a8c8c000a8a2a8431ac5818e15c4858036c23830bd148bea78b87e9534b",
        "actual": "9d2f1a8c8c000a8a2a8431ac5818e15c4858036c23830bd148bea78b87e9534b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\snapshots\\episode-0075.pt": {
        "expected": "fea4380928ce85b22d3de7b36a8be9745c49375d285763f593db52db150580f0",
        "actual": "fea4380928ce85b22d3de7b36a8be9745c49375d285763f593db52db150580f0",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\snapshots\\episode-0100.json": {
        "expected": "a0fa27d18df3c144e6a0010965ebe401a25e6df8bcb0b1a50148028c4e3b76aa",
        "actual": "a0fa27d18df3c144e6a0010965ebe401a25e6df8bcb0b1a50148028c4e3b76aa",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\snapshots\\episode-0100.pt": {
        "expected": "935c18639c58ea71f379e893978e34c5d626121ca026348b9e77b71a62f60e44",
        "actual": "935c18639c58ea71f379e893978e34c5d626121ca026348b9e77b71a62f60e44",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\snapshots\\episode-0125.json": {
        "expected": "30f370eaaec48d28090739c1230c00452dff7859397a5ef06ccb0ebe7a62366b",
        "actual": "30f370eaaec48d28090739c1230c00452dff7859397a5ef06ccb0ebe7a62366b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\snapshots\\episode-0125.pt": {
        "expected": "682b981fd07354bd2e1bb8d9b949b1f8b20c5ad4bf07880aa71cafe8f51cdb6f",
        "actual": "682b981fd07354bd2e1bb8d9b949b1f8b20c5ad4bf07880aa71cafe8f51cdb6f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\snapshots\\episode-0150.json": {
        "expected": "b0fb293ba49ee51a697d091a3b1954f1020b7a9e1c4d586a060cb7342ad7a0b6",
        "actual": "b0fb293ba49ee51a697d091a3b1954f1020b7a9e1c4d586a060cb7342ad7a0b6",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\snapshots\\episode-0150.pt": {
        "expected": "2f2b460599c0ecdeb353678ce5ab5460452fffca00e73d90778a5798f186b105",
        "actual": "2f2b460599c0ecdeb353678ce5ab5460452fffca00e73d90778a5798f186b105",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\snapshots\\episode-0175.json": {
        "expected": "62762c6ee1c90401501ff2c7d9fca087afff67dc30cb49d4eaa5a374368b6797",
        "actual": "62762c6ee1c90401501ff2c7d9fca087afff67dc30cb49d4eaa5a374368b6797",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\snapshots\\episode-0175.pt": {
        "expected": "7fb20de99e8c645334be3d3d20bd1511d07de3abc903c20480057c1e850addcd",
        "actual": "7fb20de99e8c645334be3d3d20bd1511d07de3abc903c20480057c1e850addcd",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\snapshots\\episode-0200.json": {
        "expected": "cdba46bcb3d2fe91b6a2bbf8411aab94a7605fbbf3a79163ecad753f05ef152f",
        "actual": "cdba46bcb3d2fe91b6a2bbf8411aab94a7605fbbf3a79163ecad753f05ef152f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\snapshots\\episode-0200.pt": {
        "expected": "0895fc233629d39236d76e606e52a1eb33f13982695c69404eb90989e135d991",
        "actual": "0895fc233629d39236d76e606e52a1eb33f13982695c69404eb90989e135d991",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\src\\hrl_v2\\run.py": {
        "expected": "6e7714738aeccae8ae83e566c4412d996fd5671f441cb8aafcf7d033484c3317",
        "actual": "6e7714738aeccae8ae83e566c4412d996fd5671f441cb8aafcf7d033484c3317",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\src\\hrl_v2\\archive.py": {
        "expected": "27fda1a60bd76f9f42a978aa6e685fec4caed6fc014427b9ad4aa34add91f373",
        "actual": "27fda1a60bd76f9f42a978aa6e685fec4caed6fc014427b9ad4aa34add91f373",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\src\\hrl_v2\\replay.py": {
        "expected": "5bb7f4512a42f3dc0852ad238a641d57d5dacd49d0c99a2832c74cc70cd28225",
        "actual": "5bb7f4512a42f3dc0852ad238a641d57d5dacd49d0c99a2832c74cc70cd28225",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\src\\hrl_v2\\runtime.py": {
        "expected": "02a6e33d62dd3063fa4d85c51dee859656c54be0d5b4afdc23abd4c33677df92",
        "actual": "02a6e33d62dd3063fa4d85c51dee859656c54be0d5b4afdc23abd4c33677df92",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\src\\hrl_v2\\collector.py": {
        "expected": "a8daf6106836df0c1610e988443f273c89505e845b40d2a08601eb37f637f5ab",
        "actual": "a8daf6106836df0c1610e988443f273c89505e845b40d2a08601eb37f637f5ab",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\src\\hrl_v2\\randomness.py": {
        "expected": "b2f673ba7012f46e3512f51ec8ad554701f1ab5d72546dc94887ca264d6b3214",
        "actual": "b2f673ba7012f46e3512f51ec8ad554701f1ab5d72546dc94887ca264d6b3214",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\src\\hrl_v2\\schedule.py": {
        "expected": "3c723fe8d024c583f2633f1ee38556219fc2d857663b9165993ecc8460ac68be",
        "actual": "3c723fe8d024c583f2633f1ee38556219fc2d857663b9165993ecc8460ac68be",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\pilot_train.py": {
        "expected": "ce99bda339308e8e6fdc770ccf7dd1f886b3545b96ce661964180865b98e0f5a",
        "actual": "ce99bda339308e8e6fdc770ccf7dd1f886b3545b96ce661964180865b98e0f5a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\pilot_train_r2.py": {
        "expected": "12ffc6614037437de1cae4aea8710b1250a38f37649947f893c2ad3eb9ea52dd",
        "actual": "12ffc6614037437de1cae4aea8710b1250a38f37649947f893c2ad3eb9ea52dd",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\pilot_common.py": {
        "expected": "d9b22b50a44db0e1c7eb15aaaa0177b7e4c92c0ddba0bec9f65e167c8af1235a",
        "actual": "d9b22b50a44db0e1c7eb15aaaa0177b7e4c92c0ddba0bec9f65e167c8af1235a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\checkpoints\\episode-0300-evaluated.pt": {
        "expected": "6922537220b8ef6a0cd0257be429a0fdfd075821b3245ea62e3f4daf966d7ad4",
        "actual": "6922537220b8ef6a0cd0257be429a0fdfd075821b3245ea62e3f4daf966d7ad4",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\checkpoints\\episode-0300-evaluated.json": {
        "expected": "aea8b236f82a4c8ad93f4677ca9d51a279b93d055d9195efcfdf6b0259a50985",
        "actual": "aea8b236f82a4c8ad93f4677ca9d51a279b93d055d9195efcfdf6b0259a50985",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\pipeline_data.json": {
        "expected": "6386a9989eb47c7bd7d5f2a763136fc47661ab78f0d766bbd0283984b8aeb357",
        "actual": "6386a9989eb47c7bd7d5f2a763136fc47661ab78f0d766bbd0283984b8aeb357",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step4_attempt1\\checkpoints\\U200.pt": {
        "expected": "273591ce84d1e90cb072d344412fdbe9f02cf6c1efc45a44f0800560f39e3e2e",
        "actual": "273591ce84d1e90cb072d344412fdbe9f02cf6c1efc45a44f0800560f39e3e2e",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step4_attempt1\\checkpoints\\U200.json": {
        "expected": "4546f579c611a41bcdd8c912d4dc8ae428fd0236d1e9ae7fed1a00434487923b",
        "actual": "4546f579c611a41bcdd8c912d4dc8ae428fd0236d1e9ae7fed1a00434487923b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\TFOO_selection.json": {
        "expected": "430dc9e902461e4e7dcc212c2af3c96ae0719b3619c5d563fe7d8c3b09940cc4",
        "actual": "430dc9e902461e4e7dcc212c2af3c96ae0719b3619c5d563fe7d8c3b09940cc4",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\rp\\RP-E\\E5\\summary.json": {
        "expected": "c9671bc0cb9437fbaeac9e026feb7918dab89fd4bbfadcd61cf1477a6c8b5d90",
        "actual": "c9671bc0cb9437fbaeac9e026feb7918dab89fd4bbfadcd61cf1477a6c8b5d90",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_diag_20261004\\summaries\\seed_4104\\E5.json": {
        "expected": "1a80abce1fa87f4160c2852cd84ab2a07c1851b9ca10e683b4dc49b2c4419a66",
        "actual": "1a80abce1fa87f4160c2852cd84ab2a07c1851b9ca10e683b4dc49b2c4419a66",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\rp\\RP-E\\TFOO-200\\summary.json": {
        "expected": "67ea39e09eced60e93967a95679f9bb7efb4f8c6dfb2a47f3635079d414dd1b2",
        "actual": "67ea39e09eced60e93967a95679f9bb7efb4f8c6dfb2a47f3635079d414dd1b2",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_diag_20261004\\summaries\\seed_4104\\TFOO-200.json": {
        "expected": "ca087b263b3a19c1550d8ad1b378d555eaa8a4f134ac71fce864b34bec99c32f",
        "actual": "ca087b263b3a19c1550d8ad1b378d555eaa8a4f134ac71fce864b34bec99c32f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\evaluations\\D-b_U200_cvx\\summary.json": {
        "expected": "213341bede460946318fde6fe02883498806d45cb2c8176233a5f50a1f957e18",
        "actual": "213341bede460946318fde6fe02883498806d45cb2c8176233a5f50a1f957e18",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\evaluations\\D-a_TFOO_S300r\\summary.json": {
        "expected": "b951ebadec30efb272fb979f34cb9ee5bde2414787771cc9d82f24e442d92440",
        "actual": "b951ebadec30efb272fb979f34cb9ee5bde2414787771cc9d82f24e442d92440",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\evaluations\\H_c4\\summary.json": {
        "expected": "58d71fdc846595afb45a19f3c615ac1f283f44bcfb4fd52b17dba39f51c8b341",
        "actual": "58d71fdc846595afb45a19f3c615ac1f283f44bcfb4fd52b17dba39f51c8b341",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\evaluations\\F_c9\\summary.json": {
        "expected": "6eeeab61343c5bc57bcd5c69f41c9c3a934e508730ccf0a1d190dc9495498efb",
        "actual": "6eeeab61343c5bc57bcd5c69f41c9c3a934e508730ccf0a1d190dc9495498efb",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_diag_20261004\\summaries\\seed_4104\\X1.json": {
        "expected": "9f54d2f3aa3afb5be086fc01fc0b34437b3c6a40a5df01232e0cf0197e9b00ad",
        "actual": "9f54d2f3aa3afb5be086fc01fc0b34437b3c6a40a5df01232e0cf0197e9b00ad",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_diag_20261004\\summaries\\seed_4104\\E2.json": {
        "expected": "280850ad8ba0f208ba99a2888c68bc5f0c10a7a7c6e2d4413ee593a06cc0f2e3",
        "actual": "280850ad8ba0f208ba99a2888c68bc5f0c10a7a7c6e2d4413ee593a06cc0f2e3",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_diag_20261004\\summaries\\seed_4104\\S3-c1.json": {
        "expected": "815c84903a7486557777dcff2ef277c276145240ba9de7c79440dbdb385efc61",
        "actual": "815c84903a7486557777dcff2ef277c276145240ba9de7c79440dbdb385efc61",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_diag_20261004\\summaries\\seed_4104\\S3-c2.json": {
        "expected": "e8caed242d3872f802478ea0cf06acc75ae1b155324c883b20b659343097bc39",
        "actual": "e8caed242d3872f802478ea0cf06acc75ae1b155324c883b20b659343097bc39",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_diag_20261004\\summaries\\seed_4104\\S3-c3.json": {
        "expected": "896b4e97d1593f71210b5511dfda3577eac36f73fbcbbafb3f1ae47934a3dfd5",
        "actual": "896b4e97d1593f71210b5511dfda3577eac36f73fbcbbafb3f1ae47934a3dfd5",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_diag_20261004\\summaries\\seed_4104\\S3-c4.json": {
        "expected": "b19b1db9da58de849f636e411b653e3a4033c516c2786551db0940af8ed5feb5",
        "actual": "b19b1db9da58de849f636e411b653e3a4033c516c2786551db0940af8ed5feb5",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_diag_20261004\\summaries\\seed_4104\\S3-c5.json": {
        "expected": "1003bc0344ca85076350bf66e532cb6aec0298fff9fa89d45890d96e249bfd22",
        "actual": "1003bc0344ca85076350bf66e532cb6aec0298fff9fa89d45890d96e249bfd22",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_diag_20261004\\summaries\\seed_4104\\S3-c6.json": {
        "expected": "8ef56d40280f29800158652e66bf97919851fad5c21afe069108baae00918551",
        "actual": "8ef56d40280f29800158652e66bf97919851fad5c21afe069108baae00918551",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_diag_20261004\\summaries\\seed_4104\\S3-c7.json": {
        "expected": "a91e455477ddf2a3489e24801da609240b1049bbb38d11851e02f998b09fab1a",
        "actual": "a91e455477ddf2a3489e24801da609240b1049bbb38d11851e02f998b09fab1a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_diag_20261004\\summaries\\seed_4104\\S3-c8.json": {
        "expected": "20bf04cde72d763434c8ac5d261ed995aa3b23cca13d703a479d33e9f8a060dc",
        "actual": "20bf04cde72d763434c8ac5d261ed995aa3b23cca13d703a479d33e9f8a060dc",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_diag_20261004\\summaries\\seed_4104\\S3-c9.json": {
        "expected": "1ba6a248f584bb2708177823d9f97a424573f84e1ec75d11f636e68968f74e74",
        "actual": "1ba6a248f584bb2708177823d9f97a424573f84e1ec75d11f636e68968f74e74",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_diag_20261004\\summaries\\seed_4104\\S3-c10.json": {
        "expected": "5f0b78b7e2f776bae9c6e20aca7e747243a394e045ffc7493c7a19200e5c72aa",
        "actual": "5f0b78b7e2f776bae9c6e20aca7e747243a394e045ffc7493c7a19200e5c72aa",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_diag_20261004\\summaries\\selection.json": {
        "expected": "b6b3dade868a3e6f3d320ce22490c0564709c33740f2e45b8df42d6732007dd0",
        "actual": "b6b3dade868a3e6f3d320ce22490c0564709c33740f2e45b8df42d6732007dd0",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\TFOO\\seed_4104\\checkpoints\\episode-0200-anchor.pt": {
        "expected": "e5b954fc4f01b9394cca0a48d5066eb04861ca1265e202f1eb4205a318c68e16",
        "actual": "e5b954fc4f01b9394cca0a48d5066eb04861ca1265e202f1eb4205a318c68e16",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\TFOO\\seed_4104\\checkpoints\\episode-0200-anchor.json": {
        "expected": "56bf5b3dd2e0cbfb598a28e9458b3c2524deffcc23a96a34caba9c0a865ca10b",
        "actual": "56bf5b3dd2e0cbfb598a28e9458b3c2524deffcc23a96a34caba9c0a865ca10b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\checkpoints\\episode-0200.pt": {
        "expected": "98eb3c117bf5d8e3bda863ab1c7541cd5d74defa7af6a677aba7c0e50ca78c83",
        "actual": "98eb3c117bf5d8e3bda863ab1c7541cd5d74defa7af6a677aba7c0e50ca78c83",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\checkpoints\\episode-0200.json": {
        "expected": "a006242d33769919cb70e28cc108e8f667e915a5d2270054587e6eb0c7762e96",
        "actual": "a006242d33769919cb70e28cc108e8f667e915a5d2270054587e6eb0c7762e96",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\checkpoints\\cycle-08.pt": {
        "expected": "528ef0a252791c97f9962beb7c7b650ef74293f28e735fe9cfb128de2d4d6021",
        "actual": "528ef0a252791c97f9962beb7c7b650ef74293f28e735fe9cfb128de2d4d6021",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\checkpoints\\cycle-08.json": {
        "expected": "3007fa082d7030207553da78cea95a06901e4a93badfa03be3bf5aca289a7cff",
        "actual": "3007fa082d7030207553da78cea95a06901e4a93badfa03be3bf5aca289a7cff",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\src\\hrl_v2\\configs\\v2_4_0_lite.json": {
        "expected": "7424127e2248f7318ea9128dfc254bef85e1fec0244dae5fda80c760d4559e23",
        "actual": "7424127e2248f7318ea9128dfc254bef85e1fec0244dae5fda80c760d4559e23",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\src\\hrl_v2\\configs\\physical_scene.json": {
        "expected": "6933abadbb8e34a532de6429477d52441e131996015126e458cbc97494e4c582",
        "actual": "6933abadbb8e34a532de6429477d52441e131996015126e458cbc97494e4c582",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\pilot_contract.sha256": {
        "expected": "82bc558de9183ff75cb16f06ca946673136538420f2c1b31b7f4cb2b5c13d7d1",
        "actual": "82bc558de9183ff75cb16f06ca946673136538420f2c1b31b7f4cb2b5c13d7d1",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\contract_correction.diff": {
        "expected": "0b1d8d37e52b20302a4bf9706b7d68a54bb76592efc23fae995f9c1857129f06",
        "actual": "0b1d8d37e52b20302a4bf9706b7d68a54bb76592efc23fae995f9c1857129f06",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\pilot_eval.py": {
        "expected": "152b2447e12c960d0006103d13d2a22c95d1b907cd7d6fbd7eacc2f034b62d32",
        "actual": "152b2447e12c960d0006103d13d2a22c95d1b907cd7d6fbd7eacc2f034b62d32",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\pilot_main.py": {
        "expected": "d077f87ef11c50e546e7e539b12a1865cf0851dcfae1bae51e8ae9e0369dbc50",
        "actual": "d077f87ef11c50e546e7e539b12a1865cf0851dcfae1bae51e8ae9e0369dbc50",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\pilot_main_r2.py": {
        "expected": "1a123842b707fe717abbf862ad23637426dd77f3a7bf5124537bba222705b111",
        "actual": "1a123842b707fe717abbf862ad23637426dd77f3a7bf5124537bba222705b111",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\pilot_report.py": {
        "expected": "fc50369ce56d730d68412fcbc15a9b0949a5df0b9bf375406987dd1a8a8f3d8d",
        "actual": "fc50369ce56d730d68412fcbc15a9b0949a5df0b9bf375406987dd1a8a8f3d8d",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\report.json": {
        "expected": "df8b9f72f389cc741e0558317912be31a5b890da80db5b7d1b6409568ffe48b0",
        "actual": "df8b9f72f389cc741e0558317912be31a5b890da80db5b7d1b6409568ffe48b0",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0001.json": {
        "expected": "182cde015aa60d74246e2f1122689ce41b8552129c214d56be1aa5104b1bc841",
        "actual": "182cde015aa60d74246e2f1122689ce41b8552129c214d56be1aa5104b1bc841",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0002.json": {
        "expected": "e6dccd426853a5c15a2559f220a45e4b8e5a1951d4fd43f320f904db1827a72f",
        "actual": "e6dccd426853a5c15a2559f220a45e4b8e5a1951d4fd43f320f904db1827a72f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0003.json": {
        "expected": "73007320d34b120e0adffa21c60c28495e8fb9aa3c41b71979b933f4db23ba69",
        "actual": "73007320d34b120e0adffa21c60c28495e8fb9aa3c41b71979b933f4db23ba69",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0004.json": {
        "expected": "e015a8edda5b2afaab85ccc42b0b51096e9472ced7382363af509e75455b05b0",
        "actual": "e015a8edda5b2afaab85ccc42b0b51096e9472ced7382363af509e75455b05b0",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0005.json": {
        "expected": "66a3d6deb05c408a5bc8002ae4613c457866f715e1ce8ef3d5c69296e8841926",
        "actual": "66a3d6deb05c408a5bc8002ae4613c457866f715e1ce8ef3d5c69296e8841926",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0006.json": {
        "expected": "70139168e55d1e1646cbe4c86de9e3e4969f893796e39871de73ce4e3aefceb4",
        "actual": "70139168e55d1e1646cbe4c86de9e3e4969f893796e39871de73ce4e3aefceb4",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0007.json": {
        "expected": "5624fe43c97d92758225b4db3c6936795999ab5c852ec708688898ffcfb8d077",
        "actual": "5624fe43c97d92758225b4db3c6936795999ab5c852ec708688898ffcfb8d077",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0008.json": {
        "expected": "98575946c0912ecaaaf253d8927059f3294d0e53205aafdf60700871a57e17aa",
        "actual": "98575946c0912ecaaaf253d8927059f3294d0e53205aafdf60700871a57e17aa",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0009.json": {
        "expected": "a674c3d9390a1cc477a165ff0aaf3fa387f64d641489b82dac4bb8d08042352b",
        "actual": "a674c3d9390a1cc477a165ff0aaf3fa387f64d641489b82dac4bb8d08042352b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0010.json": {
        "expected": "f053dc4bb63b28941ac5ad7e8afedf3e24ece90fa3a2b6a9d73f78ebfd7c2274",
        "actual": "f053dc4bb63b28941ac5ad7e8afedf3e24ece90fa3a2b6a9d73f78ebfd7c2274",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0411.json": {
        "expected": "3c79b48bfc1e16de326962d6c0b0df04d0f2f31b42ab7078943f5ecb7d0d73e3",
        "actual": "3c79b48bfc1e16de326962d6c0b0df04d0f2f31b42ab7078943f5ecb7d0d73e3",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0412.json": {
        "expected": "897b45e545ba7a0bd81790d6d9e7f1e3b6ebe75adf5f445d7d0bc3da77ce0e7c",
        "actual": "897b45e545ba7a0bd81790d6d9e7f1e3b6ebe75adf5f445d7d0bc3da77ce0e7c",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0413.json": {
        "expected": "eeac6ec8b482838ea5fac5ef8f770a025590099d3851643e12af7f73c434c730",
        "actual": "eeac6ec8b482838ea5fac5ef8f770a025590099d3851643e12af7f73c434c730",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0201.json": {
        "expected": "f6bdc2d0976667d978cc70ceef129ade41670d18357a10deaeaa7db516ff9941",
        "actual": "f6bdc2d0976667d978cc70ceef129ade41670d18357a10deaeaa7db516ff9941",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0202.json": {
        "expected": "382d8c5407ead7504fa70bf2eef0029b618ce6f8b3dc4c4f3455193ea48d4e0a",
        "actual": "382d8c5407ead7504fa70bf2eef0029b618ce6f8b3dc4c4f3455193ea48d4e0a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0203.json": {
        "expected": "a68ee416087085bf0707c4c7729a6fc981d7d0a22059b63723d07e7055caf518",
        "actual": "a68ee416087085bf0707c4c7729a6fc981d7d0a22059b63723d07e7055caf518",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0204.json": {
        "expected": "4084bb2180353af75fdf0965e608e3ac6c6a63dbdd99ed0248a3ce540d4003d2",
        "actual": "4084bb2180353af75fdf0965e608e3ac6c6a63dbdd99ed0248a3ce540d4003d2",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0205.json": {
        "expected": "2b13e97cc7d61f6bd0cd7486369f64496be2d513a525c3070daab1ac3b015bb0",
        "actual": "2b13e97cc7d61f6bd0cd7486369f64496be2d513a525c3070daab1ac3b015bb0",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0206.json": {
        "expected": "f2cf9b33078c47a3bcde70a7d69f2781fb8a3e2cedb075d79aa8561bda3cbba4",
        "actual": "f2cf9b33078c47a3bcde70a7d69f2781fb8a3e2cedb075d79aa8561bda3cbba4",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0207.json": {
        "expected": "466def791bedfc9f0efc3d530e57e0d9cc5ae70ef8874794fb15ef1a0de12265",
        "actual": "466def791bedfc9f0efc3d530e57e0d9cc5ae70ef8874794fb15ef1a0de12265",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0208.json": {
        "expected": "a231a4bac91cface82720b1b654fd5791c457cbe48750527e4ba0d5736898359",
        "actual": "a231a4bac91cface82720b1b654fd5791c457cbe48750527e4ba0d5736898359",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0209.json": {
        "expected": "a6000281159f3ce07e7c2de974e9a8b30331076122998484e530b3da27a09689",
        "actual": "a6000281159f3ce07e7c2de974e9a8b30331076122998484e530b3da27a09689",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0210.json": {
        "expected": "b6fea9ebcc97148084f4f1b86b99783eff01017445343b7485ceb089c65d1c16",
        "actual": "b6fea9ebcc97148084f4f1b86b99783eff01017445343b7485ceb089c65d1c16",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0211.json": {
        "expected": "ffb726044398aa6a4a5b94cb063588d1a8416fdb64e2390a4a49d68e001d7b70",
        "actual": "ffb726044398aa6a4a5b94cb063588d1a8416fdb64e2390a4a49d68e001d7b70",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0212.json": {
        "expected": "32cf882ba10853d77facc0f587619c5ea2269d113a9e36f6dad8de2ae46cff3b",
        "actual": "32cf882ba10853d77facc0f587619c5ea2269d113a9e36f6dad8de2ae46cff3b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0213.json": {
        "expected": "66e9e1f7714763862deb149962b1bef7d037a6a7d601688c70354c005f840bc5",
        "actual": "66e9e1f7714763862deb149962b1bef7d037a6a7d601688c70354c005f840bc5",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0214.json": {
        "expected": "0a4d18ba07c43e07f201aeb58d13230c887fdb4d34c2549c86cdc4517858309e",
        "actual": "0a4d18ba07c43e07f201aeb58d13230c887fdb4d34c2549c86cdc4517858309e",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0215.json": {
        "expected": "e058bda7b5c93b49dba5474dcc4d8fec8e491adbbcd30798423cccdc37313aaa",
        "actual": "e058bda7b5c93b49dba5474dcc4d8fec8e491adbbcd30798423cccdc37313aaa",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0216.json": {
        "expected": "2b4fd86d01f645608acbd8a28d6c25623d6c374114caf9d1f8d31d19b0b9fd39",
        "actual": "2b4fd86d01f645608acbd8a28d6c25623d6c374114caf9d1f8d31d19b0b9fd39",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0217.json": {
        "expected": "d0a361052db2080ec3a7020a08890fbb44dbe232d0067b8a7453f6b9bc45d5ed",
        "actual": "d0a361052db2080ec3a7020a08890fbb44dbe232d0067b8a7453f6b9bc45d5ed",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0218.json": {
        "expected": "03d94bb5522abf72a6636ae7b9d088f654fe1b247ceb58222d3d3b859525b820",
        "actual": "03d94bb5522abf72a6636ae7b9d088f654fe1b247ceb58222d3d3b859525b820",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0219.json": {
        "expected": "1767d646d7cb31fa228d54a13a479cdd94378aa193b22fa17a4674d5763e3875",
        "actual": "1767d646d7cb31fa228d54a13a479cdd94378aa193b22fa17a4674d5763e3875",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0220.json": {
        "expected": "ba9cce85199502cd8046e2bca2ff7a4e490ad695acc3e97a38802b980d7fd84e",
        "actual": "ba9cce85199502cd8046e2bca2ff7a4e490ad695acc3e97a38802b980d7fd84e",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0221.json": {
        "expected": "776cf8b587a44c5ec255b4f6b957e2470a43982723ff1c2292f6f848a1803e37",
        "actual": "776cf8b587a44c5ec255b4f6b957e2470a43982723ff1c2292f6f848a1803e37",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0222.json": {
        "expected": "054b6a95e90e2a20a09c2876f08af085459e3585152bd27344bee241bcaa56c7",
        "actual": "054b6a95e90e2a20a09c2876f08af085459e3585152bd27344bee241bcaa56c7",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0223.json": {
        "expected": "ec519e65d09d44e3ca738e4c1954b8a5b37878093c446dcf2b9f7a4d7326456a",
        "actual": "ec519e65d09d44e3ca738e4c1954b8a5b37878093c446dcf2b9f7a4d7326456a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0224.json": {
        "expected": "b0e20ede47942702aac8b91113404b5239bf4c46fdb341b63164d9d58eeb26cb",
        "actual": "b0e20ede47942702aac8b91113404b5239bf4c46fdb341b63164d9d58eeb26cb",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0225.json": {
        "expected": "2b294e63cd801ff66fc54650a1497752ac92d6083a95deec338f6538d23cff13",
        "actual": "2b294e63cd801ff66fc54650a1497752ac92d6083a95deec338f6538d23cff13",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0226.json": {
        "expected": "098dc8afaf9ba3c030fd20a35508f4acec4f509c43ef7000dca8d7ea06e030cf",
        "actual": "098dc8afaf9ba3c030fd20a35508f4acec4f509c43ef7000dca8d7ea06e030cf",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0227.json": {
        "expected": "102ba410dd3dfdc86a36f640d67d36ce978475f99f2345a271dab61af7909cae",
        "actual": "102ba410dd3dfdc86a36f640d67d36ce978475f99f2345a271dab61af7909cae",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0228.json": {
        "expected": "9627d4432e3c6e6f5124b01c266c3b71957334cd58db13494d1c763490053c19",
        "actual": "9627d4432e3c6e6f5124b01c266c3b71957334cd58db13494d1c763490053c19",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0229.json": {
        "expected": "a37e1151823c76e4a432dbd9c2d4b532fc9b23cdbf43a81858e59ca6b51d2477",
        "actual": "a37e1151823c76e4a432dbd9c2d4b532fc9b23cdbf43a81858e59ca6b51d2477",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0230.json": {
        "expected": "8e450f3ab5bd65594b5f24c9692c5f6504e9eda6fc3856ade39bfc7989499ad4",
        "actual": "8e450f3ab5bd65594b5f24c9692c5f6504e9eda6fc3856ade39bfc7989499ad4",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0231.json": {
        "expected": "d6025e215d60b0a32712beccd25487e765d1803c624a5e5c8133930057a93459",
        "actual": "d6025e215d60b0a32712beccd25487e765d1803c624a5e5c8133930057a93459",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0232.json": {
        "expected": "fe16c888c685f42fb609ce9936e4044bc4923f42452f0b71e0ed415d748fc348",
        "actual": "fe16c888c685f42fb609ce9936e4044bc4923f42452f0b71e0ed415d748fc348",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0233.json": {
        "expected": "b4cfe4ea83e51e30e8c9a9647e85466b9fefd41893a5524bc85b4ae8e4d5485e",
        "actual": "b4cfe4ea83e51e30e8c9a9647e85466b9fefd41893a5524bc85b4ae8e4d5485e",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0234.json": {
        "expected": "def6cd4649e6133ba6b0175122fafa4100ff6536bf35d50c3793ea6cb017c748",
        "actual": "def6cd4649e6133ba6b0175122fafa4100ff6536bf35d50c3793ea6cb017c748",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0235.json": {
        "expected": "6ba5bbc35204a1e3bdfc8208152035211a805e14758a767a7ca9020adb9d8b94",
        "actual": "6ba5bbc35204a1e3bdfc8208152035211a805e14758a767a7ca9020adb9d8b94",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0236.json": {
        "expected": "eefc12cd0b594cb60d1ecd11f279307d822ffe26b372aaf01b81d32a4257866a",
        "actual": "eefc12cd0b594cb60d1ecd11f279307d822ffe26b372aaf01b81d32a4257866a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0237.json": {
        "expected": "9236329604ba2c49100297efd38d67042042faae46121eca5957d1c1db08e394",
        "actual": "9236329604ba2c49100297efd38d67042042faae46121eca5957d1c1db08e394",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0238.json": {
        "expected": "89f308cde2b8ffad44d599e067cd2a6dde7ac36e88516d8eb8f35745d35c290c",
        "actual": "89f308cde2b8ffad44d599e067cd2a6dde7ac36e88516d8eb8f35745d35c290c",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0239.json": {
        "expected": "c1a8751adcc6d8baaa00600e42290d002e0f8c2bd07370965dbf31013af722bb",
        "actual": "c1a8751adcc6d8baaa00600e42290d002e0f8c2bd07370965dbf31013af722bb",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0240.json": {
        "expected": "c8320e26cc1cc7e5b9be39753cf03e76606da508573c025082e8c1279544e2dc",
        "actual": "c8320e26cc1cc7e5b9be39753cf03e76606da508573c025082e8c1279544e2dc",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0241.json": {
        "expected": "714e7e5d260fa180e488afd97ab22041f72847502b30ecc5e5f26728414bc0b8",
        "actual": "714e7e5d260fa180e488afd97ab22041f72847502b30ecc5e5f26728414bc0b8",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0242.json": {
        "expected": "2cf525b927deb89de2231c0d1c0f25ac53222e9cf48055d64005a48c7a90b3a4",
        "actual": "2cf525b927deb89de2231c0d1c0f25ac53222e9cf48055d64005a48c7a90b3a4",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0243.json": {
        "expected": "c2daaf174a482f474a17504bdf3e5d88a402db4253bfb29addcee7aecb747497",
        "actual": "c2daaf174a482f474a17504bdf3e5d88a402db4253bfb29addcee7aecb747497",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0244.json": {
        "expected": "4833613355fdf1c84a679f311cf0f58adda7befbd93e1ad66abb1769639fae77",
        "actual": "4833613355fdf1c84a679f311cf0f58adda7befbd93e1ad66abb1769639fae77",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0245.json": {
        "expected": "a4e2224ca314c00523cf021b42b31eef09936ba379357b21d8fa9e735fee0b24",
        "actual": "a4e2224ca314c00523cf021b42b31eef09936ba379357b21d8fa9e735fee0b24",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0246.json": {
        "expected": "fa0156c58eb9c802e1106763e03e89cf8d4fb43b905c99e7e3f15879af0c4628",
        "actual": "fa0156c58eb9c802e1106763e03e89cf8d4fb43b905c99e7e3f15879af0c4628",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0247.json": {
        "expected": "f83f886d9d91d3036c0f2f61a8f54a9e875f6376bd1d34dda085059bae2ef6e3",
        "actual": "f83f886d9d91d3036c0f2f61a8f54a9e875f6376bd1d34dda085059bae2ef6e3",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0248.json": {
        "expected": "955328559880d7ae0fd610bc56b3d5c08797d1ebef9da1b4248b5f4e9643ba1c",
        "actual": "955328559880d7ae0fd610bc56b3d5c08797d1ebef9da1b4248b5f4e9643ba1c",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0249.json": {
        "expected": "555e6c7d12ca7d68ed71354f66b76dc8a93fae9e4ea240c299c31eab29a9b273",
        "actual": "555e6c7d12ca7d68ed71354f66b76dc8a93fae9e4ea240c299c31eab29a9b273",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0250.json": {
        "expected": "7cb465c0986be1c0a6ed6efd17374440e1b952fcebff7b66ce1d59a4eb56c5dc",
        "actual": "7cb465c0986be1c0a6ed6efd17374440e1b952fcebff7b66ce1d59a4eb56c5dc",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0251.json": {
        "expected": "ef744fa5633ec0bd068d388eee94234e88b40dba8191829d4651d4108504923b",
        "actual": "ef744fa5633ec0bd068d388eee94234e88b40dba8191829d4651d4108504923b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0252.json": {
        "expected": "38b5ff57ce32b71d04ad552f84feb0b224f8d0c5d464367de6f14c524e37ff94",
        "actual": "38b5ff57ce32b71d04ad552f84feb0b224f8d0c5d464367de6f14c524e37ff94",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0253.json": {
        "expected": "c45ac0e219e24819c003efc00fab5a6360041180e346d8aaed15c1fc3829a7b2",
        "actual": "c45ac0e219e24819c003efc00fab5a6360041180e346d8aaed15c1fc3829a7b2",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0254.json": {
        "expected": "f19c0aeef1e63a1a41d4a3485325cd2302d1422f62ab821fd58ff26c84bc1c0a",
        "actual": "f19c0aeef1e63a1a41d4a3485325cd2302d1422f62ab821fd58ff26c84bc1c0a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0255.json": {
        "expected": "c5ab18388b2a845840e3692a17a43460cb37d9aa8e5a829ff2194b6bcc1100f0",
        "actual": "c5ab18388b2a845840e3692a17a43460cb37d9aa8e5a829ff2194b6bcc1100f0",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0256.json": {
        "expected": "2a9570f5c99681e129ba15f6c62ad0e10bd17b2483fad4ce97addd7775a47456",
        "actual": "2a9570f5c99681e129ba15f6c62ad0e10bd17b2483fad4ce97addd7775a47456",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0257.json": {
        "expected": "6c6911367f33377039026bbd58eefa8dbc13449d3d3868db831b7319c59ab366",
        "actual": "6c6911367f33377039026bbd58eefa8dbc13449d3d3868db831b7319c59ab366",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0258.json": {
        "expected": "765bf3fa712cee6270df088364a556b340a1600429473c71b161ae8c046f36be",
        "actual": "765bf3fa712cee6270df088364a556b340a1600429473c71b161ae8c046f36be",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0259.json": {
        "expected": "013819ef67f2eb29a6895749f895b66560ad935188235a7f7ac7a82b10ccfed3",
        "actual": "013819ef67f2eb29a6895749f895b66560ad935188235a7f7ac7a82b10ccfed3",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0260.json": {
        "expected": "43952e9b3187f03cc5a4e9f3e2e8f46b2075b264a374409017b9032318091afd",
        "actual": "43952e9b3187f03cc5a4e9f3e2e8f46b2075b264a374409017b9032318091afd",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0261.json": {
        "expected": "7e728c8b199a7121ea32020442292755603d94397af8b9c610813711a74791bc",
        "actual": "7e728c8b199a7121ea32020442292755603d94397af8b9c610813711a74791bc",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0262.json": {
        "expected": "fa61def1675de1dbe5f2069754cbd5f64e615359a7aea6dde867e2cbea27b075",
        "actual": "fa61def1675de1dbe5f2069754cbd5f64e615359a7aea6dde867e2cbea27b075",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0263.json": {
        "expected": "99980d40a4c80a26057623fafbb0dec8bb64aaf4cb3b0843c063b0cfac8dc462",
        "actual": "99980d40a4c80a26057623fafbb0dec8bb64aaf4cb3b0843c063b0cfac8dc462",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0264.json": {
        "expected": "17bbc5042f3ad988e1a2bbf84037f2067e468e1a79795ffccff4b14c3ae2b994",
        "actual": "17bbc5042f3ad988e1a2bbf84037f2067e468e1a79795ffccff4b14c3ae2b994",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0265.json": {
        "expected": "4501152c99a422b6f83624c9bcee34bb34abca179059826432e258641a18b32a",
        "actual": "4501152c99a422b6f83624c9bcee34bb34abca179059826432e258641a18b32a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0266.json": {
        "expected": "fb8010df0edda2c7a31f2b4c318afcf82e2529f3b88210268057578d28123bef",
        "actual": "fb8010df0edda2c7a31f2b4c318afcf82e2529f3b88210268057578d28123bef",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0267.json": {
        "expected": "6c8a525e67d20ab0c4a19c25537a6543cea97fecb23a55a3f7307509f468760f",
        "actual": "6c8a525e67d20ab0c4a19c25537a6543cea97fecb23a55a3f7307509f468760f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0268.json": {
        "expected": "47da2f492b2e1b02c42488c6f98bc0ba93d32a88455cb6561b5be78df4f213a3",
        "actual": "47da2f492b2e1b02c42488c6f98bc0ba93d32a88455cb6561b5be78df4f213a3",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0269.json": {
        "expected": "cfab0ae55adfd8bfb3cfe915b0aee0c3323a3045e4d8fa9f3fc11a0483d792da",
        "actual": "cfab0ae55adfd8bfb3cfe915b0aee0c3323a3045e4d8fa9f3fc11a0483d792da",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0270.json": {
        "expected": "89c99219bd1036c8955465cc0f11c0fd862c501cbe504d0a4e16ab98a592ec8b",
        "actual": "89c99219bd1036c8955465cc0f11c0fd862c501cbe504d0a4e16ab98a592ec8b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0271.json": {
        "expected": "db7691d7941f71a755d78ec08167f9140adeae5f44c8e345bde5cb25dbc54b81",
        "actual": "db7691d7941f71a755d78ec08167f9140adeae5f44c8e345bde5cb25dbc54b81",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0272.json": {
        "expected": "6a8634bc806c45e14db7da34b9dab1414552d5a7cd99eb1c082ab1319bc2f5aa",
        "actual": "6a8634bc806c45e14db7da34b9dab1414552d5a7cd99eb1c082ab1319bc2f5aa",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0273.json": {
        "expected": "6a391a2ad5068c3a775f70b33492b6bbca2a8b3894d82d3f01926f8dbac1accf",
        "actual": "6a391a2ad5068c3a775f70b33492b6bbca2a8b3894d82d3f01926f8dbac1accf",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0274.json": {
        "expected": "3a2931a3a49079942982547029be8e1f6634e1bffa71c31f8724d11932d052d5",
        "actual": "3a2931a3a49079942982547029be8e1f6634e1bffa71c31f8724d11932d052d5",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0275.json": {
        "expected": "b874525f56b6b55de3d59e9170cecf82c68d6e73643d28e8354d07775b369644",
        "actual": "b874525f56b6b55de3d59e9170cecf82c68d6e73643d28e8354d07775b369644",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0276.json": {
        "expected": "98676dc54c61bd530e4cee6f762186baf02fdb834d512e6cdcf814933ba39479",
        "actual": "98676dc54c61bd530e4cee6f762186baf02fdb834d512e6cdcf814933ba39479",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0277.json": {
        "expected": "6525c8ffced52f1ae9c6aaa7c0c67802bcc25c444c7fe36a0f208429c0b4e6b0",
        "actual": "6525c8ffced52f1ae9c6aaa7c0c67802bcc25c444c7fe36a0f208429c0b4e6b0",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0278.json": {
        "expected": "63d72317be7c83363b5b0651fb5a97e8cb63c7b93228177e02433fee75f262d2",
        "actual": "63d72317be7c83363b5b0651fb5a97e8cb63c7b93228177e02433fee75f262d2",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0279.json": {
        "expected": "fd93a164f74ae5c2689e164973ffd07cdf1361877239679c3a8014f83d7524cd",
        "actual": "fd93a164f74ae5c2689e164973ffd07cdf1361877239679c3a8014f83d7524cd",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0280.json": {
        "expected": "867b05e14942c128a6c8784673f6c8e67632618e5050bea21e1e90f6beb2eb69",
        "actual": "867b05e14942c128a6c8784673f6c8e67632618e5050bea21e1e90f6beb2eb69",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0281.json": {
        "expected": "9dbfc31c788711b4aae561b5908f672649cbbd6b81fc0c7972c16c78f19c405d",
        "actual": "9dbfc31c788711b4aae561b5908f672649cbbd6b81fc0c7972c16c78f19c405d",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0282.json": {
        "expected": "a97e1b9b4a0659daaea7600367cff341e701b7a3dd76ec307cf5b6a56f30dd32",
        "actual": "a97e1b9b4a0659daaea7600367cff341e701b7a3dd76ec307cf5b6a56f30dd32",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0283.json": {
        "expected": "d9739bddd99aed6592a042b405d3e29bc66dcc6498d75a271ba70f4f4714ca25",
        "actual": "d9739bddd99aed6592a042b405d3e29bc66dcc6498d75a271ba70f4f4714ca25",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0284.json": {
        "expected": "a4978aa2595fefd28624a53a95ef7cb5455cdff7f7a30c5611b12918d7e88f50",
        "actual": "a4978aa2595fefd28624a53a95ef7cb5455cdff7f7a30c5611b12918d7e88f50",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0285.json": {
        "expected": "94bf738a083e5a1fd691bb6cf9b224c07e4c9d177b486d54a2cccb50cfb09ad3",
        "actual": "94bf738a083e5a1fd691bb6cf9b224c07e4c9d177b486d54a2cccb50cfb09ad3",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0286.json": {
        "expected": "a01388f5776c8ba0157113aa40e7b114ef312084b50e4415f2cb58dc8c8b7248",
        "actual": "a01388f5776c8ba0157113aa40e7b114ef312084b50e4415f2cb58dc8c8b7248",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0287.json": {
        "expected": "bb2c979dcf0ab11f60ca79320fe032a9b31f0968bdbe945aa8050f51904b7621",
        "actual": "bb2c979dcf0ab11f60ca79320fe032a9b31f0968bdbe945aa8050f51904b7621",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0288.json": {
        "expected": "98a09efd6756d0fb07f69dfa783fecf2c08618acc69f483a73c15026a4cc24cf",
        "actual": "98a09efd6756d0fb07f69dfa783fecf2c08618acc69f483a73c15026a4cc24cf",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0289.json": {
        "expected": "e5770550240a936b3b734a34a1a8d9e21160e1ab5c604402120cefd6ec496b3c",
        "actual": "e5770550240a936b3b734a34a1a8d9e21160e1ab5c604402120cefd6ec496b3c",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0290.json": {
        "expected": "a924809703a888b297174570259bf4bc2c7f3c289e9441e17f93a9898ad14f09",
        "actual": "a924809703a888b297174570259bf4bc2c7f3c289e9441e17f93a9898ad14f09",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0291.json": {
        "expected": "60187227d22ad20146dcc9e417c4fcf2331b565a5bc9d07e8cb0b08c4dbf4cb9",
        "actual": "60187227d22ad20146dcc9e417c4fcf2331b565a5bc9d07e8cb0b08c4dbf4cb9",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0292.json": {
        "expected": "1fac2d3bf156e394147ffbfa2adbb6748728544cfdf99df3155a630df3b4f9e6",
        "actual": "1fac2d3bf156e394147ffbfa2adbb6748728544cfdf99df3155a630df3b4f9e6",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0293.json": {
        "expected": "fb995696a826ce04d8c81ecaccb9c4494ee8ca76d6203883f8fd614c2b25e578",
        "actual": "fb995696a826ce04d8c81ecaccb9c4494ee8ca76d6203883f8fd614c2b25e578",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0294.json": {
        "expected": "f926217aab3cf2aac0c28b54d0aa941c9d9a30d86300771f25b803a177da553f",
        "actual": "f926217aab3cf2aac0c28b54d0aa941c9d9a30d86300771f25b803a177da553f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0295.json": {
        "expected": "9dd2738ba63096ffdf8a876b25b307788263f689168ae54052f867c6afa6fdba",
        "actual": "9dd2738ba63096ffdf8a876b25b307788263f689168ae54052f867c6afa6fdba",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0296.json": {
        "expected": "b5f71d0272728ca7992ce1254c8c2c2554c7f20f3efaa41ce4f1dd54a8869e4e",
        "actual": "b5f71d0272728ca7992ce1254c8c2c2554c7f20f3efaa41ce4f1dd54a8869e4e",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0297.json": {
        "expected": "94d8040ce62d7b1da72c4de5c05ac40d3800ecfef72bd4b4129156860acb3b1e",
        "actual": "94d8040ce62d7b1da72c4de5c05ac40d3800ecfef72bd4b4129156860acb3b1e",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0298.json": {
        "expected": "93dc27a2156bb027de45cf3a35e7519c1689473b12daf37356a5e4f05fa30701",
        "actual": "93dc27a2156bb027de45cf3a35e7519c1689473b12daf37356a5e4f05fa30701",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0299.json": {
        "expected": "cabaf87d647719b1f29c0c40397380e10e1931f72b29617e84ead4a8f7525138",
        "actual": "cabaf87d647719b1f29c0c40397380e10e1931f72b29617e84ead4a8f7525138",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\positions\\episode-0300.json": {
        "expected": "ab744efd1d7c99135c04525374a56a2c5bdbb65bf26c996e2f122d633afa756b",
        "actual": "ab744efd1d7c99135c04525374a56a2c5bdbb65bf26c996e2f122d633afa756b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2_speedup\\outputs\\lite_v2_4_0\\multiseed_20261002_203333_r2\\runs\\Proposed\\seed_4104\\train_metrics.jsonl": {
        "expected": "4b13eed7f9089bc961493ccb9b3cd59dc9d3ccee9b52877b76ecb25b7aed607a",
        "actual": "4b13eed7f9089bc961493ccb9b3cd59dc9d3ccee9b52877b76ecb25b7aed607a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_randpre_pilot_20261005\\seed_4104\\step1_attempt1\\train_metrics.jsonl": {
        "expected": "fada2f3b1ea99b29b960de62f47308dcaa7ff2d6d565714f7fbc7dfdaf13ee53",
        "actual": "fada2f3b1ea99b29b960de62f47308dcaa7ff2d6d565714f7fbc7dfdaf13ee53",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0011.json": {
        "expected": "7ea08e894e2883be12d1d6330649ce55bdd3558efe81251a0facc454fdb787c4",
        "actual": "7ea08e894e2883be12d1d6330649ce55bdd3558efe81251a0facc454fdb787c4",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0012.json": {
        "expected": "e49695f32f913872fb84f91441740f67d49a43f64a555d397503c002057f95a4",
        "actual": "e49695f32f913872fb84f91441740f67d49a43f64a555d397503c002057f95a4",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0013.json": {
        "expected": "cd3618922875b5f3713bc23fe6875846e5a9053495587f059b89d6163e1384d1",
        "actual": "cd3618922875b5f3713bc23fe6875846e5a9053495587f059b89d6163e1384d1",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0014.json": {
        "expected": "a86061c071d55adbab472331898a595cbd618999283233520dc2ef118dbd4291",
        "actual": "a86061c071d55adbab472331898a595cbd618999283233520dc2ef118dbd4291",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0015.json": {
        "expected": "431d6916ba8557cb62830232bf249172b49850941f38eb16a67f3d79b36e2192",
        "actual": "431d6916ba8557cb62830232bf249172b49850941f38eb16a67f3d79b36e2192",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0016.json": {
        "expected": "760201e6559ca8d824a77c5d16ab35239e3509dae692bffa58b3d65b002491f7",
        "actual": "760201e6559ca8d824a77c5d16ab35239e3509dae692bffa58b3d65b002491f7",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0017.json": {
        "expected": "f0ddf303a47624f7d627d84f41c8e7a95a5a77c4c7595a1325642816866c5dd9",
        "actual": "f0ddf303a47624f7d627d84f41c8e7a95a5a77c4c7595a1325642816866c5dd9",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0018.json": {
        "expected": "b8914ddcee92844dd621fd0a7703c23f9877fdbc857ef57aa6ce945ea558c225",
        "actual": "b8914ddcee92844dd621fd0a7703c23f9877fdbc857ef57aa6ce945ea558c225",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0019.json": {
        "expected": "94ec9fab072b064d4694861b60edf062d5cc5e18ba895be96bd9b5e29d0389c3",
        "actual": "94ec9fab072b064d4694861b60edf062d5cc5e18ba895be96bd9b5e29d0389c3",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0020.json": {
        "expected": "611960b721d440ea23ec29aa56c1bc42fdfe27322c698080e93e53058be3aecf",
        "actual": "611960b721d440ea23ec29aa56c1bc42fdfe27322c698080e93e53058be3aecf",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0021.json": {
        "expected": "6aee40948d4e62a6f15ef9ad649804d5d251550df874659e24dedbb39fa1dc9b",
        "actual": "6aee40948d4e62a6f15ef9ad649804d5d251550df874659e24dedbb39fa1dc9b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0022.json": {
        "expected": "00c1381b30a6b4a1172615926423fa50c67d6a6714026ad9ba15ac6ae7d03f4a",
        "actual": "00c1381b30a6b4a1172615926423fa50c67d6a6714026ad9ba15ac6ae7d03f4a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0023.json": {
        "expected": "ac198dce794fa19c1e0644821b4a852b1cb7092c185303125fca63b94a1485d7",
        "actual": "ac198dce794fa19c1e0644821b4a852b1cb7092c185303125fca63b94a1485d7",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0024.json": {
        "expected": "7b805355674ac1ffc02730bb2fb86d9b1676cea634c1c93db0adff223eea2c45",
        "actual": "7b805355674ac1ffc02730bb2fb86d9b1676cea634c1c93db0adff223eea2c45",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0025.json": {
        "expected": "f6fc336032fbcf607077c5d8051b162ffdd7013d32baa1e10eda4ff84400f0da",
        "actual": "f6fc336032fbcf607077c5d8051b162ffdd7013d32baa1e10eda4ff84400f0da",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0026.json": {
        "expected": "9260989942b1c78ff20774f8181cd9065c4449c9fbba8a03ff32f687c718231a",
        "actual": "9260989942b1c78ff20774f8181cd9065c4449c9fbba8a03ff32f687c718231a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0027.json": {
        "expected": "9b9799098daee5ca8123aff194e6a3c3e34be437e05eb7e12df2b4d23c0ca8ca",
        "actual": "9b9799098daee5ca8123aff194e6a3c3e34be437e05eb7e12df2b4d23c0ca8ca",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0028.json": {
        "expected": "3441fd14917b36a0328a44b2d29ca7c50682191cacab5fcf08a74a80d59793d4",
        "actual": "3441fd14917b36a0328a44b2d29ca7c50682191cacab5fcf08a74a80d59793d4",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0029.json": {
        "expected": "07af826ea192c9abf7abdc1009acfdae2cd7ddb2cb29dacc0189f4a3b1aa0bf1",
        "actual": "07af826ea192c9abf7abdc1009acfdae2cd7ddb2cb29dacc0189f4a3b1aa0bf1",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0030.json": {
        "expected": "c096e917e451cb0152f484b3bf0792397c57337613d24f91b4f11ff81c3061c4",
        "actual": "c096e917e451cb0152f484b3bf0792397c57337613d24f91b4f11ff81c3061c4",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0031.json": {
        "expected": "0d7e9e2241aa79202eedfc327a05ec0f70dd68f4471a5f4fbf7b555429a9991e",
        "actual": "0d7e9e2241aa79202eedfc327a05ec0f70dd68f4471a5f4fbf7b555429a9991e",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0032.json": {
        "expected": "eb8e15848bd2f4e7cb381b4438753d65c2e81086b18deb4c6dd73203f5d318fa",
        "actual": "eb8e15848bd2f4e7cb381b4438753d65c2e81086b18deb4c6dd73203f5d318fa",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0033.json": {
        "expected": "ac52c98774c5fefcef306e9124ebc748c39c8d03149aee1fcd2f311a5c62a965",
        "actual": "ac52c98774c5fefcef306e9124ebc748c39c8d03149aee1fcd2f311a5c62a965",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0034.json": {
        "expected": "ad93b98ae2b545dbfc6d90dd733d569944055029544ba8faaf409867b4577c9d",
        "actual": "ad93b98ae2b545dbfc6d90dd733d569944055029544ba8faaf409867b4577c9d",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0035.json": {
        "expected": "8feb9b345f6ac447fa3a5260fec6c7f484b398ac8049fbb5aa78c907fb960f9f",
        "actual": "8feb9b345f6ac447fa3a5260fec6c7f484b398ac8049fbb5aa78c907fb960f9f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0036.json": {
        "expected": "59302369a1d747960a01c31c10f999a268c92bdc9560401e771cda3f6cae3253",
        "actual": "59302369a1d747960a01c31c10f999a268c92bdc9560401e771cda3f6cae3253",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0037.json": {
        "expected": "af991f86e2e0b17fef1ef50ef4feccd458d4c938dd88eb78c1ac3b0d8e2b4d23",
        "actual": "af991f86e2e0b17fef1ef50ef4feccd458d4c938dd88eb78c1ac3b0d8e2b4d23",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0038.json": {
        "expected": "8616e47425de1f30f686feef16f43d150822dcb7db1c882c928404cd8a033c29",
        "actual": "8616e47425de1f30f686feef16f43d150822dcb7db1c882c928404cd8a033c29",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0039.json": {
        "expected": "09695aa122b52fac10130d31f83a11abb98cddc0a8d4b63afe12e8b30a7cb925",
        "actual": "09695aa122b52fac10130d31f83a11abb98cddc0a8d4b63afe12e8b30a7cb925",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0040.json": {
        "expected": "18fd2ffee20deca81d6074e2f6e15286729a2f1639a476021177b3d62443e36c",
        "actual": "18fd2ffee20deca81d6074e2f6e15286729a2f1639a476021177b3d62443e36c",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0041.json": {
        "expected": "a82cb70e3e5c50aa0a99ca1882b784fa765530ed6d0ae7f982b3aedae149118e",
        "actual": "a82cb70e3e5c50aa0a99ca1882b784fa765530ed6d0ae7f982b3aedae149118e",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0042.json": {
        "expected": "bd6efc7db32ecefdbaff8541f524ef80706ed19a67ec6e9d7a1ca1653b661144",
        "actual": "bd6efc7db32ecefdbaff8541f524ef80706ed19a67ec6e9d7a1ca1653b661144",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0043.json": {
        "expected": "ac10e5a6145fc20da0148cf230ef011fa09c856d8d611151d8447520acc8d3ae",
        "actual": "ac10e5a6145fc20da0148cf230ef011fa09c856d8d611151d8447520acc8d3ae",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0044.json": {
        "expected": "457ed12c65fd25f5903f3b0476ff6d30f3a1012bba44701e81bafa09745a8218",
        "actual": "457ed12c65fd25f5903f3b0476ff6d30f3a1012bba44701e81bafa09745a8218",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0045.json": {
        "expected": "147bf8e29b0edcf2446bbd6bb448da0c659c7e8536792d56e40faf3865d6a8d0",
        "actual": "147bf8e29b0edcf2446bbd6bb448da0c659c7e8536792d56e40faf3865d6a8d0",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0046.json": {
        "expected": "f800842e5c7c8bd462cb7c33db1fc2a126a8221c58dc40612c2c295f72cd2b1d",
        "actual": "f800842e5c7c8bd462cb7c33db1fc2a126a8221c58dc40612c2c295f72cd2b1d",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0047.json": {
        "expected": "fcde5446c0c981244b97bf9ca64f66e35334243fa7be1380b46df3e3678c5f3d",
        "actual": "fcde5446c0c981244b97bf9ca64f66e35334243fa7be1380b46df3e3678c5f3d",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0048.json": {
        "expected": "53219d2abf48b4d7eafa9f327ee6b06f8592bd1c7174cc3ddd5a79ba5230046c",
        "actual": "53219d2abf48b4d7eafa9f327ee6b06f8592bd1c7174cc3ddd5a79ba5230046c",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0049.json": {
        "expected": "d7c3e32274a9abad34ee40455bd18a7603d321c6fc60d8c6bc6ba9f65e5bf2a7",
        "actual": "d7c3e32274a9abad34ee40455bd18a7603d321c6fc60d8c6bc6ba9f65e5bf2a7",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0050.json": {
        "expected": "b891deaa034e2ad3d04f9ffb5aaa4733b3e15dc926f66fa94bc20610dd4215f6",
        "actual": "b891deaa034e2ad3d04f9ffb5aaa4733b3e15dc926f66fa94bc20610dd4215f6",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0051.json": {
        "expected": "557262d52fa41f070624cf2a30d94d82cca11bfd2d4eab69340a3e5704654486",
        "actual": "557262d52fa41f070624cf2a30d94d82cca11bfd2d4eab69340a3e5704654486",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0052.json": {
        "expected": "ba0499415980104b869d4c79dccf6a1ee6b3f51777018ebfb149d82394059628",
        "actual": "ba0499415980104b869d4c79dccf6a1ee6b3f51777018ebfb149d82394059628",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0053.json": {
        "expected": "c7401b15ff9eb0716c587d4702aee98fa71563d9d3c7edd8c9c9877574f2e3f0",
        "actual": "c7401b15ff9eb0716c587d4702aee98fa71563d9d3c7edd8c9c9877574f2e3f0",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0054.json": {
        "expected": "d7c881fc71ec62406f25c85436096207dc09ffec10bd5650fd7c45a5f1e8f42b",
        "actual": "d7c881fc71ec62406f25c85436096207dc09ffec10bd5650fd7c45a5f1e8f42b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0055.json": {
        "expected": "80e0d3fce724dd0e1baddf28c1c705fe75f6db19bb664fdfe86f4deb398847cd",
        "actual": "80e0d3fce724dd0e1baddf28c1c705fe75f6db19bb664fdfe86f4deb398847cd",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0056.json": {
        "expected": "1cbb9998a5387707664441d242f40c6ab23547bb2e789aca35366dcbaf2a8e4e",
        "actual": "1cbb9998a5387707664441d242f40c6ab23547bb2e789aca35366dcbaf2a8e4e",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0057.json": {
        "expected": "e223ad4659d9726133fb9007cc281e2a9fa89b3a9035e187a1c4d653d7d8f4dd",
        "actual": "e223ad4659d9726133fb9007cc281e2a9fa89b3a9035e187a1c4d653d7d8f4dd",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0058.json": {
        "expected": "9e4457e221960d8626ae03788391590c64254e15f4c0562a3c09c436cd27bb93",
        "actual": "9e4457e221960d8626ae03788391590c64254e15f4c0562a3c09c436cd27bb93",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0059.json": {
        "expected": "889807565db536456d65db2441549b338aed9711b956df55696f4003e074378f",
        "actual": "889807565db536456d65db2441549b338aed9711b956df55696f4003e074378f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0060.json": {
        "expected": "49e6ae19f329fc529b075dbfbde875a55ff5a414c66d8a5add46276fdd0e51bf",
        "actual": "49e6ae19f329fc529b075dbfbde875a55ff5a414c66d8a5add46276fdd0e51bf",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0061.json": {
        "expected": "a91b5001eeb670be23cbced0c03e52de498ff0356287545ec1372f24b4b16f84",
        "actual": "a91b5001eeb670be23cbced0c03e52de498ff0356287545ec1372f24b4b16f84",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0062.json": {
        "expected": "be0594ba9905a53df73f4aef7f64c1fe34a17d203b46fe6f6fd2d8e4102fb4e0",
        "actual": "be0594ba9905a53df73f4aef7f64c1fe34a17d203b46fe6f6fd2d8e4102fb4e0",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0063.json": {
        "expected": "5ec0aa049735f1e5f3c780e225aa6d7d4a41cc3ef64159f46d06be55029b33b1",
        "actual": "5ec0aa049735f1e5f3c780e225aa6d7d4a41cc3ef64159f46d06be55029b33b1",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0064.json": {
        "expected": "70fabb20a069bf2a69cd8a3f5dc435100ea8bacc7b951e10273b294df6fcabf0",
        "actual": "70fabb20a069bf2a69cd8a3f5dc435100ea8bacc7b951e10273b294df6fcabf0",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0065.json": {
        "expected": "4993d47ac2bb49ac1eb9b8a2938e89a696d7cb0b4eadafdec82fabce8e71faf1",
        "actual": "4993d47ac2bb49ac1eb9b8a2938e89a696d7cb0b4eadafdec82fabce8e71faf1",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0066.json": {
        "expected": "bb0b2703fc555b2863024e1a1a60c83a13e09f30aed94f598d8454356cdbc10c",
        "actual": "bb0b2703fc555b2863024e1a1a60c83a13e09f30aed94f598d8454356cdbc10c",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0067.json": {
        "expected": "63abaa3f09971264d6c6618f34e38115db4f60e3642e406ac94e131c50570e3f",
        "actual": "63abaa3f09971264d6c6618f34e38115db4f60e3642e406ac94e131c50570e3f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0068.json": {
        "expected": "6a31124c934f88d9610d573b8c8e043a9e71f7f7dd5532fbd8efc7ac2bc555c0",
        "actual": "6a31124c934f88d9610d573b8c8e043a9e71f7f7dd5532fbd8efc7ac2bc555c0",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0069.json": {
        "expected": "4f4ccca778414cb7b017e4f0e56a70517df88a774641c9076237c4eb710b1e31",
        "actual": "4f4ccca778414cb7b017e4f0e56a70517df88a774641c9076237c4eb710b1e31",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0070.json": {
        "expected": "db70329e967b5da849f36e556401177c251b46d69f44e4ec48709fbcd81d77c9",
        "actual": "db70329e967b5da849f36e556401177c251b46d69f44e4ec48709fbcd81d77c9",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0071.json": {
        "expected": "204b61d7ba8718db7cf2029ffbb9d9954ec48ab69e95c6bd96488c97df7ef89f",
        "actual": "204b61d7ba8718db7cf2029ffbb9d9954ec48ab69e95c6bd96488c97df7ef89f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0072.json": {
        "expected": "a87a1b151e6833ac373a68c0f5c3d2b9ddcd2edef89dab38b245a602c3b48cfa",
        "actual": "a87a1b151e6833ac373a68c0f5c3d2b9ddcd2edef89dab38b245a602c3b48cfa",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0073.json": {
        "expected": "89e1211aa147faf3cea5df0fb5902406e05f178cb714c801b97677841cdb672c",
        "actual": "89e1211aa147faf3cea5df0fb5902406e05f178cb714c801b97677841cdb672c",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0074.json": {
        "expected": "a9f083b040bfe1574c7ca08c8add637a3ddd3cf98c50b18ee9ada4d95d0cef0a",
        "actual": "a9f083b040bfe1574c7ca08c8add637a3ddd3cf98c50b18ee9ada4d95d0cef0a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0075.json": {
        "expected": "88d1bcf531b17d2ab8a7ca01aac3984e1ed8e5c80b80c7056259d8a36e099976",
        "actual": "88d1bcf531b17d2ab8a7ca01aac3984e1ed8e5c80b80c7056259d8a36e099976",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0076.json": {
        "expected": "165304d815d6d3943a2b9824258561a3971fb2d5f023358f7320875c513d3917",
        "actual": "165304d815d6d3943a2b9824258561a3971fb2d5f023358f7320875c513d3917",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0077.json": {
        "expected": "7fcdf6d0e840d9dfcd444c5b2cad460ddf0d785cbc7553e2312e87b27dad8113",
        "actual": "7fcdf6d0e840d9dfcd444c5b2cad460ddf0d785cbc7553e2312e87b27dad8113",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0078.json": {
        "expected": "e28f7ad84a2c369da971c6bbf6211e27d93f8ee49b487167e9561a33d5a733b9",
        "actual": "e28f7ad84a2c369da971c6bbf6211e27d93f8ee49b487167e9561a33d5a733b9",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0079.json": {
        "expected": "e933ddd45fd6cc9cdfb437d7908385e2d17545d400cd8976e865b10fb1caa1f2",
        "actual": "e933ddd45fd6cc9cdfb437d7908385e2d17545d400cd8976e865b10fb1caa1f2",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0080.json": {
        "expected": "eedeeed236147669e9625b2280d7fe8cf57af0063ca94ea772619f8d347b7152",
        "actual": "eedeeed236147669e9625b2280d7fe8cf57af0063ca94ea772619f8d347b7152",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0081.json": {
        "expected": "cab41d03798564b40af6d9b5d821d6f6a45791ab7c67c0a76a673ddfbdf7658d",
        "actual": "cab41d03798564b40af6d9b5d821d6f6a45791ab7c67c0a76a673ddfbdf7658d",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0082.json": {
        "expected": "3a6aec16252426fab41ed70d373e842f9d74ca88b92c557d78885f84dc8ed8c8",
        "actual": "3a6aec16252426fab41ed70d373e842f9d74ca88b92c557d78885f84dc8ed8c8",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0083.json": {
        "expected": "dd9864e1b8f506ee7a3b6c23e4397b50760b62f98cde1bb55e3bf8473dc7c4e6",
        "actual": "dd9864e1b8f506ee7a3b6c23e4397b50760b62f98cde1bb55e3bf8473dc7c4e6",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0084.json": {
        "expected": "6aaf4ca9e3298a3b2b089c2c69adefaec57a8d84587adf8b5b4a22f1d98043cf",
        "actual": "6aaf4ca9e3298a3b2b089c2c69adefaec57a8d84587adf8b5b4a22f1d98043cf",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0085.json": {
        "expected": "49137d4778f916e790d8c28d2c3000fe2dc40c2761cd00cfc2990f6175a0e151",
        "actual": "49137d4778f916e790d8c28d2c3000fe2dc40c2761cd00cfc2990f6175a0e151",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0086.json": {
        "expected": "6dff76ba4eed63a05b1f1c81d39d40f57186d2e5c46b87e058f6b711dfa82b45",
        "actual": "6dff76ba4eed63a05b1f1c81d39d40f57186d2e5c46b87e058f6b711dfa82b45",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0087.json": {
        "expected": "ac58cd2d1c074424adac965dee2c32623beb7175aac7dfc03b9472b94931f7ff",
        "actual": "ac58cd2d1c074424adac965dee2c32623beb7175aac7dfc03b9472b94931f7ff",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0088.json": {
        "expected": "5b5582d6a5724a7ed1c3aa3b615367f5ea72b89e930a913f3e13216bf7eae61f",
        "actual": "5b5582d6a5724a7ed1c3aa3b615367f5ea72b89e930a913f3e13216bf7eae61f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0089.json": {
        "expected": "613b7647ef1216d30d41d609150f269dc7185d941364bfabc11e425a64eea239",
        "actual": "613b7647ef1216d30d41d609150f269dc7185d941364bfabc11e425a64eea239",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0090.json": {
        "expected": "79b52e20cab7254722d8c2067b9e5959c2b6b123009ba1642b82f6dcbe7d9656",
        "actual": "79b52e20cab7254722d8c2067b9e5959c2b6b123009ba1642b82f6dcbe7d9656",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0091.json": {
        "expected": "8a66b6006aa7f568c9adf2264fe9addfc6d6d099611e60d06a0aa09963181a71",
        "actual": "8a66b6006aa7f568c9adf2264fe9addfc6d6d099611e60d06a0aa09963181a71",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0092.json": {
        "expected": "f039ca5bd189fd9cccde6215685afbce4e067463f1ec79d9ea019151ecaa1880",
        "actual": "f039ca5bd189fd9cccde6215685afbce4e067463f1ec79d9ea019151ecaa1880",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0093.json": {
        "expected": "7f1b644668af41b6f76775856765dbb9742532978c3751add3493226c40086ad",
        "actual": "7f1b644668af41b6f76775856765dbb9742532978c3751add3493226c40086ad",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0094.json": {
        "expected": "40298cafc369ae70c6b7589732fc94425b968a15f06f7153a44fc30b1f70fb6f",
        "actual": "40298cafc369ae70c6b7589732fc94425b968a15f06f7153a44fc30b1f70fb6f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0095.json": {
        "expected": "f7b6f1462edd0e6597a34c45087dbe133944feb19bcdd6a7eee02eb2662a0a57",
        "actual": "f7b6f1462edd0e6597a34c45087dbe133944feb19bcdd6a7eee02eb2662a0a57",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0096.json": {
        "expected": "2761118a0b6c1ac732b4b0243aba2cde755c927a2d4eb49ac5782545b1da3211",
        "actual": "2761118a0b6c1ac732b4b0243aba2cde755c927a2d4eb49ac5782545b1da3211",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0097.json": {
        "expected": "94cbd31f9642ad5f31d5f8ba9061db7d34730acf886fd89c397c7c4776728f3f",
        "actual": "94cbd31f9642ad5f31d5f8ba9061db7d34730acf886fd89c397c7c4776728f3f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0098.json": {
        "expected": "515cb14a3d0dcfa2792ad01f872ba5a8d4868fbfffdafed7853319982cbf8f15",
        "actual": "515cb14a3d0dcfa2792ad01f872ba5a8d4868fbfffdafed7853319982cbf8f15",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0099.json": {
        "expected": "3a791ce3a9e6f0699bc83f010de5ad78da66a2f5cc467e33559c1f038915b72b",
        "actual": "3a791ce3a9e6f0699bc83f010de5ad78da66a2f5cc467e33559c1f038915b72b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0100.json": {
        "expected": "356415fbce8635065884eff09d6400b16fe81555be35a2a97f6121257b7aabd6",
        "actual": "356415fbce8635065884eff09d6400b16fe81555be35a2a97f6121257b7aabd6",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0101.json": {
        "expected": "d28306939945614251ea6be12da7806e0f3fd1f63e21b9089ad46f2006c2fa5e",
        "actual": "d28306939945614251ea6be12da7806e0f3fd1f63e21b9089ad46f2006c2fa5e",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0102.json": {
        "expected": "6ba30e1aab7049b6d08d3d55f21b13100935edd2cb91e9122931949dc3b22823",
        "actual": "6ba30e1aab7049b6d08d3d55f21b13100935edd2cb91e9122931949dc3b22823",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0103.json": {
        "expected": "952419501a7b3f83e9f81934eb99c7bef058725de666d7a7c83f7d313fc9dee1",
        "actual": "952419501a7b3f83e9f81934eb99c7bef058725de666d7a7c83f7d313fc9dee1",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0104.json": {
        "expected": "09773aafb9561aba7eb61818d9915e88f3462b2fcac42f5311c87992a2ae6c2a",
        "actual": "09773aafb9561aba7eb61818d9915e88f3462b2fcac42f5311c87992a2ae6c2a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0105.json": {
        "expected": "1fcc4e1822046f74178ff8c20d51274690fd266c355363a5e4d67b215ef9fad0",
        "actual": "1fcc4e1822046f74178ff8c20d51274690fd266c355363a5e4d67b215ef9fad0",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0106.json": {
        "expected": "43d690e46945e0b870a38bfedd98e8894e9fac0c05db3b0e9486a28759fc0b19",
        "actual": "43d690e46945e0b870a38bfedd98e8894e9fac0c05db3b0e9486a28759fc0b19",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0107.json": {
        "expected": "3a49fd6fb45b4be67392c4fb9e060c7476485ceddbe36a629bf2c1cc74f0b834",
        "actual": "3a49fd6fb45b4be67392c4fb9e060c7476485ceddbe36a629bf2c1cc74f0b834",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0108.json": {
        "expected": "c4eba2c8a151e50ccb73996b836403cf2c6e068b488decd6ca411f96bc96920b",
        "actual": "c4eba2c8a151e50ccb73996b836403cf2c6e068b488decd6ca411f96bc96920b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0109.json": {
        "expected": "2ca2036cf210742d71c839305a09e586f3b549d7b3aeb8a041714192394046d9",
        "actual": "2ca2036cf210742d71c839305a09e586f3b549d7b3aeb8a041714192394046d9",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0110.json": {
        "expected": "d397c6cfd7438a66f4b9c8a8b0cea3f950f36d05f3f9727fdf264344b89dfdb8",
        "actual": "d397c6cfd7438a66f4b9c8a8b0cea3f950f36d05f3f9727fdf264344b89dfdb8",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0111.json": {
        "expected": "b0b9c1480a008825d52bfdae02f5ebac02a96ba1df9411e38610d7afcc518101",
        "actual": "b0b9c1480a008825d52bfdae02f5ebac02a96ba1df9411e38610d7afcc518101",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0112.json": {
        "expected": "706bbd65a037f7303d569efbd6aa932bae8a897b80a3f5a5b91226ae30e3070c",
        "actual": "706bbd65a037f7303d569efbd6aa932bae8a897b80a3f5a5b91226ae30e3070c",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0113.json": {
        "expected": "954fc2fcd29b554934ced4f53a84164bfcae9fc87615f5e41ab667db84e8f71a",
        "actual": "954fc2fcd29b554934ced4f53a84164bfcae9fc87615f5e41ab667db84e8f71a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0114.json": {
        "expected": "d8ddb35923fe16e54db002d360676e0eb3e8f23a0ceec0a905b623c1d38a4e42",
        "actual": "d8ddb35923fe16e54db002d360676e0eb3e8f23a0ceec0a905b623c1d38a4e42",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0115.json": {
        "expected": "958b850e332e02699c0334cb6dfe88b2c890cc29514c825baeb42e157389fbf7",
        "actual": "958b850e332e02699c0334cb6dfe88b2c890cc29514c825baeb42e157389fbf7",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0116.json": {
        "expected": "29efc8312a0a4fd5eb1ec98da432436e4f0e5d15d353f36d4df74cc001d161ca",
        "actual": "29efc8312a0a4fd5eb1ec98da432436e4f0e5d15d353f36d4df74cc001d161ca",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0117.json": {
        "expected": "0cf934527a20d76ebfacbcc8c41e05972d972bb8d48afe0ae0519137d2f17db2",
        "actual": "0cf934527a20d76ebfacbcc8c41e05972d972bb8d48afe0ae0519137d2f17db2",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0118.json": {
        "expected": "dc890835f45153da02235ec448e108cbeefae651cea6940dd74ad3af177b6eec",
        "actual": "dc890835f45153da02235ec448e108cbeefae651cea6940dd74ad3af177b6eec",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0119.json": {
        "expected": "a8bc92070d182ab2154d3e093f1c6b40e0708bdc4f547602619a2f90e86cb40c",
        "actual": "a8bc92070d182ab2154d3e093f1c6b40e0708bdc4f547602619a2f90e86cb40c",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0120.json": {
        "expected": "ef11a5430491ceec096399305389e81ed6d5cd1958f7dd5759d06787bc16b2f6",
        "actual": "ef11a5430491ceec096399305389e81ed6d5cd1958f7dd5759d06787bc16b2f6",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0121.json": {
        "expected": "44a20058246fd8035c17488b4719e1b2561a5485f3237f8ca57c23adc7030d91",
        "actual": "44a20058246fd8035c17488b4719e1b2561a5485f3237f8ca57c23adc7030d91",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0122.json": {
        "expected": "cc8f7ccaad7cf3443ca6272ebae8d6edc07dd1fc8cb9030600ad227a3a95428d",
        "actual": "cc8f7ccaad7cf3443ca6272ebae8d6edc07dd1fc8cb9030600ad227a3a95428d",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0123.json": {
        "expected": "f2885530e64f9ef2d6012e7594a50cbef8543ab426e7dade94489f7e1df91e5a",
        "actual": "f2885530e64f9ef2d6012e7594a50cbef8543ab426e7dade94489f7e1df91e5a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0124.json": {
        "expected": "3bbbcbefbbcd73843526209f316d9c551a8417ef75dadcbd2434ca18bda51404",
        "actual": "3bbbcbefbbcd73843526209f316d9c551a8417ef75dadcbd2434ca18bda51404",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0125.json": {
        "expected": "3f4cbe2149d59389fee0a9db51f64bd2cd87ef3a0554f83998df9f379a030d9c",
        "actual": "3f4cbe2149d59389fee0a9db51f64bd2cd87ef3a0554f83998df9f379a030d9c",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0126.json": {
        "expected": "85982aac345c77a3a4add73355e17481603bbe0276c01913cf08fb718da97695",
        "actual": "85982aac345c77a3a4add73355e17481603bbe0276c01913cf08fb718da97695",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0127.json": {
        "expected": "562a4c00043f94901be2650257849a18ebfd99d5af444b4a4c6ad789c65e7d99",
        "actual": "562a4c00043f94901be2650257849a18ebfd99d5af444b4a4c6ad789c65e7d99",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0128.json": {
        "expected": "8c89fa9cb4f47ae6557147fdc6f3239105d08d1dd07512e833cac0dc2e893434",
        "actual": "8c89fa9cb4f47ae6557147fdc6f3239105d08d1dd07512e833cac0dc2e893434",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0129.json": {
        "expected": "441130065be6a4a9623a57be112be03bc2e11ab6d6303c9a75e0bcb8d8a83bf4",
        "actual": "441130065be6a4a9623a57be112be03bc2e11ab6d6303c9a75e0bcb8d8a83bf4",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0130.json": {
        "expected": "09c1682475e6f9254d94d71d12705f145fc7a23c6423c6da81c28b4dfcff0b3a",
        "actual": "09c1682475e6f9254d94d71d12705f145fc7a23c6423c6da81c28b4dfcff0b3a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0131.json": {
        "expected": "3f6635f390fe7b8129667786b1f3c1c967a2bd748c8d4f9802d77e958472a0e2",
        "actual": "3f6635f390fe7b8129667786b1f3c1c967a2bd748c8d4f9802d77e958472a0e2",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0132.json": {
        "expected": "aa82344f8a3bc113c14fda747a0af6376714116746a4b446f014118dce927b01",
        "actual": "aa82344f8a3bc113c14fda747a0af6376714116746a4b446f014118dce927b01",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0133.json": {
        "expected": "fe54c622783f679e1874705df46f6f28e8e98410123a826f30cb4cd3eb05c040",
        "actual": "fe54c622783f679e1874705df46f6f28e8e98410123a826f30cb4cd3eb05c040",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0134.json": {
        "expected": "e4e577933f7b022be5191f1f1422b529a90f2556c27c4a29b4aa6094715ca508",
        "actual": "e4e577933f7b022be5191f1f1422b529a90f2556c27c4a29b4aa6094715ca508",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0135.json": {
        "expected": "e7bde14f179acc545d818d047ba56bad9db0e42cec06735edde9fd27d561d4d6",
        "actual": "e7bde14f179acc545d818d047ba56bad9db0e42cec06735edde9fd27d561d4d6",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0136.json": {
        "expected": "43b94741c7919d117fe24fc79c9d58036a39603a387771e8a02686361ec1ea48",
        "actual": "43b94741c7919d117fe24fc79c9d58036a39603a387771e8a02686361ec1ea48",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0137.json": {
        "expected": "5807635fa9bfd0c29e65bcb330819d664dc8ab4cf3405a354582f7429e71686c",
        "actual": "5807635fa9bfd0c29e65bcb330819d664dc8ab4cf3405a354582f7429e71686c",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0138.json": {
        "expected": "4a08e41a28e62fd4e5f0a83b1fe189a24621ee5c8c760013374751161c42c77b",
        "actual": "4a08e41a28e62fd4e5f0a83b1fe189a24621ee5c8c760013374751161c42c77b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0139.json": {
        "expected": "83257ca4abbfc27436f92e52700fd3121d60b3b04b323e1b28a337dac3ee6890",
        "actual": "83257ca4abbfc27436f92e52700fd3121d60b3b04b323e1b28a337dac3ee6890",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0140.json": {
        "expected": "eac8c0be03313f2c784c9888ca7a3666fb262ffad39b733cd45bec7d991de492",
        "actual": "eac8c0be03313f2c784c9888ca7a3666fb262ffad39b733cd45bec7d991de492",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0141.json": {
        "expected": "4d6e53d52782b310c5e7d75984b2251884bc8747869e34dfc9fab96ce3b98ebf",
        "actual": "4d6e53d52782b310c5e7d75984b2251884bc8747869e34dfc9fab96ce3b98ebf",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0142.json": {
        "expected": "bce0af8edc9367b1d63fcb8342e5b04d551f7c93134de17cdcfb04668dd8d9a8",
        "actual": "bce0af8edc9367b1d63fcb8342e5b04d551f7c93134de17cdcfb04668dd8d9a8",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0143.json": {
        "expected": "9ceeab9d4a8f2003c80b60751f7f3f2f52274c14cc701cb90e94e116067ed84b",
        "actual": "9ceeab9d4a8f2003c80b60751f7f3f2f52274c14cc701cb90e94e116067ed84b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0144.json": {
        "expected": "9773db5b7e5391441510ac4cd9532efcb11e8d1d86228ed3f7210ec63be9b8bc",
        "actual": "9773db5b7e5391441510ac4cd9532efcb11e8d1d86228ed3f7210ec63be9b8bc",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0145.json": {
        "expected": "5a35fcb9377e60c39e72309d9bf4f93325a1d7c95b4f772996ba9dbc332f8ac1",
        "actual": "5a35fcb9377e60c39e72309d9bf4f93325a1d7c95b4f772996ba9dbc332f8ac1",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0146.json": {
        "expected": "45b872d79ed14e9185d10640be8123a5b84b49fadd53049ca1e2ed9913a4ea59",
        "actual": "45b872d79ed14e9185d10640be8123a5b84b49fadd53049ca1e2ed9913a4ea59",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0147.json": {
        "expected": "fa8c4e0000208ed7e35657439353d20636f7b22fe0519811d5cb904523063286",
        "actual": "fa8c4e0000208ed7e35657439353d20636f7b22fe0519811d5cb904523063286",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0148.json": {
        "expected": "9685853abcbb98cfd1c8030f769174110ff3e838ef70c2e75f8a2e9777120896",
        "actual": "9685853abcbb98cfd1c8030f769174110ff3e838ef70c2e75f8a2e9777120896",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0149.json": {
        "expected": "7ef347aecae50fcfe43e649d73eb46ea6ac2c6f4c0c6f04d3087b655b7fa17c6",
        "actual": "7ef347aecae50fcfe43e649d73eb46ea6ac2c6f4c0c6f04d3087b655b7fa17c6",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0150.json": {
        "expected": "e1d2001515aae4c77cc5e85abd1212a1a47681c2dc0a77ba9022a7b468d99423",
        "actual": "e1d2001515aae4c77cc5e85abd1212a1a47681c2dc0a77ba9022a7b468d99423",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0151.json": {
        "expected": "10a398f2c8629b7df89bb52ad4500f07e40b5bc6fcddab8b8371cb7339695ca6",
        "actual": "10a398f2c8629b7df89bb52ad4500f07e40b5bc6fcddab8b8371cb7339695ca6",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0152.json": {
        "expected": "8fc5b8c4417968ab0cc7eb23968ba4db25b9acae4e1679b8fe5a2b6b72dd8368",
        "actual": "8fc5b8c4417968ab0cc7eb23968ba4db25b9acae4e1679b8fe5a2b6b72dd8368",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0153.json": {
        "expected": "f9e52e2b4b8b910b8dd1b3955c7acbcc09f8d562db2021c1bb30eab58ec1e81d",
        "actual": "f9e52e2b4b8b910b8dd1b3955c7acbcc09f8d562db2021c1bb30eab58ec1e81d",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0154.json": {
        "expected": "ef07c56ba33b615fb3316a32d74cc6582ea640cd176becfe49c53a2a4c902a83",
        "actual": "ef07c56ba33b615fb3316a32d74cc6582ea640cd176becfe49c53a2a4c902a83",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0155.json": {
        "expected": "5e4e99ceec44f1da95c0a2c0c37dd18c96a1db8b0632352eb367d3b6a2b08dce",
        "actual": "5e4e99ceec44f1da95c0a2c0c37dd18c96a1db8b0632352eb367d3b6a2b08dce",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0156.json": {
        "expected": "c962c252f90dea3293c3146d2a65fda5e33aa39f2ae193f67112eff93978ef70",
        "actual": "c962c252f90dea3293c3146d2a65fda5e33aa39f2ae193f67112eff93978ef70",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0157.json": {
        "expected": "98fefaf2c4e9e4b915f6eaa5b2ceca89c464b9eaa3ee10e3f32910878a809d19",
        "actual": "98fefaf2c4e9e4b915f6eaa5b2ceca89c464b9eaa3ee10e3f32910878a809d19",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0158.json": {
        "expected": "c14b909eaa206e0192e05c9c7ae0325aa599dbcfb79f8605ab1f088f4ed8b3ef",
        "actual": "c14b909eaa206e0192e05c9c7ae0325aa599dbcfb79f8605ab1f088f4ed8b3ef",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0159.json": {
        "expected": "dc4b3d4d9cf71b79f8218ded01ace23528eb6059093b2a3687ff43423d4c47d9",
        "actual": "dc4b3d4d9cf71b79f8218ded01ace23528eb6059093b2a3687ff43423d4c47d9",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0160.json": {
        "expected": "4084afa2d7d6fdc4b1b9bd94f90b33f28b0cb9583e204ae459d429d6364eb50e",
        "actual": "4084afa2d7d6fdc4b1b9bd94f90b33f28b0cb9583e204ae459d429d6364eb50e",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0161.json": {
        "expected": "7b2d3aa1d40917f4d9a6bf7fbe5f481bf2efbd21ee72edf8951854622eaefb87",
        "actual": "7b2d3aa1d40917f4d9a6bf7fbe5f481bf2efbd21ee72edf8951854622eaefb87",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0162.json": {
        "expected": "6ead713d791839f49d9db8641ec89af0403a64f9b0c68bb6c42a46e9ae0f09ac",
        "actual": "6ead713d791839f49d9db8641ec89af0403a64f9b0c68bb6c42a46e9ae0f09ac",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0163.json": {
        "expected": "661d275ee3835fb54fb0aa6c5c3a11ba71f768c7066ee90c221f486e52266038",
        "actual": "661d275ee3835fb54fb0aa6c5c3a11ba71f768c7066ee90c221f486e52266038",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0164.json": {
        "expected": "7659aa8f664f16a350050b39c316f56d5be6f2d9160683f682ab048af9768415",
        "actual": "7659aa8f664f16a350050b39c316f56d5be6f2d9160683f682ab048af9768415",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0165.json": {
        "expected": "96260a59a9069de0e65b04d83996070dc941c11ffa83f3b7e5690f3cb1205084",
        "actual": "96260a59a9069de0e65b04d83996070dc941c11ffa83f3b7e5690f3cb1205084",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0166.json": {
        "expected": "48462299b2f2aceb151f8d67ce0906623344bb6da4c02c9f36092bdaacc10659",
        "actual": "48462299b2f2aceb151f8d67ce0906623344bb6da4c02c9f36092bdaacc10659",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0167.json": {
        "expected": "e9429a91521a80c3a3f94a14a64bb1ebc34de2793217b61f56a9cffb1bfa1e6d",
        "actual": "e9429a91521a80c3a3f94a14a64bb1ebc34de2793217b61f56a9cffb1bfa1e6d",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0168.json": {
        "expected": "e8ef4ec12ba28b561441b5913ec43162609bec71269f7735f95dbbd394ea164c",
        "actual": "e8ef4ec12ba28b561441b5913ec43162609bec71269f7735f95dbbd394ea164c",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0169.json": {
        "expected": "c06c72e36e1f8273585a15168275f075d6ebb95f5f06945d6ed41e0af95a96e1",
        "actual": "c06c72e36e1f8273585a15168275f075d6ebb95f5f06945d6ed41e0af95a96e1",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0170.json": {
        "expected": "315600e434c33dfd293f03072133ac5e9223bf0225bba5f5e9cbf92564efbc46",
        "actual": "315600e434c33dfd293f03072133ac5e9223bf0225bba5f5e9cbf92564efbc46",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0171.json": {
        "expected": "2b77ce170acd671dd98e68eea6cc2a0f3fd048fe7052bc7cbe578ec0fd743213",
        "actual": "2b77ce170acd671dd98e68eea6cc2a0f3fd048fe7052bc7cbe578ec0fd743213",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0172.json": {
        "expected": "af8e64bb8bf8cb86700e7b7e137ee0131a908942b304f4ec0751d014d620f12a",
        "actual": "af8e64bb8bf8cb86700e7b7e137ee0131a908942b304f4ec0751d014d620f12a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0173.json": {
        "expected": "7b24a129d32db8f9666bd4d7a8b657964c184a7093623ae2e555fca9a26a87c2",
        "actual": "7b24a129d32db8f9666bd4d7a8b657964c184a7093623ae2e555fca9a26a87c2",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0174.json": {
        "expected": "2427a9f7ee5a9655167a7ae1b94634a73e8004f2e8efde58c5780e62966603e7",
        "actual": "2427a9f7ee5a9655167a7ae1b94634a73e8004f2e8efde58c5780e62966603e7",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0175.json": {
        "expected": "86cb5bf066966ef6848ed9d8244b97b7bcac1bb608567b7d6366810f1d903f29",
        "actual": "86cb5bf066966ef6848ed9d8244b97b7bcac1bb608567b7d6366810f1d903f29",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0176.json": {
        "expected": "b211d2ca74d99720cc133f4f9c56086c4a43856d4f129374aa7824b57e506ec3",
        "actual": "b211d2ca74d99720cc133f4f9c56086c4a43856d4f129374aa7824b57e506ec3",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0177.json": {
        "expected": "6e34fca79c8d312a64e48525cdb80cdb939fc7fd9de2c53d312f407230c3ad8f",
        "actual": "6e34fca79c8d312a64e48525cdb80cdb939fc7fd9de2c53d312f407230c3ad8f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0178.json": {
        "expected": "f624f9b98ce38f5af04a1f6dc3bfca54cd58c029443b3d99b010ca419b0f4f03",
        "actual": "f624f9b98ce38f5af04a1f6dc3bfca54cd58c029443b3d99b010ca419b0f4f03",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0179.json": {
        "expected": "8991d38c96514a5b010b4e1456aa3b672260c05c4f4f8771aafb2f193bf3a133",
        "actual": "8991d38c96514a5b010b4e1456aa3b672260c05c4f4f8771aafb2f193bf3a133",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0180.json": {
        "expected": "80d610b0046041808163cc000125deaafb3bfec4d2e1f343d41d6a8f79054b78",
        "actual": "80d610b0046041808163cc000125deaafb3bfec4d2e1f343d41d6a8f79054b78",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0181.json": {
        "expected": "05cd2f4be413e4cf9c261d4fe738b120bd032e4aa2b8a807c3aaa394565c56d3",
        "actual": "05cd2f4be413e4cf9c261d4fe738b120bd032e4aa2b8a807c3aaa394565c56d3",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0182.json": {
        "expected": "5f178425be2948002ea295d9ace60d477b24ef4449d7ac122e29a4df20170ef1",
        "actual": "5f178425be2948002ea295d9ace60d477b24ef4449d7ac122e29a4df20170ef1",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0183.json": {
        "expected": "3cd690fc659e08a673461a6bc491013a867bbad9c62eb782a651364a208a7cce",
        "actual": "3cd690fc659e08a673461a6bc491013a867bbad9c62eb782a651364a208a7cce",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0184.json": {
        "expected": "00386eece39a932fbbadbc1e21d1f6443d82e3f8bc48b680ca3c2bc593476c93",
        "actual": "00386eece39a932fbbadbc1e21d1f6443d82e3f8bc48b680ca3c2bc593476c93",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0185.json": {
        "expected": "5bdf199eaae330dca33488d27f835a7987f88348594dc86d332dae46cef82e15",
        "actual": "5bdf199eaae330dca33488d27f835a7987f88348594dc86d332dae46cef82e15",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0186.json": {
        "expected": "8027aa155d01d7a8babeb25f84ad2a23e2fafb5670fa007a0e29c7b675cf717e",
        "actual": "8027aa155d01d7a8babeb25f84ad2a23e2fafb5670fa007a0e29c7b675cf717e",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0187.json": {
        "expected": "be4369829d2a1c996406e81093dd40eb91cd6360262b4f16d511e60a4ed59994",
        "actual": "be4369829d2a1c996406e81093dd40eb91cd6360262b4f16d511e60a4ed59994",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0188.json": {
        "expected": "c3b6003472f878319053066e2444ba665162cbd81f31060e4b4cdeae4010fc80",
        "actual": "c3b6003472f878319053066e2444ba665162cbd81f31060e4b4cdeae4010fc80",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0189.json": {
        "expected": "379397ef6fd230dc54228f062876cfabeba9acfbf4becc6d276e6cf1fed2fced",
        "actual": "379397ef6fd230dc54228f062876cfabeba9acfbf4becc6d276e6cf1fed2fced",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0190.json": {
        "expected": "f0dcbaa3015981177022b6de90a638a92c2f65051623f345aee5777c97b1549c",
        "actual": "f0dcbaa3015981177022b6de90a638a92c2f65051623f345aee5777c97b1549c",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0191.json": {
        "expected": "202343e3eb357017dc9947d7d8209325a840635482de485ea08d5b0e803f17ce",
        "actual": "202343e3eb357017dc9947d7d8209325a840635482de485ea08d5b0e803f17ce",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0192.json": {
        "expected": "128fe7cee04dcf607990c4089481dea737a80b9a4729e19ae3460b2d948aa990",
        "actual": "128fe7cee04dcf607990c4089481dea737a80b9a4729e19ae3460b2d948aa990",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0193.json": {
        "expected": "fd95e9eb04749e966d62f0689bd5e283d65e7538e6759a4a510090389c88a3b3",
        "actual": "fd95e9eb04749e966d62f0689bd5e283d65e7538e6759a4a510090389c88a3b3",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0194.json": {
        "expected": "826d407e0738f29e3ee969bad0454b5a1bc10004faf9a0dcd226c7569447c597",
        "actual": "826d407e0738f29e3ee969bad0454b5a1bc10004faf9a0dcd226c7569447c597",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0195.json": {
        "expected": "3ef830bc22874ce30e8464f20173a583c9148004e1309cdad5e9d55f3360227b",
        "actual": "3ef830bc22874ce30e8464f20173a583c9148004e1309cdad5e9d55f3360227b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0196.json": {
        "expected": "4eb5f5669a1cac6bc2c0f84a068f6071832164ee16b9ff7d6536d8673c30fc34",
        "actual": "4eb5f5669a1cac6bc2c0f84a068f6071832164ee16b9ff7d6536d8673c30fc34",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0197.json": {
        "expected": "49331be1734f8b87526e6d69cabbd1de07de1fac33e7636e1ae9b32bc213b1d5",
        "actual": "49331be1734f8b87526e6d69cabbd1de07de1fac33e7636e1ae9b32bc213b1d5",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0198.json": {
        "expected": "82b1381803c2e9a9ae9768d7d9e4fe35a1b76d3fa4c98f49ef7f54dfd54c2687",
        "actual": "82b1381803c2e9a9ae9768d7d9e4fe35a1b76d3fa4c98f49ef7f54dfd54c2687",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0199.json": {
        "expected": "dcad3397c87a09f58f283aef11b2fc4e9790bfcb3352b8d93eceb24fc8384c0e",
        "actual": "dcad3397c87a09f58f283aef11b2fc4e9790bfcb3352b8d93eceb24fc8384c0e",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage2\\runs\\sacdet\\episodes\\episode-0200.json": {
        "expected": "45d35b820c080b5e2b12a25113eb139c6740f78bfc1c972926688f0b42d3a6f5",
        "actual": "45d35b820c080b5e2b12a25113eb139c6740f78bfc1c972926688f0b42d3a6f5",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0414.json": {
        "expected": "49d082949d8b2f4659bb51b9fa108b0341a3faf545871b4de2a3fd4276a1d4d7",
        "actual": "49d082949d8b2f4659bb51b9fa108b0341a3faf545871b4de2a3fd4276a1d4d7",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0415.json": {
        "expected": "bb09290f9b5ea2b72843bc5e3e98418301c8674d062782b19d991d13a4a0f812",
        "actual": "bb09290f9b5ea2b72843bc5e3e98418301c8674d062782b19d991d13a4a0f812",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0416.json": {
        "expected": "186671ce2ded675647cfcdbcbc9bd5cedabcde63170efb06aa1b132db5ebc523",
        "actual": "186671ce2ded675647cfcdbcbc9bd5cedabcde63170efb06aa1b132db5ebc523",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0417.json": {
        "expected": "6e8f17e62c3396b05e6a7ff22e9a7ab617ee9aa37ccb138917f100e48727e497",
        "actual": "6e8f17e62c3396b05e6a7ff22e9a7ab617ee9aa37ccb138917f100e48727e497",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0418.json": {
        "expected": "ea3e227fc5c4a0e482b2e3583cac76af9eb28be7b29d203d6b244a0b68969f2e",
        "actual": "ea3e227fc5c4a0e482b2e3583cac76af9eb28be7b29d203d6b244a0b68969f2e",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0419.json": {
        "expected": "03a38056bedd921136de861931b105919467b2ace40f7f8570e6edb7f062a6f8",
        "actual": "03a38056bedd921136de861931b105919467b2ace40f7f8570e6edb7f062a6f8",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0420.json": {
        "expected": "37a1c0a9684e8c7ed4ab986add2cee953cf2fb0fb713015656da924cf51cb66d",
        "actual": "37a1c0a9684e8c7ed4ab986add2cee953cf2fb0fb713015656da924cf51cb66d",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0431.json": {
        "expected": "b7bb6d57e7bab38d9587dfb39ed48d17c509a81c1cacf538e75c69e89ad54a59",
        "actual": "b7bb6d57e7bab38d9587dfb39ed48d17c509a81c1cacf538e75c69e89ad54a59",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0432.json": {
        "expected": "cdaf1893f4f0a442636a16579c3fc4112973c8bc29abbf9f7f91e634b1a6e6c3",
        "actual": "cdaf1893f4f0a442636a16579c3fc4112973c8bc29abbf9f7f91e634b1a6e6c3",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0433.json": {
        "expected": "5f45bf701b0e390a390607ce57cba3c58e06031110fdff09b9e55786170df427",
        "actual": "5f45bf701b0e390a390607ce57cba3c58e06031110fdff09b9e55786170df427",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0434.json": {
        "expected": "2d11f6a488d4577cd5158a87bf86c06013c64b558f7132ed55c34a48f3aeca34",
        "actual": "2d11f6a488d4577cd5158a87bf86c06013c64b558f7132ed55c34a48f3aeca34",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0435.json": {
        "expected": "b585b4a5f45517800a62aaa7d7b13ecaef3807fcea8c0595a2c1e096f0bea560",
        "actual": "b585b4a5f45517800a62aaa7d7b13ecaef3807fcea8c0595a2c1e096f0bea560",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0436.json": {
        "expected": "f7464f4e0e8d3e33733fc338dccbdef4d9f7d87bd1f305e4827b2fb9fb56269c",
        "actual": "f7464f4e0e8d3e33733fc338dccbdef4d9f7d87bd1f305e4827b2fb9fb56269c",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0437.json": {
        "expected": "9d323385f3f2b0449955bd0fc6920cc47a2937a814b82a3eaa908682a71b7eb3",
        "actual": "9d323385f3f2b0449955bd0fc6920cc47a2937a814b82a3eaa908682a71b7eb3",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0438.json": {
        "expected": "80d2237eea249aed7b1b64b1590424efba98d917373f560d9af8e6bb09b3a9de",
        "actual": "80d2237eea249aed7b1b64b1590424efba98d917373f560d9af8e6bb09b3a9de",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0439.json": {
        "expected": "f4b35426c811fb5576100a83dbf69112eeb45956ff3c92fa04c7194f5d96b548",
        "actual": "f4b35426c811fb5576100a83dbf69112eeb45956ff3c92fa04c7194f5d96b548",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0440.json": {
        "expected": "48feb23532e27a44d5c3a8f25995a33db7bae4e412b90f44bcc407e3798383e4",
        "actual": "48feb23532e27a44d5c3a8f25995a33db7bae4e412b90f44bcc407e3798383e4",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0451.json": {
        "expected": "72f0a31378741a13fb1c414f8e6dfb7b7055b68281b242b96cbaff175c7de86a",
        "actual": "72f0a31378741a13fb1c414f8e6dfb7b7055b68281b242b96cbaff175c7de86a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0452.json": {
        "expected": "6a9968da479fe947d314027ffd0c21a780f6c95d7f1948ffd84b9c6d648c9007",
        "actual": "6a9968da479fe947d314027ffd0c21a780f6c95d7f1948ffd84b9c6d648c9007",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0453.json": {
        "expected": "d904f800331967df5574c8a8822b680dc9a33f79a849d878be20b1b68e7536c3",
        "actual": "d904f800331967df5574c8a8822b680dc9a33f79a849d878be20b1b68e7536c3",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0454.json": {
        "expected": "f910ac80ac1c87173e749a37f36ef7f88f35b3396787d0f6b1429266321b1fe8",
        "actual": "f910ac80ac1c87173e749a37f36ef7f88f35b3396787d0f6b1429266321b1fe8",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0455.json": {
        "expected": "1beb953da34ce79fbd03fc3b279acf25a57ffdd528be4ee3d54598217bdb1cf6",
        "actual": "1beb953da34ce79fbd03fc3b279acf25a57ffdd528be4ee3d54598217bdb1cf6",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0456.json": {
        "expected": "d3cc59292572879853718c61337ce569622c9973d3c02cbf29b0c80c10779580",
        "actual": "d3cc59292572879853718c61337ce569622c9973d3c02cbf29b0c80c10779580",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0457.json": {
        "expected": "2ff24d2571c2bf57023bda679d728c0698f680156eb4d19eb10266e5953f2802",
        "actual": "2ff24d2571c2bf57023bda679d728c0698f680156eb4d19eb10266e5953f2802",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0458.json": {
        "expected": "e7248245a09ead402414f64a4220a5d1e5e97bbe556014ac862806fe0a5c75a4",
        "actual": "e7248245a09ead402414f64a4220a5d1e5e97bbe556014ac862806fe0a5c75a4",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0459.json": {
        "expected": "02cb35acf6b2e84a3fc9e6644085dff2e76d69cf42bdf8c530752a3d479200ee",
        "actual": "02cb35acf6b2e84a3fc9e6644085dff2e76d69cf42bdf8c530752a3d479200ee",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0460.json": {
        "expected": "15ce6c66cbc387a97de65c7dadfdbaac56f3ebe58ff46089d34782d47900809d",
        "actual": "15ce6c66cbc387a97de65c7dadfdbaac56f3ebe58ff46089d34782d47900809d",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0471.json": {
        "expected": "d47f30a688d7aac2ca577a3b173efa2eb1b97b0745303b5b07bd8a945ee562f1",
        "actual": "d47f30a688d7aac2ca577a3b173efa2eb1b97b0745303b5b07bd8a945ee562f1",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0472.json": {
        "expected": "42e4bd9aebe5d6ae99a8002050f413a95125774814f475fbd2c2534fdaababbf",
        "actual": "42e4bd9aebe5d6ae99a8002050f413a95125774814f475fbd2c2534fdaababbf",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0473.json": {
        "expected": "622740563fa02481d866c787a3f51ad47fdacc3740b69438756df3695bd7aa5f",
        "actual": "622740563fa02481d866c787a3f51ad47fdacc3740b69438756df3695bd7aa5f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0474.json": {
        "expected": "82b448d2a79a13b07abff01098d738a40b6856447c123437d2bdf292f8114474",
        "actual": "82b448d2a79a13b07abff01098d738a40b6856447c123437d2bdf292f8114474",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0475.json": {
        "expected": "b21ab283585a0329ceba21f87555f6c10f62b28c3fedca9a6d5b1fe4e74df673",
        "actual": "b21ab283585a0329ceba21f87555f6c10f62b28c3fedca9a6d5b1fe4e74df673",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0476.json": {
        "expected": "5c2d551a7dcf329f7cf6c1fc590c62c797ed369908bb211de163898e2b283268",
        "actual": "5c2d551a7dcf329f7cf6c1fc590c62c797ed369908bb211de163898e2b283268",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0477.json": {
        "expected": "1006527634d66ca59aee858022fcd6b6307c2247560d0a71633b0f11109af6bb",
        "actual": "1006527634d66ca59aee858022fcd6b6307c2247560d0a71633b0f11109af6bb",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0478.json": {
        "expected": "b69342226763df2cba490766269f5e11bcb461ce4a77ca9df31526d0d04bc6f0",
        "actual": "b69342226763df2cba490766269f5e11bcb461ce4a77ca9df31526d0d04bc6f0",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0479.json": {
        "expected": "0195a39e3f1a19831c1ea7d21f943014306199d28a3bbaa72af2c714019e9814",
        "actual": "0195a39e3f1a19831c1ea7d21f943014306199d28a3bbaa72af2c714019e9814",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0480.json": {
        "expected": "dde47659ff7cc1b389d1a2fe54d7178593a23b25145d2c3e3db0b78176863ac2",
        "actual": "dde47659ff7cc1b389d1a2fe54d7178593a23b25145d2c3e3db0b78176863ac2",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0491.json": {
        "expected": "08b6ef03cb2c06085e50c04357532c4e4276c5a7a29f70a5169a8e3e897185df",
        "actual": "08b6ef03cb2c06085e50c04357532c4e4276c5a7a29f70a5169a8e3e897185df",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0492.json": {
        "expected": "7f594e35b95816824eb449d77f9e854614184631decd9727a76e32dc8338387b",
        "actual": "7f594e35b95816824eb449d77f9e854614184631decd9727a76e32dc8338387b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0493.json": {
        "expected": "472e854dae397602140d57e4133b2b49032e9caad2cbef276ac7b1dc830fcb03",
        "actual": "472e854dae397602140d57e4133b2b49032e9caad2cbef276ac7b1dc830fcb03",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0494.json": {
        "expected": "3d029da239f2d84139658dd607df7bc2553a8e4bca747372821b071d29e1c5ea",
        "actual": "3d029da239f2d84139658dd607df7bc2553a8e4bca747372821b071d29e1c5ea",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0495.json": {
        "expected": "6e168209ad910164c4c63840c7fd88dbe83e8364cee1ca06428755e06a0027d2",
        "actual": "6e168209ad910164c4c63840c7fd88dbe83e8364cee1ca06428755e06a0027d2",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0496.json": {
        "expected": "e2429180df3df62203f2dd6e570e02986f1969ba581e3e6ce2c0eaf526098d7c",
        "actual": "e2429180df3df62203f2dd6e570e02986f1969ba581e3e6ce2c0eaf526098d7c",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0497.json": {
        "expected": "249e57f5a103b6e28187f14856edbe397225cb8980269f8748970381f8544f13",
        "actual": "249e57f5a103b6e28187f14856edbe397225cb8980269f8748970381f8544f13",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0498.json": {
        "expected": "0ae08990ce41d5581d379293b82cfafbd1dd4b10d3e0bad0840d66d4747fe495",
        "actual": "0ae08990ce41d5581d379293b82cfafbd1dd4b10d3e0bad0840d66d4747fe495",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0499.json": {
        "expected": "9544a6a58a9eb89cdfc05ebf4e5d58f1b830d678be506803042cb3b57fc76993",
        "actual": "9544a6a58a9eb89cdfc05ebf4e5d58f1b830d678be506803042cb3b57fc76993",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0500.json": {
        "expected": "c80bdf134602353fa64a332ab81c73e2b8d700ece10475cfcb3c789be6057719",
        "actual": "c80bdf134602353fa64a332ab81c73e2b8d700ece10475cfcb3c789be6057719",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0511.json": {
        "expected": "b7fdc78674649868d1b6035347422db8803535bbc50701eda8ae02971758cd16",
        "actual": "b7fdc78674649868d1b6035347422db8803535bbc50701eda8ae02971758cd16",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0512.json": {
        "expected": "a0cee3db8f17a2f64457b4935578bdf14aed55e71f3b257226933d0ba1b28242",
        "actual": "a0cee3db8f17a2f64457b4935578bdf14aed55e71f3b257226933d0ba1b28242",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0513.json": {
        "expected": "92838de1e770d1518ec22b1cfd0f54f2e324549e041ad536e861b1d8798e68f3",
        "actual": "92838de1e770d1518ec22b1cfd0f54f2e324549e041ad536e861b1d8798e68f3",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0514.json": {
        "expected": "ee319d7a0a8d202b6726f113b28baaa9f5d649ac9d9c3a417984b41d74e28e80",
        "actual": "ee319d7a0a8d202b6726f113b28baaa9f5d649ac9d9c3a417984b41d74e28e80",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0515.json": {
        "expected": "38f43adc0d4b2b25e58bda0c00a4788c599424a9e5f5d76182a4f0f3292b9418",
        "actual": "38f43adc0d4b2b25e58bda0c00a4788c599424a9e5f5d76182a4f0f3292b9418",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0516.json": {
        "expected": "23f631a36707f0ff32306fa0d63e72d25c6b2dd22df5a819e87b6d794aba6345",
        "actual": "23f631a36707f0ff32306fa0d63e72d25c6b2dd22df5a819e87b6d794aba6345",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0517.json": {
        "expected": "46edeb8a7bc7d361807d7e7078eb012eb9945562b16ad093b6368a043376ccb8",
        "actual": "46edeb8a7bc7d361807d7e7078eb012eb9945562b16ad093b6368a043376ccb8",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0518.json": {
        "expected": "a572ea4846e4e59f48d6327c80aab17f6a0da7ba8fb9ee269f4dd6a2f920ef3d",
        "actual": "a572ea4846e4e59f48d6327c80aab17f6a0da7ba8fb9ee269f4dd6a2f920ef3d",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0519.json": {
        "expected": "a8691b8a671e621c8365df48d0a9813b11728c20fd17c86815594a4748af3804",
        "actual": "a8691b8a671e621c8365df48d0a9813b11728c20fd17c86815594a4748af3804",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0520.json": {
        "expected": "719871c6cc401c56bd245e5c62092e1b13d3fc73bf755484124683969baf4e6b",
        "actual": "719871c6cc401c56bd245e5c62092e1b13d3fc73bf755484124683969baf4e6b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0531.json": {
        "expected": "de117246a330d984c43ae58219949a2dc471908d2bb9a811d709c414d1fd6908",
        "actual": "de117246a330d984c43ae58219949a2dc471908d2bb9a811d709c414d1fd6908",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0532.json": {
        "expected": "8f89c599e4de684e04909253dc77c1be8502c8f21a9d24b348113d386062b05c",
        "actual": "8f89c599e4de684e04909253dc77c1be8502c8f21a9d24b348113d386062b05c",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0533.json": {
        "expected": "aa72a14860761cb553e48c6d8866c4a1cbdb9b43b2a967537c6b33fd501722d5",
        "actual": "aa72a14860761cb553e48c6d8866c4a1cbdb9b43b2a967537c6b33fd501722d5",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0534.json": {
        "expected": "685f1335a42e1f0b46b637c6a79d5a280ab33cdcb01950bb4c316bd6a59b7a3e",
        "actual": "685f1335a42e1f0b46b637c6a79d5a280ab33cdcb01950bb4c316bd6a59b7a3e",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0535.json": {
        "expected": "a77ed5ed40ee0f9069bf0d701ef29704063cdfa0a12558af7f392b3ce8b8ef2a",
        "actual": "a77ed5ed40ee0f9069bf0d701ef29704063cdfa0a12558af7f392b3ce8b8ef2a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0536.json": {
        "expected": "9288ae6845b8fb6171e2a4f84d62ea6730779c5fc3f5f8ce30b06dd4323f0bfd",
        "actual": "9288ae6845b8fb6171e2a4f84d62ea6730779c5fc3f5f8ce30b06dd4323f0bfd",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0537.json": {
        "expected": "8e37e0b340f64430f7f4425bcadae53dddbd376417b305cf2cf94fc684a80c50",
        "actual": "8e37e0b340f64430f7f4425bcadae53dddbd376417b305cf2cf94fc684a80c50",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0538.json": {
        "expected": "1e9729d8ebcb8b0db8875e21808ce75781434d2248d21a4e0d20918a7edd2beb",
        "actual": "1e9729d8ebcb8b0db8875e21808ce75781434d2248d21a4e0d20918a7edd2beb",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0539.json": {
        "expected": "c8905715a2a0ea8171588773753917dce389d09a4245b04d0b701aa7fa8bb099",
        "actual": "c8905715a2a0ea8171588773753917dce389d09a4245b04d0b701aa7fa8bb099",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0540.json": {
        "expected": "8f54cc23ec1289a6c2b265e5a4a0d08c8363cbd51b9564babed27708a102035a",
        "actual": "8f54cc23ec1289a6c2b265e5a4a0d08c8363cbd51b9564babed27708a102035a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0551.json": {
        "expected": "eadf5080f795f44e024672664a585a9304bc4f324ddd8b37e7f67914986dce5a",
        "actual": "eadf5080f795f44e024672664a585a9304bc4f324ddd8b37e7f67914986dce5a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0552.json": {
        "expected": "a24c6bb7961a7a3b9b1b3b67004004669bfa1d025ea44af0d30c024346c3e4fb",
        "actual": "a24c6bb7961a7a3b9b1b3b67004004669bfa1d025ea44af0d30c024346c3e4fb",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0553.json": {
        "expected": "9250ae538f82da192acf9fd66b8782d3f50578fbdac76da107dba69f1ea16cfe",
        "actual": "9250ae538f82da192acf9fd66b8782d3f50578fbdac76da107dba69f1ea16cfe",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0554.json": {
        "expected": "a6ac9406447dbb2a48ffef9a67414c803e05d6e070a1848d0cdd7f5fb24bc73b",
        "actual": "a6ac9406447dbb2a48ffef9a67414c803e05d6e070a1848d0cdd7f5fb24bc73b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0555.json": {
        "expected": "4e601ae317b3c9876e4f27df57f21a7d02556e00da046c03fd78f9d3e7fac257",
        "actual": "4e601ae317b3c9876e4f27df57f21a7d02556e00da046c03fd78f9d3e7fac257",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0556.json": {
        "expected": "8600d6e23160ee9ff8449e953740978a579757e5cf3a9c9f09f81b8ee9eb5cfa",
        "actual": "8600d6e23160ee9ff8449e953740978a579757e5cf3a9c9f09f81b8ee9eb5cfa",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0557.json": {
        "expected": "de744c25adae1bb1339bc4b4152c42c35e32192158b1e92868dec9c38ed5b544",
        "actual": "de744c25adae1bb1339bc4b4152c42c35e32192158b1e92868dec9c38ed5b544",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0558.json": {
        "expected": "11be11512078809edf79b8bd51607cf09935a522856c604a16cc09877ac8328a",
        "actual": "11be11512078809edf79b8bd51607cf09935a522856c604a16cc09877ac8328a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0559.json": {
        "expected": "e210514a53b703f36952311b2198f29bfeb07176e2c8a6dbaa962eb35eabeefa",
        "actual": "e210514a53b703f36952311b2198f29bfeb07176e2c8a6dbaa962eb35eabeefa",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0560.json": {
        "expected": "aec4ac39f7e6174e299a1253d2d737d9f12f9aee7f28467103d8a5aa0b8f1d45",
        "actual": "aec4ac39f7e6174e299a1253d2d737d9f12f9aee7f28467103d8a5aa0b8f1d45",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0571.json": {
        "expected": "fbf1c5dbee30e0284f17e42f66a792cfb61ac3a5c0affd8a595337aa129e8169",
        "actual": "fbf1c5dbee30e0284f17e42f66a792cfb61ac3a5c0affd8a595337aa129e8169",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0572.json": {
        "expected": "2f6cbe6bb9ef6ae6fe740d6a8e8a0c4a373a389f770e7ef25484c2c73602a3e0",
        "actual": "2f6cbe6bb9ef6ae6fe740d6a8e8a0c4a373a389f770e7ef25484c2c73602a3e0",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0573.json": {
        "expected": "9fd629e2a849b9707f0c4da74b8ae965acb98c9dc596a17f593f2af47bd953c6",
        "actual": "9fd629e2a849b9707f0c4da74b8ae965acb98c9dc596a17f593f2af47bd953c6",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0574.json": {
        "expected": "08b8fbe6d63ac979d849ff36d684b6656863e7cf214ea8299b9f8b1647466b09",
        "actual": "08b8fbe6d63ac979d849ff36d684b6656863e7cf214ea8299b9f8b1647466b09",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0575.json": {
        "expected": "ad48e8f4b28285ecbd32ccd33482c96c33e35d32d1316babf3995dfe700d9e71",
        "actual": "ad48e8f4b28285ecbd32ccd33482c96c33e35d32d1316babf3995dfe700d9e71",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0576.json": {
        "expected": "e802f8d2bb3632a2791a22d6bbf3a81b0c5724fc875cffa627bcc12b18d43f6f",
        "actual": "e802f8d2bb3632a2791a22d6bbf3a81b0c5724fc875cffa627bcc12b18d43f6f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0577.json": {
        "expected": "b6e3b6c4f88f21152af6dd25b4a3356e5032b6107803731c56dc3b5640f77726",
        "actual": "b6e3b6c4f88f21152af6dd25b4a3356e5032b6107803731c56dc3b5640f77726",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0578.json": {
        "expected": "5bd0c257a61cb494f6e9133acb6755f5ff7ee1773e7a4de2945b8c4fa25a00f3",
        "actual": "5bd0c257a61cb494f6e9133acb6755f5ff7ee1773e7a4de2945b8c4fa25a00f3",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0579.json": {
        "expected": "526826808e369980cec2b1330607ab3142f39f571d4d35bd39991423423cbe70",
        "actual": "526826808e369980cec2b1330607ab3142f39f571d4d35bd39991423423cbe70",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0580.json": {
        "expected": "94f6532773824fb13ad805c8ba7cc810b074428badcfa62175b8f2e68f5b8273",
        "actual": "94f6532773824fb13ad805c8ba7cc810b074428badcfa62175b8f2e68f5b8273",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0591.json": {
        "expected": "9afbb0daddb88eeda962c760d63ab3b3596b4810c795af565c98744178277372",
        "actual": "9afbb0daddb88eeda962c760d63ab3b3596b4810c795af565c98744178277372",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0592.json": {
        "expected": "ceac53d4d3025fedd06323811e0db7be6d503affda2f18590602c5b156995cc6",
        "actual": "ceac53d4d3025fedd06323811e0db7be6d503affda2f18590602c5b156995cc6",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0593.json": {
        "expected": "0ed6b1394047504bef026d8110448391e967da5ca13e7e7d4949f50c98464551",
        "actual": "0ed6b1394047504bef026d8110448391e967da5ca13e7e7d4949f50c98464551",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0594.json": {
        "expected": "0656c382158425c1a65a18549f111a46e57bf9ca58261bb62a5295ec95d2a1ef",
        "actual": "0656c382158425c1a65a18549f111a46e57bf9ca58261bb62a5295ec95d2a1ef",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0595.json": {
        "expected": "85e8c2fa0c5e6690962ac79702d01507d773ac2450e8fe14243ea8476d3628a7",
        "actual": "85e8c2fa0c5e6690962ac79702d01507d773ac2450e8fe14243ea8476d3628a7",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0596.json": {
        "expected": "d1eeee6824d822525df97155340f6a276adbd3f9c0f25012128d10bee078ec7e",
        "actual": "d1eeee6824d822525df97155340f6a276adbd3f9c0f25012128d10bee078ec7e",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0597.json": {
        "expected": "beab77021320f163038c0145381555cf7d7125066b07bfd2ed7ab9e2d1559f2f",
        "actual": "beab77021320f163038c0145381555cf7d7125066b07bfd2ed7ab9e2d1559f2f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0598.json": {
        "expected": "84e58f854298cc0df883f6bd75e5a81ab55400fcf9dea74316bf59bd1ef20691",
        "actual": "84e58f854298cc0df883f6bd75e5a81ab55400fcf9dea74316bf59bd1ef20691",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0599.json": {
        "expected": "4556d162eae34076a7aa87e2f3a16815bbc84bf6ee065bfc805c9fec870da2c2",
        "actual": "4556d162eae34076a7aa87e2f3a16815bbc84bf6ee065bfc805c9fec870da2c2",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_multiseed_20261002_203333_r2\\seed_4104\\stage3\\runs\\stage3\\episodes\\episode-0600.json": {
        "expected": "9a62a22af44fbd555d8fec2b541b4059613465581a3e6dfa97f48133bd1e374b",
        "actual": "9a62a22af44fbd555d8fec2b541b4059613465581a3e6dfa97f48133bd1e374b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\failure.json": {
        "expected": "2a49a3422fa5b970dc7fad1d2489c472573b499c6bf6e49ea4345f56589c34f4",
        "actual": "2a49a3422fa5b970dc7fad1d2489c472573b499c6bf6e49ea4345f56589c34f4",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\checks.json": {
        "expected": "8b35d8bcfa7b2ea89aa603116db651de96873b1281a395c98f373026e609f03c",
        "actual": "8b35d8bcfa7b2ea89aa603116db651de96873b1281a395c98f373026e609f03c",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\summary.json": {
        "expected": "bad859f0f4e8e75f8aa1ae44000ea0a64e15ba5aad6181b90ed2ba0127c4d1b8",
        "actual": "bad859f0f4e8e75f8aa1ae44000ea0a64e15ba5aad6181b90ed2ba0127c4d1b8",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\01_1.json": {
        "expected": "639a4b500b840a7dbd610e283dbc73bf7d222462fbebcdee620bb1d9de88242f",
        "actual": "639a4b500b840a7dbd610e283dbc73bf7d222462fbebcdee620bb1d9de88242f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\01_2.json": {
        "expected": "895325d7e74d760cabbfa728f50a097ad82c7b6bc2a8506daf256f55b10dae3c",
        "actual": "895325d7e74d760cabbfa728f50a097ad82c7b6bc2a8506daf256f55b10dae3c",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\02_1.json": {
        "expected": "1ac81f8cb9bbce0444b1f567c1a89eb103aa332a5f6c72e65a6c81f260a134ee",
        "actual": "1ac81f8cb9bbce0444b1f567c1a89eb103aa332a5f6c72e65a6c81f260a134ee",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\02_2.json": {
        "expected": "207baf40e69daf4a2416e950eec1a6e525222971c0a54ada230cfe38a18face9",
        "actual": "207baf40e69daf4a2416e950eec1a6e525222971c0a54ada230cfe38a18face9",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\03_1.json": {
        "expected": "efd694116f90d9a02bf5cd1e1bf8424dfc711b35009bb0ae663a6a9110d8c7b7",
        "actual": "efd694116f90d9a02bf5cd1e1bf8424dfc711b35009bb0ae663a6a9110d8c7b7",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\03_2.json": {
        "expected": "5149890a71b29e93b5758086e11d26f4408be3f13a281a1184a1e761cc65da32",
        "actual": "5149890a71b29e93b5758086e11d26f4408be3f13a281a1184a1e761cc65da32",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\04_1.json": {
        "expected": "e95fef4c0554b14a6b6d020d850da93e9ef8d9cd02992feb51a65b7d3f8c91fd",
        "actual": "e95fef4c0554b14a6b6d020d850da93e9ef8d9cd02992feb51a65b7d3f8c91fd",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\04_2.json": {
        "expected": "3345c901aa091a8934b4682d1ffae8effdb83ba794a8780cf09127fec01effcb",
        "actual": "3345c901aa091a8934b4682d1ffae8effdb83ba794a8780cf09127fec01effcb",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\05_1.json": {
        "expected": "8275c36d469e54e80ea1a9d8071059b32fe2933ec3346e30d41e8c184074acf9",
        "actual": "8275c36d469e54e80ea1a9d8071059b32fe2933ec3346e30d41e8c184074acf9",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\05_2.json": {
        "expected": "ac060af5b6b7fe8135b178d8ba0b97025d455d4fffe678b104079252c27aa1ad",
        "actual": "ac060af5b6b7fe8135b178d8ba0b97025d455d4fffe678b104079252c27aa1ad",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\06_1.json": {
        "expected": "96d3f83d641d833c4d304c4cf1913d3d8e1c35810fcf4be731d84cf8f3e72e87",
        "actual": "96d3f83d641d833c4d304c4cf1913d3d8e1c35810fcf4be731d84cf8f3e72e87",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\06_2.json": {
        "expected": "1a708a74af02b86d6574cc38d01af8d8fc3107d93d68ffde43fc7891a5be18e4",
        "actual": "1a708a74af02b86d6574cc38d01af8d8fc3107d93d68ffde43fc7891a5be18e4",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\07_1.json": {
        "expected": "f558ec131d3d989f7ec70dac5b65e011026999cea9954ddf9c551dd3213f49ce",
        "actual": "f558ec131d3d989f7ec70dac5b65e011026999cea9954ddf9c551dd3213f49ce",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\07_2.json": {
        "expected": "fd335c8a84cde7436deb99ea46d3401df479d4c8b7f5f598d8008a03ef58cafc",
        "actual": "fd335c8a84cde7436deb99ea46d3401df479d4c8b7f5f598d8008a03ef58cafc",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\08_1.json": {
        "expected": "1ae287742206d8a05f4b0233fa8e851040a38a98a56a3f8592fec8e7d86fe6ad",
        "actual": "1ae287742206d8a05f4b0233fa8e851040a38a98a56a3f8592fec8e7d86fe6ad",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\08_2.json": {
        "expected": "41ba3bfcf656e72f9bfef58257e792f262372654d93a9d4145b36bc991f32779",
        "actual": "41ba3bfcf656e72f9bfef58257e792f262372654d93a9d4145b36bc991f32779",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\09_1.json": {
        "expected": "361ae8f88f1d81c854b578f5d20dfd843cd33cf42e0b7bf976aa1cd791e50d22",
        "actual": "361ae8f88f1d81c854b578f5d20dfd843cd33cf42e0b7bf976aa1cd791e50d22",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\09_2.json": {
        "expected": "9202a4fe7c71d6290486a0ed7eef6399e8b7ae2ac3f934340db3bec9c05b8300",
        "actual": "9202a4fe7c71d6290486a0ed7eef6399e8b7ae2ac3f934340db3bec9c05b8300",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\10_1.json": {
        "expected": "323151abfeeaa60ec96e65ed684e962d9af9b11f06cc6b8a7ba50928c0a727b0",
        "actual": "323151abfeeaa60ec96e65ed684e962d9af9b11f06cc6b8a7ba50928c0a727b0",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\10_2.json": {
        "expected": "19d2ad79ef55b058afb913195ab422007f7a6486cfa28d64d4fb0c64aa9c598e",
        "actual": "19d2ad79ef55b058afb913195ab422007f7a6486cfa28d64d4fb0c64aa9c598e",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\11_1.json": {
        "expected": "29f36cb9f7414f1e7ad7c6dfc92eaf915aa3000ee54c9d5cb8d83e6605be8973",
        "actual": "29f36cb9f7414f1e7ad7c6dfc92eaf915aa3000ee54c9d5cb8d83e6605be8973",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\11_2.json": {
        "expected": "08ae0ca642ff0c402a3b0c12dba2a4632052525511649e89e821f24b6289f0e1",
        "actual": "08ae0ca642ff0c402a3b0c12dba2a4632052525511649e89e821f24b6289f0e1",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\12_1.json": {
        "expected": "f4669b4bfdae318f240346b2f46e9a8e998b01850886538a37127cff0b9795d2",
        "actual": "f4669b4bfdae318f240346b2f46e9a8e998b01850886538a37127cff0b9795d2",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\12_2.json": {
        "expected": "8cafb26eabe1752ef1fa94f5dbd89c4c71b68f480ab73f9df6dbdec1bbe0a324",
        "actual": "8cafb26eabe1752ef1fa94f5dbd89c4c71b68f480ab73f9df6dbdec1bbe0a324",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\13_1.json": {
        "expected": "eb3bb7d1317920480f9b6c41e13149db57ed571ff354f6b507fd959a0cae3523",
        "actual": "eb3bb7d1317920480f9b6c41e13149db57ed571ff354f6b507fd959a0cae3523",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\13_2.json": {
        "expected": "216ab76e0fab7cc721b1ce95214b817ce4ef19b168d0bf9bd0e20df0f4b784cf",
        "actual": "216ab76e0fab7cc721b1ce95214b817ce4ef19b168d0bf9bd0e20df0f4b784cf",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\14_1.json": {
        "expected": "12e7ca1f603826ab560052786def981ee8440f3720f7d6780d57ff419f7aa94d",
        "actual": "12e7ca1f603826ab560052786def981ee8440f3720f7d6780d57ff419f7aa94d",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\14_2.json": {
        "expected": "1bd029e61dabee913f1610fb097da6287553208cfdfd7a88a8d910165d61cf48",
        "actual": "1bd029e61dabee913f1610fb097da6287553208cfdfd7a88a8d910165d61cf48",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\15_1.json": {
        "expected": "39ff7f45aa02f9a26f617f2c4847ac8a517292b1f6f9cfb9918785f14073dc9d",
        "actual": "39ff7f45aa02f9a26f617f2c4847ac8a517292b1f6f9cfb9918785f14073dc9d",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\15_2.json": {
        "expected": "432138671e772598543941d05718c89b423748f686eec029d1fc2d2b88f35339",
        "actual": "432138671e772598543941d05718c89b423748f686eec029d1fc2d2b88f35339",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\16_1.json": {
        "expected": "cdbe7f23896ffb6669500da4028ee3eb1164a2612a7d0fee4a15e226e4c84870",
        "actual": "cdbe7f23896ffb6669500da4028ee3eb1164a2612a7d0fee4a15e226e4c84870",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\16_2.json": {
        "expected": "8079d0ab340c1b3998550c6ab150380626ed3471041a74db94ac3b39b27c51be",
        "actual": "8079d0ab340c1b3998550c6ab150380626ed3471041a74db94ac3b39b27c51be",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\17_1.json": {
        "expected": "d71716b25163901127c7be83d932663cede25a9d216a2b09c3b6f5a4b066376c",
        "actual": "d71716b25163901127c7be83d932663cede25a9d216a2b09c3b6f5a4b066376c",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\17_2.json": {
        "expected": "6da77a7d456a2431d629be057cebb57fd33dddcca48ae60adb86b6b90d855ba1",
        "actual": "6da77a7d456a2431d629be057cebb57fd33dddcca48ae60adb86b6b90d855ba1",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\18_1.json": {
        "expected": "f538a59e6cd28bbcaa6fbe96592f1489ca49a8893797b523b0f306200c7bce70",
        "actual": "f538a59e6cd28bbcaa6fbe96592f1489ca49a8893797b523b0f306200c7bce70",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\18_2.json": {
        "expected": "8c2707104db7bfcd8a07d4d745775eea36d68ceaed84a0d5e0390c0fed56887c",
        "actual": "8c2707104db7bfcd8a07d4d745775eea36d68ceaed84a0d5e0390c0fed56887c",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\19_1.json": {
        "expected": "e15761017f0bfdd281968ae30fac6e2f480ee390ebe131c58b4607ac36f656c6",
        "actual": "e15761017f0bfdd281968ae30fac6e2f480ee390ebe131c58b4607ac36f656c6",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\19_2.json": {
        "expected": "a3603f69cfc9066cd0f8b890fe4fbbab1057c4d9ea586287703e213ac064c6d1",
        "actual": "a3603f69cfc9066cd0f8b890fe4fbbab1057c4d9ea586287703e213ac064c6d1",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\20_1.json": {
        "expected": "d6d217df58910a79e714b074e7745070ce15c37bd8eafca3cb3e082d5d3063a0",
        "actual": "d6d217df58910a79e714b074e7745070ce15c37bd8eafca3cb3e082d5d3063a0",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\E5\\windows\\20_2.json": {
        "expected": "3188e273f29ae12f370b7855103e70bb12ed1c335af1d3d6847fad694ecd07a5",
        "actual": "3188e273f29ae12f370b7855103e70bb12ed1c335af1d3d6847fad694ecd07a5",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\summary.json": {
        "expected": "c1c09d9f587a8ee40b8f4bacb5fa8d1d5c063012c72584dbebdfc8704ffcb52f",
        "actual": "c1c09d9f587a8ee40b8f4bacb5fa8d1d5c063012c72584dbebdfc8704ffcb52f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\01_1.json": {
        "expected": "d6d5cdbad1707815d95767b6c8ab5b08ae8d3ddf4f5bd93e9026004913defa07",
        "actual": "d6d5cdbad1707815d95767b6c8ab5b08ae8d3ddf4f5bd93e9026004913defa07",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\01_2.json": {
        "expected": "b465036f639b9d2a7e3678593fc57a550da795e185b2773be998ad2c76b1f050",
        "actual": "b465036f639b9d2a7e3678593fc57a550da795e185b2773be998ad2c76b1f050",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\02_1.json": {
        "expected": "99a18e9fc679404d8e744a4b3243f4c5327e4d3d1a8ae62354db68b58a2b4387",
        "actual": "99a18e9fc679404d8e744a4b3243f4c5327e4d3d1a8ae62354db68b58a2b4387",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\02_2.json": {
        "expected": "2e35c8ae26dfaf17ee028631d3f3e6c4a1d32d88c0640822ec60ddbdc9e4d43c",
        "actual": "2e35c8ae26dfaf17ee028631d3f3e6c4a1d32d88c0640822ec60ddbdc9e4d43c",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\03_1.json": {
        "expected": "54accb00864644c00c62eec27b836c8e62c4a88c579d44e74c6049e7014b323b",
        "actual": "54accb00864644c00c62eec27b836c8e62c4a88c579d44e74c6049e7014b323b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\03_2.json": {
        "expected": "809fe523d0284cba98d3d336934089068555a6d6414528dd9fb56a49445865dd",
        "actual": "809fe523d0284cba98d3d336934089068555a6d6414528dd9fb56a49445865dd",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\04_1.json": {
        "expected": "0891478cfd36edab12630393b6595f1b54af902dad054a42cadb005b00a6acda",
        "actual": "0891478cfd36edab12630393b6595f1b54af902dad054a42cadb005b00a6acda",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\04_2.json": {
        "expected": "95411a0204a03fae7afea4c63a59d30578677384d67fe587949698ef5774618e",
        "actual": "95411a0204a03fae7afea4c63a59d30578677384d67fe587949698ef5774618e",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\05_1.json": {
        "expected": "cf75bff0d218fed0623893b45d668d816f47567541d0f696fbc430ec28832f83",
        "actual": "cf75bff0d218fed0623893b45d668d816f47567541d0f696fbc430ec28832f83",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\05_2.json": {
        "expected": "74d4d79d8263582dc01368733ab28b0211fda872dde2219a9280ec6291f12b64",
        "actual": "74d4d79d8263582dc01368733ab28b0211fda872dde2219a9280ec6291f12b64",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\06_1.json": {
        "expected": "df05d6ae87070ee3a1670d91131e0f6ce7a60275eacd49c11e8b3040deb6270b",
        "actual": "df05d6ae87070ee3a1670d91131e0f6ce7a60275eacd49c11e8b3040deb6270b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\06_2.json": {
        "expected": "046dcea617433c9afb1565679b1a40bb4077554e23fefaba27f4324bb041b58a",
        "actual": "046dcea617433c9afb1565679b1a40bb4077554e23fefaba27f4324bb041b58a",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\07_1.json": {
        "expected": "f1de7a363e06a644c61033e819d963f2ead41f606cc8bca06d14fd83be062d4f",
        "actual": "f1de7a363e06a644c61033e819d963f2ead41f606cc8bca06d14fd83be062d4f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\07_2.json": {
        "expected": "84d4abdfba7f1e4f0d92c3b1e3ae662a8e5ebcd6ce9a4bad9244edc36cc6ac53",
        "actual": "84d4abdfba7f1e4f0d92c3b1e3ae662a8e5ebcd6ce9a4bad9244edc36cc6ac53",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\08_1.json": {
        "expected": "666d7f2b050a1218f571fe89175f4e8ad2e36ccb52dc9a887889fa14d48e3838",
        "actual": "666d7f2b050a1218f571fe89175f4e8ad2e36ccb52dc9a887889fa14d48e3838",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\08_2.json": {
        "expected": "b7e05ba5ab498f64645ab36807bad13f920a92c6bb41d6061289777c0a1cff09",
        "actual": "b7e05ba5ab498f64645ab36807bad13f920a92c6bb41d6061289777c0a1cff09",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\09_1.json": {
        "expected": "aa271a120866d99fdd08e6cf48f4ab1c4dbcc7701cbccfeefe4ab9b74cc37cf2",
        "actual": "aa271a120866d99fdd08e6cf48f4ab1c4dbcc7701cbccfeefe4ab9b74cc37cf2",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\09_2.json": {
        "expected": "4d5b716a1eec67b10bcc8be39cb71003465c2eadc1368592a202878ddb3f8c36",
        "actual": "4d5b716a1eec67b10bcc8be39cb71003465c2eadc1368592a202878ddb3f8c36",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\10_1.json": {
        "expected": "4946d334fb36169ab4b31bd37d036792f9af5825d3b8a78ce88c8e7a821e8394",
        "actual": "4946d334fb36169ab4b31bd37d036792f9af5825d3b8a78ce88c8e7a821e8394",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\10_2.json": {
        "expected": "cc87615d9f414695e7ac8ca21e86f9409ee71621d532ad7e62fac264d2785727",
        "actual": "cc87615d9f414695e7ac8ca21e86f9409ee71621d532ad7e62fac264d2785727",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\11_1.json": {
        "expected": "44c4125570f89ab4f3e4f59cae4a98d8d4d81e2ae4c0021f6d31f1485a67158b",
        "actual": "44c4125570f89ab4f3e4f59cae4a98d8d4d81e2ae4c0021f6d31f1485a67158b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\11_2.json": {
        "expected": "6c122a84802e890a7525a65fc9ca985e10920bc36881f113d7a3600d6c4578b1",
        "actual": "6c122a84802e890a7525a65fc9ca985e10920bc36881f113d7a3600d6c4578b1",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\12_1.json": {
        "expected": "79c33f20b535aefd0721c7258257440fc3ba894ca0713df41bb819af522a28f8",
        "actual": "79c33f20b535aefd0721c7258257440fc3ba894ca0713df41bb819af522a28f8",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\12_2.json": {
        "expected": "df070540424fc42d3da4064ea84aa560e4c64552986fa7b9d8d3408ec2a469ac",
        "actual": "df070540424fc42d3da4064ea84aa560e4c64552986fa7b9d8d3408ec2a469ac",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\13_1.json": {
        "expected": "b9b0a8b9674f5f2f37277650b8a86c3a8b3e8a5c934798ef056685dc81c44a8f",
        "actual": "b9b0a8b9674f5f2f37277650b8a86c3a8b3e8a5c934798ef056685dc81c44a8f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\13_2.json": {
        "expected": "a7eef362dfe40e201f8316a0312b199bfe61442fcbbef1c0ac7d6b3aadf50036",
        "actual": "a7eef362dfe40e201f8316a0312b199bfe61442fcbbef1c0ac7d6b3aadf50036",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\14_1.json": {
        "expected": "f3be3e7ab3a11c8c44a3d55daee5d6011f850b8f8109e7fd4a3c85e3b389d92d",
        "actual": "f3be3e7ab3a11c8c44a3d55daee5d6011f850b8f8109e7fd4a3c85e3b389d92d",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\14_2.json": {
        "expected": "310ba2e927da60178add5c3f11065d809df9c847684c105cba6f2978baa7923b",
        "actual": "310ba2e927da60178add5c3f11065d809df9c847684c105cba6f2978baa7923b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\15_1.json": {
        "expected": "5febb415aec7aa3ad2a2be7058cc0939bf9622690064f7a858fb65ee6edc68ad",
        "actual": "5febb415aec7aa3ad2a2be7058cc0939bf9622690064f7a858fb65ee6edc68ad",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\15_2.json": {
        "expected": "27d8fb01f8c0ab09a3f629e7c95554150bd6f87290a45d95a8a2d34c3ff9e149",
        "actual": "27d8fb01f8c0ab09a3f629e7c95554150bd6f87290a45d95a8a2d34c3ff9e149",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\16_1.json": {
        "expected": "20fd7a137a7b3cec3f9572f630f7b826adbf96a4fe90dab434ef24f3aeaab2b4",
        "actual": "20fd7a137a7b3cec3f9572f630f7b826adbf96a4fe90dab434ef24f3aeaab2b4",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\16_2.json": {
        "expected": "fcd0f1fd6349c138c9226708aca4131b9af6bab193d713a0e9577957b58e1a72",
        "actual": "fcd0f1fd6349c138c9226708aca4131b9af6bab193d713a0e9577957b58e1a72",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\17_1.json": {
        "expected": "94e50c04cc78f0f6e8d038211498a0c205b45fe35e3b5abf306654bc00127943",
        "actual": "94e50c04cc78f0f6e8d038211498a0c205b45fe35e3b5abf306654bc00127943",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\17_2.json": {
        "expected": "93651ce35ac4e8d57f3f575b539ba1d0f15b3fa1ae7433b393f832642c716940",
        "actual": "93651ce35ac4e8d57f3f575b539ba1d0f15b3fa1ae7433b393f832642c716940",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\18_1.json": {
        "expected": "94d2f69f82a63073654694c8fb548aa75b4cbfe3cbb941bbe7f2d487d1cf3263",
        "actual": "94d2f69f82a63073654694c8fb548aa75b4cbfe3cbb941bbe7f2d487d1cf3263",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\18_2.json": {
        "expected": "6194b9bd00df73aa3887401488c611e37827b090c4cfbeb2ca24a1528d34d39c",
        "actual": "6194b9bd00df73aa3887401488c611e37827b090c4cfbeb2ca24a1528d34d39c",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\19_1.json": {
        "expected": "0803e6135b2755cbc8e8522b88ac309e4c61a48d6b1b5a04d916879d857b7a85",
        "actual": "0803e6135b2755cbc8e8522b88ac309e4c61a48d6b1b5a04d916879d857b7a85",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\19_2.json": {
        "expected": "eae94203afafdede852c7d29883dcdf3e623be9399ace3f52375c4e61d2395d5",
        "actual": "eae94203afafdede852c7d29883dcdf3e623be9399ace3f52375c4e61d2395d5",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\20_1.json": {
        "expected": "bfa06fa2b7c79233bc325a2355d2770f52618806d2de19ceb064661da9869a76",
        "actual": "bfa06fa2b7c79233bc325a2355d2770f52618806d2de19ceb064661da9869a76",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-E\\TFOO-200\\windows\\20_2.json": {
        "expected": "38f3192e7590297d874cb22c4b87048079b5d82d13e3159fa23b93901535ae84",
        "actual": "38f3192e7590297d874cb22c4b87048079b5d82d13e3159fa23b93901535ae84",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-L\\run\\episodes\\episode-0411.json": {
        "expected": "d1852be6632bc94331a13bb214e8e236ef92f8432e878acf43a2687251460d81",
        "actual": "d1852be6632bc94331a13bb214e8e236ef92f8432e878acf43a2687251460d81",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-L\\run\\episodes\\episode-0412.json": {
        "expected": "0eac0ae5aa39520cb873dd3fb702a8a813a0f73bdee7f214a13bd69884df9dc6",
        "actual": "0eac0ae5aa39520cb873dd3fb702a8a813a0f73bdee7f214a13bd69884df9dc6",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-L\\run\\episodes\\episode-0413.json": {
        "expected": "d2ec335dca12fec2d9fafe37806019d7d06426b686b81652b69ba6483549a279",
        "actual": "d2ec335dca12fec2d9fafe37806019d7d06426b686b81652b69ba6483549a279",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-L\\run\\learning.jsonl": {
        "expected": "05a2f8df52ff1da262677af21d8e08c26ce3798cc17fed5cde10da807b20ee1f",
        "actual": "05a2f8df52ff1da262677af21d8e08c26ce3798cc17fed5cde10da807b20ee1f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-L\\run\\position_25_bins.json": {
        "expected": "37b44934b0e439957a3859e74b8a07243bf4429a8864491185886f26617feefb",
        "actual": "37b44934b0e439957a3859e74b8a07243bf4429a8864491185886f26617feefb",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-L\\run\\positions\\episode-0411.json": {
        "expected": "f10c2379731f706da7ed9680b3174d8b208d7ed200497949cb93f98fbc2804ce",
        "actual": "f10c2379731f706da7ed9680b3174d8b208d7ed200497949cb93f98fbc2804ce",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-L\\run\\positions\\episode-0412.json": {
        "expected": "198445494bfea41fdc919046b5f13dc6e8cb08e2ea1ed7da36004c531cdce1a1",
        "actual": "198445494bfea41fdc919046b5f13dc6e8cb08e2ea1ed7da36004c531cdce1a1",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-L\\run\\positions\\episode-0413.json": {
        "expected": "46240eee83d2808588e70f3b32211b9349c6f8b2001b8454d6240e224404070f",
        "actual": "46240eee83d2808588e70f3b32211b9349c6f8b2001b8454d6240e224404070f",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-L\\run\\progress.json": {
        "expected": "0bddad61a87be460695845460a30bbdf5936404a6b2d535d7c2554b2624c4254",
        "actual": "0bddad61a87be460695845460a30bbdf5936404a6b2d535d7c2554b2624c4254",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-L\\run\\result.json": {
        "expected": "649b0ad4711c4112b05bdaa41d01b415ed5152d60d5c6e1898b8bcfccc4ce1c4",
        "actual": "649b0ad4711c4112b05bdaa41d01b415ed5152d60d5c6e1898b8bcfccc4ce1c4",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-L\\run\\train_metrics.jsonl": {
        "expected": "e10c0977d7c944d3bdd2cadfc1dedded179a12bc50e470a838d7991043c56fb7",
        "actual": "e10c0977d7c944d3bdd2cadfc1dedded179a12bc50e470a838d7991043c56fb7",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-U\\run\\episodes\\episode-0001.json": {
        "expected": "619113402dcff35b1a0c26a71c7855dc2a49089d74755bfb77e38b3911e8c058",
        "actual": "619113402dcff35b1a0c26a71c7855dc2a49089d74755bfb77e38b3911e8c058",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-U\\run\\episodes\\episode-0002.json": {
        "expected": "c68ec72c5fed3b23a4810b3e0e9052bbbc474453f7ccc3e6206c637ee4751b18",
        "actual": "c68ec72c5fed3b23a4810b3e0e9052bbbc474453f7ccc3e6206c637ee4751b18",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-U\\run\\episodes\\episode-0003.json": {
        "expected": "22f182c5b994c516b9ac89d9b3a412d2226ee6fa4964653ca8f3ff87e7bbad98",
        "actual": "22f182c5b994c516b9ac89d9b3a412d2226ee6fa4964653ca8f3ff87e7bbad98",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-U\\run\\episodes\\episode-0004.json": {
        "expected": "855d0eb8569ace64741eefb8b53e233ba0ed64306d127bc70f81b290d8d8a527",
        "actual": "855d0eb8569ace64741eefb8b53e233ba0ed64306d127bc70f81b290d8d8a527",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-U\\run\\episodes\\episode-0005.json": {
        "expected": "4d834d83f7fcb78343dd20c961a47ad51961d1d439a906b78640d8a1deb3e321",
        "actual": "4d834d83f7fcb78343dd20c961a47ad51961d1d439a906b78640d8a1deb3e321",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-U\\run\\episodes\\episode-0006.json": {
        "expected": "c0088b9c20599c3829b97c663c616267b7c8a9aa910501bf46f5e37638e29ab9",
        "actual": "c0088b9c20599c3829b97c663c616267b7c8a9aa910501bf46f5e37638e29ab9",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-U\\run\\episodes\\episode-0007.json": {
        "expected": "5c9dd0828be5bc8f0ad443f304c75d5ec4ac85e7940ce82d74fd057fb6ed47b6",
        "actual": "5c9dd0828be5bc8f0ad443f304c75d5ec4ac85e7940ce82d74fd057fb6ed47b6",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-U\\run\\episodes\\episode-0008.json": {
        "expected": "f68de5a8addeb54e258f2fa4c1336e924aa700699a1a99111d2e71f9158b076e",
        "actual": "f68de5a8addeb54e258f2fa4c1336e924aa700699a1a99111d2e71f9158b076e",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-U\\run\\episodes\\episode-0009.json": {
        "expected": "538fb303dec0940e2b29c0fd1d6f0ded2497d02928eb56e220ef2113d2dfffd1",
        "actual": "538fb303dec0940e2b29c0fd1d6f0ded2497d02928eb56e220ef2113d2dfffd1",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-U\\run\\episodes\\episode-0010.json": {
        "expected": "a322559bb8cb0f8ae282320121155a4a86bb416865ed158def492adb454f8899",
        "actual": "a322559bb8cb0f8ae282320121155a4a86bb416865ed158def492adb454f8899",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-U\\run\\learning.jsonl": {
        "expected": "80a24c61ae75bcca5f65d2c9b87df5feb89a92a44b3cbc352a16a554240abfce",
        "actual": "80a24c61ae75bcca5f65d2c9b87df5feb89a92a44b3cbc352a16a554240abfce",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-U\\run\\position_25_bins.json": {
        "expected": "0ea7754becb2028243280ae87aebeb7a073494b6a20d4a4285c4493ae85e5998",
        "actual": "0ea7754becb2028243280ae87aebeb7a073494b6a20d4a4285c4493ae85e5998",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-U\\run\\positions\\episode-0001.json": {
        "expected": "e81dde6b3b42ef129852f72d8e2c24b7b5dd920791abaf8c92acfa37799c31e0",
        "actual": "e81dde6b3b42ef129852f72d8e2c24b7b5dd920791abaf8c92acfa37799c31e0",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-U\\run\\positions\\episode-0002.json": {
        "expected": "cde491a354ce8070e9a3b136e91680a13b541ae3ec8e62af5fdc85ebd1f3e585",
        "actual": "cde491a354ce8070e9a3b136e91680a13b541ae3ec8e62af5fdc85ebd1f3e585",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-U\\run\\positions\\episode-0003.json": {
        "expected": "e51d9529e819e6f9af2c9e4d88a6c1ec537896c27c5fedf75756bb0b86d4db53",
        "actual": "e51d9529e819e6f9af2c9e4d88a6c1ec537896c27c5fedf75756bb0b86d4db53",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-U\\run\\positions\\episode-0004.json": {
        "expected": "bb177d77e6e89d910778794459fe8d9676dd0f72f5290a3ba45c583d7359964b",
        "actual": "bb177d77e6e89d910778794459fe8d9676dd0f72f5290a3ba45c583d7359964b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-U\\run\\positions\\episode-0005.json": {
        "expected": "3d3323f399b3c2d2b3ba0bc125e2a89b3b891445d4e9b35b39f0752ed6337e19",
        "actual": "3d3323f399b3c2d2b3ba0bc125e2a89b3b891445d4e9b35b39f0752ed6337e19",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-U\\run\\positions\\episode-0006.json": {
        "expected": "701d4eedeb74385a7dc92f04d35541ab7dc0adea4540fe3e6fad0ffeb123379b",
        "actual": "701d4eedeb74385a7dc92f04d35541ab7dc0adea4540fe3e6fad0ffeb123379b",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-U\\run\\positions\\episode-0007.json": {
        "expected": "085a816e04fb846b9a6f2abe67f04882b5e28f8de0f3424b8217fa3685714d16",
        "actual": "085a816e04fb846b9a6f2abe67f04882b5e28f8de0f3424b8217fa3685714d16",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-U\\run\\positions\\episode-0008.json": {
        "expected": "dbdf6fa77bf2b35164eb1d1c805e5942be0c3796ae41bab56982e2b9fc4a98fd",
        "actual": "dbdf6fa77bf2b35164eb1d1c805e5942be0c3796ae41bab56982e2b9fc4a98fd",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-U\\run\\positions\\episode-0009.json": {
        "expected": "6d193955a4ce8d26abb6cddd2cc61a55883221ad61f6cbe0ca1495d7fa83c762",
        "actual": "6d193955a4ce8d26abb6cddd2cc61a55883221ad61f6cbe0ca1495d7fa83c762",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-U\\run\\positions\\episode-0010.json": {
        "expected": "6604afb85a79c7015d16ac19ad31db997a5fa09d052eadfbea774993d74ef257",
        "actual": "6604afb85a79c7015d16ac19ad31db997a5fa09d052eadfbea774993d74ef257",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-U\\run\\progress.json": {
        "expected": "5bbcf89c62630ad307936da126c81aa73ab111238263ec49013cf73191b43dd9",
        "actual": "5bbcf89c62630ad307936da126c81aa73ab111238263ec49013cf73191b43dd9",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-U\\run\\result.json": {
        "expected": "20c793405a9a4a297fc337137fe7d8cf7cb8ce4970895b5e9ffca3df1fec1b81",
        "actual": "20c793405a9a4a297fc337137fe7d8cf7cb8ce4970895b5e9ffca3df1fec1b81",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\RP-U\\run\\train_metrics.jsonl": {
        "expected": "ad7f3a6cf321b1d55e3e956ddecc4ad5e60965abdf1c422e4b25a9f854d5db37",
        "actual": "ad7f3a6cf321b1d55e3e956ddecc4ad5e60965abdf1c422e4b25a9f854d5db37",
        "unchanged": true
      },
      "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_curriculum_pilot_20261008_attempt2\\seed_4104\\rp\\training_checks.json": {
        "expected": "8aba497c744246160ba6cd3a11f89a689fc01a4cbbacab1920cc35477de52885",
        "actual": "8aba497c744246160ba6cd3a11f89a689fc01a4cbbacab1920cc35477de52885",
        "unchanged": true
      }
    },
    "protocol": {
      "path": "D:\\HRL_UAV_V2\\outputs\\scene_optimality_20260928\\v2_4_0_4101_test\\protocol_v2_4_0_three_stage.json",
      "sha256": "6797ca6538e663389845bf90c63743540feabf8beeab225df9bdd5aed194106a",
      "unchanged": true
    },
    "contract_sha256": "b76d59e8af1a50673322b77b110baff8a20b486659b4b2cc6c83e3bcd105edf7",
    "git": {
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
    },
    "time": "2026-10-09T10:15:44.836728+08:00"
  },
  "git_after": {
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
}
```
