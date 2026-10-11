# v2.4.0 五 seed 汇总

4101 主终点完整对象复算逐位一致；协议SHA256：6797ca6538e663389845bf90c63743540feabf8beeab225df9bdd5aed194106a

## 主终点与次要比较

20场景；先对5 seed的场景池化值取算术均值，场景等权；Wilcoxon双侧，zero_method=wilcox；次要比较未作多重校正。

| 候选 | 参考 | 候选均值s | 参考均值s | 相对差% | 胜场 | Wilcoxon p | 配对t p | 符号p |
|---|---|---|---|---|---|---|---|---|
| Proposed | TFOO | 2.186135074488039 | 0.8238365867714856 | 165.36028013215991 | 5 | 0.001209259033203125 | 0.007743187595321528 | 0.04138946533203125 |
| Proposed | NC | 2.186135074488039 | 4.961494927565456 | -55.93797622683958 | 17 | 0.0003223419189453125 | 0.0002686666384176536 | 0.0025768280029296875 |
| Proposed | CA | 2.186135074488039 | 1.31684462025671 | 66.01313783412554 | 13 | 0.956329345703125 | 0.07347394176147057 | 0.26317596435546875 |
| Proposed | TOO | 2.186135074488039 | 0.9763757301512708 | 123.90305360717426 | 13 | 0.6476554870605469 | 0.018857903496378135 | 0.26317596435546875 |
| Proposed | Abl_TwoStage | 2.186135074488039 | 1.2325397624090162 | 77.36832037087449 | 13 | 0.7285060882568359 | 0.08690606502638973 | 0.26317596435546875 |
| Proposed | Abl_HoverSAC | 2.186135074488039 | 1.6141817854923999 | 35.43301591779311 | 13 | 0.8983173370361328 | 0.2904890064183814 | 0.26317596435546875 |
| Proposed | Abl_UpperCVX | 2.186135074488039 | 0.8904398533815179 | 145.51181825319395 | 12 | 0.5958194732666016 | 0.011055345891557421 | 0.5034446716308594 |
| Abl_TwoStage | TFOO | 1.2325397624090162 | 0.8238365867714856 | 49.60973841173868 | 2 | 6.29425048828125e-05 | 0.022622484067845006 | 0.0004024505615234375 |

## 主终点逐场景

| 场景 | Proposed s | TFOO s | 差s | 胜 |
|---|---|---|---|---|
| 1 | 0.9841629995948745 | 0.9185714485282821 | 0.06559155106659242 | False |
| 2 | 0.8020561009033205 | 0.7864488072265583 | 0.015607293676762168 | False |
| 3 | 0.7013282787781586 | 0.7309665378727137 | -0.029638259094555086 | True |
| 4 | 1.073732644896943 | 0.9356193851137157 | 0.13811325978322742 | False |
| 5 | 0.819852298787813 | 0.8334219722616574 | -0.013569673473844412 | True |
| 6 | 0.7950195679286667 | 0.7717287153977397 | 0.023290852530926998 | False |
| 7 | 0.8131930824462863 | 0.8012647724415386 | 0.0119283100047477 | False |
| 8 | 5.083377565685743 | 0.7740449579304276 | 4.309332607755316 | False |
| 9 | 5.326335359819035 | 0.838835289127668 | 4.487500070691367 | False |
| 10 | 6.087610961126887 | 1.0134499013834284 | 5.074161059743458 | False |
| 11 | 5.306308773394555 | 0.9203723648507618 | 4.3859364085437935 | False |
| 12 | 1.0968616105450335 | 1.0119040898698817 | 0.08495752067515183 | False |
| 13 | 1.3319439330543925 | 0.7158340306834036 | 0.616109902370989 | False |
| 14 | 0.7314850521846803 | 0.7701087917919708 | -0.038623739607290486 | True |
| 15 | 0.7765984137825293 | 0.7876523145115201 | -0.011053900728990818 | True |
| 16 | 0.7752793042887083 | 0.7795204237720237 | -0.004241119483315403 | True |
| 17 | 0.9409497501388115 | 0.8615263742365352 | 0.07942337590227633 | False |
| 18 | 0.7768046925070367 | 0.7449641931618695 | 0.031840499345167195 | False |
| 19 | 5.2029196210789115 | 0.7377919904702483 | 4.465127630608663 | False |
| 20 | 4.2968814788183884 | 0.7427053747977714 | 3.554176104020617 | False |

## 每seed主终点

| seed | Proposed s | TFOO s | 相对差% | 胜场 | Wilcoxon p |
|---|---|---|---|---|---|
| 4101 | 0.8511244224924497 | 0.9459839808136465 | -10.027607258170224 | 13 | 0.009435653686523438 |
| 4102 | 0.9134134471794786 | 0.8411420729393928 | 8.592053181638093 | 6 | 0.029575347900390625 |
| 4103 | 7.560720106158246 | 0.7689560310519243 | 883.24478914812 | 4 | 0.0003223419189453125 |
| 4104 | 0.7980702277519404 | 0.7669025218483826 | 4.064102674801173 | 7 | 0.15364646911621094 |
| 4105 | 0.8073471688580799 | 0.7961983272040823 | 1.400259366676604 | 12 | 0.8983173370361328 |

Proposed场景等权均值更低的seed数：1/5。

## 主比较cost/s

| 参考 | 候选cost/s | 参考cost/s | 相对差% | 胜场 |
|---|---|---|---|---|
| TFOO | 17.106744833608637 | 7.461895006260083 | 129.25469762382212 | 4 |
| NC | 17.106744833608637 | 36.94896929234326 | -53.70169950274207 | 17 |
| CA | 17.106744833608637 | 10.88314443480691 | 57.185682282201064 | 13 |
| TOO | 17.106744833608637 | 8.495679880018113 | 101.35816173869495 | 13 |

## 跨seed均值及样本标准差

| 组 | 均值s | 样本标准差s |
|---|---|---|
| Proposed | 2.186135074488039 | 3.004830201179013 |
| NC | 4.961494927565457 | 0.08259995654657365 |
| CA | 1.31684462025671 | 0.0 |
| TOO | 0.9763757301512707 | 0.19032980625998416 |
| TFOO | 0.8238365867714859 | 0.07455987905614804 |
| Abl_TwoStage | 1.232539762409016 | 0.8641185120538925 |
| Abl_HoverSAC | 1.6141817854923997 | 0.8030185678729194 |
| Abl_UpperCVX | 0.8904398533815178 | 0.049866792378822425 |

## 仅4101的Alternating

[
  {
    "candidate": "Proposed",
    "reference": "Abl_Alternating",
    "category": "ablation",
    "primary_endpoint": false,
    "scenarios": 20,
    "candidate_mean_s": 0.8511244224924497,
    "reference_mean_s": 1.5607963534607499,
    "relative_difference_percent": -45.468579510372784,
    "wins": 20,
    "pairs": [
      {
        "scenario": "1",
        "candidate_s": 0.9720701842616247,
        "reference_s": 1.7623849237947562,
        "win": true
      },
      {
        "scenario": "2",
        "candidate_s": 0.8021967726598997,
        "reference_s": 1.3917750370556785,
        "win": true
      },
      {
        "scenario": "3",
        "candidate_s": 0.6724414959539258,
        "reference_s": 1.3710271925348159,
        "win": true
      },
      {
        "scenario": "4",
        "candidate_s": 0.8764095181406694,
        "reference_s": 2.267544746058334,
        "win": true
      },
      {
        "scenario": "5",
        "candidate_s": 0.7716719449601408,
        "reference_s": 1.3700147428196983,
        "win": true
      },
      {
        "scenario": "6",
        "candidate_s": 0.7338231288616196,
        "reference_s": 1.3548418079905977,
        "win": true
      },
      {
        "scenario": "7",
        "candidate_s": 0.7527184086273843,
        "reference_s": 1.625627916072978,
        "win": true
      },
      {
        "scenario": "8",
        "candidate_s": 0.9112562267152239,
        "reference_s": 1.408983214031568,
        "win": true
      },
      {
        "scenario": "9",
        "candidate_s": 0.9750007069136175,
        "reference_s": 1.553576982892692,
        "win": true
      },
      {
        "scenario": "10",
        "candidate_s": 1.23828700919022,
        "reference_s": 1.867159982654679,
        "win": true
      },
      {
        "scenario": "11",
        "candidate_s": 1.2225158282891522,
        "reference_s": 2.045092882487999,
        "win": true
      },
      {
        "scenario": "12",
        "candidate_s": 1.0086200570154382,
        "reference_s": 1.9424073499110899,
        "win": true
      },
      {
        "scenario": "13",
        "candidate_s": 0.7568647140864706,
        "reference_s": 1.1870627615062808,
        "win": true
      },
      {
        "scenario": "14",
        "candidate_s": 0.7021434046818776,
        "reference_s": 1.3934946046347108,
        "win": true
      },
      {
        "scenario": "15",
        "candidate_s": 0.7456142062640245,
        "reference_s": 1.5719038791240951,
        "win": true
      },
      {
        "scenario": "16",
        "candidate_s": 0.724563764092689,
        "reference_s": 1.6116763922723876,
        "win": true
      },
      {
        "scenario": "17",
        "candidate_s": 0.8235508051082733,
        "reference_s": 1.811021654636314,
        "win": true
      },
      {
        "scenario": "18",
        "candidate_s": 0.7254535068409165,
        "reference_s": 1.2352049488672785,
        "win": true
      },
      {
        "scenario": "19",
        "candidate_s": 0.8008948323784704,
        "reference_s": 1.204591865675871,
        "win": true
      },
      {
        "scenario": "20",
        "candidate_s": 0.8063919348073588,
        "reference_s": 1.2405341841931816,
        "win": true
      }
    ],
    "candidate_unfinished_per_episode": {
      "total": 1409.0,
      "by_priority": [
        128.0,
        1281.0
      ]
    },
    "reference_unfinished_per_episode": {
      "total": 3833.0,
      "by_priority": [
        395.0,
        3438.0
      ]
    },
    "statistics": {
      "mean_d": -0.7096719309683004,
      "sd_d": 0.2377698256790427,
      "nonzero_differences": 20,
      "ttest_rel": {
        "statistic": -13.347990434334804,
        "pvalue": 4.206000151821033e-11,
        "alternative": "two-sided"
      },
      "wilcoxon": {
        "statistic": 0.0,
        "W_plus": 0.0,
        "pvalue": 1.9073486328125e-06,
        "alternative": "two-sided",
        "zero_method": "wilcox"
      },
      "binomtest": {
        "wins": 20,
        "n": 20,
        "pvalue": 1.9073486328125e-06,
        "alternative": "two-sided"
      }
    }
  }
]

## G1（原始检查结果）

### seed 4101

```json
{
  "gate": "G1",
  "passed": true,
  "required_wins": 12,
  "equal": {
    "scenarios": 20,
    "wins": 17,
    "candidate_mean_s": 1.2631590689120074,
    "reference_mean_s": 1.3093446376449154,
    "relative_improvement": 0.03527380599807617,
    "candidate_unfinished": 2701,
    "reference_unfinished": 4948,
    "pairs": [
      {
        "scenario": "1",
        "candidate_s": 1.0877532188531789,
        "reference_s": 1.1150467869140859,
        "win": true
      },
      {
        "scenario": "2",
        "candidate_s": 1.6157116030273704,
        "reference_s": 1.6444104592272226,
        "win": true
      },
      {
        "scenario": "3",
        "candidate_s": 1.1939639728597455,
        "reference_s": 1.286137873636988,
        "win": true
      },
      {
        "scenario": "4",
        "candidate_s": 1.2667699911436137,
        "reference_s": 1.2751635574093663,
        "win": true
      },
      {
        "scenario": "5",
        "candidate_s": 1.002453195610072,
        "reference_s": 1.0521553494923395,
        "win": true
      },
      {
        "scenario": "6",
        "candidate_s": 1.0483325001785309,
        "reference_s": 1.1384760408483925,
        "win": true
      },
      {
        "scenario": "7",
        "candidate_s": 1.8262278495243192,
        "reference_s": 1.7929975688930762,
        "win": false
      },
      {
        "scenario": "8",
        "candidate_s": 1.2158289414667018,
        "reference_s": 1.3329347947970391,
        "win": true
      },
      {
        "scenario": "9",
        "candidate_s": 1.0759518403852775,
        "reference_s": 1.1692782937736506,
        "win": true
      },
      {
        "scenario": "10",
        "candidate_s": 1.1266000168496686,
        "reference_s": 1.173628857873011,
        "win": true
      },
      {
        "scenario": "11",
        "candidate_s": 1.3961334137376493,
        "reference_s": 1.4235512864691027,
        "win": true
      },
      {
        "scenario": "12",
        "candidate_s": 1.302090489329355,
        "reference_s": 1.3679387915986276,
        "win": true
      },
      {
        "scenario": "13",
        "candidate_s": 2.1552128794491323,
        "reference_s": 2.1435304875514403,
        "win": false
      },
      {
        "scenario": "14",
        "candidate_s": 1.2709871912815802,
        "reference_s": 1.3207359705611799,
        "win": true
      },
      {
        "scenario": "15",
        "candidate_s": 0.8890164241733861,
        "reference_s": 0.9609525732166311,
        "win": true
      },
      {
        "scenario": "16",
        "candidate_s": 1.229665576995397,
        "reference_s": 1.282487632713281,
        "win": true
      },
      {
        "scenario": "17",
        "candidate_s": 1.0598654805191055,
        "reference_s": 1.117563702734941,
        "win": true
      },
      {
        "scenario": "18",
        "candidate_s": 1.029724181415606,
        "reference_s": 1.1213220522896896,
        "win": true
      },
      {
        "scenario": "19",
        "candidate_s": 1.0992297138937326,
        "reference_s": 1.0874324749192534,
        "win": false
      },
      {
        "scenario": "20",
        "candidate_s": 1.3716628975467293,
        "reference_s": 1.3811481979789915,
        "win": true
      }
    ]
  },
  "local": {
    "scenarios": 20,
    "wins": 20,
    "candidate_mean_s": 1.2631590689120074,
    "reference_mean_s": 5.7048786774891385,
    "relative_improvement": 0.7785826587520077,
    "candidate_unfinished": 2701,
    "reference_unfinished": 27090,
    "pairs": [
      {
        "scenario": "1",
        "candidate_s": 1.0877532188531789,
        "reference_s": 6.626713958216207,
        "win": true
      },
      {
        "scenario": "2",
        "candidate_s": 1.6157116030273704,
        "reference_s": 6.071214362943147,
        "win": true
      },
      {
        "scenario": "3",
        "candidate_s": 1.1939639728597455,
        "reference_s": 6.323320308642851,
        "win": true
      },
      {
        "scenario": "4",
        "candidate_s": 1.2667699911436137,
        "reference_s": 7.9137888181012945,
        "win": true
      },
      {
        "scenario": "5",
        "candidate_s": 1.002453195610072,
        "reference_s": 5.147813838840298,
        "win": true
      },
      {
        "scenario": "6",
        "candidate_s": 1.0483325001785309,
        "reference_s": 6.176136542169513,
        "win": true
      },
      {
        "scenario": "7",
        "candidate_s": 1.8262278495243192,
        "reference_s": 3.8867722488441734,
        "win": true
      },
      {
        "scenario": "8",
        "candidate_s": 1.2158289414667018,
        "reference_s": 10.172360675039249,
        "win": true
      },
      {
        "scenario": "9",
        "candidate_s": 1.0759518403852775,
        "reference_s": 7.7487416580667166,
        "win": true
      },
      {
        "scenario": "10",
        "candidate_s": 1.1266000168496686,
        "reference_s": 6.033192451346559,
        "win": true
      },
      {
        "scenario": "11",
        "candidate_s": 1.3961334137376493,
        "reference_s": 2.600165116011948,
        "win": true
      },
      {
        "scenario": "12",
        "candidate_s": 1.302090489329355,
        "reference_s": 7.064406779661031,
        "win": true
      },
      {
        "scenario": "13",
        "candidate_s": 2.1552128794491323,
        "reference_s": 6.489141970433793,
        "win": true
      },
      {
        "scenario": "14",
        "candidate_s": 1.2709871912815802,
        "reference_s": 6.132017550067241,
        "win": true
      },
      {
        "scenario": "15",
        "candidate_s": 0.8890164241733861,
        "reference_s": 4.482054535753929,
        "win": true
      },
      {
        "scenario": "16",
        "candidate_s": 1.229665576995397,
        "reference_s": 3.2290551682240287,
        "win": true
      },
      {
        "scenario": "17",
        "candidate_s": 1.0598654805191055,
        "reference_s": 3.7194262801626836,
        "win": true
      },
      {
        "scenario": "18",
        "candidate_s": 1.029724181415606,
        "reference_s": 4.9751132747775,
        "win": true
      },
      {
        "scenario": "19",
        "candidate_s": 1.0992297138937326,
        "reference_s": 2.2829779632434932,
        "win": true
      },
      {
        "scenario": "20",
        "candidate_s": 1.3716628975467293,
        "reference_s": 7.023160049237102,
        "win": true
      }
    ]
  },
  "action": "continue",
  "unfinished": {
    "rule": "report_only_not_a_pass_condition",
    "per_episode": {
      "candidate": {
        "total": 2701.0,
        "by_priority": [
          265.0,
          2436.0
        ]
      },
      "hover_equal": {
        "total": 2474.0,
        "by_priority": [
          270.0,
          2204.0
        ]
      },
      "hover_local": {
        "total": 13545.0,
        "by_priority": [
          253.0,
          13292.0
        ]
      }
    },
    "candidate_within_equal": false
  }
}
```

### seed 4102

```json
{
  "gate": "G1",
  "passed": false,
  "required_wins": 12,
  "equal": {
    "scenarios": 20,
    "wins": 6,
    "candidate_mean_s": 4.049713563872141,
    "reference_mean_s": 1.3093446376449154,
    "relative_improvement": -2.092931721289405,
    "candidate_unfinished": 9647,
    "reference_unfinished": 4948,
    "pairs": [
      {
        "scenario": "1",
        "candidate_s": 1.0769280678230053,
        "reference_s": 1.1150467869140859,
        "win": true
      },
      {
        "scenario": "2",
        "candidate_s": 1.957219882774707,
        "reference_s": 1.6444104592272226,
        "win": false
      },
      {
        "scenario": "3",
        "candidate_s": 1.1756442989942328,
        "reference_s": 1.286137873636988,
        "win": true
      },
      {
        "scenario": "4",
        "candidate_s": 1.2641245036139759,
        "reference_s": 1.2751635574093663,
        "win": true
      },
      {
        "scenario": "5",
        "candidate_s": 10.430198368448849,
        "reference_s": 1.0521553494923395,
        "win": false
      },
      {
        "scenario": "6",
        "candidate_s": 10.87548096836394,
        "reference_s": 1.1384760408483925,
        "win": false
      },
      {
        "scenario": "7",
        "candidate_s": 6.807377636626808,
        "reference_s": 1.7929975688930762,
        "win": false
      },
      {
        "scenario": "8",
        "candidate_s": 1.1935089145548354,
        "reference_s": 1.3329347947970391,
        "win": true
      },
      {
        "scenario": "9",
        "candidate_s": 11.717027863777094,
        "reference_s": 1.1692782937736506,
        "win": false
      },
      {
        "scenario": "10",
        "candidate_s": 1.1835898227976063,
        "reference_s": 1.173628857873011,
        "win": false
      },
      {
        "scenario": "11",
        "candidate_s": 1.4652366184240753,
        "reference_s": 1.4235512864691027,
        "win": false
      },
      {
        "scenario": "12",
        "candidate_s": 1.4560293372374566,
        "reference_s": 1.3679387915986276,
        "win": false
      },
      {
        "scenario": "13",
        "candidate_s": 2.1707330590077194,
        "reference_s": 2.1435304875514403,
        "win": false
      },
      {
        "scenario": "14",
        "candidate_s": 1.3149600169839422,
        "reference_s": 1.3207359705611799,
        "win": true
      },
      {
        "scenario": "15",
        "candidate_s": 9.712519286507373,
        "reference_s": 0.9609525732166311,
        "win": false
      },
      {
        "scenario": "16",
        "candidate_s": 4.577048919489682,
        "reference_s": 1.282487632713281,
        "win": false
      },
      {
        "scenario": "17",
        "candidate_s": 8.535695089176658,
        "reference_s": 1.117563702734941,
        "win": false
      },
      {
        "scenario": "18",
        "candidate_s": 1.3290555303019154,
        "reference_s": 1.1213220522896896,
        "win": false
      },
      {
        "scenario": "19",
        "candidate_s": 1.05034441363935,
        "reference_s": 1.0874324749192534,
        "win": true
      },
      {
        "scenario": "20",
        "candidate_s": 1.7015486788996053,
        "reference_s": 1.3811481979789915,
        "win": false
      }
    ]
  },
  "local": {
    "scenarios": 20,
    "wins": 13,
    "candidate_mean_s": 4.049713563872141,
    "reference_mean_s": 5.7048786774891385,
    "relative_improvement": 0.2901315185103074,
    "candidate_unfinished": 9647,
    "reference_unfinished": 27090,
    "pairs": [
      {
        "scenario": "1",
        "candidate_s": 1.0769280678230053,
        "reference_s": 6.626713958216207,
        "win": true
      },
      {
        "scenario": "2",
        "candidate_s": 1.957219882774707,
        "reference_s": 6.071214362943147,
        "win": true
      },
      {
        "scenario": "3",
        "candidate_s": 1.1756442989942328,
        "reference_s": 6.323320308642851,
        "win": true
      },
      {
        "scenario": "4",
        "candidate_s": 1.2641245036139759,
        "reference_s": 7.9137888181012945,
        "win": true
      },
      {
        "scenario": "5",
        "candidate_s": 10.430198368448849,
        "reference_s": 5.147813838840298,
        "win": false
      },
      {
        "scenario": "6",
        "candidate_s": 10.87548096836394,
        "reference_s": 6.176136542169513,
        "win": false
      },
      {
        "scenario": "7",
        "candidate_s": 6.807377636626808,
        "reference_s": 3.8867722488441734,
        "win": false
      },
      {
        "scenario": "8",
        "candidate_s": 1.1935089145548354,
        "reference_s": 10.172360675039249,
        "win": true
      },
      {
        "scenario": "9",
        "candidate_s": 11.717027863777094,
        "reference_s": 7.7487416580667166,
        "win": false
      },
      {
        "scenario": "10",
        "candidate_s": 1.1835898227976063,
        "reference_s": 6.033192451346559,
        "win": true
      },
      {
        "scenario": "11",
        "candidate_s": 1.4652366184240753,
        "reference_s": 2.600165116011948,
        "win": true
      },
      {
        "scenario": "12",
        "candidate_s": 1.4560293372374566,
        "reference_s": 7.064406779661031,
        "win": true
      },
      {
        "scenario": "13",
        "candidate_s": 2.1707330590077194,
        "reference_s": 6.489141970433793,
        "win": true
      },
      {
        "scenario": "14",
        "candidate_s": 1.3149600169839422,
        "reference_s": 6.132017550067241,
        "win": true
      },
      {
        "scenario": "15",
        "candidate_s": 9.712519286507373,
        "reference_s": 4.482054535753929,
        "win": false
      },
      {
        "scenario": "16",
        "candidate_s": 4.577048919489682,
        "reference_s": 3.2290551682240287,
        "win": false
      },
      {
        "scenario": "17",
        "candidate_s": 8.535695089176658,
        "reference_s": 3.7194262801626836,
        "win": false
      },
      {
        "scenario": "18",
        "candidate_s": 1.3290555303019154,
        "reference_s": 4.9751132747775,
        "win": true
      },
      {
        "scenario": "19",
        "candidate_s": 1.05034441363935,
        "reference_s": 2.2829779632434932,
        "win": true
      },
      {
        "scenario": "20",
        "candidate_s": 1.7015486788996053,
        "reference_s": 7.023160049237102,
        "win": true
      }
    ]
  },
  "action": "pause_before_PPO",
  "unfinished": {
    "rule": "report_only_not_a_pass_condition",
    "per_episode": {
      "candidate": {
        "total": 9647.0,
        "by_priority": [
          264.0,
          9383.0
        ]
      },
      "hover_equal": {
        "total": 2474.0,
        "by_priority": [
          270.0,
          2204.0
        ]
      },
      "hover_local": {
        "total": 13545.0,
        "by_priority": [
          253.0,
          13292.0
        ]
      }
    },
    "candidate_within_equal": false
  }
}
```

### seed 4103

```json
{
  "gate": "G1",
  "passed": true,
  "required_wins": 12,
  "equal": {
    "scenarios": 20,
    "wins": 19,
    "candidate_mean_s": 1.2335955866077004,
    "reference_mean_s": 1.3093446376449154,
    "relative_improvement": 0.05785264540698987,
    "candidate_unfinished": 2600,
    "reference_unfinished": 4948,
    "pairs": [
      {
        "scenario": "1",
        "candidate_s": 1.0282696771775008,
        "reference_s": 1.1150467869140859,
        "win": true
      },
      {
        "scenario": "2",
        "candidate_s": 1.5927914983212932,
        "reference_s": 1.6444104592272226,
        "win": true
      },
      {
        "scenario": "3",
        "candidate_s": 1.1324018563710505,
        "reference_s": 1.286137873636988,
        "win": true
      },
      {
        "scenario": "4",
        "candidate_s": 1.1933877096248915,
        "reference_s": 1.2751635574093663,
        "win": true
      },
      {
        "scenario": "5",
        "candidate_s": 0.9820837490463041,
        "reference_s": 1.0521553494923395,
        "win": true
      },
      {
        "scenario": "6",
        "candidate_s": 1.0220195672355912,
        "reference_s": 1.1384760408483925,
        "win": true
      },
      {
        "scenario": "7",
        "candidate_s": 1.880220907519574,
        "reference_s": 1.7929975688930762,
        "win": false
      },
      {
        "scenario": "8",
        "candidate_s": 1.148566102265087,
        "reference_s": 1.3329347947970391,
        "win": true
      },
      {
        "scenario": "9",
        "candidate_s": 1.0421891984864116,
        "reference_s": 1.1692782937736506,
        "win": true
      },
      {
        "scenario": "10",
        "candidate_s": 1.1166278188098513,
        "reference_s": 1.173628857873011,
        "win": true
      },
      {
        "scenario": "11",
        "candidate_s": 1.3824517574086803,
        "reference_s": 1.4235512864691027,
        "win": true
      },
      {
        "scenario": "12",
        "candidate_s": 1.2604721549636797,
        "reference_s": 1.3679387915986276,
        "win": true
      },
      {
        "scenario": "13",
        "candidate_s": 2.0761018052842286,
        "reference_s": 2.1435304875514403,
        "win": true
      },
      {
        "scenario": "14",
        "candidate_s": 1.2690241313424389,
        "reference_s": 1.3207359705611799,
        "win": true
      },
      {
        "scenario": "15",
        "candidate_s": 0.8697861540901841,
        "reference_s": 0.9609525732166311,
        "win": true
      },
      {
        "scenario": "16",
        "candidate_s": 1.2269534527150194,
        "reference_s": 1.282487632713281,
        "win": true
      },
      {
        "scenario": "17",
        "candidate_s": 1.0126830600307548,
        "reference_s": 1.117563702734941,
        "win": true
      },
      {
        "scenario": "18",
        "candidate_s": 1.0650380515271893,
        "reference_s": 1.1213220522896896,
        "win": true
      },
      {
        "scenario": "19",
        "candidate_s": 1.0340497899219707,
        "reference_s": 1.0874324749192534,
        "win": true
      },
      {
        "scenario": "20",
        "candidate_s": 1.336793290012309,
        "reference_s": 1.3811481979789915,
        "win": true
      }
    ]
  },
  "local": {
    "scenarios": 20,
    "wins": 20,
    "candidate_mean_s": 1.2335955866077004,
    "reference_mean_s": 5.7048786774891385,
    "relative_improvement": 0.7837647991577208,
    "candidate_unfinished": 2600,
    "reference_unfinished": 27090,
    "pairs": [
      {
        "scenario": "1",
        "candidate_s": 1.0282696771775008,
        "reference_s": 6.626713958216207,
        "win": true
      },
      {
        "scenario": "2",
        "candidate_s": 1.5927914983212932,
        "reference_s": 6.071214362943147,
        "win": true
      },
      {
        "scenario": "3",
        "candidate_s": 1.1324018563710505,
        "reference_s": 6.323320308642851,
        "win": true
      },
      {
        "scenario": "4",
        "candidate_s": 1.1933877096248915,
        "reference_s": 7.9137888181012945,
        "win": true
      },
      {
        "scenario": "5",
        "candidate_s": 0.9820837490463041,
        "reference_s": 5.147813838840298,
        "win": true
      },
      {
        "scenario": "6",
        "candidate_s": 1.0220195672355912,
        "reference_s": 6.176136542169513,
        "win": true
      },
      {
        "scenario": "7",
        "candidate_s": 1.880220907519574,
        "reference_s": 3.8867722488441734,
        "win": true
      },
      {
        "scenario": "8",
        "candidate_s": 1.148566102265087,
        "reference_s": 10.172360675039249,
        "win": true
      },
      {
        "scenario": "9",
        "candidate_s": 1.0421891984864116,
        "reference_s": 7.7487416580667166,
        "win": true
      },
      {
        "scenario": "10",
        "candidate_s": 1.1166278188098513,
        "reference_s": 6.033192451346559,
        "win": true
      },
      {
        "scenario": "11",
        "candidate_s": 1.3824517574086803,
        "reference_s": 2.600165116011948,
        "win": true
      },
      {
        "scenario": "12",
        "candidate_s": 1.2604721549636797,
        "reference_s": 7.064406779661031,
        "win": true
      },
      {
        "scenario": "13",
        "candidate_s": 2.0761018052842286,
        "reference_s": 6.489141970433793,
        "win": true
      },
      {
        "scenario": "14",
        "candidate_s": 1.2690241313424389,
        "reference_s": 6.132017550067241,
        "win": true
      },
      {
        "scenario": "15",
        "candidate_s": 0.8697861540901841,
        "reference_s": 4.482054535753929,
        "win": true
      },
      {
        "scenario": "16",
        "candidate_s": 1.2269534527150194,
        "reference_s": 3.2290551682240287,
        "win": true
      },
      {
        "scenario": "17",
        "candidate_s": 1.0126830600307548,
        "reference_s": 3.7194262801626836,
        "win": true
      },
      {
        "scenario": "18",
        "candidate_s": 1.0650380515271893,
        "reference_s": 4.9751132747775,
        "win": true
      },
      {
        "scenario": "19",
        "candidate_s": 1.0340497899219707,
        "reference_s": 2.2829779632434932,
        "win": true
      },
      {
        "scenario": "20",
        "candidate_s": 1.336793290012309,
        "reference_s": 7.023160049237102,
        "win": true
      }
    ]
  },
  "action": "continue",
  "unfinished": {
    "rule": "report_only_not_a_pass_condition",
    "per_episode": {
      "candidate": {
        "total": 2600.0,
        "by_priority": [
          267.0,
          2333.0
        ]
      },
      "hover_equal": {
        "total": 2474.0,
        "by_priority": [
          270.0,
          2204.0
        ]
      },
      "hover_local": {
        "total": 13545.0,
        "by_priority": [
          253.0,
          13292.0
        ]
      }
    },
    "candidate_within_equal": false
  }
}
```

### seed 4104

```json
{
  "gate": "G1",
  "passed": true,
  "required_wins": 12,
  "equal": {
    "scenarios": 20,
    "wins": 20,
    "candidate_mean_s": 1.2540270696025186,
    "reference_mean_s": 1.3093446376449154,
    "relative_improvement": 0.042248287007074836,
    "candidate_unfinished": 2556,
    "reference_unfinished": 4948,
    "pairs": [
      {
        "scenario": "1",
        "candidate_s": 1.0721224966477774,
        "reference_s": 1.1150467869140859,
        "win": true
      },
      {
        "scenario": "2",
        "candidate_s": 1.6031596767768734,
        "reference_s": 1.6444104592272226,
        "win": true
      },
      {
        "scenario": "3",
        "candidate_s": 1.2057355658687237,
        "reference_s": 1.286137873636988,
        "win": true
      },
      {
        "scenario": "4",
        "candidate_s": 1.2657972173813643,
        "reference_s": 1.2751635574093663,
        "win": true
      },
      {
        "scenario": "5",
        "candidate_s": 0.9911526498033929,
        "reference_s": 1.0521553494923395,
        "win": true
      },
      {
        "scenario": "6",
        "candidate_s": 1.0354524030564856,
        "reference_s": 1.1384760408483925,
        "win": true
      },
      {
        "scenario": "7",
        "candidate_s": 1.7587400401905509,
        "reference_s": 1.7929975688930762,
        "win": true
      },
      {
        "scenario": "8",
        "candidate_s": 1.189871607983852,
        "reference_s": 1.3329347947970391,
        "win": true
      },
      {
        "scenario": "9",
        "candidate_s": 1.0582084623322985,
        "reference_s": 1.1692782937736506,
        "win": true
      },
      {
        "scenario": "10",
        "candidate_s": 1.1595635934735598,
        "reference_s": 1.173628857873011,
        "win": true
      },
      {
        "scenario": "11",
        "candidate_s": 1.4134806455318152,
        "reference_s": 1.4235512864691027,
        "win": true
      },
      {
        "scenario": "12",
        "candidate_s": 1.2982684835857918,
        "reference_s": 1.3679387915986276,
        "win": true
      },
      {
        "scenario": "13",
        "candidate_s": 2.085251534422322,
        "reference_s": 2.1435304875514403,
        "win": true
      },
      {
        "scenario": "14",
        "candidate_s": 1.2860944023777514,
        "reference_s": 1.3207359705611799,
        "win": true
      },
      {
        "scenario": "15",
        "candidate_s": 0.8861526481996849,
        "reference_s": 0.9609525732166311,
        "win": true
      },
      {
        "scenario": "16",
        "candidate_s": 1.2232447131657367,
        "reference_s": 1.282487632713281,
        "win": true
      },
      {
        "scenario": "17",
        "candidate_s": 1.029147324700709,
        "reference_s": 1.117563702734941,
        "win": true
      },
      {
        "scenario": "18",
        "candidate_s": 1.068522075751133,
        "reference_s": 1.1213220522896896,
        "win": true
      },
      {
        "scenario": "19",
        "candidate_s": 1.0709963700803165,
        "reference_s": 1.0874324749192534,
        "win": true
      },
      {
        "scenario": "20",
        "candidate_s": 1.379579480720236,
        "reference_s": 1.3811481979789915,
        "win": true
      }
    ]
  },
  "local": {
    "scenarios": 20,
    "wins": 20,
    "candidate_mean_s": 1.2540270696025186,
    "reference_mean_s": 5.7048786774891385,
    "relative_improvement": 0.7801833938115494,
    "candidate_unfinished": 2556,
    "reference_unfinished": 27090,
    "pairs": [
      {
        "scenario": "1",
        "candidate_s": 1.0721224966477774,
        "reference_s": 6.626713958216207,
        "win": true
      },
      {
        "scenario": "2",
        "candidate_s": 1.6031596767768734,
        "reference_s": 6.071214362943147,
        "win": true
      },
      {
        "scenario": "3",
        "candidate_s": 1.2057355658687237,
        "reference_s": 6.323320308642851,
        "win": true
      },
      {
        "scenario": "4",
        "candidate_s": 1.2657972173813643,
        "reference_s": 7.9137888181012945,
        "win": true
      },
      {
        "scenario": "5",
        "candidate_s": 0.9911526498033929,
        "reference_s": 5.147813838840298,
        "win": true
      },
      {
        "scenario": "6",
        "candidate_s": 1.0354524030564856,
        "reference_s": 6.176136542169513,
        "win": true
      },
      {
        "scenario": "7",
        "candidate_s": 1.7587400401905509,
        "reference_s": 3.8867722488441734,
        "win": true
      },
      {
        "scenario": "8",
        "candidate_s": 1.189871607983852,
        "reference_s": 10.172360675039249,
        "win": true
      },
      {
        "scenario": "9",
        "candidate_s": 1.0582084623322985,
        "reference_s": 7.7487416580667166,
        "win": true
      },
      {
        "scenario": "10",
        "candidate_s": 1.1595635934735598,
        "reference_s": 6.033192451346559,
        "win": true
      },
      {
        "scenario": "11",
        "candidate_s": 1.4134806455318152,
        "reference_s": 2.600165116011948,
        "win": true
      },
      {
        "scenario": "12",
        "candidate_s": 1.2982684835857918,
        "reference_s": 7.064406779661031,
        "win": true
      },
      {
        "scenario": "13",
        "candidate_s": 2.085251534422322,
        "reference_s": 6.489141970433793,
        "win": true
      },
      {
        "scenario": "14",
        "candidate_s": 1.2860944023777514,
        "reference_s": 6.132017550067241,
        "win": true
      },
      {
        "scenario": "15",
        "candidate_s": 0.8861526481996849,
        "reference_s": 4.482054535753929,
        "win": true
      },
      {
        "scenario": "16",
        "candidate_s": 1.2232447131657367,
        "reference_s": 3.2290551682240287,
        "win": true
      },
      {
        "scenario": "17",
        "candidate_s": 1.029147324700709,
        "reference_s": 3.7194262801626836,
        "win": true
      },
      {
        "scenario": "18",
        "candidate_s": 1.068522075751133,
        "reference_s": 4.9751132747775,
        "win": true
      },
      {
        "scenario": "19",
        "candidate_s": 1.0709963700803165,
        "reference_s": 2.2829779632434932,
        "win": true
      },
      {
        "scenario": "20",
        "candidate_s": 1.379579480720236,
        "reference_s": 7.023160049237102,
        "win": true
      }
    ]
  },
  "action": "continue",
  "unfinished": {
    "rule": "report_only_not_a_pass_condition",
    "per_episode": {
      "candidate": {
        "total": 2556.0,
        "by_priority": [
          263.0,
          2293.0
        ]
      },
      "hover_equal": {
        "total": 2474.0,
        "by_priority": [
          270.0,
          2204.0
        ]
      },
      "hover_local": {
        "total": 13545.0,
        "by_priority": [
          253.0,
          13292.0
        ]
      }
    },
    "candidate_within_equal": false
  }
}
```

### seed 4105

```json
{
  "gate": "G1",
  "passed": false,
  "required_wins": 12,
  "equal": {
    "scenarios": 20,
    "wins": 15,
    "candidate_mean_s": 1.3122700022100928,
    "reference_mean_s": 1.3093446376449154,
    "relative_improvement": -0.0022342204497352025,
    "candidate_unfinished": 3522,
    "reference_unfinished": 4948,
    "pairs": [
      {
        "scenario": "1",
        "candidate_s": 1.0333823550614951,
        "reference_s": 1.1150467869140859,
        "win": true
      },
      {
        "scenario": "2",
        "candidate_s": 1.7153644795993828,
        "reference_s": 1.6444104592272226,
        "win": false
      },
      {
        "scenario": "3",
        "candidate_s": 1.1378792794571935,
        "reference_s": 1.286137873636988,
        "win": true
      },
      {
        "scenario": "4",
        "candidate_s": 1.2044196211753266,
        "reference_s": 1.2751635574093663,
        "win": true
      },
      {
        "scenario": "5",
        "candidate_s": 0.9911966664710372,
        "reference_s": 1.0521553494923395,
        "win": true
      },
      {
        "scenario": "6",
        "candidate_s": 1.0332671570377754,
        "reference_s": 1.1384760408483925,
        "win": true
      },
      {
        "scenario": "7",
        "candidate_s": 1.7676817357822412,
        "reference_s": 1.7929975688930762,
        "win": true
      },
      {
        "scenario": "8",
        "candidate_s": 1.1503335949764566,
        "reference_s": 1.3329347947970391,
        "win": true
      },
      {
        "scenario": "9",
        "candidate_s": 1.0702690058479523,
        "reference_s": 1.1692782937736506,
        "win": true
      },
      {
        "scenario": "10",
        "candidate_s": 1.149778145974333,
        "reference_s": 1.173628857873011,
        "win": true
      },
      {
        "scenario": "11",
        "candidate_s": 1.4626565012634927,
        "reference_s": 1.4235512864691027,
        "win": false
      },
      {
        "scenario": "12",
        "candidate_s": 1.3611197139478628,
        "reference_s": 1.3679387915986276,
        "win": true
      },
      {
        "scenario": "13",
        "candidate_s": 2.9489989885974737,
        "reference_s": 2.1435304875514403,
        "win": false
      },
      {
        "scenario": "14",
        "candidate_s": 1.5492449225107923,
        "reference_s": 1.3207359705611799,
        "win": false
      },
      {
        "scenario": "15",
        "candidate_s": 0.8830134536907527,
        "reference_s": 0.9609525732166311,
        "win": true
      },
      {
        "scenario": "16",
        "candidate_s": 1.221368646396858,
        "reference_s": 1.282487632713281,
        "win": true
      },
      {
        "scenario": "17",
        "candidate_s": 1.0408171771029466,
        "reference_s": 1.117563702734941,
        "win": true
      },
      {
        "scenario": "18",
        "candidate_s": 1.0858087599161266,
        "reference_s": 1.1213220522896896,
        "win": true
      },
      {
        "scenario": "19",
        "candidate_s": 1.035281676051104,
        "reference_s": 1.0874324749192534,
        "win": true
      },
      {
        "scenario": "20",
        "candidate_s": 1.4035181633412535,
        "reference_s": 1.3811481979789915,
        "win": false
      }
    ]
  },
  "local": {
    "scenarios": 20,
    "wins": 20,
    "candidate_mean_s": 1.3122700022100928,
    "reference_mean_s": 5.7048786774891385,
    "relative_improvement": 0.7699740737014839,
    "candidate_unfinished": 3522,
    "reference_unfinished": 27090,
    "pairs": [
      {
        "scenario": "1",
        "candidate_s": 1.0333823550614951,
        "reference_s": 6.626713958216207,
        "win": true
      },
      {
        "scenario": "2",
        "candidate_s": 1.7153644795993828,
        "reference_s": 6.071214362943147,
        "win": true
      },
      {
        "scenario": "3",
        "candidate_s": 1.1378792794571935,
        "reference_s": 6.323320308642851,
        "win": true
      },
      {
        "scenario": "4",
        "candidate_s": 1.2044196211753266,
        "reference_s": 7.9137888181012945,
        "win": true
      },
      {
        "scenario": "5",
        "candidate_s": 0.9911966664710372,
        "reference_s": 5.147813838840298,
        "win": true
      },
      {
        "scenario": "6",
        "candidate_s": 1.0332671570377754,
        "reference_s": 6.176136542169513,
        "win": true
      },
      {
        "scenario": "7",
        "candidate_s": 1.7676817357822412,
        "reference_s": 3.8867722488441734,
        "win": true
      },
      {
        "scenario": "8",
        "candidate_s": 1.1503335949764566,
        "reference_s": 10.172360675039249,
        "win": true
      },
      {
        "scenario": "9",
        "candidate_s": 1.0702690058479523,
        "reference_s": 7.7487416580667166,
        "win": true
      },
      {
        "scenario": "10",
        "candidate_s": 1.149778145974333,
        "reference_s": 6.033192451346559,
        "win": true
      },
      {
        "scenario": "11",
        "candidate_s": 1.4626565012634927,
        "reference_s": 2.600165116011948,
        "win": true
      },
      {
        "scenario": "12",
        "candidate_s": 1.3611197139478628,
        "reference_s": 7.064406779661031,
        "win": true
      },
      {
        "scenario": "13",
        "candidate_s": 2.9489989885974737,
        "reference_s": 6.489141970433793,
        "win": true
      },
      {
        "scenario": "14",
        "candidate_s": 1.5492449225107923,
        "reference_s": 6.132017550067241,
        "win": true
      },
      {
        "scenario": "15",
        "candidate_s": 0.8830134536907527,
        "reference_s": 4.482054535753929,
        "win": true
      },
      {
        "scenario": "16",
        "candidate_s": 1.221368646396858,
        "reference_s": 3.2290551682240287,
        "win": true
      },
      {
        "scenario": "17",
        "candidate_s": 1.0408171771029466,
        "reference_s": 3.7194262801626836,
        "win": true
      },
      {
        "scenario": "18",
        "candidate_s": 1.0858087599161266,
        "reference_s": 4.9751132747775,
        "win": true
      },
      {
        "scenario": "19",
        "candidate_s": 1.035281676051104,
        "reference_s": 2.2829779632434932,
        "win": true
      },
      {
        "scenario": "20",
        "candidate_s": 1.4035181633412535,
        "reference_s": 7.023160049237102,
        "win": true
      }
    ]
  },
  "action": "pause_before_PPO",
  "unfinished": {
    "rule": "report_only_not_a_pass_condition",
    "per_episode": {
      "candidate": {
        "total": 3522.0,
        "by_priority": [
          270.0,
          3252.0
        ]
      },
      "hover_equal": {
        "total": 2474.0,
        "by_priority": [
          270.0,
          2204.0
        ]
      },
      "hover_local": {
        "total": 13545.0,
        "by_priority": [
          253.0,
          13292.0
        ]
      }
    },
    "candidate_within_equal": false
  }
}
```

## 第三阶段每cycle训练截断时延均值

| seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 4101 | 0.9555016576594599 | 0.959105403225965 | 0.9858260507385145 | 0.9301807475694058 | 0.8551648214160885 | 0.8902548110934039 | 1.0137436074308321 | 0.833722843365074 | 0.904817172524174 | 0.8750504691367885 |
| 4102 | 4.56082029978258 | 1.267618035151656 | 1.0303367733611526 | 0.978222131248937 | 0.901785418122388 | 0.9341526155953932 | 0.9823592850386769 | 1.0255009840092628 | 1.0662655415993754 | 0.854430896865993 |
| 4103 | 0.8538041190749557 | 0.8929619567920938 | 0.7813592435547744 | 0.8188113641898896 | 0.7622005940361707 | 0.8493788715906959 | 0.8378443244234897 | 0.8105614348003645 | 0.8393920823501386 | 2.1765432792422597 |
| 4104 | 1.0636829310378408 | 0.9474282653016521 | 0.8733580095775949 | 0.800569367486841 | 0.9835933256234787 | 0.888086707359047 | 0.8046641726748609 | 0.7238240284909607 | 0.8313731974479813 | 0.7397451204098507 |
| 4105 | 0.896705206667046 | 0.9596474677082206 | 0.9714391674406133 | 1.0638001274566737 | 0.7995567334112247 | 0.9735088296014492 | 0.9126007031859669 | 0.9492399176773312 | 0.8362576928012105 | 0.7934096295412487 |

## 每步实测墙钟

| seed | 步骤 | 秒 |
|---|---|---|
| None | validation | 119.03670050000001 |
| None | validation_check | 1.99920809999999 |
| None | RP-W | 1.992780499999995 |
| None | RP-S2 | 194.2456917 |
| None | RP-S3 | 105.02618910000001 |
| None | RP-T | 163.72535520000002 |
| None | RP-REF | 28.847659999999905 |
| 4102 | reference | 639.2486046999999 |
| 4102 | stage1 | 14241.4202481 |
| 4102 | stage2 | 4166.973567300001 |
| 4102 | stage3 | 3520.6119131999985 |
| 4102 | baselines | 3971.784667 |
| 4102 | test | 2490.013923899998 |
| 4103 | reference | 586.3895402000016 |
| 4103 | stage1 | 10961.762414699999 |
| 4103 | stage2 | 2690.917072700002 |
| 4103 | stage3 | 2937.6100326000014 |
| 4103 | baselines | 3974.7364871000027 |
| 4103 | test | 2699.6804955000043 |
| 4104 | reference | 584.5319776000033 |
| 4104 | stage1 | 16445.877860200002 |
| 4104 | stage2 | 2805.417653800003 |
| 4104 | stage3 | 3024.710041600003 |
| 4104 | baselines | 4039.3930884 |
| 4104 | test | 2061.8538688000117 |
| 4105 | reference | 675.464304699999 |
| 4105 | stage1 | 12234.703220099997 |
| 4105 | stage2 | 2786.5228418999905 |
| 4105 | stage3 | 2952.7097021999944 |
| 4105 | baselines | 3940.0246805999923 |
| 4105 | test | 2145.030444899996 |

## 磁盘占用（文件字节数之和）

| seed | 字节 |
|---|---|
| 4102 | 24253680101 |
| 4103 | 24309708414 |
| 4104 | 24252275943 |
| 4105 | 24274743949 |
| 4101 | 49488325851 |

