# Fixed-workload performance comparison

Raw repetitions and medians describe these fixed workloads. The small repetition count does not establish tail latency; no P95 claim is made.

RSS peaks are sampled process memory; CUDA allocated/reserved peaks include the resident model. These are observed maxima across repetitions, not memory deltas.

Data identity and research-code identity are recorded separately. Production source changes are allowed; fixture, initial state, configuration, shared workloads and update counters must match exactly.

- Fixture SHA256: `e6e203f92da25c4688424c3552b0e7373f745df72664d7d0b1a1401cb5f55887`
- Initial state SHA256: `99775391e05bd54f44acf4c1cdbd86e26ff7b0e31a6405c56f3792257ffa4444`
- Configuration SHA256: `4f991c15ad29ba703590a1b522b9a926d4fb51ef41c3759e051e2964421045ab`

| Run | Research source manifest SHA256 | Critic microbatch |
| --- | --- | ---: |
| baseline | 21da783a9e66b1f3c8ffb735bc1976b1fe84269c923896bbe0a1319def459a48 | 16 |
| current | 1c549d8d9c0dffd0ac0b2e88e332aa36daad915591659693c406e059ed7e4c57 | 16 |
| candidate_1 | 1c549d8d9c0dffd0ac0b2e88e332aa36daad915591659693c406e059ed7e4c57 | 64 |
| candidate_2 | 1c549d8d9c0dffd0ac0b2e88e332aa36daad915591659693c406e059ed7e4c57 | 128 |

## baseline → current


Times are seconds: **median [raw repetitions]**. Speedup is before/after median. Memory columns are before → after in MiB.

| Case | Before seconds | After seconds | Speedup | Repeats | Peak RSS | Peak allocated | Peak reserved |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| actor.e1.h1800.backlog | 4.9873704 [4.9873704, 5.1421966, 4.9047598] | 3.6376744 [3.5996755, 3.6812848, 3.6376744] | 1.3710× | 3/3 | 2366.88 → 2410.21 | 553.39 → 555.39 | 1128.00 → 1256.00 |
| actor.e1.h1800.normal | 4.3987594 [4.3987594, 4.5086486, 4.308397] | 3.1737267 [3.1577412, 3.1737267, 3.248785] | 1.3860× | 3/3 | 2310.04 → 2382.69 | 428.80 → 430.79 | 864.00 → 1256.00 |
| actor.e2.h1800.backlog | 4.1063607 [4.0669189, 4.1394188, 4.1063607] | 3.0226647 [2.9983996, 3.0226647, 3.0907653] | 1.3585× | 3/3 | 2417.66 → 2427.02 | 590.67 → 591.62 | 1128.00 → 1258.00 |
| actor.e2.h1800.normal | 3.3757514 [3.3757514, 3.4059481, 3.3675198] | 2.6280459 [2.5846972, 2.7550548, 2.6280459] | 1.2845× | 3/3 | 2427.02 → 2420.21 | 470.36 → 470.88 | 1252.00 → 1258.00 |
| actor.e4.h1800.backlog | 4.0677746 [4.0677746, 4.1713502, 4.0577851] | 3.0358606 [3.1191939, 3.0358606, 2.9833397] | 1.3399× | 3/3 | 2431.91 → 2444.54 | 682.65 → 682.81 | 1384.00 → 1366.00 |
| actor.e4.h1800.normal | 3.3379412 [3.3379412, 3.3132609, 3.4451655] | 2.5902918 [2.5739784, 2.5992455, 2.5902918] | 1.2886× | 3/3 | 2432.01 → 2438.39 | 563.94 → 565.13 | 1180.00 → 1310.00 |
| checkpoint.digest2 | 0.1492704 [0.1624461, 0.1472361, 0.1492704] | 0.1457158 [0.1488585, 0.1457158, 0.1451385] | 1.0244× | 3/3 | 2651.25 → 2563.09 | 149.53 → 149.53 | 1410.00 → 1392.00 |
| checkpoint.load2 | 0.6791486 [0.6791486, 0.5947458, 0.7654797] | 0.5949976 [0.5949976, 0.5946434, 0.6015287] | 1.1414× | 3/3 | 2801.56 → 2762.52 | 149.53 → 149.53 | 1410.00 → 1392.00 |
| checkpoint.save0 | 0.5991605 [0.59098, 0.6065218, 0.5991605] | 0.5906685 [0.5906685, 0.7000131, 0.5774014] | 1.0144× | 3/3 | 2789.45 → 2724.21 | 306.64 → 254.27 | 1410.00 → 1392.00 |
| checkpoint.save2 | 1.332744 [1.332744, 1.3599629, 1.3192864] | 1.3469354 [1.5417086, 1.3469354, 1.3383648] | 0.9895× | 3/3 | 2840.90 → 2813.08 | 201.90 → 254.27 | 1410.00 → 1392.00 |
| critic.e1.h1800.backlog | 8.6946638 [8.7228983, 8.6791787, 8.6946638] | 6.5610003 [6.6617266, 6.5610003, 6.4995112] | 1.3252× | 3/3 | 2365.81 → 2409.27 | 916.63 → 915.59 | 1128.00 → 1256.00 |
| critic.e1.h1800.normal | 7.8765418 [8.5585396, 7.8765418, 7.6784528] | 5.7601533 [5.6184144, 5.7601533, 5.9282179] | 1.3674× | 3/3 | 2311.59 → 2384.31 | 686.61 → 689.19 | 864.00 → 1256.00 |
| critic.e2.h120.backlog | 3.215042 [3.215042, 3.1549894, 3.3755291] | 3.1098711 [3.1098711, 3.1376436, 3.0498271] | 1.0338× | 3/3 | 2314.40 → 2266.60 | 385.25 → 385.40 | 470.00 → 430.00 |
| critic.e2.h120.normal | 3.2922245 [3.2559142, 3.3770571, 3.2922245] | 3.0719515 [3.0719515, 3.0729505, 3.0620671] | 1.0717× | 3/3 | 2295.99 → 2249.13 | 327.64 → 327.61 | 352.00 → 350.00 |
| critic.e2.h1800.backlog | 7.5135584 [7.5135584, 7.3664575, 8.6107327] | 5.533634 [5.533634, 5.6543039, 5.4806717] | 1.3578× | 3/3 | 2447.86 → 2431.38 | 987.30 → 983.73 | 1252.00 → 1258.00 |
| critic.e2.h1800.normal | 6.2332727 [6.2332727, 6.1692805, 6.241171] | 4.6939653 [4.6895964, 4.7773492, 4.6939653] | 1.3279× | 3/3 | 2423.10 → 2419.36 | 764.54 → 763.37 | 1252.00 → 1258.00 |
| critic.e2.h600.backlog | 5.868996 [5.9117297, 5.868996, 5.7974401] | 4.7581943 [4.7581943, 4.9376696, 4.7267779] | 1.2335× | 3/3 | 2403.06 → 2354.30 | 871.98 → 873.46 | 1134.00 → 1140.00 |
| critic.e2.h600.normal | 5.5406052 [5.5406052, 5.5280866, 5.5701368] | 4.4553379 [4.3341946, 4.4553379, 4.7915702] | 1.2436× | 3/3 | 2359.89 → 2301.09 | 662.35 → 658.74 | 754.00 → 704.00 |
| critic.e4.h1800.backlog | 7.4223033 [7.3783144, 7.4223033, 7.5325647] | 6.0021793 [5.7233531, 6.4530167, 6.0021793] | 1.2366× | 3/3 | 2435.84 → 2445.09 | 1167.12 → 1168.24 | 1384.00 → 1366.00 |
| critic.e4.h1800.normal | 6.1766034 [6.1799369, 6.1480614, 6.1766034] | 4.7163454 [4.7163454, 4.9627299, 4.6702109] | 1.3096× | 3/3 | 2433.82 → 2439.68 | 943.09 → 944.43 | 1180.00 → 1310.00 |
| evaluation.replay900s | 99.5342429 [99.5342429, 99.1976162, 100.434829] | 78.6274071 [78.4976761, 78.6274071, 78.6615826] | 1.2659× | 3/3 | 3362.55 → 3426.78 | 163.59 → 163.59 | 1410.00 → 1392.00 |
| inference.lower.h1.rebuild | 0.0249551 [0.0249551, 0.0257162, 0.0248949] | 0.0227703 [0.0229513, 0.0222683, 0.0227703] | 1.0959× | 3/3 | 2421.05 → 2429.17 | 157.94 → 157.94 | 1384.00 → 1366.00 |
| inference.lower.h120.backlog.cached | 0.0308977 [0.030291, 0.0308977, 0.0326344] | 0.0293652 [0.0293617, 0.0321895, 0.0293652] | 1.0522× | 3/3 | 2412.67 → 2421.57 | 158.06 → 158.06 | 1384.00 → 1366.00 |
| inference.lower.h120.backlog.rebuild | 0.0343557 [0.0336066, 0.0349578, 0.0343557] | 0.0322105 [0.0300565, 0.0322105, 0.0349199] | 1.0666× | 3/3 | 2428.96 → 2437.87 | 163.07 → 163.07 | 1384.00 → 1366.00 |
| inference.lower.h120.cached | 0.0277305 [0.0277305, 0.0273274, 0.0285759] | 0.0277576 [0.0266945, 0.0309063, 0.0277576] | 0.9990× | 3/3 | 2408.00 → 2413.00 | 157.84 → 157.84 | 1384.00 → 1366.00 |
| inference.lower.h120.normal.cached | 0.0297853 [0.0293099, 0.0306953, 0.0297853] | 0.0311966 [0.0311966, 0.0292817, 0.0319539] | 0.9548× | 3/3 | 2412.67 → 2425.59 | 157.91 → 157.91 | 1384.00 → 1366.00 |
| inference.lower.h120.normal.rebuild | 0.0334862 [0.0340317, 0.0321312, 0.0334862] | 0.0304764 [0.0303762, 0.0304764, 0.0314423] | 1.0988× | 3/3 | 2428.97 → 2437.89 | 164.71 → 165.56 | 1384.00 → 1366.00 |
| inference.lower.h120.rebuild | 0.0363593 [0.0363593, 0.0301451, 0.0371635] | 0.0346653 [0.0346653, 0.0327277, 0.03518] | 1.0489× | 3/3 | 2420.28 → 2428.93 | 165.93 → 165.65 | 1384.00 → 1366.00 |
| inference.lower.h1800.after_actor_update | 0.0325066 [0.0335579, 0.0283473, 0.0325066] | 0.0302496 [0.0300541, 0.0333496, 0.0302496] | 1.0746× | 3/3 | 2351.75 → 2360.28 | 275.11 → 275.11 | 1384.00 → 1366.00 |
| inference.lower.h1800.backlog.cached | 0.0336569 [0.0336569, 0.0376154, 0.0324604] | 0.0302312 [0.0302312, 0.02905, 0.0306736] | 1.1133× | 3/3 | 2418.20 → 2423.20 | 158.05 → 158.05 | 1384.00 → 1366.00 |
| inference.lower.h1800.backlog.rebuild | 0.0416458 [0.0395587, 0.0416458, 0.0475508] | 0.0397592 [0.0400243, 0.0397592, 0.0363295] | 1.0475× | 3/3 | 2430.61 → 2437.89 | 268.30 → 268.30 | 1384.00 → 1366.00 |
| inference.lower.h1800.cached | 0.0277369 [0.0277369, 0.0267639, 0.0322754] | 0.0255707 [0.0255322, 0.0255707, 0.0273902] | 1.0847× | 3/3 | 2405.11 → 2414.02 | 157.84 → 157.84 | 1384.00 → 1366.00 |
| inference.lower.h1800.normal.cached | 0.033822 [0.0346948, 0.033822, 0.030846] | 0.0287883 [0.0287883, 0.0284319, 0.0371527] | 1.1749× | 3/3 | 2414.30 → 2423.20 | 157.87 → 157.87 | 1384.00 → 1366.00 |
| inference.lower.h1800.normal.rebuild | 0.036684 [0.0427134, 0.0359629, 0.036684] | 0.0363243 [0.0363243, 0.0412857, 0.0360882] | 1.0099× | 3/3 | 2428.99 → 2437.90 | 269.55 → 269.55 | 1384.00 → 1366.00 |
| inference.lower.h1800.rebuild | 0.0331082 [0.0334777, 0.0329698, 0.0331082] | 0.0336937 [0.0372476, 0.0336937, 0.0323769] | 0.9826× | 3/3 | 2419.81 → 2428.23 | 269.79 → 269.79 | 1384.00 → 1366.00 |
| inference.lower.h600.backlog.cached | 0.0336082 [0.0336082, 0.0355289, 0.0323858] | 0.0360639 [0.02965, 0.0360639, 0.0370769] | 0.9319× | 3/3 | 2412.79 → 2425.58 | 157.98 → 157.98 | 1384.00 → 1366.00 |
| inference.lower.h600.backlog.rebuild | 0.0340279 [0.0340279, 0.0329848, 0.0363992] | 0.0316831 [0.0305689, 0.0325966, 0.0316831] | 1.0740× | 3/3 | 2429.02 → 2437.91 | 194.19 → 194.19 | 1384.00 → 1366.00 |
| inference.lower.h600.cached | 0.031002 [0.0367929, 0.0301937, 0.031002] | 0.0356117 [0.0363459, 0.0311255, 0.0356117] | 0.8706× | 3/3 | 2404.11 → 2416.41 | 158.01 → 158.01 | 1384.00 → 1366.00 |
| inference.lower.h600.normal.cached | 0.0301921 [0.0313628, 0.029788, 0.0301921] | 0.0330686 [0.0367896, 0.0326825, 0.0330686] | 0.9130× | 3/3 | 2416.68 → 2421.69 | 158.01 → 158.01 | 1384.00 → 1366.00 |
| inference.lower.h600.normal.rebuild | 0.0349676 [0.0321782, 0.0349676, 0.0353345] | 0.0315138 [0.0335242, 0.0311098, 0.0315138] | 1.1096× | 3/3 | 2428.98 → 2437.91 | 194.87 → 194.87 | 1384.00 → 1366.00 |
| inference.lower.h600.rebuild | 0.0386544 [0.0386544, 0.0342849, 0.0397615] | 0.030901 [0.030901, 0.0367206, 0.0302833] | 1.2509× | 3/3 | 2419.81 → 2428.74 | 194.87 → 194.87 | 1384.00 → 1366.00 |
| inference.upper.h0.rebuild | 0.019317 [0.0183737, 0.0202586, 0.019317] | 0.0148293 [0.01416, 0.0149629, 0.0148293] | 1.3026× | 3/3 | 2429.46 → 2437.89 | 149.73 → 149.73 | 1384.00 → 1366.00 |
| inference.upper.h120.cached | 0.0259322 [0.0250784, 0.0286063, 0.0259322] | 0.0239059 [0.0239059, 0.0278602, 0.0230711] | 1.0848× | 3/3 | 2413.26 → 2425.56 | 158.18 → 158.18 | 1384.00 → 1366.00 |
| inference.upper.h120.rebuild | 0.0273022 [0.0301572, 0.0273022, 0.0268334] | 0.0241719 [0.0240391, 0.0241719, 0.0279145] | 1.1295× | 3/3 | 2429.23 → 2437.84 | 165.93 → 165.65 | 1384.00 → 1366.00 |
| inference.upper.h1800.cached | 0.0256469 [0.025574, 0.0328656, 0.0256469] | 0.0328369 [0.0328369, 0.0337862, 0.0315216] | 0.7810× | 3/3 | 2414.52 → 2423.17 | 158.18 → 158.18 | 1384.00 → 1366.00 |
| inference.upper.h1800.rebuild | 0.0318182 [0.0327888, 0.0302749, 0.0318182] | 0.0479936 [0.0479936, 0.0541428, 0.0380034] | 0.6630× | 3/3 | 2429.21 → 2437.88 | 269.79 → 269.79 | 1384.00 → 1366.00 |
| inference.upper.h600.cached | 0.02701 [0.0270989, 0.02701, 0.0257077] | 0.025302 [0.023651, 0.025302, 0.0306312] | 1.0675× | 3/3 | 2416.91 → 2421.66 | 158.18 → 158.18 | 1384.00 → 1366.00 |
| inference.upper.h600.rebuild | 0.0340423 [0.0364498, 0.0274281, 0.0340423] | 0.0258073 [0.0261215, 0.0258073, 0.0257693] | 1.3191× | 3/3 | 2429.24 → 2437.89 | 194.87 → 194.87 | 1384.00 → 1366.00 |
| inference.upper_value.h0.rebuild | 0.0300436 [0.0300436, 0.0285302, 0.0332017] | 0.0267532 [0.0322961, 0.0255593, 0.0267532] | 1.1230× | 3/3 | 2429.45 → 2437.89 | 149.74 → 149.74 | 1384.00 → 1366.00 |
| inference.upper_value.h120.cached | 0.0457627 [0.0457627, 0.0561661, 0.0426371] | 0.0449699 [0.0449699, 0.0527447, 0.0435742] | 1.0176× | 3/3 | 2413.39 → 2422.16 | 158.19 → 158.19 | 1384.00 → 1366.00 |
| inference.upper_value.h120.rebuild | 0.0495994 [0.0493061, 0.0568683, 0.0495994] | 0.04842 [0.0522283, 0.0448161, 0.04842] | 1.0244× | 3/3 | 2429.21 → 2437.85 | 165.94 → 165.65 | 1384.00 → 1366.00 |
| inference.upper_value.h1800.cached | 0.0468485 [0.0475797, 0.0468485, 0.0466137] | 0.0536749 [0.055063, 0.0536749, 0.0496873] | 0.8728× | 3/3 | 2418.17 → 2423.17 | 158.19 → 158.19 | 1384.00 → 1366.00 |
| inference.upper_value.h1800.rebuild | 0.0583581 [0.0585562, 0.0571226, 0.0583581] | 0.0620733 [0.0620733, 0.0618149, 0.0630821] | 0.9401× | 3/3 | 2428.96 → 2437.87 | 269.79 → 269.79 | 1384.00 → 1366.00 |
| inference.upper_value.h600.cached | 0.0538154 [0.0469464, 0.0538154, 0.0588263] | 0.0445516 [0.0542287, 0.0443991, 0.0445516] | 1.2079× | 3/3 | 2413.01 → 2425.56 | 158.19 → 158.19 | 1384.00 → 1366.00 |
| inference.upper_value.h600.rebuild | 0.0629655 [0.0509312, 0.0629655, 0.0643277] | 0.0595927 [0.0700994, 0.0512597, 0.0595927] | 1.0566× | 3/3 | 2429.22 → 2437.89 | 194.88 → 194.88 | 1384.00 → 1366.00 |
| io.learning_jsonl | 0.0004771 [0.0006769, 0.0004771, 0.0003785] | 0.0005172 [0.0017574, 0.0005172, 0.000428] | 0.9225× | 3/3 | 2654.42 → 2563.99 | 149.53 → 149.53 | 1410.00 → 1392.00 |
| io.metrics_json | 0.1879693 [0.1879693, 0.1882679, 0.183287] | 0.1849906 [0.1903184, 0.1849906, 0.1846666] | 1.0161× | 3/3 | 2654.42 → 2563.74 | 149.53 → 149.53 | 1410.00 → 1392.00 |
| ppo.rows900 | 82.7197132 [84.3584056, 82.4683842, 82.7197132] | 79.7936183 [78.7811729, 79.7936183, 80.3541003] | 1.0367× | 3/3 | 2727.16 → 2788.58 | 715.83 → 717.56 | 1410.00 → 1392.00 |
| replay.sample128 | 0.0005637 [0.0005637, 0.0005754, 0.0005212] | 0.0005654 [0.000565, 0.0005654, 0.0006731] | 0.9970× | 3/3 | 2649.07 → 2558.90 | 149.53 → 149.53 | 1410.00 → 1392.00 |
| source.identity | 0.0038383 [0.0045381, 0.0038383, 0.0038186] | 0.003396 [0.003396, 0.0034639, 0.0033591] | 1.1302× | 3/3 | 2648.50 → 2559.39 | 149.53 → 149.53 | 1410.00 → 1392.00 |
| target.polyak | 0.007433 [0.0063368, 0.007433, 0.0076196] | 0.0063878 [0.0061231, 0.0063878, 0.0080821] | 1.1636× | 3/3 | 2654.41 → 2563.99 | 149.53 → 149.53 | 1410.00 → 1392.00 |

## baseline → candidate_1

Diagnostic candidate only. Numerical equivalence acceptance is not supplied to this tool. A measured speed ratio is not approval to change the default.


Times are seconds: **median [raw repetitions]**. Speedup is before/after median. Memory columns are before → after in MiB.

| Case | Before seconds | After seconds | Speedup | Repeats | Peak RSS | Peak allocated | Peak reserved |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| critic.e2.h120.backlog | 3.215042 [3.215042, 3.1549894, 3.3755291] | 2.11848 [2.1363471, 2.0982407, 2.11848] | 1.5176× | 3/3 | 2314.40 → 2275.60 | 385.25 → 1019.40 | 470.00 → 1262.00 |
| critic.e2.h120.normal | 3.2922245 [3.2559142, 3.3770571, 3.2922245] | 1.9951081 [2.019231, 1.9951081, 1.9719361] | 1.6501× | 3/3 | 2295.99 → 2255.95 | 327.64 → 789.97 | 352.00 → 860.00 |
| critic.e2.h1800.backlog | 7.5135584 [7.5135584, 7.3664575, 8.6107327] | 4.7581677 [4.8862054, 4.6726638, 4.7581677] | 1.5791× | 3/3 | 2447.86 → 2430.70 | 987.30 → 2908.76 | 1252.00 → 3344.00 |
| critic.e2.h1800.normal | 6.2332727 [6.2332727, 6.1692805, 6.241171] | 3.8995793 [3.9966116, 3.8806516, 3.8995793] | 1.5984× | 3/3 | 2423.10 → 2393.50 | 764.54 → 2040.45 | 1252.00 → 3172.00 |
| critic.e2.h600.backlog | 5.868996 [5.9117297, 5.868996, 5.7974401] | 3.9716663 [4.144881, 3.9716663, 3.9670918] | 1.4777× | 3/3 | 2403.06 → 2370.88 | 871.98 → 2787.81 | 1134.00 → 3172.00 |
| critic.e2.h600.normal | 5.5406052 [5.5406052, 5.5280866, 5.5701368] | 3.473487 [3.473487, 3.5472706, 3.4668193] | 1.5951× | 3/3 | 2359.89 → 2316.59 | 662.35 → 1957.61 | 754.00 → 2290.00 |

Missing measurements (no values substituted):

- `actor.e1.h1800.backlog`: missing from candidate_1.
- `actor.e1.h1800.normal`: missing from candidate_1.
- `actor.e2.h1800.backlog`: missing from candidate_1.
- `actor.e2.h1800.normal`: missing from candidate_1.
- `actor.e4.h1800.backlog`: missing from candidate_1.
- `actor.e4.h1800.normal`: missing from candidate_1.
- `checkpoint.digest2`: missing from candidate_1.
- `checkpoint.load2`: missing from candidate_1.
- `checkpoint.save0`: missing from candidate_1.
- `checkpoint.save2`: missing from candidate_1.
- `critic.e1.h1800.backlog`: missing from candidate_1.
- `critic.e1.h1800.normal`: missing from candidate_1.
- `critic.e4.h1800.backlog`: missing from candidate_1.
- `critic.e4.h1800.normal`: missing from candidate_1.
- `evaluation.replay900s`: missing from candidate_1.
- `inference.lower.h1.rebuild`: missing from candidate_1.
- `inference.lower.h120.backlog.cached`: missing from candidate_1.
- `inference.lower.h120.backlog.rebuild`: missing from candidate_1.
- `inference.lower.h120.cached`: missing from candidate_1.
- `inference.lower.h120.normal.cached`: missing from candidate_1.
- `inference.lower.h120.normal.rebuild`: missing from candidate_1.
- `inference.lower.h120.rebuild`: missing from candidate_1.
- `inference.lower.h1800.after_actor_update`: missing from candidate_1.
- `inference.lower.h1800.backlog.cached`: missing from candidate_1.
- `inference.lower.h1800.backlog.rebuild`: missing from candidate_1.
- `inference.lower.h1800.cached`: missing from candidate_1.
- `inference.lower.h1800.normal.cached`: missing from candidate_1.
- `inference.lower.h1800.normal.rebuild`: missing from candidate_1.
- `inference.lower.h1800.rebuild`: missing from candidate_1.
- `inference.lower.h600.backlog.cached`: missing from candidate_1.
- `inference.lower.h600.backlog.rebuild`: missing from candidate_1.
- `inference.lower.h600.cached`: missing from candidate_1.
- `inference.lower.h600.normal.cached`: missing from candidate_1.
- `inference.lower.h600.normal.rebuild`: missing from candidate_1.
- `inference.lower.h600.rebuild`: missing from candidate_1.
- `inference.upper.h0.rebuild`: missing from candidate_1.
- `inference.upper.h120.cached`: missing from candidate_1.
- `inference.upper.h120.rebuild`: missing from candidate_1.
- `inference.upper.h1800.cached`: missing from candidate_1.
- `inference.upper.h1800.rebuild`: missing from candidate_1.
- `inference.upper.h600.cached`: missing from candidate_1.
- `inference.upper.h600.rebuild`: missing from candidate_1.
- `inference.upper_value.h0.rebuild`: missing from candidate_1.
- `inference.upper_value.h120.cached`: missing from candidate_1.
- `inference.upper_value.h120.rebuild`: missing from candidate_1.
- `inference.upper_value.h1800.cached`: missing from candidate_1.
- `inference.upper_value.h1800.rebuild`: missing from candidate_1.
- `inference.upper_value.h600.cached`: missing from candidate_1.
- `inference.upper_value.h600.rebuild`: missing from candidate_1.
- `io.learning_jsonl`: missing from candidate_1.
- `io.metrics_json`: missing from candidate_1.
- `ppo.rows900`: missing from candidate_1.
- `replay.sample128`: missing from candidate_1.
- `source.identity`: missing from candidate_1.
- `target.polyak`: missing from candidate_1.

## current → candidate_1

Diagnostic candidate only. Numerical equivalence acceptance is not supplied to this tool. A measured speed ratio is not approval to change the default.


Times are seconds: **median [raw repetitions]**. Speedup is before/after median. Memory columns are before → after in MiB.

| Case | Before seconds | After seconds | Speedup | Repeats | Peak RSS | Peak allocated | Peak reserved |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| critic.e2.h120.backlog | 3.1098711 [3.1098711, 3.1376436, 3.0498271] | 2.11848 [2.1363471, 2.0982407, 2.11848] | 1.4680× | 3/3 | 2266.60 → 2275.60 | 385.40 → 1019.40 | 430.00 → 1262.00 |
| critic.e2.h120.normal | 3.0719515 [3.0719515, 3.0729505, 3.0620671] | 1.9951081 [2.019231, 1.9951081, 1.9719361] | 1.5397× | 3/3 | 2249.13 → 2255.95 | 327.61 → 789.97 | 350.00 → 860.00 |
| critic.e2.h1800.backlog | 5.533634 [5.533634, 5.6543039, 5.4806717] | 4.7581677 [4.8862054, 4.6726638, 4.7581677] | 1.1630× | 3/3 | 2431.38 → 2430.70 | 983.73 → 2908.76 | 1258.00 → 3344.00 |
| critic.e2.h1800.normal | 4.6939653 [4.6895964, 4.7773492, 4.6939653] | 3.8995793 [3.9966116, 3.8806516, 3.8995793] | 1.2037× | 3/3 | 2419.36 → 2393.50 | 763.37 → 2040.45 | 1258.00 → 3172.00 |
| critic.e2.h600.backlog | 4.7581943 [4.7581943, 4.9376696, 4.7267779] | 3.9716663 [4.144881, 3.9716663, 3.9670918] | 1.1980× | 3/3 | 2354.30 → 2370.88 | 873.46 → 2787.81 | 1140.00 → 3172.00 |
| critic.e2.h600.normal | 4.4553379 [4.3341946, 4.4553379, 4.7915702] | 3.473487 [3.473487, 3.5472706, 3.4668193] | 1.2827× | 3/3 | 2301.09 → 2316.59 | 658.74 → 1957.61 | 704.00 → 2290.00 |

Missing measurements (no values substituted):

- `actor.e1.h1800.backlog`: missing from candidate_1.
- `actor.e1.h1800.normal`: missing from candidate_1.
- `actor.e2.h1800.backlog`: missing from candidate_1.
- `actor.e2.h1800.normal`: missing from candidate_1.
- `actor.e4.h1800.backlog`: missing from candidate_1.
- `actor.e4.h1800.normal`: missing from candidate_1.
- `checkpoint.digest2`: missing from candidate_1.
- `checkpoint.load2`: missing from candidate_1.
- `checkpoint.save0`: missing from candidate_1.
- `checkpoint.save2`: missing from candidate_1.
- `critic.e1.h1800.backlog`: missing from candidate_1.
- `critic.e1.h1800.normal`: missing from candidate_1.
- `critic.e4.h1800.backlog`: missing from candidate_1.
- `critic.e4.h1800.normal`: missing from candidate_1.
- `evaluation.replay900s`: missing from candidate_1.
- `inference.lower.h1.rebuild`: missing from candidate_1.
- `inference.lower.h120.backlog.cached`: missing from candidate_1.
- `inference.lower.h120.backlog.rebuild`: missing from candidate_1.
- `inference.lower.h120.cached`: missing from candidate_1.
- `inference.lower.h120.normal.cached`: missing from candidate_1.
- `inference.lower.h120.normal.rebuild`: missing from candidate_1.
- `inference.lower.h120.rebuild`: missing from candidate_1.
- `inference.lower.h1800.after_actor_update`: missing from candidate_1.
- `inference.lower.h1800.backlog.cached`: missing from candidate_1.
- `inference.lower.h1800.backlog.rebuild`: missing from candidate_1.
- `inference.lower.h1800.cached`: missing from candidate_1.
- `inference.lower.h1800.normal.cached`: missing from candidate_1.
- `inference.lower.h1800.normal.rebuild`: missing from candidate_1.
- `inference.lower.h1800.rebuild`: missing from candidate_1.
- `inference.lower.h600.backlog.cached`: missing from candidate_1.
- `inference.lower.h600.backlog.rebuild`: missing from candidate_1.
- `inference.lower.h600.cached`: missing from candidate_1.
- `inference.lower.h600.normal.cached`: missing from candidate_1.
- `inference.lower.h600.normal.rebuild`: missing from candidate_1.
- `inference.lower.h600.rebuild`: missing from candidate_1.
- `inference.upper.h0.rebuild`: missing from candidate_1.
- `inference.upper.h120.cached`: missing from candidate_1.
- `inference.upper.h120.rebuild`: missing from candidate_1.
- `inference.upper.h1800.cached`: missing from candidate_1.
- `inference.upper.h1800.rebuild`: missing from candidate_1.
- `inference.upper.h600.cached`: missing from candidate_1.
- `inference.upper.h600.rebuild`: missing from candidate_1.
- `inference.upper_value.h0.rebuild`: missing from candidate_1.
- `inference.upper_value.h120.cached`: missing from candidate_1.
- `inference.upper_value.h120.rebuild`: missing from candidate_1.
- `inference.upper_value.h1800.cached`: missing from candidate_1.
- `inference.upper_value.h1800.rebuild`: missing from candidate_1.
- `inference.upper_value.h600.cached`: missing from candidate_1.
- `inference.upper_value.h600.rebuild`: missing from candidate_1.
- `io.learning_jsonl`: missing from candidate_1.
- `io.metrics_json`: missing from candidate_1.
- `ppo.rows900`: missing from candidate_1.
- `replay.sample128`: missing from candidate_1.
- `source.identity`: missing from candidate_1.
- `target.polyak`: missing from candidate_1.

## baseline → candidate_2

Diagnostic candidate only. Numerical equivalence acceptance is not supplied to this tool. A measured speed ratio is not approval to change the default.


Times are seconds: **median [raw repetitions]**. Speedup is before/after median. Memory columns are before → after in MiB.

| Case | Before seconds | After seconds | Speedup | Repeats | Peak RSS | Peak allocated | Peak reserved |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| critic.e2.h120.backlog | 3.215042 [3.215042, 3.1549894, 3.3755291] | 2.0434133 [1.9590498, 2.0434133, 2.076719] | 1.5734× | 3/3 | 2314.40 → 2279.09 | 385.25 → 1572.24 | 470.00 → 1674.00 |
| critic.e2.h120.normal | 3.2922245 [3.2559142, 3.3770571, 3.2922245] | 1.9381597 [1.9346685, 1.9592373, 1.9381597] | 1.6986× | 3/3 | 2295.99 → 2257.66 | 327.64 → 1157.60 | 352.00 → 1214.00 |
| critic.e2.h1800.backlog | 7.5135584 [7.5135584, 7.3664575, 8.6107327] | 5.0043351 [4.9348045, 5.0371114, 5.0043351] | 1.5014× | 3/3 | 2447.86 → 2428.14 | 987.30 → 3636.63 | 1252.00 → 3980.00 |
| critic.e2.h1800.normal | 6.2332727 [6.2332727, 6.1692805, 6.241171] | 4.104132 [4.104132, 4.19317, 3.9252871] | 1.5188× | 3/3 | 2423.10 → 2395.71 | 764.54 → 2495.06 | 1252.00 → 3652.00 |
| critic.e2.h600.backlog | 5.868996 [5.9117297, 5.868996, 5.7974401] | 4.3176142 [4.4173345, 4.2902528, 4.3176142] | 1.3593× | 3/3 | 2403.06 → 2374.66 | 871.98 → 3433.93 | 1134.00 → 3652.00 |
| critic.e2.h600.normal | 5.5406052 [5.5406052, 5.5280866, 5.5701368] | 3.725618 [3.7310112, 3.6439736, 3.725618] | 1.4872× | 3/3 | 2359.89 → 2327.32 | 662.35 → 2426.05 | 754.00 → 2620.00 |

Missing measurements (no values substituted):

- `actor.e1.h1800.backlog`: missing from candidate_2.
- `actor.e1.h1800.normal`: missing from candidate_2.
- `actor.e2.h1800.backlog`: missing from candidate_2.
- `actor.e2.h1800.normal`: missing from candidate_2.
- `actor.e4.h1800.backlog`: missing from candidate_2.
- `actor.e4.h1800.normal`: missing from candidate_2.
- `checkpoint.digest2`: missing from candidate_2.
- `checkpoint.load2`: missing from candidate_2.
- `checkpoint.save0`: missing from candidate_2.
- `checkpoint.save2`: missing from candidate_2.
- `critic.e1.h1800.backlog`: missing from candidate_2.
- `critic.e1.h1800.normal`: missing from candidate_2.
- `critic.e4.h1800.backlog`: missing from candidate_2.
- `critic.e4.h1800.normal`: missing from candidate_2.
- `evaluation.replay900s`: missing from candidate_2.
- `inference.lower.h1.rebuild`: missing from candidate_2.
- `inference.lower.h120.backlog.cached`: missing from candidate_2.
- `inference.lower.h120.backlog.rebuild`: missing from candidate_2.
- `inference.lower.h120.cached`: missing from candidate_2.
- `inference.lower.h120.normal.cached`: missing from candidate_2.
- `inference.lower.h120.normal.rebuild`: missing from candidate_2.
- `inference.lower.h120.rebuild`: missing from candidate_2.
- `inference.lower.h1800.after_actor_update`: missing from candidate_2.
- `inference.lower.h1800.backlog.cached`: missing from candidate_2.
- `inference.lower.h1800.backlog.rebuild`: missing from candidate_2.
- `inference.lower.h1800.cached`: missing from candidate_2.
- `inference.lower.h1800.normal.cached`: missing from candidate_2.
- `inference.lower.h1800.normal.rebuild`: missing from candidate_2.
- `inference.lower.h1800.rebuild`: missing from candidate_2.
- `inference.lower.h600.backlog.cached`: missing from candidate_2.
- `inference.lower.h600.backlog.rebuild`: missing from candidate_2.
- `inference.lower.h600.cached`: missing from candidate_2.
- `inference.lower.h600.normal.cached`: missing from candidate_2.
- `inference.lower.h600.normal.rebuild`: missing from candidate_2.
- `inference.lower.h600.rebuild`: missing from candidate_2.
- `inference.upper.h0.rebuild`: missing from candidate_2.
- `inference.upper.h120.cached`: missing from candidate_2.
- `inference.upper.h120.rebuild`: missing from candidate_2.
- `inference.upper.h1800.cached`: missing from candidate_2.
- `inference.upper.h1800.rebuild`: missing from candidate_2.
- `inference.upper.h600.cached`: missing from candidate_2.
- `inference.upper.h600.rebuild`: missing from candidate_2.
- `inference.upper_value.h0.rebuild`: missing from candidate_2.
- `inference.upper_value.h120.cached`: missing from candidate_2.
- `inference.upper_value.h120.rebuild`: missing from candidate_2.
- `inference.upper_value.h1800.cached`: missing from candidate_2.
- `inference.upper_value.h1800.rebuild`: missing from candidate_2.
- `inference.upper_value.h600.cached`: missing from candidate_2.
- `inference.upper_value.h600.rebuild`: missing from candidate_2.
- `io.learning_jsonl`: missing from candidate_2.
- `io.metrics_json`: missing from candidate_2.
- `ppo.rows900`: missing from candidate_2.
- `replay.sample128`: missing from candidate_2.
- `source.identity`: missing from candidate_2.
- `target.polyak`: missing from candidate_2.

## current → candidate_2

Diagnostic candidate only. Numerical equivalence acceptance is not supplied to this tool. A measured speed ratio is not approval to change the default.


Times are seconds: **median [raw repetitions]**. Speedup is before/after median. Memory columns are before → after in MiB.

| Case | Before seconds | After seconds | Speedup | Repeats | Peak RSS | Peak allocated | Peak reserved |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| critic.e2.h120.backlog | 3.1098711 [3.1098711, 3.1376436, 3.0498271] | 2.0434133 [1.9590498, 2.0434133, 2.076719] | 1.5219× | 3/3 | 2266.60 → 2279.09 | 385.40 → 1572.24 | 430.00 → 1674.00 |
| critic.e2.h120.normal | 3.0719515 [3.0719515, 3.0729505, 3.0620671] | 1.9381597 [1.9346685, 1.9592373, 1.9381597] | 1.5850× | 3/3 | 2249.13 → 2257.66 | 327.61 → 1157.60 | 350.00 → 1214.00 |
| critic.e2.h1800.backlog | 5.533634 [5.533634, 5.6543039, 5.4806717] | 5.0043351 [4.9348045, 5.0371114, 5.0043351] | 1.1058× | 3/3 | 2431.38 → 2428.14 | 983.73 → 3636.63 | 1258.00 → 3980.00 |
| critic.e2.h1800.normal | 4.6939653 [4.6895964, 4.7773492, 4.6939653] | 4.104132 [4.104132, 4.19317, 3.9252871] | 1.1437× | 3/3 | 2419.36 → 2395.71 | 763.37 → 2495.06 | 1258.00 → 3652.00 |
| critic.e2.h600.backlog | 4.7581943 [4.7581943, 4.9376696, 4.7267779] | 4.3176142 [4.4173345, 4.2902528, 4.3176142] | 1.1020× | 3/3 | 2354.30 → 2374.66 | 873.46 → 3433.93 | 1140.00 → 3652.00 |
| critic.e2.h600.normal | 4.4553379 [4.3341946, 4.4553379, 4.7915702] | 3.725618 [3.7310112, 3.6439736, 3.725618] | 1.1959× | 3/3 | 2301.09 → 2327.32 | 658.74 → 2426.05 | 704.00 → 2620.00 |

Missing measurements (no values substituted):

- `actor.e1.h1800.backlog`: missing from candidate_2.
- `actor.e1.h1800.normal`: missing from candidate_2.
- `actor.e2.h1800.backlog`: missing from candidate_2.
- `actor.e2.h1800.normal`: missing from candidate_2.
- `actor.e4.h1800.backlog`: missing from candidate_2.
- `actor.e4.h1800.normal`: missing from candidate_2.
- `checkpoint.digest2`: missing from candidate_2.
- `checkpoint.load2`: missing from candidate_2.
- `checkpoint.save0`: missing from candidate_2.
- `checkpoint.save2`: missing from candidate_2.
- `critic.e1.h1800.backlog`: missing from candidate_2.
- `critic.e1.h1800.normal`: missing from candidate_2.
- `critic.e4.h1800.backlog`: missing from candidate_2.
- `critic.e4.h1800.normal`: missing from candidate_2.
- `evaluation.replay900s`: missing from candidate_2.
- `inference.lower.h1.rebuild`: missing from candidate_2.
- `inference.lower.h120.backlog.cached`: missing from candidate_2.
- `inference.lower.h120.backlog.rebuild`: missing from candidate_2.
- `inference.lower.h120.cached`: missing from candidate_2.
- `inference.lower.h120.normal.cached`: missing from candidate_2.
- `inference.lower.h120.normal.rebuild`: missing from candidate_2.
- `inference.lower.h120.rebuild`: missing from candidate_2.
- `inference.lower.h1800.after_actor_update`: missing from candidate_2.
- `inference.lower.h1800.backlog.cached`: missing from candidate_2.
- `inference.lower.h1800.backlog.rebuild`: missing from candidate_2.
- `inference.lower.h1800.cached`: missing from candidate_2.
- `inference.lower.h1800.normal.cached`: missing from candidate_2.
- `inference.lower.h1800.normal.rebuild`: missing from candidate_2.
- `inference.lower.h1800.rebuild`: missing from candidate_2.
- `inference.lower.h600.backlog.cached`: missing from candidate_2.
- `inference.lower.h600.backlog.rebuild`: missing from candidate_2.
- `inference.lower.h600.cached`: missing from candidate_2.
- `inference.lower.h600.normal.cached`: missing from candidate_2.
- `inference.lower.h600.normal.rebuild`: missing from candidate_2.
- `inference.lower.h600.rebuild`: missing from candidate_2.
- `inference.upper.h0.rebuild`: missing from candidate_2.
- `inference.upper.h120.cached`: missing from candidate_2.
- `inference.upper.h120.rebuild`: missing from candidate_2.
- `inference.upper.h1800.cached`: missing from candidate_2.
- `inference.upper.h1800.rebuild`: missing from candidate_2.
- `inference.upper.h600.cached`: missing from candidate_2.
- `inference.upper.h600.rebuild`: missing from candidate_2.
- `inference.upper_value.h0.rebuild`: missing from candidate_2.
- `inference.upper_value.h120.cached`: missing from candidate_2.
- `inference.upper_value.h120.rebuild`: missing from candidate_2.
- `inference.upper_value.h1800.cached`: missing from candidate_2.
- `inference.upper_value.h1800.rebuild`: missing from candidate_2.
- `inference.upper_value.h600.cached`: missing from candidate_2.
- `inference.upper_value.h600.rebuild`: missing from candidate_2.
- `io.learning_jsonl`: missing from candidate_2.
- `io.metrics_json`: missing from candidate_2.
- `ppo.rows900`: missing from candidate_2.
- `replay.sample128`: missing from candidate_2.
- `source.identity`: missing from candidate_2.
- `target.polyak`: missing from candidate_2.

Update counters must match exactly and are retained as records; different PPO stopping or update counts reject comparison. This report does not establish numerical equivalence and never changes or recommends a default automatically.
