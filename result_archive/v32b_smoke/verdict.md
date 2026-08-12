# v32b load-switch smoke verdict

Status: **PASS (engineering only)**

This smoke run validates the v32b execution pipeline. It is not performance evidence.

## Frozen setup exercised

- Seed: 1
- Micro pretraining: 2 episodes
- Joint TD3-SAC: 2 episodes
- Macro periods per episode: 15
- Load switching: cyclic phases with a dedicated, seed-derived random dwell in `[8, 12]`
- Switch timing: the phase is changed before the macro state is observed

## Checks

1. Both micro pretraining and joint training completed without exception.
2. The two episodes switched at macro indices 9 and 8, respectively, within the frozen dwell range.
3. The phase load vector changed from `[12.5, 5.5, 3.5]` to `[3.5, 12.5, 5.5]` on the flagged macro period.
4. Joint-training trace length was exactly `2 x 15 = 30` rows.
5. Recomputing episode power from `macro_energy_j + service_energy_j` reproduced `train_td3.csv` exactly:
   - Episode 1: `936.435530348 W`, absolute difference `0`
   - Episode 2: `929.500647592 W`, absolute difference `0`
6. Scenario metadata records `load_switch_timing=before_macro_state_observation` and no future load information.
7. The full regression suite passed: `146/146` tests.

## Interpretation boundary

The two joint-training episodes use TD3 warm-up actions and therefore show high transient power. The dual window is 20 episodes, so lambda remaining zero in this two-episode smoke run is expected and is not an algorithmic result.

## Verdict

The v32b pipeline is cleared to start seed-1 formal training. Formal results must be judged only under the frozen protocol in `docs/v32b_load_switch_formal_protocol.md`.
