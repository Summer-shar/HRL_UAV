# v32b matched-reference replay addendum

> Status: FROZEN before any v32b joint-training result was observed.
> This document clarifies reference-policy evaluation only. It does not change
> the frozen environment, training algorithm, seeds, budget, or checkpoint rule
> in `v32b_load_switch_formal_protocol.md`.

## 1. Historical result remains historical

The archived P1-prime result (`1066.687 ms`, power UCB95 `841.797 W`) remains
valid for the exact tuple under which it was measured:

`(P1-prime macro rule, historical micro checkpoint, historical multiplier)`.

Its archived `POWER_INFEASIBLE` verdict is immutable. It must not, however, be
silently transferred to a different micro policy or multiplier.

## 2. Matched v32b replay

For each selected v32b TD3 training-seed checkpoint:

1. replay P0 and P1-prime with that checkpoint's own micro SAC and frozen
   system multiplier;
2. use evaluation seeds `1000..1019` and identical exogenous random streams;
3. retain the frozen P1-prime rule (`40 m` residual target and `10 m` residual
   slew cap per macro period); no P1-double-prime or result-dependent retuning
   is allowed;
4. recompute delay, mean power, and one-sided power UCB95 from the matched
   replay rather than substituting historical values.

## 3. Pre-registered table treatment

- If matched P1-prime has power UCB95 at most `840 W`, it is a feasible fixed
  rule baseline for that training seed and participates in feasible-policy
  comparison.
- If matched P1-prime has power UCB95 above `840 W`, it remains a diagnostic
  position-value reference and is excluded from feasible-policy ranking.
- Feasibility is reported separately for every training seed. If the matched
  verdict is mixed across seeds, the main table shows a per-seed feasibility
  flag and the aggregate feasible-seed count; no single global
  "feasible"/"infeasible" label is assigned. P1-prime participates in a
  feasible-policy comparison only within seed strata where its matched UCB95
  satisfies the budget.
- The historical and matched verdicts are reported with their respective
  micro-policy/multiplier provenance; neither overwrites the other.
- Any reported TD3 capture fraction uses the matched P0/P1-prime pair for the
  same training seed. It is descriptive and never replaces the primary
  TD3-versus-P0 paired inference.

## 4. Monitoring interpretation

The system multiplier updates once per episode. Therefore:

- lambda and dual-update power are inspected across training episodes;
- switch-window effects are inspected using phase-conditioned per-macro power,
  delay, and movement;
- no claim of intra-episode lambda oscillation or response to an individual
  switch is permitted.
