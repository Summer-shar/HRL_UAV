# P1-prime analytic pre-screen and validation protocol

> Status: FROZEN before P1-prime replay. Date: 2026-08-11.
>
> The analytic grid and candidate-selection rule may not be changed after any
> P1-prime replay result is observed. The original P1 gate verdict remains
> unchanged.

## 1. Purpose

Define one causal, power-aware handcrafted load-switching baseline P1-prime
for the 840 W operating point. P1-prime retains P1's observable trigger,
support-UAV selection, and direction. Only the residual offset and residual
approach rate may differ.

## 2. Frozen grid and speed semantics

- Residual offsets: `10, 20, 30, 40 m`.
- Residual movement caps per 10 s macro period: `5, 10, 15, 20 m`.
- The corresponding macro-average approach rates are `0.5, 1.0, 1.5,
  2.0 m/s`.
- The environment flies only during the 1 s deployment segment. Therefore the
  speeds used by the propulsion curve and `0.5*m*v^2` are respectively `5,
  10, 15, 20 m/s`. The macro-average rate must never be substituted into the
  deployment energy formula.

All candidates must reach their offset within the nine active macro periods
available after a detected load switch.

## 3. Analytic movement reserve

For offset `A` and residual cap `c`, one traversal is split into

`[c, ..., c, A-(n-1)c]`, where `n=ceil(A/c)`.

The frozen 30-period episode contains five residual traversals: initial
establishment, plus return and establishment at each of two switches. For a
leg of length `d`, deployment speed is `v=d/Delta_dep`, and the additional
energy relative to one second of hovering is

`Delta E(d) = [P_fly(v)-P_hover]*Delta_dep + 0.5*m*v^2`.

The maneuver reserve is five times the sum of the leg increments divided by
the 300 s episode duration. This is the same model-specific envelope used in
the completed P1 power audit; it is not fitted to P1-prime replay data.

## 4. Frozen safety rule

Inputs fixed before candidate replay:

- P0 mean and sample standard deviation come from the archived 20-seed power
  audit.
- The P0 one-sided mean UCB95 is `mean + t_0.95,19 * std/sqrt(20)`.
- A fixed `1.0 W` cross-layer reserve covers CPU and communication response.
  This exceeds the archived original-P1 CPU-plus-communication increment.
- The required remaining budget margin equals the archived standard deviation
  of the paired original-P1 minus P0 total-power increment (about `0.984 W`).

A grid point is analytically safe only if

`P0_UCB95 + maneuver_reserve + 1.0 W <= 840 W - paired_delta_std`

and the timing requirement in Section 2 holds.

## 5. Candidate selection

Among analytically safe points, select lexicographically:

1. largest residual offset;
2. largest residual movement cap per macro period.

This prioritizes the strongest geometric intervention and then the fastest
causal response without using any observed P1-prime delay. The selected pair
is frozen for the single replay validation; no fallback grid point may be
chosen after seeing that result.

## 6. Single 20-seed CRN validation

- Policies: P0 and the selected P1-prime only.
- Seeds: the original gate seeds `65000..65019`.
- Checkpoint, micro policy, traffic phases, user mobility, and random streams:
  identical to the original load-switch gate.
- Power feasible: P1-prime one-sided mean UCB95 is at most 840 W.
- Practically effective: paired mean delay improvement exceeds 5 ms and the
  two-sided paired 95% interval excludes zero in the beneficial direction.

If power is infeasible, P1-prime is rejected without retuning and original P1
remains a power-infeasible diagnostic reference. If power is feasible but the
delay gate fails, the handcrafted P1 family is classified as infeasible or
weak at 840 W. In either case, parameters are not adjusted after replay.

## 7. Outputs

```
result_archive/p1_prime_prescreen/
|-- grid.csv
|-- selection.json
|-- prescreen_report.md
|-- prescreen_grid.png
`-- prescreen_grid.pdf
```

