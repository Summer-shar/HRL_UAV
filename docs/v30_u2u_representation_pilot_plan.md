# v30 U2U state-representation pilot plan

> Status: PRE-FREEZE DESIGN
>
> No state, reward, or training code has been changed under this plan yet.

## 1. Scientific question

Does replacing the action-contaminated nominal U2U-rate observation with an
action-independent channel-quality representation improve micro routing and
the macro Critic's ability to distinguish useful positioning actions?

The existing audit established an independent defect:

- execution allocates 3 MHz over the currently active directed links;
- the state assumes all six links are active and therefore encodes 0.5 MHz per
  link;
- executed active-link rate is 2.604 times the encoded nominal rate on average;
- division by 60 Mbps compresses the encoded mean to 0.0529.

## 2. Proposed causal representation

For every off-diagonal directed link, expose spectral efficiency

`eta_ij = log2(1 + p_u2u * beta_0 / (N0 * d_ij^2))`.

`eta_ij` depends only on geometry and fixed physical parameters. It does not
depend on the current routing action, active-link set, or bandwidth allocation.
The environment continues to execute the unchanged physical rate

`R_ij = (B_total / K_active) * eta_ij`.

Normalize `eta` by the median off-diagonal spectral efficiency at the fixed
initial anchor positions. This constant depends only on frozen geometry and
channel parameters, maps the initial topology to order one, and does not use
training outcomes. Do not clip values below 3; the zero-training audit must
verify that the physical trajectory distribution stays inside this range.
Diagonal entries remain zero.

### 2.1 Micro observation

- Replace the `N x N` nominal-rate block by normalized `eta` with the same
  matrix shape.
- Build the deterministic manager goal from availability and normalized `eta`,
  not nominal rate.
- Remove nominal six-link rate from the transmit-pressure denominator.
- Add the causal previous-frame active-link fraction so the
  worker observes recent contention without seeing the current action result.
- Use the exact previous resource-frame active-link count, without an EMA. At
  reset and after an empty previous frame, use six active directed links as the
  conservative fallback. Derive the pressure capacity from `eta`, total
  bandwidth, and this causal count.

### 2.2 Macro observation

- Replace `norm_u2u_mean` by the corresponding mean normalized spectral
  efficiency.
- Keep relative UAV coordinates and historical flow features unchanged.
- Do not add current-action active-link rates to the macro state.

Both policy layers must be retrained. Reusing old checkpoints after changing
feature semantics is prohibited even if the vector dimension is unchanged.

## 3. Engineering prerequisites

1. Isolate replay-sampling RNG from exploration/environment RNG.
2. Add a fixed-replay one-step parity test for targets, losses, and gradients.
3. Add unit tests for `R = B/K * eta`, action independence, distance
   monotonicity, normalization bounds, and causal contention-state updates.
4. Run a zero-training state audit and report distributions, clipping rates,
   and sensitivity to 20/40/60 m distance changes before training.
5. Verify numerically that `eta * realized_bandwidth` reconstructs the executed
   environment rate. Report correlation and maximum relative error.

## 4. Clean experimental arms

For causal attribution, both arms start from clean initialization and use the
same current code, seeds, budgets, and dual protocol:

| Item | Control | Treatment |
|---|---|---|
| U2U state block | Current nominal six-link rate | Spectral efficiency + causal contention |
| Delay target | 1 step | 1 step |
| Neutral coverage | Off | Off |
| Actor objective | Standard TD3 Q1 | Standard TD3 Q1 |
| Power control | v24 system dual, 840 W | Same |
| Micro SAC | Retrain | Retrain |
| Macro TD3 | Fresh start | Fresh start |

Reusing historical v24 as the sole control is not sufficient because both the
RNG plumbing and micro training path change. Historical v24 remains a
descriptive reference only.

## 5. New rulers before joint interpretation

The old fixed state vectors cannot be reused after changing observation
semantics. Before reading the joint result:

1. rebuild the actor-region and centroid-region probe manifest from common raw
   environment snapshots and re-encode it under each arm;
2. for each final checkpoint, replay a same-checkpoint deterministic centroid
   reference on 20 common seeds;
3. retain external-trace hashes and episode-level pairing;
4. evaluate both Actor and centroid state regions, not only the centroid region.

## 6. Staged gates

### Gate A: representation validity (zero training)

- no dependence on current routing action or active-link mask;
- monotonic decrease with U2U distance;
- no unintended clipping or near-constant feature block;
- exact reconstruction of executed rate when the realized `K_active` is later
  supplied for audit purposes.
- report the 20 m feature-response multiplier relative to the old encoding as a
  descriptive diagnostic. It is not a hard success gate because arbitrary
  rescaling alone can manufacture any desired multiplier.

### Gate B: micro-only value

Hold the macro policy to deterministic centroid tracking and compare freshly
trained micro SAC control/treatment checkpoints on 20 common seeds. Report
weighted delay, U2U service time, link-level flows, and power. This isolates
whether the new representation improves routing before involving trajectory
learning.

### Gate C: joint cross-layer value

Run seed=1, 100-episode joint pilots only after Gate B. Required readouts:

- fixed-probe Q1 ranking in both Actor and centroid regions;
- treatment-control paired endpoint delay;
- learned-policy versus same-checkpoint centroid-reference delay;
- mean centroid distance, path length, lambda trajectory, and power UCB95.

The final thresholds and verdict table remain OPEN until Gate A fixes the exact
normalization and causal contention statistic. No formal run starts from this
PRE-FREEZE plan.
