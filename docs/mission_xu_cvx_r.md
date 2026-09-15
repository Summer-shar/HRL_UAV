# Xu-CVX-R: fixed-CPU routing specialization, not a full Xu-SCA replication

## Frozen protocol before evaluation

Reference: Wenlong Xu et al., *Trajectory Planning and Resource Allocation for Multi-UAV Cooperative Computation*, IEEE TCOM 72(7), 2024, DOI 10.1109/TCOMM.2024.3361536. Relevant equations: (13)-(17), (29)-(34), Algorithm 1.

Repository HEAD at implementation: `9f98e5142b563e134a2f90bd2b6f555fd7ce4bf1`; existing uncommitted mission diagnostics are preserved. Source/result fingerprints are captured by the new entry point. No commit or push.

The reference jointly chooses association, per-task routing and per-task CPU frequency, with completion inside an independent frame. Current mission V1 uses fixed source association, continuous queues, aggregate priority/source routing, FIFO and priority-demand CPU. Its SAC action does not control CPU. Copying the old optimizer and silently discarding its CPU decision would be inconsistent; enabling the old CPU override would change the common service rule. Neither is done here.

**This baseline is called Xu-CVX-R, not Xu-SCA or a reproduction of the complete paper algorithm.** It retains the reference's upload + maximum branch-completion epigraph and delay/energy objective. With fixed service-frequency estimates, the `rho/f` and `rho*f^2` terms become affine in routing. Each fixed-support subproblem is a linear program, solved by CVXPY/CLARABEL. There are no fictitious SCA iterations. This is a limited, explicitly adapted model-based routing comparator; the original joint-CPU SCA comparison remains a separate unresolved interface question.

## Information and prediction model

The controller only receives the exact 68D float32 SAC observation plus static system constants. It has no environment handle and cannot read parent-task identities, individual sizes, locations, future arrivals, load schedules or RNG. Inverse observation encoding recovers compute/transfer queues, current per-source/priority arrival bits, nominal CPU capacity and reference U2U quality, to float32 accuracy.

At each 0.5-second decision, each `(priority p, source i)` current-arrival aggregate is a virtual batch of `L[p,i]` Mbit. This differs from the paper's per-task model and does not reconstruct missing task counts/sizes. A row with zero new arrivals keeps its previous executed row, since earlier uploads may still reach routing. The initial empty-row choice is Local, recorded explicitly.

Let `mu[j]` be nominal CPU processing capacity in Mbit/s, `F[j]` the configured CPU Hz, and `C` cycles/bit. For an isolated newly routed batch, the compute service-time coefficient is `L/mu[j]`. This assumes full CPU frequency while that chunk is served, **not simultaneous exclusive full CPU allocation to all batches**. It ignores competition between newly admitted batches and future arrivals. Waiting estimates use current high-only queues for high priority and high+low queues for low priority. They omit future priority preemption, release times of existing U2U chunks and packet order not present in the observation. Consequently the predictor is not exact FIFO latency or an exact finite-horizon optimizer of J.

Recover geometric gain using `g=(2^eta_ref-1)*N_eff*b_ref/P_ref`. The prediction rates use the declared all-six-directed-link allocation: `b=B/6`, `p=P_uav/2`, `R=b*log2(1+p*g/(N_eff*b))`. These are **hypothetical all-active rates**, not observed actual service rates. This conservative reference avoids using unobserved future active queues, but can overestimate transmission time compared with sparse actual activity. The same limitation must accompany performance claims. Actual environment rates/power still use original per-physical-step backlog activity and are saved independently.

For candidate support S, decision `rho[j]` sums to one, is nonnegative, and is zero outside S:

```
w[p,i,j] = Q_compute_ahead[p,j]/mu[j]
           + 1[j!=i] * Q_transfer_ahead[p,i,j]/R[i,j]
c[p,i,j] = L[p,i]/mu[j] + 1[j!=i]*L[p,i]/R[i,j]
e[p,i,j] = kappa * C * (L[p,i]*1e6) * F[j]^2
           + 1[j!=i]*p[i,j]*L[p,i]/R[i,j]
T[p,i] >= w[p,i,j] + c[p,i,j]*rho[j], j in S
min (delay_weight/holding_norm_task_s)*priority_weight[p]*T[p,i]
    + (energy_weight/energy_norm_j)*sum_j e[p,i,j]*rho[j]
```

Upload and propulsion terms are constant with respect to this instantaneous routing decision and omitted from the solve, not from measured J/energy. CPU energy is a full-frequency work estimate, not an overwrite of actual demand-dependent CPU energy. Priority weights and J normalization constants are read from the common effective configuration/environment defaults and locked; no new reward weights or queue penalty are fitted.

All seven nonempty destination supports are solved for each of the six rows (42 independent small LPs in one separable CVXPY problem). This avoids charging a queue's waiting time to a branch that is not used. The support enumeration is an explicit addition needed for persistent-queue waiting; it is not Algorithm 1 from the paper. Rows are independently optimized under the all-active link prediction, so no claim of globally joint routing optimality is made. Prefer fewer destinations on numerical ties; no other direction or role rule is imposed.

Solver tolerances and postprocessing are numerical, not performance tuning: CLARABEL absolute/relative gap and feasibility 1e-8; max 200 iterations; route entries below 1e-8 removed, rows normalized. Reject nonfinite solutions, material infeasibility, or solver failure; no fallback to a hand-tuned rule. Record residuals, cleanup magnitude, selected supports, predicted objective, solve time and actual executed routes. The same simplex action interface is used as SAC.

## Bounded experiment

- Existing mission V1, 6 MHz only, all other configuration identical to `results/mission_paper_comparison_20260914/manifest.json` base.
- Same three frozen PPO120 policies (201/202/203), same SAC101 source provenance; no learning, no replay, no optimizer steps. The SCA-derived comparator replaces SAC routing, not the macro policy.
- Scenes 33600/33601 only, each 900 seconds continuous, no 300/600 reset; six new records. No reserved scene, new load point, 3 MHz or parameter search.
- First short tests and a 20-second prefix integration; reproduce existing SAC prefix and pair exogenous processes, verify frozen state and conserved tasks/routes/time. Then run six comparisons if implementation checks pass, irrespective of which method wins.
- Reuse six existing SAC base records only after effective configuration, source and transaction checks. Keep source files/hash and reuse labels.
- Report actual J, holding/energy, high/low and weighted completed delay, completion and unfinished counts, mean/terminal queues, actual U2U resource/energy/throughput, full power, solve times. Report each scene and each PPO initialization; do not claim independent full-method training or final generalization.

No performance result is asserted in this protocol. New independent output: `results/mission_xu_cvx_r_20260914`. Stop after the six prescribed comparisons; no post-result revision of the predictor.
