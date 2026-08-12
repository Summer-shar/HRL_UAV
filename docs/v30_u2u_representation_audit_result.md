# v30 U2U representation zero-training audit result

> Status: GATE A PASS
>
> Date: 2026-08-07
>
> This result validates the proposed observation semantics only. It does not
> establish a learning or end-to-end performance improvement.

## 1. Frozen audit setup

- Checkpoint: v24 dual-on final checkpoint;
- training: none;
- evaluation episodes: 5;
- directed-link samples: 81,000, of which 36,309 were active;
- proposed feature:
  `eta_ij = log2(1 + p_u2u * beta_0 / (N0 * d_ij^2))`;
- normalization constant: the median off-diagonal `eta` at the fixed initial
  UAV anchors.

The resulting reference value is
`eta_ref = 6.490070890987665 bit/s/Hz`.

## 2. Formula and environment equivalence

The U2U channel used by the environment is exactly `beta_0 / d^2`; no
additional fading term is present in this path. For every active link, the
audit reconstructed the executed rate as

`R_reconstructed = realized_bandwidth * eta`.

The reconstruction result is exact within floating-point arithmetic:

| Metric | Result |
|---|---:|
| Pearson correlation | 1.000000 |
| Maximum absolute error | 0 bit/s |
| Maximum relative error | 0 |

Therefore `eta` is an action-independent physical primitive from which the
executed U2U rate is recovered exactly once realized bandwidth is supplied.

## 3. Dynamic range

The sampled UAV separation p05/p50/p95 was
301.72/339.15/421.81 m. Over the same samples:

| Feature | p05 | p50 | p95 |
|---|---:|---:|---:|
| Spectral efficiency (bit/s/Hz) | 5.8381 | 6.4584 | 6.7925 |
| Old nominal-rate encoding | 0.04865 | 0.05382 | 0.05660 |
| New anchor-normalized encoding | 0.89954 | 0.99513 | 1.04659 |

The maximum new feature value was 1.05920 and the fraction above 3 was zero.
The normalization therefore maps the operating geometry to order-one values
without clipping.

## 4. Position sensitivity

At the typical 300--400 m separation, moving 20 m closer changes the old
feature by only 0.00121--0.00165, while it changes the new feature by
0.02246--0.03042. Across the same physical perturbation, the new encoding has
18.49 times the numerical response of the old encoding.

This multiplier is descriptive rather than a hard gate: a scale factor alone
cannot prove that a representation improves learning. Here its relevance is
supported by the independent semantic result above: the new feature removes
the fixed-six-link bandwidth assumption and exposes the exact physical channel
quality without depending on the current action.

## 5. Verdict

Gate A passes all pre-registered validity checks:

- action-independent channel primitive: PASS;
- distance monotonicity: PASS;
- no near-zero compression or unintended clipping: PASS;
- executed-rate reconstruction with realized bandwidth: PASS;
- fixed, result-independent normalization: PASS.

The audit establishes that the old U2U state block is both semantically
misaligned with dynamic bandwidth allocation and numerically compressed. It
does not yet establish that this defect caused the macro Critic failure.

The next admissible experiment is Gate B: freshly train old- and new-feature
micro SAC arms under deterministic centroid tracking, then compare them with
20 common evaluation seeds. Joint TD3 training remains blocked until Gate B is
specified, frozen, and passed.

## 6. Artifacts

- `result_archive/v30_u2u_representation_audit/summary.json`
- `result_archive/v30_u2u_representation_audit/link_samples.csv`
- `result_archive/v30_u2u_representation_audit/distance_response.csv`
- `result_archive/v30_u2u_representation_audit/v30_u2u_representation_audit.png`
- `result_archive/v30_u2u_representation_audit/v30_u2u_representation_audit.pdf`
