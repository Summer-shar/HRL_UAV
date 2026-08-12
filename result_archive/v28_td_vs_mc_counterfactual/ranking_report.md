# v28 TD-vs-truncated-return ranking report

- Verdict: `INVALID_OR_INCONCLUSIVE`
- Pipeline gates: `{"horizon_stability": true, "crn_and_counts": true, "finite_training_losses": true, "truncated_return_train_actor_agreement_ge_90pct": false, "truncated_return_train_centroid_agreement_ge_90pct": false, "truncated_return_test_actor_correct": false}`

| State region | Arm | Agreement mean [95% CI] | Mean predicted delta | Mean true delta | Classification |
|---|---|---:|---:|---:|---|
| actor | td | 0.5000 [0.4123, 0.5877] | 0.00275176 | 0.00030787 | INCONCLUSIVE |
| actor | truncated_return | 0.5833 [0.4975, 0.6692] | 0.00219515 | 0.00030787 | INCONCLUSIVE |
| centroid | td | 0.4250 [0.3223, 0.5277] | 0.00173416 | -0.00062623 | INCONCLUSIVE |
| centroid | truncated_return | 0.4500 [0.3423, 0.5577] | 0.00099844 | -0.00062623 | INCONCLUSIVE |

Statistical unit is one episode. The six fork states are aggregated within each episode before the confidence interval is computed.
