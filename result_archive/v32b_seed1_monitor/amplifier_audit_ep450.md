# v32b seed-1 drift-amplifier audit through checkpoint 450

Status: **DESCRIPTIVE MID-RUN AUDIT; TRAINING CONTINUES**

This audit uses nine frozen-policy evaluations at episodes 50 through 450.
Correlations below are descriptive time-series associations (`n=9`), not
causal estimates, because policy, replay distribution, queues, and lambda all
change together during training.

## Checkpoint summary

| Episode | Lambda | Delay (ms) | Q-high | Q-low | Completed / arrived | Mean CPU freq. (GHz) | CPU power (W) | Propulsion power (W) | Power UCB95 (W) | U2G (m) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.000 | 965.7 | 0.6 | 17.5 | 99.7% | 8.25 | 237.1 | 594.0 | 835.3 | 0.7 |
| 100 | 0.168 | 1470.2 | 1.4 | 29.0 | 99.4% | 8.23 | 220.1 | 624.9 | 852.6 | 70.0 |
| 150 | 0.713 | 7582.0 | 14.5 | 201.5 | 93.8% | 7.87 | 201.0 | 629.3 | 835.9 | 116.1 |
| 200 | 1.042 | 9986.1 | 21.9 | 274.8 | 91.0% | 7.68 | 193.5 | 641.2 | 841.7 | 140.2 |
| 250 | 1.491 | 12564.8 | 31.2 | 423.3 | 85.6% | 7.32 | 183.8 | 673.7 | 863.3 | 149.9 |
| 300 | 3.149 | 17210.2 | 41.0 | 629.8 | 80.7% | 6.90 | 172.8 | 687.9 | 866.8 | 154.4 |
| 350 | 4.452 | 16995.9 | 36.9 | 863.4 | 74.3% | 6.42 | 149.3 | 681.4 | 836.5 | 148.6 |
| 400 | 4.246 | 16335.5 | 34.1 | 859.8 | 74.2% | 6.43 | 149.1 | 674.9 | 829.9 | 144.6 |
| 450 | 3.928 | 15555.0 | 31.6 | 842.4 | 74.6% | 6.47 | 150.2 | 675.0 | 831.0 | 145.0 |

## Finding A: queue amplification is strongly supported

- Combined queue indicators increase from `18.1` at episode 50 to about
  `874.0` at episode 450.
- The finite-window completed/arrived ratio falls from `99.7%` to `74.6%`.
- Weighted task delay rises from `0.966 s` to `15.555 s`.
- Descriptive Pearson correlations are `corr(Q_total, delay)=0.945` and
  `corr(U2G, Q_total)=0.752`.

These observations support queue accumulation as an amplifier of positioning
error. They do not by themselves prove the narrower claim that U2G service
rate crossed the arrival rate; a stage-level service/arrival counterfactual
would be required for that statement.

## Finding B: strong lambda-to-CPU suppression is supported

- Lambda rises from `0` to a peak of `4.452` at episode 350.
- Mean CPU frequency falls from `8.25 GHz` to about `6.42 GHz`.
- Compute power falls from `237.1 W` to about `149 W`.
- Descriptive correlations are `corr(lambda, CPU frequency)=-0.990` and
  `corr(lambda, compute power)=-0.961`.

This confirms that the priced-CPU response chain is active in the high-lambda
regime. It is consistent with, but does not alone prove, a positive feedback
loop in which positioning drift raises power, lambda suppresses service, and
queues amplify delay.

## Power-account interpretation

Propulsion power rises from `594 W` at episode 50 to roughly `675--688 W` in
the drifted regime, while CPU power is suppressed. Consequently, checkpoints
400 and 450 recover power feasibility mainly through lower compute power, not
through recovery of the positioning policy. Lambda is well below its maximum
of `20`, so clipping is not responsible for the observed plateau.

## Recovery status

Between episodes 350 and 450, lambda declines from `4.452` to `3.928` and delay
falls from `16.996 s` to `15.555 s`. This is partial unwinding, but performance
remains far from episode 50. The run continues to episode 1000 under the frozen
protocol; no repair or checkpoint retuning is introduced mid-run.
