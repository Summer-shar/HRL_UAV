# v32a clean online n-step verdict

- Frozen verdict: **NSTEP_REJECTED**
- Replay/target-noise hash gate: `True`
- Centroid-probe n5-n1 hierarchical 95% CI: `[0.033333333333333326, -0.0027777777777777805, 0.07666666666666666]`
- Delay n5-n1 hierarchical 95% CI (ms): `[178.50020338288252, -4.478869998033372, 387.01873536177635]`
- n=5 power UCB95 by training seed (W): `{1: 839.3000899539388, 42: 825.2313209319906, 123: 833.4140892692527}`

Only FULL_PASS authorizes n=5 in v32b. Every other valid verdict closes the n-step line and retains n=1.
