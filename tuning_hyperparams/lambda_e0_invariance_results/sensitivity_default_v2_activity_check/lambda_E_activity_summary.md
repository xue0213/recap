# lambda_E Activity Check

Verdict: **PASS**

This check scans the original sensitivity training diagnostics and measures whether
`L_var` was active. If `Lvar` and `Lvar_active` stay near zero, the
`lambda_E * L_var` term was effectively dormant in the logged optimization states.

## Summary

- Result dir: `/autodl-fs/data/recap/tuning_hyperparams/sensitivity_results/sensitivity_default_v2`
- Runs scanned: `55`
- Diagnostic points: `1815`
- Max Lvar: `4e-06`
- Max Lvar active ratio: `0.001`
- Tolerances: `Lvar <= 1e-05`, `active <= 0.001`

## Worst Runs

| run_id | diagnostic points | max Lvar | max active | mean Lvar | mean active |
|---|---:|---:|---:|---:|---:|
| cluster_init_gain__3p5 | 33 | 4e-06 | 0.0002 | 7.2727273e-07 | 6.6666667e-05 |
| num_hops__1 | 33 | 3e-06 | 0.001 | 1.5151515e-07 | 6.0606061e-05 |
| cluster_init_gain__2p5 | 33 | 3e-06 | 0.0002 | 6.969697e-07 | 6.6666667e-05 |
| cluster_init_gain__3p0 | 33 | 3e-06 | 0.0002 | 7.2727273e-07 | 6.6666667e-05 |
| lambda_H__0p2 | 33 | 3e-06 | 0.0002 | 6.3636364e-07 | 6.6666667e-05 |
| lambda_H__0p3 | 33 | 3e-06 | 0.0002 | 6.969697e-07 | 6.6666667e-05 |
| lambda_H__0p5 | 33 | 3e-06 | 0.0002 | 6.6666667e-07 | 6.6666667e-05 |
| lambda_H__0p8 | 33 | 3e-06 | 0.0002 | 7.5757576e-07 | 6.6666667e-05 |
| tau_c__0p32 | 33 | 3e-06 | 0.0002 | 6.969697e-07 | 6.6666667e-05 |
| tau_c__0p35 | 33 | 3e-06 | 0.0002 | 7.2727273e-07 | 6.6666667e-05 |
