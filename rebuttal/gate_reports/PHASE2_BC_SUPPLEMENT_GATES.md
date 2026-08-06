# Phase 2 B/C Baseline Supplement Gate Report

Date: 2026-07-26

Status: **PASS — supplementary formal manifest may start**

## Locked scope

- Methods: UNPrompt and AnomalyGFM-ZS.
- Settings: B and C.
- Seeds: 0, 1, 2.
- Formal scope: 12 training runs and 60 evaluations.
- Original 24-run primary artifact root remains immutable.
- Supplement protocol SHA-256:
  `742857e6b82e22a797186b1132a3d4788f3baa104081b8682727df1f54265d58`.

## Automated tests

Ten tests passed:

1. primary manifest remains exactly 24 runs and 156 evaluations;
2. supplementary manifest is exactly 12 runs and 60 evaluations;
3. B/C supplement contains only UNPrompt and AnomalyGFM-ZS with locked splits;
4. target labels are blocked before immutable score freeze;
5. sparse IA affinity equals its dense expression;
6. sparse normalized GraphConv equals its dense expression;
7. exact blocked UNPrompt contrastive loss equals the full denominator;
8. deterministic UNPrompt target aggregation equals dense and repeats bitwise;
9. sparse ARPACK rank-8 SVD agrees with full SVD;
10. supplementary aggregation produces 20 dataset rows and six macro rows with
    three-seed population statistics.

## One-epoch end-to-end smokes

| Setting | Method | Targets | Train seconds | Preparation seconds | Evaluation seconds | Reload max diff | Label audit |
|---|---|---:|---:|---:|---:|---:|---|
| B | UNPrompt | 5 | 1.757 | 1.021 | 0.218 | 0 | PASS |
| B | AnomalyGFM-ZS | 5 | 0.541 | 1.456 | 0.053 | 3.81e-6 | PASS |
| C | UNPrompt | 5 | 0.832 | 0.971 | 0.415 | 0 | PASS |
| C | AnomalyGFM-ZS | 5 | 0.601 | 1.580 | 0.102 | 3.81e-6 | PASS |

All 20 target score vectors were finite and frozen before metric labels were
loaded. All four label audits had zero invalid events. Feature caches were
reused only through version/raw-hash-validated cache loaders; no checkpoint,
calibration, score, or metric was reused.

## Formal decision

Proceed from a clean supplement root at 0/12 training runs and 0/60
evaluations:

`rebuttal/artifacts/phase2_bc_supplement/`
