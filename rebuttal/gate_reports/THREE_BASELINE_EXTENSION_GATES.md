# Three-Baseline Extension Acceptance Gates

Status: **PASS**

## Scope and provenance

- Locked manifest: 81 training runs and 126 final evaluations.
  SHA-256 values.
- All twelve MAT files in OWLEYE's official `raw_data.zip` matched the RECAP
  dataset files byte for byte.
  OWLEYE smoke passed.

## Numerical gates

| Gate | Maximum difference | Result |
|---|---:|---|
| DiffGAD exact sparse structure loss vs dense forward | 5.6843e-14 | PASS |
| DiffGAD exact sparse structure loss vs dense gradient | 5.6843e-14 | PASS |
| OWLEYE chunked vs full-query inference | 1.1921e-07 | PASS |
| OWLEYE effective normalization vs released tau=1 algebra | 0 | PASS |

The released OWLEYE normalization multiplier is exactly 1 under the locked
`tau=1` configuration after cancellation, so the quadratic pair-distance
materialization is unnecessary.

## Smoke artifact gates

All five smoke artifacts had finite full-population scores, passing label
audits, checkpoints, and reload evidence. Smokes were written to a separate
artifact root and were never used in formal aggregates.
