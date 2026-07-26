# Heuristics

## H01: Compute only the retained SVD subspace
- **Rationale**: AnomalyGFM retains eight singular components, so deterministic
  sparse ARPACK top-8 avoids computing and discarding the full dense basis.
- **Provenance**: ai-suggested
- **Sensitivity**: low
- **Code ref**: [`rebuttal/baselines/baseline_common.py`]

## H02: Make frozen sparse target aggregation deterministic
- **Rationale**: CUDA sparse reduction order can change last bits that are
  amplified by graph-level min-max scoring. CPU sparse aggregation is
  deterministic while the learned encoder remains unchanged.
- **Provenance**: ai-suggested
- **Sensitivity**: low
- **Code ref**: [`rebuttal/baselines/baseline_models.py`]

## H03: Calibrate fusion weights on source scores once
- **Rationale**: Source-only seed-0 selection prevents target-specific released
  settings from leaking target labels or target identities into comparison.
- **Provenance**: ai-suggested
- **Sensitivity**: medium
- **Code ref**: [`rebuttal/baselines/baseline_runner.py`]
