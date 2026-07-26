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

## H04: Expand a completed protocol through an isolated supplement
- **Rationale**: A separate manifest, artifact root, and protocol preserve the
  forensic integrity of already accepted results while allowing a
  user-requested scope revision.
- **Provenance**: ai-suggested
- **Sensitivity**: low
- **Code ref**: [`rebuttal/baselines/baseline_bc_runner.py`,
  `rebuttal/baselines/baseline_bc_analysis.py`]

## H05: Evaluate DOMINANT structure error without an N×N matrix
- **Rationale**: Expanding the squared dot-product decoder row error through
  `ZᵀZ`, positive-edge dot products, and node degree is exactly equal to the
  dense decoder error while reducing memory below quadratic.
- **Provenance**: ai-suggested
- **Sensitivity**: low
- **Code ref**: [`rebuttal/ofo_baselines/models.py`,
  `rebuttal/ofo_baselines/test_ofo_baselines.py`]

## H06: Estimate dense non-edge structure terms with locked weighting
- **Rationale**: Keeping every positive edge and sampling one deterministic
  row-wise non-edge per positive edge with inverse-probability weighting
  preserves a scalable estimate of the released dense AnomalyDAE/ADA-GAD
  objectives without labels.
- **Provenance**: ai-suggested
- **Sensitivity**: medium
- **Code ref**: [`rebuttal/ofo_baselines/common.py`,
  `rebuttal/ofo_baselines/runner.py`]

## H07: Scale reload tolerance to anomaly-score magnitude
- **Rationale**: A fixed `1e-5` absolute threshold rejects harmless CUDA
  scatter last-bit differences when reconstruction scores are large. The
  pre-result threshold `1e-5 + 5e-6 × max(abs(score))` remains tight relative
  to score scale and is recorded per run.
- **Provenance**: ai-suggested
- **Sensitivity**: medium
- **Code ref**: [`rebuttal/ofo_baselines/runner.py`]

## H08: Recompute every final metric from frozen arrays
- **Rationale**: Requiring the exact manifest and rebuilding AUROC/AUPRC from
  hashed score/query arrays catches missing runs, label-population drift, and
  aggregation errors independently of the training runner.
- **Provenance**: ai-suggested
- **Sensitivity**: low
- **Code ref**: [`rebuttal/ofo_baselines/analysis.py`]
