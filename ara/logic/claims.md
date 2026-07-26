# Claims

## C01: The locked OFA baseline reproduction is complete
- **Statement**: The confirmatory Phase 2 scope contains 24 accepted training
  runs and 156 independently verified method-target-seed evaluations.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Any missing manifest key, invalid target-label
  access, mismatched score/checkpoint hash, failed reload, or independently
  recomputed metric mismatch.
- **Proof**: [E01, `rebuttal/artifacts/phase2_baselines/analysis/artifact_audit.json`]
- **Dependencies**: []
- **Tags**: reproduction, completeness, audit

## C02: ARC and IA-GGAD lead RECAP in Settings A and B
- **Statement**: Under the locked source/target splits, ARC and IA-GGAD have
  higher dataset-macro AUROC and AUPRC than label-free RECAP in Settings A
  and B.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: A correctly recomputed three-seed macro reverses
  either comparison.
- **Proof**: [E01]
- **Dependencies**: [C01]
- **Tags**: comparative-performance, supervised-baseline

## C03: RECAP is AUROC-competitive under citation-only transfer
- **Statement**: In Setting C, label-free RECAP has higher dataset-macro AUROC
  than ARC and IA-GGAD, while its AUPRC remains slightly lower than both.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: A correctly recomputed Setting-C dataset macro
  does not preserve this ordering.
- **Proof**: [E01]
- **Dependencies**: [C01]
- **Tags**: cross-domain-transfer, label-free

## C04: Deterministic CPU sparse aggregation preserves UNPrompt target inference
- **Statement**: Moving only UNPrompt's frozen target sparse aggregation to
  CPU removes CUDA reduction nondeterminism without changing its equivalent
  dense result.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: The dense-equivalence test fails, repeated
  target inference differs, or a formal checkpoint reload exceeds the locked
  tolerance.
- **Proof**: [E02, `rebuttal/gate_reports/PHASE2_BASELINE_GATES.md`]
- **Dependencies**: []
- **Tags**: determinism, compatibility, UNPrompt

## C05: The user-revised four-baseline A/B/C comparison is complete
- **Statement**: UNPrompt and AnomalyGFM-ZS now have independently verified
  three-seed results in Settings B/C, adding 12 training runs and 60
  evaluations without modifying the original primary artifacts.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Any supplementary manifest key, checkpoint,
  frozen score, label event, calibration lock, or recomputed metric fails
  independent validation.
- **Proof**: [E03, `rebuttal/artifacts/phase2_bc_supplement/analysis/artifact_audit.json`]
- **Dependencies**: [C01]
- **Tags**: scope-completion, reproduction, audit

## C06: RECAP exceeds the added zero-context baselines in B/C
- **Statement**: RECAP has higher dataset-macro AUROC and AUPRC than UNPrompt
  and AnomalyGFM-ZS in both Settings B and C under the locked splits.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Correct three-seed recomputation reverses any of
  the four metric comparisons.
- **Proof**: [E03]
- **Dependencies**: [C05]
- **Tags**: comparative-performance, zero-context, cross-domain-transfer
