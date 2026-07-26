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

## C07: The eight-method 12-dataset OFO reproduction is complete
- **Statement**: The locked OFO scope contains 288 accepted
  method-dataset-seed training/inference runs and 576 independently
  recomputed AUROC/AUPRC values, with complete saved-model, hash, label-access,
  and reload evidence.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Any missing Cartesian-product key, invalid
  pre-freeze label access, raw/score/mask hash mismatch, missing model, failed
  reload threshold, or recomputed metric mismatch.
- **Proof**: [E04, `rebuttal/artifacts/ofo_12_baselines/formal/analysis/global_audit.json`]
- **Dependencies**: []
- **Tags**: OFO, reproduction, completeness, audit

## C08: RECAP led the original four-baseline full-graph unsupervised OFO group
- **Statement**: RECAP-OFO has higher 12-dataset macro AUROC and AUPRC than
  the original DOMINANT, AnomalyDAE, CoLA, and ADA-GAD set under the
  label-free, full-graph evaluation regime.
- **Status**: revised
- **Revision**: The later extension added GUIDE, whose 75.19/33.09 macro
  exceeds RECAP's 71.08/23.61. The statement remains true only for the
  original four-baseline set and is no longer a claim of leadership over the
  expanded unsupervised group.
- **Provenance**: ai-suggested
- **Falsification criteria**: Correct three-seed full-graph recomputation
  reverses either metric comparison against any of the four methods.
- **Proof**: [E04, E06]
- **Dependencies**: [C07, C11]
- **Tags**: OFO, comparative-performance, unsupervised

## C09: XGBGraph leads the reproduced supervised OFO group
- **Statement**: XGBGraph has the highest 12-dataset macro AUROC and AUPRC
  among GCN, GAT, BWGNN, and XGBGraph on their stratified 40% target-test
  populations.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Correct three-seed test-split recomputation
  places another supervised method above XGBGraph on either macro metric.
- **Proof**: [E04]
- **Dependencies**: [C07]
- **Tags**: OFO, comparative-performance, supervised

## C10: The full user-revised RECAP experiment protocol is complete
- **Statement**: The consolidated scope contains 450 successful training runs
  and 720 final evaluations, with seeds 0/1/2 in every required
  method-setting-dataset cell and no missing experiment or required rerun.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Any required cell is absent or duplicated, any
  source artifact audit fails, any metric lies outside `[0,1]`, or a
  seed-first/pair-first recomputation disagrees with the consolidated records.
- **Proof**: [E05,
  `rebuttal/artifacts/protocol_completion/analysis/consistency_audit.json`]
- **Dependencies**: [C01, C05, C07, C11]
- **Tags**: protocol-completion, consistency, aggregation, audit

## C11: The three-baseline extension is complete
- **Statement**: The locked extension contains 81 accepted training runs and
  126 independently verified evaluations: 72 full-graph OFO runs for DiffGAD
  and GUIDE and nine shared-source OWLEYE runs covering 54 OFA targets.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Any missing manifest key, invalid label event,
  mismatched score/checkpoint hash, failed reload, or independently
  recomputed metric mismatch.
- **Proof**: [E06,
  `rebuttal/artifacts/three_baseline_extension/formal/analysis/global_audit.json`]
- **Dependencies**: []
- **Tags**: reproduction, completeness, audit, baseline-extension

## C12: OWLEYE is source-label supervised and target-label-free
- **Statement**: OWLEYE's released zero-shot path explicitly uses source
  normal/anomaly labels for source pattern extraction and training, while the
  reproduced target score vectors are frozen before target labels are read.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: The pinned release trains without source labels,
  or any accepted target-label audit records access before score freeze.
- **Proof**: [E06, `rebuttal/THREE_BASELINE_EXTENSION_PROTOCOL.md`]
- **Dependencies**: [C11]
- **Tags**: supervision, zero-shot, OWLEYE, label-isolation

## C13: GUIDE exceeds RECAP on the expanded unsupervised OFO macro
- **Statement**: Under the same full-node, label-free, 12-dataset evaluation
  population, GUIDE reaches 75.19% AUROC and 33.09% AUPRC versus RECAP-OFO's
  71.08% and 23.61%.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Correct seed-first three-seed recomputation
  reverses either macro comparison.
- **Proof**: [E06, E05]
- **Dependencies**: [C10, C11]
- **Tags**: OFO, comparative-performance, unsupervised, GUIDE
