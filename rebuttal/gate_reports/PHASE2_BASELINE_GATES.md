# Phase 2 OFA Baseline Gate Report

Date: 2026-07-26

Status: **PASS — formal baseline manifest may start**

## Scope and provenance

- Locked primary scope: 24 training runs and 156 final evaluations.
- Seeds: 0, 1, 2.
- Official upstream revisions and archive SHA-256 values are recorded in
  `rebuttal/baselines/upstream_manifest.json`.
- All 12 project MAT files are byte-identical to the official ARC copies.
- Formal runtime: Python 3.12.3, PyTorch 2.11.0+cu128, PyG 2.7.0,
  einops 0.8.1, RTX PRO 6000 Blackwell.

## Automated compatibility tests

Six tests passed:

1. the manifest contains exactly 24 runs and 156 evaluations;
2. target labels cannot be read before an immutable score freeze;
3. IA-GGAD edge-wise sparse affinity equals the released dense masked
   expression;
4. sparse normalized GraphConv equals its dense expression;
5. UNPrompt's checkpointed row-block contrastive loss equals the full
   all-negative loss;
6. AnomalyGFM's sparse ARPACK top-8 SVD agrees with deterministic full SVD
   within `atol=2e-4, rtol=2e-4`.

## Setting A seed-0 one-epoch smoke gates

The smoke setting uses one epoch for each training stage only; metrics are
sanity checks and are not formal results.

| Method | Targets | Train seconds | Preparation seconds | Peak GiB | Reload max abs diff | Label audit |
|---|---:|---:|---:|---:|---:|---|
| ARC | 8 | 0.636 | 2.090 | 4.038 | 0 | PASS |
| IA-GGAD | 8 | 0.984 | 2.000 | 5.005 | 3.81e-6 | PASS |
| UNPrompt | 8 | 1.444 | 43.452 | 3.197 | 5.96e-7 | PASS |
| AnomalyGFM-ZS | 8 | 0.552 | 3.165 | 0.896 | 4.29e-6 | PASS |

All 32 target score vectors were finite, all AUROC/AUPRC computations
completed, and every method checkpoint reproduced its saved scores within the
locked `1e-5` reload tolerance.

UNPrompt's preparation time includes the first generation of all exact rank-8
SVD caches. AnomalyGFM's initial released full-dense SVD attempt was stopped as
redundant preprocessing before producing a smoke result; direct top-8 ARPACK
preprocessing reduced its accepted preparation time to 3.165 seconds.

## Formal execution decision

Proceed with the immutable manifest at:

`rebuttal/artifacts/phase2_baselines/manifest.json`

The formal directory was empty at gate acceptance: 0/24 training runs and
0/156 evaluations.

