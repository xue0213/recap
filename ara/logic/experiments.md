# Experiments

## E01: OFA baseline reproduction

- Settings: A, B, C
- Methods: ARC, UNPrompt, AnomalyGFM-ZS, IA-GGAD
- Seeds: 0, 1, 2
- Scope: 24 training runs, 156 final evaluations
- Metrics: AUROC, AUPRC, dataset macro, Setting-C domain macro
- Acceptance: score/checkpoint hashes, label audit, checkpoint reload, and
  independent metric recomputation must pass.

## E02: Compatibility and determinism gates

- Dense-versus-sparse affinity and GraphConv equivalence
- Exact blocked UNPrompt contrastive loss versus the full denominator
- Sparse ARPACK top-8 SVD versus full SVD
- Deterministic UNPrompt target aggregation versus dense and repeated output

## E03: Eight-method 12-dataset OFO baseline reproduction

- Methods: GCN, GAT, BWGNN, XGBGraph, DOMINANT, AnomalyDAE, CoLA, ADA-GAD
- Datasets: PubMed, Cora, CiteSeer, ACM, Flickr, BlogCatalog, Facebook,
  Weibo, Reddit, Questions, YelpChi, Amazon
- Seeds: 0, 1, 2
- Scope: 288 independent training/inference runs and 288 frozen score vectors
- Metrics: AUROC, AUPRC, dataset macro, domain macro, timing and resources
- Acceptance: exact manifest coverage, raw/score/mask hashes, label-access
  order, model persistence, checkpoint reload, and independent metric
  recomputation must pass.

## E05: Protocol-wide completion and consistency audit

- Scope: RECAP-OFO, RECAP-OFA A/B/C, eight OFO baselines, and four OFA
  baselines in A/B/C
- Evidence: 369 training runs, 594 final evaluations, 90 stability pair
  records, 486 diagnostic rows, and 225 RECAP checkpoints
- Aggregation: dataset macro within seed, Setting-C domain macro within seed,
  stability macro within seed pair, and timing macro within seed
- Acceptance: exact cell/seed coverage, finite unit-interval metrics, five
  passing source artifact audits, explicit evaluation-population strata,
  deterministic table regeneration, and all protocol/equivalence tests
