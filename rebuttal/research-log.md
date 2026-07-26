# RECAP Phase 1 Research Log

Append-only decision and experiment record.

| # | Date | Type | Summary |
|---|------|------|---------|
| 1 | 2026-07-26 | bootstrap | Reviewed the manuscript, full experiment protocol, server environment, all 12 Phase 1 datasets, and current code. Identified a paper-implementation drift in the dirty ANN worktree and decided to execute Phase 1 from a clean worktree based on commit `c94c4d7`. |
| 2 | 2026-07-26 | protocol | Locked OFO to 11 datasets by excluding Questions; OFA remains unchanged. Locked 42 training runs, 87 final evaluations, 87 stability comparisons, seeds 0/1/2, paper hyperparameters, strict label isolation, exact KNN, and population standard deviation. |
| 3 | 2026-07-26 | infrastructure | Created a 20-minute heartbeat to audit progress, failures, label isolation, result completeness, and outer-loop reflection checkpoints. |
| 4 | 2026-07-26 | infrastructure | Added a resumable 42-run manifest runner, atomic checkpoints at epochs 25/50/75/100, post-score label evaluation, compact fixed-epoch diagnostics, final community export, exact C×C soft co-assignment, automatic raw-result aggregation, and completion validation. |
| 5 | 2026-07-26 | inner-loop | Six infrastructure unit tests passed: manifest scope/counts, explicit paper defaults, `tau_c` assignment formula, label-free graph boundary, exact co-assignment identity, and equivalence of explicit `lambda_E=0` with the legacy zero default. |
| 6 | 2026-07-26 | inner-loop | Full 12-dataset preflight passed in 27.87 seconds. Locked raw SHA-256 hashes, node/feature/anomaly counts, adjacency-nnz and unique-undirected edge counts. Confirmed Amazon has 10,224 nodes and Questions has 48,921 nodes. |
| 7 | 2026-07-26 | outer-loop | Infrastructure risk is now controlled: Phase 1 runs from the clean paper worktree, ANN is absent from the execution code, labels are excluded from model-facing graphs, and deterministic caches are versioned. Direction: DEEPEN with the two formal seed-0 gates before running the remaining manifest. |
