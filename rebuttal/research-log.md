# RECAP Phase 1 Research Log

Append-only decision and experiment record.

| # | Date | Type | Summary |
|---|------|------|---------|
| 1 | 2026-07-26 | bootstrap | Reviewed the manuscript, full experiment protocol, server environment, all 12 Phase 1 datasets, and current code. Identified a paper-implementation drift in the dirty ANN worktree and decided to execute Phase 1 from a clean worktree based on commit `c94c4d7`. |
| 2 | 2026-07-26 | protocol | Locked OFO to 11 datasets by excluding Questions; OFA remains unchanged. Locked 42 training runs, 87 final evaluations, 87 stability comparisons, seeds 0/1/2, paper hyperparameters, strict label isolation, exact KNN, and population standard deviation. |
| 3 | 2026-07-26 | infrastructure | Created a 20-minute heartbeat to audit progress, failures, label isolation, result completeness, and outer-loop reflection checkpoints. |
