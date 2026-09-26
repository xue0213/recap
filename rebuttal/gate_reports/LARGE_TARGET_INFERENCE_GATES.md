# Large-Target Inference Acceptance Gates

Date: 2026-07-26

Status: **passed before formal large-target inference**

The gates validate the separate, memory-bounded full-graph inference adapter
without changing the original inference route used by the twelve standard
datasets.

| Gate | Result |
|---|---|
| Locked manifest | 0 training runs; 9 primary target evaluations |
| Canonical data preflight | T-Finance, DGraph-Fin and T-Social shapes, label semantics, feature finiteness and component hashes passed |
| Checkpoint preflight | Setting-A seeds 0/1/2 hashes, embedded sources, epoch and model configuration passed |
| Native vs chunked total score | maximum absolute difference `7.152557e-07` |
| Native vs chunked adhesion term | maximum absolute difference `3.814697e-06` |
| Native vs chunked context term | maximum absolute difference `1.043081e-07` |
| Fixed-seed FAISS determinism | two 20,000-node candidate files were byte-identical (`1db074e0365724542ab911ab594fd7a1c275f3b818d8cc88abdbd2bf819fb25f`) |
| Candidate validity | no invalid IDs, self-neighbors or duplicate neighbors |
| Unit/regression tests | 31 passed, 1 skipped in the formal Python environment |
| T-Finance seed-0 end-to-end smoke | full-node exact and ANN scores froze before labels; independent metrics matched exactly |

The smoke-test exact primary result was AUROC `0.2297559508356054` and AUPRC
`0.02706981839529329`. Its ANN candidate recall@64 over all 39,357 nodes was
`0.8931801363162842`; exact/ANN score Spearman was `0.99970348912562`.
These values were treated strictly as a label-isolation and fidelity gate and
were not used to tune, replace, or invert a formal result.

The first machine-readable preflight attempt stopped before any result existed
because the seed-2 SHA-256 string in the Python declaration was missing two
hexadecimal characters. The locked Markdown protocol already contained the
correct hash. The declaration was corrected, preflight was rerun from scratch,
and the accepted preflight hash is
`c573941ced4b87ddae3eb96b900396574fd5ad57e17d2b722b95f8ab48d638b9`.
