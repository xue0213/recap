# E04: Eight-method 12-dataset OFO baseline reproduction

All values are 12-dataset macro AUROC/AUPRC means over seeds 0, 1, and 2.
Population standard deviations and all per-dataset values are in
`rebuttal/reports/OFO_12_BASELINE_REPORT.md`.

| Regime | Method | AUROC/AUPRC |
|---|---|---:|
| Supervised, stratified 40% test nodes | GCN | 0.81525/0.42827 |
| Supervised, stratified 40% test nodes | GAT | 0.83085/0.43336 |
| Supervised, stratified 40% test nodes | BWGNN | 0.74378/0.29687 |
| Supervised, stratified 40% test nodes | XGBGraph | 0.88840/0.61981 |
| Unsupervised, full graph | DOMINANT | 0.53321/0.10402 |
| Unsupervised, full graph | AnomalyDAE | 0.56238/0.09969 |
| Unsupervised, full graph | CoLA | 0.54363/0.11881 |
| Unsupervised, full graph | ADA-GAD | 0.57575/0.07172 |
| Unsupervised, full graph | RECAP-OFO | 0.71083/0.23606 |

Formal scope: 288 training-and-inference runs and 576 AUROC/AUPRC values.
Independent audit checked the complete method/dataset/seed Cartesian product,
raw-data, score and query-mask hashes, all label-access records, all saved
models and checkpoint reloads, and recomputed every metric. All checks passed.

The supervised and unsupervised rows have different supervision rights and
evaluation populations. The table records both regimes but does not assert a
strictly identical-condition comparison across that boundary.
