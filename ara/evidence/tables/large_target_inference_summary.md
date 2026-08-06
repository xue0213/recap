# Large-Target Inference Evidence Summary

Scope: target-side full-graph inference only; zero large-target training runs.
Values are mean ± population standard deviation over the three accepted
RECAP-OFA Setting-A checkpoints.

| Target | Full nodes | Primary KNN | AUROC | AUPRC | Shared setup (s) | Warm inference (s) | Peak GPU allocated/reserved (GiB) | ANN recall@64 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| T-Finance | 39,357 | exact | 0.25566 ± 0.02002 | 0.02801 ± 0.00077 | 9.71 | 0.16 ± 0.12 | 0.98 / 1.25 | 0.8934 (all nodes) |
| DGraph-Fin | 3,700,550 | FAISS-IVFPQ | 0.36747 ± 0.00651 | 0.00904 ± 0.00009 | 503.73 | 6.61 ± 0.11 | 48.10 / 65.75 | 0.5735 (512 queries) |
| T-Social | 5,781,065 | FAISS-IVFPQ | 0.43785 ± 0.00515 | 0.02595 ± 0.00012 | 841.69 | 10.51 ± 0.09 | 75.13 / 80.66 | 0.4338 (512 queries) |

All nine full-node score vectors passed independent hash, finiteness,
label-order, and metric recomputation audits. Accounted formal setup and run
phases total 1,420.29 seconds.

The evidence supports computational target-side inference scalability on the
recorded hardware. It does not test million-node training scalability. All
three AUROC means are below 0.5 and all three AUPRC means are below anomaly
prevalence, so predictive effectiveness of the unadapted checkpoints is not
supported.
