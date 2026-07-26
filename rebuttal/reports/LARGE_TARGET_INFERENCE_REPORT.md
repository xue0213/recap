# RECAP Large-Target Full-Graph Inference Results

Scope: **target-side inference scalability only**. These experiments reuse the accepted RECAP-OFA Setting-A checkpoints and perform no training or target-label tuning.

## Primary results

| Target | Full nodes | Eval nodes | Anomaly rate (%) | KNN | AUROC (%) | AUPRC (%) |
|---|---:|---:|---:|---|---:|---:|
| T-Finance | 39,357 | 39,357 | 4.5837 | exact | 25.5656 ± 2.0025 | 2.8012 ± 0.0766 |
| DGraph-Fin | 3,700,550 | 1,225,601 | 1.2654 | faiss_ivfpq | 36.7466 ± 0.6507 | 0.9044 ± 0.0089 |
| T-Social | 5,781,065 | 5,781,065 | 3.0147 | faiss_ivfpq | 43.7851 ± 0.5152 | 2.5952 ± 0.0116 |

Mean ± population standard deviation over the three immutable Setting-A checkpoints. Every target receives a score for every node; DGraph-Fin metrics use only its frozen 0/1 evaluation mask.
AUROC 50% and the listed anomaly rate for AUPRC are the random-ranking references; computational completion is not treated as evidence of predictive effectiveness.

## Scalability

| Target | Shared cold setup (s) | Warm checkpoint inference (s) | Estimated cold latency (s) | Nodes/s | Adjacency nnz/s | Peak GPU alloc. (GiB) | Peak GPU reserved (GiB) | Peak RSS (GiB) | KNN cache (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T-Finance | 9.71 | 0.16 ± 0.12 | 9.87 | 390,288 | 4,372,650 | 0.98 | 1.25 | 3.18 | 0.02 |
| DGraph-Fin | 503.73 | 6.61 ± 0.11 | 510.33 | 560,202 | 15,871 | 48.10 | 65.75 | 4.97 | 0.88 |
| T-Social | 841.69 | 10.51 ± 0.09 | 852.21 | 549,882 | 173,711 | 75.13 | 80.66 | 9.26 | 1.38 |

Shared setup includes canonical loading/normalization, four-hop propagation, initial residual construction, KNN construction and the pre-registered ANN-fidelity query. Warm checkpoint inference includes checkpoint load, model residual forward, primary score construction and score serialization; it excludes metric calculation. Peak memory is the maximum observed in setup or any checkpoint run.
All three shared setups plus all nine checkpoint runs account for 1420.29 seconds (23.67 minutes), excluding preflight, gates, orchestration and independent auditing.

## Approximation fidelity

| Target | Fidelity population | IVF lists / probes | Retrieved | Top-64 recall |
|---|---|---:|---:|---:|
| T-Finance | all nodes | 614 / 16 | 256 | 0.8934 |
| DGraph-Fin | 512 fixed label-blind queries | 4096 / 16 | 256 | 0.5735 |
| T-Social | 512 fixed label-blind queries | 4096 / 16 | 256 | 0.4338 |

For T-Finance, the auxiliary ANN route has mean exact/ANN score Spearman 0.999608, mean absolute AUROC difference 0.000251, and mean absolute AUPRC difference 0.000009. Its primary results remain exact KNN.

The million-node primary routes use the locked 4,096 IVF lists. T-Finance's descriptive auxiliary ANN index uses 614 lists (the adapter's predeclared 64 training vectors/list safety cap on a 39,357-node graph); this does not affect its exact primary result.

## Per-checkpoint records

| Target | Seed | AUROC | AUPRC | Primary inference (s) | Peak GPU alloc. (GiB) | Peak RSS (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| T-Finance | 0 | 0.229756 | 0.027070 | 0.33 | 0.70 | 1.82 |
| T-Finance | 1 | 0.258687 | 0.028022 | 0.08 | 0.70 | 1.82 |
| T-Finance | 2 | 0.278524 | 0.028945 | 0.07 | 0.70 | 1.82 |
| DGraph-Fin | 0 | 0.371793 | 0.009116 | 6.76 | 48.10 | 2.86 |
| DGraph-Fin | 1 | 0.372335 | 0.009096 | 6.54 | 48.10 | 2.88 |
| DGraph-Fin | 2 | 0.358268 | 0.008919 | 6.52 | 48.10 | 2.89 |
| T-Social | 0 | 0.430682 | 0.025872 | 10.60 | 75.13 | 3.42 |
| T-Social | 1 | 0.440310 | 0.025869 | 10.55 | 75.13 | 3.46 |
| T-Social | 2 | 0.442562 | 0.026117 | 10.39 | 75.13 | 3.46 |

## Acceptance and interpretation

- 9/9 pre-registered checkpoint-target cells completed with a finite full-node score vector.
- Every checkpoint, dataset component, candidate cache, score vector and query mask matched its recorded SHA-256.
- Every label audit froze all declared score routes before the single label-unlock event.
- Independent AUROC/AUPRC recomputation matched every stored metric within 1e-12.
- This establishes computational applicability of target-side full-graph inference under the recorded hardware and adapter. It does not establish million-node training scalability, and predictive quality must be read from the primary table rather than inferred from successful completion.
- All three target means are below the random-ranking references (AUROC 50%; AUPRC equal to anomaly prevalence). The experiment therefore does **not** support predictive effectiveness of the unadapted Setting-A checkpoints on these targets.

Preflight file SHA-256: `f8a98c31350ecbf821a9c54101562b8189c6841f22d8d38a3550d02e27553fea`  
Independent audit hash: `a060c0f654da78973a8e2e477070a436b0f741393dee28b1a3058b95e9c9dac6`
