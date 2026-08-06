# Large-dataset preparation for RECAP

## Storage layout

Raw licensed downloads stay under `dataset_large/raw/`. Converted bundles go
under `dataset/<name>/`, where the existing `utils.Dataset` loader discovers
them automatically.

Each bundle contains `features.npy`, `labels.npy`, `adjacency.npz`, and
`metadata.json`. Features and labels are memory-mappable; adjacency is CSR.
For million-node inputs, 200,000 deterministic evenly spaced rows fit the
robust statistics and PCA basis. All rows are then transformed in 100,000-row
chunks, and exact dataset-wide projected means/variances are accumulated for
the final z-score pass. The cache version is
`robust_sampled_pca_post_zscore_v1`.

## T-Finance and T-Social

Download the two individual files from the BWGNN authors' Drive folder. Do not
rename or reinterpret them before conversion. BWGNN stores them as serialized
DGL graphs.

Use a separate DGL conversion environment:

```bash
python tools/prepare_large_datasets.py \
  --dataset tfinance \
  --source dataset_large/raw/bwgnn/tfinance \
  --output-root dataset

python tools/prepare_large_datasets.py \
  --dataset tsocial \
  --source dataset_large/raw/bwgnn/tsocial \
  --output-root dataset
```

The converter accepts either `feature` or `feat` and either `label` or
`labels`, handles T-Finance's two-column one-hot label, and binarizes anomalies.
It reuses DGL's native CSR arrays and preserves the released topology exactly;
this avoids several edge-sized memory copies on T-Social.

## DGraph-Fin

Download `DGraphFin.zip` only after logging in to the official DGraph website
and accepting its terms. Extract `dgraphfin.npz`, then run:

```bash
python tools/prepare_large_datasets.py \
  --dataset dgraphfin \
  --source dataset_large/raw/dgraphfin/dgraphfin.npz \
  --output-root dataset
```

Official label 1 is mapped to anomaly and label 0 to normal. Labels 2 and 3 are
background nodes: they remain in the graph for message propagation but are
excluded from AUROC/AUPRC through `evaluation_mask.npy`. The converter
preserves the list of all source fields in metadata, while RECAP currently
consumes node features, binary labels, the evaluation mask, and symmetrized
topology.

## Acceptance

```bash
python tools/validate_large_dataset.py dataset/tfinance \
  --expected-nodes 39357 --expected-features 10
python tools/validate_large_dataset.py dataset/tsocial \
  --expected-nodes 5781065 --expected-features 10
python tools/validate_large_dataset.py dataset/dgraphfin \
  --expected-nodes 3700550 --expected-features 17
```

Then smoke-test RECAP loading and propagation:

```bash
python -c "from utils import Dataset; d=Dataset(32,'tfinance'); d.propagated(1,'cpu'); print(d.graph)"
```

## Verified artifacts (2026-07-26)

| Dataset | Raw archive SHA-256 | Raw archive | Current result |
|---|---|---:|---|
| T-Finance | `0a0091978425a0e9f1cd0e001240517ecd786344bd580c2bd00ef32ad1c80475` | 78,068,255 bytes | Conversion, validation, 32-D alignment, loading, and four-hop GPU propagation pass |
| T-Social | `fd504505f6e629551d6e913f923117d32d812043e47a1520268fcefee06ff654` | 744,239,223 bytes | Conversion, validation, 32-D alignment, loading, and four-hop GPU propagation pass |
| DGraph-Fin | `e4db83e319492dd174c0be6896f799e796731d43685e152c678baf4bd2d3bb4a` | 150,476,320 bytes | Conversion and structural validation pass; 32-D alignment and RECAP loading pass |

DGraph-Fin acceptance values:

- 3,700,550 nodes, 17 raw features, 7,994,520 entries after
  symmetrization.
- 15,509 fraud nodes.
- 1,225,601 foreground nodes included in evaluation; labels 2 and 3 remain
  graph context but are masked out of metrics.
- The canonical bundle occupies about 288 MiB before the 32-D aligned cache.

Conversion-only environment:

- Python 3.10.8
- DGL 1.1.3
- PyTorch 2.11.0+cu128
- NumPy 1.26.4
- SciPy 1.11.4

The primary RECAP environment remains Python 3.12.3 and does not depend on
DGL.

BWGNN acceptance values on the cloned 120 GiB RAM / RTX PRO 6000 server:

- T-Finance: 39,357 nodes, 42,445,086 directed CSR entries, 10 raw
  features, and 1,804 anomalies. Its canonical bundle is about 47 MiB.
  Loading took about 8.62 seconds; four-hop propagation took 0.66 seconds
  with 0.98 GiB peak allocated GPU memory.
- T-Social: 5,781,065 nodes, 146,211,016 directed CSR entries, 10 raw
  features, and 174,280 anomalies. Its canonical bundle is about 428 MiB
  before the aligned cache. Loading took about 35.61 seconds; four-hop
  propagation took 2.12 seconds with 8.25 GiB peak allocated GPU memory.
- Both released DGL graphs already store both directions. The adapter
  preserves those records instead of symmetrizing them again.

## Full-graph inference status

The large-graph adapter is complete. T-Finance retains the original exact
blockwise KNN route. T-Social and DGraph-Fin use a dedicated FAISS IVF-PQ route
with exact reranking inside a bounded candidate pool and memory-bounded score
construction. The routing is dataset-specific, so all ordinary datasets retain
the original exact inference path. See `docs/APPROXIMATE_KNN.md` for the locked
algorithm and parameters.

The accepted run scored every node in all three graphs. It establishes
target-side full-graph inference scalability under the recorded hardware, not
million-node training scalability. The unadapted Setting-A checkpoints were
below random-ranking references on all three targets, so computational success
must not be presented as predictive effectiveness. Complete latency, memory,
ANN-fidelity, metrics, and audit results are in
`rebuttal/reports/LARGE_TARGET_INFERENCE_REPORT.md`.
