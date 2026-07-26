# RECAP Rebuttal Experiment Protocol

## 0. Purpose and Phase Scope

This document specifies the experiments and reporting format for the RECAP rebuttal.
It covers two evaluation paradigms:

1. **One-for-One (OFO):** train one RECAP model independently on each target graph.
2. **One-for-All (OFA):** train one shared RECAP model on source graphs and directly
   infer on unseen target graphs.

### Phase 1 scope

Phase 1 runs **RECAP only** and fills only the RECAP rows/cells in the tables below.

- Run `RECAP-OFO` for the target-specific one-for-one evaluation.
- Run `RECAP` for the three one-for-all source-target settings.
- Record the final community outputs required for the three-seed community
  stability audit described in Section 6.
- Record training and inference wall-clock times using the timing protocol in
  Section 6.
- Do **not** run GCN, GAT, BWGNN, XGBGraph, DOMINANT, AnomalyDAE, CoLA,
  ADA-GAD, ARC, UNPrompt, AnomalyGFM, or IA-GGAD in Phase 1.
- Leave all baseline cells marked `—` for Phase 2.

### Phase 1 expected training count

- RECAP-OFO: 8 target graphs × 3 seeds = **24 training runs**.
- RECAP-OFA Setting B: 1 split × 3 seeds = **3 training runs**.
- RECAP-OFA Setting C: 1 split × 3 seeds = **3 training runs**.
- RECAP-OFA Setting A:
  - reuse the existing three-seed raw results only if their configuration exactly
    matches this protocol; otherwise rerun 3 training runs.

Therefore, Phase 1 requires **30 new runs** if Setting A is safely reusable, or
**33 runs** if Setting A must be rerun.

---

## 1. Datasets and Domains

| Domain | Datasets |
|---|---|
| Citation | PubMed, Cora, CiteSeer, ACM |
| Social | Flickr, BlogCatalog, Facebook, Weibo, Reddit |
| E-commerce/Review | YelpChi, Amazon |
| Q&A/Interaction | Questions |

Use the exact graph data, anomaly labels, feature preparation, and adjacency
construction used by the submitted paper.

Labels must never enter RECAP training, hyperparameter selection, early stopping,
community construction, prototype estimation, or inference. Labels are used only
after scoring to compute AUROC and AUPRC.

---

## 2. Common RECAP Configuration

All RECAP experiments must use the submitted-paper configuration unless an
experiment below explicitly changes the source/target graphs.

| Item | Value |
|---|---:|
| Aligned feature dimension | 32 |
| Optimizer | Adam |
| Training epochs | 100 |
| Learning rate | \(5\times10^{-5}\) |
| Weight decay | \(5\times10^{-5}\) |
| Propagation hops \(L\) | 4 |
| Community number \(C\) | 36 |
| KNN size \(k\) | 64 |
| Residual-similarity temperature \(\tau_s\) | 0.3 |
| Assignment temperature \(\tau_c\) | 0.3 |
| Prototype score scale \(\tau_e\) | 1.0 / released implementation default |
| Assignment regularization weight \(\lambda_H\) | 0.1 |
| Community-usage weight \(\lambda_{\mathrm{usage}}\) | 0.1 |
| Context score coefficient \(\beta\) | 0.02 |
| Training seeds | 0, 1, 2 |

### Configuration warning

The current local `params/recap.json` may not match the submitted-paper
configuration (for example, the local file may contain different values for
`num_hops`, `num_clusters`, `knn_k`, or `beta`). **Do not run the final
experiments by blindly loading that file.**

Before launching final runs:

1. Resolve the runtime configuration printed by the experiment.
2. Confirm that it matches the table above and the submitted manuscript.
3. Save a copy of the resolved configuration with every result directory.
4. If a manuscript parameter is absent from this table, preserve the exact
   released implementation/default used to obtain the submitted results.
5. Do not tune any parameter using target labels.

### Common optimization rules

- Use fixed 100-epoch training and the final checkpoint.
- Do not select epochs using target AUROC or AUPRC.
- Do not perform target-label-based early stopping.
- For OFA, train one shared model across all source graphs.
- Preserve the released implementation's graph-normalized loss and source-graph
  sampling procedure.
- All three OFA settings contain four source graphs, so they must use the same
  epoch and optimizer-step schedule.
- Use the same preprocessing code and cache version for every setting.
- Use the same device type and numerical precision where possible.

### Random seeds

The seeds `0, 1, 2` are **training seeds**, controlling parameter initialization,
dropout, data/graph order, and optimization. For a fixed trained checkpoint,
standard RECAP inference is deterministic and should be run once per target graph.

Do not resample target datasets for different training seeds.

---

## 3. Experiment I: One-for-One RECAP

### 3.1 Goal

Evaluate RECAP in the same target-specific paradigm as conventional supervised
and unsupervised one-for-one graph anomaly detectors.

The Phase 1 method name is:

> **RECAP-OFO**

### 3.2 Target datasets

Run independently on:

1. Cora
2. CiteSeer
3. ACM
4. BlogCatalog
5. Facebook
6. Weibo
7. Reddit
8. Amazon

### 3.3 RECAP-OFO protocol

For each target dataset \(D\) and each training seed \(r\in\{0,1,2\}\):

1. Initialize a fresh RECAP model from scratch.
2. Use only the attributes and graph structure of \(D\).
3. Do not load a source-pretrained RECAP checkpoint.
4. Do not use any other graph for training.
5. Train RECAP's unsupervised residual-community objective on \(D\) for 100 epochs.
6. Keep anomaly labels inaccessible to the training loop.
7. After training, compute target-graph community assignments, prototypes,
   prototype-adhesion scores, context scores, and final anomaly scores on \(D\).
8. Compute AUROC and AUPRC using labels only after anomaly scores are finalized.
9. Save the seed-level metrics and resolved configuration.

There is no node-label train/test split for RECAP-OFO because its training is
unsupervised and transductive. The full graph structure and attributes may be
used, while anomaly labels remain evaluation-only.

### 3.4 Independence requirements

- Each dataset-seed pair must start from a fresh model.
- Do not reuse a model trained on another target graph.
- Do not reuse optimizer state across datasets.
- Do not choose dataset-specific hyperparameters.
- Do not modify the score coefficient based on target performance.

---

## 4. Experiment II: One-for-All RECAP

### 4.1 Common OFA protocol

For each OFA setting and each training seed:

1. Initialize one RECAP model.
2. Train it jointly on the four specified source graphs without source labels.
3. Freeze all learnable parameters after source training.
4. Apply the same frozen checkpoint to every target graph in that setting.
5. Do not fine-tune, optimize, or early-stop on target graphs.
6. Do not use target prompts or normal/anomalous context samples.
7. Target-side feature alignment, graph propagation, soft assignments, and
   closed-form community prototype estimation follow standard RECAP inference.
8. Save per-target AUROC/AUPRC and training/inference metadata.

### 4.2 Setting A: Multi-domain source transfer

This is the submitted-paper source-target split.

#### Source graphs

- PubMed
- Flickr
- Questions
- YelpChi

#### Target graphs

- Cora
- CiteSeer
- ACM
- BlogCatalog
- Facebook
- Weibo
- Reddit
- Amazon

#### Scale

- 4 source graphs
- 4 source domains
- 8 target graphs

Existing Setting A results may be reused only when raw seed-level results,
resolved hyperparameters, and preprocessing versions are available and match
this protocol exactly.

### 4.3 Setting B: Leave-Social-Domain-Out transfer

The Social domain is completely absent from source training.

#### Source graphs

- PubMed
- Cora
- Questions
- YelpChi

#### Target graphs

- Flickr
- BlogCatalog
- Facebook
- Weibo
- Reddit

#### Scale

- 4 source graphs
- 3 source domains
- 5 target graphs
- 0 Social-domain source graphs

Relative to Setting A, Flickr is replaced by Cora while the number of source
graphs remains fixed.

### 4.4 Setting C: Citation-only source transfer

All source graphs come from the Citation domain.

#### Source graphs

- PubMed
- Cora
- CiteSeer
- ACM

#### Target sampling rule

The target set was fixed before model training using a stratified random
selection:

- Split seed: `2026`
- Sort candidate dataset names alphabetically before sampling.
- Sample 3 of the 5 Social datasets.
- Sample 1 of the 2 E-commerce datasets.
- Include Questions as the sole Q&A dataset.

The resulting fixed target set is:

- BlogCatalog
- Flickr
- Reddit
- Amazon
- Questions

Do not resample this set.

#### Scale

- 4 source graphs
- 1 source domain
- 5 target graphs from 3 unseen domains

---

## 5. Metrics and Aggregation

### 5.1 Per-dataset metrics

For every dataset and seed, record:

- AUROC
- AUPRC
- training time
- inference time
- resolved configuration
- checkpoint/result path

Report metrics as percentages:

\[
M_{\%}=100M.
\]

Each displayed per-dataset value is:

\[
\operatorname{mean}_{r\in\{0,1,2\}} M_r
\!/\!-
\operatorname{std}_{r\in\{0,1,2\}} M_r.
\]

Use population or sample standard deviation consistently with the submitted
paper. Record the exact choice in the result metadata.

### 5.2 Dataset-macro result

For each seed, first average over the target datasets:

\[
M^{(r)}_{\mathrm{dataset\mbox{-}macro}}
=
\frac{1}{|\mathcal T|}
\sum_{D\in\mathcal T} M_D^{(r)}.
\]

Then report the mean and standard deviation of the three seed-level macro
values. Do not average already-rounded table cells.

### 5.3 Setting B Social Macro

For each seed:

\[
M^{(r)}_{\mathrm{social}}
=
\frac{
M_{\mathrm{Flickr}}^{(r)}
+M_{\mathrm{BlogCatalog}}^{(r)}
+M_{\mathrm{Facebook}}^{(r)}
+M_{\mathrm{Weibo}}^{(r)}
+M_{\mathrm{Reddit}}^{(r)}
}{5}.
\]

Report mean ± standard deviation across seeds.

### 5.4 Setting C Domain Macro

For each seed, calculate:

\[
M^{(r)}_{\mathrm{social}}
=
\frac{
M_{\mathrm{BlogCatalog}}^{(r)}
+M_{\mathrm{Flickr}}^{(r)}
+M_{\mathrm{Reddit}}^{(r)}
}{3},
\]

\[
M^{(r)}_{\mathrm{domain\mbox{-}macro}}
=
\frac{
M^{(r)}_{\mathrm{social}}
+M_{\mathrm{Amazon}}^{(r)}
+M_{\mathrm{Questions}}^{(r)}
}{3}.
\]

Compute this separately for AUROC and AUPRC, then report mean ± standard
deviation over seeds.

### 5.5 Cross-setting interpretation

Settings A, B, and C contain different target sets. Their macro values should
not be averaged into a single overall score or interpreted as a strict
single-variable causal curve. They are complementary transfer protocols
demonstrating robustness to different source-domain coverage and source-target
partitions.

---

## 6. Community Stability and Runtime Recording

These records must be collected during the Phase 1 runs. The community
stability audit uses the existing training seeds `0, 1, 2` and requires **no
additional training runs**.

### 6.1 Community stability scope

Collect the required community outputs for:

- every RECAP-OFO target graph in Section 3;
- every RECAP-OFA target graph in Settings A, B, and C;
- all three training seeds.

Community IDs are arbitrary and can be permuted between runs. Do not measure
stability by directly checking whether raw community IDs are equal.

### 6.2 Final community data to save

After final-epoch inference, save the following for each
method/setting/dataset/seed:

1. Soft assignment matrix
   \[
   H\in\mathbb R^{n\times C}.
   \]
2. Hard assignment vector
   \[
   c_i=\arg\max_c H_{ic}.
   \]
3. Community usage vector
   \[
   \pi_c=\frac{1}{n}\sum_iH_{ic}.
   \]
4. Per-node assignment entropy.
5. Per-node raw and standardized prototype-adhesion scores.
6. Per-node raw and standardized context scores.
7. Per-node final anomaly scores.
8. Deterministic node indices in the exact order used by all saved arrays.
9. Effective community number
   \[
   C_{\mathrm{eff}}
   =
   \exp\left(-\sum_c\pi_c\log(\pi_c+\epsilon)\right).
   \]

Save floating-point arrays in compressed `.npz` or equivalent lossless format.
`float32` is preferred; `float16` may be used for `H` only when storage becomes
material, but the choice must be recorded. Hard assignments and node indices
must be saved exactly.

Recommended directory structure:

```text
community_stability/
  one_for_one/
    Cora/
      seed_0.npz
      seed_1.npz
      seed_2.npz
    ...
  one_for_all/
    setting_A/
      Cora/
        seed_0.npz
        seed_1.npz
        seed_2.npz
      ...
    setting_B/
    setting_C/
```

### 6.3 Training-process diagnostics

At epochs `1, 10, 25, 50, 75, 100`, append the following diagnostics to a CSV
file without using labels:

- epoch;
- seed;
- paradigm and setting;
- source/target dataset name as applicable;
- total training loss;
- \(L_{\mathrm{con}}\);
- assignment regularization loss;
- community-usage regularization loss;
- average node-assignment entropy;
- community-usage entropy;
- \(C_{\mathrm{eff}}\);
- minimum, maximum, and standard deviation of community usage;
- residual embedding mean standard deviation;
- number of communities whose usage exceeds the implementation's active-usage
  threshold.

For OFA training, record these diagnostics per source graph when available and
also record the source-graph macro average. For OFO, record them for the single
training graph.

These diagnostics are observational only. They must not be used to select an
epoch, tune a hyperparameter, or stop training.

### 6.4 Stability metrics to compute

For each fixed setting and dataset, compare the three seed pairs:

```text
(0, 1), (0, 2), (1, 2)
```

Compute:

1. **NMI** between hard assignments (higher is better).
2. **ARI** between hard assignments (higher is better).
3. **Soft co-assignment similarity** (higher is better):
   \[
   \rho_{ab}
   =
   \frac{\|H^{(a)\top}H^{(b)}\|_F^2}
   {\|H^{(a)\top}H^{(a)}\|_F
    \|H^{(b)\top}H^{(b)}\|_F}.
   \]
   This is permutation-invariant and avoids constructing an \(n\times n\)
   co-assignment matrix.
4. **Final-score Spearman correlation** between node anomaly-score vectors
   (higher is better).
5. Mean and standard deviation of \(C_{\mathrm{eff}}\) across the three seeds.

NMI, ARI, soft co-assignment similarity, and score Spearman are summarized as
the mean and standard deviation across the three seed pairs. Save the
pair-level values, not only their averages.

No anomaly labels are required for these stability metrics.

### 6.5 Runtime definitions

Record wall-clock time in seconds using a monotonic high-resolution timer such
as `time.perf_counter()`.

For CUDA execution, call `torch.cuda.synchronize()` immediately before starting
and immediately after stopping every timed region. Without synchronization,
reported GPU times are invalid.

Record these timing components separately:

1. **Data preparation time**
   - dataset loading;
   - feature alignment;
   - graph normalization;
   - multi-hop propagation;
   - preprocessing/cache construction.
2. **Training time**
   - starts immediately before the first training forward/optimizer step;
   - ends immediately after the final optimizer step;
   - includes loss computation, backward passes, optimizer updates, and
     training-time community/KNN construction;
   - excludes data preparation and result serialization.
3. **Inference time**
   - measured separately for each target graph;
   - starts immediately before the frozen-model target forward pass;
   - includes target residual encoding, KNN/community graph construction,
     soft assignment, prototype estimation, context scoring, and final score
     construction;
   - ends when the complete node anomaly-score vector is available;
   - excludes AUROC/AUPRC computation and file serialization.

For RECAP-OFO, record one training time and one inference time for every
dataset/seed run.

For RECAP-OFA, record one shared-model training time for every setting/seed and
one inference time for every target dataset/seed.

### 6.6 Runtime controls and metadata

- Use the same hardware, device type, precision, and software environment for
  all Phase 1 runs.
- Record CPU model, GPU model, GPU memory, PyTorch version, CUDA version, and
  device identifier.
- Record whether a KNN/preprocessing cache was cold, newly created, or reused.
- Do not compare a cold-cache run against a warm-cache run without marking the
  difference.
- Prefer a unique cache namespace per setting/seed for final timing runs, or
  explicitly report cache state for every record.
- Do not include failed/retried partial runs in the timing summary.
- The three training seeds provide three runtime observations. Do not add
  repeated inference solely for timing in the minimum-cost Phase 1 protocol.
- Optionally record peak allocated GPU memory, but this is supplementary and
  does not replace training/inference time.

---

## 7. Required Raw Result Format

Save one record per method/setting/dataset/seed. The preferred CSV schema is:

```text
method,paradigm,setting,seed,source_graphs,target_graph,dataset_domain,auroc,auprc,data_prepare_seconds,train_seconds,inference_seconds,cache_state,peak_gpu_memory_mb,config_path,checkpoint_path,community_output_path
```

Examples of `method`, `paradigm`, and `setting`:

```text
RECAP-OFO,one-for-one,OFO,0,... 
RECAP,one-for-all,A,0,...
RECAP,one-for-all,B,0,...
RECAP,one-for-all,C,0,...
```

Also save:

- a JSON copy of all seed-level records;
- a resolved configuration file;
- stdout/stderr logs;
- the training-process diagnostics CSV from Section 6.3;
- the final community arrays from Section 6.2;
- pair-level and summarized stability metrics from Section 6.4;
- runtime hardware/software metadata from Section 6.6;
- a short failure log for any unsuccessful run.

Do not fill a table from console output alone. Tables must be generated from
saved raw seed-level results.

---

## 8. Phase 1 Execution Checklist

### 8.1 Pre-run validation

- [ ] Confirm all 12 datasets load correctly.
- [ ] Confirm anomaly labels are not passed to the RECAP loss.
- [ ] Confirm feature-alignment cache/version consistency.
- [ ] Confirm the resolved configuration matches Section 2.
- [ ] Confirm seeds are exactly `0, 1, 2`.
- [ ] Confirm Setting C target graphs are fixed and not resampled.
- [ ] Confirm target metrics are not used for early stopping.
- [ ] Confirm the timer uses `time.perf_counter()` or an equivalent monotonic
      wall-clock timer.
- [ ] Confirm CUDA synchronization is applied around timed regions.
- [ ] Confirm cache state is recorded for every timed run.
- [ ] Confirm final community outputs contain node indices and use identical
      node ordering across seeds.

### 8.2 RECAP-OFO

- [ ] Cora × 3 seeds
- [ ] CiteSeer × 3 seeds
- [ ] ACM × 3 seeds
- [ ] BlogCatalog × 3 seeds
- [ ] Facebook × 3 seeds
- [ ] Weibo × 3 seeds
- [ ] Reddit × 3 seeds
- [ ] Amazon × 3 seeds

### 8.3 RECAP-OFA

- [ ] Validate or rerun Setting A × 3 seeds
- [ ] Run Setting B × 3 seeds
- [ ] Run Setting C × 3 seeds

### 8.4 Community-stability records

- [ ] Final \(H\), hard assignments, usage, entropy, score components, and
      node indices exist for every required dataset/seed.
- [ ] Epoch diagnostics exist for epochs 1, 10, 25, 50, 75, and 100.
- [ ] NMI, ARI, soft co-assignment, and score Spearman exist for all three seed
      pairs.
- [ ] \(C_{\mathrm{eff}}\) is recorded for every seed.
- [ ] No raw community-ID equality was used as a stability metric.

### 8.5 Runtime records

- [ ] Data preparation, training, and inference times are stored separately.
- [ ] OFA inference time is stored per target graph.
- [ ] Hardware/software and cache metadata are saved.
- [ ] Failed or partial runs are excluded from runtime summaries.
- [ ] Runtime table values come from the same three successful seeds as the
      performance results.

### 8.6 Post-run validation

- [ ] Exactly three successful seed records exist for every required result.
- [ ] Every metric is within \([0,1]\) before percentage conversion.
- [ ] No target-label-based model selection occurred.
- [ ] Macro metrics were computed from unrounded seed-level results.
- [ ] AUROC and AUPRC tables were generated from the same raw records.
- [ ] RECAP-only Phase 1 tables contain no fabricated baseline values.

---

## 9. Result Tables to Complete

### Table 1. One-for-One AUROC (%)

All methods are trained independently on each target graph. In Phase 1, fill
only the `RECAP-OFO` row.

| Method | Target labels | Cora | CiteSeer | ACM | BlogCatalog | Facebook | Weibo | Reddit | Amazon | Avg. |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Supervised One-for-One** ||||||||||||
| GCN | ✓ | — | — | — | — | — | — | — | — | — |
| GAT | ✓ | — | — | — | — | — | — | — | — | — |
| BWGNN | ✓ | — | — | — | — | — | — | — | — | — |
| XGBGraph | ✓ | — | — | — | — | — | — | — | — | — |
| **Unsupervised One-for-One** ||||||||||||
| DOMINANT | ✗ | — | — | — | — | — | — | — | — | — |
| AnomalyDAE | ✗ | — | — | — | — | — | — | — | — | — |
| CoLA | ✗ | — | — | — | — | — | — | — | — | — |
| ADA-GAD | ✗ | — | — | — | — | — | — | — | — | — |
| **RECAP-OFO** | ✗ | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD |

### Table 2. One-for-One AUPRC (%)

| Method | Target labels | Cora | CiteSeer | ACM | BlogCatalog | Facebook | Weibo | Reddit | Amazon | Avg. |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Supervised One-for-One** ||||||||||||
| GCN | ✓ | — | — | — | — | — | — | — | — | — |
| GAT | ✓ | — | — | — | — | — | — | — | — | — |
| BWGNN | ✓ | — | — | — | — | — | — | — | — | — |
| XGBGraph | ✓ | — | — | — | — | — | — | — | — | — |
| **Unsupervised One-for-One** ||||||||||||
| DOMINANT | ✗ | — | — | — | — | — | — | — | — | — |
| AnomalyDAE | ✗ | — | — | — | — | — | — | — | — | — |
| CoLA | ✗ | — | — | — | — | — | — | — | — | — |
| ADA-GAD | ✗ | — | — | — | — | — | — | — | — | — |
| **RECAP-OFO** | ✗ | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD |

### Table 3. One-for-All Setting A AUROC (%)

`SL`: source labels. `TC`: target context/prompts. `TT`: target tuning.
In Phase 1, fill only the `RECAP` row.

| Method | SL | TC | TT | Cora | CiteSeer | ACM | BlogCatalog | Facebook | Weibo | Reddit | Amazon | Avg. |
|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Source-supervised Zero-shot OFA** |||||||||||||
| AnomalyGFM-ZS | ✓ | ✗ | ✗ | — | — | — | — | — | — | — | — | — |
| IA-GGAD | ✓ | ✗ | ✗ | — | — | — | — | — | — | — | — | — |
| **Context/Prompt-assisted OFA** |||||||||||||
| ARC | ✓ | ✓ | ✗ | — | — | — | — | — | — | — | — | — |
| UNPrompt | ✓ | ✓ | ✗ | — | — | — | — | — | — | — | — | — |
| **Label-free Zero-shot OFA** |||||||||||||
| **RECAP** | ✗ | ✗ | ✗ | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD |

### Table 4. One-for-All Setting A AUPRC (%)

| Method | SL | TC | TT | Cora | CiteSeer | ACM | BlogCatalog | Facebook | Weibo | Reddit | Amazon | Avg. |
|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Source-supervised Zero-shot OFA** |||||||||||||
| AnomalyGFM-ZS | ✓ | ✗ | ✗ | — | — | — | — | — | — | — | — | — |
| IA-GGAD | ✓ | ✗ | ✗ | — | — | — | — | — | — | — | — | — |
| **Context/Prompt-assisted OFA** |||||||||||||
| ARC | ✓ | ✓ | ✗ | — | — | — | — | — | — | — | — | — |
| UNPrompt | ✓ | ✓ | ✗ | — | — | — | — | — | — | — | — | — |
| **Label-free Zero-shot OFA** |||||||||||||
| **RECAP** | ✗ | ✗ | ✗ | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD |

### Table 5. Source-Target Robustness Summary

Each cell is `AUROC mean±std / AUPRC mean±std`. Settings contain different target
sets; do not compute an overall average across A/B/C.

| Method | SL | TC | Setting A Dataset-Macro | Setting B Social-Macro | Setting C Dataset-Macro | Setting C Domain-Macro |
|---|:---:|:---:|---:|---:|---:|---:|
| ARC | ✓ | ✓ | — | — | — | — |
| IA-GGAD | ✓ | ✗ | — | — | — | — |
| **RECAP** | ✗ | ✗ | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD |

### Table 6. Setting B: Leave-Social-Domain-Out

Each cell is `AUROC mean±std / AUPRC mean±std`.

| Method | Flickr | BlogCatalog | Facebook | Weibo | Reddit | Social Macro |
|---|---:|---:|---:|---:|---:|---:|
| ARC | — | — | — | — | — | — |
| IA-GGAD | — | — | — | — | — | — |
| **RECAP** | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD |

### Table 7. Setting C: Citation-only Source Transfer

Each cell is `AUROC mean±std / AUPRC mean±std`.

| Method | BlogCatalog | Flickr | Reddit | Amazon | Questions | Dataset-Macro | Domain-Macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| ARC | — | — | — | — | — | — | — |
| IA-GGAD | — | — | — | — | — | — | — |
| **RECAP** | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD |

### Table 8. Three-Seed Community Stability

NMI, ARI, soft co-assignment, and score Spearman are mean ± standard deviation
over seed pairs `(0,1)`, `(0,2)`, and `(1,2)`. \(C_{\mathrm{eff}}\) is mean ±
standard deviation over seeds.

| Paradigm/Setting | Evaluation scope | NMI ↑ | ARI ↑ | Soft Co-assignment ↑ | Score Spearman ↑ | \(C_{\mathrm{eff}}\) |
|---|---|---:|---:|---:|---:|---:|
| RECAP-OFO | Macro over 8 target graphs | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD |
| RECAP-OFA A | Macro over 8 target graphs | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD |
| RECAP-OFA B | Macro over 5 Social graphs | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD |
| RECAP-OFA C | Macro over 5 selected graphs | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD | RECAP-TBD |

The per-dataset stability table must also be saved as CSV even if only the
macro summary appears in the paper.

### Table 9. RECAP Training and Inference Time

Times are wall-clock seconds, mean ± standard deviation over the three
successful training seeds. OFA training time is for one shared model; OFA
inference time is averaged over target graphs only after first computing the
per-seed target-graph mean.

| Paradigm/Setting | # Source/Training Graphs | # Target Graphs | Data Preparation (s) | Training (s) | Inference per Target Graph (s) |
|---|---:|---:|---:|---:|---:|
| RECAP-OFO | 1 per model | 8 independently trained graphs | RECAP-TBD | RECAP-TBD | RECAP-TBD |
| RECAP-OFA A | 4 | 8 | RECAP-TBD | RECAP-TBD | RECAP-TBD |
| RECAP-OFA B | 4 | 5 | RECAP-TBD | RECAP-TBD | RECAP-TBD |
| RECAP-OFA C | 4 | 5 | RECAP-TBD | RECAP-TBD | RECAP-TBD |

For RECAP-OFO, the summary is a macro over the eight dataset-specific
train/inference runs. Save per-dataset timing results separately.

---

## 10. Phase 1 Completion Criteria

Phase 1 is complete only when:

1. All required RECAP-OFO and RECAP-OFA runs have three successful training
   seeds.
2. Raw seed-level records and resolved configurations are saved.
3. Every `RECAP-TBD` cell in Tables 1–9 is replaced by a value computed from raw
   results.
4. Baseline cells remain `—`.
5. No result is copied from an incompatible configuration or a one-seed smoke
   test.
6. Community stability metrics are computed from the three fixed seeds without
   additional training or target labels.
7. Training and inference timings follow the synchronized wall-clock protocol
   and include cache/hardware metadata.
8. A short execution report lists:
   - completed runs;
   - reused Setting A runs and their provenance;
   - failed/retried runs;
   - exact aggregation script or command;
   - community-stability output paths;
   - timing methodology and cache state;
   - final result file paths.
