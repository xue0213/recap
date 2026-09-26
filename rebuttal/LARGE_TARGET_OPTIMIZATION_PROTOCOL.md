# RECAP Large-Target Predictive Optimization Protocol

Status: **locked before the new checkpoint scan and target-adaptation runs**

Date: 2026-07-27

This addendum seeks the strongest defensible RECAP results on T-Finance,
DGraph-Fin, and T-Social without overwriting the completed large-target
scalability experiment. The original nine Setting-A results and their
artifacts remain immutable.

## 1. Questions

1. Is the weak large-target result mainly caused by the chosen Setting-A
   source mixture?
2. Can a label-free, target-adapted RECAP model improve predictive quality
   while preserving the full-graph inference path?
3. For T-Finance and T-Social, which share the released 10-dimensional
   feature schema, does a shared source-fitted feature coordinate system
   improve cross-graph transfer?

## 2. Evidence strata

### Confirmatory / deployable

- Target labels and evaluation masks are inaccessible during preprocessing,
  training, checkpoint selection, scoring, and score-route selection.
- A complete score vector and SHA-256 are frozen before the single label
  unlock.
- Score direction is always the paper direction; no post-hoc inversion is
  allowed.
- The paper score `z(adhesion) + 0.02 z(context)` is primary.
- `z(adhesion)` is a predeclared efficiency ablation and is reported
  separately, never silently substituted for full RECAP.
- Three independent seeds are required for a final selected configuration.

### Exploratory / oracle

- All 45 accepted existing checkpoints (OFA A/B/C and 12 OFO source models,
  three seeds each) are scored before any target label is loaded.
- Metrics are computed only after every candidate score route is frozen.
- Per-target selection by observed AUROC is an oracle source-sensitivity
  upper bound. It is not a deployable zero-shot selection result.
- Component and hyperparameter diagnostics may motivate a later
  pre-registered run, but their own best values remain exploratory.

## 3. Experiment order

1. Reuse the immutable propagated-feature and exact/ANN candidate definitions
   from the completed scalability experiment.
2. Run the 45-checkpoint source-sensitivity matrix on all three targets.
   Before labels are unlocked, also freeze three seed-aligned ensembles:
   - `AllSources`: mean of all 15 source families within each seed;
   - `OFA-ABC`: mean of the three accepted OFA settings within each seed;
   - a domain-name-only ensemble, fixed without metrics: for financial
     targets use OFO-Amazon, OFO-YelpChi, OFA-A and OFA-B; for T-Social use
     OFO-Facebook, OFO-Flickr, OFO-Reddit, OFO-weibo, OFO-BlogCatalog and
     OFA-A.
3. Compare the paper score with the predeclared adhesion-only ablation and
   inspect seed stability.
4. Train label-free target-adapted models:
   - full T-Finance when feasible;
   - deterministic uniform node samples for the million-node targets, while
     retaining full-graph propagation context and full-graph inference;
   - fixed ANN candidates and a chunked algebraically equivalent community
     loss for memory-bounded training.
5. Test the shared T-Finance-fitted feature transform on T-Social. This route
   is valid only because their raw feature schemas are the same.
6. Select final configurations without target labels, run seeds 0/1/2, freeze
   all scores, and independently recompute AUROC/AUPRC.

## 4. Fixed settings

- Core model: paper-locked RECAP, 32 dimensions, four propagation hops,
  256 hidden units, two layers, 36 communities, K=64, 100 epochs.
- Optimizer and model hyperparameters remain those in the accepted Phase-1
  protocol unless a change is explicitly classified exploratory.
- Large-target inference candidates remain exact for T-Finance and the
  immutable FAISS-IVFPQ top-64 candidates for DGraph-Fin/T-Social.
- Primary metrics are AUROC and AUPRC on the same evaluation populations as
  the completed scalability experiment.
- Mean and population standard deviation are reported over seeds 0/1/2.

## 5. Acceptance

- No label or evaluation-mask object enters a model-facing graph.
- Every candidate checkpoint, dataset bundle, candidate array, score array,
  and final checkpoint is hashed.
- Resume logic may reuse an artifact only after its recorded hash and
  provenance match.
- Final tables separate Setting-A baseline, deployable label-free
  target-adapted results, and oracle source-scan results.
- Any change to feature alignment, training population, or score component is
  named in the table rather than folded into “RECAP” without qualification.
- Independent metric recomputation and a three-seed completeness audit must
  pass before a result is called final.
