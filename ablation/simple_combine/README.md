# Simple Combine Ablation

This folder contains the residual + simple detector ablation for RECAP.

Run the experiment from the RECAP project root:

```bash
/root/miniconda3/bin/python ablation/simple_combine/run_simple_combine.py \
  --device cuda:0 \
  --model recap_auprc_best \
  --trials 5 \
  --epochs 100 \
  --no-diagnostics \
  --continue-on-error \
  --output-dir ablation/simple_combine/results
```

The script compares:

- Residual + KMeans
- Residual + Spectral Clustering
- Residual + GMM
- Residual + LOF
- RECAP

Outputs are written to `ablation/simple_combine/results/` as JSON, CSV, and Markdown
tables. The residual baselines use the same feature alignment and multi-hop
propagation as RECAP, then score nodes directly in the raw residual space.

For large target graphs, spectral clustering defaults to `--spectral-large-mode sample`:
it runs spectral clustering on a sampled residual subgraph capped by
`--spectral-max-nodes 3000` and scores all nodes by distance to the sampled spectral
centroids. Use `--spectral-large-mode error --spectral-max-nodes 20000` if you
want exact full-graph spectral clustering and prefer explicit failures on graphs
that exceed the cap.

GMM uses a robust fitting fallback: near-constant residual dimensions are removed,
then failed fits are retried with larger `reg_covar`, and finally fewer mixture
components if needed. This prevents covariance-degeneracy failures on graphs such
as Reddit while keeping the baseline in the Gaussian-mixture family.
