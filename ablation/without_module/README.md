# Without-Module Ablation

This folder contains the core module ablation for RECAP. The script keeps the
main implementation untouched and injects controlled variants at runtime:

- `RECAP`: full model.
- `RECAP w/o Residual`: replaces residual hop differences with non-residual
  propagated-hop concatenation.
- `RECAP w/o L_con`: removes the residual-similarity graph consistency loss.
- `RECAP w/o L_H`: removes assignment sharpness/balance regularization.
- `RECAP w/o Adhesion Score`: evaluates with only the KNN community context
  inconsistency score.
- `RECAP w/o Context Score`: evaluates with only the prototype adhesion score.
- `RECAP C=1`: collapses the community layer to one community to test whether
  multi-community structure is necessary.

The default model configuration is `params/recap_auprc_best.json`
(`k=64`, `C=36`); `RECAP C=1` overrides only the community count.

Formal run:

```bash
cd /root/autodl-fs/recap

/root/miniconda3/bin/python ablation/without_module/run_without_module.py \
  --device cuda:0 \
  --model recap_auprc_best \
  --trials 5 \
  --epochs 100 \
  --no-diagnostics \
  --continue-on-error \
  --output-dir ablation/without_module/results
```

Use `--device cpu` if CUDA is unavailable. Results are written as JSON, CSV, and
Markdown tables under the output directory.
