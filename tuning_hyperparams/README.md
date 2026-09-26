# Optional sensitivity tools

These scripts are optional diagnostics and are not required for the main RECAP
training or inference path. Run them from the repository root after installing
`requirements.txt` and placing the datasets described in the top-level README.

Run a sensitivity sweep with the paper configuration:

```bash
./tuning_hyperparams/run_sensitivity.sh
```

Use a smaller smoke run with:

```bash
TRAIN_DATASETS="pubmed Flickr" \
TEST_DATASETS="cora ACM weibo" \
EPOCHS=80 TRIALS=1 RUN_NAME=recap_sens_quick \
./tuning_hyperparams/run_sensitivity.sh
```

The lambda_E=0 comparison requires a completed sensitivity sweep because it
compares against that sweep's summary file. Pass both paths explicitly:

```bash
python tuning_hyperparams/lambda_e0_invariance.py \
  --base-config params/recap_auprc_best.json \
  --reference-summary tuning_hyperparams/sensitivity_results/<run>/sensitivity_summary.csv
```

All generated results are written to ignored local directories.
