# 12-Dataset OFO Baseline Environment

Recorded on 2026-07-26 before formal execution.

| Component | Locked value |
|---|---|
| Environment | `/root/autodl-tmp/envs/recap-ofo-baselines` |
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu128 |
| PyG | 2.7.0 |
| PyGOD | 1.1.0 |
| XGBoost | 3.0.2 |
| NumPy | 2.1.3 |
| SciPy | 1.17.1 |
| scikit-learn | 1.8.0 |
| pandas | 2.3.1 |
| PyYAML | 6.0.2 |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Compute capability | 12.0 |
| GPU memory | 97,887 MiB |
| CPU memory | 1.0 TiB total, 912 GiB available at gate |
| Data disk | 100 GiB total, 68 GiB available at gate |

The environment is a `--system-site-packages` virtual environment based on the
already verified RECAP Python 3.12.3 environment. Only the locked OFO baseline
packages were installed into the child environment. This preserves the
completed RECAP and OFA environments.

The legacy releases target TensorFlow 1.x, DGL 0.4–0.8, and CUDA 11-era
PyTorch. They are retained as pinned source archives for provenance but are not
used as executable environments on the Blackwell GPU. Current PyTorch/PyG
adapters and the exact compatibility limits are locked in
`OFO_12_BASELINE_PROTOCOL.md`.
