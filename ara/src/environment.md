# Environment

- Server GPU: NVIDIA RTX PRO 6000 Blackwell
- Python: 3.12.3
- PyTorch: 2.11.0+cu128
- PyTorch Geometric: 2.7.0
- NumPy: 2.1.3
- scikit-learn: 1.8.0
- einops: 0.8.1
- PyGOD: 1.1.0
- XGBoost: 3.0.2

The authoritative matrix is `rebuttal/baselines/environment_matrix.md`.
The isolated OFO baseline matrix is
`rebuttal/ofo_baselines/environment_matrix.md`.

DiffGAD, GUIDE, and OWLEYE use the same isolated OFO baseline environment.
Their pinned upstream revisions, archive hashes, and ORCA revision are in
`rebuttal/new_baselines/upstream_manifest.json`.
