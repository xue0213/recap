# Phase 2 Baseline Environment Matrix

Recorded on 2026-07-26 before formal baseline execution.

| Environment | Python | Torch / CUDA | PyG | DGL | GPU probe | Decision |
|---|---|---|---|---|---|---|
| `/root/autodl-tmp/envs/recap-py310` | 3.10.8 | import fails: missing `libcublasLt.so` | absent | import blocked by Torch | failed | reject |
| `/root/autodl-tmp/envs/recap-py3123` | 3.12.3 | 2.11.0+cu128 | 2.7.0 | absent | RTX PRO 6000 Blackwell, CC 12.0 | formal environment |

Formal adapters avoid a legacy DGL dependency by replacing only DGL graph
convolutions and dense masked adjacency operations with algebraically
equivalent PyTorch sparse operations. Equivalence tests are mandatory under
`BASELINE_OFA_REPROTOCOL.md`.

Other observed formal-environment packages:

- NumPy 2.1.3
- SciPy 1.17.1
- scikit-learn 1.8.0
- NVIDIA driver 595.58.03
- GPU memory 97,887 MiB

