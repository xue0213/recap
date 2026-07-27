# T-Finance Single-Source Large-Target Transfer Protocol

Status: locked before any new DGraph-Fin or T-Social score is computed.

## Question

Does label-free training on the large financial graph T-Finance improve RECAP
inference on T-Finance itself and transfer to DGraph-Fin or T-Social?

## Fixed source training

- Reuse the completed uniform T-Finance target-adaptation runs at epoch 100.
- The training graph is the full 39,357-node T-Finance graph.
- Training uses no anomaly labels or evaluation masks.
- Seeds are exactly 0, 1 and 2.
- Checkpoint SHA-256:
  - seed 0: `9ab4717011ece6ef31934a3d54dd03511e79bd86d87e63c6c38e23658b0be9b0`
  - seed 1: `e71e9fde5cd11bc36e15376c1de3cb65bcbaef8feb55bb94d3e0307d2c96d1fd`
  - seed 2: `1cd24e8fde981fe90ece805e801f363902652f9299d4837fdb506a421c87ae33`
- No checkpoint, epoch, seed, score direction or score component may be
  selected using any of the three targets' labels.

## Fixed targets and inference

- Targets: T-Finance, DGraph-Fin and T-Social.
- Each target uses its existing canonical independently aligned 32-dimensional
  features. This is the standard RECAP alignment route, not the exploratory
  source-fitted shared-coordinate route.
- Fixed candidate graphs:
  - T-Finance exact top-64:
    `f16f69f6dafc2d6c75d919d61916320798d320baef880884e45a4f7baf0400b8`
  - DGraph-Fin ANN top-64:
    `925d5c65249a3c279ce39b6920a0f0ef88d6f57d0ac3e6e5d2e6da60eda414d6`
  - T-Social ANN top-64:
    `4430c335dad8899435599c567897d65786a9eec73c0d805a46fe7db970e2f1c0`
- The primary route is the paper's full score. Adhesion-only and context-only
  are frozen named diagnostics and are never substituted for the primary
  result.
- DGraph-Fin scores all nodes but metrics use only its immutable official
  evaluation mask.

## Evidence boundary and acceptance

- T-Finance is an unsupervised same-graph training/evaluation result.
- DGraph-Fin and T-Social are target-label-free cross-graph transfer results.
- These strata must be reported separately and must not be combined into one
  macro average.
- All 27 score vectors (3 targets x 3 seeds x 3 routes) must be finite,
  atomically saved and SHA-256 frozen before any target label is loaded.
- Evaluation reports AUROC and AUPRC as mean plus population standard
  deviation over seeds 0/1/2. Score inversion is forbidden.
- An independent audit must rehash every score and recompute every metric.

## Predeclared interpretation

Training on T-Finance may help DGraph-Fin if the two financial datasets share
transferable residual-community structure. Failure to improve DGraph-Fin
would indicate that domain identity alone is insufficient under their feature
and graph-distribution shift. T-Social is a cross-domain negative control.
