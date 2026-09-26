# RECAP Large-Target Training-Milestone Diagnostic Protocol

Status: **locked before milestone scoring**

Date: 2026-07-28

Classification: **exploratory/oracle only**. The uniform target-adaptation
epoch-100 metrics were already observed before this diagnostic was proposed.

## Question

Did the fixed 100-epoch target-adaptation schedule overfit the label-free
community objective, such that an earlier saved checkpoint gives materially
better target ranking?

## Fixed diagnostic

- Reuse only the completed uniform target-adaptation checkpoints; do not
  retrain or modify them.
- Score epochs 25, 50, 75 and 100 for seeds 0/1/2 on all three full targets.
- Reuse the same immutable exact/ANN inference candidates and paper model
  settings as the confirmatory target-adaptation run.
- Freeze and hash paper, adhesion-only and context-only scores for every
  epoch/seed cell before the first label access in each target process.
- Use the original target evaluation populations.
- Report every epoch and route. Selecting the best epoch or component using
  target AUROC is explicitly an oracle diagnostic, never a deployable model
  selection rule.
- Never invert a score and never use negative component weights.

The diagnostic is accepted when all 36 score cells per target are present,
finite, hash-verified and independently recomputed.
