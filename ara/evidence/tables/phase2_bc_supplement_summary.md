# E03: B/C baseline completion supplement

All values are AUROC/AUPRC dataset-macro means over three seeds.

| Setting | UNPrompt | AnomalyGFM-ZS | RECAP |
|---|---:|---:|---:|
| B | 0.5792/0.1169 | 0.4964/0.0843 | 0.6775/0.2198 |
| C | 0.5925/0.1137 | 0.5103/0.0557 | 0.6731/0.1750 |

Formal scope: 12 training runs and 60 evaluations. Independent audit checked
12 checkpoints, 60 frozen score vectors, 60 recomputed metrics, and 168 label
events with zero problem.
