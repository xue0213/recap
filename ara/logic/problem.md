# Problem

Determine whether RECAP remains effective under the locked OFO/OFA protocol
and whether the supervised generalist baselines required by OFA Settings A,
B, and C can be reproduced fairly under identical source/target splits,
three-seed aggregation, and strict target-label boundaries.

The later OFO extension asks whether eight target-specific supervised and
unsupervised graph anomaly baselines can be reproduced on all twelve RECAP
graphs, including Questions, without quadratic-memory failures, unsupported
legacy dependencies, target-test tuning, or selective exclusion of weak
results.

The final extension asks whether DiffGAD and GUIDE can be added to the
full-graph unsupervised OFO stratum and whether OWLEYE can be reproduced in
OFA A/B/C under its actual source-label-supervised, target-label-free
zero-shot rights, without inheriting released label-selection leakage or
quadratic preprocessing.
