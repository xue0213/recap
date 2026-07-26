# Constraints

- No target-label tuning, checkpoint selection, or calibration.
- ARC alone may use 10 labeled-normal target context nodes.
- IA-GGAD samples 10 unlabeled target references and excludes them from query.
- UNPrompt and AnomalyGFM-ZS use the full target query population.
- Standard deviation uses `ddof=0`.
- Compatibility changes must preserve objectives, samples, and label rights.
- OFO supervised methods use a 40/20/40 stratified split and only the 40% test
  nodes enter final metrics; OFO unsupervised methods use the full graph.
- Questions is mandatory for every OFO baseline method and seed.
- DOMINANT's structure term must be algebraically exact. AnomalyDAE and
  ADA-GAD may use only the pre-registered weighted 1:1 non-edge estimator.
- Weak and high-variance formal results cannot trigger target-test tuning,
  selective seed reruns, or dataset removal.
