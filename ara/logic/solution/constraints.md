# Constraints

- No target-label tuning, checkpoint selection, or calibration.
- ARC alone may use 10 labeled-normal target context nodes.
- IA-GGAD samples 10 unlabeled target references and excludes them from query.
- UNPrompt and AnomalyGFM-ZS use the full target query population.
- Standard deviation uses `ddof=0`.
- Compatibility changes must preserve objectives, samples, and label rights.
