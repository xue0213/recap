# RECAP Large-Target Predictive Optimization Report

Date: 2026-07-28

## Scope and conclusion

This phase tested whether RECAP's weak zero-shot predictions on T-Finance,
DGraph-Fin and T-Social could be repaired without target labels. It preserves
the completed scalability result and never reverses score direction.

The computational result remains valid, but the predictive conclusion is
negative. No primary full-RECAP configuration approaches 0.70 AUROC. The
strongest full-score means found are 0.3211 on T-Finance, 0.3839 on
DGraph-Fin and 0.4845 on T-Social; the first is exploratory and the latter
two require oracle target-metric source selection. A named adhesion-only
exploratory transfer
reaches 0.5187 on T-Social; this is not the paper's full score and its shared
coordinate reproduction gate failed.

## Evidence strata

- **Confirmatory/deployable addendum:** the locked uniform, label-free target
  adaptation; all full-target scores froze before labels were loaded.
- **Exploratory:** robust-core adaptation proposed after the uniform results,
  and T-Finance-to-T-Social shared-coordinate transfer whose strict
  pointwise reproduction gate failed.
- **Oracle:** per-target source-family/checkpoint selection and
  target-metric selection among epochs 25/50/75/100. These diagnose upper
  bounds only and are not deployable selection rules.

## Primary paper-score results

Mean ± population standard deviation over seeds 0/1/2:

| Evidence/configuration | T-Finance AUROC / AUPRC | DGraph-Fin AUROC / AUPRC | T-Social AUROC / AUPRC |
|---|---:|---:|---:|
| Original Setting-A zero-shot | 0.2557 ± 0.0200 / 0.0280 ± 0.0008 | 0.3675 ± 0.0065 / 0.00904 ± 0.00009 | 0.4379 ± 0.0052 / 0.02595 ± 0.00012 |
| Uniform target adaptation | 0.3116 ± 0.0186 / 0.03019 ± 0.00077 | 0.3711 ± 0.0072 / 0.00908 ± 0.00011 | 0.4216 ± 0.0040 / 0.02485 ± 0.00017 |
| Robust-core adaptation (exploratory) | 0.3211 ± 0.0113 / 0.03057 ± 0.00050 | 0.3700 ± 0.0043 / 0.00905 ± 0.00005 | 0.4191 ± 0.0074 / 0.02462 ± 0.00049 |
| Shared-coordinate transfer (exploratory) | — | — | 0.4549 ± 0.0125 / 0.02656 ± 0.00095 |
| Best source family selected by target metric (oracle) | OFA-A: 0.2557 / 0.02801 | OFA-C: 0.3839 / 0.00931 | OFO-Facebook: 0.4845 / 0.02932 |
| Best uniform milestone selected by target metric (oracle) | epoch 100: 0.3116 / 0.03019 | epoch 75: 0.3716 / 0.00909 | epoch 25: 0.4316 / 0.02528 |

The evaluation prevalences/AUPRC random references are 0.04584,
0.01265 and 0.03015, respectively. Thus even the best primary full-score
AUPRC remains below prevalence on every target.

## Named component diagnostics

| Target | Strongest three-seed named component result | Status |
|---|---:|---|
| T-Finance | Robust-core adhesion-only: 0.3221 / 0.03068 | Exploratory |
| DGraph-Fin | OFA-C context-only: 0.4646 / 0.01106 | Source-selected oracle |
| T-Social | Shared-coordinate adhesion-only: 0.5187 / 0.03045 | Exploratory; strict coordinate gate failed |

Additional boundaries:

- The predeclared social-domain ensemble reaches 0.5022 AUROC with
  adhesion-only. It is label-blind by construction and avoids choosing one
  source family using target metrics.
- The best target-selected T-Social source family for adhesion-only is
  OFO-weibo at 0.5126/0.03144. The best individual checkpoint reaches
  0.5209, but seed selection is an even stronger oracle and is not used as a
  final result.
- The best individual DGraph-Fin context-only checkpoint reaches 0.4909.
  The corresponding three-seed family mean is 0.4646.
- The best individual T-Finance context-only checkpoint reaches 0.4539, but
  its three-seed family mean is only 0.3707.

## Mechanistic findings

1. **Existing source choice is not enough.** All T-Finance full-score source
   families remain below random. DGraph-Fin and T-Social improve under oracle
   source selection, but not to a practically strong level.
2. **Uniform target adaptation is not a general fix.** It modestly improves
   T-Finance, is nearly neutral on DGraph-Fin and hurts T-Social.
3. **Normal-core filtering is not the missing mechanism.** Removing the most
   extreme 10% of nodes by a fixed label-free feature/degree proxy yields only
   +0.0095 AUROC on T-Finance and degrades both larger graphs.
4. **Independent feature coordinates matter somewhat on T-Social.** The
   shared T-Finance coordinate system improves T-Social adhesion, but the
   source reproduction maximum error (3.18e-4) exceeded the locked 1e-4 gate,
   so the route remains exploratory.
5. **Epoch-100 overfitting is not the main cause.** Earlier checkpoints do
   not help T-Finance, change DGraph-Fin by at most about 0.0005 full-score
   AUROC, and improve T-Social full-score AUROC by only about 0.010.

## Integrity controls

- Three canonical bundles, aligned features, fixed KNN candidates and all 45
  accepted source checkpoints match their prior SHA-256 values after server
  migration.
- The original paper score direction is retained; no `1-score`, sign flip or
  negative component weighting is used.
- DGraph-Fin metrics use only its immutable 0/1 evaluation mask; every node
  is still scored.
- Every training graph excludes labels and evaluation masks.
- All score vectors are finite and frozen before the first label access in
  the corresponding process.
- The final combined independent audit passed 657 frozen score-route hashes
  and 657 metric recomputations with maximum difference 0. Its SHA-256 is
  `d70681e4388052bd661f451707acd99607b9da0b617f734e79705fac61c9bf88`.

## Practical interpretation

These experiments support RECAP's ability to execute target-side inference on
million-node graphs, but they do not support a claim of strong predictive
effectiveness on these three cross-domain targets. The strongest defensible
rebuttal position is therefore computational scalability plus transparent
negative predictive evidence, not a claim that the optimized results are
competitive with approximately 0.70 AUROC supervised or in-domain methods.
