# FTTransformer: Baseline vs. Dedup (Seed 42)

Both experiments use the same validation/test files and
the same recorded training configuration. Only the training
dataset differs.

| Metric | Baseline | Dedup | Difference (Dedup − Baseline) |
|---|---:|---:|---:|
| ROC-AUC | 0.9642 | 0.9653 | +0.0010 |
| PR-AUC | 0.5188 | 0.5118 | -0.0071 |
| Precision | 0.2082 | 0.1920 | -0.0162 |
| Recall | 0.9096 | 0.9299 | +0.0203 |
| F1 | 0.3388 | 0.3183 | -0.0205 |

## Interpretation

In this single-seed experiment, test PR-AUC and F1 are lower
after deduplication, while Recall is higher. This does not
establish that deduplication is consistently harmful; more
random seeds are needed to assess variability.

The reported classification metrics use the thresholds and
evaluation procedure implemented in the training scripts.
See the baseline and dedup test-metrics JSON files for the
underlying recorded values.
