# FTTransformer Deduplication Experiment — Seed 42

## Purpose

Test whether removing repeated training records with identical
actual model inputs and consistent labels changes performance.

## Data handling

- Original training rows: 744,658
- Deduplicated training rows: 744,078
- Rows removed: 580
- Negative rows removed: 487
- Positive rows removed: 93
- Validation and test files: unchanged and hash-verified

Patient-level data and processed arrays are intentionally excluded
from version control. Reproduction requires authorized access to
the corresponding locally prepared dataset.

## Training configuration

- Seed: 42
- Batch size: 512
- Epochs: 8
- Learning rate: 1e-4
- Weight decay: 1e-5

## Data and output paths

Set `TRANSFUSION_DATA_DIR` to the local directory containing
`preprocessing_metadata.json`, `train.npz`, `val.npz`, and
`test.npz`.

Optionally set `TRANSFUSION_OUTPUT_DIR` to choose where checkpoints
and training metrics are written. The default output directory is
this experiment's local `output/` directory.

The FTTransformer implementation and its Python dependencies must
also be installed and importable in the selected environment.

## Result

See `../../results/ft_rbc_seed42_comparison.md` for the comparison
with the original baseline.
