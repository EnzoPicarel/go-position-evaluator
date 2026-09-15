# Validation record

Validation performed while preparing the portfolio repository:

- Four focused unit tests pass: board encoding, symmetry augmentation, split isolation, and checkpoint round-trip inference.
- The Jupyter notebook passes `nbformat` structural validation.
- The local dataset used for the final submission contains 41,553 positions. The original handout text mentions 21,854, so both figures are documented rather than silently conflated.
- The corrected split has no overlap between training and validation symmetry groups.
- A one-epoch CPU smoke test on 256 positions completed training, validation, checkpoint export, and metrics export.

The smoke-test metrics are deliberately not presented as model performance: one epoch on a small prefix of the data is only an execution check. A corrected full-dataset training run has not yet been used to make benchmark claims.

## Reproduce the checks

```bash
python -m unittest discover -s tests -v
python train.py \
  --data data/samples-8x8.json.gz \
  --limit 256 \
  --epochs 1 \
  --output artifacts/smoke-test
```
