# Experiments

The planned comparison is in [the matrix](../configs/experiment-matrix.csv). Keep
model fitting under `runs/fit/<id>` and final evaluation under `runs/<id>`.
Run `python scripts/compare_runs.py` to generate the dashboard report from the
evaluation JSON files actually present. An unevaluated row retains its planned
status. Keep validation results separate from fixed holdout scores and record
the dataset version, checkpoints and metric definitions alongside results.
