# Scientific report

No road-crack model was fitted or evaluated on real road data in this
environment. The [experiment matrix](../configs/experiment-matrix.csv) is a
protocol, and published road-crack scores, selected models, demo media and
derived weights are pending. Software tests use synthetic fixtures. COCO
checkpoints were loaded for compatibility checks and are not road models.

## Evaluation protocol

- Build a versioned and reviewed group split. Keep GPS-correlated sites,
  repeated traversals and video sequences together. Leave entire source/site
  groups out for transfer evaluation when there are enough independent
  groups. On RDD2022 use a labeled holdout from the archive's annotated
  portion or another verified external source.
- Record sample counts per class, source and split. Report precision, recall,
  AP50 and AP50–95 by class alongside summary mAP, confusion matrices and
  precision–recall plots. Report latency and throughput on the target device,
  with warmup and size, and with the same confidence policy across contenders.
- For segmentation report Dice, IoU, precision and recall from pixel confusion
  counts. Report recall in a narrow ground-truth boundary band as a proxy for
  missed thin structure, and review thin-crack cases separately by visual
  inspection. Background-only masks with undefined denominators yield null
  metrics. Keep thresholds and tiling choices fixed on test.
- Review errors on independent road scenes with thin and overlapping cracks,
  seam lines, paint markings, shadows, dark pavement, wet pavement and
  high-resolution inputs. Slice by source and camera where metadata exists.
  Document missing strata and label ambiguity. If estimating uncertainty,
  bootstrap groups rather than correlated frames.
- Select a detection/segmentation pair using validation, considering
  per-class recall, false positives, robustness and the application's latency
  budget. Use a still-untouched test set for the final comparison. A single
  accuracy number cannot answer these questions.
