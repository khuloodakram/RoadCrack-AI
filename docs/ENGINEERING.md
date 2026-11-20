# Engineering and training guide

## Prepare the data

Obtain a dataset release whose terms permit your use. RDD2022 typically uses
`<region>/train/images/` and `<region>/train/annotations/xmls/`. DeepCrack ships
a ZIP with `train_img`, `train_lab`, `test_img`, and `test_lab`.

```bash
python scripts/source_manifest.py RDD2022 data/raw/rdd data/rdd-inventory.csv
python scripts/source_manifest.py RDD2022 data/raw/rdd data/detection.csv --groups data/reviewed-rdd-groups.csv
python -m roadcrack.data detect data/detection.csv data/processed-detection
python -m roadcrack.data segment data/segmentation.csv data/processed-segmentation
```

Use [the sample schema](../configs/manifest.example.csv). The reviewed groups
CSV uses `key,group,split` and joins on the relative source path in the
inventory. Use the same group identifier for all views of a site, route
segment or video sequence even across sources. A file name or country alone
cannot establish site identity. Reserve independent groups with `split=test`,
and explicitly mark reviewed negatives with `negative=true`. Store labels
and images with matching orientation. For DeepCrack, use
`source_manifest.py DeepCrack` on the extracted ZIP. For CrackForest, use
`scripts/crackforest_masks.py --help` and verify foreground polarity before
pairing converted masks with source images. For CRACK500, use paired masks
only after confirming rights, matching geometry, and capture identity across
crops. Use `--mask-threshold` when source masks need a different grayscale
cutoff. Inspection metadata stays local.

Preparation rejects malformed inputs, incompatible dimensions and cross-group
perceptual duplicates. It removes exact intra-group duplicates with identical
labels. A difference hash catches some near duplicates; manual verification
and capture metadata are required for other scene similarity, including
different perspectives. The approximate 70/15/15 allocation balances source
counts without splitting a group. Small or fixed pools can deviate. Read
`report.json` and per-split previews before fitting. D40 is an optional
pothole class. The default omits its boxes; review pothole-only images before
treating them as crack negatives. Segmentation learns binary masks independently.

## Fit and evaluate

For GPU, use Python 3.12 with an official mutually compatible CUDA Torch and
Torchvision build. Keep all other direct pins; remove the CPU wheel lines
before installing them into that environment. Set `PYTHONPATH=src`.
The Colab notebook has a GPU setup and persistent run directory. Allow the
encoder download destination or stage authorized weights in the framework
cache using the official tools. Use trusted artifacts only.

```bash
python -m roadcrack.detection train --data data/processed-detection \
  --weights models/yolo26s.pt --run runs/fit/det-01 --device 0
python -m roadcrack.segmentation train --data data/processed-segmentation \
  --pretrained --run runs/fit/seg-01 --device cuda:0
```

Run YOLO11s, YOLO26m and DeepLabV3+ in separate directories and preserve the
same dataset version and seed. For fine-tuning use a previous checkpoint via
`--weights` and a new run, optionally tuning with `--freeze` and `--lr`.
For interrupted detection training use the original run and its
`fit/weights/last.pt` with `--resume`. For segmentation use `--resume
<run>/last.pt` with original optimization and data settings. Preserve the
complete run. Update the absolute YOLO dataset YAML path only when relocating
the prepared dataset; a resumable Ultralytics run also stores its original
dataset path. Training batches alone receive optional horizontal flips and
limited detection exposure/scale changes.

```bash
python -m roadcrack.detection evaluate --data data/processed-detection \
  --weights runs/fit/det-01/fit/weights/best.pt --run runs/det-01 --split test
python -m roadcrack.segmentation evaluate --data data/processed-segmentation \
  --resume runs/fit/seg-01/best.pt --run runs/seg-01
python scripts/compare_runs.py
```

Run validation separately for comparison and threshold selection. The
segmentation learner picks an initial threshold using resized validation masks.
To calibrate with original-resolution tiling, run segmentation evaluation with
`--split val --calibrate` and inspect `val.json`. Use the resulting threshold
as `--threshold` for the fixed test run and `ROADCRACK_SEG_THRESHOLD` when
serving. Freeze tile size and overlap on validation. Configure
`ROADCRACK_IMAGE_SIZE` to serve the detection resolution selected in validation.
Annotations and predictions in segmentation example images use blue for
missed foreground, orange for false positives, green for true positives.

## Serve

FastAPI serves artifacts by opaque job IDs. Streamlit requests them through
the API. Model instances are cached by checkpoint metadata and use serialized
inference within one process. Run one API worker for in-memory video job
state. Put a deployed service behind authenticated access and manage quotas,
retention and persistent job state before multi-user use. Remove old local
artifacts with your data retention policy. Result IDs are bearer references in
this implementation. Uploaded originals are discarded for video after the job;
image originals and completed outputs are stored for downloads. Audio is
removed from result video. Video IDs use greedy class-and-overlap association
between adjacent sampled frames. Track totals are estimates susceptible to
viewpoint changes, occlusion, and frame spacing. H.264 requires FFmpeg with
libx264; without FFmpeg, native OpenCV MP4 output depends on codec availability.

For local prediction use `scripts/analyze_image.py --help` or
`python -m roadcrack.detection predict --help`. The pipeline outputs boxes in
pixels, optional original-sized masks, an overlay and a checkpoint SHA-256.
Detection overlap handling uses the selected model's post-processing. The
two tasks retain independent outputs and share one coordinate system.


On managed networks with a trusted HTTPS inspection CA, the Docker build
accepts a BuildKit CA bundle for pip. For the tested cloud image use
`docker build --secret id=pip_ca,src=/etc/ssl/certs/ca-certificates.crt -t roadcrack-ai:local .`.
Then use `docker compose up --no-build` after configuring `.env`.
