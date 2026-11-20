# Final status

RoadCrack AI includes the data-preparation, PyTorch training, evaluation,
image and video inference, dashboard, API, Docker, experiments and reporting
code. The application starts on CPU. Serving real analyses requires trusted,
trained road-damage checkpoints.

| Deliverable | Status |
| --- | --- |
| Data code | VOC-to-YOLO, binary masks, reviewed site/video groups, source-balanced allocation, duplicate screening, EDA and archive inspection. Real reviewed train/validation/test manifests are pending. |
| Detection | YOLO26s, YOLO11s and YOLO26m official initialization checkpoints loaded on CPU. YOLO26s train/save/evaluate/predict paths exercised with temporary synthetic software fixtures. Road-trained weights and detector selection are pending. |
| Segmentation | U-Net++ and DeepLabV3+ forwards passed. U-Net++ checkpointing, evaluation and resume initialization ran against synthetic software fixtures. Tiled validation calibration and original-resolution mask operations have software tests. Road-trained weights and architecture selection are pending. |
| App | FastAPI and Streamlit start locally. Tests exercise image APIs, upload limits, missing weights, video jobs, cancellation at the processing layer, reports and dashboard startup. |
| Experiments and scientific evidence | Comparison plan and artifact collection are present. Real-data scores, error slices, independent-source testing, measured inference performance on road data, and actual demo images and video remain pending. |

## Prerequisites for the real-data run

- Make the authorized RDD2022 archive available locally or allow the publisher's
  S3 hostname `bigdatacup.s3.ap-northeast-1.amazonaws.com` or the exact Figshare
  download destinations in the managed environment. The tested S3 and
  `api.figshare.com` connections returned a proxy 403. Confirm reuse terms for
  the chosen archive. Confirm the CRACK500 permissions and download route if it
  is part of the experiment.
- Confirm whether the intended use permits non-commercial-only DeepCrack and
  CrackForest data. Use rights-cleared alternatives if the intended use is
  commercial. Supply reviewed capture groups and a fixed external site/source
  split where possible.
- Use a CUDA machine or the supplied Colab workflow. This instance reported
  Python 3.12.14 and PyTorch 2.7.1+cpu with CUDA unavailable. The default
  segmentation pretrained encoder host `huggingface.co` returned a proxy 403
  when checked. Stage authorized weights through a supported cache or allow
  the provider's required download hosts.
- Fit and freeze on validation, then score the untouched test split and publish
  metrics and demonstration media only when the relevant data and model terms
  permit them. See [the evaluation protocol](SCIENTIFIC_REPORT.md).

See [progress](PROGRESS.md) and [data terms](DATA_AND_LICENSES.md).

Verified checks include 15 passing pytest tests, compilation, dependency
consistency, a Docker image build with the cloud's trusted CA, and the API
health route inside the container. The live host dashboard health endpoint
returned HTTP 200.
