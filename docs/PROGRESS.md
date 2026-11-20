# Progress

## 2026-11-20

| Phase | Work completed and checks |
| --- | --- |
| 1. Foundation | Inspected the README-only checkout, added a Python 3.12 CPU environment, dependency pins and resolved constraints, CI, container definitions and environment diagnostics. CPU package install and `pip check` passed. CUDA and NVIDIA device nodes were unavailable. |
| 2. Data | Read all four publisher repositories. Retrieved and inspected DeepCrack and CrackForest archives outside Git. RDD2022 S3/Figshare probes and the CRACK500 Google Drive probe returned proxy 403. Added manifest inventory, group splitting, XML conversion, mask ingestion, validation, duplicate screening and EDA exports. Synthetic tests exercise conversion, deterministic splitting, duplicate and corrupt image handling, and mask encoding. |
| 3. Detection | Verified CPU load and forward passes for official YOLO26s, YOLO11s and YOLO26m weights. A temporary synthetic YOLO26s fit passed checkpoint reload, held-out evaluation plumbing and image prediction. Artifacts were removed after the check. |
| 4. Segmentation | Verified both architecture forwards. A temporary synthetic U-Net++ fit passed checkpoint save, reload, evaluation and resume initialization. Tests cover loss, original-coordinate tiles, threshold calibration and empty masks. |
| 5. Comparison | Added a planned experiment matrix, validation operating-point workflow and a generator for observed metrics. No real-data optimization experiment was run. |
| 6. Image analysis | Implemented fast and detailed modes, JSON, CSV, PDF and PNG exports. Software tests check coordinate clipping and empty detections. |
| 7. Video | Added sampled decoding, local trajectory estimates, cancellation and H.264 export when FFmpeg is available. Tested short synthetic video decode, output playback via OpenCV, cleanup, and FastAPI job polling and download paths. |
| 8. Service and dashboard | Started FastAPI and Streamlit and checked both local health endpoints. Streamlit home rendered through AppTest. Tested API limits, missing assets, corrupt input, missing weights and exports. |
| 9. Quality | Exercised routes using synthetic data and test doubles, and separately exercised the actual CPU training tools. Independent real-road image/video cases remain open. |
| 10. Delivery | Added Arabic usage instructions, engineering and scientific notes, Colab notebook and final status. The notebook's GPU cells were prepared for a supported runtime and remain unexecuted here. |

The first container build failed pip TLS verification behind the managed proxy.
A second build used the host's trusted CA bundle as a BuildKit secret while
keeping TLS verification enabled. The application does not include the CA bundle.

See [final status](FINAL_STATUS.md) for blockers and the final checks.

The final test run passed all 15 tests. The Starlette test client emitted one
AnyIO deprecation warning. The build completed using the trusted CA and the
container's API health check passed.
