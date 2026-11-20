"""Explicit software-only image, model architecture, and video fixtures."""
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from roadcrack.pipeline import Analysis, Pipeline, WeightsUnavailable
from roadcrack.segmentation import build_model, metrics, segmentation_loss, tiled_probabilities
from roadcrack.video import Cancelled, process_video


class ConstantModel(torch.nn.Module):
    def forward(self, x):
        return x[:, :1] * 0


class FakePipeline:
    def __init__(self, *args):
        pass

    def analyze(self, image):
        return Analysis(dict(width=image.width, height=image.height, mode="fast", model_version={},
                             detections=[{"class": "D00", "confidence": .8, "xyxy": [3, 4, 17, 8]}],
                             processing_seconds=.001, segmentation=None), image.copy(), None)


@pytest.mark.parametrize("architecture", ["UnetPlusPlus", "DeepLabV3Plus"])
def test_architecture_forward(architecture):
    torch.set_num_threads(1)
    model = build_model(architecture, "resnet18").eval()
    with torch.inference_mode():
        assert model(torch.zeros(1, 3, 64, 64)).shape == (1, 1, 64, 64)


def test_tiling_odd_dimensions_and_loss():
    image = Image.new("RGB", (175, 93))
    probs = tiled_probabilities(ConstantModel(), image, 64, 17, "cpu")
    assert probs.shape == (93, 175)
    np.testing.assert_array_equal(probs, .5)
    logits = torch.zeros(2, 1, 3, 4, requires_grad=True)
    segmentation_loss(logits, torch.zeros_like(logits)).backward()
    assert torch.isfinite(logits.grad).all()
    assert metrics([0, 0, 0])["iou"] is None


def test_no_weights(monkeypatch, tmp_path):
    monkeypatch.setenv("ROADCRACK_DETECTORS", f"yolo26s={tmp_path / 'missing.pt'}")
    with pytest.raises(WeightsUnavailable):
        Pipeline("yolo26s")


def test_video_export_and_cancel(tmp_path):
    input_path = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(str(input_path), cv2.VideoWriter_fourcc(*"mp4v"), 12, (64, 48))
    assert writer.isOpened()
    for i in range(12):
        writer.write(np.full((48, 64, 3), i * 12, dtype=np.uint8))
    writer.release()
    report = process_video(input_path, tmp_path, FakePipeline(), 3)
    assert report["frames"] == 12 and report["analyzed_frames"] == 4
    assert report["estimated_tracks"] == 1
    capture = cv2.VideoCapture(str(tmp_path / "annotated.mp4"))
    assert capture.read()[0]
    capture.release()
    event = threading.Event()
    event.set()
    with pytest.raises(Cancelled):
        process_video(input_path, tmp_path, FakePipeline(), cancelled=event.is_set)
    assert not (tmp_path / "intermediate.mp4").exists()
    with pytest.raises(ValueError):
        process_video(tmp_path / "missing.mp4", tmp_path, FakePipeline())


@pytest.mark.parametrize("rows", [[], [[-2, 4, 177, 8, .7, 0], [2, 5, 77, 8, .6, 1]]])
def test_pipeline_original_coordinates_and_empty_detections(tmp_path, monkeypatch, rows):
    from types import SimpleNamespace
    from roadcrack import pipeline

    checkpoint = tmp_path / "test.pt"
    checkpoint.write_bytes(b"fixture for hash only")
    monkeypatch.setenv("ROADCRACK_DETECTORS", f"fixture={checkpoint}")
    monkeypatch.setenv("ROADCRACK_SEGMENTER", str(checkpoint))
    monkeypatch.setenv("ROADCRACK_OVERLAP", "16")
    stub = SimpleNamespace(predict=lambda *args, **kwargs: [
        SimpleNamespace(names={0: "D00", 1: "D10"}, boxes=SimpleNamespace(data=torch.tensor(rows)))])
    monkeypatch.setattr(pipeline, "detector", lambda *args: stub)
    monkeypatch.setattr(pipeline, "segmenter", lambda *args: (
        ConstantModel(), {"config": {"size": 64}, "threshold": .75}))
    result = Pipeline("fixture", "detailed").analyze(Image.new("RGB", (175, 93)))
    assert result.overlay.size == result.mask.size == (175, 93)
    assert result.result["segmentation"]["foreground_pixels"] == 0
    assert len(result.result["detections"]) == len(rows)
    if rows:
        assert result.result["detections"][0]["xyxy"] == [0, 4, 175, 8]


def test_tiled_validation_calibration(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from roadcrack import segmentation

    for folder in ("images/val", "masks/val"):
        (tmp_path / folder).mkdir(parents=True)
    Image.new("RGB", (37, 41)).save(tmp_path / "images/val/one.png")
    Image.new("L", (37, 41), 255).save(tmp_path / "masks/val/one.png")
    (tmp_path / "report.json").write_text('{"dataset_version": "synthetic"}')
    monkeypatch.setattr(segmentation, "load_model", lambda *args: (ConstantModel(), {"threshold": .75}))
    segmentation.test_checkpoint(SimpleNamespace(
        device="cpu", resume=None, threshold=None, calibrate=True, run=tmp_path / "calibration",
        data=tmp_path, size=32, overlap=8, split="val"))
    result = json.loads((tmp_path / "calibration/val.json").read_text())
    assert result["dice"] == 1 and result["threshold"] == .25
    assert result["candidates"]["0.75"]["recall"] == 0
