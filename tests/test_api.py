import io
import time

import cv2
import numpy as np

from fastapi.testclient import TestClient
from PIL import Image
from streamlit.testing.v1 import AppTest

from app import api
from tests.test_inference import FakePipeline

client = TestClient(api.app)


def png():
    stream = io.BytesIO()
    Image.new("RGB", (128, 81)).save(stream, format="PNG")
    return stream.getvalue()


def test_health_and_error_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("ROADCRACK_ARTIFACTS", str(tmp_path))
    monkeypatch.setenv("ROADCRACK_DETECTORS", f"yolo26s={tmp_path / 'missing.pt'}")
    assert client.get("/health").status_code == 200
    assert client.post("/images", files={"file": ("a.txt", b"file", "text/plain")}).status_code == 415
    assert client.post("/images", files={"file": ("a.png", b"file", "image/png")}).status_code == 422
    assert client.post("/images", files={"file": ("a.png", png(), "image/png")}).status_code == 503
    assert not list(tmp_path.iterdir())
    monkeypatch.setattr(api, "MAX_BODY", 200)
    assert client.post("/images", files={"file": ("a.png", png(), "image/png")}).status_code == 413
    def chunks():
        yield b"--abc\r\nContent-Disposition: form-data; name=\"file\"; filename=\"x.png\"\r\n\r\n"
        yield b"x" * 250
        yield b"\r\n--abc--\r\n"
    assert client.post("/images", content=chunks(),
                       headers={"content-type": "multipart/form-data; boundary=abc"}).status_code == 413


def test_image_downloads(tmp_path, monkeypatch):
    monkeypatch.setenv("ROADCRACK_ARTIFACTS", str(tmp_path))
    monkeypatch.setattr(api, "Pipeline", FakePipeline)
    response = client.post("/images", files={"file": ("a.png", png(), "image/png")})
    assert response.status_code == 200
    job = response.json()["id"]
    for filename in ("analysis.json", "detections.csv", "report.pdf", "overlay.png", "original.png"):
        assert client.get(f"/artifacts/{job}/{filename}").status_code == 200
    assert client.get(f"/artifacts/{job}/mask.png").status_code == 404


def test_dashboard_home():
    app = AppTest.from_file("app/dashboard.py").run()
    assert not app.exception
    assert "ابدأ تحليل الطريق" in [heading.value for heading in app.header]


def test_video_job(tmp_path, monkeypatch):
    monkeypatch.setenv("ROADCRACK_ARTIFACTS", str(tmp_path / "artifacts"))
    monkeypatch.setattr(api, "Pipeline", FakePipeline)
    path = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.release()
    response = client.post("/videos", files={"file": ("tiny.mp4", path.read_bytes(), "video/mp4")},
                           data={"sample": 2})
    assert response.status_code == 202
    job = response.json()["id"]
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        result = client.get(f"/videos/{job}").json()
        if result["status"] in {"done", "failed"}:
            break
        time.sleep(.05)
    assert result["status"] == "done", result
    assert result["analyzed_frames"] == 2
    assert client.get(f"/artifacts/{job}/annotated.mp4").status_code == 200
    assert len(client.get(f"/artifacts/{job}/frames.jsonl").text.splitlines()) == 2
    assert client.get(f"/artifacts/{job}/analysis.json").json()["observations"] == 2
    assert client.get(f"/artifacts/{job}/detections.csv").status_code == 200
    assert client.get(f"/artifacts/{job}/report.pdf").content.startswith(b"%PDF")
    assert not (tmp_path / "artifacts" / job / "upload").exists()
