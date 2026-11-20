"""Inference over trusted, locally configured model weights."""
import hashlib
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


class WeightsUnavailable(RuntimeError):
    pass


DETECTION_NAMES = {"D00": "Longitudinal crack", "D10": "Transverse crack",
                   "D20": "Alligator crack", "D40": "Pothole"}


@lru_cache(maxsize=8)
def detector(path, signature):
    from ultralytics import YOLO
    model = YOLO(str(path))
    if model.task != 'detect' or not set(model.names.values()).issubset(DETECTION_NAMES):
        raise ValueError('Configure a road-damage detector checkpoint with supported class names')
    return model


@lru_cache(maxsize=8)
def segmenter(path, signature, device):
    from .segmentation import load_model
    return load_model(path, device)


def weight(path):
    if not path or not Path(path).is_file():
        raise WeightsUnavailable("ملف الأوزان المطلوب غير متاح")
    return Path(path).resolve()


@lru_cache(maxsize=16)
def version(path, signature):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def signature(path):
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def registry():
    values = os.getenv("ROADCRACK_DETECTORS", "yolo26s=models/yolo26s-trained.pt")
    return dict(item.split("=", 1) for item in values.split(",") if "=" in item)


@dataclass
class Analysis:
    result: dict
    overlay: Image.Image
    mask: Image.Image | None


INFERENCE_LOCK = threading.RLock()


@contextmanager
def inference_slot():
    with INFERENCE_LOCK:
        yield


class Pipeline:
    def __init__(self, model_name, mode="fast", confidence=.25):
        if model_name not in registry():
            raise ValueError("النموذج المحدد غير مسجل")
        if mode not in ("fast", "detailed") or not 0 <= confidence <= 1:
            raise ValueError("إعدادات التحليل غير صالحة")
        detect_path = weight(registry()[model_name])
        seg_path = weight(os.getenv("ROADCRACK_SEGMENTER", "models/segmentation.pt")) if mode == "detailed" else None
        self.model_name, self.mode, self.confidence = model_name, mode, confidence
        self.device = os.getenv("ROADCRACK_DEVICE", "cpu")
        self.image_size = int(os.getenv("ROADCRACK_IMAGE_SIZE", "640"))
        d_signature = signature(detect_path)
        self.versions = {"detector": version(detect_path, d_signature)}
        with inference_slot():
            self.detector = detector(detect_path, d_signature)
            self.segmenter = None
            if seg_path:
                s_signature = signature(seg_path)
                self.segmenter, checkpoint = segmenter(seg_path, s_signature, self.device)
                self.threshold = float(os.getenv("ROADCRACK_SEG_THRESHOLD", checkpoint["threshold"]))
                if not 0 <= self.threshold <= 1:
                    raise ValueError("الحد الأدنى لقناع التقسيم غير صالح")
                self.tile = int(os.getenv("ROADCRACK_TILE", checkpoint["config"]["size"]))
                self.overlap = int(os.getenv("ROADCRACK_OVERLAP", "96"))
                self.versions["segmenter"] = version(seg_path, s_signature)

    def analyze(self, image):
        start = time.perf_counter()
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        overlay = image.copy()
        mask = None
        segmentation = None
        with inference_slot():
            output = self.detector.predict(image, conf=self.confidence, device=self.device,
                                           imgsz=self.image_size, verbose=False)[0]
            if self.segmenter is not None:
                from .segmentation import tiled_probabilities
                pixels = tiled_probabilities(self.segmenter, image, self.tile,
                                             self.overlap, self.device) >= self.threshold
                mask = Image.fromarray(pixels.astype(np.uint8) * 255)
                overlay.paste(Image.new("RGB", image.size, "#ff6134"), mask=Image.fromarray(pixels.astype(np.uint8)*115))
                segmentation = {"foreground_pixels": int(pixels.sum()), "threshold": self.threshold,
                                "tile": self.tile, "overlap": self.overlap}
        detections = []
        draw = ImageDraw.Draw(overlay)
        for row in output.boxes.data.cpu().tolist():
            left, top, right, bottom, score, class_id = row
            label = output.names[int(class_id)]
            box = [max(0, min(left, width)), max(0, min(top, height)),
                   max(0, min(right, width)), max(0, min(bottom, height))]
            detections.append({"class": label, "name": DETECTION_NAMES.get(label, label),
                               "confidence": score, "xyxy": box})
            draw.rectangle(box, outline="#ffbe4a", width=3)
            draw.text((box[0]+2, box[1]+2), f"{label} {score:.2f}",
                      fill="white", stroke_width=2, stroke_fill="black")
        return Analysis({"width": width, "height": height, "mode": self.mode,
                         "model": self.model_name, "model_version": self.versions,
                         "detection_image_size": self.image_size, "confidence_threshold": self.confidence,
                         "processing_seconds": time.perf_counter()-start,
                         "detections": detections, "segmentation": segmentation}, overlay, mask)
