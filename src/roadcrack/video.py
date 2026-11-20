"""Sample video frames and export an annotated video with timestamped observations."""
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class Cancelled(Exception):
    pass


def iou(first, second):
    w = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    h = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    union = sum((box[2]-box[0])*(box[3]-box[1]) for box in (first, second)) - w*h
    return w*h / union if union else 0


def process_video(path, folder, pipeline, sample=5, cancelled=lambda: False, max_frames=18000):
    if sample < 1:
        raise ValueError("Frame sampling must be positive")
    start = time.perf_counter()
    capture = cv2.VideoCapture(str(path))
    writer = None
    intermediate = folder / "intermediate.mp4"
    report = folder / "frames.jsonl"
    output = folder / "annotated.mp4"
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not math.isfinite(fps) or fps <= 0:
            raise ValueError("تعذر قراءة الفيديو أو معدل إطاراته")
        total = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if total > max_frames:
            raise ValueError("تجاوز الفيديو الحد الأقصى لعدد الإطارات")
        frames, analyzed, next_id = 0, 0, 1
        previous = []
        with report.open("w") as stream:
            while True:
                if cancelled():
                    raise Cancelled()
                decoded, frame = capture.read()
                if not decoded:
                    break
                if frames >= max_frames or frame.shape[0]*frame.shape[1] > 4096*2160:
                    raise ValueError("تجاوز الفيديو حدود المعالجة")
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps, (width + width % 2, height + height % 2))
                    if not writer.isOpened():
                        raise RuntimeError("تعذر إنشاء فيديو النتائج")
                if frames % sample == 0:
                    analysis = pipeline.analyze(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                    current = analysis.result["detections"]
                    available = list(previous)
                    for detection in sorted(current, key=lambda row: -row["confidence"]):
                        match = max((row for row in available if row["class"] == detection["class"]),
                                    key=lambda row: iou(row["xyxy"], detection["xyxy"]), default=None)
                        if match and iou(match["xyxy"], detection["xyxy"]) >= .5:
                            detection["track_id"] = match["track_id"]
                            available.remove(match)
                        else:
                            detection["track_id"] = next_id
                            next_id += 1
                    analysis.result.update(frame=frames, timestamp_seconds=frames/fps)
                    stream.write(json.dumps(analysis.result) + "\n")
                    frame = cv2.cvtColor(np.asarray(analysis.overlay), cv2.COLOR_RGB2BGR)
                    previous = current
                    analyzed += 1
                if width % 2 or height % 2:
                    frame = cv2.copyMakeBorder(frame, 0, height % 2, 0, width % 2, cv2.BORDER_REPLICATE)
                writer.write(frame)
                frames += 1
        if not frames or (total > 0 and frames + 1 < total):
            raise ValueError("الفيديو فارغ أو لم يكتمل فك ترميزه")
        writer.release()
        writer = None
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            args = [ffmpeg, "-nostdin", "-loglevel", "error", "-y", "-i", str(intermediate),
                    "-c:v", "libx264", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
            with subprocess.Popen(args, stderr=subprocess.DEVNULL) as process:
                while process.poll() is None:
                    if cancelled():
                        process.terminate()
                        raise Cancelled()
                    time.sleep(.1)
                if process.returncode:
                    raise RuntimeError("تعذر ترميز فيديو النتائج")
        else:
            intermediate.replace(output)
        elapsed = time.perf_counter() - start
        return {"frames": frames, "analyzed_frames": analyzed, "input_fps": fps,
                "processing_seconds": elapsed, "processing_fps": frames / elapsed,
                "estimated_tracks": next_id-1, "codec": "h264" if ffmpeg else "mp4v"}
    except Exception:
        output.unlink(missing_ok=True)
        report.unlink(missing_ok=True)
        raise
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        intermediate.unlink(missing_ok=True)
