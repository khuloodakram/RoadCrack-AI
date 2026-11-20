"""Produce machine-readable downloads and a PDF summary."""
import csv
import json
import textwrap

from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas


def image_exports(folder, image, analysis):
    image.save(folder / "original.png")
    analysis.overlay.save(folder / "overlay.png")
    result = analysis.result
    if analysis.mask is not None:
        analysis.mask.save(folder / "mask.png")
    (folder / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (folder / "detections.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("class", "confidence", "left", "top", "right", "bottom"))
        writer.writerows([row["class"], row["confidence"], *row["xyxy"]] for row in result["detections"])
    pdf = Canvas(str(folder / "report.pdf"))
    pdf.setTitle("RoadCrack AI analysis")
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(42, 800, "RoadCrack AI")
    pdf.setFont("Helvetica", 11)
    pdf.drawString(42, 775, f'{result["width"]} x {result["height"]} px | {result["mode"]} | '
                             f'{result["processing_seconds"]:.3f} s')
    pdf.drawString(42, 755, f'Detections: {len(result["detections"])}')
    preview = analysis.overlay.copy()
    preview.thumbnail((1000, 1000))
    pdf.drawImage(ImageReader(preview), 42, 270, 510, 460, preserveAspectRatio=True, anchor="c")
    y = 245
    for key, checksum in result["model_version"].items():
        for line in textwrap.wrap(f"{key}: {checksum}", 75):
            pdf.drawString(42, y, line)
            y -= 15
    pdf.save()


def video_exports(folder, summary):
    with (folder / "frames.jsonl").open(encoding="utf-8") as frames, (folder / "detections.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("frame", "timestamp_seconds", "track_id", "class", "confidence",
                         "left", "top", "right", "bottom"))
        observations = 0
        for line in frames:
            frame = json.loads(line)
            for detection in frame["detections"]:
                observations += 1
                writer.writerow([frame["frame"], frame["timestamp_seconds"], detection["track_id"],
                                 detection["class"], detection["confidence"], *detection["xyxy"]])
    (folder / "analysis.json").write_text(json.dumps(dict(summary, observations=observations), indent=2),
                                         encoding="utf-8")
    pdf = Canvas(str(folder / "report.pdf"))
    pdf.setTitle("RoadCrack AI video analysis")
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(42, 800, "RoadCrack AI | Video")
    pdf.setFont("Helvetica", 11)
    for index, line in enumerate((
        f'Frames: {summary["frames"]} | Analyzed: {summary["analyzed_frames"]}',
        f'Processing: {summary["processing_seconds"]:.2f} s | {summary["processing_fps"]:.2f} FPS',
        f'Observations: {observations} | Estimated tracks: {summary["estimated_tracks"]}',
    )):
        pdf.drawString(42, 775 - 20 * index, line)
    pdf.save()
