"""Build reproducible datasets from a CSV of reviewed source records."""
import argparse
import csv
import hashlib
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

CLASSES = ("D00", "D10", "D20", "D40")
SPLITS = ("train", "val", "test")


def voc_labels(path, width, height, classes):
    """Read 1-based inclusive Pascal VOC boxes into normalized YOLO boxes."""
    labels = []
    root = ET.parse(path).getroot()
    size = root.find("size")
    if size is not None and (int(size.findtext("width")), int(size.findtext("height"))) != (width, height):
        raise ValueError(f"VOC image dimensions differ from {path}")
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name == "D40" and name not in classes:
            continue
        if name not in classes:
            raise ValueError(f"Unknown VOC class {name!r} in {path}")
        box = obj.find("bndbox")
        left, top, right, bottom = (float(box.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax"))
        if not (1 <= left <= right <= width and 1 <= top <= bottom <= height):
            raise ValueError(f"Invalid VOC rectangle in {path}")
        left, top = left - 1, top - 1
        labels.append((classes.index(name), (left + right) / (2 * width),
                       (top + bottom) / (2 * height), (right - left) / width,
                       (bottom - top) / height))
    return labels


def difference_hash(image):
    pixels = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    return int.from_bytes(np.packbits(pixels[:, 1:] > pixels[:, :-1]).tobytes())


def grouped_split(group_sources, seed, fixed=None):
    """Keep global groups intact and balance counts across source strata."""
    fixed = fixed or {}
    order = sorted(group_sources.keys() - fixed.keys())
    random.Random(seed).shuffle(order)
    order.sort(key=lambda group: -sum(group_sources[group].values()))
    targets = sum((group_sources[g] for g in group_sources), Counter())
    counts = {split: Counter() for split in SPLITS}
    ratios = dict(zip(SPLITS, (0.70, 0.15, 0.15)))
    assignment = dict(fixed)
    for group, split in fixed.items():
        counts[split].update(group_sources[group])
    for index, group in enumerate(order):
        remaining = len(order) - index
        empty = [split for split in SPLITS if not counts[split]]
        choices = empty if remaining == len(empty) else SPLITS
        best = min(choices, key=lambda split: sum(
            n * (2 * counts[split][source] + n - 2 * targets[source] * ratios[split])
            / max(1, targets[source] * ratios[split])
            for source, n in group_sources[group].items()
        ))
        assignment[group] = best
        counts[best].update(group_sources[group])
    return assignment


def prepare(manifest, dest, task, seed=42, include_d40=False, mask_threshold=127):
    if not 0 <= mask_threshold < 255:
        raise ValueError("mask_threshold must be between 0 and 254")
    classes = CLASSES if include_d40 else CLASSES[:3]
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    groups, records, exact = defaultdict(Counter), [], {}
    near, duplicates, fixed = [], [], {}
    buckets = defaultdict(list)
    fingerprint = hashlib.sha256(manifest.read_bytes())
    for row in rows:
        source, group = row["source"].strip(), row["group"].strip()
        if not source or not group:
            raise ValueError("Each image needs a source and a global site/video group")
        split_hint = row.get("split", "").strip()
        if split_hint:
            if split_hint not in SPLITS or fixed.setdefault(group, split_hint) != split_hint:
                raise ValueError(f"Invalid or conflicting fixed split for group {group}")
        path = manifest.parent / row["image"]
        annotation = manifest.parent / row["annotation"] if row["annotation"] else None
        with Image.open(path) as image:
            image.load()
            if image.format not in {"JPEG", "PNG", "BMP", "WEBP"}:
                raise ValueError(f"Unsupported image format in {path}")
            if image.getexif().get(274, 1) != 1:
                raise ValueError(f"Normalize EXIF orientation and annotations for {path}")
            width, height = image.size
            suffix = {"JPEG": ".jpg", "PNG": ".png", "BMP": ".bmp", "WEBP": ".webp"}[image.format]
            digest = hashlib.sha256(f"{width}x{height}:".encode() + image.convert("RGB").tobytes()).hexdigest()
            perceptual = difference_hash(image)
        labels = []
        if task == "detect":
            if annotation is None and row.get("negative", "").lower() != "true":
                raise ValueError(f"Mark reviewed negatives explicitly for {path}")
            labels = voc_labels(annotation, width, height, classes) if annotation else []
        else:
            if annotation is None:
                raise ValueError(f"Missing mask for {path}")
            with Image.open(annotation) as mask:
                if mask.size != (width, height):
                    raise ValueError(f"Mask dimensions differ for {path}")
                mask.load()
        fingerprint.update(digest.encode())
        if annotation:
            fingerprint.update(hashlib.sha256(annotation.read_bytes()).digest())
        annotation_digest = hashlib.sha256(annotation.read_bytes()).digest() if annotation else None
        if digest in exact:
            if exact[digest] != (group, annotation_digest):
                raise ValueError(f"Duplicate image with a different group or annotation: {path}")
            duplicates.append(str(path))
            continue
        exact[digest] = group, annotation_digest
        # Four differing bits cannot touch all five disjoint bands.
        band_keys = [(band, (perceptual >> (band * 13)) & 8191) for band in range(5)]
        candidates = set(candidate for key in band_keys for candidate in buckets[key])
        for previous, other_group, other_path in candidates:
            if group != other_group and (perceptual ^ previous).bit_count() <= 4:
                if len(near) < 20:
                    near.append([str(path), str(other_path)])
        for key in band_keys:
            buckets[key].append((perceptual, group, path))
        groups[group][source] += 1
        records.append((path, annotation, group, source, width, height, labels, suffix))
    if near:
        raise ValueError(f"Review cross-group near duplicates and merge groups or remove duplicates: {near[:20]}")
    if len(groups) < 3:
        raise ValueError("At least three independent groups are required")
    assignment = grouped_split(groups, seed, fixed)
    if set(assignment.values()) != set(SPLITS):
        raise ValueError("Fixed groups leave at least one split empty")
    dest.mkdir(parents=True, exist_ok=False)
    fingerprint.update(f"{task}:{seed}:{include_d40}:{mask_threshold}".encode())
    report = {"seed": seed, "dataset_version": fingerprint.hexdigest(), "duplicates_removed": duplicates,
              "groups": assignment, "splits": {}}
    for split in SPLITS:
        report["splits"][split] = {"images": 0, "classes": Counter(), "sources": Counter(), "dimensions": Counter(),
                                    "negative_images": 0, "foreground_pixels": 0}
    for index, (path, annotation, group, source, width, height, labels, suffix) in enumerate(records):
        split = assignment[group]
        image_out = dest / "images" / split / f"{index:08d}{suffix}"
        target = dest / ("labels" if task == "detect" else "masks") / split / f"{index:08d}"
        image_out.parent.mkdir(parents=True, exist_ok=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, image_out)
        stats = report["splits"][split]
        stats["images"] += 1
        stats["sources"][source] += 1
        stats["dimensions"][f"{width}x{height}"] += 1
        if task == "detect":
            target.with_suffix(".txt").write_text("".join(
                f"{c} {x:.8f} {y:.8f} {w:.8f} {h:.8f}\n" for c, x, y, w, h in labels))
            stats["classes"].update(classes[label[0]] for label in labels)
            stats["negative_images"] += not labels
        else:
            with Image.open(annotation) as mask:
                binary = np.asarray(mask.convert("L")) > mask_threshold
            Image.fromarray(binary.astype(np.uint8) * 255).save(target.with_suffix(".png"))
            stats["foreground_pixels"] += int(binary.sum())
            stats["negative_images"] += not binary.any()
        example = dest / f"example-{split}.jpg"
        if not example.exists():
            with Image.open(path) as image:
                preview = image.convert("RGB")
            if task == "detect":
                draw = ImageDraw.Draw(preview)
                for cls, x, y, w, h in labels:
                    draw.rectangle(((x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height),
                                   outline="orange", width=2)
                    draw.text(((x-w/2)*width, (y-h/2)*height), classes[cls], fill="orange")
            else:
                preview.paste(Image.new("RGB", preview.size, (235, 90, 0)), mask=Image.fromarray(binary.astype(np.uint8)*120))
            preview.thumbnail((1000, 1000))
            preview.save(example)
    if task == "detect":
        (dest / "dataset.yaml").write_text(yaml.safe_dump({"path": str(dest.resolve()),
            **{split: f"images/{split}" for split in SPLITS}, "names": list(classes)}))
    (dest / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["detect", "segment"])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-d40", action="store_true")
    parser.add_argument("--mask-threshold", type=int, default=127)
    args = parser.parse_args()
    prepare(args.manifest, args.output, args.task, args.seed, args.include_d40, args.mask_threshold)


if __name__ == "__main__":
    main()
