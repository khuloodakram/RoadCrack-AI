"""Inventory RDD2022 or DeepCrack files and join human-reviewed capture groups."""
import argparse
import csv
import os
from pathlib import Path


def pairs(root, source):
    if source == "RDD2022":
        for annotations in sorted(root.rglob("annotations/xmls")):
            partition = annotations.parent.parent
            if partition.name != "train":
                continue
            for xml in sorted(annotations.glob("*.xml")):
                image = partition / "images" / f"{xml.stem}.jpg"
                yield image, xml, str(partition.parent.relative_to(root))
    else:
        for split in ("train", "test"):
            labels = {path.stem: path for path in (root / f"{split}_lab").iterdir() if path.is_file()}
            for image in sorted((root / f"{split}_img").iterdir()):
                if image.suffix.lower() in (".jpg", ".png", ".jpeg", ".bmp"):
                    label = labels.get(image.stem)
                    if label is None:
                        raise ValueError(f"No paired label for {image}")
                    yield image, label, source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=("RDD2022", "DeepCrack"))
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--groups", type=Path, help="Reviewed CSV keyed by path relative to root. Omit for an inventory.")
    args = parser.parse_args()
    groups = {}
    if args.groups:
        with args.groups.open(newline="", encoding="utf-8-sig") as stream:
            groups = {row["key"]: row for row in csv.DictReader(stream)}
    records = []
    for image, annotation, source in pairs(args.root, args.source):
        if not image.is_file():
            raise ValueError(f"Missing image {image}")
        key = image.relative_to(args.root).as_posix()
        metadata = groups[key] if args.groups else {}
        records.append({"key": key, "image": os.path.relpath(image, args.output.parent),
                        "annotation": os.path.relpath(annotation, args.output.parent),
                        "source": source, "group": metadata.get("group", ""),
                        "split": metadata.get("split", "")})
    if not records:
        raise ValueError("No paired images and annotations found")
    with args.output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, records[0].keys())
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
