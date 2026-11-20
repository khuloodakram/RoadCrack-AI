"""Synthetic fixtures exercise ingestion and split logic only."""
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from roadcrack.data import SPLITS, grouped_split, prepare, voc_labels


def test_voc_and_optional_potholes(tmp_path):
    annotation = tmp_path / "annotation.xml"
    annotation.write_text("<annotation><object><name>D40</name><bndbox><xmin>1</xmin>"
                          "<ymin>1</ymin><xmax>10</xmax><ymax>10</ymax></bndbox></object></annotation>")
    assert voc_labels(annotation, 10, 10, ("D00",)) == []
    assert voc_labels(annotation, 10, 10, ("D00", "D40")) == [(1, .5, .5, 1., 1.)]
    with pytest.raises(ValueError):
        voc_labels(annotation, 8, 8, ("D40",))


def test_split_is_deterministic():
    groups = {str(i): Counter({"source": 1}) for i in range(20)}
    split = grouped_split(groups, 42)
    assert split == grouped_split(dict(reversed(list(groups.items()))), 42)
    assert Counter(split.values()) == dict(zip(SPLITS, (14, 3, 3)))


def test_binary_masks_and_cross_group_duplicates(tmp_path):
    rng = np.random.default_rng(2)
    manifest = tmp_path / "manifest.csv"
    records = []
    for i in range(6):
        image = tmp_path / f"{i}.png"
        mask = tmp_path / f"{i}-mask.png"
        Image.fromarray(rng.integers(0, 255, size=(32, 32), dtype=np.uint8)).save(image)
        Image.fromarray(np.tile([31, 210], (32, 16)).astype(np.uint8)).save(mask)
        records.append([image.name, mask.name, "synthetic", f"site-{i // 2}"])
    def write_manifest():
        with manifest.open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(("image", "annotation", "source", "group"))
            writer.writerows(records)
    write_manifest()
    output = tmp_path / "processed"
    report = prepare(manifest, output, "segment")
    assert set(report["groups"].values()) == set(SPLITS)
    for path in output.glob("masks/*/*.png"):
        with Image.open(path) as mask:
            assert set(mask.getdata()) == {0, 255}
    Path(tmp_path / "2.png").write_bytes((tmp_path / "0.png").read_bytes())
    with pytest.raises(ValueError, match="Duplicate"):
        prepare(manifest, tmp_path / "other", "segment")
    (tmp_path / "2.png").write_bytes(b"invalid")
    with pytest.raises(OSError):
        prepare(manifest, tmp_path / "corrupt", "segment")
