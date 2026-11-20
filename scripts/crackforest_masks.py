"""Convert reviewed CrackForest MATLAB segmentation values to 0/255 PNG."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--foreground", type=int, required=True, choices=(1, 2),
                        help="Confirm the foreground value visually on local samples")
    args = parser.parse_args()
    annotations = sorted(args.ground_truth.glob("*.mat"))
    if not annotations:
        parser.error("No MATLAB annotations found")
    args.output.mkdir(parents=True, exist_ok=False)
    for annotation in annotations:
        mask = loadmat(annotation, simplify_cells=True)["groundTruth"]["Segmentation"]
        if mask.ndim != 2 or not set(np.unique(mask)).issubset({1, 2}):
            raise ValueError(f"Unexpected segmentation encoding in {annotation}")
        Image.fromarray((mask == args.foreground).astype(np.uint8) * 255).save(
            args.output / f"{annotation.stem}.png")


if __name__ == "__main__":
    main()
