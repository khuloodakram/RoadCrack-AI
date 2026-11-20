"""Generate JSON, CSV, PDF, overlay and optional mask for one image."""
import argparse
from pathlib import Path

from PIL import Image, ImageOps

from roadcrack.exports import image_exports
from roadcrack.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="yolo26s")
    parser.add_argument("--mode", choices=("fast", "detailed"), default="fast")
    parser.add_argument("--confidence", type=float, default=.25)
    args = parser.parse_args()
    pipeline = Pipeline(args.model, args.mode, args.confidence)
    with Image.open(args.image) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    analysis = pipeline.analyze(image)
    args.output.mkdir(parents=True, exist_ok=False)
    image_exports(args.output, image, analysis)


if __name__ == "__main__":
    main()
