"""Run detection baseline, fine-tuning, held-out evaluation, or prediction."""
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import yaml

def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


BASELINES = ("yolo26s.pt", "yolo11s.pt", "yolo26m.pt")


def detector_metrics(model, dataset, args, output):
    result = model.val(data=str(dataset), split=args.split, imgsz=args.size, batch=args.batch, workers=args.workers,
                       device=args.device, plots=True, project=str(output.parent), name=output.name)
    scores = {key: float(value) for key, value in result.results_dict.items()}
    scores["per_class"] = [
        dict(zip(("precision", "recall", "ap50", "ap50_95"), map(float, result.box.class_result(i))))
        | {"class_id": int(class_id), "class_name": model.names[int(class_id)]}
        for i, class_id in enumerate(result.box.ap_class_index)
    ]
    return scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("train", "evaluate", "predict"))
    p.add_argument("--weights", default=BASELINES[0])
    p.add_argument("--data", type=Path)
    p.add_argument("--source")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--freeze", type=int, default=0)
    p.add_argument("--resume", type=Path)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--confidence", type=float, default=.25)
    p.add_argument("--split", choices=("val", "test"), default="test")
    a = p.parse_args()
    if a.resume and a.action != "train":
        p.error("--resume is for training only; use --weights for evaluation or prediction")
    if a.action != "train" and not Path(a.weights).is_file():
        p.error("Supply a local, trusted road-crack checkpoint via --weights")
    if a.action != "predict" and not a.data:
        p.error("--data must point to a prepared dataset directory")
    if a.action == "predict" and not a.source:
        p.error("--source is required for prediction")
    from ultralytics import YOLO
    import torch

    data = None
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    if a.data:
        data = a.data / "dataset.yaml"
        if Path(yaml.safe_load(data.read_text())["path"]).resolve() != a.data.resolve():
            p.error("Update dataset.yaml path after moving a prepared dataset")
        config["dataset_version"] = json.loads((a.data / "report.json").read_text())["dataset_version"]
    a.run = a.run.resolve()
    a.run.mkdir(parents=True, exist_ok=bool(a.resume))
    metadata = a.run / "configuration.json"
    if a.resume:
        old = json.loads(metadata.read_text())
        if old["dataset_version"] != config["dataset_version"] or a.resume.resolve().parent != a.run / "fit/weights":
            p.error("Resume in the original run directory with the same prepared dataset")
    model = YOLO(str(a.resume or a.weights))
    checkpoint = Path(a.resume or a.weights)
    if not checkpoint.is_file():
        checkpoint = Path(model.ckpt_path)
    config["checkpoint_sha256"] = checksum(checkpoint)
    config["versions"] = {name: version(name) for name in ("torch", "ultralytics", "numpy")}
    config["hardware"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if a.device != "cpu" else ["CPU"]
    if not a.resume:
        metadata.write_text(json.dumps(config, indent=2))
    if a.action == "train":
        if a.resume:
            model.train(resume=True, device=a.device)
        else:
            model.train(data=str(data.resolve()), project=str(a.run), name="fit",
                        imgsz=a.size, batch=a.batch, workers=a.workers, epochs=a.epochs, patience=a.patience,
                        device=a.device, seed=a.seed, deterministic=True, optimizer="AdamW",
                        lr0=a.lr, cos_lr=True, freeze=a.freeze, save_period=1, plots=True,
                        amp=a.device != "cpu" and torch.cuda.is_available(), degrees=0, shear=0, perspective=0,
                        flipud=0, fliplr=.5 if a.augment else 0, mosaic=0, mixup=0,
                        scale=.1 if a.augment else 0, translate=.05 if a.augment else 0,
                        hsv_h=0, hsv_s=.2 if a.augment else 0, hsv_v=.2 if a.augment else 0)
    elif a.action == "evaluate":
        scores = detector_metrics(model, data.resolve(), a, a.run / "plots")
        scores["checkpoint_sha256"] = config["checkpoint_sha256"]
        scores["split"] = a.split
        scores["dataset_version"] = config["dataset_version"]
        (a.run / "metrics.json").write_text(json.dumps(scores, indent=2))
    else:
        for _ in model.predict(a.source, imgsz=a.size, conf=a.confidence, device=a.device,
                               save=True, save_txt=True, save_conf=True, stream=True,
                               project=str(a.run), name="predictions"):
            pass


if __name__ == "__main__":
    main()
