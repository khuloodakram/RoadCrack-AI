"""Binary segmentation training, evaluation, and original-resolution inference."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

ARCHITECTURES = ("UnetPlusPlus", "DeepLabV3Plus")


def build_model(architecture, encoder, pretrained=False):
    import segmentation_models_pytorch as smp

    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {architecture}")
    return getattr(smp, architecture)(encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None, in_channels=3, classes=1)


def tensor(image):
    pixels = np.asarray(image, dtype=np.float32) / 255
    pixels = (pixels - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
    return torch.from_numpy(pixels.transpose(2, 0, 1).copy())


class CrackMasks(Dataset):
    def __init__(self, root, split, size, augment=False):
        self.images = sorted((root / "images" / split).glob("*"))
        if not self.images:
            raise ValueError(f"No images in split {split}")
        self.masks, self.size, self.augment = root / "masks" / split, size, augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        with Image.open(path) as image, Image.open(self.masks / f"{path.stem}.png") as mask:
            x = tensor(image.convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR))
            y = torch.from_numpy((np.asarray(mask.convert("L").resize(
                (self.size, self.size), Image.Resampling.NEAREST)) > 127).astype(np.float32))[None]
        if self.augment and random.random() < 0.5:
            x, y = x.flip(-1), y.flip(-1)
        return x, y


def segmentation_loss(logits, target, loss="bce-dice"):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    if loss == "bce":
        return bce
    probs = logits.sigmoid()
    dims = (1, 2, 3)
    dice = (2 * (probs * target).sum(dims) + 1) / (probs.sum(dims) + target.sum(dims) + 1)
    return bce + (1 - dice).mean()


def confusion(pred, truth):
    return np.array([(pred & truth).sum().item(), (pred & ~truth).sum().item(),
                     (~pred & truth).sum().item()], dtype=np.int64)


def metrics(counts):
    tp, fp, fn = map(int, counts)
    def ratio(n, d):
        return n / d if d else None
    return dict(dice=ratio(2*tp, 2*tp+fp+fn), iou=ratio(tp, tp+fp+fn),
                precision=ratio(tp, tp+fp), recall=ratio(tp, tp+fn))


@torch.inference_mode()
def evaluate(model, loader, device, thresholds, loss):
    model.eval()
    counts = {t: np.zeros(3, dtype=np.int64) for t in thresholds}
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += segmentation_loss(logits, y, loss).item() * len(x)
        p = logits.sigmoid()
        for threshold in thresholds:
            counts[threshold] += confusion(p >= threshold, y.bool())
    return total_loss / len(loader.dataset), {t: metrics(c) for t, c in counts.items()}


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = build_model(config["architecture"], config["encoder"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


@torch.inference_mode()
def tiled_probabilities(model, image, tile, overlap, device):
    if tile < 32 or tile % 32 or not 0 <= overlap < tile:
        raise ValueError("Tile must be a multiple of 32 and overlap must be smaller")
    width, height = image.size
    sums = np.zeros((height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    def starts(length):
        last = max(0, length - tile)
        return sorted(set(range(0, last, tile - overlap)) | {last})
    window = np.maximum(np.outer(np.hanning(tile), np.hanning(tile)), 0.01).astype(np.float32)
    for top in starts(height):
        for left in starts(width):
            right, bottom = min(left + tile, width), min(top + tile, height)
            patch = Image.new("RGB", (tile, tile))
            patch.paste(image.crop((left, top, right, bottom)))
            prob = model(tensor(patch)[None].to(device)).sigmoid()[0, 0].cpu().numpy()
            h, w = bottom - top, right - left
            sums[top:bottom, left:right] += prob[:h, :w] * window[:h, :w]
            weights[top:bottom, left:right] += window[:h, :w]
    return sums / weights


def train(args):
    from importlib.metadata import version

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device)
    amp = device.type == "cuda"
    if args.batch < 2 or args.size < 32 or args.size % 32:
        raise ValueError("Use batch >= 2 and a size divisible by 32")
    generator = torch.Generator().manual_seed(args.seed)
    data = CrackMasks(args.data, "train", args.size, args.flip)
    if len(data) < args.batch:
        raise ValueError("The train split is smaller than the requested batch")
    train_loader = DataLoader(data, args.batch, shuffle=True, drop_last=len(data) % args.batch == 1, generator=generator)
    val_loader = DataLoader(CrackMasks(args.data, "val", args.size), args.batch)
    resume = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    config = vars(args).copy()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    config["dataset_version"] = json.loads((args.data / "report.json").read_text())["dataset_version"]
    config["versions"] = {name: version(name) for name in ("torch", "segmentation-models-pytorch", "numpy")}
    config["hardware"] = torch.cuda.get_device_name(device) if amp else "CPU"
    keys = ("architecture", "encoder", "batch", "size", "seed", "lr", "epochs", "loss", "flip", "dataset_version")
    if resume and any(config[key] != resume["config"][key] for key in keys):
        raise ValueError("Resume with the same training configuration and dataset")
    args.run.mkdir(parents=True, exist_ok=bool(resume))
    model = build_model(args.architecture, args.encoder, args.pretrained and not resume).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, best, stale, history = 0, -1, 0, []
    if resume:
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        generator.set_state(resume["generator"])
        torch.set_rng_state(resume["torch_rng"])
        random.setstate(resume["python_rng"])
        if amp and resume["cuda_rng"]:
            torch.cuda.set_rng_state_all(resume["cuda_rng"])
        start, best, stale, history = resume["epoch"] + 1, resume["best"], resume["stale"], resume["history"]
    for epoch in range(start, args.epochs):
        began = time.perf_counter()
        model.train()
        total, seen = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=amp):
                loss = segmentation_loss(model(x), y, args.loss)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(x)
            seen += len(x)
        val_loss, scores = evaluate(model, val_loader, device, (.25, .35, .5, .65, .75), args.loss)
        threshold = max(scores, key=lambda t: scores[t]["dice"] if scores[t]["dice"] is not None else -1)
        score = scores[threshold]["dice"]
        improved = epoch == 0 or (score is not None and score > best)
        if improved:
            best, stale = score if score is not None else -1, 0
        else:
            stale += 1
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": total / seen, "val_loss": val_loss,
                        "threshold": threshold, "scores": scores, "seconds": time.perf_counter() - began})
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     scaler=scaler.state_dict(), generator=generator.get_state(), torch_rng=torch.get_rng_state(),
                     python_rng=random.getstate(), cuda_rng=torch.cuda.get_rng_state_all() if amp else [],
                     config=config, best=best, stale=stale, history=history, epoch=epoch, threshold=threshold)
        for name in (("last", "best") if improved else ("last",)):
            temp = args.run / f"{name}.tmp"
            torch.save(state, temp)
            temp.replace(args.run / f"{name}.pt")
        (args.run / "history.json").write_text(json.dumps(history, indent=2))
        if stale >= args.patience:
            break


def test_checkpoint(args):
    device = torch.device(args.device)
    model, checkpoint = load_model(args.resume, device)
    threshold = checkpoint["threshold"] if args.threshold is None else args.threshold
    thresholds = (.25, .35, .5, .65, .75) if args.calibrate else (threshold,)
    counts = {t: np.zeros(3, dtype=np.int64) for t in thresholds}
    thin_hits = {t: 0 for t in thresholds}
    thin_total = 0
    import cv2
    examples = args.run / "examples"
    examples.mkdir(parents=True, exist_ok=True)
    paths = sorted((args.data / "images" / args.split).glob("*"))
    if not paths:
        raise ValueError(f"The {args.split} split is empty")
    for index, path in enumerate(paths):
        with Image.open(path) as opened, Image.open(args.data / "masks" / args.split / f"{path.stem}.png") as mask:
            image = opened.convert("RGB")
            truth = np.asarray(mask.convert("L")) > 127
        probabilities = tiled_probabilities(model, image, args.size, args.overlap, device)
        distances = cv2.distanceTransform(np.pad(truth.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        thin = truth & (distances <= 2)
        thin_total += int(thin.sum())
        truth_tensor = torch.from_numpy(truth)
        for value in thresholds:
            predicted = probabilities >= value
            counts[value] += confusion(torch.from_numpy(predicted), truth_tensor)
            thin_hits[value] += int((thin & predicted).sum())
        if index < 8 and not args.calibrate:
            pred = probabilities >= threshold
            overlay = np.asarray(image).copy()
            overlay[truth & ~pred] = (40, 150, 250)
            overlay[pred & ~truth] = (255, 80, 40)
            overlay[pred & truth] = (34, 200, 120)
            Image.fromarray(overlay).save(examples / f"{path.stem}.png")
    scores = {t: metrics(count) for t, count in counts.items()}
    if args.calibrate:
        threshold = max(scores, key=lambda t: scores[t]["dice"] if scores[t]["dice"] is not None else -1)
    result = dict(scores[threshold], threshold=threshold, tile=args.size, overlap=args.overlap, split=args.split,
                  boundary_band_recall=thin_hits[threshold] / thin_total if thin_total else None,
                  dataset_version=json.loads((args.data / "report.json").read_text())["dataset_version"])
    if args.calibrate:
        result["candidates"] = scores
    (args.run / f"{args.split}.json").write_text(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("train", "evaluate"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--architecture", choices=ARCHITECTURES, default="UnetPlusPlus")
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--overlap", type=int, default=96)
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--calibrate", action="store_true", help="Select tiled threshold on validation only")
    p.add_argument("--threshold", type=float, help="Fixed threshold chosen on validation")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr", type=float, default=.0003)
    p.add_argument("--loss", choices=("bce-dice", "bce"), default="bce-dice")
    p.add_argument("--flip", action="store_true")
    args = p.parse_args()
    if args.action == "evaluate" and not args.resume:
        p.error("--resume must point to the selected checkpoint for evaluation")
    if args.calibrate and (args.action != "evaluate" or args.split != "val" or args.threshold is not None):
        p.error("--calibrate requires evaluation on val and chooses its own threshold")
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        p.error("--threshold must be in [0, 1]")
    (train if args.action == "train" else test_checkpoint)(args)


if __name__ == "__main__":
    main()
