#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import (
    LRASPP_MobileNet_V3_Large_Weights,
    lraspp_mobilenet_v3_large,
)
from torchvision.models.segmentation.lraspp import LRASPPHead
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contracts/phase5/phase5c3_contract.json"
CHANNELS = (
    "traversable_floor",
    "floor_marking",
    "curb_or_step",
    "wall_or_door",
    "fence_or_guardrail",
    "shelf_or_rack",
    "pole_or_column",
    "traffic_control_or_sign",
    "pallet_or_load",
    "box_or_small_obstacle",
    "person",
    "robot_or_cart",
    "forklift_or_heavy_vehicle",
    "unknown_or_unlabeled",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_path(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


class WarehouseDataset(Dataset):
    def __init__(self, root, rows, training):
        self.root = root
        self.rows = rows
        self.training = training

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = cv2.imread(str(self.root / row["image"]), cv2.IMREAD_COLOR)
        label = cv2.imread(str(self.root / row["label"]), cv2.IMREAD_GRAYSCALE)
        if image is None or label is None:
            raise RuntimeError(f"failed to load dataset frame: {row}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.training and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            label = np.ascontiguousarray(label[:, ::-1])
        image = torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)
        if self.training:
            image = TF.adjust_brightness(image, random.uniform(0.85, 1.15))
            image = TF.adjust_contrast(image, random.uniform(0.85, 1.15))
            image = TF.adjust_saturation(image, random.uniform(0.85, 1.15))
        image = TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return image, torch.from_numpy(label.astype(np.int64))


class LogitsOnly(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model(image)["out"]


def confusion_metrics(confusion):
    confusion = confusion.astype(np.float64)
    true_positive = np.diag(confusion)
    union = confusion.sum(axis=1) + confusion.sum(axis=0) - true_positive
    iou = np.divide(true_positive, union, out=np.full_like(true_positive, np.nan), where=union > 0)
    present = np.isfinite(iou)
    return {
        "mean_iou": float(np.nanmean(iou)),
        "pixel_accuracy": float(true_positive.sum() / max(confusion.sum(), 1.0)),
        "per_class_iou": {
            CHANNELS[index]: (float(value) if math.isfinite(value) else None)
            for index, value in enumerate(iou)
        },
        "present_classes": int(present.sum()),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    confusion = np.zeros((len(CHANNELS), len(CHANNELS)), dtype=np.int64)
    for images, labels in loader:
        logits = model(images.to(device, non_blocking=True))["out"]
        predictions = logits.argmax(dim=1).cpu().numpy()
        truth = labels.numpy()
        encoded = truth.ravel() * len(CHANNELS) + predictions.ravel()
        confusion += np.bincount(encoded, minlength=len(CHANNELS) ** 2).reshape(
            len(CHANNELS), len(CHANNELS)
        )
    return confusion_metrics(confusion)


def onnx_check(model, onnx_path, sample, device):
    wrapper = LogitsOnly(model).eval().to(device)
    with torch.no_grad():
        expected = wrapper(sample.to(device)).cpu().numpy()
    torch.onnx.export(
        wrapper,
        sample.to(device),
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
        dynamo=False,
    )
    session = ort.InferenceSession(
        str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    actual = session.run(["output"], {"input": sample.numpy()})[0]
    max_abs_error = float(np.max(np.abs(expected - actual)))
    for _ in range(10):
        session.run(["output"], {"input": sample.numpy()})
    timings = []
    for _ in range(100):
        started = time.perf_counter_ns()
        session.run(["output"], {"input": sample.numpy()})
        timings.append((time.perf_counter_ns() - started) / 1e6)
    return {
        "providers": session.get_providers(),
        "max_abs_logit_error": max_abs_error,
        "latency_ms": {
            "mean": float(np.mean(timings)),
            "p95": float(np.percentile(timings, 95)),
            "max": float(np.max(timings)),
        },
        "input_shape": list(session.get_inputs()[0].shape),
        "output_shape": list(session.get_outputs()[0].shape),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-validation", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-deploy", action="store_true")
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    dataset_summary = json.loads((dataset / "summary.json").read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if dataset_summary["status"] != "complete":
        raise SystemExit("Phase 5-C3 dataset does not cover all 14 classes")
    if dataset_summary["contract_sha256"] != sha256(CONTRACT):
        raise SystemExit("Phase 5-C3 dataset was generated from a different contract")
    epochs = args.epochs or contract["training"]["epochs"]
    batch_size = args.batch_size or contract["training"]["batch_size"]
    if epochs < 1 or batch_size < 1:
        raise SystemExit("epochs and batch size must be positive")
    output = args.output or ROOT / "artifacts/phase5c3_training" / time.strftime(
        "%Y%m%d_%H%M%S"
    )
    output.mkdir(parents=True, exist_ok=False)

    with (dataset / "frames.csv").open(newline="", encoding="ascii") as source:
        rows = list(csv.DictReader(source))
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    if args.max_train:
        train_rows = train_rows[: args.max_train]
    if args.max_validation:
        validation_rows = validation_rows[: args.max_validation]
    torch.manual_seed(contract["dataset"]["seed"])
    random.seed(contract["dataset"]["seed"])
    np.random.seed(contract["dataset"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        WarehouseDataset(dataset, train_rows, training=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
    )
    validation_loader = DataLoader(
        WarehouseDataset(dataset, validation_rows, training=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
    )

    weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
    model = lraspp_mobilenet_v3_large(weights=weights)
    model.classifier = LRASPPHead(40, 960, len(CHANNELS), inter_channels=128)
    model.to(device)
    pixel_counts = np.asarray(
        [dataset_summary["splits"]["train"]["class_pixels"][name] for name in CHANNELS],
        dtype=np.float64,
    )
    class_weights = np.sqrt(pixel_counts.sum() / np.maximum(pixel_counts, 1.0))
    class_weights /= class_weights.mean()
    class_weights = np.clip(class_weights, 0.25, 8.0)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=contract["training"]["learning_rate"],
        weight_decay=contract["training"]["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    best_miou = -1.0
    best_path = output / "warehouse_nav14_best.pth"
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)["out"]
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        validation = evaluate(model, validation_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation": validation,
        }
        history.append(record)
        print(
            f"Epoch {epoch:02d}/{epochs} loss={record['train_loss']:.4f} "
            f"val_mIoU={validation['mean_iou']:.4f} val_acc={validation['pixel_accuracy']:.4f}"
        )
        if validation["mean_iou"] > best_miou:
            best_miou = validation["mean_iou"]
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    sample, _ = next(iter(validation_loader))
    sample = sample[:1].cpu()
    onnx_path = output / "warehouse_nav14_candidate.onnx"
    onnx_validation = onnx_check(model, onnx_path, sample, device)
    deployed_model = ROOT / "model/warehouse_nav14_candidate.onnx"
    deployed_manifest = ROOT / "model/warehouse_nav14_candidate.json"
    deploy_allowed = not args.no_deploy and not args.max_train and not args.max_validation
    if deploy_allowed:
        shutil.copy2(onnx_path, deployed_model)
    summary = {
        "schema_version": "phase5c3-training-v1",
        "status": "candidate_trained_shadow_only",
        "control_promotion_allowed": False,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "contract": str(CONTRACT.relative_to(ROOT)),
        "contract_sha256": sha256(CONTRACT),
        "dataset": project_path(dataset),
        "dataset_summary_sha256": sha256(dataset / "summary.json"),
        "train_frames": len(train_rows),
        "validation_frames": len(validation_rows),
        "epochs": epochs,
        "batch_size": batch_size,
        "class_weights": {
            name: float(class_weights[index]) for index, name in enumerate(CHANNELS)
        },
        "best_validation_miou": best_miou,
        "best_checkpoint": best_path.name,
        "best_checkpoint_sha256": sha256(best_path),
        "onnx": onnx_path.name,
        "onnx_sha256": sha256(onnx_path),
        "onnx_validation": onnx_validation,
        "history": history,
        "next_gate": "1000-frame Phase 5-C3 candidate shadow replay against frozen Phase 5-C2 teacher",
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    if deploy_allowed:
        deployed_manifest.write_text(
            json.dumps(
                {
                    "taxonomy_id": "warehouse_nav14_v1",
                    "model": str(deployed_model.relative_to(ROOT)),
                    "model_sha256": sha256(deployed_model),
                    "training_summary": str(summary_path.relative_to(ROOT)),
                    "training_summary_sha256": sha256(summary_path),
                    "control_promotion_allowed": False,
                },
                indent=2,
                ensure_ascii=True,
            )
            + "\n",
            encoding="ascii",
        )
    print(json.dumps({"best_validation_miou": best_miou, "onnx": onnx_validation}, indent=2))
    print(f"Phase 5-C3 training artifacts: {output}")


if __name__ == "__main__":
    main()
