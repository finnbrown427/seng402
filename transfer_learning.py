"""Transfer learning from synthetic target detection to medical segmentation."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_cnn import SpatialTargetCNN, WatermarkCNN


@dataclass(frozen=True)
class ImageMaskPair:
    image_path: Path
    mask_path: Path
    source_id: str


@dataclass(frozen=True)
class PatchRecord:
    pair_index: int
    source_id: str
    y: int
    x: int
    y_end: int
    x_end: int
    positive: bool


def discover_image_mask_pairs(
    root: str | Path,
    image_patterns: Sequence[str] = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"),
    mask_tokens: Sequence[str] = ("_mask", "-mask", "mask_", "_label", "-label"),
) -> list[ImageMaskPair]:
    """Find image/mask pairs using configurable, conservative filename matching."""
    root_path = Path(root)
    files = sorted(path for pattern in image_patterns for path in root_path.rglob(pattern))
    masks_by_stem = {path.stem.lower(): path for path in files if any(token in path.stem.lower() for token in mask_tokens)}
    pairs = []
    for image_path in files:
        stem = image_path.stem.lower()
        if stem in masks_by_stem:
            continue
        candidates = []
        for token in mask_tokens:
            candidates.extend(
                [
                    masks_by_stem.get(f"{stem}{token}"),
                    masks_by_stem.get(f"{token.lstrip('_-')}{stem}"),
                ]
            )
        mask_path = next((candidate for candidate in candidates if candidate is not None), None)
        if mask_path is not None:
            pairs.append(ImageMaskPair(image_path, mask_path, image_path.stem))
    if not pairs:
        raise FileNotFoundError(
            f"No image/mask pairs found under {root_path}. "
            "Pass explicit pairs or configure image_patterns/mask_tokens for the dataset layout."
        )
    return pairs


def discover_breast_cancer_pairs(root: str | Path) -> list[ImageMaskPair]:
    """Discover the downloaded breast-cancer dataset's Images/Masks layout."""
    root_path = Path(root)
    image_dir = root_path / "Images"
    mask_dir = root_path / "Masks"
    images = sorted(image_dir.glob("*.tif"))
    masks = {path.stem.lower(): path for path in mask_dir.glob("*.TIF")}
    pairs = []
    for image_path in images:
        image_key = image_path.stem.removesuffix("_ccd").lower()
        mask_path = masks.get(image_key)
        if mask_path is None:
            raise FileNotFoundError(f"No mask found for {image_path.name}")
        pairs.append(ImageMaskPair(image_path, mask_path, image_path.stem))
    if not pairs:
        raise FileNotFoundError(f"No .tif images found in {image_dir}")
    return pairs


def split_pairs(
    pairs: Sequence[ImageMaskPair],
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, list[ImageMaskPair]]:
    """Split original images before patches are extracted to prevent leakage."""
    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction and test_fraction must be non-negative and sum to less than one")
    unique_ids = list(dict.fromkeys(pair.source_id for pair in pairs))
    random.Random(seed).shuffle(unique_ids)
    test_count = max(1, round(len(unique_ids) * test_fraction)) if test_fraction else 0
    validation_count = max(1, round(len(unique_ids) * validation_fraction)) if validation_fraction else 0
    test_ids = set(unique_ids[:test_count])
    validation_ids = set(unique_ids[test_count:test_count + validation_count])
    train_ids = set(unique_ids) - test_ids - validation_ids
    if not train_ids and len(unique_ids) > 2:
        raise ValueError("The split left no training images")
    result = {
        "train": [pair for pair in pairs if pair.source_id in train_ids],
        "validation": [pair for pair in pairs if pair.source_id in validation_ids],
        "test": [pair for pair in pairs if pair.source_id in test_ids],
    }
    assert not (set(pair.source_id for pair in result["train"]) & set(pair.source_id for pair in result["validation"]))
    assert not (set(pair.source_id for pair in result["train"]) & set(pair.source_id for pair in result["test"]))
    assert not (set(pair.source_id for pair in result["validation"]) & set(pair.source_id for pair in result["test"]))
    return result


def _positions(height: int, width: int, patch_size: int, overlap: int) -> Iterable[tuple[int, int, int, int]]:
    if patch_size <= 0 or overlap < 0 or overlap >= patch_size:
        raise ValueError("patch_size must be positive and overlap must be smaller than patch_size")
    step = patch_size - overlap
    for y in range(0, height, step):
        for x in range(0, width, step):
            yield y, x, min(y + patch_size, height), min(x + patch_size, width)


def crop_and_pad(array: np.ndarray, y: int, x: int, y_end: int, x_end: int, patch_size: int) -> np.ndarray:
    crop = array[y:y_end, x:x_end]
    padding = ((0, patch_size - crop.shape[0]), (0, patch_size - crop.shape[1]))
    return np.pad(crop, padding, mode="constant")


def build_patch_records(
    pairs: Sequence[ImageMaskPair],
    patch_size: int,
    overlap: int,
    positive_threshold: int = 1,
    negative_ratio: float | None = None,
    seed: int = 42,
) -> list[PatchRecord]:
    """Create patch metadata after splitting; optionally subsample only negatives."""
    records = []
    for pair_index, pair in enumerate(pairs):
        image = cv2.imread(str(pair.image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(pair.mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise ValueError(f"Could not read image/mask pair: {pair.image_path}, {pair.mask_path}")
        if image.shape != mask.shape:
            raise ValueError(f"Image and mask shapes differ for {pair.source_id}: {image.shape} vs {mask.shape}")
        for y, x, y_end, x_end in _positions(*image.shape, patch_size, overlap):
            mask_patch = mask[y:y_end, x:x_end]
            records.append(PatchRecord(pair_index, pair.source_id, y, x, y_end, x_end, bool(np.count_nonzero(mask_patch) >= positive_threshold)))
    if negative_ratio is None:
        return records
    if negative_ratio < 0:
        raise ValueError("negative_ratio must be non-negative")
    positives = [record for record in records if record.positive]
    negatives = [record for record in records if not record.positive]
    random.Random(seed).shuffle(negatives)
    negative_count = min(len(negatives), round(len(positives) * negative_ratio))
    return positives + negatives[:negative_count]


def _read_pair(pair: ImageMaskPair) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(pair.image_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(pair.mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None or image.shape != mask.shape:
        raise ValueError(f"Invalid image/mask pair: {pair.image_path}, {pair.mask_path}")
    return image, (mask > 0).astype(np.float32)


class MedicalPatchDataset(Dataset):
    def __init__(self, pairs: Sequence[ImageMaskPair], records: Sequence[PatchRecord], patch_size: int):
        self.pairs = list(pairs)
        self.records = list(records)
        self.patch_size = patch_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image, mask = _read_pair(self.pairs[record.pair_index])
        image_patch = crop_and_pad(image, record.y, record.x, record.y_end, record.x_end, self.patch_size)
        mask_patch = crop_and_pad(mask, record.y, record.x, record.y_end, record.x_end, self.patch_size)
        return (
            torch.from_numpy(image_patch[None].astype(np.float32) / 255.0),
            torch.from_numpy(mask_patch[None].astype(np.float32)),
        )


def load_synthetic_encoder(model: SpatialTargetCNN, checkpoint: str | Path | dict) -> tuple[list[str], list[str]]:
    """Load only encoder weights from a WatermarkCNN checkpoint."""
    state = torch.load(checkpoint, map_location="cpu") if isinstance(checkpoint, (str, Path)) else checkpoint
    state = state.get("model_state_dict", state.get("state_dict", state))
    encoder_state = {key: value for key, value in state.items() if key.startswith("features.")}
    if not encoder_state:
        raise ValueError("Checkpoint contains no features.* weights")
    result = model.load_state_dict(encoder_state, strict=False)
    unexpected = [key for key in result.unexpected_keys if key.startswith("features.")]
    missing = [key for key in result.missing_keys if key.startswith("features.")]
    if unexpected or missing:
        raise ValueError(f"Encoder checkpoint mismatch; missing={missing}, unexpected={unexpected}")
    return missing, unexpected


def set_encoder_trainable(model: SpatialTargetCNN, trainable: bool) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = trainable


def segmentation_metrics(logits: torch.Tensor, masks: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    predictions = (torch.sigmoid(logits) >= threshold)
    targets = masks >= 0.5
    tp = (predictions & targets).sum().item()
    fp = (predictions & ~targets).sum().item()
    fn = (~predictions & targets).sum().item()
    tn = (~predictions & ~targets).sum().item()
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall + eps),
        "accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
    }


def _run_epoch(model, loader, optimizer=None, device="cpu"):
    training = optimizer is not None
    model.train(training)
    if training and not any(parameter.requires_grad for parameter in model.features.parameters()):
        # BatchNorm buffers also need to remain fixed during the frozen stage.
        model.features.eval()
    criterion = torch.nn.BCEWithLogitsLoss()
    totals = {"loss": 0.0, "count": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            predictions = torch.sigmoid(logits) >= 0.5
            targets = masks >= 0.5
            totals["loss"] += loss.item() * images.size(0)
            totals["count"] += images.size(0)
            totals["tp"] += (predictions & targets).sum().item()
            totals["fp"] += (predictions & ~targets).sum().item()
            totals["fn"] += (~predictions & targets).sum().item()
            totals["tn"] += (~predictions & ~targets).sum().item()
    eps = 1e-8
    precision = totals["tp"] / (totals["tp"] + totals["fp"] + eps)
    recall = totals["tp"] / (totals["tp"] + totals["fn"] + eps)
    return {
        "loss": totals["loss"] / max(totals["count"], 1),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall + eps),
        "accuracy": (totals["tp"] + totals["tn"]) / max(sum(totals[key] for key in ("tp", "fp", "fn", "tn")), 1),
    }


def train_stage(model, train_loader, validation_loader, epochs, learning_rate, device="cpu"):
    optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=learning_rate)
    best_state = copy.deepcopy(model.state_dict())
    best_metrics = None
    for _ in range(epochs):
        _run_epoch(model, train_loader, optimizer, device)
        metrics = _run_epoch(model, validation_loader, None, device)
        if best_metrics is None or metrics["f1"] > best_metrics["f1"]:
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best_metrics


def evaluate_segmentation(model, loader, device="cpu") -> dict[str, float]:
    """Evaluate a segmentation model without updating its parameters."""
    return _run_epoch(model, loader, None, device)


def predict_full_image(model, image: np.ndarray, patch_size: int, overlap: int, device="cpu") -> np.ndarray:
    """Average overlapping patch probabilities back into one image-sized map."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape[:2]
    patches = []
    positions = []
    for y, x, y_end, x_end in _positions(height, width, patch_size, overlap):
        patch = crop_and_pad(image, y, x, y_end, x_end, patch_size)
        patches.append(patch.astype(np.float32) / 255.0)
        positions.append((y, x, y_end, x_end))
    batch = torch.from_numpy(np.stack(patches)[:, None]).to(device)
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(batch)).cpu().numpy()[:, 0]
    heat = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros_like(heat)
    for probability, (y, x, y_end, x_end) in zip(probabilities, positions):
        crop_height, crop_width = y_end - y, x_end - x
        heat[y:y_end, x:x_end] += probability[:crop_height, :crop_width]
        counts[y:y_end, x:x_end] += 1.0
    return np.divide(heat, counts, out=np.zeros_like(heat), where=counts > 0)


def save_prediction_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    prediction: np.ndarray,
    output_path: str | Path,
    threshold: float = 0.5,
) -> None:
    """Save a compact qualitative image with truth and predicted locations."""
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    overlay = base.copy()
    overlay[mask > 0] = (0, 255, 0)
    overlay[prediction >= threshold] = (0, 0, 255)
    cv2.imwrite(str(output_path), cv2.addWeighted(base, 0.65, overlay, 0.35, 0.0))


def build_loaders(splits, patch_size, overlap, batch_size=16, negative_ratio=None, seed=42):
    datasets = {}
    for split_name, pairs in splits.items():
        records = build_patch_records(pairs, patch_size, overlap, negative_ratio=negative_ratio, seed=seed)
        datasets[split_name] = MedicalPatchDataset(pairs, records, patch_size)
    return {
        split_name: DataLoader(dataset, batch_size=batch_size, shuffle=split_name == "train", num_workers=0)
        for split_name, dataset in datasets.items()
    }


def run_transfer_experiment(
    synthetic_checkpoint: str | Path | dict,
    loaders: dict[str, DataLoader],
    head_epochs: int = 5,
    fine_tune_epochs: int = 5,
    head_learning_rate: float = 1e-3,
    fine_tune_learning_rate: float = 1e-4,
    device: str | torch.device = "cpu",
):
    """Run frozen-head and fine-tuned stages from the same synthetic encoder."""
    model = SpatialTargetCNN().to(device)
    load_synthetic_encoder(model, synthetic_checkpoint)
    set_encoder_trainable(model, False)
    stage1_model, stage1_metrics = train_stage(
        model, loaders["train"], loaders["validation"], head_epochs, head_learning_rate, device
    )
    frozen_head_model = copy.deepcopy(stage1_model)
    set_encoder_trainable(stage1_model, True)
    stage2_model, stage2_metrics = train_stage(
        stage1_model, loaders["train"], loaders["validation"], fine_tune_epochs, fine_tune_learning_rate, device
    )
    test_metrics = _run_epoch(stage2_model, loaders["test"], None, device)
    return {"frozen_head": frozen_head_model, "fine_tuned": stage2_model, "stage1": stage1_metrics, "stage2": stage2_metrics, "test": test_metrics}


def evaluate_synthetic_classifier(model: WatermarkCNN, loader: DataLoader, device="cpu") -> dict[str, float]:
    """Evaluate baseline A using patch occupancy as its binary target."""
    model.eval()
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for images, masks in loader:
            labels = (masks.flatten(1).sum(dim=1) > 0).float()
            predictions = (torch.sigmoid(model(images.to(device)).flatten()) >= 0.5).float().cpu()
            tp += ((predictions == 1) & (labels == 1)).sum().item()
            fp += ((predictions == 1) & (labels == 0)).sum().item()
            fn += ((predictions == 0) & (labels == 1)).sum().item()
            tn += ((predictions == 0) & (labels == 0)).sum().item()
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall + eps), "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1)}

def run_medical_from_config(config) -> dict[str, dict[str, float]]:
    """Run the medical experiment from the shared project configuration."""
    settings = config["medical"]
    pairs = discover_breast_cancer_pairs(settings["dataset"])
    splits = split_pairs(
        pairs,
        validation_fraction=settings["validation_fraction"],
        test_fraction=settings["test_fraction"],
        seed=config["training"]["seed"],
    )
    loaders = build_loaders(
        splits,
        patch_size=settings["patch_size"],
        overlap=settings["overlap"],
        batch_size=settings["batch_size"],
        negative_ratio=settings["negative_ratio"],
        seed=config["training"]["seed"],
    )
    device = torch.device(settings["device"])
    experiment = run_transfer_experiment(
        config["outputs"]["synthetic_checkpoint"],
        loaders,
        head_epochs=settings["head_epochs"],
        fine_tune_epochs=settings["fine_tune_epochs"],
        head_learning_rate=settings["head_learning_rate"],
        fine_tune_learning_rate=settings["fine_tune_learning_rate"],
        device=device,
    )
    baseline = WatermarkCNN().to(device)
    baseline.load_state_dict(torch.load(config["outputs"]["synthetic_checkpoint"], map_location=device))
    metrics = {
        "A_synthetic_classifier": evaluate_synthetic_classifier(baseline, loaders["test"], device),
        "B_frozen_encoder": evaluate_segmentation(experiment["frozen_head"], loaders["test"], device),
        "C_fine_tuned": evaluate_segmentation(experiment["fine_tuned"], loaders["test"], device),
    }
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(experiment["frozen_head"].state_dict(), output_dir / "frozen_head.pt")
    torch.save(experiment["fine_tuned"].state_dict(), output_dir / "fine_tuned.pt")
    test_pair = splits["test"][0]
    image = cv2.imread(str(test_pair.image_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(test_pair.mask_path), cv2.IMREAD_GRAYSCALE)
    prediction = predict_full_image(experiment["fine_tuned"], image, settings["patch_size"], settings["overlap"], device)
    save_prediction_overlay(image, mask, prediction, output_dir / "fine_tuned_overlay.png")
    print("Medical transfer metrics:")
    for name, values in metrics.items():
        print(name, values)
    return metrics
