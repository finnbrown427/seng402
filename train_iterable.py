import csv
import copy
import random
import os
import numpy as np

import cv2
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader

from pytorch_cnn import WatermarkCNN


from image_generation import (create_noise, create_target, embed_targets, segment_image, validate_segment_config)


class TargetIterableDataset(IterableDataset):
    def __init__(self, num_samples=5000, image_size=1024, segment_size=224, overlap=64,
                 target_prob=0.95, max_targets=78, target_size=8, target_mode="bw",
                 block_size=1, positive_threshold=0.5, target_shape="square",
                 mix_mode="per_target", target_kwargs=None, seed=None):
        self.num_samples = num_samples
        self.image_size = image_size
        self.segment_size = segment_size
        self.overlap = overlap
        self.target_prob = target_prob
        self.max_targets = max_targets
        self.target_size = target_size
        self.target_mode = target_mode
        self.block_size = block_size
        self.positive_threshold = positive_threshold
        self.target_shape = target_shape
        self.mix_mode = mix_mode
        self.target_kwargs = dict(target_kwargs) if target_kwargs is not None else {}
        self.seed = seed

    def __iter__(self):
        random_state = random.getstate()
        numpy_state = np.random.get_state()
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        try:
            for _ in range(self.num_samples):
                base_target_kwargs = {
                    "size": self.target_size,
                    "mode": self.target_mode,
                    "block_size": self.block_size,
                    **self.target_kwargs,
                }
                background_noise = create_noise(self.image_size, self.image_size)

                if random.random() < self.target_prob:
                    num_targets = random.randint(1, self.max_targets)
                    full_image, mask = embed_targets(
                        background_noise,
                        num_targets,
                        target_kwargs=base_target_kwargs,
                        target_shape=self.target_shape,
                        mix_mode=self.mix_mode,
                    )
                else:
                    full_image = background_noise
                    mask = np.zeros_like(background_noise, dtype=np.uint8)

                segments, labels, _ = segment_image(
                    full_image, mask, self.segment_size, self.segment_size,
                    self.overlap, self.positive_threshold
                )

                for segment, label in zip(segments, labels):
                    segment = segment.astype(np.float32) / 255.0
                    segment_tensor = torch.from_numpy(segment).unsqueeze(0)
                    label_tensor = torch.tensor([float(label)], dtype=torch.float32)
                    yield segment_tensor, label_tensor
        finally:
            random.setstate(random_state)
            np.random.set_state(numpy_state)


def run_train(
    epochs=3,
    batch_size=32,
    lr=1e-3,
    train_samples=750,
    val_samples=100,
    image_size=516,
    segment_size=64,
    overlap=16,
    target_prob=0.5,
    target_size=8,
    train_target_size=None,
    val_target_size=None,
    decision_threshold=0.3,
    seed=67,
    return_metrics=False,
    train_target_shape="circle",
    val_target_shape=None,
    train_mix_mode=None,
    val_mix_mode=None,
    train_target_kwargs=None,
    val_target_kwargs=None,
    restore_best=True,
    validation_seed=None,
    device="auto",
):
    if torch is None or nn is None or WatermarkCNN is None:
        raise ImportError("PyTorch is required to run training.")

    validate_segment_config(segment_size, segment_size, overlap)

    if val_target_shape is None:
        val_target_shape = train_target_shape
    if val_mix_mode is None:
        val_mix_mode = train_mix_mode

    if train_target_size is None:
        train_target_size = target_size
    if val_target_size is None:
        val_target_size = train_target_size

    train_target_kwargs = dict(train_target_kwargs) if train_target_kwargs is not None else {}
    val_target_kwargs = dict(val_target_kwargs) if val_target_kwargs is not None else {}
    train_target_kwargs.setdefault("size", train_target_size)
    val_target_kwargs.setdefault("size", val_target_size)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'mps'.")
    device = torch.device(device)

    train_ds = TargetIterableDataset(
        num_samples=train_samples,
        image_size=image_size,
        segment_size=segment_size,
        overlap=overlap,
        target_prob=target_prob,
        target_shape=train_target_shape,
        mix_mode=train_mix_mode,
        target_kwargs=train_target_kwargs,
    )
    val_ds = TargetIterableDataset(
        num_samples=val_samples,
        image_size=image_size,
        segment_size=segment_size,
        overlap=overlap,
        target_prob=target_prob,
        target_shape=val_target_shape,
        mix_mode=val_mix_mode,
        target_kwargs=val_target_kwargs,
        seed=validation_seed if validation_seed is not None else (None if seed is None else seed + 1),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    val_batches = list(val_loader)

    model = WatermarkCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    last_metrics = None
    best_val_f1 = -1.0
    best_model_state = None
    best_metrics = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        tp = 0
        fp = 0
        fn = 0

        # train loop
        for segments, labels in train_loader:
            segments = segments.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(segments)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * segments.size(0)
            preds = (torch.sigmoid(logits) > decision_threshold).float()
            running_correct += (preds == labels).sum().item()
            running_total += labels.numel()

            labels_i = labels.int()
            preds_i = preds.int()
            tp += ((preds_i == 1) & (labels_i == 1)).sum().item()
            fp += ((preds_i == 1) & (labels_i == 0)).sum().item()
            fn += ((preds_i == 0) & (labels_i == 1)).sum().item()

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total
        eps = 1e-8
        train_precision = tp / (tp + fp + eps)
        train_recall = tp / (tp + fn + eps)
        train_f1 = 2 * train_precision * train_recall / (train_precision + train_recall + eps)

        # validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_tp = 0
        val_fp = 0
        val_fn = 0
        val_probabilities = []

        with torch.no_grad():
            for segments, labels in val_batches:
                segments = segments.to(device)
                labels = labels.to(device)
                logits = model(segments)
                loss = criterion(logits, labels)

                val_loss += loss.item() * segments.size(0)
                probabilities = torch.sigmoid(logits)
                val_probabilities.append(probabilities.detach().cpu())
                preds = (probabilities > decision_threshold).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.numel()

                labels_i = labels.int()
                preds_i = preds.int()
                val_tp += ((preds_i == 1) & (labels_i == 1)).sum().item()
                val_fp += ((preds_i == 1) & (labels_i == 0)).sum().item()
                val_fn += ((preds_i == 0) & (labels_i == 1)).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total
        val_precision = val_tp / (val_tp + val_fp + eps)
        val_recall = val_tp / (val_tp + val_fn + eps)
        val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall + eps)
        val_label_positive_rate = (val_tp + val_fn) / val_total
        val_predicted_positive_rate = (val_tp + val_fp) / val_total
        val_probabilities = torch.cat(val_probabilities)
        val_probability_min = float(val_probabilities.min())
        val_probability_max = float(val_probabilities.max())
        val_probability_mean = float(val_probabilities.mean())

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())

        last_metrics = {
            "segment_size": segment_size,
            "overlap": overlap,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_precision": train_precision,
            "train_recall": train_recall,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1,
            "val_label_positive_rate": val_label_positive_rate,
            "val_predicted_positive_rate": val_predicted_positive_rate,
            "val_probability_min": val_probability_min,
            "val_probability_max": val_probability_max,
            "val_probability_mean": val_probability_mean,
        }
        if is_best:
            best_metrics = copy.deepcopy(last_metrics)

        print(
            f"Epoch {epoch}: \n"
            f"training: loss={train_loss:.4f} acc={train_acc:.3f} prec={train_precision:.3f} recall={train_recall:.3f} F1={train_f1:.3f} \n"
            f"validation: loss={val_loss:.4f} acc={val_acc:.3f} prec={val_precision:.3f} recall={val_recall:.3f} F1={val_f1:.3f} "
            f"labels+={val_label_positive_rate:.3f} predicted+={val_predicted_positive_rate:.3f} "
            f"probability={val_probability_min:.3f}-{val_probability_max:.3f}"
        )

    if restore_best and best_model_state is not None:
        model.load_state_dict(best_model_state)
        last_metrics = best_metrics

    if return_metrics:
        return model, last_metrics
    return model


def build_sweep_configs(segment_sizes, overlaps):
    return [
        {"segment_size": segment_size, "overlap": overlap}
        for segment_size in segment_sizes
        for overlap in overlaps
    ]


def rank_sweep_results(results):
    return sorted(results, key=lambda item: item.get("val_f1", -1), reverse=True)


def build_target_transfer_configs(train_target_sizes, val_target_sizes):
    return [
        {"train_target_size": train_target_size, "val_target_size": val_target_size}
        for train_target_size in train_target_sizes
        for val_target_size in val_target_sizes
    ]


def summarize_target_transfer_results(results):
    summary_results = []
    unique_train_sizes = sorted({item["train_target_size"] for item in results})
    unique_val_sizes = sorted({item["val_target_size"] for item in results})

    for train_target_size in unique_train_sizes:
        for val_target_size in unique_val_sizes:
            matching = [
                item
                for item in results
                if item["train_target_size"] == train_target_size and item["val_target_size"] == val_target_size
            ]
            if not matching:
                continue

            summary_results.append(
                {
                    "train_target_size": train_target_size,
                    "val_target_size": val_target_size,
                    "val_loss": float(np.mean([item["val_loss"] for item in matching])),
                    "val_acc": float(np.mean([item["val_acc"] for item in matching])),
                    "val_precision": float(np.mean([item["val_precision"] for item in matching])),
                    "val_recall": float(np.mean([item["val_recall"] for item in matching])),
                    "val_f1": float(np.mean([item["val_f1"] for item in matching])),
                }
            )

    return summary_results


def save_sweep_results(results, output_path="segment_sweep_results.csv"):
    fieldnames = [
        "segment_size",
        "overlap",
        "val_loss",
        "val_acc",
        "val_precision",
        "val_recall",
        "val_f1",
        "seed",
    ]

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def save_target_transfer_results(results, output_path="target_transfer_results.csv"):
    fieldnames = [
        "train_target_size",
        "val_target_size",
        "val_loss",
        "val_acc",
        "val_precision",
        "val_recall",
        "val_f1",
        "seed",
        "run",
    ]

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def plot_sweep_results(results, output_path="segment_sweep_results.png"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot export.")
        return None

    sizes = sorted({item["segment_size"] for item in results})
    overlaps = sorted({item["overlap"] for item in results})

    matrix = np.zeros((len(sizes), len(overlaps)), dtype=np.float32)
    for item in results:
        row = sizes.index(item["segment_size"])
        col = overlaps.index(item["overlap"])
        matrix[row, col] = item["val_f1"]

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(overlaps)), max(4, 1.2 * len(sizes))))
    image = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(np.arange(len(overlaps)))
    ax.set_xticklabels(overlaps)
    ax.set_yticks(np.arange(len(sizes)))
    ax.set_yticklabels(sizes)
    ax.set_xlabel("Overlap")
    ax.set_ylabel("Segment size")
    ax.set_title("Validation F1 by segment size and overlap")

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(col_idx, row_idx, f"{matrix[row_idx, col_idx]:.3f}", ha="center", va="center")

    fig.colorbar(image, ax=ax, label="Validation F1")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_target_transfer_results(results, output_path="target_transfer_results.png"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot export.")
        return None

    train_sizes = sorted({item["train_target_size"] for item in results})
    val_sizes = sorted({item["val_target_size"] for item in results})

    matrix = np.zeros((len(train_sizes), len(val_sizes)), dtype=np.float32)
    for item in results:
        row = train_sizes.index(item["train_target_size"])
        col = val_sizes.index(item["val_target_size"])
        matrix[row, col] = item["val_acc"]

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(val_sizes)), max(4, 1.2 * len(train_sizes))))
    image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(val_sizes)))
    ax.set_xticklabels(val_sizes)
    ax.set_yticks(np.arange(len(train_sizes)))
    ax.set_yticklabels(train_sizes)
    ax.set_xlabel("Evaluation target size")
    ax.set_ylabel("Training target size")
    ax.set_title("Validation accuracy by train/eval target size")

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(col_idx, row_idx, f"{matrix[row_idx, col_idx]:.3f}", ha="center", va="center")

    fig.colorbar(image, ax=ax, label="Validation accuracy")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def run_target_transfer_experiment(
    train_target_sizes=(4, 12),
    val_target_sizes=(4, 12),
    epochs=3,
    batch_size=32,
    lr=1e-3,
    train_samples=300,
    val_samples=80,
    image_size=516,
    target_prob=0.5,
    repeat_runs=1,
    seed_base=42,
    output_csv="target_transfer_results.csv",
    plot_path="target_transfer_results.png",
    target_shape="circle",
    mix_mode="per_target",
    train_target_shape=None,
    val_target_shape=None,
    train_mix_mode=None,
    val_mix_mode=None,
):
    if train_target_shape is None:
        train_target_shape = target_shape
    if val_target_shape is None:
        val_target_shape = train_target_shape
    if train_mix_mode is None:
        train_mix_mode = mix_mode
    if val_mix_mode is None:
        val_mix_mode = train_mix_mode

    configs = build_target_transfer_configs(train_target_sizes, val_target_sizes)
    all_results = []

    for config in configs:
        for run_idx in range(repeat_runs):
            seed = seed_base + run_idx
            _, metrics = run_train(
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                train_samples=train_samples,
                val_samples=val_samples,
                image_size=image_size,
                segment_size=64,
                overlap=32,
                target_prob=target_prob,
                seed=seed,
                return_metrics=True,
                train_target_shape=train_target_shape,
                val_target_shape=val_target_shape,
                train_mix_mode=train_mix_mode,
                val_mix_mode=val_mix_mode,
                train_target_size=config["train_target_size"],
                val_target_size=config["val_target_size"],
            )
            metrics["train_target_size"] = config["train_target_size"]
            metrics["val_target_size"] = config["val_target_size"]
            metrics["seed"] = seed
            metrics["run"] = run_idx + 1
            all_results.append(metrics)

    summary_results = summarize_target_transfer_results(all_results)
    ranked_results = sorted(summary_results, key=lambda item: item.get("val_acc", -1), reverse=True)
    save_target_transfer_results(all_results, output_csv)
    plot_target_transfer_results(ranked_results, plot_path)

    print("Target transfer results:")
    for result in ranked_results:
        print(
            f"train_target_size={result['train_target_size']} val_target_size={result['val_target_size']} "
            f"val_acc={result['val_acc']:.3f} val_f1={result['val_f1']:.3f}"
        )

    return ranked_results


def run_segment_sweep(
    segment_sizes=(32, 64, 96),
    overlaps=(0, 8, 16, 32),
    epochs=3,
    batch_size=32,
    lr=1e-3,
    train_samples=300,
    val_samples=80,
    image_size=516,
    target_prob=0.5,
    repeat_runs=1,
    seed_base=42,
    output_csv="segment_sweep_results.csv",
    plot_path="segment_sweep_results.png",
    target_shape="square",
    mix_mode="per_target",
    train_target_shape=None,
    val_target_shape=None,
    train_mix_mode=None,
    val_mix_mode=None,
    device="auto",
):
    if train_target_shape is None:
        train_target_shape = target_shape
    if val_target_shape is None:
        val_target_shape = train_target_shape
    if train_mix_mode is None:
        train_mix_mode = mix_mode
    if val_mix_mode is None:
        val_mix_mode = train_mix_mode

    configs = build_sweep_configs(segment_sizes, overlaps)
    all_results = []

    for config in configs:
        try:
            validate_segment_config(config["segment_size"], config["segment_size"], config["overlap"])
        except ValueError as exc:
            print(f"Skipping invalid config segment_size={config['segment_size']} overlap={config['overlap']}: {exc}")
            continue

        for run_idx in range(repeat_runs):
            seed = seed_base + run_idx
            _, metrics = run_train(
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                train_samples=train_samples,
                val_samples=val_samples,
                image_size=image_size,
                segment_size=config["segment_size"],
                overlap=config["overlap"],
                target_prob=target_prob,
                seed=seed,
                return_metrics=True,
                train_target_shape=train_target_shape,
                val_target_shape=val_target_shape,
                train_mix_mode=train_mix_mode,
                val_mix_mode=val_mix_mode,
                device=device,
            )
            metrics["seed"] = seed
            metrics["run"] = run_idx + 1
            all_results.append(metrics)

    summary_results = []
    for config in configs:
        matching = [item for item in all_results if item["segment_size"] == config["segment_size"] and item["overlap"] == config["overlap"]]
        if not matching:
            continue
        summary = {
            "segment_size": config["segment_size"],
            "overlap": config["overlap"],
            "val_loss": float(np.mean([item["val_loss"] for item in matching])),
            "val_acc": float(np.mean([item["val_acc"] for item in matching])),
            "val_precision": float(np.mean([item["val_precision"] for item in matching])),
            "val_recall": float(np.mean([item["val_recall"] for item in matching])),
            "val_f1": float(np.mean([item["val_f1"] for item in matching])),
        }
        summary_results.append(summary)

    ranked_results = rank_sweep_results(summary_results)
    save_sweep_results(ranked_results, output_csv)
    plot_sweep_results(ranked_results, plot_path)

    print("Sweep results:")
    for result in ranked_results:
        print(
            f"segment_size={result['segment_size']} overlap={result['overlap']} "
            f"val_f1={result['val_f1']:.3f}"
        )

    return ranked_results


def visualise_predictions(model, image, mask, segment_size, overlap, vis_img_path, mask_img_path=None, device=None):
    if cv2 is None:
        raise ImportError("OpenCV is required to create prediction visualisations.")

    model_device = next(model.parameters()).device
    if device is not None:
        model_device = torch.device(device)
        model = model.to(model_device)

    segments, _, positions = segment_image(image, mask, segment_size, segment_size, overlap)

    segment_array = np.stack(segments).astype(np.float32) / 255.0
    segment_tensor = torch.from_numpy(segment_array).unsqueeze(1).to(model_device)

    model.eval()
    with torch.no_grad():
        logits = model(segment_tensor)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

    heat = np.zeros_like(image, dtype=np.float32)
    count = np.zeros_like(image, dtype=np.float32)
    for (y, x, y_end, x_end), p in zip(positions, probs):
        heat[y:y_end, x:x_end] += p
        count[y:y_end, x:x_end] += 1.0

    heat = np.divide(heat, count, out=np.zeros_like(heat), where=count > 0)
    heat_u8 = (heat * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base, 0.6, heat_color, 0.4, 0.0)
    cv2.imwrite(vis_img_path, overlay)

    if mask_img_path is not None:
        mask_u8 = (mask > 0).astype(np.uint8) * 255
        cv2.imwrite(mask_img_path, mask_u8)


RUN_CONFIG = {
    "run_sweep": False,
    "training": {
        "epochs": 3,
        "batch_size": 32,
        "lr": 1e-3,
        "train_samples": 1000,
        "val_samples": 250,
        "image_size": 516,
        "segment_size": 64,
        "overlap": 32,
        "target_prob": 0.5,
        "target_size": 8,
        "decision_threshold": 0.3,
        "seed": 67,
        "validation_seed": 68,
        "device": "cpu",
    },
    "sweep": {
        "segment_sizes": (32, 64, 96),
        "overlaps": (0, 8, 16, 32),
        "epochs": 5,
        "batch_size": 32,
        "lr": 1e-3,
        "train_samples": 5000,
        "val_samples": 1000,
        "image_size": 516,
        "target_prob": 0.5,
        "repeat_runs": 1,
        "seed_base": 67,
        "target_shape": "circle",
        "mix_mode": None,
        "device": "cpu",
    },
    "target": {
        "count": 5,
        "size": 8,
        "mode": "bw",
        "block_size": 1,
        "shape": "circle",
        "mix_mode": None,
    },
    "visualisation": {
        "output_path": "prediction_visual.png",
        "mask_output_path": "target_mask_visual.png",
    },
}


def print_run_modes(config, segment_size, overlap):
    target_config = config["target"]
    training_config = config["training"]
    requested_device = training_config["device"]
    if requested_device == "auto":
        if torch.cuda.is_available():
            resolved_device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            resolved_device = "mps"
        else:
            resolved_device = "cpu"
    else:
        resolved_device = requested_device

    device_detail = resolved_device
    if resolved_device == "cuda" and torch.cuda.is_available():
        device_detail = f"cuda ({torch.cuda.get_device_name(0)})"
    print(
        "Run configuration: "
        f"mode={'segment sweep' if config['run_sweep'] else 'standard training'}, "
        f"target_shape={target_config['shape']}, "
        f"target_mix_mode={target_config['mix_mode']}, "
        f"target_size={target_config['size']}, "
        f"segment_size={segment_size}, overlap={overlap}, "
        f"threshold={training_config['decision_threshold']}, "
        f"device={device_detail}"
    )


if __name__ == "__main__":
    run_sweep = RUN_CONFIG["run_sweep"]
    training_config = RUN_CONFIG["training"]
    sweep_config = RUN_CONFIG["sweep"]
    target_config = RUN_CONFIG["target"]
    visualisation_config = RUN_CONFIG["visualisation"]

    if run_sweep:
        print_run_modes(
            RUN_CONFIG,
            sweep_config["segment_sizes"],
            sweep_config["overlaps"],
        )
        sweep_results = run_segment_sweep(**sweep_config)
        segment_size = sweep_results[0]["segment_size"]
        overlap = sweep_results[0]["overlap"]
        sweep_training_config = {
            **training_config,
            "epochs": sweep_config["epochs"],
            "batch_size": sweep_config["batch_size"],
            "lr": sweep_config["lr"],
            "train_samples": sweep_config["train_samples"],
            "val_samples": sweep_config["val_samples"],
            "image_size": sweep_config["image_size"],
            "target_prob": sweep_config["target_prob"],
            "seed": sweep_config["seed_base"],
        }
        model = run_train(
            **sweep_training_config,
            segment_size=segment_size,
            overlap=overlap,
        )
    else:
        segment_size = training_config["segment_size"]
        overlap = training_config["overlap"]
        print_run_modes(RUN_CONFIG, segment_size, overlap)
        model = run_train(**training_config)

    background_noise = create_noise(training_config["image_size"], training_config["image_size"])
    full_image, mask = embed_targets(
        background_noise,
        target_config["count"],
        target_kwargs={
            "size": target_config["size"],
            "mode": target_config["mode"],
            "block_size": target_config["block_size"],
            "shape": target_config["shape"],
        },
        target_shape=target_config["shape"],
        mix_mode=target_config["mix_mode"],
    )

    visualise_predictions(
        model,
        full_image,
        mask,
        segment_size=segment_size,
        overlap=overlap,
        vis_img_path=visualisation_config["output_path"],
        mask_img_path=visualisation_config["mask_output_path"],
    )