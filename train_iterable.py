import csv
import copy
import random
import numpy as np

import cv2



import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader



from pytorch_cnn import WatermarkCNN


from image_generation import create_noise, embed_targets, segment_image, validate_segment_config



class TargetIterableDataset(IterableDataset):
    def __init__(self, config, split):
        data = config["data"]
        training = config["training"]
        targets = config["targets"]
        self.num_samples = training[f"{split}_samples"]
        self.image_size = data["image_size"]
        self.segment_size = data["segment_size"]
        self.overlap = data["overlap"]
        self.target_prob = data["target_prob"] if split == "train" else data["val_target_prob"]
        self.min_targets = data["min_targets"] if split == "train" else data["val_min_targets"]
        self.max_targets = data["max_targets"] if split == "train" else data["val_max_targets"]
        self.target_size = targets["target_size"]
        self.target_mode = targets["target_mode"]
        self.block_size = targets["block_size"]
        self.positive_threshold = data["positive_threshold"]
        self.target_shape = targets["target_shape"]
        self.mix_mode = targets["mix_mode"]
        self.target_kwargs = dict(targets["target_kwargs"])

        if split == "val":
            self.target_shape = targets["val_target_shape"] or self.target_shape
            self.mix_mode = targets["val_mix_mode"] or self.mix_mode

    def __iter__(self):
        for _ in range(self.num_samples):
            base_target_kwargs = {
                "size": self.target_size,
                "mode": self.target_mode,
                "block_size": self.block_size,
                **self.target_kwargs,
            }
            background_noise = create_noise(self.image_size, self.image_size)

            if random.random() < self.target_prob:
                num_targets = random.randint(self.min_targets, self.max_targets)
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


def run_train(config):
    training = config["training"]
    data = config["data"]

    epochs = training["epochs"]
    batch_size = training["batch_size"]
    lr = training["lr"]
    seed = training["seed"]
    return_metrics = training["return_metrics"]
    segment_size = data["segment_size"]
    overlap = data["overlap"]

    if torch is None or nn is None or WatermarkCNN is None:
        raise ImportError("PyTorch is required to run training.")

    validate_segment_config(segment_size, segment_size, overlap)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TargetIterableDataset(config, "train")
    val_ds = TargetIterableDataset(config, "val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = WatermarkCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    last_metrics = None

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
            preds = (torch.sigmoid(logits) > 0.5).float()
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

        with torch.no_grad():
            for segments, labels in val_loader:
                segments = segments.to(device)
                labels = labels.to(device)
                logits = model(segments)
                loss = criterion(logits, labels)

                val_loss += loss.item() * segments.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
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
        }

        print(
            f"Epoch {epoch}: \n"
            f"training: loss={train_loss:.4f} acc={train_acc:.3f} prec={train_precision:.3f} recall={train_recall:.3f} F1={train_f1:.3f} \n"
            f"validation: loss={val_loss:.4f} acc={val_acc:.3f} prec={val_precision:.3f} recall={val_recall:.3f} F1={val_f1:.3f}"
        )

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


def run_segment_sweep(config):
    sweep = config["sweep"]
    outputs = config["outputs"]
    configs = build_sweep_configs(sweep["segment_sizes"], sweep["overlaps"])
    all_results = []

    for sweep_config in configs:
        try:
            validate_segment_config(sweep_config["segment_size"], sweep_config["segment_size"], sweep_config["overlap"])
        except ValueError as exc:
            print(f"Skipping invalid config segment_size={sweep_config['segment_size']} overlap={sweep_config['overlap']}: {exc}")
            continue

        for run_idx in range(sweep["repeat_runs"]):
            seed = sweep["seed_base"] + run_idx
            candidate_config = copy.deepcopy(config)
            candidate_config["training"]["return_metrics"] = True
            candidate_config["training"]["seed"] = seed
            candidate_config["data"]["segment_size"] = sweep_config["segment_size"]
            candidate_config["data"]["overlap"] = sweep_config["overlap"]
            _, metrics = run_train(
                candidate_config,
            )
            metrics["seed"] = seed
            metrics["run"] = run_idx + 1
            all_results.append(metrics)

    summary_results = []
    for sweep_config in configs:
        matching = [item for item in all_results if item["segment_size"] == sweep_config["segment_size"] and item["overlap"] == sweep_config["overlap"]]
        if not matching:
            continue
        summary = {
            "segment_size": sweep_config["segment_size"],
            "overlap": sweep_config["overlap"],
            "val_loss": float(np.mean([item["val_loss"] for item in matching])),
            "val_acc": float(np.mean([item["val_acc"] for item in matching])),
            "val_precision": float(np.mean([item["val_precision"] for item in matching])),
            "val_recall": float(np.mean([item["val_recall"] for item in matching])),
            "val_f1": float(np.mean([item["val_f1"] for item in matching])),
        }
        summary_results.append(summary)

    ranked_results = rank_sweep_results(summary_results)
    save_sweep_results(ranked_results, outputs["output_csv"])
    plot_sweep_results(ranked_results, outputs["plot_path"])

    print("Sweep results:")
    for result in ranked_results:
        print(
            f"segment_size={result['segment_size']} overlap={result['overlap']} "
            f"val_f1={result['val_f1']:.3f}"
        )

    return ranked_results


def visualise_predictions(model, image, mask, config):
    if cv2 is None:
        raise ImportError("OpenCV is required to create prediction visualisations.")

    visualization = config["visualization"]
    segment_size = visualization["segment_size"]
    overlap = visualization["overlap"]
    vis_img_path = config["outputs"]["visualization_path"]
    model_device = next(model.parameters()).device

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


run_config = {
    "mode": {
        "run_sweep": False,
    },
    "training": {
        "epochs": 3,
        "batch_size": 32,
        "lr": 1e-3,
        "train_samples": 2500,
        "val_samples": 500,
        "seed": 42,
        "return_metrics": False,
    },
    "data": {
        "image_size": 516,
        "segment_size": 64,
        "overlap": 8,
        "target_prob": 0.99,
        "min_targets": 10,
        "max_targets": 40,
        "positive_threshold": 0.5,
        "val_target_prob": 0.5,
        "val_min_targets": 4,
        "val_max_targets": 10,
    },
    "targets": {
        "target_size": 8,
        "target_mode": "bw",
        "block_size": 1,
        "target_shape": "square",
        "mix_mode": None,
        "target_kwargs": {},
        "val_target_shape": None,
        "val_mix_mode": None,
    },
    "sweep": {
        "segment_sizes": (32, 64, 96),
        "overlaps": (0, 8, 16, 32),
        "repeat_runs": 1,
        "seed_base": 42,
    },
    "outputs": {
        "output_csv": "segment_sweep_results.csv",
        "plot_path": "segment_sweep_results.png",
        "visualization_path": "prediction_visual_new.png",
    },
    "visualization": {
        "num_targets": 10,
        "target_shape": "circle",
        "mix_mode": None,
        "target_kwargs": {"size": 8, "mode": "bw", "block_size": 1},
        "segment_size": 32,
        "overlap": 20,
    },
}


if __name__ == "__main__":
    config = run_config

    print(f"running. mode: {config["mode"]}")

    if config["mode"]["run_sweep"]:
        sweep_results = run_segment_sweep(config)
        selected = sweep_results[0]
        selected_config = copy.deepcopy(config)
        selected_config["data"]["segment_size"] = selected["segment_size"]
        selected_config["data"]["overlap"] = selected["overlap"]
        model = run_train(selected_config)
    else:
        model = run_train(config)

    visualization = config["visualization"]
    background_noise = create_noise(config["data"]["image_size"], config["data"]["image_size"])
    full_image, mask = embed_targets(
        background_noise,
        visualization["num_targets"],
        target_kwargs=visualization["target_kwargs"],
        target_shape=visualization["target_shape"],
        mix_mode=visualization["mix_mode"],
    )
    visualise_predictions(model, full_image, mask, config)