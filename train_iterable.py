import csv
import copy
import random
import threading
import time
import tracemalloc
from contextlib import contextmanager
import numpy as np

import cv2



import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader



from pytorch_cnn import WatermarkCNN


from image_generation import create_noise, embed_targets, segment_image, validate_segment_config


@contextmanager
def measure_run():
    """Measure wall time, Python allocations, and peak process RSS."""
    try:
        import psutil
    except ImportError as exc:
        raise ImportError("Install psutil to measure sweep memory usage.") from exc

    process = psutil.Process()
    rss_samples = []
    stop_sampling = threading.Event()

    def sample_rss():
        while not stop_sampling.is_set():
            rss_samples.append(process.memory_info().rss)
            stop_sampling.wait(0.01)

    sampler = threading.Thread(target=sample_rss, daemon=True)
    result = {}
    tracemalloc.start()
    start_time = time.perf_counter()
    sampler.start()
    try:
        yield result
    finally:
        elapsed_seconds = time.perf_counter() - start_time
        stop_sampling.set()
        sampler.join()
        _, peak_python_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if rss_samples:
            peak_rss_bytes = max(rss_samples)
        else:
            peak_rss_bytes = process.memory_info().rss
        result.update({
            "time_seconds": elapsed_seconds,
            "peak_python_memory_mb": peak_python_bytes / (1024 ** 2),
            "peak_rss_mb": peak_rss_bytes / (1024 ** 2),
        })


def download_breast_cancer_dataset():
    """Download the medical dataset only when explicitly requested."""
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError("Install kagglehub to download the medical dataset.") from exc

    path = kagglehub.dataset_download("andrewmvd/breast-cancer-cell-segmentation")
    print("Path to dataset files:", path)
    return path

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
    outputs = config.get("outputs", {})

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
    best_state = None
    best_metrics = None
    best_val_f1 = float("-inf")

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

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = last_metrics

        print(
            f"Epoch {epoch}: \n"
            f"training: loss={train_loss:.4f} acc={train_acc:.3f} prec={train_precision:.3f} recall={train_recall:.3f} F1={train_f1:.3f} \n"
            f"validation: loss={val_loss:.4f} acc={val_acc:.3f} prec={val_precision:.3f} recall={val_recall:.3f} F1={val_f1:.3f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    if return_metrics:
        checkpoint_path = outputs.get("synthetic_checkpoint")
        if checkpoint_path:
            torch.save(model.state_dict(), checkpoint_path)
        return model, best_metrics
    checkpoint_path = outputs.get("synthetic_checkpoint")
    if checkpoint_path:
        torch.save(model.state_dict(), checkpoint_path)
    return model


def build_sweep_configs(segment_sizes, overlap_fractions):
    configs = []
    for size in segment_sizes:
        for frac in overlap_fractions:
            configs.append({
                "segment_size": size,
                "overlap_frac": frac,
                "overlap": int(round(size * frac)),
            })
    return configs


def rank_sweep_results(results):
    return sorted(results, key=lambda item: item.get("val_f1", -1), reverse=True)


def save_sweep_results(results, output_path="segment_sweep_results.csv"):
    fieldnames = [
        "segment_size",
        "overlap",
        "time_seconds",
        "peak_python_memory_mb",
        "peak_rss_mb",
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
    overlaps = sorted({item["overlap_frac"] for item in results})

    time_matrix = np.full((len(sizes), len(overlaps)), np.nan, dtype=np.float32)
    f1_matrix = np.full((len(sizes), len(overlaps)), np.nan, dtype=np.float32)
    for item in results:
        row = sizes.index(item["segment_size"])
        col = overlaps.index(item["overlap_frac"])
        time_matrix[row, col] = item["time_seconds"]
        f1_matrix[row, col] = item["val_f1"]

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(overlaps)), max(4, 1.2 * len(sizes))))
    # Reversed colormap: low time = green (good), high time = red (bad)
    image = ax.imshow(time_matrix, cmap="RdYlGn_r")
    ax.set_xticks(np.arange(len(overlaps)))
    ax.set_xticklabels([f"{frac:.0%}" for frac in overlaps])
    ax.set_yticks(np.arange(len(sizes)))
    ax.set_yticklabels(sizes)
    ax.set_xlabel("Overlap %")
    ax.set_ylabel("Segment size")
    ax.set_title("Time taken by segment size and overlap")

    for row_idx in range(time_matrix.shape[0]):
        for col_idx in range(time_matrix.shape[1]):
            if np.isnan(time_matrix[row_idx, col_idx]):
                continue
            ax.text(
                col_idx,
                row_idx,
                f"F1 = {f1_matrix[row_idx, col_idx]:.3f} \n ({time_matrix[row_idx, col_idx]:.1f}s)",
                ha="center",
                va="center",
            )

    fig.colorbar(image, ax=ax, label="Time taken (seconds)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def run_segment_sweep(config):
    sweep = config["sweep"]
    outputs = config["outputs"]
    configs = build_sweep_configs(sweep["segment_sizes"], sweep["overlap_fractions"])
    all_results = []

    for sweep_config in configs:
        size = sweep_config["segment_size"]
        frac = sweep_config["overlap_frac"]
        overlap = sweep_config["overlap"]
        try:
            validate_segment_config(size, size, overlap)
        except ValueError as exc:
            print(f"Skipping invalid config segment_size={size} overlap={overlap} ({frac:.0%}): {exc}")
            continue

        for run_idx in range(sweep["repeat_runs"]):
            print(f"run id = {run_idx}, size = {size}, frac = {frac}")

            seed = sweep["seed_base"] + run_idx
            candidate_config = copy.deepcopy(config)
            candidate_config["training"]["return_metrics"] = True
            candidate_config["training"]["seed"] = seed
            candidate_config["data"]["segment_size"] = size
            candidate_config["data"]["overlap"] = overlap
            with measure_run() as measurement:
                _, metrics = run_train(candidate_config)
            metrics.update(measurement)
            metrics["segment_size"] = size
            metrics["overlap"] = overlap
            metrics["overlap_frac"] = frac
            metrics["seed"] = seed
            metrics["run"] = run_idx + 1
            all_results.append(metrics)

    summary_results = []
    for sweep_config in configs:
        matching = [
            item for item in all_results
            if item["segment_size"] == sweep_config["segment_size"]
            and item["overlap_frac"] == sweep_config["overlap_frac"]
        ]
        if not matching:
            continue
        summary = {
            "segment_size": sweep_config["segment_size"],
            "overlap_frac": sweep_config["overlap_frac"],
            "overlap": sweep_config["overlap"],
            "time_seconds": float(np.mean([item["time_seconds"] for item in matching])),
            "peak_python_memory_mb": float(np.mean([item["peak_python_memory_mb"] for item in matching])),
            "peak_rss_mb": float(np.mean([item["peak_rss_mb"] for item in matching])),
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
            f"segment_size={result['segment_size']} "
            f"overlap={result['overlap']} ({result['overlap_frac']:.0%}) "
            f"val_f1={result['val_f1']:.3f} time={result['time_seconds']:.1f}s"
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
        "run_sweep": True,
        "run_medical_transfer": False,
    },
    "training": {
        "epochs": 3,
        "batch_size": 64,
        "lr": 1e-3,
        "train_samples": 200,
        "val_samples": 50,
        "seed": 43,
        "return_metrics": False,
    },
    "data": {
        "image_size": 1024,
        "segment_size": 128,
        "overlap": 64,
        "target_prob": 0.75,
        "min_targets": 25,
        "max_targets": 100,
        "positive_threshold": 1.0,
        "val_target_prob": 0.5,
        "val_min_targets": 4,
        "val_max_targets": 10,
    },
    "targets": {
        "target_size": 12,
        "target_mode": "bw",
        "block_size": 1,
        "target_shape": "circle",
        "mix_mode": None,
        "target_kwargs": {},
        "val_target_shape": None,
        "val_mix_mode": None,
    },
    "sweep": {
        "segment_sizes": (32, 64, 96),
        "overlap_fractions": (0, 0.25, 0.50, 0.75),
        "repeat_runs": 3,
        "seed_base": 67,
    },
    "outputs": {
        "output_csv": "segment_sweep_results_b.csv",
        "plot_path": "segment_sweep_results_b.png",
        "visualization_path": "prediction_visual_new.png",
        "synthetic_checkpoint": "synthetic_watermark_cnn.pt",
    },
    "medical": {
        "dataset": "data/breast-cancer-cell-segmentation",
        "output_dir": "transfer_outputs",
        "patch_size": 128,
        "overlap": 64,
        "batch_size": 16,
        "negative_ratio": 1.0,
        "validation_fraction": 0.2,
        "test_fraction": 0.2,
        "head_epochs": 5,
        "fine_tune_epochs": 5,
        "head_learning_rate": 1e-3,
        "fine_tune_learning_rate": 1e-4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
    "visualization": {
        "num_targets": 10,
        "target_shape": "circle",
        "mix_mode": None,
        "target_kwargs": {"size": 16, "mode": "bw", "block_size": 2},
        "segment_size": 48,
        "overlap": 32,
    },
}


if __name__ == "__main__":
    config = run_config

    print(f"running. mode: {config['mode']}")

    if config["mode"]["run_medical_transfer"]:
        from transfer_learning import run_medical_from_config

        run_medical_from_config(config)
    elif config["mode"]["run_sweep"]:
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