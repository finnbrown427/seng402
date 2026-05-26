import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader

from pytorch_cnn import WatermarkCNN
from image_generation import (create_noise, create_target, embed_targets, segment_image)

class TargetIterableDataset(IterableDataset):
    def __init__(self, num_samples=5000, image_size=1024, segment_size=224, overlap=64,
                 target_prob=0.5, max_targets=10, target_size=8, target_mode="bw",
                 block_size=1, positive_threshold=0.5):
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

    def __iter__(self):
        for _ in range(self.num_samples):
            target_args = (self.target_size, self.target_mode, self.block_size)
            background_noise = create_noise(self.image_size, self.image_size)

            if random.random() < self.target_prob:
                num_targets = random.randint(1, self.max_targets)
                full_image, mask = embed_targets(background_noise, num_targets, target_args)
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


def run_train(
    epochs=5,
    batch_size=32,
    lr=1e-3,
    train_samples=750,
    val_samples=100,
    image_size=516,
    segment_size=64,
    overlap=16,
    target_prob=0.5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TargetIterableDataset(
        num_samples=train_samples,
        image_size=image_size,
        segment_size=segment_size,
        overlap=overlap,
        target_prob=target_prob,
    )
    val_ds = TargetIterableDataset(
        num_samples=val_samples,
        image_size=image_size,
        segment_size=segment_size,
        overlap=overlap,
        target_prob=target_prob,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = WatermarkCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

        print(
            f"Epoch {epoch}: \n"
            f"training: loss={train_loss:.4f} acc={train_acc:.3f} prec={train_precision:.3f} recall={train_recall:.3f} F1={train_f1:.3f} \n"
            f"validation: loss={val_loss:.4f} acc={val_acc:.3f} prec={val_precision:.3f} recall={val_recall:.3f} F1={val_f1:.3f}"
        )
    return model

def visualise_predictions(model, image, mask, segment_size, overlap, vis_img_path, device="cpu"):
    segments, _, positions = segment_image(image, mask, segment_size, segment_size, overlap)

    segment_array = np.stack(segments).astype(np.float32) / 255.0
    segment_tensor = torch.from_numpy(segment_array).unsqueeze(1).to(device)

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

if __name__ == "__main__":
    model = run_train()

    # for making visualisation
    target_args = (8, "bw", 1)
    background_noise = create_noise(516, 516)
    full_image, mask = embed_targets(background_noise, 5, target_args)

    visualise_predictions(model, full_image, mask, segment_size=32, overlap=20, vis_img_path="prediction_visual.png")