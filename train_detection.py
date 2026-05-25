import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from pytorch_cnn import WatermarkCNN
from image_generation import (create_noise, create_target, embed_targets, segment_image)

class TargetDataset(Dataset):

    def __init__(self,
                 num_samples=5000,
                 image_size=1024,
                 segment_size=224,
                 overlap=64,
                 target_prob=0.5,
                 max_targets=10,
                 target_size=8,
                 target_mode="bw",
                 block_size=1,
                 positive_threshold=0.5):
        
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

    def __len__(self):
        return self.num_samples
    
    # calls for creation of a training sample
    def __getitem__(self, index):
        target_args = (self.target_size, self.target_mode, self.block_size)

        background_noise = create_noise()

        if random.random() < self.target_prob:
            num_targets = random.randint(1, self.max_targets)
            full_image, mask = embed_targets(background_noise, num_targets, target_args)
        else:
            full_image = background_noise
            mask = np.zeros_like(background_noise, dtype=np.uint8)



        # when randomly picking one segment from the image
        # TODO: take all segments
        segments, labels = segment_image(full_image, mask, self.segment_size, self.segment_size, self.overlap, self.positive_threshold)
        segment_index = random.randrange(len(segments))
        segment = segments[segment_index].astype(np.float32) / 255.0
        label = float(labels[segment_index])

        segment_tensor = torch.from_numpy(segment).unsqueeze(0)
        label_tensor = torch.tensor([label], dtype=torch.float32)

        return segment_tensor, label_tensor
    

def run_train(
            epochs=5,
            batch_size=32,
            lr=1e-3,
            train_samples=10000,
            val_samples=1000,
            image_size=1024,
            segment_size=224,
            overlap=64,
            target_prob=0.5,
):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TargetDataset(
        num_samples=train_samples,
        image_size=image_size,
        segment_size=segment_size,
        overlap=overlap,
        target_prob=target_prob,
    )
    val_ds = TargetDataset(
        num_samples=val_samples,
        image_size=image_size,
        segment_size=segment_size,
        overlap=overlap,
        target_prob=target_prob,
    )


    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = WatermarkCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

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

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

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

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

if __name__ == "__main__":
    # visualise_sample()
    run_train()
