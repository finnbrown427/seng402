from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from transfer_learning import (
    ImageMaskPair,
    MedicalPatchDataset,
    build_patch_records,
    crop_and_pad,
    save_prediction_overlay,
    segmentation_metrics,
    split_pairs,
)


def test_crop_and_pad_returns_requested_patch_size():
    array = np.ones((3, 4), dtype=np.uint8)

    patch = crop_and_pad(array, 1, 2, 3, 4, patch_size=4)

    assert patch.shape == (4, 4)
    assert np.all(patch[:2, :2] == 1)
    assert np.all(patch[2:, :] == 0)
    assert np.all(patch[:, 2:] == 0)


def test_split_pairs_keeps_source_images_in_one_split():
    pairs = [
        ImageMaskPair(Path(f"image_{index}.png"), Path(f"mask_{index}.png"), f"source-{index}")
        for index in range(4)
    ]

    splits = split_pairs(pairs, validation_fraction=0.25, test_fraction=0.25, seed=7)

    split_ids = [
        {pair.source_id for pair in splits[name]}
        for name in ("train", "validation", "test")
    ]
    assert all(split_ids)
    assert not (split_ids[0] & split_ids[1])
    assert not (split_ids[0] & split_ids[2])
    assert not (split_ids[1] & split_ids[2])


def test_split_pairs_rejects_invalid_fractions():
    with pytest.raises(ValueError):
        split_pairs([], validation_fraction=0.8, test_fraction=0.3)


def test_segmentation_metrics_counts_binary_predictions():
    logits = torch.tensor([[[[10.0, -10.0], [10.0, -10.0]]]])
    masks = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])

    metrics = segmentation_metrics(logits, masks)

    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["accuracy"] == pytest.approx(0.75)


def test_medical_patch_dataset_reads_and_normalises_patch(tmp_path):
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(image_path), np.full((4, 4), 128, dtype=np.uint8))
    cv2.imwrite(str(mask_path), np.zeros((4, 4), dtype=np.uint8))
    pair = ImageMaskPair(image_path, mask_path, "sample")
    records = build_patch_records([pair], patch_size=4, overlap=0)

    dataset = MedicalPatchDataset([pair], records, patch_size=4)
    image, mask = dataset[0]

    assert len(dataset) == 1
    assert image.shape == (1, 4, 4)
    assert mask.shape == (1, 4, 4)
    assert image.dtype == torch.float32
    assert float(image.mean()) == pytest.approx(128 / 255)
    assert torch.count_nonzero(mask) == 0


def test_save_prediction_overlay_creates_image(tmp_path):
    output_path = tmp_path / "overlay.png"
    image = np.zeros((4, 4), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    prediction = np.zeros((4, 4), dtype=np.float32)
    mask[0, 0] = 1
    prediction[1, 1] = 1

    save_prediction_overlay(image, mask, prediction, output_path)

    saved = cv2.imread(str(output_path))
    assert saved is not None
    assert saved.shape == (4, 4, 3)
