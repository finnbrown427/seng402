"""Generate example circle targets for visual debugging.

This script creates a small contact sheet of synthetic circle targets so it is
easy to inspect the mask shape, block pattern, and edge behavior.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from image_generation import create_circle_target


def build_circle_mask(size: int) -> np.ndarray:
    center = (size - 1) / 2.0
    y_idx, x_idx = np.indices((size, size), dtype=np.float32)
    dist_sq = (x_idx - center) ** 2 + (y_idx - center) ** 2
    radius = size / 2.0
    return (dist_sq <= (radius ** 2)).astype(np.uint8) * 255


def to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def add_label(canvas: np.ndarray, text: str, origin: tuple[int, int]) -> None:
    cv2.putText(
        canvas,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def make_contact_sheet(images: list[np.ndarray], labels: list[str], columns: int = 4, padding: int = 16) -> np.ndarray:
    if not images:
        raise ValueError("At least one image is required")

    tile_height, tile_width = images[0].shape[:2]
    rows = (len(images) + columns - 1) // columns
    sheet_height = rows * tile_height + (rows + 1) * padding
    sheet_width = columns * tile_width + (columns + 1) * padding

    sheet = np.zeros((sheet_height, sheet_width, 3), dtype=np.uint8)
    sheet[:] = (18, 18, 18)

    for index, (image, label) in enumerate(zip(images, labels)):
        row = index // columns
        col = index % columns

        y0 = padding + row * (tile_height + padding)
        x0 = padding + col * (tile_width + padding)

        tile = to_bgr(image)
        sheet[y0:y0 + tile_height, x0:x0 + tile_width] = tile
        add_label(sheet, label, (x0 + 4, y0 + 16))

    return sheet


def generate_examples(size: int, mode: str, block_size: int, count: int, seed: int | None) -> tuple[list[np.ndarray], list[str]]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    mask = build_circle_mask(size)
    images: list[np.ndarray] = []
    labels: list[str] = []

    for index in range(count):
        target = create_circle_target(size=size, mode=mode, block_size=block_size)

        images.append(target)
        labels.append(f"target {index + 1}")

        if index == 0:
            images.append(mask)
            labels.append("circle mask")

    return images, labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate example circle targets for debugging.")
    parser.add_argument("--size", type=int, default=32, help="Target size in pixels")
    parser.add_argument("--mode", choices=["bw", "gray", "grey"], default="bw", help="Target fill mode")
    parser.add_argument("--block-size", type=int, default=4, help="Block size used inside the target")
    parser.add_argument("--count", type=int, default=8, help="Number of example targets to generate")
    parser.add_argument("--columns", type=int, default=4, help="Number of columns in the contact sheet")
    parser.add_argument("--output", type=Path, default=Path("circle_targets_debug.png"), help="Output PNG path")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible examples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images, labels = generate_examples(
        size=8,
        mode=args.mode,
        block_size=1,
        count=args.count,
        seed=args.seed,
    )
    sheet = make_contact_sheet(images, labels, columns=args.columns)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), sheet):
        raise RuntimeError(f"Failed to write output image to {args.output}")

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()