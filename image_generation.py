import random
import numpy as np
import cv2


def create_noise(width=1024, height=1024):
    noise = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    return noise

def create_target(size=8, mode="bw", block_size=1):
    target_pattern = np.zeros((size, size), dtype=np.uint8)

    for i in range(0, size, block_size):
        for j in range(0, size, block_size):
            if random.choice([True, False]):
                target_pattern[i:i+block_size, j:j+block_size] = 255

    # option to make qr code grey scale
    if mode == "grey" or mode == "gray":
        target_pattern = np.zeros((size, size), dtype=np.uint8)
        for i in range(0, size, block_size):
            for j in range(0, size, block_size):
                target_pattern[i:i+block_size, j:j+block_size] = random.randint(0, 255)

    return target_pattern

def embed_targets(noise, num_targets, target_args):
    background_height, background_width = noise.shape[:2]
    mask = np.zeros((background_height, background_width), dtype=np.uint8)

    for _ in range(num_targets):
        current_target = create_target(*target_args)
        target_height, target_width = current_target.shape[:2]
        y_pos = random.randint(0, background_height - target_height)
        x_pos = random.randint(0, background_width - target_width)

        noise[y_pos:y_pos + target_height, x_pos:x_pos + target_width] = current_target
        mask[y_pos:y_pos + target_height, x_pos:x_pos + target_width] = 1

    return noise, mask


def validate_segment_config(segment_height, segment_width, overlap):
    if segment_height <= 0 or segment_width <= 0:
        raise ValueError("segment size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= segment_height or overlap >= segment_width:
        raise ValueError(
            f"overlap must be smaller than the segment size; got overlap={overlap}, segment_size={segment_height}"
        )
    return True


def segment_image(image, mask, segment_height, segment_width, overlap, positive_threshold=1):
    validate_segment_config(segment_height, segment_width, overlap)

    segments = []
    positions = []
    image_height, image_width = image.shape[:2]

    y_step = segment_height - overlap
    x_step = segment_width - overlap

    for y in range(0, image_height, y_step):
        for x in range(0, image_width, x_step):
            y_end = min(y + segment_height, image_height)
            x_end = min(x + segment_width, image_width)

            segment = image[y:y_end, x:x_end].copy()

            # padding for incomplete edge segments
            if segment.shape[0] < segment_height or segment.shape[1] < segment_width:
                padding_height = segment_height - segment.shape[0]
                padding_width = segment_width - segment.shape[1]
                segment = np.pad(segment, ((0, padding_height), (0, padding_width)), mode="constant", constant_values=0)
            
            segments.append(segment)
            positions.append((y, x, y_end, x_end))

    # label segments
    labels = []
    for (y, x, y_end, x_end) in positions:
        mask_crop = mask[y:y_end, x:x_end]
        label = 1 if int(mask_crop.sum()) >= positive_threshold else 0
        labels.append(label)

    return segments, labels, positions


# target configs:
SIZE=8
MODE='bw'
BLOCK_SIZE=1
SEGMENT_SIZE=224
SEGMENT_OVERLAP=64


if __name__ == "main":
    target_args = (SIZE, MODE, BLOCK_SIZE)
    NUM_TARGETS = random.randint(1, 10)

    background_noise = create_noise()
    full_image, mask = embed_targets(background_noise, NUM_TARGETS, target_args)

    segments, labels, positions = segment_image(full_image, mask, SEGMENT_SIZE, SEGMENT_SIZE, SEGMENT_OVERLAP)