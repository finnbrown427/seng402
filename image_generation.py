import random
import numpy as np
import cv2


def create_noise(width=1024, height=1024):
    noise = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    return noise


def get_target_shape(shape, fallback="square"):
    if shape is None:
        return fallback
    shape_name = str(shape).lower()
    if shape_name in {"square", "circle", "mixed"}:
        return shape_name
    raise ValueError(f"Unsupported target shape '{shape}'. Use 'square', 'circle', or 'mixed'.")


def create_square_target(size, mode, block_size):
    target_pattern = np.zeros((size, size), dtype=np.uint8)

    for i in range(0, size, block_size):
        for j in range(0, size, block_size):
            if random.choice([True, False]):
                target_pattern[i:i+block_size, j:j+block_size] = 255

    if mode == "grey" or mode == "gray":
        target_pattern = np.zeros((size, size), dtype=np.uint8)
        for i in range(0, size, block_size):
            for j in range(0, size, block_size):
                target_pattern[i:i+block_size, j:j+block_size] = random.randint(0, 255)

    return target_pattern


def create_circle_target(size, mode, block_size=1):

    target_pattern = np.zeros((size, size), dtype=np.uint8)
    center = (size - 1) / 2.0
    y_idx, x_idx = np.indices((size, size), dtype=np.float32)
    dist_sq = (x_idx - center) ** 2 + (y_idx - center) ** 2
    radius = size / 2.0
    mask = dist_sq <= (radius ** 2)

    # Fill pattern in blocks, then mask to circle shape
    for i in range(0, size, block_size):
        for j in range(0, size, block_size):
            h = min(block_size, size - i)
            w = min(block_size, size - j)
            if mode == "grey" or mode == "gray":
                value = random.randint(0, 255)
            else:
                value = 255 if random.choice([True, False]) else 0
            target_pattern[i:i + h, j:j + w] = value

    # Zero out pixels outside the circular mask so embedding can preserve
    # the underlying background outside the circle.
    target_pattern[~mask] = 0

    return target_pattern


def create_target(size=8, mode="bw", block_size=1, shape="square"):
    shape_name = get_target_shape(shape)

    if shape_name == "mixed":
        shape_name = random.choice(["square", "circle"])

    if shape_name == "square":
        return create_square_target(int(size), mode, int(block_size))
    if shape_name == "circle":
        return create_circle_target(int(size), mode, int(block_size))

    raise ValueError(f"Unsupported target shape '{shape_name}'.")


def _coerce_target_kwargs(target_args=None, target_kwargs=None, target_shape="square", mix_mode=None):
    if target_kwargs is not None:
        kwargs = dict(target_kwargs)
    elif target_args is not None:
        if isinstance(target_args, dict):
            kwargs = dict(target_args)
        elif isinstance(target_args, (tuple, list)):
            if len(target_args) == 3:
                size, mode, block_size = target_args
                kwargs = {"size": size, "mode": mode, "block_size": block_size}
            else:
                raise ValueError("Legacy target_args must be a tuple/list of (size, mode, block_size).")
        else:
            raise TypeError("target_args must be a tuple/list or dict.")
    else:
        kwargs = {}

    kwargs.setdefault("size", 8)
    kwargs.setdefault("mode", "bw")
    kwargs.setdefault("block_size", 1)
    kwargs["shape"] = kwargs.get("shape", target_shape)

    if mix_mode == "per_target" and target_shape == "mixed":
        kwargs["shape"] = random.choice(["square", "circle"])

    return kwargs


def embed_targets(noise, num_targets, target_args=None, target_kwargs=None, target_shape="square", mix_mode="per_target"):
    background_height, background_width = noise.shape[:2]
    mask = np.zeros((background_height, background_width), dtype=np.uint8)

    if target_shape == "mixed" and mix_mode == "per_image":
        target_shape = random.choice(["square", "circle"])

    for _ in range(num_targets):
        if target_shape == "mixed" and mix_mode in {"per_target", None}:
            current_shape = random.choice(["square", "circle"])
        else:
            current_shape = target_shape

        kwargs = _coerce_target_kwargs(
            target_args=target_args,
            target_kwargs=target_kwargs,
            target_shape=current_shape,
            mix_mode=mix_mode,
        )
        current_target = create_target(**kwargs)
        target_height, target_width = current_target.shape[:2]
        y_pos = random.randint(0, background_height - target_height)
        x_pos = random.randint(0, background_width - target_width)

        # Only write pixels where the target has non-zero values so the
        # underlying background (noise) remains outside the target's mask.
        region = noise[y_pos:y_pos + target_height, x_pos:x_pos + target_width]
        write_mask = current_target > 0
        if write_mask.any():
            region[write_mask] = current_target[write_mask]
            noise[y_pos:y_pos + target_height, x_pos:x_pos + target_width] = region

        mask[y_pos:y_pos + target_height, x_pos:x_pos + target_width] = np.maximum(
            mask[y_pos:y_pos + target_height, x_pos:x_pos + target_width],
            (current_target > 0).astype(np.uint8),
        )

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