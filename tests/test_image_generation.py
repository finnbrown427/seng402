import numpy as np
import pytest

from image_generation import (
    create_circle_target,
    create_noise,
    create_target,
    embed_targets,
    get_target_shape,
    segment_image,
    validate_segment_config,
)


def test_create_noise_has_requested_shape_and_dtype():
    noise = create_noise(width=7, height=5)

    assert noise.shape == (5, 7)
    assert noise.dtype == np.uint8
    assert noise.min() >= 0
    assert noise.max() <= 255


@pytest.mark.parametrize("shape", ["square", "circle", "mixed", "SQUARE"])
def test_get_target_shape_accepts_supported_shapes(shape):
    assert get_target_shape(shape) in {"square", "circle", "mixed"}


def test_get_target_shape_uses_fallback_for_none():
    assert get_target_shape(None, fallback="circle") == "circle"


def test_get_target_shape_rejects_unknown_shape():
    with pytest.raises(ValueError, match="Unsupported target shape"):
        get_target_shape("triangle")


def test_circle_target_has_expected_shape_and_mask():
    target, mask = create_circle_target(size=9, mode="bw")

    assert target.shape == (9, 9)
    assert mask.shape == (9, 9)
    assert mask.dtype == bool
    assert mask[4, 4]
    assert not mask[0, 0]
    assert np.all(target[~mask] == 0)


def test_create_target_mixed_returns_valid_target():
    target, mask = create_target(size=6, mode="bw", block_size=2, shape="mixed")

    assert target.shape == (6, 6)
    assert mask.shape == (6, 6)
    assert mask.dtype == bool


def test_embed_targets_marks_embedded_pixels():
    noise = np.zeros((20, 20), dtype=np.uint8)

    embedded, mask = embed_targets(
        noise,
        num_targets=1,
        target_kwargs={"size": 4, "mode": "grey", "block_size": 1},
        target_shape="square",
    )

    assert embedded.shape == noise.shape
    assert mask.shape == noise.shape
    assert mask.dtype == np.uint8
    assert np.count_nonzero(mask) == 16
    assert np.count_nonzero(embedded) > 0


@pytest.mark.parametrize(
    "segment_height,segment_width,overlap",
    [(0, 4, 0), (4, 0, 0), (4, 4, -1), (4, 4, 4)],
)
def test_validate_segment_config_rejects_invalid_values(
    segment_height, segment_width, overlap
):
    with pytest.raises(ValueError):
        validate_segment_config(segment_height, segment_width, overlap)


def test_segment_image_pads_edges_and_labels_positive_segments():
    image = np.arange(25, dtype=np.uint8).reshape(5, 5)
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[4, 4] = 1

    segments, labels, positions = segment_image(
        image, mask, segment_height=4, segment_width=4, overlap=0
    )

    assert len(segments) == 4
    assert all(segment.shape == (4, 4) for segment in segments)
    assert len(labels) == len(positions) == 4
    assert 1 in labels
    assert segments[-1][-1, -1] == 0
