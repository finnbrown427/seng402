import torch

from pytorch_cnn import SpatialTargetCNN, WatermarkCNN


def test_watermark_cnn_returns_one_logit_per_image():
    model = WatermarkCNN()
    output = model(torch.randn(3, 1, 16, 16))

    assert output.shape == (3, 1)


def test_spatial_target_cnn_preserves_input_spatial_dimensions():
    model = SpatialTargetCNN()
    output = model(torch.randn(2, 1, 20, 24))

    assert output.shape == (2, 1, 20, 24)


def test_spatial_target_encoder_can_be_frozen_and_unfrozen():
    model = SpatialTargetCNN()

    for parameter in model.features.parameters():
        parameter.requires_grad = False
    assert not any(parameter.requires_grad for parameter in model.features.parameters())

    for parameter in model.features.parameters():
        parameter.requires_grad = True
    assert all(parameter.requires_grad for parameter in model.features.parameters())
