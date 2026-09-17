import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from train_iterable import (
    build_sweep_configs,
    build_target_transfer_configs,
    rank_sweep_results,
    summarize_target_transfer_results,
)


def test_build_sweep_configs_and_rank():
    configs = build_sweep_configs([32, 64], [0, 16])

    assert len(configs) == 4
    assert configs[0]["segment_size"] == 32
    assert configs[0]["overlap"] == 0

    results = [
        {"segment_size": 32, "overlap": 16, "val_f1": 0.61},
        {"segment_size": 64, "overlap": 0, "val_f1": 0.72},
        {"segment_size": 64, "overlap": 16, "val_f1": 0.68},
    ]

    ranked = rank_sweep_results(results)

    assert ranked[0]["segment_size"] == 64
    assert ranked[0]["overlap"] == 0
    assert ranked[1]["segment_size"] == 64
    assert ranked[1]["overlap"] == 16


def test_build_target_transfer_configs_and_summary():
    configs = build_target_transfer_configs([4, 12], [4, 12])

    assert len(configs) == 4
    assert configs[0]["train_target_size"] == 4
    assert configs[0]["val_target_size"] == 4

    results = [
        {"train_target_size": 4, "val_target_size": 4, "val_loss": 0.7, "val_acc": 0.6, "val_precision": 0.5, "val_recall": 0.4, "val_f1": 0.44},
        {"train_target_size": 4, "val_target_size": 12, "val_loss": 0.6, "val_acc": 0.7, "val_precision": 0.6, "val_recall": 0.5, "val_f1": 0.54},
        {"train_target_size": 12, "val_target_size": 4, "val_loss": 0.5, "val_acc": 0.8, "val_precision": 0.7, "val_recall": 0.6, "val_f1": 0.64},
        {"train_target_size": 12, "val_target_size": 12, "val_loss": 0.4, "val_acc": 0.9, "val_precision": 0.8, "val_recall": 0.7, "val_f1": 0.74},
    ]

    summary = summarize_target_transfer_results(results)

    assert len(summary) == 4
    assert summary[0]["train_target_size"] == 4
    assert summary[0]["val_target_size"] == 4
    assert summary[-1]["train_target_size"] == 12
    assert summary[-1]["val_target_size"] == 12
