import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from train_iterable import build_sweep_configs, rank_sweep_results


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
