import numpy as np

from hypok_mimic3.losses import build_class_weights, sqrt_inverse_frequency_weights


def test_sqrt_inverse_frequency_weights_are_normalized_and_ordered():
    labels = np.repeat([0, 1, 2], [4, 16, 1])
    weights = sqrt_inverse_frequency_weights(labels, num_classes=3)
    assert np.isclose(weights.mean(), 1.0)
    assert weights[2] > weights[0] > weights[1]


def test_build_class_weights_supports_none():
    config = {
        "model": {"num_classes": 3},
        "training": {"class_weight_method": "none"},
    }
    assert np.array_equal(build_class_weights(config, [0, 1, 2]), np.ones(3))
