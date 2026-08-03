import numpy as np

from hypok_mimic3.metrics import classification_metrics


def test_macro_auroc_is_mean_of_per_class_ovr_metrics():
    true = np.array([0, 0, 1, 1, 2, 2])
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.6, 0.3, 0.1],
            [0.2, 0.7, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
            [0.1, 0.3, 0.6],
        ],
        dtype=np.float32,
    )
    result = classification_metrics(true, probabilities.argmax(axis=1), probabilities)
    class_mean = np.mean(
        [result["per_class"][name]["auroc"] for name in ("HypoK", "NK", "HyperK")]
    )
    assert np.isfinite(result["macro_auroc_ovr"])
    assert np.isclose(result["macro_auroc_ovr"], class_mean)
