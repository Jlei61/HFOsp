import numpy as np

from scripts.run_topic4_zm_fig5_workpoint_probe import _jsonable


def test_probe_sidecar_converts_numpy_values_recursively():
    payload = {
        "site": {"xy_mm": np.asarray([4.0, 9.0], dtype=np.float32)},
        "valid": np.bool_(True),
        "count": np.int64(64),
        "nested": (np.float64(0.5),),
    }

    assert _jsonable(payload) == {
        "site": {"xy_mm": [4.0, 9.0]},
        "valid": True,
        "count": 64,
        "nested": [0.5],
    }
