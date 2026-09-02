from __future__ import annotations

import numpy as np

from src.topic5_group_event_state.v032_eval.h2a_probe import _standardize_from_base_fit


def test_h2a_context_standardisation_uses_base_fit_only():
    x = np.array([[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]])
    got, meta = _standardize_from_base_fit(x, np.array([0, 1]))
    assert np.allclose(got[:2].mean(axis=0), 0.0)
    assert np.allclose(got[:2].std(axis=0), 1.0)
    assert meta["fit_phase"] == "base_fit"
    # The late row is transformed, never used to recenter the prefix rows.
    assert np.all(got[2] > 10.0)

