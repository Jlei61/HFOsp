import numpy as np
import pytest

from scripts.audit_topic4_rev12_population_excursion_canary import (
    _pair_sensitivity,
    fragment_partition,
)


def _episode(indices):
    return {"detector_fragment_indices": indices}


def test_fragment_partition_is_complete_and_detects_overlap():
    assert fragment_partition([
        _episode([0, 1]), _episode([2]),
    ], 3).tolist() == [0, 0, 1]
    with pytest.raises(RuntimeError, match="multiple excursions"):
        fragment_partition([_episode([0, 1]), _episode([1, 2])], 3)
    with pytest.raises(RuntimeError, match="dropped"):
        fragment_partition([_episode([0])], 2)


def test_pair_sensitivity_separates_boundary_and_mode_instability():
    primary = {
        "n_detector_fragments": 4,
        "fragment_partition": np.asarray([0, 0, 1, 1]),
        "fragment_modes": np.asarray([0, 0, 1, 1]),
        "n_returned_excursions": 2,
    }
    other = {
        "n_detector_fragments": 4,
        "fragment_partition": np.asarray([0, 0, 0, 0]),
        "fragment_modes": np.asarray([1, 1, 1, 1]),
        "n_returned_excursions": 1,
    }
    result = _pair_sensitivity(primary, other)
    assert result["boundary_partition_ari"] == 0.0
    assert result["patient_mode_agreement_on_common_fragments"] == 0.5
    assert result["returned_excursion_count_difference"] == -1
