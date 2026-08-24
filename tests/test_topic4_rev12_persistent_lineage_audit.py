import numpy as np

from scripts.audit_topic4_rev12_persistent_lineage_canary import (
    clean_fragment_jaccard,
    partition_coassignment_jaccard,
)


def test_partition_jaccard_ignores_compound_singletons():
    assert partition_coassignment_jaccard(
        np.asarray([0, 0, -3, 1, 1]),
        np.asarray([4, 4, -8, 9, 9]),
    ) == 1.0


def test_partition_jaccard_detects_event_split():
    observed = partition_coassignment_jaccard(
        np.asarray([0, 0, 0, 0]),
        np.asarray([1, 1, 2, 2]),
    )
    assert np.isclose(observed, 2 / 6)


def test_clean_fragment_jaccard_detects_compound_threshold_changes():
    assert np.isclose(clean_fragment_jaccard(
        np.asarray([0, 1, -3, -4]),
        np.asarray([0, -2, 2, -4]),
    ), 1 / 3)
