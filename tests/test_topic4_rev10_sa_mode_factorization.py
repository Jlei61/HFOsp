import numpy as np

from scripts.audit_topic4_rev10_sa_mode_factorization import (
    build_strata,
    matched_label_draw,
)


def test_matched_draw_equalizes_block_and_exact_shaft_counts():
    # Four strata, each containing both labels with deliberately unequal counts.
    blocks = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    labels = np.asarray([0, 0, 0, 1, 0, 0, 1, 1, 1, 1])
    onsets = np.full((10, 4), np.nan)
    onsets[:, :2] = [[0, 1]] * 10
    onsets[:4, 2] = 2
    onsets[4:, 2:] = [2, 3]
    groups = {"ICL": np.asarray([0, 1]), "SCL": np.asarray([2, 3])}
    strata = build_strata(blocks, onsets, labels, groups)
    observed, null, used = matched_label_draw(
        strata, np.random.default_rng(2), max_events_per_label_per_stratum=5,
    )
    assert len(used) == 2
    assert len(observed[0]) == len(observed[1]) == 3
    assert len(null[0]) == len(null[1]) == 3
    for selected in (observed, null):
        for block in (0, 1):
            assert np.sum(blocks[selected[0]] == block) == np.sum(blocks[selected[1]] == block)
