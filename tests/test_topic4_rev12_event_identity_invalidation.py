import numpy as np

from scripts.audit_topic4_rev12_exact_fit_event_identity_invalidation import (
    event_identity_summary,
)


def test_event_identity_invalidation_summary_uses_worker_level_units():
    payloads = [{
        "event_unit": {
            "raw_detector_fragment_count": fragments,
            "n_directed_roots": roots,
            "compound_detector_fragment_fraction": compound,
        },
        "events": [{"duration_ms": duration}],
    } for fragments, roots, compound, duration in (
        (10, 100, 0.2, 40), (20, 200, 0.4, 80),
    )]
    result = event_identity_summary(payloads)
    assert result["n_workers"] == 2
    assert result["median_detector_fragments_per_worker"] == 15
    assert result["median_directed_roots_per_worker"] == 150
    assert np.isclose(result["median_compound_fragment_fraction"], 0.3)
    assert result["median_event_duration_ms"] == 60
