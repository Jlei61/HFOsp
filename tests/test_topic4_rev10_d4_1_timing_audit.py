from scripts.aggregate_topic4_rev10_d4_1_timing_audit import adjudicate_timing


SOURCES = ["A", "B"]


def _rows(broad_a=5, broad_b=5, overlap=False):
    rows = []
    for source, broad in (("A", broad_a), ("B", broad_b)):
        for index in range(6):
            rows.append({
                "source_id": source,
                "original_triggered": index < 4,
                "broad_triggered": index < broad,
                "late_after_40ms": 4 <= index < broad,
                "sham_overlap": overlap and index == 0,
                "expected_mode_match": True,
                "joint_shaft": True,
                "ood": False,
            })
    return rows


def test_broad_timing_support_is_secondary_and_preserves_formal_failure():
    verdict = adjudicate_timing(_rows(), source_ids=SOURCES, minimum_networks=5)
    assert verdict["status"].endswith("BROAD_ROUTE_TIMING_SUPPORTED")
    assert verdict["formal_D4_1_verdict_unchanged"].endswith("NOT_CONFIRMED")


def test_sham_overlap_or_sparse_broad_response_remains_unresolved():
    overlap = adjudicate_timing(
        _rows(overlap=True), source_ids=SOURCES, minimum_networks=5,
    )
    sparse = adjudicate_timing(
        _rows(broad_b=4), source_ids=SOURCES, minimum_networks=5,
    )
    assert overlap["status"].endswith("REMAINS_UNRESOLVED")
    assert sparse["status"].endswith("REMAINS_UNRESOLVED")
