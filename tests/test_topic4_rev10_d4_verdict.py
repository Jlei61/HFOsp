from scripts.aggregate_topic4_rev10_d4_uniform_source_map import adjudicate


def _rows(shared=False, incidental=False):
    rows = []
    for index in range(25):
        rows.append({
            "source_id": f"source_{index:02d}",
            "clean_A_networks": 2 if shared and index == 7 else (
                1 if incidental and index == 7 else 0
            ),
            "clean_B_networks": 3 if index == 3 else 0,
        })
    return rows


def test_same_source_in_two_networks_is_required_for_forced_A_capacity():
    verdict = adjudicate(_rows(shared=True), 2)
    assert verdict["status"].endswith("OBSERVED")
    assert verdict["selected_source_id"] == "source_07"


def test_single_network_A_is_descriptive_only():
    verdict = adjudicate(_rows(incidental=True), 2)
    assert verdict["status"].endswith("NOT_OBSERVED")
    assert verdict["single_network_A_only"] is True
