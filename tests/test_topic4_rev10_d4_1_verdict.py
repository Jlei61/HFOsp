from scripts.aggregate_topic4_rev10_d4_1_packet_dose import adjudicate


SOURCE_IDS = ["route_A_x18_y06", "route_B_x02_y14"]


def _row(fraction, a, b, runaway=0):
    return {
        "packet_fraction_of_E": fraction,
        "packet_n_E": int(round(32000 * fraction)),
        "clean_route_A_x18_y06_networks": a,
        "clean_route_B_x02_y14_networks": b,
        "n_runaway": runaway,
    }


def test_adjudication_selects_smallest_jointly_confirmed_dose():
    rows = [_row(0.00125, 5, 4), _row(0.0025, 5, 5), _row(0.005, 6, 6)]
    verdict = adjudicate(rows, minimum_networks=5, source_ids=SOURCE_IDS)
    assert verdict["status"] == "REV10D4_1_FRESH_NETWORK_FORCED_AB_ROUTE_CONFIRMED"
    assert verdict["selected_packet_fraction_of_E"] == 0.0025
    assert verdict["selected_packet_n_E"] == 80


def test_adjudication_requires_both_sources_and_no_runaway():
    rows = [_row(0.0025, 6, 4), _row(0.005, 6, 6, runaway=1)]
    verdict = adjudicate(rows, minimum_networks=5, source_ids=SOURCE_IDS)
    assert verdict["status"] == "REV10D4_1_FRESH_NETWORK_FORCED_AB_ROUTE_NOT_CONFIRMED"
    assert verdict["selected_packet_fraction_of_E"] is None
