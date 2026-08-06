import numpy as np

from scripts.paper_figures.plot_fig_mz_stim_site_near_runaway import _common_unbroken_window
from scripts.run_topic4_mz_stim_site_compare import (
    choose_capture_snapshot,
    electrode_target_mask,
    select_icl_contacts,
)


def test_select_icl_contacts_respects_shaft_and_anchor():
    names = ["SCL1", "ICL1", "ICL2", "ICL3", "ICL4", "ICL5"]
    contacts = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [3.0, 1.0],
        [4.0, 1.0],
    ])
    selected = select_icl_contacts(names, contacts, np.array([3.4, 1.0]), n_contacts=2)
    assert selected.tolist() == [4, 5]


def test_electrode_target_mask_is_e_indexed_union():
    pos_e = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    contacts = np.array([[0.0, 0.0], [3.0, 0.0]])
    mask = electrode_target_mask(pos_e, contacts, radius_mm=0.25)
    assert mask.dtype == bool
    assert mask.tolist() == [True, False, True]


def test_choose_capture_snapshot_is_latest_before_lead_target():
    snapshots = {
        "t80": {"step": 800, "z_E": np.ones(2), "m_E": np.zeros(2)},
        "t100": {"step": 1000, "z_E": np.ones(2), "m_E": np.zeros(2)},
        "t120": {"step": 1200, "z_E": np.ones(2), "m_E": np.zeros(2)},
    }
    label, payload, time_ms = choose_capture_snapshot(snapshots, t_run_ms=125.0, lead_ms=20.0)
    assert label == "t100"
    assert payload["step"] == 1000
    assert time_ms == 100.0


def test_common_unbroken_window_contains_endpoint_only():
    payloads = [
        {
            "meta": {"arm": "endpoint", "stim_on_ms": 8000.0, "stim_off_ms": 14000.0, "t_run_ms": 14024.6},
            "data": {"times": np.array([0.0, 14164.2])},
        },
        {
            "meta": {"arm": "middle", "stim_on_ms": 8000.0, "stim_off_ms": 14000.0, "t_run_ms": 15460.0},
            "data": {"times": np.array([0.0, 15599.6])},
        },
    ]
    assert _common_unbroken_window(payloads) == (6000.0, 14164.2)
