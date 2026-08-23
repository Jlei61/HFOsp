import numpy as np

from scripts.audit_topic4_rev12_native_lineage_canary import compare_shared_arrays


def test_native_lineage_parity_comparison_accepts_nan_identical_arrays():
    template = {
        key: np.asarray([np.nan, 1.0])
        for key in (
            "contact_names", "shaft_ids", "contact_xy_mm", "onsets", "ranks",
            "event_t_on_ms", "event_t_off_ms", "event_trigger_t_on_ms",
            "event_returned", "event_fragment_count", "active_fraction",
            "active_fraction_bin_ms", "contact_envelope", "contact_envelope_dt_ms",
            "sheet_activity_counts", "sheet_activity_frame_ms",
            "directed_lineage_labels", "directed_lineage_collision_mask",
            "detector_fragment_dominant_lineage_id", "detector_fragment_dominance",
            "detector_fragment_collision_fraction", "detector_fragment_compound",
            "source_onset_maps_ms", "source_onset_evaluable", "source_bin_mm",
            "source_sheet_mm", "positions_E", "h", "delta_vtheta",
            "edge_coefficients",
        )
    }
    template["contact_names"] = np.asarray(["A1", "A2"])
    template["shaft_ids"] = np.asarray(["A", "A"])
    assert compare_shared_arrays(template, template) == []
    changed = dict(template)
    changed["ranks"] = np.asarray([0.0, 1.0])
    assert compare_shared_arrays(changed, template) == ["ranks"]
