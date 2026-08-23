import numpy as np

from scripts.run_topic4_rev12_node_worker import _event_contact_readout
from src.sef_hfo_observation import VirtualMontage
from src.topic4_node_dualmode import (
    binned_contact_envelope,
    sheet_contact_sampling_weights,
)


def test_worker_uses_only_the_selected_directed_root_for_contact_ranks():
    counts = np.zeros((5, 3, 3), float)
    labels = np.zeros_like(counts, int)
    counts[:, 0, 0] = [0.0, 4.0, 8.0, 4.0, 0.0]
    labels[1:4, 0, 0] = 1
    counts[:, 0, 2] = [0.0, 20.0, 10.0, 20.0, 0.0]
    labels[1:4, 0, 2] = 2
    positions = np.vstack([
        np.repeat([[0.5, 0.5]], 20, axis=0),
        np.repeat([[1.5, 0.5]], 20, axis=0),
        np.repeat([[2.5, 0.5]], 20, axis=0),
    ])
    contacts = np.asarray([[0.5, 0.5], [2.5, 0.5]])
    montage = VirtualMontage(contacts, ["A1", "A2"], "test")
    population = np.zeros((3, 3), float)
    population[0] = 20.0
    weights = sheet_contact_sampling_weights(
        contacts, population,
        bin_mm=1.0, kernel_width_mm=0.1,
    )
    envelope = binned_contact_envelope(
        counts, weights, frame_ms=1.0, smooth_ms=0.01,
    )

    onsets, ranks, audit = _event_contact_readout(
        events=[{"cascade_id": 1, "t_on": 0.0, "t_off": 5.0}],
        envelope=envelope, envelope_dt_ms=1.0, montage=montage,
        valid_contacts=np.ones(2, bool), positions_e=positions,
        movie={
            "activity_counts": counts, "frame_ms": 1.0,
            "bin_mm": 1.0, "sheet_mm": 3.0,
        },
        lineage_labels=labels,
        readout={
            "source": "lineage_restricted_sheet_activity",
            "kernel_width_mm": 0.1, "smooth_ms": 0.01,
            "minimum_full_trace_pearson": 0.999,
            "participation_margin_fraction": 0.1,
            "timing_fraction": 0.5,
        },
    )

    assert np.isfinite(onsets[0, 0])
    assert np.isnan(onsets[0, 1])
    assert ranks[0, 0] == 0.0
    assert np.isnan(ranks[0, 1])
    assert audit["source"] == "lineage_restricted_sheet_activity"
    assert audit["parity_status"] == "PASS"
