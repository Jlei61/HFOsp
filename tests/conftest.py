from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@pytest.fixture
def synthetic_group_analysis_npz(tmp_path: Path) -> str:
    ch_names = np.array(["A1", "B1", "C1"], dtype=object)
    n_ch = int(ch_names.shape[0])
    n_events = 12

    starts = np.arange(n_events, dtype=np.float64)
    event_windows = np.stack([starts, starts + 0.5], axis=1)

    events_bool = np.ones((n_ch, n_events), dtype=bool)
    centroid_time = np.vstack(
        [
            np.full(n_events, 0.020, dtype=np.float64),
            np.full(n_events, 0.032, dtype=np.float64),
            np.full(n_events, 0.045, dtype=np.float64),
        ]
    )
    lag_raw = centroid_time.copy()
    lag_rank = np.tile(np.array([[0], [1], [2]], dtype=np.int64), (1, n_events))

    # Diagonal = events per node; off-diagonal = co-activation counts.
    coact_event_count = np.array(
        [
            [12, 11, 10],
            [11, 12, 9],
            [10, 9, 12],
        ],
        dtype=np.int64,
    )

    p = tmp_path / "synthetic_group_analysis.npz"
    np.savez_compressed(
        str(p),
        sfreq=np.array([1000.0], dtype=np.float64),
        band=np.array(["ripple"], dtype=object),
        ch_names=ch_names,
        window_sec=np.array([0.5], dtype=np.float64),
        n_events=np.array([n_events], dtype=np.int64),
        n_channels=np.array([n_ch], dtype=np.int64),
        event_windows=event_windows,
        centroid_time=centroid_time,
        events_bool=events_bool,
        lag_raw=lag_raw,
        lag_rank=lag_rank,
        coact_event_count=coact_event_count,
        coact_all_event_count=coact_event_count,
        coact_all_ch_names=ch_names,
    )
    return str(p)


@pytest.fixture
def synthetic_gpu_npz(tmp_path: Path) -> str:
    chns_names = np.array(["A1", "B1", "C1"], dtype=object)
    starts = np.arange(12, dtype=np.float64)

    # A leads B leads C with stable positive delay.
    det_a = np.stack([starts + 0.020, starts + 0.080], axis=1)
    det_b = np.stack([starts + 0.032, starts + 0.092], axis=1)
    det_c = np.stack([starts + 0.045, starts + 0.105], axis=1)

    whole_dets = np.array([det_a, det_b, det_c], dtype=object)
    p = tmp_path / "synthetic_gpu.npz"
    np.savez_compressed(str(p), chns_names=chns_names, whole_dets=whole_dets)
    return str(p)
