from __future__ import annotations

from pathlib import Path

import numpy as np

from src.event_periodicity import (
    _compute_half_life,
    _epoch_to_hour,
    compute_daynight_stratified_detrending,
    compute_detrended_serial_correlation,
    compute_multiscale_detrend_fraction,
    compute_nparticipating_autocorrelation,
    compute_serial_correlation_decay,
    compute_serial_corr_soz_stratified,
    compute_within_block_serial_corr,
    merge_contiguous_blocks,
)


def _ar1_iei(n: int, phi: float = 0.8, sigma: float = 0.2, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(n, dtype=float)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + sigma * rng.normal()
    return np.exp(x + 0.4)


def _events_from_starts(starts: np.ndarray, dur: float = 0.05) -> np.ndarray:
    starts = np.asarray(starts, dtype=float)
    return np.column_stack([starts, starts + dur])


def test_compute_serial_correlation_decay_reports_positive_decay() -> None:
    iei = _ar1_iei(1500, phi=0.88, sigma=0.15, seed=3)
    out = compute_serial_correlation_decay(iei, max_lag=20, min_pairs=100)

    assert out["lags"][0] == 1
    assert np.isfinite(out["rs"][0])
    assert out["rs"][0] > 0.5
    assert np.nanmedian(out["rs"][5:10]) < out["rs"][0]
    assert np.isfinite(out["half_life_lag"])
    assert out["half_life_lag"] > 1.0


def test_compute_detrended_serial_correlation_reduces_slow_rate_drift() -> None:
    rng = np.random.default_rng(7)
    slow = 1.4 + 0.45 * np.sin(np.linspace(0.0, 8.0 * np.pi, 500))
    iei = slow * np.exp(0.07 * rng.normal(size=slow.size))
    event_times = np.concatenate([[0.0], np.cumsum(iei)])

    out = compute_detrended_serial_correlation(
        event_times,
        iei,
        window_sec=120.0,
    )

    assert out["raw_r"] > 0.6
    assert out["detrended_r"] < out["raw_r"] - 0.1
    assert out["detrend_fraction"] > 0.1
    assert out["n_valid_pairs"] > 100


def test_compute_within_block_serial_corr_ignores_cross_block_gap() -> None:
    iei1 = _ar1_iei(120, phi=0.82, sigma=0.18, seed=11)
    iei2 = _ar1_iei(100, phi=0.78, sigma=0.18, seed=12)

    starts1 = np.concatenate([[0.0], np.cumsum(iei1)])
    starts2 = 1000.0 + np.concatenate([[0.0], np.cumsum(iei2)])
    events = np.vstack([
        _events_from_starts(starts1),
        _events_from_starts(starts2),
    ])
    block_ranges = [
        (0.0, float(starts1[-1] + 1.0)),
        (1000.0, float(starts2[-1] + 1.0)),
    ]

    out = compute_within_block_serial_corr(events, block_ranges)

    assert out["n_blocks_total"] == 2
    assert out["n_blocks_with_events"] == 2
    assert out["n_blocks_used"] == 2
    assert len(out["per_block"]) == 2
    assert out["pooled_n_pairs"] == (len(iei1) - 1) + (len(iei2) - 1)
    assert out["pooled_r"] > 0.3


def test_compute_serial_corr_soz_stratified_splits_group_events(tmp_path: Path) -> None:
    subject_dir = tmp_path / "demo" / "all_recs"
    subject_dir.mkdir(parents=True)

    iei_soz = _ar1_iei(70, phi=0.85, sigma=0.12, seed=21)
    iei_nsoz = _ar1_iei(70, phi=0.75, sigma=0.15, seed=22)
    starts_soz = np.concatenate([[0.0], np.cumsum(iei_soz)])
    starts_nsoz = 400.0 + np.concatenate([[0.0], np.cumsum(iei_nsoz)])
    starts = np.concatenate([starts_soz, starts_nsoz])

    packed = _events_from_starts(starts, dur=0.10)
    lag_raw = np.zeros((3, packed.shape[0]), dtype=float)
    events_bool = np.zeros((3, packed.shape[0]), dtype=int)
    events_bool[0, :starts_soz.size] = 1
    events_bool[1, starts_soz.size:] = 1
    chn_names = np.array(["A1", "B1", "C1"], dtype=object)

    np.save(subject_dir / "block_packedTimes.npy", packed)
    np.savez_compressed(
        subject_dir / "block_lagPat.npz",
        lagPatRaw=lag_raw,
        eventsBool=events_bool,
        chnNames=chn_names,
        start_t=np.array([0.0], dtype=float),
    )

    out = compute_serial_corr_soz_stratified(
        subject_dir=subject_dir,
        dataset="epilepsiae",
        soz_channels=["A1"],
        block_ranges=[(0.0, 1200.0)],
    )

    assert out["n_soz_channels_matched"] == 1
    assert out["all"]["n_events"] == starts.size
    assert out["soz"]["n_events"] == starts_soz.size
    assert out["nonsoz"]["n_events"] == starts_nsoz.size
    assert np.isfinite(out["soz"]["lag1_r"])
    assert np.isfinite(out["nonsoz"]["lag1_r"])


# -----------------------------------------------------------------------
# PR-2.5: _compute_half_life
# -----------------------------------------------------------------------

def test_compute_half_life_exact() -> None:
    lags = [1, 2, 3, 4, 5]
    rs = np.array([1.0, 0.7, 0.5, 0.3, 0.1])
    hl = _compute_half_life(lags, rs)
    assert hl == 3.0, f"Expected 3.0, got {hl}"


def test_compute_half_life_interpolation() -> None:
    lags = [1, 2, 3]
    rs = np.array([1.0, 0.8, 0.2])
    hl = _compute_half_life(lags, rs)
    assert 2.0 < hl < 3.0, f"Expected between 2 and 3, got {hl}"


def test_compute_half_life_never_reached() -> None:
    lags = [1, 2, 3]
    rs = np.array([1.0, 0.9, 0.8])
    hl = _compute_half_life(lags, rs)
    assert np.isnan(hl)


def test_compute_half_life_empty() -> None:
    assert np.isnan(_compute_half_life([], np.array([])))


def test_compute_half_life_negative_start() -> None:
    lags = [1, 2]
    rs = np.array([-0.3, -0.5])
    assert np.isnan(_compute_half_life(lags, rs))


# -----------------------------------------------------------------------
# PR-2.5: merge_contiguous_blocks
# -----------------------------------------------------------------------

def test_merge_contiguous_blocks_basic() -> None:
    blocks = [(0.0, 100.0), (100.5, 200.0), (201.0, 300.0)]
    merged = merge_contiguous_blocks(blocks, max_gap_sec=5.0)
    assert len(merged) == 1

    merged2 = merge_contiguous_blocks(blocks, max_gap_sec=0.3)
    assert len(merged2) == 3


def test_merge_contiguous_blocks_overlap() -> None:
    blocks = [(0.0, 110.0), (100.0, 200.0)]
    merged = merge_contiguous_blocks(blocks, max_gap_sec=0.0)
    assert len(merged) == 1
    assert merged[0] == (0.0, 200.0)


def test_merge_contiguous_blocks_empty() -> None:
    assert merge_contiguous_blocks([]) == []


def test_merge_contiguous_blocks_single() -> None:
    assert merge_contiguous_blocks([(5.0, 10.0)]) == [(5.0, 10.0)]


def test_merge_contiguous_blocks_large_gap() -> None:
    blocks = [(0.0, 100.0), (500.0, 600.0)]
    merged = merge_contiguous_blocks(blocks, max_gap_sec=5.0)
    assert len(merged) == 2


# -----------------------------------------------------------------------
# PR-2.5: _epoch_to_hour
# -----------------------------------------------------------------------

def test_epoch_to_hour_berlin_noon() -> None:
    import datetime
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("Europe/Berlin")
    except ImportError:
        import pytz
        tz = pytz.timezone("Europe/Berlin")
    dt = datetime.datetime(2024, 7, 15, 12, 30, 0, tzinfo=tz)
    epoch = dt.timestamp()
    h = _epoch_to_hour(epoch, "Europe/Berlin")
    assert 12.4 < h < 12.6, f"Expected ~12.5, got {h}"


def test_epoch_to_hour_shanghai_midnight() -> None:
    import datetime
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("Asia/Shanghai")
    except ImportError:
        import pytz
        tz = pytz.timezone("Asia/Shanghai")
    dt = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=tz)
    epoch = dt.timestamp()
    h = _epoch_to_hour(epoch, "Asia/Shanghai")
    assert h < 0.01, f"Expected ~0.0, got {h}"


# -----------------------------------------------------------------------
# PR-2.5: compute_multiscale_detrend_fraction
# -----------------------------------------------------------------------

def test_multiscale_detrend_fraction_removes_modulation() -> None:
    rng = np.random.default_rng(42)
    slow = 1.5 + 0.6 * np.sin(np.linspace(0, 6 * np.pi, 2000))
    iei = slow * np.exp(0.05 * rng.normal(size=2000))
    event_times = np.concatenate([[0.0], np.cumsum(iei)])

    out = compute_multiscale_detrend_fraction(
        event_times, iei, windows=(30, 90, 300),
    )

    assert out["raw_r"] > 0.3
    assert len(out["per_window"]) == 3
    assert len(out["delta_frac"]) == 2

    fracs = [pw["detrend_fraction"] for pw in out["per_window"]]
    assert all(np.isfinite(f) for f in fracs)
    assert all(f > 0 for f in fracs), "all windows should remove some modulation"


def test_multiscale_detrend_fraction_structure() -> None:
    iei = _ar1_iei(300, phi=0.5, sigma=0.2, seed=10)
    event_times = np.concatenate([[0.0], np.cumsum(iei)])

    out = compute_multiscale_detrend_fraction(
        event_times, iei, windows=(30, 90, 300),
    )

    assert "raw_r" in out
    assert "windows" in out
    assert "per_window" in out
    assert "delta_frac" in out

    for pw in out["per_window"]:
        assert "window_sec" in pw
        assert "detrended_r" in pw
        assert "detrend_fraction" in pw
        assert "n_valid_pairs" in pw

    for df in out["delta_frac"]:
        assert "midpoint_sec" in df
        assert "delta_frac" in df


# -----------------------------------------------------------------------
# PR-2.5: compute_nparticipating_autocorrelation (with tmp_path fixtures)
# -----------------------------------------------------------------------

def test_nparticipating_autocorrelation_basic(tmp_path: Path) -> None:
    subj_dir = tmp_path / "subj" / "all_recs"
    subj_dir.mkdir(parents=True)

    n_events = 200
    rng = np.random.default_rng(99)
    events_bool = np.zeros((4, n_events), dtype=int)
    for j in range(n_events):
        k = rng.integers(1, 5)
        chosen = rng.choice(4, size=k, replace=False)
        events_bool[chosen, j] = 1

    starts = np.sort(rng.uniform(0, 800, size=n_events))
    packed = np.column_stack([starts, starts + 0.05])

    np.save(subj_dir / "block_packedTimes.npy", packed)
    np.savez_compressed(
        subj_dir / "block_lagPat.npz",
        lagPatRaw=np.zeros((4, n_events)),
        eventsBool=events_bool,
        chnNames=np.array(["A", "B", "C", "D"], dtype=object),
        start_t=np.array([0.0]),
    )

    out = compute_nparticipating_autocorrelation(
        subject_dir=subj_dir,
        dataset="epilepsiae",
        block_ranges=[(0.0, 900.0)],
        max_lag=10,
        min_pairs=20,
    )

    assert out["n_events"] == n_events
    assert out["n_blocks_used"] == 1
    assert len(out["lags"]) > 0
    assert len(out["rs"]) == len(out["lags"])
    assert np.isfinite(out["lag1_r"])
    assert np.isfinite(out["half_life_lag"]) or np.isnan(out["half_life_lag"])


def test_nparticipating_autocorrelation_no_data(tmp_path: Path) -> None:
    subj_dir = tmp_path / "empty" / "all_recs"
    subj_dir.mkdir(parents=True)

    out = compute_nparticipating_autocorrelation(
        subject_dir=subj_dir,
        dataset="epilepsiae",
        block_ranges=[(0.0, 100.0)],
    )
    assert out.get("warning") == "no_data"


# -----------------------------------------------------------------------
# PR-2.5: compute_daynight_stratified_detrending
# -----------------------------------------------------------------------

def test_daynight_detrending_splits_correctly() -> None:
    import datetime
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("Europe/Berlin")
    except ImportError:
        import pytz
        tz = pytz.timezone("Europe/Berlin")

    day_start = datetime.datetime(2024, 6, 15, 10, 0, 0, tzinfo=tz)
    night_start = datetime.datetime(2024, 6, 15, 22, 0, 0, tzinfo=tz)

    rng = np.random.default_rng(77)
    n_day, n_night = 150, 100

    day_epoch = day_start.timestamp()
    night_epoch = night_start.timestamp()

    day_iei = np.abs(rng.normal(1.0, 0.2, size=n_day))
    night_iei = np.abs(rng.normal(1.5, 0.3, size=n_night))

    day_starts = day_epoch + np.concatenate([[0], np.cumsum(day_iei[:-1])])
    night_starts = night_epoch + np.concatenate([[0], np.cumsum(night_iei[:-1])])

    all_starts = np.concatenate([day_starts, night_starts])
    all_iei = np.concatenate([day_iei, night_iei])
    order = np.argsort(all_starts)
    all_starts = all_starts[order]
    all_iei = all_iei[order]

    out = compute_daynight_stratified_detrending(
        event_times=all_starts,
        iei=all_iei,
        dataset="epilepsiae",
        window_sec=30.0,
    )

    assert out["timezone"] == "Europe/Berlin"
    assert out["n_day"] > 50
    assert out["n_night"] > 50
    assert out["n_day"] + out["n_night"] == n_day + n_night

    for key in ("day", "night", "combined"):
        seg = out[key]
        assert "n_iei" in seg
        assert "raw_r" in seg
        assert "detrended_r" in seg


def test_daynight_detrending_too_few_events() -> None:
    event_times = np.array([0.0, 1.0, 2.0])
    iei = np.array([1.0, 1.0, 1.0])

    out = compute_daynight_stratified_detrending(
        event_times, iei, dataset="yuquan", window_sec=10.0,
    )

    assert out["timezone"] == "Asia/Shanghai"
    assert out["day"]["n_iei"] + out["night"]["n_iei"] <= 3
