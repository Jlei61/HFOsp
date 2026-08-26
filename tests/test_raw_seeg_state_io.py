"""Worker B IO tests for the Raw-SEEG evolvable-state model (R0.1).

Everything here runs on synthetic data written in the Epilepsiae ``.data/.head``
format, so the whole chain -- native reader, bipolar montage, filter, decimation,
int16 quantisation, Zarr layout, Welch target, train-only statistics, artifact
mask, torch Dataset -- is exercised end to end without touching either mount.

The tests that matter most are the boring ones: minute alignment (1), channel
order (2) and filled minutes (12).  Those three are hard-invalidity conditions
in the scientific spec; a silent failure there contaminates every downstream
number.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.topic5_raw_seeg_state import contract
from src.topic5_raw_seeg_state import raw_cache as rc
from src.topic5_raw_seeg_state import spectral_target as st
from src.topic5_raw_seeg_state import windows as wd

FS_FAST = 256
CONVERSION_FACTOR = 0.179          # same magnitude as the real Epilepsiae heads


# --------------------------------------------------------------------------
# synthetic Epilepsiae subject
# --------------------------------------------------------------------------


def _write_block(dirpath: Path, stem: str, uv: np.ndarray, fs: float, start_epoch: float) -> Path:
    """``uv`` is (n_samples, n_channels) in microvolts; writes .data + .head."""
    n_samples, n_ch = uv.shape
    conversion = -1.0 * CONVERSION_FACTOR
    ints = np.clip(np.rint(uv / conversion), -32768, 32767).astype("<i2")
    data_path = dirpath / f"{stem}.data"
    ints.tofile(data_path)
    names = ",".join(f"C{i}" for i in range(n_ch))
    (dirpath / f"{stem}.head").write_text(
        "\n".join([
            f"elec_names=[{names}]",
            f"num_channels={n_ch}",
            f"sample_freq={fs}",
            f"duration_in_sec={n_samples / fs}",
            f"num_samples={n_samples}",
            "sample_bytes=2",
            f"conversion_factor={CONVERSION_FACTOR}",
        ]) + "\n"
    )
    return data_path


def make_subject(
    tmp_path: Path,
    signal_fn,
    *,
    subject: str = "synth_a",
    n_native: int = 4,
    fs: float = FS_FAST,
    minutes_per_block: int = 3,
    n_blocks: int = 1,
    start_epoch: float = 1_000_000_000.0,
    native_perm=None,
    train_minutes=None,
    covered=None,
    horizons=(1,),
):
    """Build blocks + the three Worker-A frames for one synthetic subject.

    ``signal_fn(t_sec, native_channel)`` returns microvolts; ``t_sec`` is
    seconds since the subject's first block start.  ``native_perm[j]`` gives the
    position of logical channel ``j`` inside the native file, so a test can
    scramble the on-disk order and check the cache still follows
    ``contact_metadata.channel_index``.
    """
    import pandas as pd

    src = tmp_path / "src"
    src.mkdir(parents=True, exist_ok=True)
    perm = np.arange(n_native) if native_perm is None else np.asarray(native_perm, dtype=int)
    n_min = minutes_per_block * n_blocks
    spb = int(round(minutes_per_block * 60 * fs))

    manifest_rows = []
    for b in range(n_blocks):
        t0 = b * minutes_per_block * 60.0
        t = t0 + np.arange(spb) / fs
        uv = np.empty((spb, n_native), dtype=np.float64)
        for j in range(n_native):
            uv[:, perm[j]] = signal_fn(t, j)
        stem = f"blk{b:02d}"
        path = _write_block(src, stem, uv, fs, start_epoch + t0)
        manifest_rows.append({
            "subject": subject, "dataset": "synth", "session_id": 0, "block_id": stem,
            "block_start_epoch": start_epoch + t0,
            "block_end_epoch": start_epoch + t0 + spb / fs,
            "duration_sec": spb / fs, "native_sampling_rate": float(fs),
            "n_channels_native": n_native, "source_path": str(path),
            "source_kind": "epilepsiae", "gap_to_prev_sec": 0.0,
            "opens_session": b == 0, "split": "train",
        })
    manifest = pd.DataFrame(manifest_rows, columns=list(contract.DATASET_MANIFEST_COLUMNS))

    contacts = []
    for ci in range(n_native - 1):
        contacts.append({
            "subject": subject, "dataset": "synth", "channel_index": ci,
            "channel_name": f"A{ci + 1}-A{ci + 2}", "anode": f"A{ci + 1}",
            "cathode": f"A{ci + 2}", "shaft": "A", "shaft_index": ci,
            "x_mm": float(ci), "y_mm": 0.0, "z_mm": 0.0, "coord_space": "native",
            "coord_valid": True, "native_index_anode": int(perm[ci]),
            "native_index_cathode": int(perm[ci + 1]), "contact_valid": True,
            "drop_reason": "", "coord_mode": contract.COORD_MODE_FULL,
        })
    contact_df = pd.DataFrame(contacts, columns=list(contract.CONTACT_METADATA_COLUMNS))

    mi = np.arange(n_min)
    starts = start_epoch + mi * 60.0
    cov = np.ones(n_min, dtype=bool) if covered is None else np.asarray(covered, dtype=bool)
    n_train = n_min if train_minutes is None else int(train_minutes)
    split = np.where(mi < n_train, "train", "validation")
    usable = cov.copy()
    ctx_ok = np.array([
        bool(m - contract.CONTEXT_MINUTES + 1 >= 0
             and usable[max(0, m - contract.CONTEXT_MINUTES + 1):m + 1].all()
             and len(set(split[max(0, m - contract.CONTEXT_MINUTES + 1):m + 1])) == 1)
        for m in mi
    ])
    wi = {
        "subject": subject, "minute_index": mi, "minute_start_epoch": starts,
        "session_id": 0, "split": split, "covered": cov, "guard_free": True,
        "n_valid_contacts": n_native - 1, "minute_usable": usable, "ctx_ok": ctx_ok,
    }
    for h in (1, 5, 10, 100):
        tt = mi + h
        ok = (tt < n_min)
        okv = np.zeros(n_min, dtype=bool)
        okv[ok] = usable[tt[ok]] & (split[tt[ok]] == split[mi[ok]])
        wi[f"h{h}_ok"] = ctx_ok & okv
    window_index = pd.DataFrame(wi, columns=list(contract.WINDOW_INDEX_COLUMNS))

    return {
        "subject": subject, "manifest": manifest, "contacts": contact_df,
        "window_index": window_index, "n_minutes": n_min, "fs": fs,
        "start_epoch": start_epoch,
        "train_end_epoch": start_epoch + n_train * 60.0,
        "dev_end_epoch": start_epoch + n_min * 60.0,
        "n_contacts": n_native - 1,
    }


def build_cache(spec, out_dir: Path, **kw):
    return rc.build_subject_cache(
        spec["subject"], spec["manifest"], spec["contacts"], spec["window_index"],
        out_path=out_dir / "raw_256hz.zarr",
        train_end_epoch=spec["train_end_epoch"], dev_end_epoch=spec["dev_end_epoch"],
        cache_cap=kw.pop("cache_cap", False), **kw,
    )


def read_minute_uv(out_dir: Path, minute: int) -> np.ndarray:
    """(MINUTE_SAMPLES, C) microvolts straight out of the cache."""
    import zarr

    arr = zarr.open_array(str(out_dir / "raw_256hz.zarr"), mode="r")
    scale = np.asarray(arr.attrs["contact_scale_uv"], dtype=np.float32)
    lo = minute * contract.MINUTE_SAMPLES
    return np.asarray(arr[lo:lo + contract.MINUTE_SAMPLES, :], dtype=np.float32) * scale


def peak_freq(x: np.ndarray, fs: float = contract.ANALYSIS_RATE_HZ) -> float:
    from scipy.signal import welch

    f, p = welch(x, fs=fs, nperseg=min(2048, len(x)))
    return float(f[int(np.argmax(p))])


# --------------------------------------------------------------------------
# 1. minute alignment
# --------------------------------------------------------------------------


def test_minute_alignment_is_exact(tmp_path):
    """Minute m of the cache must be minute m of the recording, exactly.

    A DC ramp cannot be used as the marker because the contract mandates a
    0.5 Hz high-pass that removes it, so each minute instead carries its own
    frequency (10 + 3*m Hz) and its own amplitude (100 * (1 + m)).  Both must
    land in the right 15360-sample slot.
    """
    def sig(t, ch):
        m = np.floor(t / 60.0).astype(int)
        return (100.0 * (1 + m)) * np.sin(2 * np.pi * (10.0 + 3.0 * m) * t) * (1 + 0.1 * ch)

    spec = make_subject(tmp_path, sig, minutes_per_block=6, n_blocks=1)
    out = tmp_path / "cache"
    build_cache(spec, out)
    for m in range(6):
        x = read_minute_uv(out, m)[:, 0]
        core = x[contract.ANALYSIS_RATE_HZ:-contract.ANALYSIS_RATE_HZ]
        assert abs(peak_freq(core) - (10.0 + 3.0 * m)) < 0.5, f"minute {m} frequency"
        amp = np.sqrt(2.0) * core.std()
        want = 100.0 * (1 + m) * abs(1.0 - 1.1)   # bipolar A(m) - A(m+1)
        assert abs(amp - want) / want < 0.05, f"minute {m} amplitude {amp} vs {want}"


# --------------------------------------------------------------------------
# 2. channel order
# --------------------------------------------------------------------------


def test_cache_columns_follow_channel_index_not_native_order(tmp_path):
    """Hard-invalidity condition #4: cache column order == contact_metadata."""
    def sig(t, ch):
        return 100.0 * (ch + 1) * np.sin(2 * np.pi * 20.0 * t)

    perm = [3, 0, 2, 1]      # logical channel j lives at native position perm[j]
    spec = make_subject(tmp_path, sig, n_native=4, native_perm=perm, minutes_per_block=2)
    out = tmp_path / "cache"
    build_cache(spec, out)
    x = read_minute_uv(out, 1)
    core = x[contract.ANALYSIS_RATE_HZ:-contract.ANALYSIS_RATE_HZ, :]
    amp = np.sqrt(2.0) * core.std(axis=0)
    want = np.array([100.0 * abs((j + 1) - (j + 2)) for j in range(3)])
    assert np.allclose(amp, want, rtol=0.05), f"{amp} != {want}"
    import zarr
    arr = zarr.open_array(str(out / "raw_256hz.zarr"), mode="r")
    assert list(arr.attrs["channel_names"]) == ["A1-A2", "A2-A3", "A3-A4"]


# --------------------------------------------------------------------------
# 3. decimation / anti-aliasing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fs", [2000, 1024, 512, 256])
def test_decimation_preserves_in_band_and_kills_out_of_band(fs):
    """40 Hz survives at the right frequency; 300 Hz does not alias back in."""
    secs = 70.0
    n = int(secs * fs)
    n -= n % rc.native_alignment(fs)
    t = np.arange(n) / fs
    sos = rc.design_prefilter(fs)

    y40 = rc.process_native_segment(
        (100.0 * np.sin(2 * np.pi * 40.0 * t))[None, :], fs, sos
    )[0]
    assert y40.size == n * contract.ANALYSIS_RATE_HZ // fs - 2 * rc.pad_analysis_samples(fs)
    assert abs(peak_freq(y40) - 40.0) < 0.5
    assert abs(np.sqrt(2.0) * y40.std() - 100.0) / 100.0 < 0.05

    if fs / 2 > 300.0:
        y300 = rc.process_native_segment(
            (100.0 * np.sin(2 * np.pi * 300.0 * t))[None, :], fs, sos
        )[0]
        att_db = 10.0 * np.log10(np.var(y300) / np.var(y40))
        assert att_db < -40.0, f"{fs} Hz: 300 Hz only attenuated {att_db:.1f} dB"


# --------------------------------------------------------------------------
# 4. filter-edge padding at a block boundary
# --------------------------------------------------------------------------


def test_block_boundary_leaves_no_visible_transient(tmp_path):
    """A sine spanning two blocks must not dent the boundary minutes' power."""
    def sig(t, ch):
        return 100.0 * (1 + 0.3 * ch) * np.sin(2 * np.pi * 20.0 * t)

    spec = make_subject(tmp_path, sig, minutes_per_block=3, n_blocks=2)
    out = tmp_path / "cache"
    build_cache(spec, out)
    power = []
    for m in range(6):
        x = read_minute_uv(out, m)[:, 0]
        power.append(float(np.var(x)))
    interior = power[1]
    for m in (2, 3):    # last minute of block 0, first minute of block 1
        rel = abs(power[m] - interior) / interior
        assert rel < 0.05, f"minute {m} power differs from interior by {rel:.3%}"


# --------------------------------------------------------------------------
# 5. band definition
# --------------------------------------------------------------------------


def test_band_power_lands_in_the_bands_that_overlap_the_signal():
    """8-12 Hz power must sit in the bands covering 8-12 Hz, not elsewhere."""
    from scipy.signal import butter, sosfiltfilt

    rng = np.random.default_rng(0)
    fs = contract.ANALYSIS_RATE_HZ
    x = rng.standard_normal((contract.MINUTE_SAMPLES, 2))
    sos = butter(8, [8.0 / (fs / 2), 12.0 / (fs / 2)], btype="bandpass", output="sos")
    x = sosfiltfilt(sos, x, axis=0) * 1000.0
    x = x + 1e-3 * rng.standard_normal(x.shape)
    field = st.minute_spectral_field(x)
    power = 10.0 ** field
    lo, hi = contract.FREQ_EDGES[:-1], contract.FREQ_EDGES[1:]
    overlap = (hi > 8.0) & (lo < 12.0)
    frac = power[:, overlap].sum(axis=1) / power.sum(axis=1)
    assert (frac > 0.90).all(), f"only {frac} of band power in the 8-12 Hz bands"


# --------------------------------------------------------------------------
# 6. line-noise exclusion
# --------------------------------------------------------------------------


def test_line_noise_does_not_dominate_the_band_that_contains_it():
    """Band 10 is 46.4-68.1 Hz and straddles 50 Hz; notch + bin exclusion must
    keep a large 50 Hz tone from lifting it."""
    rng = np.random.default_rng(1)
    fs = 1024
    secs = 70.0
    n = int(secs * fs)
    n -= n % rc.native_alignment(fs)
    t = np.arange(n) / fs
    base = 50.0 * rng.standard_normal(n)
    sos = rc.design_prefilter(fs)

    clean = rc.process_native_segment(base[None, :], fs, sos)[0]
    dirty = rc.process_native_segment((base + 1000.0 * np.sin(2 * np.pi * 50.0 * t))[None, :], fs, sos)[0]
    keep = slice(0, contract.MINUTE_SAMPLES)
    f_clean = st.minute_spectral_field(clean[keep][:, None])[0]
    f_dirty = st.minute_spectral_field(dirty[keep][:, None])[0]
    band10 = 10
    assert contract.FREQ_EDGES[band10] < 50.0 < contract.FREQ_EDGES[band10 + 1]
    rise = float(f_dirty[band10] - f_clean[band10])
    assert rise < 0.5, f"band 10 log power rose by {rise:.3f} because of 50 Hz"


# --------------------------------------------------------------------------
# 7. train-only normalisation
# --------------------------------------------------------------------------


def _numeric_stats(stats):
    drop = {"code_revision", "subject", "n_validation_minutes"}
    return {k: v for k, v in stats.items() if k not in drop}


def test_train_stats_ignore_the_validation_half(tmp_path):
    """Hard-invalidity condition #5: scaling validation x10 must change nothing."""
    def make(gain):
        rng = np.random.default_rng(7)
        noise = rng.standard_normal((int(14 * 60 * FS_FAST) + 10, 4)) * 60.0

        def sig(t, ch):
            idx = np.rint(t * FS_FAST).astype(int)
            g = np.where(t >= 10 * 60.0, gain, 1.0)
            return noise[idx, ch] * g * (1 + 0.2 * ch)
        return sig

    stats = {}
    for gain, tag in ((1.0, "plain"), (10.0, "loud")):
        spec = make_subject(
            tmp_path / tag, make(gain), n_native=4, minutes_per_block=14, train_minutes=10
        )
        out = tmp_path / tag / "cache"
        build_cache(spec, out)
        st.build_subject_targets(spec["subject"], out / "raw_256hz.zarr",
                                 out / "spectral_target.zarr")
        stats[tag] = st.compute_train_stats(
            spec["subject"], out / "raw_256hz.zarr", out / "spectral_target.zarr",
            out / "train_stats.json",
        )
    a, b = _numeric_stats(stats["plain"]), _numeric_stats(stats["loud"])
    assert set(a) == set(b)
    for k in a:
        assert a[k] == b[k], f"train statistic {k!r} depends on the validation half"

    # non-vacuity: the x10 really did reach the validation minutes
    import zarr
    va = np.asarray(zarr.open_array(str(tmp_path / "plain/cache/spectral_target.zarr"),
                                    mode="r")[10:14, :, :])
    vb = np.asarray(zarr.open_array(str(tmp_path / "loud/cache/spectral_target.zarr"),
                                    mode="r")[10:14, :, :])
    assert np.nanmax(np.abs(vb - va)) > 0.5, "the validation half was not actually louder"


# --------------------------------------------------------------------------
# 8. sealed gate
# --------------------------------------------------------------------------


def test_sealed_minutes_are_excluded_and_the_gate_still_fires(tmp_path):
    """Two layers: the selector drops sealed minutes, the gate refuses them."""
    def sig(t, ch):
        return 100.0 * np.sin(2 * np.pi * 20.0 * t) * (1 + 0.1 * ch)

    spec = make_subject(tmp_path, sig, minutes_per_block=4)
    out = tmp_path / "cache"
    # layer 1: a dev_end that lands 90 s in keeps only the minute that ends before it
    rc.build_subject_cache(
        spec["subject"], spec["manifest"], spec["contacts"], spec["window_index"],
        out_path=out / "raw_256hz.zarr",
        train_end_epoch=spec["train_end_epoch"],
        dev_end_epoch=spec["start_epoch"] + 90.0,
        cache_cap=False,
    )
    import pandas as pd
    ci = pd.read_parquet(out / "cache_index.parquet")
    # cache_index is truncated at the sealed bound so no artifact of the build
    # carries a sealed timestamp; only minute 0 ends before start+90 s.
    assert ci["minute_index"].tolist() == [0]
    assert ci["cached"].tolist() == [True]
    assert float(ci["minute_start_epoch"].max()) < spec["start_epoch"] + 90.0
    import numpy as _np
    assert _np.load(out / "minute_filled.npy").tolist() == [False, True, True, True]

    # layer 2: if anything sealed ever reaches the writer, the gate raises
    with pytest.raises(ValueError, match="SEALED-PARTITION VIOLATION"):
        rc._assert_not_sealed("synth_a", np.array([1_000_000_180.0]), 1_000_000_090.0)
    rc._assert_not_sealed("synth_a", np.array([1_000_000_000.0]), 1_000_000_090.0)


def test_real_subject_must_use_the_frozen_dev_end_epoch():
    """A cohort subject may not be cached against a hand-supplied sealed bound."""
    subject = contract.cohort_subjects()[0]
    frozen = contract.dev_end_epoch(subject)
    rc._assert_not_sealed(subject, np.array([frozen - 3600.0]), frozen)
    with pytest.raises(ValueError, match="disagrees with the frozen"):
        rc._assert_not_sealed(subject, np.array([frozen - 3600.0]), frozen + 7200.0)


def test_select_cached_minutes_respects_the_caps():
    grid = rc.MinuteGrid(
        minute_index=np.arange(600),
        minute_start_epoch=1e9 + np.arange(600) * 60.0,
        covered=np.ones(600, dtype=bool),
        split=np.array(["train"] * 400 + ["validation"] * 200, dtype=object),
    )
    keep = rc.select_cached_minutes(
        grid, 1e9 + 400 * 60.0, 1e9 + 600 * 60.0,
        cache_cap=True, train_hours_cap=1.0, val_hours_cap=0.5,
    )
    assert keep.sum() == 90
    assert np.flatnonzero(keep[:400]).tolist() == list(range(340, 400))   # most recent hour
    assert np.flatnonzero(keep[400:]).tolist() == list(range(170, 200))


# --------------------------------------------------------------------------
# 9-12. the training reader
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset_fixture(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("ds")
    rng = np.random.default_rng(3)
    n_min = 26
    noise = rng.standard_normal((int(n_min * 60 * FS_FAST) + 10, 5)) * 80.0

    def sig(t, ch):
        idx = np.rint(t * FS_FAST).astype(int)
        return noise[idx, ch] * (1 + 0.2 * ch) + 30.0 * np.sin(2 * np.pi * 12.0 * t)

    covered = np.ones(n_min, dtype=bool)
    covered[20] = False           # one never-recorded minute in the middle
    spec = make_subject(
        tmp_path, sig, n_native=5, minutes_per_block=n_min, train_minutes=n_min,
        covered=covered,
    )
    out = tmp_path / "cache"
    build_cache(spec, out)
    st.build_subject_targets(spec["subject"], out / "raw_256hz.zarr", out / "spectral_target.zarr")
    st.compute_train_stats(spec["subject"], out / "raw_256hz.zarr",
                           out / "spectral_target.zarr", out / "train_stats.json")
    st.artifact_mask(spec["subject"], out / "raw_256hz.zarr", out / "train_stats.json",
                     out / "artifact_mask.zarr")
    spec["out"] = out
    return spec


def _make_dataset(spec, **kw):
    out = spec["out"]
    return wd.SubjectWindowDataset(
        spec["subject"], "train", spec["window_index"], spec["contacts"],
        cache_path=out / "raw_256hz.zarr", target_path=out / "spectral_target.zarr",
        mask_path=out / "artifact_mask.zarr", stats_path=out / "train_stats.json",
        cache_index_path=out / "cache_index.parquet", **kw,
    )


def test_item_exposes_exactly_the_allowed_encoder_inputs(dataset_fixture):
    ds = _make_dataset(dataset_fixture, horizons=(1, 5))
    assert len(ds) > 0
    item = ds[0]
    contract.assert_no_forbidden_inputs({k: item[k] for k in contract.ALLOWED_INPUT_KEYS})
    ds.encoder_inputs(item)
    assert set(item) == set(contract.ALLOWED_INPUT_KEYS) | set(wd.TARGET_KEYS)
    C = dataset_fixture["n_contacts"]
    assert item["raw"].shape == (C, contract.CONTEXT_MINUTES * contract.MINUTE_SAMPLES)
    assert item["coords_mm"].shape == (C, 3)
    assert item["coord_valid"].shape == (C,)
    assert item["shaft_id"].shape == (C,)
    assert item["shaft_index"].shape == (C,)
    assert item["contact_valid"].shape == (C,)
    assert item["minute_valid"].shape == (C, contract.CONTEXT_MINUTES)
    assert item["persistence"].shape == (C, contract.N_FREQ_BINS)
    for h in (1, 5):
        assert item["target"][str(h)].shape == (C, contract.N_FREQ_BINS)


def test_consistency_keys_only_when_requested(dataset_fixture):
    plain = _make_dataset(dataset_fixture, horizons=(1,))
    assert not (set(wd.CONSISTENCY_KEYS) & set(plain[0]))
    cons = _make_dataset(dataset_fixture, horizons=(1,), need_consistency=True)
    item = cons[0]
    assert set(item) >= set(wd.CONSISTENCY_KEYS)
    assert item["raw_next"].shape == item["raw"].shape
    # raw_next is the context shifted by exactly one minute
    ms = contract.MINUTE_SAMPLES
    assert np.allclose(item["raw"][:, ms:].numpy(), item["raw_next"][:, :-ms].numpy())


def test_context_and_target_times_are_ordered(dataset_fixture):
    ds = _make_dataset(dataset_fixture, horizons=(1, 5))
    for i in (0, len(ds) // 2, len(ds) - 1):
        item = ds[i]
        t_epoch = item["t_epoch"]
        for h in (1, 5):
            assert item["target_epoch"][str(h)] == pytest.approx(t_epoch + 60.0 * h)
        ctx = wd.context_range(item["t_index"], ds.context_convention)
        ctx_start = t_epoch - 60.0 * (contract.CONTEXT_MINUTES - 1)
        assert ds._minute_start[ctx[0]] == pytest.approx(ctx_start)
        # the context ends at origin_epoch, strictly before every target minute
        assert item["origin_epoch"] == pytest.approx(t_epoch + 60.0)
        for h in (1, 5):
            assert item["origin_epoch"] <= item["target_epoch"][str(h)]
            assert ctx.max() < item["t_index"] + h


def test_dataloader_workers_match_single_process(dataset_fixture):
    from torch.utils.data import DataLoader

    ds = _make_dataset(dataset_fixture, horizons=(1,))
    common = dict(batch_size=2, shuffle=False, collate_fn=wd.collate_windows)
    a = list(DataLoader(ds, num_workers=0, **common))
    b = list(DataLoader(ds, num_workers=4, worker_init_fn=wd.worker_init_fn, **common))
    assert len(a) == len(b) > 0
    for ba, bb in zip(a, b):
        for k in ("raw", "coords_mm", "coord_valid", "shaft_id", "shaft_index",
                  "contact_valid", "minute_valid", "persistence"):
            assert np.array_equal(ba[k].numpy(), bb[k].numpy()), k
        assert np.array_equal(ba["target"]["1"].numpy(), bb["target"]["1"].numpy())
        assert np.array_equal(ba["target_mask"]["1"].numpy(), bb["target_mask"]["1"].numpy())
        assert ba["t_index"].tolist() == bb["t_index"].tolist()


def test_uncovered_minutes_never_enter_an_item(dataset_fixture):
    ds = _make_dataset(dataset_fixture, horizons=(1,))
    filled = np.load(dataset_fixture["out"] / "minute_filled.npy")
    assert filled[20], "fixture must contain a filled minute"
    touched = set()
    for i in range(len(ds)):
        t = int(ds._index[i])
        touched.update(int(m) for m in wd.context_range(t, ds.context_convention))
        touched.add(t + 1)
    assert not (touched & set(np.flatnonzero(filled).tolist())), (
        "a filled minute reached an emitted item"
    )
    assert len(ds) > 0


def test_require_all_horizons_toggle(dataset_fixture):
    strict = _make_dataset(dataset_fixture, horizons=(1, 10), require_all_horizons=True)
    loose = _make_dataset(dataset_fixture, horizons=(1, 10), require_all_horizons=False)
    # train defaults to the per-horizon set (contract.TRAIN_REQUIRE_ALL_HORIZONS)
    assert not wd.default_require_all_horizons("train")
    assert wd.default_require_all_horizons("validation")
    assert len(_make_dataset(dataset_fixture, horizons=(1, 10))) == len(loose)
    assert len(loose) >= len(strict)
    if len(loose) > len(strict):
        extra = sorted(set(loose._index.tolist()) - set(strict._index.tolist()))
        item = loose[loose._index.tolist().index(extra[0])]
        assert not (item["target_mask"]["1"].numpy().all()
                    and item["target_mask"]["10"].numpy().all())


def test_missing_coordinates_do_not_invalidate_the_contact(dataset_fixture):
    """contact_valid and coord_valid are independent axes (contract section 5).

    A contact with no localisation must still carry signal, a shaft id and a
    shaft index; only its mm projection is switched off.  Mean-centring uses the
    coord_valid subset alone, and nothing is imputed.
    """
    c = dataset_fixture["contacts"].copy()
    c.loc[c["channel_index"].isin([0, 1]), ["x_mm", "y_mm", "z_mm"]] = np.nan
    c.loc[c["channel_index"].isin([0, 1]), "coord_valid"] = False
    ds = wd.SubjectWindowDataset(
        dataset_fixture["subject"], "train", dataset_fixture["window_index"], c,
        horizons=(1,), cache_path=dataset_fixture["out"] / "raw_256hz.zarr",
        target_path=dataset_fixture["out"] / "spectral_target.zarr",
        mask_path=dataset_fixture["out"] / "artifact_mask.zarr",
        stats_path=dataset_fixture["out"] / "train_stats.json",
        cache_index_path=dataset_fixture["out"] / "cache_index.parquet",
    )
    item = ds[0]
    cv = item["coord_valid"].numpy()
    assert cv.tolist() == [False, False, True, True]
    assert item["contact_valid"].numpy().all(), "signal validity must be untouched"
    assert np.array_equal(item["coords_mm"].numpy()[~cv], np.zeros((2, 3), dtype=np.float32))
    assert np.allclose(item["coords_mm"].numpy()[cv].mean(axis=0), 0.0, atol=1e-5)
    assert item["shaft_index"].numpy().tolist() == [0, 1, 2, 3]
    assert item["raw"].shape[0] == 4 and np.isfinite(item["raw"].numpy()).all()


def test_eligible_indices_matches_the_dataset_without_touching_zarr(dataset_fixture):
    """Worker D must be able to enumerate both eval sets from parquet alone."""
    out = dataset_fixture["out"]
    for require_all in (True, False):
        idx = wd.eligible_indices(
            dataset_fixture["subject"], "train", dataset_fixture["window_index"],
            horizons=(1, 10), require_all=require_all,
            cache_index_path=out / "cache_index.parquet",
        )
        ds = _make_dataset(dataset_fixture, horizons=(1, 10),
                           require_all_horizons=require_all)
        assert idx.tolist() == ds._index.tolist()
    primary = wd.eligible_indices(
        dataset_fixture["subject"], "train", dataset_fixture["window_index"],
        horizons=(1, 10), cache_index_path=out / "cache_index.parquet")
    assert primary.tolist() == wd.eligible_indices(
        dataset_fixture["subject"], "train", dataset_fixture["window_index"],
        horizons=(1, 10), require_all=False,
        cache_index_path=out / "cache_index.parquet").tolist()


def test_window_field_is_diagnostic_and_shaped(dataset_fixture):
    rng = np.random.default_rng(11)
    x = rng.standard_normal((contract.WINDOW_SAMPLES, 3)) * 50.0
    f = st.window_spectral_field(x)
    assert f.shape == (3, contract.N_FREQ_BINS)
    assert np.isfinite(f).all()
