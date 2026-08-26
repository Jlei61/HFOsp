"""Decimated int16 raw cache for the Raw-SEEG evolvable-state model (R0.1).

Owner: Worker B.  Reads ``contract.py`` constants; never hard-codes them.

What this module does, in plain words
-------------------------------------
The native recordings are 256 / 512 / 1024 / 2000 Hz, spread over hundreds of
files on two mounts, and are far too slow to read inside a training loop (an
EDF open alone costs seconds).  This module walks every recorded block once,
turns it into the bipolar montage the contract asks for, filters and decimates
it to a single common rate, and stores the result as one int16 Zarr array per
subject laid out on an exact one-minute grid:

    raw_256hz.zarr[minute_index * MINUTE_SAMPLES : (minute_index+1) * MINUTE_SAMPLES, :]

is *by construction* the samples of that wall-clock minute, for every minute of
the subject's timeline -- including the minutes that were never recorded, which
are written as zeros and flagged in ``minute_filled`` so no consumer can mistake
"not recorded" for "quiet brain".  That exact alignment is the whole point of
the file; ``tests/test_raw_seeg_state_io.py`` pins it.

Signal path per block (in this order):
    1. read native samples for the contacts we need
    2. bipolar difference (anode - cathode) per ``contact_metadata``
    3. zero-phase 0.5 Hz Butterworth high-pass + IIR notch at each line-noise
       harmonic below the native Nyquist (Q=30), applied as ONE sos cascade
    4. anti-alias + decimate to ``contract.ANALYSIS_RATE_HZ``
    5. int16 quantisation with a per-contact scale estimated on TRAIN only

Every written segment is processed with >= ``PAD_SECONDS`` of real (or
edge-reflected) signal on each side which is then discarded, so no filter edge
transient ever lands inside a kept minute.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, decimate, iirnotch, resample_poly, sosfiltfilt, tf2sos

from . import contract

# --------------------------------------------------------------------------
# Signal-path constants (module-local; not part of the frozen contract)
# --------------------------------------------------------------------------

HIGHPASS_HZ = 0.5
"""Zero-phase Butterworth high-pass corner, order 4 (applied at native rate)."""

HIGHPASS_ORDER = 4
NOTCH_Q = 30.0

PAD_SECONDS = 5.0
"""Filter-edge guard discarded from both ends of every processed segment."""

CHUNK_MINUTES_DEFAULT = 5
"""Minutes decoded per pass.  5 min x 2000 Hz x 145 ch float64 ~= 0.7 GB."""

CALIB_MINUTES_DEFAULT = 40
"""Train minutes decoded twice to calibrate the int16 scale."""

CALIB_STRETCHES = 8
"""Those minutes are drawn as this many CONTIGUOUS stretches (see
_calibrate_int16_scale) so the reads stay sequential on a rotational disk."""

INT16_TARGET_COUNTS = 8000.0
"""6 x train MAD is mapped to this many int16 counts (rail = 32767)."""

INT16_MAX = 32767
INT16_MIN = -32768

FALLBACK_SCALE_UV = 1.0
"""Used when the train MAD of a contact is 0 (dead / disconnected channel)."""


# --------------------------------------------------------------------------
# 1. Signal path primitives
# --------------------------------------------------------------------------


def native_alignment(native_rate: float) -> int:
    """Native samples per output sample-group: lengths must be a multiple of it.

    ``out_len = in_len * ANALYSIS_RATE_HZ // native_rate`` is exact iff
    ``in_len`` is a multiple of the returned value.
    """
    fs = int(round(float(native_rate)))
    if fs <= 0:
        raise ValueError(f"bad native rate {native_rate!r}")
    return fs // gcd(fs, contract.ANALYSIS_RATE_HZ)


def pad_native_samples(native_rate: float) -> int:
    """Filter-edge pad in native samples, rounded up to the alignment grid."""
    fs = int(round(float(native_rate)))
    align = native_alignment(fs)
    raw = int(math.ceil(PAD_SECONDS * fs))
    return int(math.ceil(raw / align) * align)


def pad_analysis_samples(native_rate: float) -> int:
    """The same pad expressed in decimated samples (always exact)."""
    fs = int(round(float(native_rate)))
    return pad_native_samples(fs) * contract.ANALYSIS_RATE_HZ // fs


def design_prefilter(native_rate: float) -> np.ndarray:
    """0.5 Hz high-pass + a notch per line-noise harmonic below native Nyquist.

    Returned as a single second-order-section cascade so one ``sosfiltfilt``
    call does all of it (measured ~3x faster than three separate passes).
    """
    fs = float(native_rate)
    nyq = fs / 2.0
    sections = [butter(HIGHPASS_ORDER, HIGHPASS_HZ / nyq, btype="highpass", output="sos")]
    for f0 in contract.LINE_NOISE_HZ:
        if f0 + contract.LINE_NOISE_HALFWIDTH_HZ < nyq:
            b, a = iirnotch(float(f0), NOTCH_Q, fs)
            sections.append(tf2sos(b, a))
    return np.vstack(sections)


def decimate_to_analysis(x: np.ndarray, native_rate: float) -> np.ndarray:
    """Anti-alias + resample the last axis from ``native_rate`` to 256 Hz.

    ``x`` must already have a length that is a multiple of
    ``native_alignment(native_rate)`` so the output length is exact.

    Measured 2026-08-21 on 90 ch x 300 s @ 2000 Hz (float32, 1 thread):
      single stage ``resample_poly(16, 125)``            -> 1.18 s
      two stage ``decimate(4)`` + ``resample_poly(64,125)`` -> 1.46 s
    The single stage is faster *and* has one filter instead of two, so it wins.
    """
    fs = int(round(float(native_rate)))
    target = contract.ANALYSIS_RATE_HZ
    n = x.shape[-1]
    align = native_alignment(fs)
    if n % align:
        raise ValueError(
            f"segment length {n} is not a multiple of the {fs} Hz alignment {align}; "
            "exact minute alignment would be lost"
        )
    if fs == target:
        return np.ascontiguousarray(x)
    if fs % target == 0:
        q = fs // target
        return decimate(x, q, ftype="fir", zero_phase=True, axis=-1)
    g = gcd(fs, target)
    return resample_poly(x, target // g, fs // g, axis=-1)


def process_native_segment(
    x_padded: np.ndarray,
    native_rate: float,
    sos: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Filter + decimate a padded native segment and strip the pad.

    ``x_padded`` is (C, pad + L + pad) at ``native_rate``; the return is
    (C, L * 256 / native_rate).  The pad is what keeps filter edge transients
    out of the kept region.
    """
    fs = int(round(float(native_rate)))
    if sos is None:
        sos = design_prefilter(fs)
    x = np.ascontiguousarray(x_padded, dtype=np.float32)
    y = sosfiltfilt(sos.astype(np.float32), x, axis=-1)
    y = decimate_to_analysis(y, fs)
    pad_out = pad_analysis_samples(fs)
    return np.ascontiguousarray(y[:, pad_out:y.shape[-1] - pad_out], dtype=np.float32)


# --------------------------------------------------------------------------
# 2. Native block readers
# --------------------------------------------------------------------------


class BlockReader:
    """Read native samples in microvolts from one recorded block."""

    n_samples: int = 0
    n_channels: int = 0
    native_rate: float = 0.0

    def read(self, start: int, stop: int, native_indices: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - trivial
        pass


class EpilepsiaeBlockReader(BlockReader):
    """``.data`` is int16 little-endian, sample-major / channel-fastest.

    Verified two ways: ``src/preprocessing.py::load_epilepsiae_block`` reshapes
    ``(-1, num_channels)`` and transposes, and
    ``filesize == num_samples * num_channels * 2``.  The repo-wide convention
    is ``uV = int16 * (-conversion_factor)`` -- the sign flip is irrelevant for
    band power and cancels in any bipolar difference, but we keep it so this
    cache is polarity-consistent with every other Epilepsiae artifact here.
    """

    def __init__(self, data_path: Path, head_path: Optional[Path] = None):
        try:
            from ..preprocessing import _read_epilepsiae_head_for_streaming
        except ImportError:  # pragma: no cover - flat sys.path layouts
            from src.preprocessing import _read_epilepsiae_head_for_streaming

        data_path = Path(data_path)
        if head_path is None:
            head_path = data_path.with_suffix(".head")
        head = _read_epilepsiae_head_for_streaming(Path(head_path))
        if int(head["sample_bytes"]) != 2:
            raise ValueError(f"unsupported sample_bytes for {head_path}")
        self.n_channels = int(head["num_channels"])
        self.native_rate = float(head["sample_freq"])
        self.channel_names = list(head["channel_names"])
        self.conversion = -1.0 * float(head["conversion_factor"])
        file_bytes = os.path.getsize(data_path)
        n_from_file = file_bytes // (2 * self.n_channels)
        self.n_samples = int(min(int(head["num_samples"]), n_from_file))
        self._mm = np.memmap(
            data_path, dtype="<i2", mode="r", shape=(n_from_file, self.n_channels)
        )

    def read(self, start: int, stop: int, native_indices: np.ndarray) -> np.ndarray:
        block = np.asarray(self._mm[start:stop, :][:, native_indices], dtype=np.float32)
        return np.ascontiguousarray(block.T) * np.float32(self.conversion)

    def close(self) -> None:
        self._mm = None


class YuquanEdfBlockReader(BlockReader):
    """One MNE EDF handle per file, reused for every read (open costs 4-30 s).

    ``encoding='latin1'`` is mandatory -- the Chinese hospital headers are not
    valid UTF-8 and MNE raises without it.
    """

    def __init__(self, edf_path: Path):
        import mne

        self._n_signals, self._first_annotation = _edf_signal_layout(Path(edf_path))
        self._raw = mne.io.read_raw_edf(
            str(edf_path), preload=False, verbose="ERROR", encoding="latin1"
        )
        self.native_rate = float(self._raw.info["sfreq"])
        self.n_channels = len(self._raw.ch_names)
        self.n_samples = int(self._raw.n_times)
        self.channel_names = list(self._raw.ch_names)

    def read(self, start: int, stop: int, native_indices: np.ndarray) -> np.ndarray:
        picks = np.asarray(native_indices, dtype=int)
        if picks.size and int(picks.max()) >= self._first_annotation:
            raise ValueError(
                f"native index {int(picks.max())} sits at or after the EDF annotation "
                f"signal (position {self._first_annotation} of {self._n_signals}); MNE "
                "drops annotation signals from ch_names, so every index at or beyond it "
                "would silently address the wrong channel.  contact_metadata's "
                "native_index_* must index the EDF signal list, and this file's "
                "annotation signal is not last."
            )
        order = np.argsort(picks, kind="stable")
        uniq, inverse = np.unique(picks[order], return_inverse=True)
        data = self._raw.get_data(picks=uniq, start=int(start), stop=int(stop))
        out = np.empty((picks.size, data.shape[1]), dtype=np.float32)
        out[order] = data[inverse].astype(np.float32)
        return out * np.float32(1e6)  # MNE returns volts

    def close(self) -> None:
        try:
            self._raw.close()
        except Exception:  # pragma: no cover
            pass
        self._raw = None


def _edf_signal_layout(path: Path) -> Tuple[int, int]:
    """(n_signals, index of the first 'EDF Annotations' signal or n_signals).

    MNE removes annotation signals from ``ch_names``, so a Yuquan native index
    only equals the MNE row when every annotation signal sits after the last
    real contact.  On this cohort the annotation channel is the last of 104 in
    every file checked, but that is verified per file rather than assumed.
    """
    with open(path, "rb") as fh:
        header = fh.read(256)
        n_signals = int(header[252:256].decode("latin1").strip())
        labels = fh.read(16 * n_signals).decode("latin1")
    names = [labels[i * 16:(i + 1) * 16].strip() for i in range(n_signals)]
    ann = [i for i, nm in enumerate(names) if "annotation" in nm.lower()]
    return n_signals, (ann[0] if ann else n_signals)


def open_block_reader(source_path: str, source_kind: str) -> BlockReader:
    kind = str(source_kind).lower()
    path = Path(source_path)
    if kind.startswith("epilepsiae") or path.suffix == ".data":
        return EpilepsiaeBlockReader(path)
    if kind.startswith("yuquan") or kind == "edf_header" or path.suffix.lower() == ".edf":
        return YuquanEdfBlockReader(path)
    raise ValueError(f"cannot pick a reader for source_kind={source_kind!r} path={path}")


# --------------------------------------------------------------------------
# 3. Subject-level plan
# --------------------------------------------------------------------------


@dataclass
class ContactPlan:
    """Bipolar channel wiring for one subject, in ``channel_index`` order."""

    channel_names: List[str]
    anode_native: np.ndarray          # (C,) int, index into the native file
    cathode_native: np.ndarray        # (C,) int
    native_indices: np.ndarray        # (K,) sorted unique native indices we read
    anode_row: np.ndarray             # (C,) position of anode inside native_indices
    cathode_row: np.ndarray           # (C,) position of cathode inside native_indices

    @property
    def n_contacts(self) -> int:
        return len(self.channel_names)


def build_contact_plan(contact_df, subject: str) -> ContactPlan:
    sub = contact_df[contact_df["subject"] == subject].sort_values("channel_index")
    if sub.empty:
        raise ValueError(f"no contact_metadata rows for subject {subject!r}")
    idx = sub["channel_index"].to_numpy()
    if not np.array_equal(idx, np.arange(len(sub))):
        raise ValueError(
            f"{subject}: channel_index must be a dense 0..C-1 range, got "
            f"{idx[:5]}...{idx[-3:]}"
        )
    anode = sub["native_index_anode"].to_numpy().astype(int)
    cathode = sub["native_index_cathode"].to_numpy().astype(int)
    native = np.unique(np.concatenate([anode, cathode]))
    pos = {int(v): i for i, v in enumerate(native)}
    return ContactPlan(
        channel_names=[str(v) for v in sub["channel_name"].tolist()],
        anode_native=anode,
        cathode_native=cathode,
        native_indices=native,
        anode_row=np.array([pos[int(v)] for v in anode], dtype=int),
        cathode_row=np.array([pos[int(v)] for v in cathode], dtype=int),
    )


@dataclass
class MinuteGrid:
    minute_index: np.ndarray
    minute_start_epoch: np.ndarray
    covered: np.ndarray
    split: np.ndarray

    @property
    def n_minutes(self) -> int:
        return int(self.minute_index.size)


def build_minute_grid(minute_df, subject: str) -> MinuteGrid:
    sub = minute_df[minute_df["subject"] == subject].sort_values("minute_index")
    if sub.empty:
        raise ValueError(f"no window_index rows for subject {subject!r}")
    mi = sub["minute_index"].to_numpy().astype(int)
    if not np.array_equal(mi, np.arange(mi.size)):
        raise ValueError(f"{subject}: minute_index must be a dense 0..N-1 range")
    starts = sub["minute_start_epoch"].to_numpy().astype(float)
    if mi.size > 1:
        step = np.diff(starts)
        if not np.allclose(step, 60.0, atol=1e-3):
            raise ValueError(
                f"{subject}: minute grid is not a uniform 60 s grid "
                f"(min step {step.min():.3f}, max {step.max():.3f})"
            )
    return MinuteGrid(
        minute_index=mi,
        minute_start_epoch=starts,
        covered=sub["covered"].to_numpy().astype(bool),
        split=sub["split"].to_numpy().astype(object),
    )


def select_cached_minutes(
    grid: MinuteGrid,
    train_end_epoch: float,
    dev_end_epoch: float,
    cache_cap: bool = True,
    train_hours_cap: Optional[float] = contract.CACHE_TRAIN_HOURS_CAP,
    val_hours_cap: Optional[float] = contract.CACHE_VAL_HOURS_CAP,
) -> np.ndarray:
    """Boolean (n_minutes,): which minutes get real signal written.

    A minute qualifies when it is ``covered``, its split is train/validation,
    and it ends at or before ``dev_end_epoch``.  With ``cache_cap`` the most
    recent ``train_hours_cap`` covered hours before ``train_end_epoch`` and the
    most recent ``val_hours_cap`` covered hours before ``dev_end_epoch`` are
    kept -- so train and validation stay chronologically adjacent to each other
    and to the sealed bound.
    """
    ends = grid.minute_start_epoch + 60.0
    base = grid.covered & (ends <= dev_end_epoch + 1e-6)
    split = np.array([str(s) for s in grid.split])
    is_train = base & (split == "train")
    is_val = base & (split == "validation")
    if not cache_cap:
        return is_train | is_val
    out = np.zeros(grid.n_minutes, dtype=bool)
    for mask, cap_hours in ((is_train, train_hours_cap), (is_val, val_hours_cap)):
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        if cap_hours is None:          # no cap: keep every covered minute
            out[idx] = True
            continue
        keep = int(round(float(cap_hours) * 60.0))
        out[idx[-keep:] if idx.size > keep else idx] = True
    return out


def _assert_not_sealed(subject: str, epochs: np.ndarray, dev_end_epoch: float) -> None:
    """Sealed-partition gate; works for real cohort subjects and synthetic ones."""
    arr = np.atleast_1d(np.asarray(epochs, dtype=float))
    if arr.size and float(np.nanmax(arr)) >= float(dev_end_epoch):
        raise ValueError(
            f"SEALED-PARTITION VIOLATION for {subject}: max cached minute start "
            f"{float(np.nanmax(arr)):.3f} >= dev_end_epoch {float(dev_end_epoch):.3f}"
        )
    try:
        splits = contract.load_subject_splits()
    except Exception:  # pragma: no cover - manifest unavailable in unit tests
        return
    if subject in splits:
        frozen = contract.dev_end_epoch(subject)
        if abs(frozen - float(dev_end_epoch)) > 1e-6:
            raise ValueError(
                f"{subject}: dev_end_epoch {dev_end_epoch!r} disagrees with the frozen "
                f"upstream split bound {frozen!r}; contract.dev_end_epoch is the only "
                "sanctioned source"
            )
        contract.assert_not_sealed(subject, arr)


# --------------------------------------------------------------------------
# 4. Reading one run of minutes
# --------------------------------------------------------------------------


def _assign_minutes_to_blocks(grid: MinuteGrid, cached: np.ndarray, blocks) -> Dict[int, List[int]]:
    """Map block row -> the cached minute indices whose midpoint lies inside it."""
    starts = blocks["block_start_epoch"].to_numpy().astype(float)
    ends = blocks["block_end_epoch"].to_numpy().astype(float)
    mid = grid.minute_start_epoch + 30.0
    out: Dict[int, List[int]] = {}
    order = np.argsort(starts, kind="stable")
    for m in np.flatnonzero(cached):
        hit = -1
        for b in order:
            if starts[b] <= mid[m] < ends[b]:
                hit = int(b)
                break
        if hit < 0:
            raise ValueError(
                f"minute {int(m)} (mid epoch {mid[m]:.1f}) is marked covered but falls "
                "in no block; window_index and dataset_manifest disagree"
            )
        out.setdefault(hit, []).append(int(m))
    return out


def _contiguous_runs(values: Sequence[int]) -> List[List[int]]:
    runs: List[List[int]] = []
    for v in sorted(values):
        if runs and v == runs[-1][-1] + 1:
            runs[-1].append(v)
        else:
            runs.append([v])
    return runs


@dataclass
class MinuteRun:
    """A maximal stretch of cached minutes inside one block AND one split.

    Runs never cross a split boundary.  That is deliberate and load-bearing:
    the zero-phase filter reaches ``PAD_SECONDS`` past each end of whatever it
    processes, so a run spanning the train/validation cut would let a few
    seconds of validation signal bleed into the last train minute -- tiny, but
    it makes hard-invalidity condition #5 ("normalisation used validation
    data") impossible to prove clean.  Clipping the pad at the split boundary
    and reflecting instead makes the train half bit-identical no matter what
    the validation half contains, which is exactly what
    ``test_train_stats_ignore_the_validation_half`` pins.
    """

    block: int
    split: str
    minutes: List[int]
    avail_lo: int          # native sample index the pad may not reach below
    avail_hi: int          # native sample index the pad may not reach at/above


def plan_minute_runs(grid: MinuteGrid, cached: np.ndarray,
                     minute_to_block: Dict[int, List[int]], blocks) -> List[MinuteRun]:
    starts = blocks["block_start_epoch"].to_numpy().astype(float)
    rates = blocks["native_sampling_rate"].to_numpy().astype(float)
    split = np.array([str(s) for s in grid.split])
    runs: List[MinuteRun] = []
    for b in sorted(minute_to_block):
        fs = int(round(rates[b]))
        for sp in sorted({split[m] for m in minute_to_block[b]}):
            members = [m for m in minute_to_block[b] if split[m] == sp]
            for part in _contiguous_runs(members):
                lo = int(round((grid.minute_start_epoch[part[0]] - starts[b]) * fs))
                runs.append(MinuteRun(
                    block=int(b), split=sp, minutes=list(part),
                    avail_lo=lo, avail_hi=lo + len(part) * 60 * fs,
                ))
    return runs


def _read_run(
    reader: BlockReader,
    plan: ContactPlan,
    block_start_epoch: float,
    native_rate: float,
    minute_start_epoch: float,
    n_minutes: int,
    sos: np.ndarray,
    avail_lo: Optional[int] = None,
    avail_hi: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Decode ``n_minutes`` consecutive minutes starting at ``minute_start_epoch``.

    Returns (C, n_minutes * MINUTE_SAMPLES) float32 microvolts, plus the number
    of native samples that had to be edge-reflected because the run reached
    past a block edge (>= 0; should only ever be the <=5 % of a boundary minute
    that ``MINUTE_COVERAGE_FRACTION`` tolerates).
    """
    fs = int(round(float(native_rate)))
    pad = pad_native_samples(fs)
    length = n_minutes * 60 * fs
    s0 = int(round((minute_start_epoch - block_start_epoch) * fs))
    want_lo, want_hi = s0 - pad, s0 + length + pad
    floor = 0 if avail_lo is None else max(0, int(avail_lo))
    ceil = reader.n_samples if avail_hi is None else min(reader.n_samples, int(avail_hi))
    lo, hi = max(floor, want_lo), min(ceil, want_hi)
    if hi <= lo:
        raise ValueError("run falls entirely outside the block")
    raw = reader.read(lo, hi, plan.native_indices)
    missing = (lo - want_lo) + (want_hi - hi)
    if missing:
        raw = np.pad(
            raw,
            ((0, 0), (lo - want_lo, want_hi - hi)),
            mode="reflect" if raw.shape[-1] > 1 else "edge",
        )
    bip = raw[plan.anode_row] - raw[plan.cathode_row]
    del raw
    return process_native_segment(bip, fs, sos=sos), int(missing)


# --------------------------------------------------------------------------
# 5. int16 calibration
# --------------------------------------------------------------------------


def _calibrate_int16_scale(
    subject: str,
    plan: ContactPlan,
    grid: MinuteGrid,
    cached: np.ndarray,
    runs: List[MinuteRun],
    blocks,
    readers: Dict[int, BlockReader],
    sos_by_rate: Dict[int, np.ndarray],
    calib_minutes: int,
) -> Dict[str, np.ndarray]:
    """Per-contact int16 scale from a subsample of TRAIN minutes only.

    ``6 x MAD`` of the decimated train signal is mapped to
    ``INT16_TARGET_COUNTS``; anything larger clips at the int16 rail and is
    counted.  Validation minutes are never touched here (hard invalidity
    condition #5: normalisation must not see validation).
    """
    split = np.array([str(s) for s in grid.split])
    train_cached = np.flatnonzero(cached & (split == "train"))
    if train_cached.size == 0:
        raise ValueError(f"{subject}: no cached TRAIN minutes to calibrate the int16 scale")
    # Draw the calibration minutes as a few CONTIGUOUS stretches rather than as
    # `calib_minutes` scattered singletons. The int16 scale is a per-contact MAD,
    # so a handful of stretches spread across train is statistically the same
    # sample -- but on a rotational disk the scattered version is 100x slower:
    # the first pilot build spent 52 minutes calibrating one Yuquan subject
    # (~77 s per single-minute read) while the same read costs 0.23 s on an idle
    # spindle. Sequential reads survive contention; random ones do not.
    take = train_cached
    if take.size > calib_minutes:
        n_stretch = max(1, min(CALIB_STRETCHES, calib_minutes))
        per = max(1, calib_minutes // n_stretch)
        anchors = np.linspace(0, max(0, take.size - per), n_stretch).round().astype(int)
        picked: List[int] = []
        for a in anchors:
            seg = take[a:a + per]
            # only keep a stretch that is actually contiguous in minute index,
            # otherwise it is no cheaper than scattered reads
            if seg.size and int(seg[-1] - seg[0]) == seg.size - 1:
                picked.extend(int(v) for v in seg)
            else:
                picked.append(int(seg[0]))
        take = np.array(sorted(set(picked)), dtype=int)
        if take.size == 0:
            take = train_cached[: min(calib_minutes, train_cached.size)]
    run_of = {m: r for r in runs for m in r.minutes}
    starts = blocks["block_start_epoch"].to_numpy().astype(float)
    rates = blocks["native_sampling_rate"].to_numpy().astype(float)
    chunks = []
    for group in _contiguous_runs([int(m) for m in take]):
        # one read per contiguous stretch, and never across two source blocks
        by_block: Dict[int, List[int]] = {}
        for m in group:
            by_block.setdefault(run_of[m].block, []).append(m)
        for b, ms in by_block.items():
            for part in _contiguous_runs(ms):
                run = run_of[part[0]]
                fs = int(round(rates[b]))
                y, _ = _read_run(
                    readers[b], plan, starts[b], fs,
                    grid.minute_start_epoch[part[0]], len(part),
                    sos_by_rate[fs], run.avail_lo, run.avail_hi,
                )
                chunks.append(y)
    sample = np.concatenate(chunks, axis=1)
    median = np.median(sample, axis=1).astype(np.float64)
    mad = np.median(np.abs(sample - median[:, None]), axis=1).astype(np.float64)
    scale = 6.0 * mad / INT16_TARGET_COUNTS
    dead = ~np.isfinite(scale) | (scale <= 0)
    scale[dead] = FALLBACK_SCALE_UV
    return {
        "int16_scale_uv": scale.astype(np.float32),
        "calib_median_uv": median.astype(np.float32),
        "calib_mad_uv": mad.astype(np.float32),
        "calib_dead_contacts": np.flatnonzero(dead).astype(int),
        "calib_minutes_used": np.asarray(take, dtype=int),
    }


# --------------------------------------------------------------------------
# 6. The builder
# --------------------------------------------------------------------------


def _zarr_int16_array(path: Path, shape, chunks, overwrite: bool):
    import zarr
    from zarr.codecs import BloscCodec, BloscShuffle

    return zarr.create_array(
        store=str(path),
        shape=shape,
        chunks=chunks,
        dtype="int16",
        compressors=[BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.shuffle)],
        overwrite=overwrite,
    )


def build_subject_cache(
    subject: str,
    manifest_df,
    contact_df,
    minute_df,
    out_path: Optional[Path] = None,
    train_end_epoch: Optional[float] = None,
    dev_end_epoch: Optional[float] = None,
    *,
    cache_cap: bool = True,
    train_hours_cap: Optional[float] = contract.CACHE_TRAIN_HOURS_CAP,
    val_hours_cap: Optional[float] = contract.CACHE_VAL_HOURS_CAP,
    chunk_minutes: int = CHUNK_MINUTES_DEFAULT,
    calib_minutes: int = CALIB_MINUTES_DEFAULT,
    overwrite: bool = True,
    log=None,
) -> Dict[str, object]:
    """Build ``raw_256hz.zarr`` + ``cache_index.parquet`` for one subject.

    Returns a JSON-serialisable summary (timings, bytes, clipping rates, counts)
    that ``build_raw_cache.py`` folds into ``BUILD_STATUS.json``.
    """
    import pandas as pd

    t_start = time.time()
    say = log or (lambda *_a, **_k: None)

    if train_end_epoch is None:
        train_end_epoch = contract.load_subject_splits()[subject].train_end_epoch
    if dev_end_epoch is None:
        dev_end_epoch = contract.dev_end_epoch(subject)
    out_path = Path(out_path) if out_path is not None else contract.raw_cache_path(subject)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = build_contact_plan(contact_df, subject)
    grid = build_minute_grid(minute_df, subject)
    if plan.native_indices.min() < 0:
        raise ValueError(f"{subject}: negative native channel index in contact_metadata")
    blocks = manifest_df[manifest_df["subject"] == subject].sort_values("block_start_epoch")
    if blocks.empty:
        raise ValueError(f"no dataset_manifest rows for subject {subject!r}")
    blocks = blocks.reset_index(drop=True)

    cached = select_cached_minutes(
        grid, train_end_epoch, dev_end_epoch, cache_cap, train_hours_cap, val_hours_cap
    )
    if not cached.any():
        raise ValueError(f"{subject}: no minutes selected for caching")
    _assert_not_sealed(subject, grid.minute_start_epoch[cached], dev_end_epoch)

    minute_to_block = _assign_minutes_to_blocks(grid, cached, blocks)
    runs = plan_minute_runs(grid, cached, minute_to_block, blocks)
    rates = blocks["native_sampling_rate"].to_numpy().astype(float)
    starts = blocks["block_start_epoch"].to_numpy().astype(float)
    sos_by_rate = {int(round(r)): design_prefilter(r) for r in np.unique(rates)}

    C = plan.n_contacts
    n_minutes = grid.n_minutes
    say(f"{subject}: C={C} n_minutes={n_minutes} cached={int(cached.sum())} "
        f"blocks_touched={len(minute_to_block)}")

    readers: Dict[int, BlockReader] = {}
    summary: Dict[str, object] = {}
    arr = None
    try:
        # -- calibration pass (train only) ---------------------------------
        t_cal = time.time()
        for b in sorted(minute_to_block):
            readers[b] = open_block_reader(
                str(blocks["source_path"].iloc[b]), str(blocks["source_kind"].iloc[b])
            )
            if plan.native_indices.max() >= readers[b].n_channels:
                raise ValueError(
                    f"{subject}: contact_metadata references native channel "
                    f"{int(plan.native_indices.max())} but block "
                    f"{blocks['block_id'].iloc[b]!r} only has {readers[b].n_channels} "
                    "channels; native_index_* must index the source file's own channel "
                    "list (MNE ch_names order for EDF, .head elec_names order for "
                    "Epilepsiae)"
                )
        calib = _calibrate_int16_scale(
            subject, plan, grid, cached, runs, blocks, readers,
            sos_by_rate, calib_minutes,
        )
        scale = calib["int16_scale_uv"].astype(np.float64)
        cal_sec = time.time() - t_cal
        say(f"{subject}: calibrated {len(calib['calib_minutes_used'])} train minutes "
            f"in {cal_sec:.1f}s; {len(calib['calib_dead_contacts'])} dead contacts")

        # -- write pass -----------------------------------------------------
        arr = _zarr_int16_array(
            out_path, (n_minutes * contract.MINUTE_SAMPLES, C),
            (contract.MINUTE_SAMPLES, C), overwrite,
        )
        n_clipped = np.zeros(C, dtype=np.int64)
        n_written_samples = 0
        n_reflected_native = 0
        t_dec = 0.0
        minutes_done = 0
        for b in sorted(minute_to_block):
            fs = int(round(rates[b]))
            sos = sos_by_rate[fs]
            for run in [r for r in runs if r.block == b]:
                for k in range(0, len(run.minutes), chunk_minutes):
                    part = run.minutes[k:k + chunk_minutes]
                    t0 = time.time()
                    y, missing = _read_run(
                        readers[b], plan, starts[b], fs,
                        grid.minute_start_epoch[part[0]], len(part), sos,
                        run.avail_lo, run.avail_hi,
                    )
                    t_dec += time.time() - t0
                    n_reflected_native += missing
                    expect = len(part) * contract.MINUTE_SAMPLES
                    if y.shape[-1] != expect:
                        raise AssertionError(
                            f"{subject}: decoded {y.shape[-1]} samples, expected {expect}"
                        )
                    q = y / scale[:, None]
                    n_clipped += ((q > INT16_MAX) | (q < INT16_MIN)).sum(axis=1)
                    np.clip(q, INT16_MIN, INT16_MAX, out=q)
                    lo = part[0] * contract.MINUTE_SAMPLES
                    arr[lo:lo + expect, :] = np.rint(q).astype(np.int16).T
                    n_written_samples += expect
                    minutes_done += len(part)
            readers[b].close()
            readers.pop(b)
        arr.attrs["contact_scale_uv"] = [float(v) for v in scale]
        arr.attrs["channel_names"] = list(plan.channel_names)
        arr.attrs["analysis_rate_hz"] = contract.ANALYSIS_RATE_HZ
        arr.attrs["minute_samples"] = contract.MINUTE_SAMPLES
        arr.attrs["n_minutes_grid"] = int(n_minutes)
        arr.attrs["contract_version"] = contract.CONTRACT_VERSION
        arr.attrs["subject"] = subject
    finally:
        for r in readers.values():
            r.close()

    np.save(out_dir / "contact_scale_uv.npy", scale.astype(np.float32))
    np.save(out_dir / "minute_filled.npy", ~cached)
    split = np.array([str(s) for s in grid.split])
    # cache_index carries only minutes that end strictly before the sealed bound.
    # The zarr keeps the FULL minute grid so minute_index * MINUTE_SAMPLES stays
    # exact, but no artifact of this build may contain a sealed timestamp, so the
    # index is truncated and every consumer scatters by ``minute_index`` rather
    # than by row position (``load_cache_index``).
    dev_ok = (grid.minute_start_epoch + 60.0) <= float(dev_end_epoch)
    pd.DataFrame({
        "minute_index": grid.minute_index[dev_ok],
        "minute_start_epoch": grid.minute_start_epoch[dev_ok],
        "split": split[dev_ok],
        "cached": cached[dev_ok],
        "filled": ~cached[dev_ok],
    }).to_parquet(out_dir / "cache_index.parquet", index=False)

    bytes_on_disk = sum(p.stat().st_size for p in out_path.rglob("*") if p.is_file())
    total = float(n_written_samples) * C
    summary = {
        "subject": subject,
        "contract_version": contract.CONTRACT_VERSION,
        "n_contacts": int(C),
        "n_minutes_total": int(n_minutes),
        "n_minutes_cached": int(cached.sum()),
        "n_minutes_train": int((cached & (split == "train")).sum()),
        "n_minutes_validation": int((cached & (split == "validation")).sum()),
        "native_rates": sorted({int(round(r)) for r in rates}),
        "bytes_on_disk": int(bytes_on_disk),
        "bytes_uncompressed": int(total * 2),
        "compression_ratio": float(total * 2 / max(bytes_on_disk, 1)),
        "clip_fraction_max": float(n_clipped.max() / max(n_written_samples, 1)),
        "clip_fraction_mean": float(n_clipped.sum() / max(total, 1)),
        "n_dead_contacts": int(len(calib["calib_dead_contacts"])),
        "dead_contacts": [int(v) for v in calib["calib_dead_contacts"]],
        "n_reflected_native_samples": int(n_reflected_native),
        "n_minute_runs": int(len(runs)),
        "int16_scale_uv_median": float(np.median(scale)),
        "calibration_seconds": float(cal_sec),
        "decode_seconds": float(t_dec),
        "wall_seconds": float(time.time() - t_start),
        "cached_hours": float(cached.sum() / 60.0),
        "decode_x_realtime": float(cached.sum() * 60.0 / max(t_dec, 1e-9)),
        "n_minutes_written": int(minutes_done),
        "cache_cap_applied": bool(cache_cap),
        "train_end_epoch": float(train_end_epoch),
        "dev_end_epoch": float(dev_end_epoch),
        "raw_cache_path": str(out_path),
    }
    say(f"{subject}: wrote {bytes_on_disk/1e9:.2f} GB "
        f"(ratio {summary['compression_ratio']:.2f}x) in {summary['wall_seconds']:.0f}s")
    return summary


def load_cache(subject: str, cache_path: Optional[Path] = None):
    """Open the int16 cache read-only; returns (zarr array, scale_uv (C,))."""
    import zarr

    p = Path(cache_path) if cache_path is not None else contract.raw_cache_path(subject)
    arr = zarr.open_array(str(p), mode="r")
    scale = np.asarray(arr.attrs["contact_scale_uv"], dtype=np.float32)
    return arr, scale


def load_cache_index(cache_dir: Path, n_minutes: int) -> Dict[str, np.ndarray]:
    """Scatter ``cache_index.parquet`` back onto the FULL minute grid.

    The parquet is truncated at the sealed bound (so it contains no sealed
    timestamp), while the zarr keeps every minute of the grid.  Consumers must
    therefore index by ``minute_index``, never by row position.
    """
    import pandas as pd

    df = pd.read_parquet(Path(cache_dir) / "cache_index.parquet")
    mi = df["minute_index"].to_numpy().astype(int)
    if mi.size and int(mi.max()) >= n_minutes:
        raise ValueError(
            f"cache_index minute_index {int(mi.max())} exceeds the grid ({n_minutes})"
        )
    cached = np.zeros(n_minutes, dtype=bool)
    split = np.full(n_minutes, "sealed", dtype=object)
    start = np.full(n_minutes, np.nan, dtype=float)
    cached[mi] = df["cached"].to_numpy().astype(bool)
    split[mi] = df["split"].to_numpy().astype(str)
    start[mi] = df["minute_start_epoch"].to_numpy().astype(float)
    return {"cached": cached, "split": split.astype(str), "minute_start_epoch": start,
            "filled": ~cached}


def read_minutes_uv(arr, scale_uv: np.ndarray, minute_index: int, n_minutes: int = 1) -> np.ndarray:
    """(n_minutes*MINUTE_SAMPLES, C) microvolts for a contiguous minute run."""
    lo = int(minute_index) * contract.MINUTE_SAMPLES
    hi = lo + int(n_minutes) * contract.MINUTE_SAMPLES
    return np.asarray(arr[lo:hi, :], dtype=np.float32) * scale_uv[None, :]
