"""Training reader: one (context, target) pair per item.  Owner: Worker B.

Plain words: the model is shown the last 10 minutes of decimated raw signal
from every bipolar contact and asked what the contact x frequency power field
will look like 1 / 5 / 10 / 100 minutes later.  This module turns the frozen
minute grid into exactly those pairs, and refuses to emit a pair that would
smuggle in a minute that was never recorded, that failed the artifact rule,
that sits on the other side of a recording gap or a seizure guard, or that
lies at/after the sealed bound.

Index convention (checked against Worker A's ``window_index.parquet`` at
construction time, see ``CONTEXT_CONVENTION``):

    t_index          the reference minute; the row of window_index
    context minutes  [t_index - CONTEXT_MINUTES + 1 .. t_index]   (inclusive)
    origin_epoch     end of the context  = t_epoch + 60
    target minute h  t_index + h,  epoch  = t_epoch + h*60

``ctx_ok`` in Worker A's schema is documented as "the 10 minutes ending at this
index are all usable, one session", which is the inclusive convention.  The
dataset re-verifies that against ``minute_usable`` on construction and raises
with an explicit instruction if the upstream convention turns out to be the
exclusive one -- a silent off-by-one here is hard-invalidity condition #1
(temporal leakage), so it must fail loudly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

# Blosc runs its own thread pool inside the C library. A forked DataLoader
# worker inherits that pool's mutexes in whatever state the parent left them,
# and the first decompression in the child deadlocks: on the very first real
# training job a worker went zombie and the parent sat on a futex with the GPU
# at 0% for nine minutes. Turning blosc's internal threading off makes the codec
# fork-safe, and costs nothing measurable here -- a warm 10-minute window
# decompresses at ~550 MB/s single-threaded, well above what the GPU consumes.
try:  # pragma: no cover - depends on the installed codec stack
    import numcodecs.blosc as _blosc

    _blosc.use_threads = False
except Exception:  # pragma: no cover
    pass

import os

from . import contract
from .spectral_target import artifact_mask_path

CONTEXT_CONVENTION = "inclusive"
"""``inclusive``: context = [t-CTX+1 .. t].  ``exclusive``: [t-CTX .. t-1]."""

TARGET_KEYS = (
    "target", "target_mask", "target_epoch",
    "persistence", "persistence_mask",
    "t_index", "t_epoch", "origin_epoch", "subject",
)
"""Everything an item carries beyond ``contract.ALLOWED_INPUT_KEYS``."""

CONSISTENCY_KEYS = ("raw_next", "minute_valid_next")


def context_range(t_index: int, convention: str = CONTEXT_CONVENTION) -> np.ndarray:
    n = contract.CONTEXT_MINUTES
    if convention == "inclusive":
        return np.arange(t_index - n + 1, t_index + 1)
    if convention == "exclusive":
        return np.arange(t_index - n, t_index)
    raise ValueError(f"unknown context convention {convention!r}")


class WindowEligibility:
    """Which reference minutes may be emitted -- pure index arithmetic.

    Deliberately free of any Zarr / torch dependency so Worker D can enumerate
    both evaluation index sets (``contract.EVAL_SET_PRIMARY`` =
    ``common_all_horizons``, ``contract.EVAL_SET_SECONDARY`` = ``per_horizon``)
    from the parquet files alone, without instantiating a dataset or touching
    the cache.
    """

    def __init__(
        self,
        subject: str,
        split: str,
        window_index,
        horizons: Sequence[int],
        cached: np.ndarray,
        *,
        require_all_horizons: bool = True,
        need_consistency: bool = False,
        context_convention: str = CONTEXT_CONVENTION,
        verify_convention: bool = True,
    ):
        self.subject = str(subject)
        self.split = str(split)
        self.horizons = [int(h) for h in horizons]
        self.require_all_horizons = bool(require_all_horizons)
        self.need_consistency = bool(need_consistency)
        self.context_convention = context_convention

        wi = window_index[window_index["subject"] == self.subject].sort_values("minute_index")
        if wi.empty:
            raise ValueError(f"no window_index rows for {self.subject!r}")
        self.n_minutes = int(wi["minute_index"].max()) + 1
        self.minute_start = wi["minute_start_epoch"].to_numpy().astype(float)
        self.minute_split = wi["split"].to_numpy().astype(str)
        self.minute_usable = wi["minute_usable"].to_numpy().astype(bool)
        self.ctx_ok = wi["ctx_ok"].to_numpy().astype(bool)
        self.h_ok = {h: wi[f"h{h}_ok"].to_numpy().astype(bool) for h in self.horizons}
        self.cached = np.asarray(cached, dtype=bool)
        if self.cached.size != self.n_minutes:
            raise ValueError(
                f"{self.subject}: cached mask has {self.cached.size} entries, window "
                f"index has {self.n_minutes} minutes"
            )
        if verify_convention:
            self._verify_convention()
        self.index = self._build_index()

    def _verify_convention(self) -> None:
        """``ctx_ok`` must agree with the context minutes we intend to read."""
        rows = np.flatnonzero(self.ctx_ok)
        if rows.size == 0:
            return
        probe = rows[np.linspace(0, rows.size - 1, min(200, rows.size)).round().astype(int)]

        def consistent(conv: str) -> bool:
            for t in probe:
                c = context_range(int(t), conv)
                if c.min() < 0 or c.max() >= self.n_minutes:
                    return False
                if not self.minute_usable[c].all():
                    return False
            return True

        conv = self.context_convention
        if not consistent(conv):
            other = "exclusive" if conv == "inclusive" else "inclusive"
            raise ValueError(
                f"{self.subject}: ctx_ok is inconsistent with the {conv!r} context "
                f"convention.  Worker A's window_index appears to use the {other!r} "
                f"convention ({'consistent' if consistent(other) else 'also inconsistent'}); "
                f"pass context_convention={other!r} after confirming with Worker A. "
                "This is refused rather than guessed because an off-by-one here is "
                "hard-invalidity condition #1 (temporal leakage)."
            )

    def row_ok(self, t: int) -> bool:
        ctx = context_range(t, self.context_convention)
        if ctx.min() < 0 or ctx.max() >= self.n_minutes:
            return False
        if not self.ctx_ok[t]:
            return False
        if not (self.cached[ctx].all() and self.minute_usable[ctx].all()):
            return False
        if not (self.minute_split[ctx] == self.split).all():
            return False
        if self.need_consistency:
            nxt = t + 1
            if nxt >= self.n_minutes or not self.ctx_ok[nxt]:
                return False
            nctx = context_range(nxt, self.context_convention)
            if not (self.cached[nctx].all() and self.minute_usable[nctx].all()):
                return False
            if not (self.minute_split[nctx] == self.split).all():
                return False
        return True

    def horizon_ok(self, t: int, h: int) -> bool:
        tt = t + h
        if tt >= self.n_minutes or not self.h_ok[h][t]:
            return False
        return bool(
            self.cached[tt] and self.minute_usable[tt]
            and self.minute_split[tt] == self.split
        )

    def _build_index(self) -> np.ndarray:
        rows: List[int] = []
        for t in np.flatnonzero(self.ctx_ok):
            t = int(t)
            if not self.row_ok(t):
                continue
            oks = [self.horizon_ok(t, h) for h in self.horizons]
            if (all(oks) if self.require_all_horizons else any(oks)):
                rows.append(t)
        return np.asarray(rows, dtype=np.int64)

    def counts(self) -> Dict[int, int]:
        return {h: int(sum(self.horizon_ok(int(t), h) for t in self.index))
                for h in self.horizons}


def default_require_all_horizons(split: str) -> bool:
    """``train`` keeps every window (per-horizon masks); eval defaults to the
    primary ``common_all_horizons`` set."""
    return False if str(split) == "train" else True


def load_cached_mask(cache_index_path: Path, n_minutes: int) -> np.ndarray:
    import pandas as pd

    ci = pd.read_parquet(cache_index_path)
    mi = ci["minute_index"].to_numpy().astype(int)
    if mi.size and int(mi.max()) >= n_minutes:
        raise ValueError(
            f"cache_index reaches minute {int(mi.max())} but the window index only "
            f"has {n_minutes} minutes"
        )
    out = np.zeros(n_minutes, dtype=bool)
    out[mi] = ci["cached"].to_numpy().astype(bool)
    return out


def eligible_indices(
    subject: str,
    split: str,
    window_index,
    horizons: Sequence[int] = contract.HORIZONS_MIN,
    *,
    require_all: Optional[bool] = None,
    cache_index_path: Optional[Path] = None,
    need_consistency: bool = False,
    context_convention: str = CONTEXT_CONVENTION,
) -> np.ndarray:
    """Reference-minute indices for one (subject, split, horizon set), no Zarr.

    ``require_all=True`` gives ``contract.EVAL_SET_PRIMARY``
    (``common_all_horizons``); ``False`` gives ``contract.EVAL_SET_SECONDARY``
    (``per_horizon``).  ``None`` follows ``default_require_all_horizons``.
    """
    if require_all is None:
        require_all = default_require_all_horizons(split)
    wi = window_index[window_index["subject"] == str(subject)]
    if wi.empty:
        raise ValueError(f"no window_index rows for {subject!r}")
    n_minutes = int(wi["minute_index"].max()) + 1
    path = (Path(cache_index_path) if cache_index_path is not None
            else contract.cache_dir(subject) / "cache_index.parquet")
    return WindowEligibility(
        subject, split, window_index, horizons, load_cached_mask(path, n_minutes),
        require_all_horizons=require_all, need_consistency=need_consistency,
        context_convention=context_convention,
    ).index


class SubjectWindowDataset(Dataset):
    """(context, target) pairs for one subject and one split.

    Parameters
    ----------
    subject, split
        ``split`` is ``train`` or ``validation``; the sealed partition can never
        be requested because it is not in ``window_index``.
    horizons
        Minutes ahead to predict.  With ``require_all_horizons=True`` (default,
        and what the pilot uses) a reference minute is emitted only if EVERY
        requested horizon is eligible, so all horizons see the same t's and the
        horizon curve is a within-t comparison.  With ``False`` a t eligible for
        h=1 but not h=100 is still emitted and ``target_mask[100]`` is all-False
        -- the loss must then divide by the per-horizon mask sum.
    need_consistency
        Also return ``raw_next`` (the context shifted +1 minute) for the
        ``L_cons`` term.  Off by default so the h=1-only pilot does not pay the
        extra 30 MB read per item.
    """

    def __init__(
        self,
        subject: str,
        split: str,
        window_index,
        contact_df,
        horizons: Sequence[int] = contract.HORIZONS_MIN,
        *,
        cache_path: Optional[Path] = None,
        target_path: Optional[Path] = None,
        mask_path: Optional[Path] = None,
        stats_path: Optional[Path] = None,
        cache_index_path: Optional[Path] = None,
        require_all_horizons: Optional[bool] = None,
        need_consistency: bool = False,
        context_convention: str = CONTEXT_CONVENTION,
        verify_convention: bool = True,
    ):
        import json

        self.subject = str(subject)
        self.split = str(split)
        self.horizons = [int(h) for h in horizons]
        if require_all_horizons is None:
            require_all_horizons = default_require_all_horizons(split)
        self.require_all_horizons = bool(require_all_horizons)
        self.need_consistency = bool(need_consistency)
        self.context_convention = context_convention

        self._cache_path = Path(cache_path) if cache_path else contract.raw_cache_path(subject)
        cdir = self._cache_path.parent
        self._target_path = Path(target_path) if target_path else contract.spectral_target_path(subject)
        self._mask_path = Path(mask_path) if mask_path else artifact_mask_path(subject, cdir)
        stats_path = Path(stats_path) if stats_path else contract.subject_stats_path(subject)
        cache_index_path = Path(cache_index_path) if cache_index_path else cdir / "cache_index.parquet"

        stats = json.loads(stats_path.read_text())
        self._target_mean = np.asarray(stats["target_mean"], dtype=np.float32)
        self._target_std = np.asarray(stats["target_std"], dtype=np.float32)
        self._int16_scale = np.asarray(stats["int16_scale_uv"], dtype=np.float32)
        self._raw_center = np.asarray(stats["raw_center_uv"], dtype=np.float32)
        self._raw_scale = np.asarray(stats["raw_scale_uv"], dtype=np.float32)

        # -- contacts -------------------------------------------------------
        # contact_valid ("a well-formed bipolar signal exists") and coord_valid
        # ("we know where it is") are independent axes in the contract.  Five
        # Yuquan subjects record fine but have no electrode localisation at all;
        # collapsing the two would delete them.  So coordinates are mean-centred
        # over the coord_valid subset only, zeroed elsewhere, and never imputed --
        # coord_valid travels with them so the encoder can gate the mm projection
        # and fall back on the shaft_id / shaft_index topology floor.
        sub = contact_df[contact_df["subject"] == self.subject].sort_values("channel_index")
        if sub.empty:
            raise ValueError(f"no contact_metadata rows for {self.subject!r}")
        coords = sub[["x_mm", "y_mm", "z_mm"]].to_numpy().astype(np.float32)
        coord_valid = sub["coord_valid"].to_numpy().astype(bool) & np.isfinite(coords).all(axis=1)
        coords = np.where(np.isfinite(coords), coords, 0.0).astype(np.float32)
        if coord_valid.any():
            coords = coords - coords[coord_valid].mean(axis=0, keepdims=True)
        coords[~coord_valid] = 0.0
        self.coords_mm = coords
        self.coord_valid = coord_valid
        self.contact_valid = sub["contact_valid"].to_numpy().astype(bool)
        self.coord_mode = (str(sub["coord_mode"].iloc[0]) if "coord_mode" in sub.columns
                           else contract.COORD_MODE_FULL)
        self.n_contacts = int(len(sub))
        self.n_contacts_without_coord = int((~coord_valid).sum())
        shafts = [str(s) for s in sub["shaft"].tolist()]
        order = {s: i for i, s in enumerate(sorted(set(shafts)))}
        self.shaft_id = np.array([order[s] for s in shafts], dtype=np.int64)
        self.shaft_index = sub["shaft_index"].to_numpy().astype(np.int64)

        # -- minute grid ----------------------------------------------------
        wi = window_index[window_index["subject"] == self.subject]
        if wi.empty:
            raise ValueError(f"no window_index rows for {self.subject!r}")
        n_minutes = int(wi["minute_index"].max()) + 1
        self._elig = WindowEligibility(
            self.subject, self.split, window_index, self.horizons,
            load_cached_mask(cache_index_path, n_minutes),
            require_all_horizons=self.require_all_horizons,
            need_consistency=self.need_consistency,
            context_convention=self.context_convention,
            verify_convention=verify_convention,
        )
        self._n_minutes = self._elig.n_minutes
        self._minute_start = self._elig.minute_start
        self._index = self._elig.index
        self._raw = None
        self._tgt = None
        self._mask = None

    # -- eligibility (delegated to WindowEligibility) -----------------------

    def eligible_counts(self) -> Dict[int, int]:
        """Per-horizon count of emitted items whose target is actually valid."""
        return self._elig.counts()

    # -- lazy zarr handles (never opened before a fork) ---------------------

    def _ensure_open(self) -> None:
        if self._raw is not None:
            return
        import zarr

        self._raw_gain = None
        self._raw_bias = None
        self._raw = zarr.open_array(str(self._cache_path), mode="r")
        # The spectral target and the artifact mask are small -- (n_minutes, C,
        # 12) float32 and (n_minutes, C) bool, i.e. 19 MB and 0.4 MB for
        # epilepsiae_620, 123 MB and 2.6 MB for the largest implantation -- and
        # every item touches them a dozen times (persistence, four horizons, and
        # a mask for each). Left as zarr handles that was twelve round trips per
        # item through zarr 3's async-to-sync bridge, and profiling showed half
        # of a 180 ms item was spent in _thread.lock.acquire waiting on that
        # bridge. Materialising them once per worker removes it. Only the raw
        # waveform stays on disk; it is the part that does not fit.
        self._tgt = np.asarray(zarr.open_array(str(self._target_path), mode="r")[:],
                               dtype=np.float32)
        self._mask = np.asarray(zarr.open_array(str(self._mask_path), mode="r")[:],
                                dtype=bool)

    def close(self) -> None:
        self._raw = self._tgt = self._mask = None

    def __len__(self) -> int:
        return int(self._index.size)

    # -- item ---------------------------------------------------------------

    def _read_raw_norm(self, ctx: np.ndarray) -> np.ndarray:
        return self._read_raw_span(int(ctx[0]), int(ctx[-1]) + 1)

    def _read_raw_span(self, m_lo: int, m_hi: int) -> np.ndarray:
        """Normalised waveform for minutes ``[m_lo, m_hi)`` as (C, n*15360)."""
        blk = np.asarray(self._raw[m_lo * contract.MINUTE_SAMPLES:
                                   m_hi * contract.MINUTE_SAMPLES, :], dtype=np.float32)
        # int16 -> microvolts -> train-normalised, fused into one multiply-add
        # done in place. Written out as (blk*int16_scale - center)/scale it made
        # three full temporaries of a 21 MB array per item; the fused form makes
        # none. Exactly equal in float32 up to the associativity of a*b/c, which
        # is why _raw_gain / _raw_bias are precomputed in float64 and cast once.
        if getattr(self, "_raw_gain", None) is None:
            g = (np.asarray(self._int16_scale, dtype=np.float64)
                 / np.asarray(self._raw_scale, dtype=np.float64))
            b = (-np.asarray(self._raw_center, dtype=np.float64)
                 / np.asarray(self._raw_scale, dtype=np.float64))
            self._raw_gain = g.astype(np.float32)[None, :]
            self._raw_bias = b.astype(np.float32)[None, :]
        np.multiply(blk, self._raw_gain, out=blk)
        np.add(blk, self._raw_bias, out=blk)
        return np.ascontiguousarray(blk.T)

    def _read_raw_pair(self, ctx: np.ndarray, nctx: np.ndarray):
        """Both context windows from ONE read of their union.

        The consistency term needs the context at t and the context at t+1, and
        those overlap in nine of their ten minutes. Reading them separately
        fetched, decompressed and normalised the same nine minutes twice --
        about a third of the time in a warm item. One read of the eleven-minute
        union and two views costs one.
        """
        lo = min(int(ctx[0]), int(nctx[0]))
        hi = max(int(ctx[-1]), int(nctx[-1])) + 1
        span = self._read_raw_span(lo, hi)
        M = contract.MINUTE_SAMPLES
        def _slice(c):
            a = (int(c[0]) - lo) * M
            return np.ascontiguousarray(span[:, a:a + len(c) * M])
        return _slice(ctx), _slice(nctx)

    def _minute_valid(self, ctx: np.ndarray) -> np.ndarray:
        art = np.asarray(self._mask[ctx, :], dtype=bool)        # (n_ctx, C)
        return (~art).T & self.contact_valid[:, None]           # (C, n_ctx)

    def _norm_field(self, minute: int) -> np.ndarray:
        f = np.asarray(self._tgt[int(minute), :, :], dtype=np.float32)
        return (f - self._target_mean) / self._target_std

    def __getitem__(self, i: int) -> Dict[str, object]:
        self._ensure_open()
        t = int(self._index[int(i)])
        ctx = context_range(t, self.context_convention)
        t_epoch = float(self._minute_start[t])

        item: Dict[str, object] = {
            "raw": torch.from_numpy(self._read_raw_norm(ctx)),
            "coords_mm": torch.from_numpy(self.coords_mm.copy()),
            "coord_valid": torch.from_numpy(self.coord_valid.copy()),
            "shaft_id": torch.from_numpy(self.shaft_id.copy()),
            "shaft_index": torch.from_numpy(self.shaft_index.copy()),
            "contact_valid": torch.from_numpy(self.contact_valid.copy()),
            "minute_valid": torch.from_numpy(np.ascontiguousarray(self._minute_valid(ctx))),
        }

        last = int(ctx[-1])
        pers = self._norm_field(last)
        pers_mask = (
            np.isfinite(pers)
            & (~np.asarray(self._mask[last, :], dtype=bool))[:, None]
            & self.contact_valid[:, None]
        )
        item["persistence"] = torch.from_numpy(np.nan_to_num(pers, nan=0.0))
        item["persistence_mask"] = torch.from_numpy(pers_mask)

        tgt, tmask, tepoch = {}, {}, {}
        for h in self.horizons:
            key = str(h)
            if self._elig.horizon_ok(t, h):
                y = self._norm_field(t + h)
                m = (
                    np.isfinite(y)
                    & (~np.asarray(self._mask[t + h, :], dtype=bool))[:, None]
                    & self.contact_valid[:, None]
                )
                tgt[key] = torch.from_numpy(np.nan_to_num(y, nan=0.0))
                tmask[key] = torch.from_numpy(m)
            else:
                tgt[key] = torch.zeros(
                    (self.n_contacts, contract.N_FREQ_BINS), dtype=torch.float32
                )
                tmask[key] = torch.zeros(
                    (self.n_contacts, contract.N_FREQ_BINS), dtype=torch.bool
                )
            tepoch[key] = t_epoch + 60.0 * h
        item["target"] = tgt
        item["target_mask"] = tmask
        item["target_epoch"] = tepoch

        item["t_index"] = t
        item["t_epoch"] = t_epoch
        item["origin_epoch"] = t_epoch + 60.0
        item["subject"] = self.subject

        if self.need_consistency:
            nctx = context_range(t + 1, self.context_convention)
            raw_now, raw_next = self._read_raw_pair(ctx, nctx)
            item["raw"] = torch.from_numpy(raw_now)
            item["raw_next"] = torch.from_numpy(raw_next)
            item["minute_valid_next"] = torch.from_numpy(
                np.ascontiguousarray(self._minute_valid(nctx))
            )
        return item

    # -- introspection ------------------------------------------------------

    def encoder_inputs(self, item: Dict[str, object]) -> Dict[str, object]:
        """The subset of an item that may reach the encoder (contract section 5)."""
        payload = {k: item[k] for k in contract.ALLOWED_INPUT_KEYS}
        contract.assert_no_forbidden_inputs(payload)
        return payload

    def summary(self) -> Dict[str, object]:
        return {
            "subject": self.subject,
            "split": self.split,
            "n_items": len(self),
            "n_contacts": self.n_contacts,
            "n_contacts_without_coord": self.n_contacts_without_coord,
            "n_contacts_valid": int(self.contact_valid.sum()),
            "coord_mode": self.coord_mode,
            "horizons": list(self.horizons),
            "require_all_horizons": self.require_all_horizons,
            "need_consistency": self.need_consistency,
            "context_convention": self.context_convention,
            "eligible_per_horizon": self.eligible_counts(),
        }


def worker_init_fn(worker_id: int) -> None:  # pragma: no cover - exercised via DataLoader
    """Close any handle inherited across the fork; each worker reopens lazily."""
    from torch.utils.data import get_worker_info

    try:  # belt and braces: also disable it inside the child
        import numcodecs.blosc as _b

        _b.use_threads = False
    except Exception:
        pass
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    import torch as _t

    _t.set_num_threads(1)
    info = get_worker_info()
    if info is not None and hasattr(info.dataset, "close"):
        info.dataset.close()


def collate_windows(batch: List[Dict[str, object]]):
    """Default collate, after asserting C is constant across the batch.

    C is constant *within* a subject and R0.1 never mixes subjects in a batch,
    so ``torch.utils.data.default_collate`` is sufficient; the assert is what
    turns a future cross-subject batch into a loud error instead of a stack
    shape crash deep inside the model.
    """
    from torch.utils.data._utils.collate import default_collate

    shapes = {tuple(b["coords_mm"].shape) for b in batch}
    if len(shapes) != 1:
        raise ValueError(f"batch mixes contact counts {sorted(shapes)}")
    subjects = {b["subject"] for b in batch}
    if len(subjects) != 1:
        raise ValueError(f"batch mixes subjects {sorted(subjects)}")
    return default_collate(batch)
