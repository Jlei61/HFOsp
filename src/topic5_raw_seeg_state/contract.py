"""Frozen contract for the Raw-SEEG evolvable-state model, revision R0.1.

Owner: main agent. **No worker may edit this file.** Every constant, schema and
path helper below is the single source of truth that the four workers
(A data-contract / B spectral-target+IO / C model / D train+figures) compile
against. If a worker believes a constant is wrong, it reports back instead of
editing.

Scientific spec  : docs/archive/topic5/raw_seeg_state_scientific_spec_2026-08-21.md
Execution plan   : docs/archive/topic5/raw_seeg_state_execution_plan_2026-08-21.md

Two hard boundaries encoded here:

1. ``SEALED`` — the formal-test partition is defined by the *already frozen*
   Epi-PRSSM v0.1 split manifest. R0.1 may only read wall-clock time strictly
   before ``validation.last_epoch`` for each subject. ``dev_end_epoch()`` is the
   only function that returns that bound and every consumer must route through
   it.
2. ``R0.1 target space`` — the model predicts a contact x log-frequency power
   field and nothing else. No IED marks, no seizure labels, no SOZ, no arrival
   likelihood. ``FORBIDDEN_INPUT_KEYS`` lists what must never reach the encoder.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# 0. Revision identity
# --------------------------------------------------------------------------

REVISION = "r0_1"
CONTRACT_VERSION = "raw_seeg_state_contract_v2_2026-08-21"
#: v2 (same day, before any training run): omega bound halved to the Nyquist
#: rate of a minute-sampled state; encoder inputs went from 5 to 7 keys
#: (coord_valid + shaft_index) so contacts without a coordinate stay usable;
#: cache hour caps lifted after the measured 4.68x compression; Yuquan seizure
#: guard now unions the frozen inventory with the EDF annotation scan.

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Small artifacts (manifests, metrics, checkpoints, figures) live in the repo.
RESULT_ROOT = REPO_ROOT / "results" / "epi_prssm" / "raw_seeg_state" / REVISION

#: The decimated raw cache must NOT go on the repo volume (``/`` has <120 GB
#: free). It also must not share a spindle with the raw source it is being built
#: from: every disk on this box is rotational, and the first pilot build showed
#: Yuquan calibration reads collapsing from 0.23 s to ~77 s per minute while six
#: writers hammered the same platter. So the two datasets cross over --
#: Epilepsiae reads from sdc and writes to sdd, Yuquan reads from sdd and writes
#: to sdc -- and neither ever contends with itself.
#:   Epilepsiae cache ~206 GB raw  -> /mnt/yuquan_data   (816 GB free)
#:   Yuquan     cache ~60 GB raw   -> /mnt/epilepsia_data (193 GB free)
CACHE_ROOT = Path("/mnt/yuquan_data/hfosp_cache/raw_seeg_state_r0_1")
CACHE_ROOT_BY_DATASET: Dict[str, Path] = {
    "epilepsiae": Path("/mnt/yuquan_data/hfosp_cache/raw_seeg_state_r0_1"),
    "yuquan": Path("/mnt/epilepsia_data/hfosp_cache/raw_seeg_state_r0_1"),
}

DATA_DIR = RESULT_ROOT / "data"
PER_SUBJECT_DIR = RESULT_ROOT / "per_subject"
FIGURE_DIR = RESULT_ROOT / "figures"
LOG_DIR = RESULT_ROOT / "logs"
JOB_DIR = RESULT_ROOT / "jobs"
MANIFEST_DIR = RESULT_ROOT / "manifests"
REPORT_DIR = RESULT_ROOT / "reports"

#: The conda interpreter and the LD_LIBRARY_PATH entry pandas needs on this box.
PYTHON_BIN = "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python"
CONDA_LIB = "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib"

# --------------------------------------------------------------------------
# 1. Upstream frozen inputs (read-only)
# --------------------------------------------------------------------------

#: Chronological 60/20/20 split frozen by Epi-PRSSM v0.1. R0.1 inherits it
#: verbatim; ``test`` stays sealed.
UPSTREAM_SPLIT_MANIFEST = (
    REPO_ROOT / "results/epi_prssm/v0_1/manifests/SPLIT_MANIFEST.json"
)
EPILEPSIAE_BLOCK_INVENTORY = REPO_ROOT / "results/epilepsiae_block_inventory.csv"
YUQUAN_BLOCK_INVENTORY = REPO_ROOT / "results/dataset_inventory/yuquan_block_inventory.csv"
EPILEPSIAE_SEIZURE_INVENTORY = REPO_ROOT / "results/epilepsiae_seizure_inventory.csv"
YUQUAN_SEIZURE_INVENTORY = REPO_ROOT / "results/dataset_inventory/yuquan_seizure_inventory.csv"
YUQUAN_EDF_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")

YUQUAN_SUPPLEMENTARY_SEIZURE_DIR = REPO_ROOT / "results/seizure_detection"
"""``pr1_seizure_<subject>.json`` holds the raw EDF-annotation scan. The frozen
``YUQUAN_SEIZURE_INVENTORY`` drops zero-duration marks (onset == offset fails
``has_complete_eeg_interval``), which silently loses two real onsets inside the
dev window (zhangbichen in train, chenziyang in validation). The guard must be
the UNION of both sources, de-duplicated by onset within 1 s. Losing an hour of
training data is cheap; letting an ictal transition into the backbone is not."""

SEIZURE_OFFSET_FALLBACK_SECONDS = 120.0
"""When a seizure has an onset but no usable offset, treat the offset as
``onset + 120 s`` before applying POSTICTAL_GUARD_SECONDS."""

#: The nine Yuquan subjects that have no row in the frozen block inventory.
#: Their block intervals must be rebuilt from the EDF fixed header (same rule
#: Epi-PRSSM v0.1 used, see its DATA_MANIFEST.recorded_coverage_rule).
YUQUAN_SUBJECTS_WITHOUT_INVENTORY = (
    "chengshuai",
    "chenziyang",
    "hanyuxuan",
    "liyouran",
    "songzishuo",
    "wangyiyang",
    "zhangbichen",
    "zhangjiaqi",
    "zhaochenxi",
)

# --------------------------------------------------------------------------
# 2. Signal contract  (FROZEN 2026-08-21 after the Nyquist audit)
# --------------------------------------------------------------------------
#
# Native rates present in the 34-subject cohort:
#   Yuquan      : 2000 Hz (all 16)
#   Epilepsiae  : 1024 Hz (14 subjects), 512 Hz (253, and part of 139),
#                  256 Hz (part of 139, 384, 583)
# The cohort-wide Nyquist floor is therefore 128 Hz, set by the 256 Hz
# recordings. A single common analysis rate keeps the cohort comparable, so:

ANALYSIS_RATE_HZ = 256
"""Common decimated rate for every subject and every block."""

FREQ_LOW_HZ = 1.0
FREQ_HIGH_HZ = 100.0
"""Upper edge sits ~22 % below the 128 Hz floor, leaving anti-alias margin.
R0.1 therefore says NOTHING about ripple / HFO band (>100 Hz) predictability."""

N_FREQ_BINS = 12
FREQ_EDGES: np.ndarray = np.logspace(
    np.log10(FREQ_LOW_HZ), np.log10(FREQ_HIGH_HZ), N_FREQ_BINS + 1
)

#: Mains frequency is 50 Hz on both mounts (China / Germany). These bands are
#: notch-filtered before decimation AND excluded from band-power integration.
LINE_NOISE_HZ: Tuple[float, ...] = (50.0, 100.0)
LINE_NOISE_HALFWIDTH_HZ = 1.0

PATCH_SECONDS = 0.25
PATCH_SAMPLES = int(round(PATCH_SECONDS * ANALYSIS_RATE_HZ))          # 64
WINDOW_SECONDS = 5.0
WINDOW_SAMPLES = int(round(WINDOW_SECONDS * ANALYSIS_RATE_HZ))        # 1280
PATCHES_PER_WINDOW = int(round(WINDOW_SECONDS / PATCH_SECONDS))       # 20
WINDOWS_PER_MINUTE = int(round(60.0 / WINDOW_SECONDS))                # 12
MINUTE_SAMPLES = WINDOW_SAMPLES * WINDOWS_PER_MINUTE                  # 15360

#: Welch parameters for the one-minute target field, on the 256 Hz signal.
TARGET_WELCH_NPERSEG = 2048        # 8 s  -> 0.125 Hz resolution
TARGET_WELCH_NOVERLAP = 1024       # 50 % overlap -> 14 segments per minute
TARGET_WELCH_WINDOW = "hann"
TARGET_LOG_EPS = 1e-12

#: Montage. Bipolar between adjacent contacts on the same shaft. Rationale:
#: both mounts are stored against a common/scalp reference, and a shared
#: reference injects a global component that would make "predicting the whole
#: contact field" trivially easy for reasons that have nothing to do with brain
#: state. Coordinate of a bipolar channel = midpoint of its two contacts.
MONTAGE = "bipolar_within_shaft"

# --------------------------------------------------------------------------
# 3. Time / eligibility contract
# --------------------------------------------------------------------------

CONTEXT_MINUTES = 10
"""Encoder input history, in minutes of wall-clock time."""

HORIZONS_MIN: Tuple[int, ...] = (1, 5, 10, 100)

MINUTE_COVERAGE_FRACTION = 0.95
"""A minute counts as recorded only if its covered fraction is STRICTLY greater
than 0.95 (i.e. > 57.0 s of 60 s). The strict reading is the conservative one and
is what the unit tests pin (57 s -> False, 58 s -> True); prose elsewhere that
says ">=95 %" means this."""

SESSION_JOIN_SECONDS = 300.0
"""Gaps <= 300 s do not open a new session (same convention Epi-PRSSM v0.1
froze as ``session_join_seconds``). Gaps > 300 s DO break the session, and no
context/target pair may straddle a session boundary. Minutes overlapping any
gap still fail MINUTE_COVERAGE_FRACTION and are individually unusable."""

PREICTAL_GUARD_SECONDS = 3600.0
"""60 min before EEG onset is removed from the backbone-training pool so that
the preictal raw windows a later E0.5 / H2b analysis needs stay untouched."""

POSTICTAL_GUARD_SECONDS = 3600.0
"""60 min after EEG offset is the frozen postictal interval for R0.1."""

ARTIFACT_ROBUST_Z = 6.0
"""Per contact, a minute whose broadband log power deviates more than 6 robust
SD (median / 1.4826*MAD, both estimated on TRAIN only) from that contact's
train median is flagged as an artifact minute for that contact."""

ARTIFACT_SATURATION_FRACTION = 0.01
"""Per contact-minute, >1 % of samples at the ADC rail also flags an artifact."""

MINUTE_MIN_VALID_CONTACT_FRACTION = 0.70
"""If fewer than 70 % of a subject's contacts survive the artifact rule in a
given minute, the whole minute is unusable (cannot be context or target)."""

TRAIN_REQUIRE_ALL_HORIZONS = False
"""Training uses every window that is valid for *any* horizon, with per-horizon
masks, so no data is thrown away. The loss already drops a horizon with zero
valid entries from its denominator."""

EVAL_SET_PRIMARY = "common_all_horizons"
EVAL_SET_SECONDARY = "per_horizon"
"""Evaluation reports TWO window sets, and the distinction is load-bearing for
Figure R2.

``common_all_horizons`` — only windows valid for all four horizons at once.
This is the PRIMARY horizon curve: it is the only way the curve measures
"harder horizon" rather than "different windows". Every arm (model, mean,
persistence, feature-AR, identity) is scored on exactly this index set.

``per_horizon`` — every window valid for that horizon individually, with its own
denominator. This is the SECONDARY table, and it exists because four subjects
(gaolan, litengsheng, songzishuo, sunyuanxin) have ZERO validation windows at
h=100 — their validation span is shorter than 110 min — and requiring all four
horizons would delete them from the h=1/5/10 results as well.

A subject with an empty primary set is reported in the secondary table only,
and is excluded from the cohort horizon curve with its exclusion stated, never
silently imputed.
"""

#: Cache caps, in hours of *covered* time, applied as "the most recent covered
#: hours before the partition boundary" so train and validation stay
#: chronologically adjacent to each other and to the sealed bound.
#:
#: LIFTED 2026-08-21, before any subject beyond the first was built. The caps
#: existed only as a storage budget, sized against an assumed ~1.4x zstd ratio
#: on int16 neural data. The first real build (yuquan_huanghanwen, 1289 minutes
#: x 87 contacts) measured **4.68x**, which puts the whole 34-subject cache at
#: ~68 GB on disk with NO cap at all, against 834 GB free on the mount. An
#: arbitrary engineering limit that costs epilepsiae_620 three quarters of its
#: 213 recorded hours is not worth keeping once it buys nothing, so both caps
#: are now ``None`` = cache every covered dev minute. Build cost rises from
#: ~1056 h to ~2415 h of source, roughly 1.5 h wall clock at 8-10 processes.
CACHE_TRAIN_HOURS_CAP: Optional[float] = None
CACHE_VAL_HOURS_CAP: Optional[float] = None

# --------------------------------------------------------------------------
# 4. Model contract
# --------------------------------------------------------------------------

LATENT_DIM = 32
N_ROTATION_MODES = 16          # 16 x 2D blocks == LATENT_DIM
assert 2 * N_ROTATION_MODES == LATENT_DIM

D_MODEL = 128
N_TEMPORAL_LAYERS = 2          # within-contact, over the 20 patches of a 5 s window
N_SPATIAL_LAYERS = 2           # across contacts, within a 5 s window
N_CONTEXT_LAYERS = 3           # causal, over the 10 minute-tokens
N_HEADS = 4
FFN_MULT = 4
DROPOUT = 0.1

TAU_MIN_MINUTES = 1.0
TAU_MAX_MINUTES = 48.0 * 60.0
"""Mode time constants are learned in log space and hard-clamped to
[1 min, 48 h]; every mode is strictly stable (exp(-h/tau) < 1)."""

OMEGA_MAX_RAD_PER_MIN = float(np.pi / TAU_MIN_MINUTES)
"""Rotation-rate bound = the Nyquist rate of a minute-sampled state.

Originally set to ``2*pi/TAU_MIN_MINUTES`` (one full turn per minute); Worker C
showed that is exactly 2x the identifiable range. Every horizon in
``HORIZONS_MIN`` and the consistency step are whole numbers of minutes, so for
integer ``h`` the pair ``omega`` and ``omega + 2*pi`` give bit-identical
``cos(omega*h)/sin(omega*h)`` — the upper half of the old box was a perfect
alias of the lower half, and a fitted ``omega > pi`` in dynamics_modes.json
would have been mis-read as a fast mode when it is the alias of a slow one.
``pi/TAU_MIN_MINUTES`` (half a turn per minute) is the non-aliased bound.
Corrected 2026-08-21, before any training run.
"""

LAMBDA_CONS_DEFAULT = 0.1
"""Only two settings are compared in R0.1: LAMBDA_CONS_DEFAULT and 0.0."""

CONSISTENCY_HUBER_DELTA = 1.0

# --------------------------------------------------------------------------
# 5. What must never enter the encoder
# --------------------------------------------------------------------------

FORBIDDEN_INPUT_KEYS: Tuple[str, ...] = (
    "ied_label",
    "ied_event_mark",
    "iei",
    "event_rate",
    "seizure_label",
    "seizure_onset",
    "soz",
    "soz_core",
    "contact_rank",
    "lagpat_rank",
    "template_rank",
    "epi_prssm_latent",
    "epi_prssm_state",
    "swap_class",
    "vigilance",
    "day_night",
)
"""R0.1's encoder sees exactly: decimated raw SEEG, contact coordinates,
shaft id, valid/artifact mask. Anything in this tuple appearing as a key in a
model input dict is an immediate hard error, not a warning."""

ALLOWED_INPUT_KEYS: Tuple[str, ...] = (
    "raw",            # (C, CONTEXT_MINUTES*MINUTE_SAMPLES) float32, train-normalised
    "coords_mm",      # (C, 3) float32, per-subject native space, mean-centred; 0 where invalid
    "coord_valid",    # (C,) bool  - this contact HAS an anatomical mm coordinate
    "shaft_id",       # (C,) int64 - which electrode shaft
    "shaft_index",    # (C,) int64 - 0-based position along that shaft
    "contact_valid",  # (C,) bool  - contact is ELECTRICALLY usable (independent of coords)
    "minute_valid",   # (C, CONTEXT_MINUTES) bool - artifact mask per contact-minute
)

#: ``contact_valid`` and ``coord_valid`` are deliberately independent axes.
#: ``contact_valid`` answers "is there a well-formed bipolar signal here"; it is
#: what decides whether the channel exists at all. ``coord_valid`` answers "do we
#: know where it is in the brain". Collapsing the two would have thrown away five
#: whole Yuquan subjects (chenziyang / gaolan / hanyuxuan / sunyuanxin /
#: wangyiyang) whose recordings are fine but for which no electrode-localisation
#: artifact exists anywhere on the mount — only raw MRI/CT, which R0.1 is not in
#: the business of processing. Position encoding therefore always has a floor:
#: ``shaft_id`` + ``shaft_index`` embeddings are present for every contact, and
#: the mm-coordinate projection is added only where ``coord_valid`` is True.
COORD_MODE_FULL = "mm"
COORD_MODE_TOPOLOGY_ONLY = "shaft_index_only"
"""Per-subject label recorded in contact_metadata / run_manifest. Cohort
statistics must report the two groups separately; a shaft_index_only subject
carries no anatomical distance information and cannot support any spatial
claim."""

# --------------------------------------------------------------------------
# 6. Artifact schemas (column order is part of the contract)
# --------------------------------------------------------------------------

DATASET_MANIFEST_COLUMNS: Tuple[str, ...] = (
    "subject", "dataset", "session_id", "block_id",
    "block_start_epoch", "block_end_epoch", "duration_sec",
    "native_sampling_rate", "n_channels_native",
    "source_path", "source_kind", "gap_to_prev_sec", "opens_session",
    "split",  # train | validation | sealed | dropped
)

WINDOW_INDEX_COLUMNS: Tuple[str, ...] = (
    "subject", "minute_index", "minute_start_epoch", "session_id", "split",
    "covered", "guard_free", "n_valid_contacts", "minute_usable",
    "ctx_ok",            # the 10 minutes ending at this index are all usable, one session
    "h1_ok", "h5_ok", "h10_ok", "h100_ok",
)

CONTACT_METADATA_COLUMNS: Tuple[str, ...] = (
    "subject", "dataset", "channel_index", "channel_name",
    "anode", "cathode", "shaft", "shaft_index",
    "x_mm", "y_mm", "z_mm", "coord_space", "coord_valid",
    "native_index_anode", "native_index_cathode", "contact_valid", "drop_reason",
    "coord_mode",
)

ELIGIBILITY_COLUMNS: Tuple[str, ...] = (
    "subject", "dataset", "n_contacts", "native_rates", "nyquist_limited",
    "recorded_hours_total", "dev_covered_hours", "train_hours", "val_hours",
    "cached_train_hours", "cached_val_hours",
    "n_sessions", "n_seizures_in_dev", "n_seizures_from_supplement",
    "seizure_guard_source", "coord_mode", "guard_hours_removed",
    "n_train_h1", "n_train_h5", "n_train_h10", "n_train_h100",
    "n_val_h1", "n_val_h5", "n_val_h10", "n_val_h100",
    "pilot_tier", "status",
)

# --------------------------------------------------------------------------
# 7. Helpers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SubjectSplit:
    """Wall-clock split bounds inherited from the Epi-PRSSM v0.1 manifest."""

    subject: str
    dataset: str
    first_epoch: float
    train_end_epoch: float
    dev_end_epoch: float          # == upstream validation.last_epoch; SEALED beyond
    sealed_first_epoch: float     # upstream test.first_epoch, for auditing only


_SPLIT_CACHE: Optional[Dict[str, SubjectSplit]] = None


def load_subject_splits(
    manifest_path: Path = UPSTREAM_SPLIT_MANIFEST,
) -> Dict[str, SubjectSplit]:
    """Read the frozen upstream split and expose it as wall-clock bounds.

    The upstream manifest indexes *events*; ``boundaries.<part>.{first,last}_epoch``
    are the wall-clock times of the first/last event of each partition. R0.1
    consumes continuous time, so it uses those epochs directly as cut points and
    is deliberately conservative at the sealed edge: everything at or after
    ``validation.last_epoch`` is off limits, which is earlier than the upstream
    test partition actually starts.
    """
    global _SPLIT_CACHE
    if _SPLIT_CACHE is not None and manifest_path == UPSTREAM_SPLIT_MANIFEST:
        return _SPLIT_CACHE
    payload = json.loads(Path(manifest_path).read_text())
    if payload.get("test_status") != "SEALED_UNTIL_FORMAL_TEST_RELEASE":
        raise ValueError(
            "upstream split manifest no longer declares the test partition sealed; "
            f"got {payload.get('test_status')!r}"
        )
    out: Dict[str, SubjectSplit] = {}
    for row in payload["subjects"]:
        b = row["boundaries"]
        subject = row["subject"]
        out[subject] = SubjectSplit(
            subject=subject,
            dataset=subject.split("_", 1)[0],
            first_epoch=float(b["train"]["first_epoch"]),
            train_end_epoch=float(b["train"]["last_epoch"]),
            dev_end_epoch=float(b["validation"]["last_epoch"]),
            sealed_first_epoch=float(b["test"]["first_epoch"]),
        )
    if manifest_path == UPSTREAM_SPLIT_MANIFEST:
        _SPLIT_CACHE = out
    return out


def cohort_subjects() -> List[str]:
    """The 34 frozen subjects, sorted."""
    return sorted(load_subject_splits())


def dev_end_epoch(subject: str) -> float:
    """The ONLY sanctioned way to learn where the sealed partition begins."""
    return load_subject_splits()[subject].dev_end_epoch


def assert_not_sealed(subject: str, epochs) -> None:
    """Raise if any timestamp reaches into the sealed formal-test partition."""
    bound = dev_end_epoch(subject)
    arr = np.atleast_1d(np.asarray(epochs, dtype=float))
    if arr.size and float(np.nanmax(arr)) >= bound:
        raise ValueError(
            f"SEALED-PARTITION VIOLATION for {subject}: max epoch "
            f"{float(np.nanmax(arr)):.3f} >= dev_end_epoch {bound:.3f}"
        )


def assert_no_forbidden_inputs(payload: Dict[str, object]) -> None:
    """Hard gate for the encoder input dict (contract section 5)."""
    bad = sorted(set(payload) & set(FORBIDDEN_INPUT_KEYS))
    if bad:
        raise ValueError(f"forbidden R0.1 encoder inputs present: {bad}")
    unknown = sorted(set(payload) - set(ALLOWED_INPUT_KEYS))
    if unknown:
        raise ValueError(
            f"unrecognised encoder input keys {unknown}; allowed = {list(ALLOWED_INPUT_KEYS)}"
        )


def band_indices(freqs: np.ndarray) -> List[np.ndarray]:
    """Indices of ``freqs`` falling in each of the N_FREQ_BINS log bands.

    Line-noise neighbourhoods are dropped from every band so 50/100 Hz residue
    cannot leak into a band-power value.
    """
    freqs = np.asarray(freqs, dtype=float)
    keep = np.ones_like(freqs, dtype=bool)
    for f0 in LINE_NOISE_HZ:
        keep &= np.abs(freqs - f0) > LINE_NOISE_HALFWIDTH_HZ
    out: List[np.ndarray] = []
    for i in range(N_FREQ_BINS):
        lo, hi = FREQ_EDGES[i], FREQ_EDGES[i + 1]
        sel = (freqs >= lo) & (freqs < hi) & keep
        if not sel.any():  # pragma: no cover - guarded by test
            raise ValueError(
                f"band {i} [{lo:.2f},{hi:.2f}) Hz has no usable FFT bin; "
                "check TARGET_WELCH_NPERSEG"
            )
        out.append(np.flatnonzero(sel))
    return out


# -- per-subject paths ------------------------------------------------------


def cache_dir(subject: str) -> Path:
    """Per-subject cache directory, on the spindle that is NOT its raw source."""
    dataset = str(subject).split("_", 1)[0]
    return CACHE_ROOT_BY_DATASET.get(dataset, CACHE_ROOT) / subject


def raw_cache_path(subject: str) -> Path:
    """Decimated int16 raw, zarr, shape (n_minutes*MINUTE_SAMPLES, C)."""
    return cache_dir(subject) / "raw_256hz.zarr"


def spectral_target_path(subject: str) -> Path:
    """Minute log-power field, zarr, shape (n_minutes, C, N_FREQ_BINS)."""
    return cache_dir(subject) / "spectral_target.zarr"


def subject_stats_path(subject: str) -> Path:
    """Train-only normalisation + artifact thresholds."""
    return cache_dir(subject) / "train_stats.json"


def subject_dir(subject: str) -> Path:
    return PER_SUBJECT_DIR / subject


def code_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def package_hash(paths: Sequence[Path]) -> str:
    """Stable hash of the R0.1 source files, for run manifests."""
    h = hashlib.sha256()
    for p in sorted(Path(x) for x in paths):
        if p.is_file():
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def r0_1_source_files() -> List[Path]:
    src = REPO_ROOT / "src" / "topic5_raw_seeg_state"
    scr = REPO_ROOT / "scripts" / "topic5_raw_seeg_state"
    return sorted(list(src.glob("*.py")) + list(scr.glob("*.py")))


def atomic_write_json(path: Path, payload: object) -> None:
    """Write JSON via a temp file + rename so a killed job never truncates."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    tmp.replace(path)
