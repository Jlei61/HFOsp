"""Topic 5.2D v0.2 — frozen prefix/horizon sample construction.

The module turns the parent ``GEOMETRY_ONLY_PCA2`` frame cache (28 SEEG
patients) and the ECoG rank cache (E958 / E1084) into an immutable per-patient
sample cache.  Every downstream model — the two unordered baselines, the
structured low-dimensional operators and the free low-rank control — reads the
*same* arrays, so no arm can silently see a different task.

Contract anchors (spec 2026-08-17):

* §5.1 primary prefix is the first three real rank sets; ``prefix_len=2`` is the
  pre-registered harder sensitivity.
* §5.1 horizons ``h=1..5`` each carry their own eligibility denominator.
* §5.2 already-recruited contacts are masked out of the candidate set; missing
  horizons mask only that horizon and never fabricate a STOP contact.
* §4.3 the autonomous suffix field is the fixed 1–5 step accumulation over the
  contacts that are still available at the end of the prefix — it never reads
  the true remaining event length.
* §7.2 the endpoint is the late-field centroid over the final 20% of rank sets.
* §7.4 the time variable is a spectral-centroid latency proxy, never a
  conduction delay.
* §2.3 the 25% ⊂ 50% ⊂ 100% split-0 subsets are block- and length-stratified and
  strictly nested, identical across every arm.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

HORIZONS: tuple[int, ...] = (1, 2, 3, 4, 5)
PRIMARY_PREFIX_LEN = 3
SENSITIVITY_PREFIX_LEN = 2
PRIMARY_HORIZONS: tuple[int, ...] = (1, 2, 3)
LONG_HORIZONS: tuple[int, ...] = (4, 5)
LATE_FIELD_TAIL_FRACTION = 0.2
BLOCK_GAP_SECONDS = 300.0
LENGTH_BIN_EDGES: tuple[int, ...] = (4, 6, 8, 12)
DATA_FRACTIONS: tuple[int, ...] = (25, 50, 100)
SUBSET_SEED = 20260817

# ``target_available`` convention (audited, identical for every arm including
# the two unordered baselines): the candidate set for horizon ``h`` excludes
# every contact recruited strictly before rank-set index ``prefix_len+h-1``.
# This is a property of the prediction task, not a model input, and it is the
# only place where within-event teacher forcing enters.  The suffix field uses
# the prefix-only mask instead, so the autonomous suffix never depends on the
# true remaining event.
AVAILABILITY_CONTRACT = "teacher_forced_no_repeat_per_horizon"
SUFFIX_MASK_CONTRACT = "prefix_only_no_repeat"


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# patient level input
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PatientInput:
    dataset: str
    patient: str
    contact_names: list[str]
    shafts: list[str]
    coords_3d_mm: np.ndarray
    contacts_xy_mm: np.ndarray
    ranks: np.ndarray
    split: np.ndarray
    event_abs_time: np.ndarray
    event_lag_raw: np.ndarray
    recording_block: np.ndarray
    provenance: dict

    @property
    def n_contacts(self) -> int:
        return len(self.contact_names)

    @property
    def n_events(self) -> int:
        return int(self.ranks.shape[0])


def derive_recording_blocks(event_abs_time: np.ndarray, gap_seconds: float = BLOCK_GAP_SECONDS) -> np.ndarray:
    """Maximal runs of events separated by at most ``gap_seconds``.

    Events are already chronological in both caches; the function asserts that
    rather than re-sorting, because re-sorting would silently break the frozen
    chronological split.
    """
    times = np.asarray(event_abs_time, dtype=np.float64)
    if times.size and not np.all(np.diff(times) >= -1e-6):
        raise ValueError("event_abs_time is not chronological; refusing to reorder a frozen split")
    if times.size == 0:
        return np.zeros(0, dtype=np.int32)
    boundaries = np.concatenate([[0], (np.diff(times) > gap_seconds).astype(np.int32)])
    return np.cumsum(boundaries).astype(np.int32)


def load_seeg_patient(frame_root: Path, patient: str) -> PatientInput:
    directory = Path(frame_root) / patient
    plane = np.load(directory / "plane.npz", allow_pickle=False)
    events = np.load(directory / "events.npz", allow_pickle=True)
    provenance = json.loads((directory / "provenance.json").read_text())
    ranks = np.asarray(events["ranks"], dtype=np.int16)
    abs_time = np.asarray(events["event_abs_time"], dtype=np.float64)
    return PatientInput(
        dataset="SEEG",
        patient=patient,
        contact_names=[str(v) for v in events["contact_names"]],
        shafts=[str(v) for v in events["shafts"]],
        coords_3d_mm=np.asarray(plane["coords_3d_mm"], dtype=np.float64),
        contacts_xy_mm=np.asarray(plane["contacts_xy_mm"], dtype=np.float64),
        ranks=ranks,
        split=np.asarray(events["split"], dtype=np.int8),
        event_abs_time=abs_time,
        event_lag_raw=np.asarray(events["event_lag_raw"], dtype=np.float32),
        recording_block=derive_recording_blocks(abs_time),
        provenance=provenance,
    )


def load_ecog_patient(cache_root: Path, subject: str) -> PatientInput:
    """E958 / E1084 rank cache.

    The ECoG cache stores ties (≈60% of rank sets have cardinality > 1), a
    ``participation`` mask and a per-block index.  Splits are named
    train/validation/test and map onto 0/1/2; there is no ``split == -1``
    model-unseen tier, so the ECoG case series never enters the compact
    confirmation.
    """
    directory = Path(cache_root) / subject
    events = np.load(directory / "events.npz", allow_pickle=True)
    provenance = json.loads((directory / "provenance.json").read_text())
    raw_ranks = np.asarray(events["ranks"], dtype=np.int32)
    participation = np.asarray(events["participation"], dtype=bool)
    ranks = np.where(participation, raw_ranks, -1).astype(np.int16)
    grid = _ecog_grid_coordinates([str(v) for v in events["channel_names"]])
    abs_time = np.asarray(events["event_epoch"], dtype=np.float64)
    order = np.argsort(abs_time, kind="stable")
    return PatientInput(
        dataset="ECOG",
        patient=f"E{subject}",
        contact_names=[str(v) for v in events["channel_names"]],
        shafts=[str(v)[:2] for v in events["channel_names"]],
        coords_3d_mm=np.column_stack([grid, np.zeros(grid.shape[0])]),
        contacts_xy_mm=grid,
        ranks=ranks[order],
        split=np.asarray(events["split"], dtype=np.int8)[order],
        event_abs_time=abs_time[order],
        event_lag_raw=np.asarray(events["lag_sec"], dtype=np.float32)[order],
        recording_block=np.asarray(events["block_index"], dtype=np.int32)[order],
        provenance=provenance,
    )


def _ecog_grid_coordinates(channel_names: list[str]) -> np.ndarray:
    """Physical 8x8 grid layout in millimetres from ``G<row><index>`` names."""
    rows = sorted({name[1] for name in channel_names})
    coords = np.zeros((len(channel_names), 2), dtype=np.float64)
    for position, name in enumerate(channel_names):
        row = rows.index(name[1])
        column = int(name[2:]) - 1
        coords[position] = (column * 10.0, row * 10.0)
    return coords


# ---------------------------------------------------------------------------
# rank-set decomposition
# ---------------------------------------------------------------------------
def rank_sets_for_event(rank_row: np.ndarray) -> list[np.ndarray]:
    """Ordered list of contact-index arrays, one per distinct rank."""
    participating = np.flatnonzero(rank_row >= 0)
    if participating.size == 0:
        return []
    values = rank_row[participating]
    order = np.argsort(values, kind="stable")
    participating = participating[order]
    values = values[order]
    boundaries = np.flatnonzero(np.diff(values)) + 1
    return [np.asarray(part, dtype=np.int64) for part in np.split(participating, boundaries)]


def count_rank_sets(ranks: np.ndarray) -> np.ndarray:
    out = np.zeros(ranks.shape[0], dtype=np.int32)
    for index in range(ranks.shape[0]):
        row = ranks[index]
        participating = row[row >= 0]
        out[index] = np.unique(participating).size
    return out


# ---------------------------------------------------------------------------
# sample construction
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SampleSet:
    patient: str
    dataset: str
    prefix_len: int
    n_contacts: int
    max_cardinality: int
    event_index: np.ndarray
    split: np.ndarray
    recording_block: np.ndarray
    n_rank_sets: np.ndarray
    prefix_sets: np.ndarray
    start_set: np.ndarray
    cumulative_set: np.ndarray
    target_sets: np.ndarray
    target_valid: np.ndarray
    target_available: np.ndarray
    target_cardinality: np.ndarray
    suffix5_field: np.ndarray
    full_suffix_field: np.ndarray
    suffix_eval_mask: np.ndarray
    late_field_centroid: np.ndarray
    latency_proxy: np.ndarray
    latency_valid: np.ndarray
    subset_25: np.ndarray
    subset_50: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(self.event_index.size)

    def split_mask(self, split_value: int) -> np.ndarray:
        return self.split == split_value

    def fraction_mask(self, fraction: int) -> np.ndarray:
        """Split-0 events belonging to the nested ``fraction``% subset."""
        train = self.split == 0
        if fraction == 100:
            return train
        if fraction == 50:
            return train & self.subset_50
        if fraction == 25:
            return train & self.subset_25
        raise ValueError(f"unsupported data fraction {fraction}")


def build_sample_set(patient: PatientInput, prefix_len: int) -> SampleSet:
    n_contacts = patient.n_contacts
    n_horizons = len(HORIZONS)
    rows: list[dict] = []
    for event in range(patient.n_events):
        sets = rank_sets_for_event(patient.ranks[event])
        if len(sets) < prefix_len + 1:
            continue
        rows.append({"event": event, "sets": sets})
    n_samples = len(rows)

    event_index = np.zeros(n_samples, dtype=np.int64)
    n_rank_sets = np.zeros(n_samples, dtype=np.int32)
    prefix_sets = np.zeros((n_samples, prefix_len, n_contacts), dtype=np.uint8)
    start_set = np.zeros((n_samples, n_contacts), dtype=np.uint8)
    cumulative_set = np.zeros((n_samples, n_contacts), dtype=np.uint8)
    target_sets = np.zeros((n_samples, n_horizons, n_contacts), dtype=np.uint8)
    target_valid = np.zeros((n_samples, n_horizons), dtype=bool)
    target_available = np.zeros((n_samples, n_horizons, n_contacts), dtype=np.uint8)
    target_cardinality = np.zeros((n_samples, n_horizons), dtype=np.int16)
    suffix5_field = np.zeros((n_samples, n_contacts), dtype=np.uint8)
    full_suffix_field = np.zeros((n_samples, n_contacts), dtype=np.uint8)
    late_field_centroid = np.zeros((n_samples, 2), dtype=np.float32)
    latency_proxy = np.zeros((n_samples, n_horizons), dtype=np.float32)
    latency_valid = np.zeros((n_samples, n_horizons), dtype=bool)

    for position, row in enumerate(rows):
        event = row["event"]
        sets = row["sets"]
        event_index[position] = event
        n_rank_sets[position] = len(sets)

        recruited = np.zeros(n_contacts, dtype=bool)
        for step in range(prefix_len):
            prefix_sets[position, step, sets[step]] = 1
            recruited[sets[step]] = True
        start_set[position, sets[0]] = 1
        cumulative_set[position] = recruited.astype(np.uint8)

        running = recruited.copy()
        for slot, horizon in enumerate(HORIZONS):
            index = prefix_len + horizon - 1
            target_available[position, slot] = (~running).astype(np.uint8)
            if index < len(sets):
                target_sets[position, slot, sets[index]] = 1
                target_valid[position, slot] = True
                target_cardinality[position, slot] = len(sets[index])
                running[sets[index]] = True

        suffix_slice = sets[prefix_len:prefix_len + len(HORIZONS)]
        for part in suffix_slice:
            suffix5_field[position, part] = 1
        for part in sets[prefix_len:]:
            full_suffix_field[position, part] = 1
        suffix5_field[position][recruited] = 0
        full_suffix_field[position][recruited] = 0

        tail_start = max(prefix_len, int(np.ceil((1.0 - LATE_FIELD_TAIL_FRACTION) * len(sets))))
        tail_start = min(tail_start, len(sets) - 1)
        tail_contacts = np.concatenate(sets[tail_start:])
        late_field_centroid[position] = patient.contacts_xy_mm[tail_contacts].mean(axis=0)

        lag = patient.event_lag_raw[event]
        tau_prefix = float(np.mean(lag[sets[prefix_len - 1]]))
        for slot, horizon in enumerate(HORIZONS):
            index = prefix_len + horizon - 1
            if index < len(sets):
                latency_proxy[position, slot] = float(np.mean(lag[sets[index]])) - tau_prefix
                latency_valid[position, slot] = True

    suffix_eval_mask = (1 - cumulative_set).astype(np.uint8)
    split = patient.split[event_index]
    recording_block = patient.recording_block[event_index]
    subset_25, subset_50 = _nested_fraction_masks(split, recording_block, n_rank_sets, patient.patient)
    max_cardinality = int(target_cardinality.max()) if n_samples else 1

    return SampleSet(
        patient=patient.patient,
        dataset=patient.dataset,
        prefix_len=prefix_len,
        n_contacts=n_contacts,
        max_cardinality=max(1, max_cardinality),
        event_index=event_index,
        split=split,
        recording_block=recording_block,
        n_rank_sets=n_rank_sets,
        prefix_sets=prefix_sets,
        start_set=start_set,
        cumulative_set=cumulative_set,
        target_sets=target_sets,
        target_valid=target_valid,
        target_available=target_available,
        target_cardinality=target_cardinality,
        suffix5_field=suffix5_field,
        full_suffix_field=full_suffix_field,
        suffix_eval_mask=suffix_eval_mask,
        late_field_centroid=late_field_centroid,
        latency_proxy=latency_proxy,
        latency_valid=latency_valid,
        subset_25=subset_25,
        subset_50=subset_50,
    )


def _nested_fraction_masks(
    split: np.ndarray,
    recording_block: np.ndarray,
    n_rank_sets: np.ndarray,
    patient: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Strictly nested 25% ⊂ 50% ⊂ 100% split-0 subsets, block/length stratified.

    The score is a frozen per-sample uniform draw; a subset takes the lowest
    scoring share inside each (recording block, event-length bin) stratum, so
    25% ⊂ 50% holds by construction and every arm receives bit-identical event
    identifiers.
    """
    subset_25 = np.zeros(split.size, dtype=bool)
    subset_50 = np.zeros(split.size, dtype=bool)
    train = np.flatnonzero(split == 0)
    if train.size == 0:
        return subset_25, subset_50
    seed = SUBSET_SEED ^ (int(hashlib.sha256(patient.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF)
    rng = np.random.default_rng(seed)
    scores = rng.random(train.size)
    length_bin = np.digitize(n_rank_sets[train], LENGTH_BIN_EDGES)
    strata = recording_block[train].astype(np.int64) * 1000 + length_bin
    for stratum in np.unique(strata):
        members = np.flatnonzero(strata == stratum)
        order = members[np.argsort(scores[members], kind="stable")]
        for fraction, mask in ((25, subset_25), (50, subset_50)):
            take = int(np.floor(order.size * fraction / 100.0))
            if order.size > 0 and take == 0:
                take = 1
            mask[train[order[:take]]] = True
    return subset_25, subset_50


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------
_ARRAY_FIELDS = (
    "event_index", "split", "recording_block", "n_rank_sets", "prefix_sets",
    "start_set", "cumulative_set", "target_sets", "target_valid",
    "target_available", "target_cardinality", "suffix5_field",
    "full_suffix_field", "suffix_eval_mask", "late_field_centroid",
    "latency_proxy", "latency_valid", "subset_25", "subset_50",
)


def save_sample_set(samples: SampleSet, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: getattr(samples, name) for name in _ARRAY_FIELDS}
    payload["meta"] = np.asarray(
        json.dumps(
            {
                "patient": samples.patient,
                "dataset": samples.dataset,
                "prefix_len": samples.prefix_len,
                "n_contacts": samples.n_contacts,
                "max_cardinality": samples.max_cardinality,
                "horizons": list(HORIZONS),
                "availability_contract": AVAILABILITY_CONTRACT,
                "suffix_mask_contract": SUFFIX_MASK_CONTRACT,
            }
        )
    )
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)
    return sha256_file(path)


def load_sample_set(path: Path) -> SampleSet:
    payload = np.load(path, allow_pickle=False)
    meta = json.loads(str(payload["meta"]))
    fields = {name: payload[name] for name in _ARRAY_FIELDS}
    return SampleSet(
        patient=meta["patient"],
        dataset=meta["dataset"],
        prefix_len=meta["prefix_len"],
        n_contacts=meta["n_contacts"],
        max_cardinality=meta["max_cardinality"],
        **fields,
    )


# ---------------------------------------------------------------------------
# census
# ---------------------------------------------------------------------------
def horizon_census(samples: SampleSet) -> list[dict]:
    rows = []
    for split_value, split_name in ((0, "train"), (1, "calibration"), (2, "development_test"), (-1, "model_unseen")):
        in_split = samples.split == split_value
        if not in_split.any():
            continue
        base = {
            "patient": samples.patient,
            "dataset": samples.dataset,
            "prefix_len": samples.prefix_len,
            "split": split_name,
            "n_eligible_events": int(in_split.sum()),
            "rank_sets_median": float(np.median(samples.n_rank_sets[in_split])),
            "rank_sets_min": int(samples.n_rank_sets[in_split].min()),
            "rank_sets_max": int(samples.n_rank_sets[in_split].max()),
        }
        for slot, horizon in enumerate(HORIZONS):
            valid = in_split & samples.target_valid[:, slot]
            base[f"h{horizon}_denominator"] = int(valid.sum())
            base[f"h{horizon}_cardinality_median"] = (
                float(np.median(samples.target_cardinality[valid, slot])) if valid.any() else float("nan")
            )
            base[f"h{horizon}_available_median"] = (
                float(np.median(samples.target_available[valid, slot].sum(axis=1))) if valid.any() else float("nan")
            )
            # A decision whose candidate set has exactly ``n`` contacts left is
            # forced: its exact-subset likelihood is identically zero and it
            # carries no information.  Small montages hit this at later horizons,
            # so the fraction is reported next to every denominator.
            base[f"h{horizon}_forced_fraction"] = (
                float(
                    (
                        samples.target_available[valid, slot].sum(axis=1)
                        <= samples.target_cardinality[valid, slot]
                    ).mean()
                )
                if valid.any() else float("nan")
            )
        base["suffix_denominator"] = int(in_split.sum())
        base["suffix5_positive_rate"] = float(
            samples.suffix5_field[in_split].sum() / max(1, samples.suffix_eval_mask[in_split].sum())
        )
        rows.append(base)
    return rows
