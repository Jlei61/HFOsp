"""Native-rate readers and the three reference views of one group event.

A group event is stored once as a pointer (file, sample range, montage) and is
rebuilt on demand into three explicitly-labelled views:

``detector``
    the montage the HFO detector actually ran on -- adjacent bipolar for Yuquan
    (``reference_type='bipolar'`` plus ``bipolar_pairs`` in ``*_gpu.npz``), and
    the global common average over retained intracranial channels for Epilepsiae
    (``epilepsiae_detectHFOs.avg_rerefAndDrop_eeg``).
``bipolar``
    adjacent bipolar on the contact's own shaft.  Identical to ``detector`` for
    Yuquan, which is recorded rather than silently duplicated.
``shaft_car``
    the contact minus the mean of its own shaft.

Every view carries its own reference token so no consumer can concatenate two
montages as if they were the same signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np


_CLEAN_PREFIXES = ("POL ", "EEG ", "SEEG ")
_CLEAN_SUFFIXES = ("-Ref", "-REF", "-ref")
_CONTACT_RE = re.compile(r"^([A-Za-z]+'?)\s*(\d+)$")

VIEW_DETECTOR = "detector"
VIEW_BIPOLAR = "bipolar"
VIEW_SHAFT_CAR = "shaft_car"
VIEW_NAMES = (VIEW_DETECTOR, VIEW_BIPOLAR, VIEW_SHAFT_CAR)


def clean_contact(name: str) -> str:
    text = str(name).strip()
    for prefix in _CLEAN_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):]
    for suffix in _CLEAN_SUFFIXES:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text.strip().upper()


def split_contact(name: str) -> tuple[str, int] | None:
    match = _CONTACT_RE.match(clean_contact(name))
    if match is None:
        return None
    return match.group(1).upper(), int(match.group(2))


@dataclass(frozen=True)
class UniverseContact:
    """One row of the patient's group-event contact universe."""

    lagpat_label: str
    detector_label: str
    anode: str
    cathode: str | None
    shaft: str
    number: int


@dataclass(frozen=True)
class ContactUniverse:
    dataset: str
    subject: str
    detector_reference: str
    contacts: tuple[UniverseContact, ...]
    bipolar_equals_detector: bool

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(c.lagpat_label for c in self.contacts)

    def __len__(self) -> int:
        return len(self.contacts)


def build_contact_universe(
    dataset: str,
    subject: str,
    lagpat_labels: Sequence[str],
    detector_labels: Sequence[str],
    detector_reference: str,
    bipolar_pairs: Sequence[Sequence[str]] | None = None,
) -> ContactUniverse:
    """Resolve each lagPat row to its true montage channel and single contacts.

    Yuquan lagPat rows store only the *anode* of an adjacent bipolar detector
    channel, so ``E11`` is really ``E11-E12``.  Epilepsiae rows are single CAR
    contacts.  Refusing to guess here is what keeps the two datasets from being
    silently pooled under one contact semantics.
    """

    dataset = str(dataset).lower()
    detector_by_clean = {clean_contact(d): str(d) for d in detector_labels}
    pair_by_anode: dict[str, tuple[str, str]] = {}
    if bipolar_pairs:
        for pair in bipolar_pairs:
            anode, cathode = clean_contact(pair[0]), clean_contact(pair[1])
            pair_by_anode.setdefault(anode, (anode, cathode))

    contacts: list[UniverseContact] = []
    for label in lagpat_labels:
        clean = clean_contact(label)
        parsed = split_contact(clean)
        if parsed is None:
            raise ValueError(f"{subject}: unparsable lagPat contact {label!r}")
        shaft, number = parsed
        if dataset == "yuquan":
            if clean in pair_by_anode:
                anode, cathode = pair_by_anode[clean]
            else:
                anode, cathode = clean, f"{shaft}{number + 1}"
            detector_label = detector_by_clean.get(
                f"{anode}-{cathode}", f"{anode}-{cathode}"
            )
        else:
            anode, cathode = clean, None
            detector_label = detector_by_clean.get(clean, clean)
        contacts.append(
            UniverseContact(
                lagpat_label=str(label),
                detector_label=detector_label,
                anode=anode,
                cathode=cathode,
                shaft=shaft,
                number=number,
            )
        )
    return ContactUniverse(
        dataset=dataset,
        subject=str(subject),
        detector_reference=str(detector_reference),
        contacts=tuple(contacts),
        bipolar_equals_detector=(dataset == "yuquan"),
    )


class BlockReader:
    """Native-rate windowed access to one recording block."""

    native_rate_hz: float
    n_samples: int
    labels: tuple[str, ...]

    def read(self, start: int, stop: int, channels: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError


class EpilepsiaeBlockReader(BlockReader):
    """``.head`` + ``.data`` pair; int16 samples stored sample-major in µV/CF."""

    def __init__(self, data_path: Path, head_path: Path):
        meta = dict(
            line.split("=", 1)
            for line in Path(head_path).read_text().strip().splitlines()
            if "=" in line
        )
        self.n_samples = int(meta["num_samples"])
        self.n_channels = int(meta["num_channels"])
        self.native_rate_hz = float(meta["sample_freq"])
        self.conversion_factor = float(meta["conversion_factor"])
        raw_names = [n.strip() for n in meta["elec_names"].strip("[]").split(",")]
        self.labels = tuple(clean_contact(n) for n in raw_names)
        self.raw_labels = tuple(raw_names)
        expected = self.n_samples * self.n_channels * 2
        actual = Path(data_path).stat().st_size
        if actual != expected:
            raise ValueError(
                f"{data_path}: {actual} bytes but head implies {expected}"
            )
        self._mm = np.memmap(
            data_path, dtype="<i2", mode="r", shape=(self.n_samples, self.n_channels)
        )
        self.index = {label: i for i, label in enumerate(self.labels)}

    def read(self, start: int, stop: int, channels: np.ndarray | None = None) -> np.ndarray:
        start = max(0, int(start))
        stop = min(int(stop), self.n_samples)
        n_out = self.n_channels if channels is None else len(channels)
        if stop <= start:
            return np.zeros((n_out, 0), dtype=np.float32)
        window = self._mm[start:stop, :]
        if channels is not None:
            window = window[:, channels]
        chunk = np.asarray(window, dtype=np.float32).T
        return chunk * np.float32(self.conversion_factor)


class YuquanEdfBlockReader(BlockReader):
    """EDF record layout; SEEG signals only, scaled to physical units."""

    def __init__(self, edf_path: Path, header: Mapping[str, object]):
        self.header = header
        self.spr = int(header["spr"])
        self.native_rate_hz = float(header["sfreq"])
        self.n_records = int(header["n_records"])
        self.n_samples = self.n_records * self.spr
        self.labels = tuple(clean_contact(l) for l in header["seeg_labels"])  # type: ignore[index]
        self.raw_labels = tuple(str(l) for l in header["seeg_labels"])  # type: ignore[index]
        self._offsets = np.asarray(header["sample_offsets"], dtype=np.int64)
        self._gains = np.asarray(header["gains"], dtype=np.float32)
        self._dc = np.asarray(header["offsets"], dtype=np.float32)
        record_total_samples = int(header["record_total_bytes"]) // 2  # type: ignore[arg-type]
        self._mm = np.memmap(
            edf_path,
            dtype="<i2",
            mode="r",
            offset=int(header["header_n_bytes"]),  # type: ignore[arg-type]
            shape=(self.n_records, record_total_samples),
        )
        self.index = {label: i for i, label in enumerate(self.labels)}

    def read(self, start: int, stop: int, channels: np.ndarray | None = None) -> np.ndarray:
        start = max(0, int(start))
        stop = min(int(stop), self.n_samples)
        picks = np.arange(len(self.labels)) if channels is None else np.asarray(channels)
        if stop <= start:
            return np.zeros((picks.size, 0), dtype=np.float32)
        r0 = start // self.spr
        r1 = (stop - 1) // self.spr + 1
        block = np.asarray(self._mm[r0:r1, :], dtype=np.float32)
        out = np.empty((picks.size, (r1 - r0) * self.spr), dtype=np.float32)
        for k, i in enumerate(picks):
            off = self._offsets[i]
            out[k] = block[:, off : off + self.spr].reshape(-1)
        out *= self._gains[picks, None]
        out += self._dc[picks, None]
        lo = start - r0 * self.spr
        return out[:, lo : lo + (stop - start)]


def open_block_reader(dataset: str, raw_path: Path, head_path: Path | None) -> BlockReader:
    if str(dataset).lower() == "epilepsiae":
        if head_path is None:
            raise ValueError("epilepsiae blocks require a .head file")
        return EpilepsiaeBlockReader(Path(raw_path), Path(head_path))
    import sys

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.preprocessing import _parse_edf_header_for_streaming

    return YuquanEdfBlockReader(Path(raw_path), _parse_edf_header_for_streaming(Path(raw_path)))


@dataclass(frozen=True)
class ViewPlan:
    """Which raw rows one block actually needs, and where they land after subsetting."""

    picks: np.ndarray
    anode: np.ndarray
    cathode: np.ndarray
    neighbour: np.ndarray
    neighbour_sign: np.ndarray
    shaft_members: tuple[tuple[int, ...], ...]
    car: np.ndarray | None


def build_view_plan(
    reader: BlockReader,
    universe: "ContactUniverse",
    car_channel_indices: Sequence[int] | None = None,
) -> ViewPlan:
    index: Mapping[str, int] = reader.index  # type: ignore[attr-defined]
    needed: list[int] = []
    seen: dict[int, int] = {}

    def _slot(raw_idx: int | None) -> int:
        if raw_idx is None:
            return -1
        if raw_idx not in seen:
            seen[raw_idx] = len(needed)
            needed.append(raw_idx)
        return seen[raw_idx]

    if universe.dataset == "epilepsiae":
        if car_channel_indices is None:
            raise ValueError("epilepsiae detector view needs the producer's CAR channel set")
        car = np.array([_slot(int(i)) for i in car_channel_indices], dtype=np.int64)
    else:
        car = None

    n = len(universe)
    anode = np.full(n, -1, dtype=np.int64)
    cathode = np.full(n, -1, dtype=np.int64)
    neighbour = np.full(n, -1, dtype=np.int64)
    sign = np.ones(n, dtype=np.float32)
    shafts: list[tuple[int, ...]] = []
    shaft_cache: dict[str, tuple[int, ...]] = {}
    for ci, contact in enumerate(universe.contacts):
        anode[ci] = _slot(index.get(contact.anode))
        if contact.cathode:
            cathode[ci] = _slot(index.get(contact.cathode))
        if not universe.bipolar_equals_detector:
            nb = index.get(f"{contact.shaft}{contact.number + 1}")
            if nb is None:
                nb = index.get(f"{contact.shaft}{contact.number - 1}")
                sign[ci] = -1.0
            neighbour[ci] = _slot(nb)
        if contact.shaft not in shaft_cache:
            shaft_cache[contact.shaft] = tuple(
                _slot(i) for i in _shaft_members(reader, contact.shaft)
            )
        shafts.append(shaft_cache[contact.shaft])
    return ViewPlan(
        picks=np.asarray(needed, dtype=np.int64),
        anode=anode,
        cathode=cathode,
        neighbour=neighbour,
        neighbour_sign=sign,
        shaft_members=tuple(shafts),
        car=car,
    )


def _shaft_members(reader: BlockReader, shaft: str) -> list[int]:
    return [
        idx
        for label, idx in reader.index.items()  # type: ignore[attr-defined]
        if (parsed := split_contact(label)) is not None and parsed[0] == shaft
    ]


def build_event_views(
    reader: BlockReader,
    universe: ContactUniverse,
    start_sample: int,
    stop_sample: int,
    *,
    car_channel_indices: Sequence[int] | None = None,
    plan: ViewPlan | None = None,
) -> dict[str, np.ndarray]:
    """Return ``{view_name: (n_universe_contacts, n_samples) float32}``.

    ``car_channel_indices`` names the exact channel set the Epilepsiae producer
    averaged over (the retained intracranial channels); passing ``None`` there
    would quietly average a different channel set than the detector did.
    Passing a reusable ``plan`` avoids re-reading channels no view consumes.
    """

    if plan is None:
        plan = build_view_plan(reader, universe, car_channel_indices)
    raw = reader.read(start_sample, stop_sample, plan.picks)
    n_time = raw.shape[1]
    n_contacts = len(universe)
    out = {
        name: np.full((n_contacts, n_time), np.nan, dtype=np.float32)
        for name in VIEW_NAMES
    }
    car = raw[plan.car].mean(axis=0) if plan.car is not None else None

    shaft_mean_cache: dict[tuple[int, ...], np.ndarray] = {}
    for ci in range(n_contacts):
        if plan.anode[ci] < 0:
            continue
        anode = raw[plan.anode[ci]]

        if car is not None:
            out[VIEW_DETECTOR][ci] = anode - car
        elif plan.cathode[ci] >= 0:
            out[VIEW_DETECTOR][ci] = anode - raw[plan.cathode[ci]]

        if universe.bipolar_equals_detector:
            out[VIEW_BIPOLAR][ci] = out[VIEW_DETECTOR][ci]
        elif plan.neighbour[ci] >= 0:
            out[VIEW_BIPOLAR][ci] = plan.neighbour_sign[ci] * (anode - raw[plan.neighbour[ci]])

        members = plan.shaft_members[ci]
        if members not in shaft_mean_cache:
            shaft_mean_cache[members] = (
                raw[np.asarray(members, dtype=np.int64)].mean(axis=0)
                if members
                else np.full(n_time, np.nan, dtype=np.float32)
            )
        out[VIEW_SHAFT_CAR][ci] = anode - shaft_mean_cache[members]

    return out


@dataclass(frozen=True)
class MontageResolution:
    """Where the detector montage came from, never silently guessed."""

    detector_labels: tuple[str, ...]
    reference: str
    bipolar_pairs: tuple[tuple[str, str], ...] | None
    provenance: str
    unresolvable: tuple[str, ...]


def resolve_montage(
    dataset: str,
    lagpat_labels: Sequence[str],
    gpu_path: Path | None,
    raw_path: Path,
) -> MontageResolution:
    """Recover the detector montage for one block.

    ``*_gpu.npz`` is authoritative when present.  For Yuquan it is missing on 83
    blocks (7 patients, 181k events), and the adjacent-bipolar anode rule can be
    re-derived from the recording's own channel list instead: every one of the
    972 recorded pairs in this cohort is ``(n, n+1)`` on one shaft, and on the
    single ``pengzihang`` block that does keep its ``_gpu.npz`` the derived pairs
    reproduce the recorded ones for all 12 lagPat rows.  The provenance token
    keeps the derived case distinguishable from the read case forever.
    """

    dataset = str(dataset).lower()
    if gpu_path is not None and Path(gpu_path).exists():
        with np.load(gpu_path, allow_pickle=True) as gpu:
            files = list(gpu.files)
            labels = tuple(str(v) for v in gpu["chns_names"]) if "chns_names" in files else ()
            pairs = (
                tuple(tuple(str(y) for y in x) for x in gpu["bipolar_pairs"])
                if "bipolar_pairs" in files
                else None
            )
            if "reference_type" in files:
                reference = str(np.asarray(gpu["reference_type"]).reshape(-1)[0])
            elif dataset == "epilepsiae":
                reference = "car_global_intracranial_from_producer"
            else:
                reference = "bipolar"
        return MontageResolution(labels, reference, pairs, "gpu_npz", ())

    if dataset != "yuquan":
        raise FileNotFoundError(
            f"{dataset} block needs its _gpu.npz to name the detector montage: {gpu_path}"
        )

    import sys

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.preprocessing import _parse_edf_header_for_streaming

    header = _parse_edf_header_for_streaming(Path(raw_path))
    available = {clean_contact(label) for label in header["seeg_labels"]}
    pairs_out: list[tuple[str, str]] = []
    unresolvable: list[str] = []
    for label in lagpat_labels:
        parsed = split_contact(label)
        if parsed is None:
            unresolvable.append(str(label))
            continue
        shaft, number = parsed
        anode, cathode = f"{shaft}{number}", f"{shaft}{number + 1}"
        if anode in available and cathode in available:
            pairs_out.append((anode, cathode))
        else:
            unresolvable.append(str(label))
    return MontageResolution(
        tuple(f"{a}-{b}" for a, b in pairs_out),
        "bipolar",
        tuple(pairs_out),
        "derived_from_label_rule_no_gpu",
        tuple(unresolvable),
    )
