"""Yuquan SEEG record loader for ictal-window pipelines.

Mirrors the consumer surface of :func:`src.preprocessing.load_epilepsiae_block`
so that :func:`src.ictal_onset_extraction.extract_seizure_window` can route
to either dataset transparently.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.preprocessing import PreprocessingResult, _build_bipolar_pairs


# Single uppercase letter, optional apostrophe (contralateral SEEG probes use
# A', B', etc.), 1-3 digits. Deliberately rejects multi-letter prefixes like
# DC10 / EMG1 / ECG that real yuquan EDFs emit as auxiliary channels.
_INTRACRANIAL_NAME_RE = re.compile(r"^[A-Z]'?\d{1,3}$")

_SCALP_REF = frozenset({"A1", "A2"})
_SCALP_10_20 = frozenset({
    # Frontal
    "Fp1", "Fp2", "Fz",
    "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10",
    # Central / temporal
    "C1", "C2", "C3", "C4", "C5", "C6", "Cz",
    "T3", "T4", "T5", "T6",
    "FT9", "FT10",
    # Parietal / occipital
    "P3", "P4", "Pz",
    "O1", "O2",
})
_AUX_PREFIXES = (
    "DC", "EKG", "ECG", "EMG", "OSAT", "PULSE", "SPO2", "BR", "PR", "BP",
)


def normalize_yuquan_channel_name(raw: str) -> str:
    """Strip 'EEG '/'POL ' prefix and '-Ref' suffix; collapse whitespace.

    Apostrophes (e.g., POL A'1 -> A'1) are preserved — they encode
    contralateral SEEG probes and must not be lost.
    """
    s = raw.strip()
    for prefix in ("EEG ", "POL "):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    if s.endswith("-Ref"):
        s = s[:-len("-Ref")]
    return s.strip()


def classify_yuquan_channel(raw: str) -> str:
    """Return one of {'intracranial', 'scalp', 'scalp_ref', 'aux'}.

    Order of checks matters: aux prefixes are checked BEFORE the
    intracranial regex because real yuquan EDFs emit DC10/EMG1/ECG as
    auxiliary channels whose normalized names (DC10, EMG1, ECG) would
    otherwise pass a naive [A-Z]{1,3}\\d{1,2} pattern.
    """
    norm = normalize_yuquan_channel_name(raw)
    if norm in _SCALP_REF:
        return "scalp_ref"
    if norm in _SCALP_10_20:
        return "scalp"
    if any(norm.startswith(p) for p in _AUX_PREFIXES):
        return "aux"
    if _INTRACRANIAL_NAME_RE.match(norm):
        return "intracranial"
    return "aux"


def _select_intracranial_indices(ch_names: List[str]) -> Tuple[List[int], List[str]]:
    keep_idx: List[int] = []
    keep_names: List[str] = []
    for i, raw in enumerate(ch_names):
        if classify_yuquan_channel(raw) == "intracranial":
            keep_idx.append(i)
            keep_names.append(normalize_yuquan_channel_name(raw))
    return keep_idx, keep_names


def load_yuquan_record(
    edf_path: Path | str,
    *,
    reference: str = "car",
    segment_sec: float = 200.0,  # accepted for API parity; not used (preload=True)
    intracranial_only: bool = True,
) -> PreprocessingResult:
    """Load a yuquan SEEG EDF, filter to intracranial channels, apply reference.

    Parameters
    ----------
    edf_path : str or Path
        Path to the EDF file.
    reference : str
        'car' (default) for common-average reference (zero-mean per sample),
        'bipolar' for adjacent-contact pairs within each probe.
    intracranial_only : bool
        If True (default) drop scalp/scalp_ref/aux channels via
        :func:`classify_yuquan_channel`. Yuquan analyses always want this.
    """
    import mne  # local import to keep module import cheap

    edf_path = Path(edf_path)
    if not edf_path.exists():
        raise FileNotFoundError(edf_path)

    raw = mne.io.read_raw_edf(
        str(edf_path), preload=True, verbose=False, encoding="latin1"
    )
    sfreq = float(raw.info["sfreq"])
    data = raw.get_data().astype(np.float64, copy=False)
    ch_names_raw = list(raw.ch_names)

    if intracranial_only:
        keep_idx, keep_names = _select_intracranial_indices(ch_names_raw)
        if not keep_idx:
            raise ValueError(
                f"No intracranial channels found in {edf_path.name}; "
                f"first 8 raw channels: {ch_names_raw[:8]}"
            )
        data = data[keep_idx]
        ch_names = keep_names
    else:
        ch_names = [normalize_yuquan_channel_name(c) for c in ch_names_raw]

    bipolar_pairs = None
    if reference == "car":
        data = data - data.mean(axis=0, keepdims=True)
        out_names = ch_names
        ref_type = "car"
    elif reference == "bipolar":
        pairs, pair_names = _build_bipolar_pairs(ch_names)
        if not pairs:
            raise ValueError(
                f"No valid bipolar pairs in {edf_path.name} after filtering"
            )
        data = np.stack([data[a] - data[b] for a, b in pairs], axis=0)
        out_names = pair_names
        bipolar_pairs = [
            (ch_names[a], ch_names[b]) for a, b in pairs
        ]
        ref_type = "bipolar"
    else:
        raise ValueError(f"reference must be 'car' or 'bipolar', got {reference!r}")

    return PreprocessingResult(
        data=data.astype(np.float64, copy=False),
        sfreq=sfreq,
        ch_names=out_names,
        original_ch_names=list(ch_names_raw),
        bipolar_pairs=bipolar_pairs,
        reference_type=ref_type,
    )
