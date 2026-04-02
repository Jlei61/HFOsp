"""Epilepsiae dataset inventory and synchrony-manifest utilities.

This module formalizes the raw-data/SQL/artifact contract discovered in PR1.5:
  - raw blocks live under ``inv*`` / ``epilepsiae_3patient``
  - metadata truth lives in ``all_data_sqls/*.sql``
  - interictal artifacts live under ``interilca_inter_results/all_data_lns``

The goal is to keep this logic out of one-off scripts so downstream analyses
can build subject lists and timelines from one reusable source of truth.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo


SCALP_OR_AUX_CHANNELS = {
    "FP1",
    "FP2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "FZ",
    "CZ",
    "PZ",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "T1",
    "T2",
    "EOG",
    "EOG1",
    "EOG2",
    "ECG",
    "EMG",
    "EMG1",
    "EMG2",
    "HP1",
    "HP2",
    "HP3",
    "PHO",
    "FT9",
    "FT10",
    "T7",
    "T8",
    "P7",
    "P8",
}

NONTRIVIAL_GAP_SEC = 2.0
DEFAULT_MIN_SYNC_INTERVAL_SEC = 3.0 * 3600.0
DEFAULT_EPILEPSIAE_TIMEZONE = "Europe/Berlin"
DEFAULT_DAY_START_HOUR = 8
DEFAULT_NIGHT_START_HOUR = 20
DEFAULT_TIMEZONE_BY_HOSPITAL = {"UKLFR": DEFAULT_EPILEPSIAE_TIMEZONE}
DEFAULT_TIMEZONE_BY_PATIENT_PREFIX = {"FR": DEFAULT_EPILEPSIAE_TIMEZONE}


@dataclass(frozen=True)
class EpilepsiaePaths:
    data_root: Path = Path("/mnt/epilepsia_data")
    sql_root: Path = Path("/mnt/epilepsia_data/all_data_sqls")
    artifact_root: Path = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")
    raw_roots: Tuple[Path, ...] = (
        Path("/mnt/epilepsia_data/inv"),
        Path("/mnt/epilepsia_data/inv2"),
        Path("/mnt/epilepsia_data/inv_1_part"),
        Path("/mnt/epilepsia_data/epilepsiae_3patient"),
    )


@dataclass(frozen=True)
class RawBlockFiles:
    stem: str
    head_path: Optional[Path]
    data_path: Optional[Path]
    raw_gpu_path: Optional[Path]


@dataclass(frozen=True)
class EpilepsiaeTimeConfig:
    """
    Explicit timezone/day-night contract for Epilepsiae.

    Search order:
      recording override -> subject/patient override -> hospital map
      -> patient prefix map -> dataset default -> unknown
    """

    timezone_default: Optional[str] = DEFAULT_EPILEPSIAE_TIMEZONE
    timezone_by_hospital: Dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_TIMEZONE_BY_HOSPITAL)
    )
    timezone_by_patient_prefix: Dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_TIMEZONE_BY_PATIENT_PREFIX)
    )
    timezone_overrides: Dict[str, str] = field(default_factory=dict)
    recording_timezone_overrides: Dict[str, str] = field(default_factory=dict)
    day_start_hour: int = DEFAULT_DAY_START_HOUR
    night_start_hour: int = DEFAULT_NIGHT_START_HOUR


@dataclass
class EpilepsiaeInventory:
    subject_rows: List[Dict[str, object]]
    recording_rows: List[Dict[str, object]]
    block_rows: List[Dict[str, object]]
    seizure_rows: List[Dict[str, object]]
    summary: Dict[str, object]


def _parse_sql_values(line: str) -> List[str]:
    match = re.search(r"VALUES\s*\((.*)\);?$", line.strip())
    if not match:
        raise ValueError(f"Could not parse SQL VALUES from line: {line[:120]}")
    payload = match.group(1)
    return next(
        csv.reader([payload], delimiter=",", quotechar="'", skipinitialspace=True)
    )


def _normalize_sql_value(value: str) -> Optional[str]:
    val = value.strip()
    if val.upper() == "NULL":
        return None
    return val


def _sql_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    upper = value.upper()
    if upper == "TRUE":
        return True
    if upper == "FALSE":
        return False
    return None


def _parse_ts(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt).timestamp()
        except ValueError:
            continue
    raise ValueError(f"Unsupported timestamp: {value}")


def _validate_time_config(time_config: EpilepsiaeTimeConfig) -> None:
    if not (0 <= int(time_config.day_start_hour) <= 23):
        raise ValueError("day_start_hour must be in [0, 23].")
    if not (0 <= int(time_config.night_start_hour) <= 23):
        raise ValueError("night_start_hour must be in [0, 23].")
    if int(time_config.day_start_hour) >= int(time_config.night_start_hour):
        raise ValueError("day_start_hour must be smaller than night_start_hour.")


def _day_night_rule_string(time_config: EpilepsiaeTimeConfig) -> str:
    return (
        f"day={int(time_config.day_start_hour):02d}:00-"
        f"{int(time_config.night_start_hour):02d}:00 local"
    )


def _epoch_to_local_hour(epoch_ts: float, timezone_name: str) -> int:
    dt_utc = datetime.fromtimestamp(float(epoch_ts), tz=timezone.utc)
    return int(dt_utc.astimezone(ZoneInfo(str(timezone_name))).hour)


def _classify_day_night(local_hour: int, time_config: EpilepsiaeTimeConfig) -> str:
    if int(time_config.day_start_hour) <= int(local_hour) < int(time_config.night_start_hour):
        return "day"
    return "night"


def _annotate_epoch_local_time(
    epoch_ts: Optional[float],
    timezone_name: Optional[str],
    time_config: EpilepsiaeTimeConfig,
) -> Dict[str, object]:
    if epoch_ts is None or not timezone_name:
        return {"local_hour": None, "day_night": None}
    local_hour = _epoch_to_local_hour(float(epoch_ts), str(timezone_name))
    return {
        "local_hour": local_hour,
        "day_night": _classify_day_night(local_hour, time_config),
    }


def resolve_epilepsiae_timezone(
    *,
    subject: str,
    patient_code: Optional[str] = None,
    hospital: Optional[str] = None,
    recording_id: Optional[str] = None,
    time_config: Optional[EpilepsiaeTimeConfig] = None,
) -> Dict[str, object]:
    cfg = time_config or EpilepsiaeTimeConfig()
    _validate_time_config(cfg)

    if recording_id:
        for key in (f"{subject}/{recording_id}", str(recording_id)):
            timezone_name = cfg.recording_timezone_overrides.get(key)
            if timezone_name:
                return {
                    "timezone_name": timezone_name,
                    "timezone_source": "recording_override",
                    "timezone_known": True,
                    "reliable_without_override": False,
                }

    for key in (str(subject), str(patient_code or "")):
        if not key:
            continue
        timezone_name = cfg.timezone_overrides.get(key)
        if timezone_name:
            return {
                "timezone_name": timezone_name,
                "timezone_source": "subject_override",
                "timezone_known": True,
                "reliable_without_override": False,
            }

    if hospital:
        timezone_name = cfg.timezone_by_hospital.get(str(hospital))
        if timezone_name:
            return {
                "timezone_name": timezone_name,
                "timezone_source": "hospital",
                "timezone_known": True,
                "reliable_without_override": True,
            }

    patient_prefix = str(patient_code or "").split("_")[0]
    if patient_prefix:
        timezone_name = cfg.timezone_by_patient_prefix.get(patient_prefix)
        if timezone_name:
            return {
                "timezone_name": timezone_name,
                "timezone_source": "patient_code_prefix",
                "timezone_known": True,
                "reliable_without_override": True,
            }

    if cfg.timezone_default:
        return {
            "timezone_name": cfg.timezone_default,
            "timezone_source": "dataset_default",
            "timezone_known": True,
            "reliable_without_override": False,
        }

    return {
        "timezone_name": None,
        "timezone_source": "unknown",
        "timezone_known": False,
        "reliable_without_override": False,
    }


def _parse_duration_commentary(
    commentary: Optional[str],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not commentary:
        return None, None, None
    match = re.search(
        r"duration:\s*([0-9.]+)\s*\(net\),\s*([0-9.]+)\s*\(gross\),\s*([0-9.]+)%\s*complete",
        commentary,
    )
    if not match:
        return None, None, None
    return float(match.group(1)), float(match.group(2)), float(match.group(3))


def _read_head_info(head_path: Path) -> Dict[str, object]:
    info: Dict[str, object] = {}
    with open(head_path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            info[key] = value
    elec_names = str(info.get("elec_names", ""))[1:-1]
    ch_names = [x.strip() for x in elec_names.split(",") if x.strip()]
    return {
        "start_ts": _parse_ts(str(info["start_ts"])),
        "duration_sec": float(info["duration_in_sec"]),
        "sample_freq": float(info["sample_freq"]),
        "n_channels": int(info["num_channels"]),
        "channel_names": ch_names,
    }


def _classify_channels(channel_names: Iterable[str]) -> Dict[str, object]:
    names = [str(x).strip() for x in channel_names if str(x).strip()]
    scalp = [x for x in names if x.upper() in SCALP_OR_AUX_CHANNELS]
    intracranial = [x for x in names if x.upper() not in SCALP_OR_AUX_CHANNELS]
    return {
        "has_eeg": bool(scalp),
        "scalp_or_aux_channels": len(scalp),
        "intracranial_channels": len(intracranial),
        "sample_intracranial": "|".join(intracranial[:8]),
    }


def _subject_from_sql_name(path: Path) -> str:
    match = re.match(r"pat_(\d+)02_", path.name)
    if not match:
        raise ValueError(f"Unexpected SQL filename: {path.name}")
    return match.group(1)


def _collect_raw_blocks(subject: str, paths: EpilepsiaePaths) -> Dict[str, RawBlockFiles]:
    by_stem: Dict[str, RawBlockFiles] = {}
    patient_dirname = f"pat_{subject}02"
    for raw_root in paths.raw_roots:
        patient_dir = raw_root / patient_dirname
        if not patient_dir.exists():
            continue
        for adm_dir in sorted(
            p for p in patient_dir.iterdir() if p.is_dir() and p.name.startswith("adm_")
        ):
            for rec_dir in sorted(
                p for p in adm_dir.iterdir() if p.is_dir() and p.name.startswith("rec_")
            ):
                for head_path in sorted(rec_dir.glob("*.head")):
                    stem = head_path.stem
                    data_path = rec_dir / f"{stem}.data"
                    gpu_path = rec_dir / f"{stem}_gpu.npz"
                    by_stem[stem] = RawBlockFiles(
                        stem=stem,
                        head_path=head_path,
                        data_path=data_path if data_path.exists() else None,
                        raw_gpu_path=gpu_path if gpu_path.exists() else None,
                    )
    return by_stem


def _parse_sql_subject(sql_path: Path) -> Dict[str, object]:
    subject = _subject_from_sql_name(sql_path)
    patient_code = None
    admission: Dict[str, object] = {}
    recordings: Dict[str, Dict[str, object]] = {}
    blocks: List[Dict[str, object]] = []
    seizures: List[Dict[str, object]] = []

    with open(sql_path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("INSERT INTO patient "):
                vals = [_normalize_sql_value(x) for x in _parse_sql_values(stripped)]
                patient_code = vals[1]
            elif stripped.startswith("INSERT INTO admission "):
                vals = [_normalize_sql_value(x) for x in _parse_sql_values(stripped)]
                admission = {
                    "admission_id": vals[0],
                    "hospital": vals[4],
                    "seeg": _sql_bool(vals[7]),
                    "ieeg": _sql_bool(vals[8]),
                    "admission_date": vals[2],
                }
            elif stripped.startswith("INSERT INTO recording "):
                vals = [_normalize_sql_value(x) for x in _parse_sql_values(stripped)]
                net_sec, gross_sec, completeness = _parse_duration_commentary(vals[12])
                recordings[str(vals[0])] = {
                    "recording_id": str(vals[0]),
                    "recording_str_id": vals[1],
                    "begin": vals[3],
                    "end": vals[4],
                    "begin_epoch": _parse_ts(vals[3]),
                    "end_epoch": _parse_ts(vals[4]),
                    "duration_sec_sql": float(vals[5]) if vals[5] else None,
                    "blocks_sql": int(vals[6]) if vals[6] else None,
                    "channels_sql": int(vals[7]) if vals[7] else None,
                    "sample_rate_sql": float(vals[8]) if vals[8] else None,
                    "commentary": vals[12],
                    "net_duration_sec": net_sec,
                    "gross_duration_sec": gross_sec,
                    "completeness_pct": completeness,
                }
            elif stripped.startswith("INSERT INTO block "):
                vals = [_normalize_sql_value(x) for x in _parse_sql_values(stripped)]
                blocks.append(
                    {
                        "block_id": str(vals[0]),
                        "recording_id": str(vals[1]),
                        "eeg_file_id": str(vals[2]),
                        "block_no": int(vals[3]) if vals[3] else None,
                        "samples": int(vals[4]) if vals[4] else None,
                        "sample_bytes": int(vals[5]) if vals[5] else None,
                        "channels_sql": int(vals[6]) if vals[6] else None,
                        "factor": float(vals[7]) if vals[7] else None,
                        "begin": vals[8],
                        "end": vals[9],
                        "begin_epoch": _parse_ts(vals[8]),
                        "end_epoch": _parse_ts(vals[9]),
                        "gap_sec": float(vals[10]) if vals[10] else 0.0,
                    }
                )
            elif stripped.startswith("INSERT INTO seizure "):
                vals = [_normalize_sql_value(x) for x in _parse_sql_values(stripped)]
                seizures.append(
                    {
                        "seizure_id": str(vals[0]),
                        "recording_id": str(vals[1]),
                        "block_id": str(vals[2]),
                        "eeg_onset": vals[3],
                        "clin_onset": vals[4],
                        "first_eeg_change": vals[5],
                        "first_clin_sign": vals[6],
                        "eeg_offset": vals[7],
                        "clin_offset": vals[8],
                        "pattern": vals[9],
                        "classification": vals[10],
                        "vigilance": vals[11],
                        "focus": vals[12],
                        "commentary": vals[13],
                    }
                )

    return {
        "subject": subject,
        "sql_path": str(sql_path),
        "patient_code": patient_code,
        "admission": admission,
        "recordings": recordings,
        "blocks": blocks,
        "seizures": seizures,
    }


def _artifact_dir(subject: str, paths: EpilepsiaePaths) -> Path:
    return paths.artifact_root / subject / "all_recs"


def _artifact_counts(subject: str, paths: EpilepsiaePaths) -> Dict[str, object]:
    artifact_dir = _artifact_dir(subject, paths)
    if not artifact_dir.exists():
        return {
            "artifact_dir": str(artifact_dir),
            "artifact_subject_present": False,
            "has_refine_gpu": False,
        }
    return {
        "artifact_dir": str(artifact_dir),
        "artifact_subject_present": True,
        "has_refine_gpu": (artifact_dir / "sub_refineGpu.npz").exists(),
    }


def _build_rows_for_subject(
    sql_info: Dict[str, object],
    paths: EpilepsiaePaths,
    time_config: EpilepsiaeTimeConfig,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    subject = str(sql_info["subject"])
    patient_code = sql_info["patient_code"]
    hospital = sql_info["admission"].get("hospital")
    recordings = dict(sql_info["recordings"])
    blocks = list(sql_info["blocks"])
    seizures = list(sql_info["seizures"])
    raw_by_stem = _collect_raw_blocks(subject, paths)
    artifact_dir = _artifact_dir(subject, paths)

    block_rows: List[Dict[str, object]] = []
    for block in blocks:
        recording_id = str(block["recording_id"])
        stem = f"{recording_id}_{int(block['block_no']):04d}"
        tz_info = resolve_epilepsiae_timezone(
            subject=subject,
            patient_code=patient_code,
            hospital=hospital,
            recording_id=recording_id,
            time_config=time_config,
        )
        start_local = _annotate_epoch_local_time(
            block["begin_epoch"], tz_info["timezone_name"], time_config
        )
        end_local = _annotate_epoch_local_time(
            block["end_epoch"], tz_info["timezone_name"], time_config
        )
        raw_block = raw_by_stem.get(stem)
        head_info = _read_head_info(raw_block.head_path) if raw_block and raw_block.head_path else None
        head_delta = None
        if head_info and block["begin_epoch"] is not None:
            head_delta = float(head_info["start_ts"] - float(block["begin_epoch"]))
        channel_info = _classify_channels(head_info["channel_names"]) if head_info else {
            "has_eeg": False,
            "scalp_or_aux_channels": 0,
            "intracranial_channels": 0,
            "sample_intracranial": "",
        }
        block_rows.append(
            {
                "subject": subject,
                "patient_code": patient_code,
                "recording_id": recording_id,
                "block_id": block["block_id"],
                "block_no": block["block_no"],
                "block_stem": stem,
                "block_start_epoch": block["begin_epoch"],
                "block_end_epoch": block["end_epoch"],
                "gap_to_prev_sec": block["gap_sec"],
                "sample_rate_sql": recordings[recording_id]["sample_rate_sql"],
                "n_channels_sql": block["channels_sql"],
                "head_exists": bool(raw_block and raw_block.head_path),
                "data_exists": bool(raw_block and raw_block.data_path),
                "raw_gpu_exists": bool(raw_block and raw_block.raw_gpu_path),
                "artifact_gpu_exists": (artifact_dir / f"{stem}_gpu.npz").exists(),
                "has_lagpat": (artifact_dir / f"{stem}_lagPat.npz").exists(),
                "has_lagpat_freq": (artifact_dir / f"{stem}_lagPat_withFreqCent.npz").exists(),
                "has_packed_times": (artifact_dir / f"{stem}_packedTimes.npy").exists(),
                "has_packed_times_freq": (artifact_dir / f"{stem}_packedTimes_withFreqCent.npy").exists(),
                "head_start_epoch": None if not head_info else head_info["start_ts"],
                "head_duration_sec": None if not head_info else head_info["duration_sec"],
                "head_sample_rate": None if not head_info else head_info["sample_freq"],
                "head_n_channels": None if not head_info else head_info["n_channels"],
                "head_sql_start_delta_sec": head_delta,
                "has_eeg": channel_info["has_eeg"],
                "intracranial_channels": channel_info["intracranial_channels"],
                "scalp_or_aux_channels": channel_info["scalp_or_aux_channels"],
                "sample_intracranial": channel_info["sample_intracranial"],
                "timezone_name": tz_info["timezone_name"],
                "timezone_source": tz_info["timezone_source"],
                "block_start_local_hour": start_local["local_hour"],
                "block_start_day_night": start_local["day_night"],
                "block_end_local_hour": end_local["local_hour"],
                "block_end_day_night": end_local["day_night"],
            }
        )

    seizure_rows: List[Dict[str, object]] = []
    sorted_seizures = sorted(
        seizures,
        key=lambda x: (_parse_ts(x["eeg_onset"]) if x["eeg_onset"] else float("inf")),
    )
    prev_onset: Optional[float] = None
    for seizure in sorted_seizures:
        recording_id = str(seizure["recording_id"])
        tz_info = resolve_epilepsiae_timezone(
            subject=subject,
            patient_code=patient_code,
            hospital=hospital,
            recording_id=recording_id,
            time_config=time_config,
        )
        eeg_onset_epoch = _parse_ts(seizure["eeg_onset"])
        eeg_offset_epoch = _parse_ts(seizure["eeg_offset"])
        clin_onset_epoch = _parse_ts(seizure["clin_onset"])
        clin_offset_epoch = _parse_ts(seizure["clin_offset"])
        eeg_local = _annotate_epoch_local_time(
            eeg_onset_epoch, tz_info["timezone_name"], time_config
        )
        clin_local = _annotate_epoch_local_time(
            clin_onset_epoch, tz_info["timezone_name"], time_config
        )
        seizure_rows.append(
            {
                "subject": subject,
                "patient_code": patient_code,
                "recording_id": recording_id,
                "block_id": seizure["block_id"],
                "seizure_id": seizure["seizure_id"],
                "eeg_onset_epoch": eeg_onset_epoch,
                "eeg_offset_epoch": eeg_offset_epoch,
                "clin_onset_epoch": clin_onset_epoch,
                "clin_offset_epoch": clin_offset_epoch,
                "has_complete_eeg_interval": bool(
                    eeg_onset_epoch is not None and eeg_offset_epoch is not None
                ),
                "has_complete_clin_interval": bool(
                    clin_onset_epoch is not None and clin_offset_epoch is not None
                ),
                "eeg_duration_sec": None
                if eeg_onset_epoch is None or eeg_offset_epoch is None
                else eeg_offset_epoch - eeg_onset_epoch,
                "clin_duration_sec": None
                if clin_onset_epoch is None or clin_offset_epoch is None
                else clin_offset_epoch - clin_onset_epoch,
                "seizure_interval_from_prev_sec": None
                if prev_onset is None or eeg_onset_epoch is None
                else eeg_onset_epoch - prev_onset,
                "pattern": seizure["pattern"],
                "classification": seizure["classification"],
                "vigilance": seizure["vigilance"],
                "focus": seizure["focus"],
                "timezone_name": tz_info["timezone_name"],
                "timezone_source": tz_info["timezone_source"],
                "eeg_onset_local_hour": eeg_local["local_hour"],
                "eeg_onset_day_night": eeg_local["day_night"],
                "clin_onset_local_hour": clin_local["local_hour"],
                "clin_onset_day_night": clin_local["day_night"],
            }
        )
        if eeg_onset_epoch is not None:
            prev_onset = eeg_onset_epoch

    recording_rows: List[Dict[str, object]] = []
    seizure_by_recording: Dict[str, List[Dict[str, object]]] = {}
    for seizure_row in seizure_rows:
        seizure_by_recording.setdefault(str(seizure_row["recording_id"]), []).append(
            seizure_row
        )
    blocks_by_recording: Dict[str, List[Dict[str, object]]] = {}
    for block_row in block_rows:
        blocks_by_recording.setdefault(str(block_row["recording_id"]), []).append(
            block_row
        )

    for recording_id, recording in sorted(
        recordings.items(), key=lambda x: float(x[1]["begin_epoch"] or 0.0)
    ):
        tz_info = resolve_epilepsiae_timezone(
            subject=subject,
            patient_code=patient_code,
            hospital=hospital,
            recording_id=recording_id,
            time_config=time_config,
        )
        begin_local = _annotate_epoch_local_time(
            recording["begin_epoch"], tz_info["timezone_name"], time_config
        )
        rec_blocks = sorted(
            blocks_by_recording.get(recording_id, []), key=lambda x: int(x["block_no"])
        )
        rec_seizures = seizure_by_recording.get(recording_id, [])
        gap_values = [float(x["gap_to_prev_sec"] or 0.0) for x in rec_blocks]
        nontrivial_gaps = [x for x in gap_values if x > NONTRIVIAL_GAP_SEC]
        first_head = next((x for x in rec_blocks if x["head_exists"]), None)
        intracranial_values = sorted(int(x["intracranial_channels"] or 0) for x in rec_blocks)
        recording_rows.append(
            {
                "subject": subject,
                "patient_code": patient_code,
                "recording_id": recording_id,
                "recording_str_id": recording["recording_str_id"],
                "recording_begin_epoch": recording["begin_epoch"],
                "recording_end_epoch": recording["end_epoch"],
                "recording_duration_sec_sql": recording["duration_sec_sql"],
                "net_duration_sec": recording["net_duration_sec"],
                "gross_duration_sec": recording["gross_duration_sec"],
                "completeness_pct": recording["completeness_pct"],
                "n_blocks_sql": recording["blocks_sql"],
                "n_blocks_found_raw": len(rec_blocks),
                "n_nontrivial_block_gaps": len(nontrivial_gaps),
                "max_block_gap_sec": max(gap_values) if gap_values else None,
                "is_block_continuous": len(nontrivial_gaps) == 0,
                "sample_rate_sql": recording["sample_rate_sql"],
                "sample_rate_head": None if not first_head else first_head["head_sample_rate"],
                "n_channels_sql": recording["channels_sql"],
                "n_channels_head": None if not first_head else first_head["head_n_channels"],
                "has_eeg": any(bool(x["has_eeg"]) for x in rec_blocks),
                "intracranial_channels_median": None
                if not intracranial_values
                else intracranial_values[len(intracranial_values) // 2],
                "has_any_seizure": bool(rec_seizures),
                "n_seizures": len(rec_seizures),
                "n_complete_eeg_intervals": sum(
                    1 for x in rec_seizures if x["has_complete_eeg_interval"]
                ),
                "has_any_lagpat": any(bool(x["has_lagpat"]) for x in rec_blocks),
                "has_all_lagpat": bool(rec_blocks)
                and all(bool(x["has_lagpat"]) for x in rec_blocks),
                "has_any_packed_times": any(bool(x["has_packed_times"]) for x in rec_blocks),
                "has_all_packed_times": bool(rec_blocks)
                and all(bool(x["has_packed_times"]) for x in rec_blocks),
                "timezone_name": tz_info["timezone_name"],
                "timezone_source": tz_info["timezone_source"],
                "recording_begin_local_hour": begin_local["local_hour"],
                "recording_begin_day_night": begin_local["day_night"],
            }
        )

    return recording_rows, block_rows, seizure_rows


def _build_subject_row(
    subject: str,
    sql_info: Dict[str, object],
    recording_rows: List[Dict[str, object]],
    block_rows: List[Dict[str, object]],
    seizure_rows: List[Dict[str, object]],
    paths: EpilepsiaePaths,
    time_config: EpilepsiaeTimeConfig,
) -> Dict[str, object]:
    artifact_counts = _artifact_counts(subject, paths)
    tz_info = resolve_epilepsiae_timezone(
        subject=subject,
        patient_code=sql_info["patient_code"],
        hospital=sql_info["admission"].get("hospital"),
        time_config=time_config,
    )
    recording_rows = sorted(
        recording_rows, key=lambda x: float(x["recording_begin_epoch"] or 0.0)
    )
    inter_record_gaps: List[float] = []
    for prev, curr in zip(recording_rows[:-1], recording_rows[1:]):
        prev_end = prev["recording_end_epoch"]
        curr_begin = curr["recording_begin_epoch"]
        if prev_end is not None and curr_begin is not None:
            inter_record_gaps.append(float(curr_begin - prev_end))
    nontrivial_inter_record = [x for x in inter_record_gaps if x > NONTRIVIAL_GAP_SEC]
    seizure_intervals = [
        float(x["seizure_interval_from_prev_sec"])
        for x in seizure_rows
        if x["seizure_interval_from_prev_sec"] is not None
    ]
    max_block_gap = max(
        (float(x["gap_to_prev_sec"] or 0.0) for x in block_rows), default=0.0
    )

    return {
        "subject": subject,
        "patient_code": sql_info["patient_code"],
        "sql_path": sql_info["sql_path"],
        "hospital": sql_info["admission"].get("hospital"),
        "seeg_flag_sql": sql_info["admission"].get("seeg"),
        "ieeg_flag_sql": sql_info["admission"].get("ieeg"),
        "n_recordings": len(recording_rows),
        "n_blocks_sql": len(block_rows),
        "n_recordings_with_nontrivial_gap": sum(
            1 for x in recording_rows if not x["is_block_continuous"]
        ),
        "max_block_gap_sec": max_block_gap,
        "n_inter_record_gaps_gt2s": len(nontrivial_inter_record),
        "max_inter_record_gap_sec": max(inter_record_gaps) if inter_record_gaps else None,
        "is_subject_continuous_by_recording": len(nontrivial_inter_record) == 0,
        "total_net_duration_sec": sum(
            float(x["net_duration_sec"] or 0.0) for x in recording_rows
        ),
        "total_gross_duration_sec": sum(
            float(x["gross_duration_sec"] or 0.0) for x in recording_rows
        ),
        "n_seizures": len(seizure_rows),
        "n_complete_eeg_intervals": sum(
            1 for x in seizure_rows if x["has_complete_eeg_interval"]
        ),
        "n_complete_clin_intervals": sum(
            1 for x in seizure_rows if x["has_complete_clin_interval"]
        ),
        "median_seizure_interval_sec": None
        if not seizure_intervals
        else sorted(seizure_intervals)[len(seizure_intervals) // 2],
        "min_seizure_interval_sec": min(seizure_intervals) if seizure_intervals else None,
        "max_seizure_interval_sec": max(seizure_intervals) if seizure_intervals else None,
        "has_eeg_channels_in_heads": any(bool(x["has_eeg"]) for x in block_rows),
        "has_intracranial_heads": any(
            int(x["intracranial_channels"] or 0) > 0 for x in block_rows
        ),
        "artifact_subject_present": artifact_counts["artifact_subject_present"],
        "has_refine_gpu": artifact_counts["has_refine_gpu"],
        "n_gpu_raw_blocks": sum(1 for x in block_rows if x["raw_gpu_exists"]),
        "n_gpu_artifact_blocks": sum(1 for x in block_rows if x["artifact_gpu_exists"]),
        "n_lagpat_blocks": sum(1 for x in block_rows if x["has_lagpat"]),
        "n_packed_times_blocks": sum(1 for x in block_rows if x["has_packed_times"]),
        "n_lagpat_freq_blocks": sum(1 for x in block_rows if x["has_lagpat_freq"]),
        "timezone_name": tz_info["timezone_name"],
        "timezone_source": tz_info["timezone_source"],
        "day_start_hour": int(time_config.day_start_hour),
        "night_start_hour": int(time_config.night_start_hour),
        "day_night_rule": _day_night_rule_string(time_config),
        "day_night_wall_clock_possible": True,
        "day_night_timezone_known": tz_info["timezone_known"],
        "day_night_reliable_without_override": tz_info["reliable_without_override"],
    }


def survey_epilepsiae_dataset(
    paths: Optional[EpilepsiaePaths] = None,
    *,
    time_config: Optional[EpilepsiaeTimeConfig] = None,
) -> EpilepsiaeInventory:
    paths = paths or EpilepsiaePaths()
    time_config = time_config or EpilepsiaeTimeConfig()
    _validate_time_config(time_config)
    sql_paths = sorted(paths.sql_root.glob("pat_*_*.sql"))
    subject_rows: List[Dict[str, object]] = []
    recording_rows_all: List[Dict[str, object]] = []
    block_rows_all: List[Dict[str, object]] = []
    seizure_rows_all: List[Dict[str, object]] = []

    for sql_path in sql_paths:
        sql_info = _parse_sql_subject(sql_path)
        subject = str(sql_info["subject"])
        recording_rows, block_rows, seizure_rows = _build_rows_for_subject(
            sql_info, paths, time_config
        )
        subject_rows.append(
            _build_subject_row(
                subject,
                sql_info,
                recording_rows,
                block_rows,
                seizure_rows,
                paths,
                time_config,
            )
        )
        recording_rows_all.extend(recording_rows)
        block_rows_all.extend(block_rows)
        seizure_rows_all.extend(seizure_rows)

    summary = {
        "n_sql_subjects": len(subject_rows),
        "n_recordings": len(recording_rows_all),
        "n_blocks": len(block_rows_all),
        "n_seizures": len(seizure_rows_all),
        "n_subjects_with_artifacts": sum(
            1 for x in subject_rows if x["artifact_subject_present"]
        ),
        "n_subjects_with_refine_gpu": sum(
            1 for x in subject_rows if x["has_refine_gpu"]
        ),
        "n_subjects_with_seeg_flag": sum(
            1 for x in subject_rows if x["seeg_flag_sql"] is True
        ),
        "n_subjects_with_ieeg_flag": sum(
            1 for x in subject_rows if x["ieeg_flag_sql"] is True
        ),
        "n_subjects_continuous_by_recording": sum(
            1 for x in subject_rows if x["is_subject_continuous_by_recording"]
        ),
        "n_subjects_with_nontrivial_inter_record_gap": sum(
            1 for x in subject_rows if x["n_inter_record_gaps_gt2s"] > 0
        ),
        "n_subjects_with_complete_eeg_intervals": sum(
            1 for x in subject_rows if x["n_complete_eeg_intervals"] > 0
        ),
    }
    return EpilepsiaeInventory(
        subject_rows=subject_rows,
        recording_rows=recording_rows_all,
        block_rows=block_rows_all,
        seizure_rows=seizure_rows_all,
        summary=summary,
    )


def build_epilepsiae_sync_subject_manifest(
    inventory: EpilepsiaeInventory,
    *,
    min_complete_eeg_intervals: int = 2,
    min_sync_interval_sec: float = DEFAULT_MIN_SYNC_INTERVAL_SEC,
    require_artifacts: bool = True,
) -> List[Dict[str, object]]:
    """
    Build a subject-level manifest for interictal synchrony analysis.

    Readiness is intentionally simple:
    - at least `min_complete_eeg_intervals` complete EEG seizures
    - at least one inter-seizure interval >= `min_sync_interval_sec`
    - artifacts (`lagPat` + `packedTimes`) available when `require_artifacts=True`
    """
    seizure_by_subject: Dict[str, List[Dict[str, object]]] = {}
    for row in inventory.seizure_rows:
        seizure_by_subject.setdefault(str(row["subject"]), []).append(row)

    manifest_rows: List[Dict[str, object]] = []
    for subj in sorted(inventory.subject_rows, key=lambda x: str(x["subject"])):
        subject = str(subj["subject"])
        seizures = seizure_by_subject.get(subject, [])
        complete_eeg = [x for x in seizures if x["has_complete_eeg_interval"]]
        long_intervals = [
            float(x["seizure_interval_from_prev_sec"])
            for x in seizures
            if x["seizure_interval_from_prev_sec"] not in ("", None)
            and float(x["seizure_interval_from_prev_sec"]) >= float(min_sync_interval_sec)
        ]
        lagpat_cov = 0.0
        if int(subj["n_gpu_artifact_blocks"] or 0) > 0:
            lagpat_cov = float(subj["n_lagpat_blocks"]) / float(subj["n_gpu_artifact_blocks"])
        artifact_ready = (
            int(subj["n_lagpat_blocks"] or 0) > 0
            and int(subj["n_packed_times_blocks"] or 0) > 0
            and (not require_artifacts or bool(subj["artifact_subject_present"]))
        )
        ready = (
            len(complete_eeg) >= int(min_complete_eeg_intervals)
            and len(long_intervals) >= 1
            and artifact_ready
        )
        if ready and lagpat_cov >= 0.95:
            tier = "ready_full_artifacts"
        elif ready:
            tier = "ready_partial_artifacts"
        elif int(subj["n_complete_eeg_intervals"] or 0) < int(min_complete_eeg_intervals):
            tier = "not_enough_complete_seizures"
        elif len(long_intervals) == 0:
            tier = "no_long_interictal_interval"
        elif not artifact_ready:
            tier = "missing_interictal_artifacts"
        else:
            tier = "not_ready"

        reasons: List[str] = []
        if int(subj["n_complete_eeg_intervals"] or 0) < int(min_complete_eeg_intervals):
            reasons.append("fewer_than_required_complete_eeg_seizures")
        if len(long_intervals) == 0:
            reasons.append("no_interseizure_interval_ge_threshold")
        if not artifact_ready:
            reasons.append("missing_lagpat_or_packed_times")
        if float(subj["n_inter_record_gaps_gt2s"] or 0) > 0:
            reasons.append("has_inter_recording_gaps")
        if str(subj["day_night_reliable_without_override"]) != "True":
            reasons.append("timezone_override_needed_for_day_night")

        manifest_rows.append(
            {
                "subject": subject,
                "patient_code": subj["patient_code"],
                "tier": tier,
                "ready_for_sync_analysis": ready,
                "n_complete_eeg_intervals": len(complete_eeg),
                "n_intervals_ge_threshold": len(long_intervals),
                "min_sync_interval_sec": float(min_sync_interval_sec),
                "artifact_subject_present": subj["artifact_subject_present"],
                "has_refine_gpu": subj["has_refine_gpu"],
                "n_lagpat_blocks": subj["n_lagpat_blocks"],
                "n_packed_times_blocks": subj["n_packed_times_blocks"],
                "n_gpu_artifact_blocks": subj["n_gpu_artifact_blocks"],
                "lagpat_coverage_ratio": round(lagpat_cov, 6),
                "n_inter_record_gaps_gt2s": subj["n_inter_record_gaps_gt2s"],
                "max_inter_record_gap_sec": subj["max_inter_record_gap_sec"],
                "max_block_gap_sec": subj["max_block_gap_sec"],
                "timezone_name": subj["timezone_name"],
                "timezone_source": subj["timezone_source"],
                "day_night_rule": subj["day_night_rule"],
                "day_night_reliable_without_override": subj["day_night_reliable_without_override"],
                "reasons": "|".join(reasons),
            }
        )
    return manifest_rows


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            fh.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_epilepsiae_csv_rows(csv_path: Path | str) -> List[Dict[str, str]]:
    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_epilepsiae_sync_subject_manifest(csv_path: Path | str) -> List[Dict[str, str]]:
    return load_epilepsiae_csv_rows(csv_path)


def save_epilepsiae_inventory(
    inventory: EpilepsiaeInventory, output_dir: Path | str
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    subject_path = output_dir / "epilepsiae_subject_inventory.csv"
    recording_path = output_dir / "epilepsiae_recording_inventory.csv"
    block_path = output_dir / "epilepsiae_block_inventory.csv"
    seizure_path = output_dir / "epilepsiae_seizure_inventory.csv"
    summary_path = output_dir / "epilepsiae_dataset_summary.json"

    _write_csv(subject_path, inventory.subject_rows)
    _write_csv(recording_path, inventory.recording_rows)
    _write_csv(block_path, inventory.block_rows)
    _write_csv(seizure_path, inventory.seizure_rows)
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"summary": inventory.summary, "subjects": inventory.subject_rows},
            fh,
            indent=2,
            ensure_ascii=False,
        )
    return {
        "subject_csv": str(subject_path),
        "recording_csv": str(recording_path),
        "block_csv": str(block_path),
        "seizure_csv": str(seizure_path),
        "summary_json": str(summary_path),
    }


def save_epilepsiae_sync_subject_manifest(
    rows: Sequence[Dict[str, object]], output_path: Path | str
) -> str:
    output_path = Path(output_path)
    _write_csv(output_path, list(rows))
    return str(output_path)
