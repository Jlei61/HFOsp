"""SEEG 3D channel coordinate loader (v2 strict schema, 2026-05-21).

Plan: docs/archive/topic4/sef_itp_phase1/coord_loader_plan_2026-05-21.md
Data contract: docs/topic0_methodology_audits.md §3.2.4 (v2 strict invariants)

Design philosophy: 7 hard invariants, every violation is a science pollution
risk. Returns ordered numpy array aligned to caller's channel_names_requested
— NO dict-only interface, NO silent reordering. Yuquan → mm directly;
Epilepsiae → voxel + explicit marker; voxel→mm only via explicit affine arg.

Invariants (every test in tests/test_seeg_coord_loader.py covers one):
  1. NULL coord → explicit missing[]; coords_array NaN row + mask False
  2. Bipolar midpoint requires BOTH endpoints found; one NULL = pair missing
  3. Channel name match three-state: found / not_found / bipolar_partial_endpoint
  4. Provenance records absolute paths + loader_version
  5. Output array order is request order (channel_names_requested as anchor)
  6. coord_space + coord_units are explicit; voxel != mm
  7. No cross-subject registration; per-subject native space only
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np


LOADER_VERSION = "coord_loader_v3_2026-05-21"

YUQUAN_ELEC_ROOT = Path(
    "/mnt/yuquan_data/yuquan_images/nii格式及点电极坐标/caseAndMRI/yuquan_24h_mriCT/patients_elecs_reGen"
)
EPILEPSIAE_SQL_ROOT = Path("/mnt/epilepsia_data/all_data_sqls")

# v3 (2026-05-21): Epilepsiae MRIs verified to share the MNI152 1mm standard
# grid affine across all 27 subjects (identical shape + affine; differing md5
# = real subject anatomy warped to MNI152 grid). Loader auto-discovers MRI
# files across these mount roots and applies the canonical affine.
EPILEPSIAE_MRI_ROOTS: Tuple[Path, ...] = (
    Path("/mnt/epilepsia_data/inv"),
    Path("/mnt/epilepsia_data/inv_1_part"),
    Path("/mnt/epilepsia_data/inv2"),
    Path("/mnt/epilepsia_data/epilepsiae_3patient"),
)

# Verified across 27 Epilepsiae MRIs (2026-05-21): all share this affine.
MNI152_1MM_AFFINE = np.array(
    [
        [-1.0, 0.0, 0.0, 90.0],
        [0.0, 1.0, 0.0, -126.0],
        [0.0, 0.0, 1.0, -72.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)
MNI152_1MM_SHAPE = (182, 218, 182)


# =============================================================================
# Schema dataclasses
# =============================================================================


@dataclass
class MissingEntry:
    """One entry in CoordResult.missing[]."""

    channel: str
    reason: str  # "sql_null" | "name_not_found" | "bipolar_partial_endpoint" | "commentary_*"
    index_in_requested: int
    commentary: Optional[str] = None


@dataclass
class BipolarRes:
    """Per-bipolar-channel resolution log."""

    left_endpoint: str
    right_endpoint: str
    left_coord: Optional[Tuple[float, float, float]]
    right_coord: Optional[Tuple[float, float, float]]
    midpoint_strategy: str = "mean_both_required"


@dataclass
class CoordResult:
    """SEEG coord loader output (v3 schema, see coord_loader_plan §4).

    INVARIANT (v2 #5 — most important): channel_names_requested,
    coords_array_in_requested_order, mapped_mask_in_requested_order are ALL
    aligned by index. Any reordering is a science pollution bug.

    v3 (2026-05-21) added fields:
      - source_coord_type: what the raw input was (e.g., "sql_voxel_ijk")
      - normalization_certainty: honest tag for cohort-comparability claim
    """

    schema_version: str
    dataset: str
    subject_id: str
    channel_names_requested: List[str]
    coords_array_in_requested_order: np.ndarray  # (n_requested, 3) NaN for missing
    mapped_mask_in_requested_order: np.ndarray  # (n_requested,) bool
    coord_space: str  # "fs_native_ras_mm" | "mri_native_voxel_ijk" | "mni152_1mm" | "ras_mm_via_affine"
    coord_units: str  # "mm" | "voxel"
    provenance: Dict[str, Any]
    missing: List[MissingEntry] = field(default_factory=list)
    bipolar_resolution: Dict[str, BipolarRes] = field(default_factory=dict)
    # v3 new fields:
    source_coord_type: str = ""  # "direct_ras_mm" | "sql_voxel_ijk" | ...
    normalization_certainty: str = ""  # "subject_native" | "grid_confirmed_warp_type_unverified" | "mni_normalized_verified"


# =============================================================================
# Channel name parsing
# =============================================================================


@dataclass
class ChannelQuery:
    raw_name: str
    is_bipolar: bool
    left: Optional[Tuple[str, int]]  # (shaft_prefix, ordinal)
    right: Optional[Tuple[str, int]]


_NAME_RE = re.compile(r"^([A-Za-z]+\'?)\s*(\d+)$")


def _parse_endpoint(token: str) -> Optional[Tuple[str, int]]:
    """Parse 'HLA1' → ('HLA', 1); 'A\\'1' → (\"A'\", 1); None on failure."""
    m = _NAME_RE.match(token.strip())
    if not m:
        return None
    prefix, num_str = m.group(1), m.group(2)
    try:
        return (prefix, int(num_str))
    except ValueError:
        return None


def _parse_channel_to_query_keys(channel_name: str) -> ChannelQuery:
    """Parse a channel name into a ChannelQuery (monopolar or bipolar)."""
    name = channel_name.strip()
    if "-" in name:
        left_tok, right_tok = name.split("-", 1)
        left = _parse_endpoint(left_tok)
        right = _parse_endpoint(right_tok)
        return ChannelQuery(raw_name=channel_name, is_bipolar=True, left=left, right=right)
    else:
        left = _parse_endpoint(name)
        return ChannelQuery(raw_name=channel_name, is_bipolar=False, left=left, right=None)


# =============================================================================
# Bipolar midpoint (both-required contract)
# =============================================================================


def _resolve_bipolar(
    left_coord: Optional[np.ndarray],
    right_coord: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Midpoint with BOTH endpoints required (v2 invariant #2).

    Returns None if either endpoint is missing — caller must mark pair missing.
    """
    if left_coord is None or right_coord is None:
        return None
    return (np.asarray(left_coord, dtype=float) + np.asarray(right_coord, dtype=float)) / 2.0


# =============================================================================
# Affine transformation (voxel → mm)
# =============================================================================


def _apply_affine(voxel_coord: np.ndarray, mri_affine: np.ndarray) -> np.ndarray:
    """Apply 4x4 affine to (3,) or (N, 3) voxel coords, return ras_mm."""
    voxel_coord = np.asarray(voxel_coord, dtype=float)
    if voxel_coord.ndim == 1:
        homog = np.concatenate([voxel_coord, [1.0]])
        out = mri_affine @ homog
        return out[:3]
    elif voxel_coord.ndim == 2:
        n = voxel_coord.shape[0]
        homog = np.hstack([voxel_coord, np.ones((n, 1))])
        out = (mri_affine @ homog.T).T
        return out[:, :3]
    else:
        raise ValueError(f"voxel_coord must be 1D or 2D, got shape {voxel_coord.shape}")


# =============================================================================
# Yuquan reader
# =============================================================================


def _read_yuquan_chnXyzDict(subject_id: str, root: Path = YUQUAN_ELEC_ROOT) -> Dict[str, np.ndarray]:
    """Load Yuquan chnXyzDict.npy — dict[shaft_prefix → (n_contacts, 3) array]."""
    path = root / subject_id / "chnXyzDict.npy"
    if not path.exists():
        raise FileNotFoundError(f"Yuquan coord file not found: {path}")
    data = np.load(path, allow_pickle=True).item()
    return data


def _lookup_yuquan_contact(
    shaft_dict: Dict[str, np.ndarray],
    shaft_prefix: str,
    contact_num: int,
) -> Optional[np.ndarray]:
    """Look up one monopolar contact in Yuquan dict.

    Contacts in chnXyzDict[shaft] are 0-indexed; SEEG channel names are 1-indexed.
    So 'HLA1' → row 0, 'HLA2' → row 1, etc.
    """
    if shaft_prefix not in shaft_dict:
        return None
    arr = shaft_dict[shaft_prefix]
    row_idx = contact_num - 1
    if row_idx < 0 or row_idx >= len(arr):
        return None
    return np.asarray(arr[row_idx], dtype=float)


# =============================================================================
# Epilepsiae reader
# =============================================================================


_EPILEPSIAE_INSERT_RE = re.compile(
    r"INSERT INTO electrode \([^)]+\) VALUES \(([^)]*)\);",
    re.IGNORECASE,
)


@dataclass
class EpilepsiaeElectrodeRow:
    name: str
    coord_x: Optional[float]
    coord_y: Optional[float]
    coord_z: Optional[float]
    commentary: Optional[str]


def _parse_epilepsiae_sql_value(token: str) -> Any:
    """Parse one VALUES token: 'NULL' → None, "'text'" → str, '131.0' → float."""
    token = token.strip()
    if token == "NULL":
        return None
    if token.startswith("'") and token.endswith("'"):
        return token[1:-1]
    try:
        return float(token)
    except ValueError:
        return token


def _split_sql_values(values_str: str) -> List[str]:
    """Split a SQL VALUES tuple respecting single-quoted strings.

    `131.0, 139.0, 119.0, NULL, 'nicht abgrenzbar!'` → 5 tokens.
    """
    tokens: List[str] = []
    buf: List[str] = []
    in_quote = False
    for ch in values_str:
        if ch == "'" and (not buf or buf[-1] != "\\"):
            in_quote = not in_quote
            buf.append(ch)
        elif ch == "," and not in_quote:
            tokens.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        tokens.append("".join(buf))
    return tokens


def _canonicalize_epilepsiae_subject_id(subject_id: str) -> str:
    """Canonicalize Epilepsiae subject_id to the canonical pat_<id> form.

    Rule (v3.1 lock 2026-05-21 — fixes cross-patient pollution):
      - if subject_id ends in '02' → already canonical, return as-is
      - else → append '02' (numeric short ID convention)

    Examples:
        '115'    -> '11502'   (pat_11502, subject 115)
        '1150'   -> '115002'  (pat_115002, subject 1150)
        '108402' -> '108402'  (already canonical)
        '11502'  -> '11502'   (already canonical, ends in 02)
        '115002' -> '115002'  (already canonical)

    NOTE: this rule is UNAMBIGUOUS but requires caller to be explicit. A user
    passing '11502' MUST mean pat_11502 (subject 115), NOT pat_1150202 (subject
    11502). If you have only the numeric short form, pass it directly.
    """
    sid = str(subject_id).strip()
    if not sid:
        raise ValueError("subject_id must be non-empty")
    if not sid.isdigit():
        raise ValueError(
            f"subject_id {sid!r} must be numeric (no 'pat_' prefix, no date suffix)"
        )
    return sid if sid.endswith("02") else f"{sid}02"


def _read_epilepsiae_electrode_sql(
    canonical_id: str, root: Path = EPILEPSIAE_SQL_ROOT
) -> Tuple[List["EpilepsiaeElectrodeRow"], Path]:
    """Read electrode rows from pat_<canonical_id>_*.sql.

    v3.1 (2026-05-21) — replaces fuzzy `*<id>*.sql` glob that caused
    cross-patient pollution (e.g., 'pat_115_*.sql' fallback to '*115*.sql'
    silently matched pat_115002 when caller meant pat_11502). Now exact-match
    only; 0 or >1 candidates → raise.

    Returns:
        (rows, sql_path) — sql_path returned so caller can cross-check
        provenance against MRI source.
    """
    candidates = sorted(root.glob(f"pat_{canonical_id}_*.sql"))
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"Epilepsiae SQL not found for pat_{canonical_id} in {root}. "
            f"Glob: pat_{canonical_id}_*.sql"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Ambiguous Epilepsiae SQL for pat_{canonical_id}: "
            f"{len(candidates)} matches {[p.name for p in candidates]}. "
            f"Expected exactly 1."
        )
    sql_path = candidates[0]

    rows: List[EpilepsiaeElectrodeRow] = []
    with open(sql_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "INSERT INTO electrode " not in line:
                continue
            m = _EPILEPSIAE_INSERT_RE.search(line)
            if not m:
                continue
            tokens = [_parse_epilepsiae_sql_value(t) for t in _split_sql_values(m.group(1))]
            # Schema: id, array, name, moniker, artifact, focus_rel, invasive, supplier,
            #         coord_x, coord_y, coord_z, commentary
            if len(tokens) < 12:
                continue
            rows.append(
                EpilepsiaeElectrodeRow(
                    name=str(tokens[2]),
                    coord_x=tokens[8] if isinstance(tokens[8], float) else None,
                    coord_y=tokens[9] if isinstance(tokens[9], float) else None,
                    coord_z=tokens[10] if isinstance(tokens[10], float) else None,
                    commentary=tokens[11] if isinstance(tokens[11], str) else None,
                )
            )
    return rows, sql_path


def _lookup_epilepsiae_contact(
    rows: List[EpilepsiaeElectrodeRow],
    shaft_prefix: str,
    contact_num: int,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Look up one monopolar Epilepsiae electrode. Returns (coord_or_None, commentary)."""
    target_name = f"{shaft_prefix}{contact_num}"
    for row in rows:
        if row.name == target_name:
            if row.coord_x is None or row.coord_y is None or row.coord_z is None:
                return None, row.commentary
            return (
                np.array([row.coord_x, row.coord_y, row.coord_z], dtype=float),
                row.commentary,
            )
    return None, None


def _categorize_commentary(commentary: Optional[str]) -> str:
    """Classify Epilepsiae commentary into missing.reason category."""
    if commentary is None:
        return "sql_null"
    low = commentary.lower()
    if "mikrokontakt" in low:
        return "commentary_mikrokontakt"
    if "abgrenzbar" in low:
        return "commentary_nicht_abgrenzbar"
    return "commentary_other"


# =============================================================================
# Main entry
# =============================================================================


def load_subject_coords(
    dataset: Literal["yuquan", "epilepsiae"],
    subject_id: str,
    channel_names_requested: Sequence[str],
    mri_affine: Optional[np.ndarray] = None,
    *,
    yuquan_root: Path = YUQUAN_ELEC_ROOT,
    epilepsiae_root: Path = EPILEPSIAE_SQL_ROOT,
    epilepsiae_mri_search_roots: Optional[Sequence[Path]] = None,
    allow_voxel_fallback: bool = False,
) -> CoordResult:
    """Load SEEG 3D coords for one subject's channels.

    Args:
        dataset: "yuquan" → fs_native_ras_mm; "epilepsiae" → mni152_1mm via
                 auto-discovered MRI affine.
        subject_id: subject identifier.
                    - yuquan: directory name (e.g., "chengshuai")
                    - epilepsiae: numeric ID; canonicalized via
                      _canonicalize_epilepsiae_subject_id (e.g., "115" → "11502",
                      "1150" → "115002", "108402" → "108402")
        channel_names_requested: ordered list; output array indexing matches this
        mri_affine: optional 4x4 affine to OVERRIDE auto-discovery (Epilepsiae).
                    When provided, coord_space = "ras_mm_via_affine".
        epilepsiae_mri_search_roots: MRI .img/.hdr search roots
                    (default: EPILEPSIAE_MRI_ROOTS = 4 production mounts).
        allow_voxel_fallback: v3.1 (Epilepsiae only). Default False — MRI miss
                    raises FileNotFoundError. Pass True to opt-in to voxel
                    sensitivity mode. Silent fallback is forbidden by default
                    to prevent science pollution from typos / mount issues.

    Returns:
        CoordResult with v3 schema. See module docstring for invariants.

    Raises:
        ValueError: unknown dataset; ambiguous SQL match; affine mismatch;
                    cross-patient provenance mismatch; etc.
        FileNotFoundError: SQL or MRI not found and no fallback authorized.
    """
    if dataset == "yuquan":
        return _load_yuquan(subject_id, list(channel_names_requested), yuquan_root, mri_affine)
    elif dataset == "epilepsiae":
        if epilepsiae_mri_search_roots is None:
            epilepsiae_mri_search_roots = EPILEPSIAE_MRI_ROOTS
        return _load_epilepsiae(
            subject_id,
            list(channel_names_requested),
            epilepsiae_root,
            mri_affine,
            tuple(epilepsiae_mri_search_roots),
            allow_voxel_fallback,
        )
    else:
        raise ValueError(f"unknown dataset: {dataset!r}. Use 'yuquan' or 'epilepsiae'.")


def _find_epilepsiae_mri(
    canonical_id: str, roots: Sequence[Path]
) -> Optional[Path]:
    """Find mri_*.img for pat_<canonical_id> across roots.

    v3.1 (2026-05-21) — caller MUST pass canonical_id (no internal short/full
    ID expansion). Eliminates the cross-patient pollution where '115' would
    try both '115' and '11502', silently matching pat_11502 when the matching
    SQL was pat_115002.
    """
    for root in roots:
        if not root.exists():
            continue
        pat_dir = root / f"pat_{canonical_id}"
        if not pat_dir.exists():
            continue
        mri_files = sorted(pat_dir.glob("adm_*/MRI/mri_*.img"))
        if mri_files:
            return mri_files[0]
    return None


def _load_mri_affine(mri_img_path: Path) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Load 4x4 affine + 3D shape from an Analyze .img/.hdr pair via nibabel."""
    try:
        import nibabel as nib  # local import — only required for auto-discovery
    except ImportError as e:
        raise ImportError(
            "nibabel required for Epilepsiae MRI auto-discovery; "
            "pip install nibabel, or pass mri_affine explicitly."
        ) from e
    img = nib.load(str(mri_img_path))
    aff = np.asarray(img.affine, dtype=float)
    shape = tuple(int(s) for s in img.shape[:3])
    return aff, shape


def _load_yuquan(
    subject_id: str,
    channel_names_requested: List[str],
    root: Path,
    mri_affine: Optional[np.ndarray],
) -> CoordResult:
    if mri_affine is not None:
        raise ValueError(
            "Yuquan coords are already fs_native_ras_mm; mri_affine must be None for Yuquan. "
            "If you need a different space, transform downstream of the loader."
        )

    shaft_dict = _read_yuquan_chnXyzDict(subject_id, root)
    source_path = str((root / subject_id / "chnXyzDict.npy").resolve())

    return _build_result(
        dataset="yuquan",
        subject_id=subject_id,
        channel_names_requested=channel_names_requested,
        lookup_fn=lambda prefix, num: (_lookup_yuquan_contact(shaft_dict, prefix, num), None),
        coord_space="fs_native_ras_mm",
        coord_units="mm",
        provenance={
            "source_path": source_path,
            "affine_path": None,
            "loader_version": LOADER_VERSION,
        },
        mri_affine=None,
        source_coord_type="direct_ras_mm",
        normalization_certainty="subject_native",
    )


def _load_epilepsiae(
    subject_id: str,
    channel_names_requested: List[str],
    sql_root: Path,
    mri_affine_override: Optional[np.ndarray],
    mri_search_roots: Tuple[Path, ...],
    allow_voxel_fallback: bool,
) -> CoordResult:
    """Load Epilepsiae coords (v3.1 hardened — canonicalized ID, loud on miss).

    Precedence:
      1. mri_affine_override given → use it, coord_space="ras_mm_via_affine"
      2. MRI file auto-discovered + affine matches MNI152 + provenance pat_id
         matches SQL pat_id → coord_space="mni152_1mm"
      3. MRI affine doesn't match MNI152 OR shape doesn't match → raise
      4. MRI not found AND allow_voxel_fallback=False → raise (default)
      5. MRI not found AND allow_voxel_fallback=True → voxel sensitivity mode

    v3.1 fixes (2026-05-21):
      - subject_id canonicalized via _canonicalize_epilepsiae_subject_id
        (rules out '115' silently matching pat_115002)
      - SQL glob exact pat_<canonical>_*.sql, 0 or >1 → raise
      - MRI provenance cross-checked against canonical id
      - MRI miss is HARD failure by default (was: silent voxel fallback)
    """
    canonical_id = _canonicalize_epilepsiae_subject_id(subject_id)
    rows, sql_path = _read_epilepsiae_electrode_sql(canonical_id, sql_root)
    sql_source_path = str(sql_path.resolve())

    # === Precedence resolution ===
    if mri_affine_override is not None:
        affine_to_apply = mri_affine_override
        coord_space = "ras_mm_via_affine"
        coord_units = "mm"
        affine_source = "<external_override>"
        normalization_certainty = "external_affine_provided"
    else:
        mri_path = (
            _find_epilepsiae_mri(canonical_id, mri_search_roots)
            if mri_search_roots
            else None
        )
        if mri_path is not None:
            # Cross-check: MRI dir must be pat_<canonical> (same as SQL)
            # Defensive — _find_epilepsiae_mri already searches exact pat_<canonical>/
            # but verify post-hoc since this is the science-pollution-critical step
            mri_pat_dir_name = mri_path.parent.parent.parent.name  # pat_<canonical>
            expected_pat_name = f"pat_{canonical_id}"
            if mri_pat_dir_name != expected_pat_name:
                raise ValueError(
                    f"Provenance mismatch (cross-patient pollution risk): "
                    f"SQL is from {expected_pat_name} but MRI is from "
                    f"{mri_pat_dir_name}. MRI path: {mri_path}"
                )

            aff, shape = _load_mri_affine(mri_path)
            if not np.allclose(aff, MNI152_1MM_AFFINE, atol=1e-3):
                raise ValueError(
                    f"Epilepsiae subject {subject_id!r} (canonical {canonical_id!r}): "
                    f"MRI affine at {mri_path} does not match canonical MNI152 1mm.\n"
                    f"  expected: {MNI152_1MM_AFFINE.tolist()}\n"
                    f"  observed: {aff.tolist()}\n"
                    f"Cohort assumption (all 27 subjects share MNI152 affine) "
                    f"is broken — review data."
                )
            if shape != MNI152_1MM_SHAPE:
                raise ValueError(
                    f"Epilepsiae subject {subject_id!r}: MRI shape {shape} "
                    f"!= MNI152 1mm shape {MNI152_1MM_SHAPE}"
                )
            affine_to_apply = aff
            coord_space = "mni152_1mm"
            coord_units = "mm"
            affine_source = str(mri_path)
            normalization_certainty = "grid_confirmed_warp_type_unverified"
        else:
            # No MRI auto-discovered. v3.1: default = HARD FAIL.
            if not allow_voxel_fallback:
                raise FileNotFoundError(
                    f"Epilepsiae MRI not found for pat_{canonical_id} "
                    f"(searched roots: {[str(r) for r in mri_search_roots]}). "
                    f"To opt into voxel sensitivity mode, pass "
                    f"allow_voxel_fallback=True; or pass an explicit "
                    f"mri_affine override. Silent voxel fallback is forbidden "
                    f"by default (v3.1 safety lock)."
                )
            affine_to_apply = None
            coord_space = "mri_native_voxel_ijk"
            coord_units = "voxel"
            affine_source = None
            normalization_certainty = "subject_native_voxel_no_affine"

    return _build_result(
        dataset="epilepsiae",
        subject_id=canonical_id,  # store canonical for downstream provenance
        channel_names_requested=channel_names_requested,
        lookup_fn=lambda prefix, num: _lookup_epilepsiae_contact(rows, prefix, num),
        coord_space=coord_space,
        coord_units=coord_units,
        provenance={
            "source_path": sql_source_path,
            "affine_path": affine_source,
            "loader_version": LOADER_VERSION,
            "canonical_subject_id": canonical_id,
            "input_subject_id": subject_id,
        },
        mri_affine=affine_to_apply,
        source_coord_type="sql_voxel_ijk",
        normalization_certainty=normalization_certainty,
    )


def _build_result(
    *,
    dataset: str,
    subject_id: str,
    channel_names_requested: List[str],
    lookup_fn,  # (shaft_prefix, contact_num) -> (coord or None, commentary or None)
    coord_space: str,
    coord_units: str,
    provenance: Dict[str, Any],
    mri_affine: Optional[np.ndarray],
    source_coord_type: str = "direct",
    normalization_certainty: str = "subject_native",
) -> CoordResult:
    """Shared per-channel resolution loop. Order-anchor invariant locked here."""
    n = len(channel_names_requested)
    coords_array = np.full((n, 3), np.nan, dtype=float)
    mapped_mask = np.zeros(n, dtype=bool)
    missing: List[MissingEntry] = []
    bipolar_log: Dict[str, BipolarRes] = {}

    for i, name in enumerate(channel_names_requested):
        query = _parse_channel_to_query_keys(name)

        if query.is_bipolar:
            if query.left is None or query.right is None:
                missing.append(
                    MissingEntry(
                        channel=name, reason="name_not_found",
                        index_in_requested=i,
                        commentary=f"could not parse bipolar endpoints from {name!r}",
                    )
                )
                continue
            left_lookup = lookup_fn(query.left[0], query.left[1])
            right_lookup = lookup_fn(query.right[0], query.right[1])
            left_coord, left_comm = left_lookup
            right_coord, right_comm = right_lookup

            bipolar_log[name] = BipolarRes(
                left_endpoint=f"{query.left[0]}{query.left[1]}",
                right_endpoint=f"{query.right[0]}{query.right[1]}",
                left_coord=tuple(left_coord) if left_coord is not None else None,
                right_coord=tuple(right_coord) if right_coord is not None else None,
            )

            midpoint = _resolve_bipolar(left_coord, right_coord)
            if midpoint is None:
                # invariant #2: both endpoints required
                reason = "bipolar_partial_endpoint"
                comm = None
                if left_coord is None and left_comm:
                    reason = _categorize_commentary(left_comm)
                    comm = left_comm
                elif right_coord is None and right_comm:
                    reason = _categorize_commentary(right_comm)
                    comm = right_comm
                missing.append(
                    MissingEntry(
                        channel=name, reason=reason,
                        index_in_requested=i, commentary=comm,
                    )
                )
                continue
            coords_array[i] = midpoint
            mapped_mask[i] = True

        else:  # monopolar
            if query.left is None:
                missing.append(
                    MissingEntry(
                        channel=name, reason="name_not_found",
                        index_in_requested=i,
                        commentary=f"could not parse monopolar {name!r}",
                    )
                )
                continue
            coord, commentary = lookup_fn(query.left[0], query.left[1])
            if coord is None:
                reason = _categorize_commentary(commentary)
                missing.append(
                    MissingEntry(
                        channel=name, reason=reason,
                        index_in_requested=i, commentary=commentary,
                    )
                )
                continue
            coords_array[i] = coord
            mapped_mask[i] = True

    # Apply affine if requested (Epilepsiae voxel → ras_mm)
    if mri_affine is not None:
        mri_affine = np.asarray(mri_affine, dtype=float)
        if mri_affine.shape != (4, 4):
            raise ValueError(f"mri_affine must be 4x4, got shape {mri_affine.shape}")
        mapped_rows = np.where(mapped_mask)[0]
        if len(mapped_rows) > 0:
            mapped_coords = coords_array[mapped_rows]
            coords_array[mapped_rows] = _apply_affine(mapped_coords, mri_affine)

    result = CoordResult(
        schema_version="coord_loader_v3",
        dataset=dataset,
        subject_id=subject_id,
        channel_names_requested=channel_names_requested,
        coords_array_in_requested_order=coords_array,
        mapped_mask_in_requested_order=mapped_mask,
        coord_space=coord_space,
        coord_units=coord_units,
        provenance=provenance,
        missing=missing,
        bipolar_resolution=bipolar_log,
        source_coord_type=source_coord_type,
        normalization_certainty=normalization_certainty,
    )
    _validate_coord_result(result)
    return result


def _validate_coord_result(result: CoordResult) -> None:
    """Check v2 invariants. Raise ValueError on violation."""
    n = len(result.channel_names_requested)
    if result.coords_array_in_requested_order.shape != (n, 3):
        raise ValueError(
            f"coords_array shape {result.coords_array_in_requested_order.shape} != ({n}, 3)"
        )
    if result.mapped_mask_in_requested_order.shape != (n,):
        raise ValueError(
            f"mapped_mask shape {result.mapped_mask_in_requested_order.shape} != ({n},)"
        )

    # mask True ↔ coords finite (invariant #1)
    finite_rows = ~np.any(np.isnan(result.coords_array_in_requested_order), axis=1)
    if not np.array_equal(finite_rows, result.mapped_mask_in_requested_order):
        raise ValueError(
            "mapped_mask must match finite-row mask of coords_array (invariant #1)"
        )

    # missing entries must have NaN rows + mask False (forward direction)
    for entry in result.missing:
        if entry.index_in_requested < 0 or entry.index_in_requested >= n:
            raise ValueError(f"missing entry index out of range: {entry}")
        if result.mapped_mask_in_requested_order[entry.index_in_requested]:
            raise ValueError(
                f"channel {entry.channel} listed as missing but mask says mapped"
            )

    # Inverse direction (v1.0.5 fix, advisor 2026-05-21 #2): every mask=False
    # index MUST have a corresponding missing[] entry. A NaN row with no
    # missing[] entry would silently lose the reason — caller can't tell why.
    missing_indices = {entry.index_in_requested for entry in result.missing}
    for i in range(n):
        if not result.mapped_mask_in_requested_order[i] and i not in missing_indices:
            raise ValueError(
                f"channel index {i} ({result.channel_names_requested[i]!r}) "
                f"has mask=False but no missing[] entry — every unmapped channel "
                f"must have an explicit missing entry with reason"
            )

    # space / units consistency (invariant #6)
    if result.coord_space == "fs_native_ras_mm" and result.coord_units != "mm":
        raise ValueError("fs_native_ras_mm must have coord_units='mm'")
    if result.coord_space == "mri_native_voxel_ijk" and result.coord_units != "voxel":
        raise ValueError("mri_native_voxel_ijk must have coord_units='voxel'")
    if result.coord_space == "ras_mm_via_affine" and result.coord_units != "mm":
        raise ValueError("ras_mm_via_affine must have coord_units='mm'")
    if result.coord_space == "mni152_1mm" and result.coord_units != "mm":
        raise ValueError("mni152_1mm must have coord_units='mm'")


# =============================================================================
# Phase 1 consumer contract: assert coord_units == "mm" before geometric analysis
# =============================================================================


def assert_coord_result_is_mm_for_main_analysis(result: CoordResult) -> None:
    """Phase 1 / H1 / H2 main analyses must consume mm coords only.

    Call this at the top of any function that takes a CoordResult and computes
    Euclidean distances. Voxel coords from Epilepsiae default loader must NOT
    silently feed main analysis (invariant #6 from coord_loader plan §4).
    """
    if result.coord_units != "mm":
        raise ValueError(
            f"Main analysis requires coord_units='mm'; got {result.coord_units!r} "
            f"(coord_space={result.coord_space!r}). "
            f"For Epilepsiae voxel coords, pass an explicit mri_affine to the loader."
        )
