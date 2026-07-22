"""MZ gradient-corridor stimulation — patient-layout virtual-SEEG site comparison on the Z+M SNN.

*** IMPORT-SAFE / SIDE-EFFECT-FREE *** No simulation runs on import and this module writes no files;
the runner (scripts/run_topic4_mz_gradient_corridor_stimulation.py) owns all I/O and simulation.

Scientific question (falsifiable): in a spiking network whose pathology axis + electrode montage are
mapped from a subject's FROZEN interictal template-gradient field, does virtual inhibition of the
mid-corridor bipolar pair, versus either axis-endpoint pair, (1) delay operational runaway, (2) reduce
cross-corridor global spread, (3) keep local events, WITHOUT (4) globally silencing everything.

Upper bound of any claim: "in this patient-layout-mapped Z+M SNN, mid-corridor virtual inhibition
changes operational runaway and model propagation extent." NOT clinical efficacy / real seizure / DBS.

Geometry contract (BINDING; CLAUDE.md §6 clauses enforced inline):
  - Axis + montage come ONLY from the frozen record's axis_pair.shared_axis.u,
    interictal_field.planes.shared.points, interictal_field.contact_order (+ shafts).
  - FORBIDDEN axis inputs: source/sink centroid, rank-displacement, decision_k, swap endpoints,
    template_source_foci, register_to_sheet, fixed top-k, D_AB, any ictal quantity. This module does
    NOT import sef_hfo_subject_placement.
  - Fingerprint fail / contract mismatch / missing shared plane -> FAIL CLOSED (raise). No fallback.

Model contract: frozen Z+M candidate zA_q75_tz5000__mA0p001_tau500 on the Stage-5 blessed SNN
organization (L=20, density=100, g=3.6, drive=0.6, AR=2.0, twoend_equal cores 17.5/18.0). theta_EE=0
(corridor aligned to sheet-x, anisotropy along corridor) identically for all patients; only the two
cores (gradient Q10/Q90) and the electrode/stim geometry differ per patient. Z/M params are NEVER
tuned per patient.
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.topic5_template_axis_field import (  # noqa: E402
    INTERICTAL_FIELD_CONTRACT,
    INTERICTAL_FIELD_FINGERPRINT_ALGORITHM,
    TEMPLATE_AXIS_DEFINITION,
    TEMPLATE_AXIS_DIRECTION,
    interictal_field_fingerprint,
)

SCHEMA_VERSION = "mz-gradient-corridor-stim-1.0"

# frozen Z+M candidate (given; NEVER re-tuned per patient)
MZ_CANDIDATE = dict(label="zA_q75_tz5000__mA0p001_tau500", use_z=True, use_m=True,
                    I_th_EI=95.19851312666987, tau_z=5000.0, tau_adp=500.0,
                    eta_m=0.007451594355587098)

PRIMARY_COHORT = ("epilepsiae_1084", "epilepsiae_1146", "epilepsiae_583",
                  "epilepsiae_590", "epilepsiae_958", "yuquan_zhaochenxi")
SENSITIVITY_COHORT = ("epilepsiae_384",)

# frozen SNN organization
SNN = dict(L=20.0, density=100.0, g=3.6, drive=0.6, dt=0.1, theta_EE_deg=0.0, AR=2.0,
           core_mean=17.5, core_std=1.0, core_r=1.5, base_mean=18.0,
           sheet_margin_mm=2.0, core_quantiles=(0.10, 0.90))


# ============================================================ frozen gradient loader (fail-closed)
def gradient_record_path(subject_id: str, input_root: str) -> str:
    return os.path.join(str(input_root), "per_subject", f"{subject_id}.json")


def load_gradient_record(subject_id: str, input_root: str) -> Dict[str, object]:
    """Load one frozen interictal template-gradient record and VERIFY it fail-closed.

    Enforces (raise on any failure, never fall back to an endpoint axis):
      - file exists;
      - contract == topic5_interictal_template_fields_v1;
      - interictal_field.status == ok and fingerprint_algorithm == sha256_v1p1_nonfinite_canonical;
      - recomputed interictal_field_fingerprint(record) == stored fingerprint_sha256;
      - axis_definition / direction convention match the template-gradient contract;
      - a shared plane (planes.shared.points) + contact_order exist.
    """
    path = gradient_record_path(subject_id, input_root)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"gradient record not found: {path}")
    record = json.loads(open(path).read())
    if record.get("contract") != INTERICTAL_FIELD_CONTRACT:
        raise ValueError(f"{subject_id}: unsupported gradient contract {record.get('contract')!r}")
    if record.get("axis_definition") != TEMPLATE_AXIS_DEFINITION:
        raise ValueError(f"{subject_id}: unexpected axis_definition {record.get('axis_definition')!r}")
    if record.get("axis_direction_convention") != TEMPLATE_AXIS_DIRECTION:
        raise ValueError(f"{subject_id}: unexpected axis direction {record.get('axis_direction_convention')!r}")
    field_ = record.get("interictal_field") or {}
    if field_.get("status") != "ok":
        raise ValueError(f"{subject_id}: interictal_field status {field_.get('status')!r} != ok")
    if field_.get("fingerprint_algorithm") != INTERICTAL_FIELD_FINGERPRINT_ALGORITHM:
        raise ValueError(f"{subject_id}: unexpected fingerprint algorithm {field_.get('fingerprint_algorithm')!r}")
    recomputed = interictal_field_fingerprint(record)         # raises internally on contract issues
    stored = field_.get("fingerprint_sha256")
    if not stored or str(stored) != recomputed:
        raise ValueError(f"{subject_id}: interictal field fingerprint MISMATCH after JSON load "
                         f"(stored={stored}, recomputed={recomputed})")
    planes = field_.get("planes") or {}
    if "shared" not in planes or (planes["shared"] or {}).get("status") != "ok":
        raise ValueError(f"{subject_id}: no usable shared gradient plane (planes.shared)")
    if not field_.get("contact_order"):
        raise ValueError(f"{subject_id}: empty contact_order")
    return record


def subject_gate_flags(record: Mapping[str, object]) -> Dict[str, object]:
    """Cohort gate booleans read straight from the frozen record (no re-derivation)."""
    pair = record.get("axis_pair") or {}
    rel = (pair.get("relation") or {}).get("relation")
    shared = (pair.get("shared_axis") or {}).get("status")
    field_ = record.get("interictal_field") or {}
    return dict(
        axis_pair_estimable=bool(pair.get("axis_pair_estimable")),
        geometry_2d_supported=bool(pair.get("geometry_2d_supported")),
        strict_stability_pass=bool(pair.get("strict_stability_pass")),
        relation=str(rel),
        relation_reversed=bool(rel == "reversed"),
        shared_axis_ok=bool(shared == "ok"),
        interictal_field_ok=bool(field_.get("status") == "ok"),
        n_field_contacts=int(field_.get("n_contacts", 0) or 0),
    )


# ============================================================ patient-specific sheet montage
@dataclass
class SheetMontage:
    """Isotropic (single-scale) map of the frozen shared gradient plane onto the L=20 SNN sheet.

    contacts/names/shafts are aligned 1:1 to the frozen contact_order. along/transverse are the
    normalized shared-plane coordinates (along = points[:,0], early->late by contract). src_xy/snk_xy
    are the two pathology cores at the Q10/Q90 along-quantiles (transverse=0, on the axis line).
    """
    names: List[str]
    shafts: List[str]
    contacts: np.ndarray            # (N,2) sheet mm
    along: np.ndarray               # (N,) normalized along (early->late)
    transverse: np.ndarray          # (N,) normalized transverse
    src_xy: np.ndarray              # (2,) core at along-Q10 (early)
    snk_xy: np.ndarray              # (2,) core at along-Q90 (late)
    axis_center_along: float
    axis_unit: np.ndarray           # (2,) sheet-x (+ = early->late)
    scale: float
    center: np.ndarray              # (2,) plane-frame bbox center
    L: float
    margin: float

    @property
    def core_separation_mm(self) -> float:
        return float(np.linalg.norm(self.snk_xy - self.src_xy))


def _trailing_int(name: str) -> Optional[int]:
    m = re.search(r"(\d+)\s*$", str(name))
    return int(m.group(1)) if m else None


def build_sheet_montage(record: Mapping[str, object], *, L: float = 20.0, margin: float = 2.0,
                        core_quantiles: Tuple[float, float] = (0.10, 0.90)) -> SheetMontage:
    """Map the frozen shared-plane points onto the sheet with ONE isotropic scale (no x/y stretch).

    Uses ONLY interictal_field.planes.shared.points + contact_order + shafts. along -> sheet-x so the
    corridor is horizontal (theta_EE=0 for every patient). Cores are the Q10/Q90 along-quantiles at
    transverse=0. Sign-flipping the along axis only swaps which core is src vs snk; the contact set,
    the transform, and the axis-center are invariant.
    """
    field_ = record["interictal_field"]
    plane = field_["planes"]["shared"]
    pts = np.asarray(plane["points"], float)                  # (N,2): col0 along(early->late), col1 transverse
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("shared plane points must be (N,2)")
    names = [str(x) for x in field_["contact_order"]]
    shafts = [str(x) for x in field_["shafts"]]
    if not (len(names) == len(shafts) == pts.shape[0]):
        raise ValueError("contact_order / shafts / shared points are not aligned")
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2.0
    span = float((maxs - mins).max())
    if not np.isfinite(span) or span <= 1e-9:
        raise ValueError("degenerate shared-plane bounding box")
    scale = float((L - 2.0 * margin) / span)                  # isotropic single scale

    def _to_sheet(p2: np.ndarray) -> np.ndarray:
        return (np.asarray(p2, float) - center) * scale + np.array([L / 2.0, L / 2.0])

    contacts = _to_sheet(pts)
    along = pts[:, 0].copy()
    transverse = pts[:, 1].copy()
    q_lo, q_hi = core_quantiles
    core_neg_along = float(np.quantile(along, q_lo))
    core_pos_along = float(np.quantile(along, q_hi))
    src_xy = _to_sheet(np.array([core_neg_along, 0.0]))       # early core
    snk_xy = _to_sheet(np.array([core_pos_along, 0.0]))       # late core
    axis_center_along = 0.5 * (core_neg_along + core_pos_along)
    axis_vec = snk_xy - src_xy
    axis_unit = axis_vec / (np.linalg.norm(axis_vec) + 1e-12)
    return SheetMontage(names=names, shafts=shafts, contacts=contacts, along=along, transverse=transverse,
                        src_xy=src_xy, snk_xy=snk_xy, axis_center_along=axis_center_along,
                        axis_unit=axis_unit, scale=scale, center=center, L=float(L), margin=float(margin))


# ============================================================ bipolar stimulation-site selection
@dataclass
class BipolarSite:
    kind: str                       # endpoint_negative / endpoint_positive / middle / offaxis
    name_a: str
    name_b: str
    idx_a: int
    idx_b: int
    shaft: str
    mid_sheet: np.ndarray           # (2,)
    mid_along: float
    mid_transverse: float


def _adjacent_pairs(montage: SheetMontage) -> List[BipolarSite]:
    """All same-shaft, adjacent-numbered (|Δ|==1) bipolar pairs among the frozen contacts."""
    nums = [_trailing_int(n) for n in montage.names]
    out: List[BipolarSite] = []
    for i, j in combinations(range(len(montage.names)), 2):
        if montage.shafts[i] != montage.shafts[j]:
            continue
        if nums[i] is None or nums[j] is None or abs(nums[i] - nums[j]) != 1:
            continue
        mid_sheet = 0.5 * (montage.contacts[i] + montage.contacts[j])
        out.append(BipolarSite(kind="candidate", name_a=montage.names[i], name_b=montage.names[j],
                               idx_a=i, idx_b=j, shaft=montage.shafts[i], mid_sheet=mid_sheet,
                               mid_along=0.5 * (montage.along[i] + montage.along[j]),
                               mid_transverse=0.5 * (montage.transverse[i] + montage.transverse[j])))
    return out


def select_bipolar_sites(montage: SheetMontage, *, offaxis_along_tol: float = 0.20
                         ) -> Dict[str, object]:
    """Pick the three primary bipolar stim sites (+ optional off-axis) from the frozen montage.

    endpoint_negative/positive = adjacent pair whose midpoint is the lowest/highest along the shared
    gradient axis (the x-quantile ends). middle = adjacent pair whose midpoint is closest, IN 2D, to the
    corridor-center point (along-center on the axis line, transverse=0) so it sits on the corridor
    between the two cores rather than merely at a central along-coordinate; disjoint from both endpoints.
    off-axis = a legal pair near the corridor-center along but with the largest lateral (transverse)
    offset, disjoint from all three primaries (None if none qualifies).

    Sign of the along axis only swaps endpoint_negative<->positive; the middle and the site SET are
    invariant (the 2D distance to (center, 0) is symmetric under along-negation). RAISES if fewer than
    three distinct non-overlapping bipolar sites exist.
    """
    pairs = _adjacent_pairs(montage)
    result: Dict[str, object] = {"n_adjacent_pairs": len(pairs), "sites": {}, "reason": None}
    if len(pairs) < 3:
        result["reason"] = f"only {len(pairs)} adjacent same-shaft bipolar pair(s) among frozen contacts"
        return result
    by_along = sorted(pairs, key=lambda s: s.mid_along)
    neg = by_along[0]
    pos = by_along[-1]
    used = {neg.name_a, neg.name_b, pos.name_a, pos.name_b}
    mid_cands = [s for s in pairs if not ({s.name_a, s.name_b} & used)]
    if not mid_cands:
        result["reason"] = "no middle bipolar pair disjoint from both endpoints"
        return result

    def _dist2_to_center(s):
        # 2D distance from the pair midpoint to the corridor center (along-center, transverse=0)
        return (s.mid_along - montage.axis_center_along) ** 2 + s.mid_transverse ** 2

    middle = min(mid_cands, key=_dist2_to_center)
    used_all = used | {middle.name_a, middle.name_b}
    off_cands = [s for s in pairs if not ({s.name_a, s.name_b} & used_all)
                 and abs(s.mid_along - montage.axis_center_along) <= offaxis_along_tol]
    offaxis = None
    if off_cands:
        offaxis = max(off_cands, key=lambda s: abs(s.mid_transverse))
    neg = _relabel(neg, "endpoint_negative")
    pos = _relabel(pos, "endpoint_positive")
    middle = _relabel(middle, "middle")
    sites = {"endpoint_negative": neg, "endpoint_positive": pos, "middle": middle}
    if offaxis is not None:
        sites["offaxis"] = _relabel(offaxis, "offaxis")
    result["sites"] = sites
    return result


def _relabel(s: BipolarSite, kind: str) -> BipolarSite:
    return BipolarSite(kind=kind, name_a=s.name_a, name_b=s.name_b, idx_a=s.idx_a, idx_b=s.idx_b,
                       shaft=s.shaft, mid_sheet=s.mid_sheet, mid_along=s.mid_along,
                       mid_transverse=s.mid_transverse)


# ============================================================ geometry audit row (per subject)
def audit_subject_geometry(subject_id: str, input_root: str, *, tier: str) -> Dict[str, object]:
    """One geometry_audit.csv row: fingerprint/contract/relation/stability/shared-plane + bipolar sites.

    Never raises for a scientific exclusion (records exclusion_reason instead); DOES raise only for a
    truly unreadable / fingerprint-broken artifact via load_gradient_record (that is a data-integrity
    stop, not an exclusion). tier in {primary_candidate, sensitivity}.
    """
    row: Dict[str, object] = {"subject_id": subject_id, "tier": tier,
                              "gradient_contract": None, "fingerprint_ok": False,
                              "relation": None, "strict_stability_pass": None,
                              "geometry_2d_supported": None, "shared_plane_ok": None,
                              "n_field_contacts": None, "n_shafts": None, "n_adjacent_pairs": None,
                              "core_separation_mm": None, "cores_nonoverlapping": None,
                              "endpoint_negative_pair": None, "endpoint_positive_pair": None,
                              "middle_pair": None, "offaxis_pair": None,
                              "three_sites_ok": False, "admitted": False, "exclusion_reason": None}
    try:
        record = load_gradient_record(subject_id, input_root)
    except Exception as exc:
        row["exclusion_reason"] = f"load/fingerprint failure: {type(exc).__name__}: {exc}"
        return row
    row["fingerprint_ok"] = True
    row["gradient_contract"] = record.get("contract")
    flags = subject_gate_flags(record)
    row.update(relation=flags["relation"], strict_stability_pass=flags["strict_stability_pass"],
               geometry_2d_supported=flags["geometry_2d_supported"], shared_plane_ok=flags["shared_axis_ok"],
               n_field_contacts=flags["n_field_contacts"])
    montage = build_sheet_montage(record, L=SNN["L"], margin=SNN["sheet_margin_mm"],
                                  core_quantiles=SNN["core_quantiles"])
    row["n_shafts"] = len(set(montage.shafts))
    sep = montage.core_separation_mm
    min_sep = 2.0 * float(SNN["core_r"])                      # cores must be geometrically distinct
    row["core_separation_mm"] = round(sep, 3)
    row["cores_nonoverlapping"] = bool(sep > min_sep)
    sel = select_bipolar_sites(montage)
    row["n_adjacent_pairs"] = sel["n_adjacent_pairs"]
    reasons = []
    if not flags["relation_reversed"]:
        reasons.append(f"relation={flags['relation']}!=reversed")
    if not flags["geometry_2d_supported"]:
        reasons.append("geometry_2d_unsupported")
    if tier == "primary_candidate" and not flags["strict_stability_pass"]:
        reasons.append("not_strict_stability")
    if sep <= min_sep:
        reasons.append(f"corridor_degenerate (core_sep {sep:.2f}mm <= 2*core_r {min_sep:.1f}mm)")
    if not sel["sites"]:
        reasons.append(sel["reason"] or "no_three_sites")
    else:
        s = sel["sites"]
        row["three_sites_ok"] = True
        row["endpoint_negative_pair"] = f"{s['endpoint_negative'].name_a}-{s['endpoint_negative'].name_b}"
        row["endpoint_positive_pair"] = f"{s['endpoint_positive'].name_a}-{s['endpoint_positive'].name_b}"
        row["middle_pair"] = f"{s['middle'].name_a}-{s['middle'].name_b}"
        row["offaxis_pair"] = (f"{s['offaxis'].name_a}-{s['offaxis'].name_b}" if "offaxis" in s else None)
    row["exclusion_reason"] = "; ".join(reasons) if reasons else None
    row["admitted"] = bool(not reasons and row["three_sites_ok"])
    return row


# ============================================================ SNN substrate (frozen organization)
# Import the blessed engine numerics + accepted MZ slow object. Importing topic4_mz_onset_dynamics
# also puts src/snn_engine on sys.path (its module header does the insert) -> params/connectivity load.
from src.topic4_mz_onset_dynamics import (  # noqa: E402
    _loop_consts, score_runaway, MZOnsetProbe,
)
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402
from kick_probe import _flatten_by_source, membrane_step  # noqa: E402
from params import Params  # noqa: E402
from connectivity import place_neurons  # noqa: E402
from connectivity_rot import build_connectivity_rot  # noqa: E402
from src.sef_hfo_heterogeneity import sample_core_field  # noqa: E402
from src.sef_hfo_events import detect_events  # noqa: E402

# frozen active-fraction / event-bar constants (mirror run_sef_hfo_snn_cm_spontaneous_readout C.*)
ACTIVE_BIN_MS = 1.0
BASELINE_MS = (5.0, 50.0)
CAL_FRAC = 0.5


def mz_config(candidate: Mapping[str, object] = MZ_CANDIDATE) -> "MZSlowVarsConfig":
    return MZSlowVarsConfig(use_z=bool(candidate["use_z"]), use_m=bool(candidate["use_m"]),
                            I_th_EI=float(candidate["I_th_EI"]), tau_z=float(candidate["tau_z"]),
                            tau_adp=float(candidate["tau_adp"]), eta_m=float(candidate["eta_m"]))


def build_shared_net(seed: int) -> Dict[str, object]:
    """Frozen SNN organization (identical across patients for a given seed): Params, uniform E/I
    placement, and E->E anisotropic connectivity with theta_EE=0 (corridor aligned to sheet-x, AR=2).
    Nothing here depends on the patient montage -> the same net is reused for every patient at a seed.
    """
    p = Params(g=SNN["g"], L=SNN["L"], density=SNN["density"], T=1.0, dt=SNN["dt"],
               nu_ext_ratio=SNN["drive"], seed=int(seed))
    rng = np.random.default_rng(int(seed))
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng,
                                 theta_EE=np.deg2rad(SNN["theta_EE_deg"]), AR=SNN["AR"], verbose=False)
    # precompute the flattened scatter tables ONCE in the parent so fork workers share them (COW).
    _loop_consts(p, net)
    return dict(p=p, net=net, NE=int(NE), NI=int(NI), N=int(NE + NI),
                posE=net["pos"][:NE], posI=net["pos"][NE:], labels=labels, seed=int(seed))


def build_patient_vth(shared: Mapping[str, object], montage: SheetMontage) -> Dict[str, object]:
    """Two low-V_th cores at the patient gradient Q10/Q90 (src/snk), same core magnitude/radius as the
    blessed twoend_equal substrate. Returns vth (N,) and the E-indexed union core mask."""
    net = shared["net"]
    NE = shared["NE"]
    is_E = np.zeros(shared["N"], bool)
    is_E[:NE] = True
    seed = shared["seed"]

    def _core_vth(xy, s):
        return sample_core_field(net["pos"], is_E, np.asarray(xy, float), SNN["core_r"],
                                 np.random.default_rng(s), core_mean=SNN["core_mean"],
                                 core_std=SNN["core_std"], base_mean=SNN["base_mean"])["vth"]

    vth = np.minimum(_core_vth(montage.src_xy, seed + 7), _core_vth(montage.snk_xy, seed + 8))
    posE = shared["posE"]
    core_mask_E = ((np.linalg.norm(posE - montage.src_xy, axis=1) <= SNN["core_r"])
                   | (np.linalg.norm(posE - montage.snk_xy, axis=1) <= SNN["core_r"]))
    return dict(vth=vth, core_mask_E=core_mask_E)


def build_stim_targets(posE: np.ndarray, sites: Mapping[str, BipolarSite], *, radius_mm: float
                       ) -> Dict[str, object]:
    """Match the number of clamped E cells across every stim site (dose-matched arms).

    Each site's candidates = E cells within ``radius_mm`` of its bipolar-pair midpoint; N_target =
    min candidate count over all this subject's sites; each site clamps its nearest N_target E cells.
    Records the per-site effective radius (distance to the N_target-th nearest E cell).
    """
    posE = np.asarray(posE, float)
    order = [k for k in ("endpoint_negative", "endpoint_positive", "middle", "offaxis") if k in sites]
    dists = {k: np.linalg.norm(posE - sites[k].mid_sheet, axis=1) for k in order}
    cand_counts = {k: int((dists[k] <= radius_mm).sum()) for k in order}
    n_target = int(min(cand_counts.values())) if cand_counts else 0
    masks: Dict[str, np.ndarray] = {}
    eff_radius: Dict[str, float] = {}
    for k in order:
        idx = np.argsort(dists[k])[:n_target]
        m = np.zeros(len(posE), bool)
        m[idx] = True
        masks[k] = m
        eff_radius[k] = float(dists[k][idx].max()) if n_target > 0 else 0.0
    return dict(n_target=n_target, masks=masks, eff_radius=eff_radius,
                candidate_counts=cand_counts, radius_mm=float(radius_mm))


# ============================================================ streaming spatial observer
class SpatialStreamObserver:
    """Reduce per-step E spikes to compact spatial summaries WITHOUT a T x NE raster (spec: memory).

    Records (a) the 1-ms active-fraction trace (mean over E of "fired in bin", identical to
    C.active_fraction), and (b) the fraction of E cells active per axial / transverse spatial bin over
    ``spatial_bin_ms`` windows. Axial coordinate = projection on the corridor axis (0 at corridor
    center); the two cores sit near +/- core_separation/2.
    """

    def __init__(self, posE, axis_center_xy, axis_unit, *, dt, L, n_steps,
                 active_bin_ms=ACTIVE_BIN_MS, spatial_bin_ms=5.0, axial_bins=20, transverse_bins=12):
        posE = np.asarray(posE, float)
        self.NE = len(posE)
        u = np.asarray(axis_unit, float)
        u = u / (np.linalg.norm(u) + 1e-12)
        perp = np.array([-u[1], u[0]])
        c = np.asarray(axis_center_xy, float)
        self.axial_coord = (posE - c) @ u
        self.trans_coord = (posE - c) @ perp
        self.ax_edges = np.linspace(-L / 2.0, L / 2.0, axial_bins + 1)
        self.tr_edges = np.linspace(-L / 2.0, L / 2.0, transverse_bins + 1)
        self.ax_idx = np.clip(np.digitize(self.axial_coord, self.ax_edges) - 1, 0, axial_bins - 1)
        self.tr_idx = np.clip(np.digitize(self.trans_coord, self.tr_edges) - 1, 0, transverse_bins - 1)
        self.ax_cell_count = np.maximum(np.bincount(self.ax_idx, minlength=axial_bins).astype(float), 1.0)
        self.tr_cell_count = np.maximum(np.bincount(self.tr_idx, minlength=transverse_bins).astype(float), 1.0)
        self.axial_bins = axial_bins
        self.transverse_bins = transverse_bins
        self.dt = float(dt)
        self.ab = max(1, int(round(active_bin_ms / dt)))
        self.sb = max(1, int(round(spatial_bin_ms / dt)))
        self.active_frac = np.zeros(n_steps // self.ab + 2)
        self.axial_act = np.zeros((n_steps // self.sb + 2, axial_bins))
        self.trans_act = np.zeros((n_steps // self.sb + 2, transverse_bins))
        self.spatial_bin_ms = float(spatial_bin_ms)
        self.active_bin_ms = float(active_bin_ms)
        self._fa = np.zeros(self.NE, bool)
        self._fs = np.zeros(self.NE, bool)
        self._na = 0
        self._ns = 0

    def record(self, k, spkE):
        np.logical_or(self._fa, spkE, out=self._fa)
        np.logical_or(self._fs, spkE, out=self._fs)
        if (k + 1) % self.ab == 0:
            self.active_frac[self._na] = self._fa.mean()
            self._na += 1
            self._fa[:] = False
        if (k + 1) % self.sb == 0:
            cnt = np.bincount(self.ax_idx[self._fs], minlength=self.axial_bins).astype(float)
            self.axial_act[self._ns] = cnt / self.ax_cell_count
            cnt_t = np.bincount(self.tr_idx[self._fs], minlength=self.transverse_bins).astype(float)
            self.trans_act[self._ns] = cnt_t / self.tr_cell_count
            self._ns += 1
            self._fs[:] = False

    def finalize(self):
        self.active_frac = self.active_frac[:self._na]
        self.axial_act = self.axial_act[:self._ns]
        self.trans_act = self.trans_act[:self._ns]
        return self


# ============================================================ observed integration loop (run_loop + hooks)
def run_observed_loop(p, net, slow, V_th_per_neuron, *, n_steps, observer=None, lfp_recorder=None,
                      lfp_every=10, early_stop_runaway=True, es_thresh_hz=120.0, es_dur_ms=100.0):
    """Faithful copy of topic4_mz_onset_dynamics.run_loop (store_spikes=False) with two read-only hooks:
    a per-step spatial ``observer`` and a strided ``lfp_recorder``. With observer=None and
    lfp_recorder=None the per-step work and RNG draw order are identical to run_loop -> a bit-parity gate
    (tests) guards against numeric drift. NEVER allocates a T x NE raster.
    """
    c = _loop_consts(p, net)
    NE, NI, N, M, dt = c["NE"], c["NI"], c["N"], c["M"], c["dt"]
    labels = c["labels"]
    a_indptr, a_dst, a_dly, a_w = net["ampa_flat"]
    g_indptr, g_dst, g_dly, g_w = net["gaba_flat"]
    rng = net["rng"]
    base_vth = p.V_th if V_th_per_neuron is None else np.asarray(V_th_per_neuron, float)

    t0 = 0
    V = np.full(N, p.V_reset, dtype=np.float64)
    ref = np.zeros(N, dtype=np.int32)
    s_E = np.zeros(N); I_E = np.zeros(N); s_I = np.zeros(N); I_I = np.zeros(N)
    ring_sE = np.zeros((M, N)); ring_sI = np.zeros((M, N))
    xi = 0.0
    _ = rng.choice(NE, size=min(80, NE), replace=False)          # stream fidelity: match run_loop / simulate_kick
    _ = NE + rng.choice(NI, size=min(20, NI), replace=False)

    rate_E = np.zeros(n_steps); rate_I = np.zeros(n_steps)
    lfp_trace = None
    lfp_idx = None
    if lfp_recorder is not None:
        n_lfp = n_steps // lfp_every + 1
        lfp_trace = np.zeros((n_lfp, len(lfp_recorder.sites)))
        lfp_idx = np.zeros(n_lfp, dtype=np.int64)
    _nl = 0
    _es_alpha = 1.0 - np.exp(-dt / 20.0); _es_ema = 0.0
    _es_dur = int(round(es_dur_ms / dt)); _es_run = 0; _stop_k = n_steps

    for k in range(n_steps):
        t = t0 + k
        xi = c["ou_a"] * xi + c["ou_b"] * rng.standard_normal()
        nu_now = c["nu_sig_const"] + xi
        if nu_now < 0.0:
            nu_now = 0.0
        s_E *= c["decay_sE"]; s_I *= c["decay_sI"]
        slot = t % M
        s_E += ring_sE[slot]; ring_sE[slot] = 0.0
        s_I += ring_sI[slot]; ring_sI[slot] = 0.0
        nu_vec = np.full(N, max(nu_now, 0.0))
        ext = rng.poisson(nu_vec * dt, size=N).astype(np.float64)
        s_E += ext * c["ext_incr"]
        I_E = s_E + (I_E - s_E) * c["decay_IE"]
        I_I = s_I + (I_I - s_I) * c["decay_II"]
        if lfp_trace is not None and (k % lfp_every == 0):
            lfp_trace[_nl] = lfp_recorder.sample(I_E, I_I)       # current-based LFP (reuse LFPRecorder)
            lfp_idx[_nl] = k
            _nl += 1
        if slow is not None:
            I_net = slow.apply_currents(I_E, I_I, labels)
            V_th_eff = slow.threshold(base_vth)
        else:
            I_net = I_E - I_I
            V_th_eff = base_vth
        ref -= 1
        np.maximum(ref, 0, out=ref)
        free = ref == 0
        if slow is not None:
            Vtmp = I_net + (V - I_net) * c["decay_V"]
        else:
            Vtmp = membrane_step(V, I_E, I_I, c["decay_V"])
        V = np.where(free, Vtmp, p.V_reset)
        spk = free & (V >= (V_th_eff if np.isscalar(V_th_eff) else V_th_eff))
        V[spk] = p.V_reset
        ref[spk] = c["ref_steps"][spk]
        if slow is not None:
            slow.step(spk, labels, dt)
        rate_E[k] = spk[:NE].sum(); rate_I[k] = spk[NE:].sum()
        if observer is not None:
            observer.record(k, spk[:NE])
        spE = np.where(spk[:NE])[0]; spI = np.where(spk[NE:])[0]
        if spE.size:
            st = a_indptr[spE]; cnt = a_indptr[spE + 1] - st; tot = int(cnt.sum())
            if tot:
                idx = (np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt) + np.repeat(st, cnt))
                np.add.at(ring_sE, ((t + a_dly[idx]) % M, a_dst[idx]), a_w[idx])
        if spI.size:
            st = g_indptr[spI]; cnt = g_indptr[spI + 1] - st; tot = int(cnt.sum())
            if tot:
                idx = (np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt) + np.repeat(st, cnt))
                np.add.at(ring_sI, ((t + g_dly[idx]) % M, g_dst[idx]), g_w[idx])
        if early_stop_runaway:
            _es_ema += _es_alpha * (rate_E[k] / NE / dt * 1e3 - _es_ema)
            _es_run = _es_run + 1 if _es_ema >= es_thresh_hz else 0
            if _es_run >= _es_dur:
                _stop_k = k + 1
                break

    if _stop_k < n_steps:
        rate_E = rate_E[:_stop_k]; rate_I = rate_I[:_stop_k]
    if observer is not None:
        observer.finalize()
    if lfp_trace is not None:
        lfp_trace = lfp_trace[:_nl]; lfp_idx = lfp_idx[:_nl]
    return dict(rate_E=rate_E / NE / dt * 1e3, rate_I=rate_I / NI / dt * 1e3,
                n_steps=len(rate_E), runaway_early_stop_step=(None if _stop_k >= n_steps else _stop_k),
                lfp_trace=lfp_trace, lfp_step_idx=lfp_idx)


# ============================================================ event / runaway / propagation metrics
AXIAL_ACTIVE_FRAC = 0.05     # an axial bin counts as "active" in a 5-ms window if this frac of its E cells fired
LOCAL_SPAN_FRAC = 0.5        # a discrete event is "local" if its axial span < this * corridor length


def _downsample(a, target=3000):
    a = np.asarray(a, np.float32)
    if a.size <= target:
        return a
    return a[:: max(1, a.size // target)]


INTERICTAL_BAR_WINDOW_MS = 4000.0    # early z-undepleted window that sets the interictal event scale


def frozen_event_bar(active_frac: np.ndarray, *, bin_ms: float = ACTIVE_BIN_MS,
                     interictal_window_ms: float = INTERICTAL_BAR_WINDOW_MS) -> float:
    """Interictal-scale frozen event-onset bar: floor + CAL_FRAC*(early-window af max - floor).

    floor = P95 over the [5,50] ms quiet window. The event scale is taken from the FIRST
    ``interictal_window_ms`` (z still ~undepleted ~= the slow-off interictal regime) rather than the
    whole-run af max, because in this candidate the active fraction ramps up toward the late runaway
    (~0.14) which would inflate the bar and miss the smaller early interictal events (~0.05). This
    approximates the blessed slow-off calibration without an extra full slow-off run. Computed ONCE on
    the baseline; reused for every arm.
    """
    af = np.asarray(active_frac, float)
    nb0, nb1 = int(BASELINE_MS[0] / bin_ms), int(BASELINE_MS[1] / bin_ms)
    floor = float(np.percentile(af[nb0:nb1], 95)) if nb1 > nb0 and af.size > nb1 else float(af.min())
    ew = int(interictal_window_ms / bin_ms)
    early = af[:ew] if af.size > ew else af
    scale = float(early.max())
    if not np.isfinite(scale) or scale <= floor:
        scale = float(af.max())                              # fallback: no early events -> whole-run scale
    return floor + CAL_FRAC * (scale - floor)


def lfp_band_summary(lfp_trace, lfp_step_idx, dt, *, lo=30.0, hi=80.0, window_steps=None):
    """30-80 Hz band power per site + total, optionally restricted to a step window. lfp sampled at
    lfp_step_idx (strided); fs = 1000/(stride_ms). Returns dict(total_power, per_window powers)."""
    if lfp_trace is None or len(lfp_trace) < 16:
        return dict(available=False)
    from scipy.signal import butter, filtfilt
    x = np.asarray(lfp_trace, float)                          # (n_lfp, n_sites)
    idx = np.asarray(lfp_step_idx, int)
    stride = int(np.median(np.diff(idx))) if len(idx) > 1 else 10
    fs = 1000.0 / (stride * dt)                               # Hz
    ny = fs / 2.0
    if hi >= ny:
        hi = 0.95 * ny
    b, a = butter(4, [lo / ny, hi / ny], btype="band")
    filt = filtfilt(b, a, x, axis=0)
    power_all = (filt ** 2)                                   # (n_lfp, n_sites)

    def _win_power(lo_s, hi_s):
        m = (idx >= lo_s) & (idx < hi_s)
        return float(power_all[m].mean()) if m.any() else float("nan")

    out = dict(available=True, fs_hz=float(fs), stride_steps=stride,
               total_power=float(power_all.mean()), per_site_power=power_all.mean(axis=0).tolist())
    if window_steps is not None:
        on_s, off_s, end_s = window_steps
        out["pre_power"] = _win_power(0, on_s)
        out["during_power"] = _win_power(on_s, off_s)
        out["post_power"] = _win_power(off_s, end_s)
    return out


def propagation_metrics(observer: SpatialStreamObserver, coredist_mm: float, dt: float, *,
                        post_bin0: int, core_r: float, active_frac: float = AXIAL_ACTIVE_FRAC):
    """Cross-corridor spread metrics from the axial activity matrix, restricted to POST-stim windows.

    Regions along the corridor axis (origin=corridor center; cores at +/- coredist/2):
      source-side  axial <= -coredist/2 + core_r ;  sink-side axial >= coredist/2 - core_r ;
      far-beyond   |axial| > coredist/2 + core_r .
    Metrics (all restricted to spatial bins >= post_bin0):
      escape_prob        = frac of post windows where the active axial span >= 0.8 * corridor length
      far_side_recruit   = max over post windows of min(source-side, sink-side) mean activity (both ends lit)
      max_axial_span_norm= max over post windows of (active axial extent / corridor length)
      participation_peak = max over post windows of the mean active fraction across all axial bins
      beyond_reach       = max over post windows of the mean activity in far-beyond bins
    """
    ax = np.asarray(observer.axial_act, float)                # (n_sbins, n_axial_bins)
    if ax.shape[0] <= post_bin0:
        return dict(escape_prob=0.0, far_side_recruit=0.0, max_axial_span_norm=0.0,
                    participation_peak=0.0, beyond_reach=0.0, n_post_windows=0)
    post = ax[post_bin0:]                                     # (T_post, n_axial_bins)
    centers = 0.5 * (observer.ax_edges[:-1] + observer.ax_edges[1:])
    half = coredist_mm / 2.0
    src_mask = centers <= (-half + core_r)
    snk_mask = centers >= (half - core_r)
    far_mask = np.abs(centers) > (half + core_r)
    active_bins = post > active_frac                          # (T_post, n_axial_bins)
    span_norm = np.zeros(post.shape[0])
    for i in range(post.shape[0]):
        on = np.where(active_bins[i])[0]
        if on.size >= 1:
            extent = centers[on.max()] - centers[on.min()]
            span_norm[i] = extent / max(coredist_mm, 1e-6)
    src_act = post[:, src_mask].mean(axis=1) if src_mask.any() else np.zeros(post.shape[0])
    snk_act = post[:, snk_mask].mean(axis=1) if snk_mask.any() else np.zeros(post.shape[0])
    far_act = post[:, far_mask].mean(axis=1) if far_mask.any() else np.zeros(post.shape[0])
    both_ends = np.minimum(src_act, snk_act)
    return dict(
        escape_prob=float(np.mean(span_norm >= 0.8)),
        far_side_recruit=float(both_ends.max()),
        max_axial_span_norm=float(span_norm.max()),
        participation_peak=float(post.mean(axis=1).max()),
        beyond_reach=float(far_act.max()),
        n_post_windows=int(post.shape[0]),
    )


def local_event_metrics(active_frac, axial_act, bar, dt, *, spatial_bin_ms, coredist_mm, core_r,
                        stim_off_ms, axial_edges, active_axial=AXIAL_ACTIVE_FRAC):
    """Post-stim discrete-event bookkeeping: how many events survive and are LOCAL (small axial span)
    vs the run being globally silenced. Events from the frozen-bar detector on the active-fraction trace.
    """
    events = detect_events(np.asarray(active_frac, float), ACTIVE_BIN_MS, event_on_frac=float(bar))
    post = [e for e in events if e["t_on"] >= stim_off_ms]
    centers = 0.5 * (np.asarray(axial_edges)[:-1] + np.asarray(axial_edges)[1:])
    sb = float(spatial_bin_ms)
    n_local = 0
    n_global = 0
    for e in post:
        b0, b1 = int(e["t_on"] / sb), int(e["t_off"] / sb) + 1
        seg = np.asarray(axial_act)[b0:b1]
        if seg.size == 0:
            continue
        peak = seg.max(axis=0)
        on = np.where(peak > active_axial)[0]
        span = (centers[on.max()] - centers[on.min()]) if on.size >= 1 else 0.0
        if span < LOCAL_SPAN_FRAC * coredist_mm:
            n_local += 1
        else:
            n_global += 1
    return dict(n_post_events=len(post), n_post_local_events=n_local, n_post_global_events=n_global,
                post_events=[dict(t_on=e["t_on"], t_off=e["t_off"], dur_ms=e["dur_ms"],
                                  peak_ext=e["peak_ext"], returned=e["returned"]) for e in post])


def prerunaway_propagation(axial_act, ax_edges, coredist_mm, core_r, spatial_bin_ms, *,
                           stim_off_ms, t_run_ms, active_frac=AXIAL_ACTIVE_FRAC):
    """Cross-corridor spread restricted to the POST-stim, PRE-runaway window [stim_off, t_run).

    The whole-run escape/far metrics saturate because the terminal global runaway floods the sheet
    regardless of stim site. This isolates the pre-runaway window (or the full post-stim window when the
    arm never runs away = censored) so the question is whether stimulation slows/limits cross-corridor
    recruitment BEFORE the global runaway, not after.

    Returns (all NaN/None if the window is empty):
      cross_corridor_latency_ms  first time (rel. stim_off) BOTH core regions are simultaneously active
      far_reach_prob             fraction of window bins with both core regions active
      mid_crossing_ms            first time the corridor-center region is active
      max_axial_span_norm_pre    max active axial extent / corridor length in the window
      window_ms                  length of the pre-runaway window (report so a short window is visible)
    """
    ax = np.asarray(axial_act, float)
    centers = 0.5 * (np.asarray(ax_edges)[:-1] + np.asarray(ax_edges)[1:])
    half = float(coredist_mm) / 2.0
    src_mask = centers <= (-half + core_r)
    snk_mask = centers >= (half - core_r)
    mid_mask = np.abs(centers) <= core_r
    sb = float(spatial_bin_ms)
    b0 = int(stim_off_ms / sb)
    b1 = int(t_run_ms / sb) if t_run_ms is not None else ax.shape[0]
    b1 = min(b1, ax.shape[0])
    nan = float("nan")
    if b1 <= b0:
        return dict(cross_corridor_latency_ms=None, far_reach_prob=nan, mid_crossing_ms=None,
                    max_axial_span_norm_pre=nan, window_ms=0.0, n_window_bins=0)
    win = ax[b0:b1]
    src_act = win[:, src_mask].mean(axis=1) if src_mask.any() else np.zeros(win.shape[0])
    snk_act = win[:, snk_mask].mean(axis=1) if snk_mask.any() else np.zeros(win.shape[0])
    mid_act = win[:, mid_mask].mean(axis=1) if mid_mask.any() else np.zeros(win.shape[0])
    both = np.minimum(src_act, snk_act) > active_frac
    idx = np.where(both)[0]
    midx = np.where(mid_act > active_frac)[0]
    span = np.zeros(win.shape[0])
    active_bins = win > active_frac
    for i in range(win.shape[0]):
        on = np.where(active_bins[i])[0]
        if on.size:
            span[i] = (centers[on.max()] - centers[on.min()]) / max(coredist_mm, 1e-6)
    return dict(cross_corridor_latency_ms=(float(idx[0] * sb) if idx.size else None),
                far_reach_prob=float(both.mean()),
                mid_crossing_ms=(float(midx[0] * sb) if midx.size else None),
                max_axial_span_norm_pre=float(span.max()),
                window_ms=float((b1 - b0) * sb), n_window_bins=int(b1 - b0))


# ============================================================ per-arm driver + summary
def run_arm(shared: Mapping[str, object], patient: Mapping[str, object], montage: SheetMontage, *,
            arm: str, target_E: Optional[np.ndarray], stim_window_steps: Optional[Tuple[int, int]],
            delta_mv: float, n_steps: int, mz_candidate: Mapping[str, object] = MZ_CANDIDATE,
            spatial_bin_ms: float = 5.0, axial_bins: int = 20, transverse_bins: int = 12,
            lfp_every: int = 10):
    """Run ONE arm (baseline or a stim site) on the patient substrate. Identical noise (net rng reset to
    the subject-seed) across arms so pre-stim trajectories match. Stim arms raise V_th (virtual
    inhibition) on target_E over stim_window_steps. Returns (res, observer, slow)."""
    from lfp import LFPRecorder
    p = shared["p"]; net = shared["net"]; NE = shared["NE"]; N = shared["N"]
    cfg = mz_config(mz_candidate)
    slow = MZOnsetProbe(N, 18.0, cfg, NE=NE, core_mask_E=patient["core_mask_E"])
    if arm != "baseline_no_stim" and target_E is not None and stim_window_steps is not None:
        lo, hi = stim_window_steps
        slow.set_suppression(lo=int(lo), hi=int(hi), target_E=np.asarray(target_E, bool), delta=float(delta_mv))
    lfp_rec = LFPRecorder(p, net["pos"], shared["labels"], sites=np.asarray(montage.contacts, float))
    axis_center = 0.5 * (montage.src_xy + montage.snk_xy)
    obs = SpatialStreamObserver(shared["posE"], axis_center, montage.axis_unit, dt=p.dt, L=montage.L,
                                n_steps=n_steps, spatial_bin_ms=spatial_bin_ms,
                                axial_bins=axial_bins, transverse_bins=transverse_bins)
    net["rng"] = np.random.default_rng(int(shared["seed"]))    # identical noise realization across arms
    res = run_observed_loop(p, net, slow, patient["vth"], n_steps=n_steps, observer=obs,
                            lfp_recorder=lfp_rec, lfp_every=lfp_every, early_stop_runaway=True)
    return res, obs, slow


def summarize_run(res: Mapping[str, object], obs: SpatialStreamObserver, slow, *, arm: str, dt: float,
                  frozen_bar: float, stim_on_ms: float, stim_off_ms: float, t_max_ms: float,
                  coredist_mm: float, core_r: float, spatial_bin_ms: float,
                  baseline_total_activity: Optional[float] = None) -> Dict[str, object]:
    """Build the compact per-run summary: restricted runaway-free time (with real t_run + censor flag),
    pre/during/post windows, propagation spread, local-event preservation, rebound, LFP band power."""
    rate = np.asarray(res["rate_E"], float)
    runaway_ms = score_runaway(rate, dt)
    if res.get("runaway_early_stop_step") is not None and runaway_ms is None:
        runaway_ms = res["runaway_early_stop_step"] * dt
    censored = runaway_ms is None
    t_run = float(t_max_ms) if censored else float(runaway_ms)
    rrt = min(t_run, t_max_ms) - stim_off_ms
    af = np.asarray(obs.active_frac, float)
    events = detect_events(af, ACTIVE_BIN_MS, event_on_frac=float(frozen_bar))
    n_pre = sum(1 for e in events if e["t_off"] <= stim_on_ms)
    n_pre_recover = sum(1 for e in events if e["t_off"] <= stim_on_ms and e["returned"])
    post_bin0 = int(stim_off_ms / spatial_bin_ms)
    prop = propagation_metrics(obs, coredist_mm, dt, post_bin0=post_bin0, core_r=core_r)
    loc = local_event_metrics(af, obs.axial_act, frozen_bar, dt, spatial_bin_ms=spatial_bin_ms,
                              coredist_mm=coredist_mm, core_r=core_r, stim_off_ms=stim_off_ms,
                              axial_edges=obs.ax_edges)
    on_s, off_s, end_s = int(stim_on_ms / dt), int(stim_off_ms / dt), len(rate)

    def _mrate(a, b):
        s = rate[max(0, a):min(len(rate), b)]
        return float(s.mean()) if s.size else float("nan")

    total_activity = float(af.sum())
    reb1 = off_s + int(200.0 / dt)
    lfp = lfp_band_summary(res.get("lfp_trace"), res.get("lfp_step_idx"), dt, window_steps=(on_s, off_s, end_s))
    summary = dict(
        arm=arm, runaway_ms=(None if censored else round(float(runaway_ms), 1)), censored=bool(censored),
        restricted_runaway_free_time_ms=round(float(rrt), 1), t_run_used_ms=round(t_run, 1),
        n_events=len(events), n_pre_stim_events=int(n_pre), n_pre_stim_recoverable=int(n_pre_recover),
        pre_rate_hz=_mrate(0, on_s), during_rate_hz=_mrate(on_s, off_s), post_rate_hz=_mrate(off_s, end_s),
        rebound_rate_200ms_hz=_mrate(off_s, reb1), total_activity_integral=total_activity,
        total_activity_ratio=(round(total_activity / baseline_total_activity, 4)
                              if baseline_total_activity else None),
        escape_prob=prop["escape_prob"], far_side_recruit=round(prop["far_side_recruit"], 4),
        max_axial_span_norm=round(prop["max_axial_span_norm"], 4),
        participation_peak=round(prop["participation_peak"], 4), beyond_reach=round(prop["beyond_reach"], 4),
        n_post_events=loc["n_post_events"], n_post_local_events=loc["n_post_local_events"],
        n_post_global_events=loc["n_post_global_events"], lfp=lfp)
    return summary


def arm_arrays(res: Mapping[str, object], obs: SpatialStreamObserver, slow) -> Dict[str, np.ndarray]:
    """Compact arrays for the per-run NPZ (never the T x NE raster)."""
    return dict(rate=_downsample(res["rate_E"]), active_frac=_downsample(obs.active_frac, 6000),
                axial_act=np.asarray(obs.axial_act, np.float32),
                trans_act=np.asarray(obs.trans_act, np.float32),
                ax_edges=np.asarray(obs.ax_edges, np.float32),
                z_mean=_downsample(slow.trace_z_mean), z_min=_downsample(slow.trace_z_min),
                m_mean=_downsample(slow.trace_m_mean), adap=_downsample(slow.trace_adap_current),
                lfp_per_site_power=np.asarray((res.get("lfp_trace") if res.get("lfp_trace") is not None
                                               else np.zeros((0, 0))), np.float32))


# ============================================================ statistics (unit = subject)
def paired_sign_flip_test(diffs: Sequence[float]) -> Dict[str, object]:
    """Exact two-sided paired sign-flip (randomization) test on per-subject differences (H0: median 0).
    Enumerates all 2^n sign assignments of the observed |diffs| (exact for small n). Reports the mean
    diff, its exact p, and the count of positive subjects."""
    d = np.asarray([x for x in diffs if np.isfinite(x)], float)
    n = len(d)
    if n == 0:
        return dict(n=0, mean=float("nan"), median=float("nan"), p_value=float("nan"), n_positive=0)
    obs = float(np.mean(d))
    mag = np.abs(d)
    count = 0
    total = 1 << n
    for mask in range(total):
        signs = np.array([1.0 if (mask >> i) & 1 else -1.0 for i in range(n)])
        if abs(float(np.mean(signs * mag))) >= abs(obs) - 1e-12:
            count += 1
    return dict(n=int(n), mean=obs, median=float(np.median(d)), p_value=count / total,
                n_positive=int((d > 0).sum()), values=d.tolist())


def wilcoxon_signed_rank(diffs: Sequence[float]) -> Dict[str, object]:
    d = np.asarray([x for x in diffs if np.isfinite(x) and x != 0.0], float)
    if len(d) < 1:
        return dict(n=0, statistic=float("nan"), p_value=float("nan"))
    try:
        from scipy.stats import wilcoxon
        st = wilcoxon(d, zero_method="wilcox", alternative="two-sided", mode="exact")
        return dict(n=int(len(d)), statistic=float(st.statistic), p_value=float(st.pvalue))
    except Exception as exc:
        return dict(n=int(len(d)), statistic=float("nan"), p_value=float("nan"), error=str(exc))


def subject_median_effects(per_seed_rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    """Collapse a subject's seed rows to per-subject median C_run / C_best (unit = subject, not seed)."""
    def _med(key):
        vals = [float(r[key]) for r in per_seed_rows if r.get(key) is not None and np.isfinite(r.get(key))]
        return float(np.median(vals)) if vals else float("nan")
    return dict(n_seeds=len(per_seed_rows), c_run=_med("c_run"), c_best=_med("c_best"),
                rrt_middle=_med("rrt_middle"), rrt_neg=_med("rrt_endpoint_negative"),
                rrt_pos=_med("rrt_endpoint_positive"))
