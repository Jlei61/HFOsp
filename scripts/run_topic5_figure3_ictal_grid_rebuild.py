#!/usr/bin/env python
"""Figure 3 ictal gradient R3 dense-grid field-concordance full recompute.

Implements docs/archive/topic5/fig3_ictal_gradient_r3_full_recompute_handoff_2026-07-18.md.

Primary estimand: R3 dense-grid support-gated maxAB field concordance.
Paired sensitivity: R2 contact-evaluated smoothed field concordance, rerun on
identical inputs in the same pass (never read from an old R2 summary).

The 167-event / 17-subject `all_phenotype_matched` parent list is the single
event universe. One coherent all-contact permutation per (subject, seizure,
draw) is reused across A/B, all seven primary bands, the BB150 parent anchor,
R2 and R3. Subject is the cohort unit; seizures are folded to a subject median
before any cohort test. No result-based drop, reorder, or method switch.

CLI:
    python scripts/run_topic5_figure3_ictal_grid_rebuild.py --validate-only
    python scripts/run_topic5_figure3_ictal_grid_rebuild.py --n-perm 20 --outdir <smoke>
    python scripts/run_topic5_figure3_ictal_grid_rebuild.py --n-perm 1000 \
        --outdir results/topic5_ictal_recruitment/field_concordance_grid_parent_matched
    python scripts/run_topic5_figure3_ictal_grid_rebuild.py --verify-only --outdir <root>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import src.topic5_gradient_grid_field as gg
from src.topic5_template_axis_field import (
    make_field_scorer, score_field_batch, scorers_from_interictal_record,
)
from src.topic5_tspectral_field_concordance import (
    apply_fixed_permutations, bootstrap_median_ci, fold_seizure_null_draws,
    make_contact_permutations, paired_sign_flip_p, phenotype_selector_sets,
)
from src.topic5_t0_features import window_activation

# ---- canonical inputs (handoff §2.4) -------------------------------------
ANALYSIS = REPO / "results/topic5_ictal_recruitment/tspectral_field_concordance"
PARENT_EVENT_CSV = ANALYSIS / "clinical_onset_gradient_field_cohort_stat_event.csv"
FIELD_ROOT = REPO / "results/interictal_propagation_masked/template_gradient_fields/per_subject"
V2_BAND_CACHE = REPO / "results/topic5_ictal_recruitment/v2_band_scan/cache"
BB150_CACHE = REPO / "results/topic5_ictal_recruitment/t0_feature_cache_bb150_1_150"
PHENO_EPI = REPO / "results/topic5_ictal_recruitment/v2_band_scan/cache_tspectral_v1p2_common_1_80hz"
PHENO_YUQ = REPO / "results/topic5_ictal_recruitment/v2_band_scan/cache_tspectral_v1p2_yuquan_common_1_80hz"
CONFIG = REPO / "config/topic5_v2_phase1.yaml"

CONTRACT = "topic5_figure3_ictal_gradient_r3_dense_grid_maxab_v1"
BASE_SEED = 20260718            # handoff §4.1 lock
N_PERM_FINAL = 1000
WINDOW = (0.0, 10.0)
MIN_CONTACTS = 6
GRID_PRIMARY = 161   # N=161 is the locked primary (81 failed the subject convergence gate)
GRID_SENS = 81
FS_EDGE_SUBJECTS = {"epilepsiae_139", "epilepsiae_253"}  # ripple_high fs_edge_flag (§2.5)

EXPECTED_SUBJECTS = 17
EXPECTED_EVENTS = 167
EXPECTED_STRICT = 106
EXPECTED_GAMMA = 61
EXPECTED_SHARED = {"epilepsiae_1084", "epilepsiae_1146", "epilepsiae_139",
                   "epilepsiae_384", "epilepsiae_548", "epilepsiae_590", "epilepsiae_958"}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def event_seed(subject: str, seizure_idx: int, base: int = BASE_SEED) -> int:
    """Deterministic per-(subject, seizure) seed, band/method-independent (§4.1)."""
    key = f"{subject}:{int(seizure_idx)}".encode("utf-8")
    return int((base ^ zlib.crc32(key)) & 0xFFFFFFFF)


def load_primary_bands() -> List[Dict[str, object]]:
    cfg = yaml.safe_load(CONFIG.read_text())
    bands = []
    for name, lo, hi in cfg["bands"]["primary"]:
        bands.append({"name": str(name), "lo": float(lo), "hi": float(hi)})
    if len(bands) != 7:
        raise SystemExit(f"expected 7 primary bands, got {len(bands)}")
    return bands


def load_parent_events() -> pd.DataFrame:
    """The 167-event / 17-subject parent list with per-event phenotype (§2.1)."""
    df = pd.read_csv(PARENT_EVENT_CSV)
    pooled = df[df.group_id == "all_phenotype_matched"][["dataset", "subject", "seizure_idx"]].drop_duplicates()
    strict = set(map(tuple, df[df.group_id == "strict_broadband"][["subject", "seizure_idx"]].values))
    gamma = set(map(tuple, df[df.group_id == "gamma_nonbroadband"][["subject", "seizure_idx"]].values))
    if strict & gamma:
        raise SystemExit(f"C1 violation: strict∩gamma non-empty ({len(strict & gamma)})")
    rows = []
    for _, r in pooled.iterrows():
        key = (r.subject, int(r.seizure_idx))
        pheno = "strict" if key in strict else "gamma" if key in gamma else None
        if pheno is None:
            raise SystemExit(f"C1 violation: event {key} has no phenotype")
        rows.append({"dataset": r.dataset, "subject": r.subject,
                     "seizure_idx": int(r.seizure_idx), "phenotype": pheno})
    out = pd.DataFrame(rows).sort_values(["subject", "seizure_idx"]).reset_index(drop=True)
    # C1 fail-closed cohort lock
    n_sub = out.subject.nunique()
    n_ev = len(out.drop_duplicates(["subject", "seizure_idx"]))
    n_strict = int((out.phenotype == "strict").sum())
    n_gamma = int((out.phenotype == "gamma").sum())
    if n_sub != EXPECTED_SUBJECTS or n_ev != EXPECTED_EVENTS:
        raise SystemExit(f"C1: expected {EXPECTED_SUBJECTS}/{EXPECTED_EVENTS}, got {n_sub}/{n_ev}")
    if n_strict != EXPECTED_STRICT or n_gamma != EXPECTED_GAMMA:
        raise SystemExit(f"C1: expected strict {EXPECTED_STRICT}/gamma {EXPECTED_GAMMA}, "
                         f"got {n_strict}/{n_gamma}")
    return out


class SubjectField:
    """Frozen interictal geometry for one subject (routing + sigma_common + planes)."""

    def __init__(self, subject: str, axis: str = "gradient"):
        self.subject = subject
        path = FIELD_ROOT / f"{subject}.json"
        if not path.exists():
            raise SystemExit(f"missing frozen field: {path}")
        self.record = json.loads(path.read_text())
        # C19 fingerprint fail-closed (raises on mismatch) — same frozen source for both axes
        self.scorers = scorers_from_interictal_record(self.record)
        field = self.record["interictal_field"]
        planes = field["planes"]
        self.contact_order = [str(x) for x in field["contact_order"]]
        self.support_a = np.asarray(field["support_a"], float)
        self.support_b = np.asarray(field["support_b"], float)
        self.earliness_a = np.asarray(field["earliness_a"], float)
        self.earliness_b = np.asarray(field["earliness_b"], float)
        self.axis = axis
        if axis == "endpoint":
            # AXIS-ONLY: same frozen support/earliness/coords; only the projection plane
            # changes (endpoint source->sink cores instead of the gradient axis). Per-template
            # A/B for all subjects; fail closed on degenerate cores (NO gradient fallback).
            coords = np.asarray(field["coords"], float)
            ep_a = gg.build_endpoint_plane(coords, np.asarray(field["rank_a"], float), k_primary=3)
            ep_b = gg.build_endpoint_plane(coords, np.asarray(field["rank_b"], float), k_primary=3)
            if ep_a is None or ep_b is None:
                raise SystemExit(f"endpoint axis degenerate for {subject} "
                                 f"(fail closed; no gradient fallback)")
            self.route = "endpoint"
            self.pts_a = ep_a["points"]
            self.pts_b = ep_b["points"]
            self.sigma_common = float(ep_a["sigma"])   # subject_fixed: endpoint-A sigma for A & B
            self.sigma_own_b = float(ep_b["sigma"])
            self.endpoint_cores = {"a": {"tier": ep_a["tier"], "source_idx": ep_a["source_idx"],
                                         "sink_idx": ep_a["sink_idx"]},
                                   "b": {"tier": ep_b["tier"], "source_idx": ep_b["source_idx"],
                                         "sink_idx": ep_b["sink_idx"]}}
        # C3 outcome-independent gradient routing
        elif "shared_a" in self.scorers and "shared_b" in self.scorers:
            self.route = "shared"
            self.pts_a = np.asarray(planes["shared"]["points"], float)
            self.pts_b = self.pts_a
            self.sigma_common = float(planes["shared"]["sigma"])   # C4
            self.sigma_own_b = float(planes["shared"]["sigma"])    # shared: B uses shared sigma in both policies
        elif "own_a" in self.scorers and "own_b" in self.scorers:
            self.route = "own_fallback"
            self.pts_a = np.asarray(planes["own_a"]["points"], float)
            self.pts_b = np.asarray(planes["own_b"]["points"], float)
            self.sigma_common = float(planes["own_a"]["sigma"])    # own_a sigma (subject-fixed A & B)
            self.sigma_own_b = float(planes["own_b"]["sigma"])     # own_b frozen sigma (frozen_per_model B)
        else:
            raise SystemExit(f"C3: {subject} neither complete shared nor own field pair")
        self.fingerprint = field.get("fingerprint_sha256")

    def sigmas(self, policy: str):
        """(sigma_a, sigma_b) for a smoothing policy.

        subject_fixed: sigma_a = sigma_b = sigma_common (shared or own_a plane sigma).
        frozen_per_model: shared -> both = shared sigma; own -> A = own_a, B = own_b.
        """
        if policy == "subject_fixed":
            return self.sigma_common, self.sigma_common
        if policy == "frozen_per_model":
            return self.sigma_common, self.sigma_own_b
        raise SystemExit(f"unknown smoothing policy: {policy}")

    def build_event_scorers(self, finite: np.ndarray, n: int,
                            policy: str = "subject_fixed") -> Dict[str, object]:
        sigma_a, sigma_b = self.sigmas(policy)
        return gg.build_event_scorer(
            pts_a=self.pts_a, support_a=self.support_a, earliness_a=self.earliness_a,
            pts_b=self.pts_b, support_b=self.support_b, earliness_b=self.earliness_b,
            sigma_a=sigma_a, sigma_b=sigma_b, finite=finite, n=n,
            shared_grid=(self.route == "shared"),
            model_a=f"{self.subject}_{self.route}_A", model_b=f"{self.subject}_{self.route}_B")

    def build_r2_scorers(self, policy: str = "subject_fixed"):
        """R2 contact-evaluated scorers at the same per-template sigma as the R3 cell (§3.6)."""
        sigma_a, sigma_b = self.sigmas(policy)
        sa = make_field_scorer(self.earliness_a, self.pts_a, self.support_a, sigma_a)
        sb = make_field_scorer(self.earliness_b, self.pts_b, self.support_b, sigma_b)
        return sa, sb


class ActivationCache:
    """Per-subject ictal activation: 7 primary bands + BB150 anchor."""

    def __init__(self, subject: str, bands: List[Dict[str, object]]):
        self.subject = subject
        self.bands = bands
        v2 = np.load(V2_BAND_CACHE / f"{subject}.npz", allow_pickle=True)
        self.v2 = {k: v2[k] for k in v2.files}
        self.v2_channels = [str(x) for x in self.v2["channels"]]
        bb = np.load(BB150_CACHE / f"{subject}.npz", allow_pickle=True)
        self.bb = {k: bb[k] for k in bb.files}
        self.bb_channels = [str(x) for x in self.bb["channels"]]

    def band_activation(self, band: str, idx: int) -> Optional[np.ndarray]:
        zt = self.v2.get(f"{band}__zt__{idx}")
        relt = self.v2.get(f"{band}__relt__{idx}")
        if zt is None or relt is None:
            return None
        return window_activation(np.asarray(zt, float), np.asarray(relt, float), *WINDOW)

    def anchor_activation(self, idx: int) -> Optional[np.ndarray]:
        v = self.bb.get(f"bb150_auc__{idx}")
        return None if v is None else np.asarray(v, float)


def align_by_name(source_names: Sequence[str], values: np.ndarray, target: Sequence[str]) -> np.ndarray:
    idx = {str(n): i for i, n in enumerate(source_names)}
    out = np.full(len(target), np.nan)
    for i, name in enumerate(target):
        j = idx.get(str(name))
        if j is not None:
            out[i] = values[j]
    return out


def median_event(observed: Sequence[float]) -> float:
    v = np.asarray([x for x in observed if np.isfinite(x)], float)
    return float(np.median(v)) if v.size else float("nan")


def fold_null(null_by_event: List[np.ndarray]) -> np.ndarray:
    """Event→subject median per draw (§4.3) via the canonical helper (no pooling)."""
    shaped = [np.asarray(a, float).reshape(-1, 1) for a in null_by_event]
    return fold_seizure_null_draws(shaped)[:, 0]


# --------------------------------------------------------------------------
# per-event scoring (C2, C7, C8, C9, C11)
# --------------------------------------------------------------------------
def build_common_mask(sf: SubjectField, ac: ActivationCache, bands, idx: int):
    """Strict common mask: contact_order ∩ caches ∩ finite in all 7 bands ∩ anchor (§2.6)."""
    order = sf.contact_order
    in_v2 = np.array([n in ac.v2_channels for n in order])
    in_bb = np.array([n in ac.bb_channels for n in order])
    finite = in_v2 & in_bb
    band_acts = {}
    for b in bands:
        raw = ac.band_activation(b["name"], idx)
        if raw is None:
            return None, None, None, f"missing_band:{b['name']}"
        aligned = align_by_name(ac.v2_channels, raw, order)
        band_acts[b["name"]] = aligned
        finite &= np.isfinite(aligned)
    anchor_raw = ac.anchor_activation(idx)
    if anchor_raw is None:
        return None, None, None, "missing_anchor_bb150"
    anchor = align_by_name(ac.bb_channels, anchor_raw, order)
    band_acts["bb150_anchor"] = anchor
    finite &= np.isfinite(anchor)
    return finite, band_acts, anchor, None


def score_event(sf: SubjectField, finite: np.ndarray, activations: Dict[str, np.ndarray],
                perms_all: np.ndarray, ws_perms: Optional[np.ndarray], grids: Sequence[int],
                policy: str = "subject_fixed"):
    """Return per-activation observed + null maxAB for R3 (each grid) and R2.

    `activations` keys -> full-length (n_contact,) activation aligned to
    contact_order (NaN outside common mask). `perms_all` is the ONE coherent
    all-contact permutation reused across every activation/method (C11).
    """
    n_perm = perms_all.shape[0]
    # masked activations: NaN outside the common finite mask so every method
    # (R2, R3@81, R3@161) sees exactly the same available contacts (C2, C7).
    masked = {k: np.where(finite, v, np.nan) for k, v in activations.items()}
    # activation matrices: row 0 observed, rows 1.. permuted (same perms) (C11)
    act_mats = {}
    for k, v in masked.items():
        permuted = apply_fixed_permutations(v[None, :], perms_all)[:, 0, :]  # (n_perm, n_contact)
        act_mats[k] = np.vstack([v[None, :], permuted])                      # (1+n_perm, n_contact)
    ws_mats = None
    if ws_perms is not None:
        ws_mats = {}
        for k, v in masked.items():
            permuted = apply_fixed_permutations(v[None, :], ws_perms)[:, 0, :]
            ws_mats[k] = np.vstack([v[None, :], permuted])

    out: Dict[str, object] = {"observed": {}, "null": {}, "detail": {}, "within_shaft_null": {}}
    # R3 at each resolution
    evs = {n: sf.build_event_scorers(finite, n, policy) for n in grids}
    # grid boundary assertion (C6): S>=0.15 region must not touch the edge
    for n, ev in evs.items():
        for tag in ("A", "B"):
            S = ev[tag].S_inter.reshape(ev[f"grid_{tag.lower()}"]["X"].shape)
            if gg.support_region_touches_boundary(S):
                raise SystemExit(f"C6: support region touches grid boundary "
                                 f"({sf.subject} grid {n} template {tag})")
    for k, mat in act_mats.items():
        for n, ev in evs.items():
            vals = gg.score_event_maxab_batch(ev, mat)   # (1+n_perm,)
            out["observed"][(f"R3_{n}", k)] = float(vals[0])
            out["null"][(f"R3_{n}", k)] = vals[1:]
            if n == grids[0]:
                out["detail"][(f"R3_{n}", k)] = gg.score_event_detail_single(ev, mat[0])
        if ws_mats is not None:
            for n in (grids[0],):  # within-shaft only at primary resolution (§4.2 secondary)
                vals = gg.score_event_maxab_batch(evs[n], ws_mats[k])
                out["within_shaft_null"][(f"R3_{n}", k)] = vals[1:]
    # R2 contact-evaluated (C9): same masked activation, same perms, same per-template sigma as R3 cell
    sa, sb = sf.build_r2_scorers(policy)
    for k, mat in act_mats.items():
        ra = score_field_batch(sa, mat)["abs_r"]
        rb = score_field_batch(sb, mat)["abs_r"]
        stack = np.vstack([ra, rb])
        with np.errstate(invalid="ignore"):
            maxab = np.nanmax(stack, axis=0)
        allnan = ~np.isfinite(ra) & ~np.isfinite(rb)
        maxab[allnan] = np.nan
        out["observed"][("R2", k)] = float(maxab[0])
        out["null"][("R2", k)] = maxab[1:]
    return out


# --------------------------------------------------------------------------
# cohort assembly
# --------------------------------------------------------------------------
def cohort_spatial_null_p(D, null_draws):
    """Coherent cohort spatial-null permutation p (handoff §5.2/§5.3).

    ``D`` is the (n_subject,) observed subject statistic; ``null_draws`` is the
    (n_subject, n_perm) subject-level null (event->subject median per draw, using
    the SAME coherent all-contact permutation). The cohort statistic is the
    across-subject median; the p is the fraction of draws whose cohort median
    reaches the observed cohort median. This is the pre-registered coherent
    cohort permutation p, NOT the subject-vs-own-null Wilcoxon.
    """
    D = np.asarray(D, float)
    M = np.asarray(null_draws, float)
    if M.ndim != 2 or M.shape[0] != D.shape[0]:
        return float("nan")
    ok = np.isfinite(D)
    if int(ok.sum()) < 1:
        return float("nan")
    cobs = float(np.nanmedian(D[ok]))
    cnull = np.nanmedian(M[ok, :], axis=0)              # (n_perm,)
    K = cnull.size
    return float((1 + int(np.sum(cnull >= cobs - 1e-15))) / (K + 1))


def cohort_from_margins(data_vals, null_meds, margins, null_draws=None):
    data = np.asarray(data_vals, float)
    null = np.asarray(null_meds, float)
    marg = np.asarray(margins, float)
    ok = np.isfinite(data) & np.isfinite(null)
    data, null, marg = data[ok], null[ok], marg[ok]
    spatial_p = float("nan")
    if null_draws is not None:
        spatial_p = cohort_spatial_null_p(np.asarray(data_vals, float)[ok],
                                          np.asarray(null_draws, float)[ok, :])
    return {
        "coherent_cohort_spatial_null_p": spatial_p,
        "n_subjects": int(data.size),
        "data_median": float(np.median(data)) if data.size else float("nan"),
        "data_iqr_low": float(np.percentile(data, 25)) if data.size else float("nan"),
        "data_iqr_high": float(np.percentile(data, 75)) if data.size else float("nan"),
        "null_median": float(np.median(null)) if null.size else float("nan"),
        "null_iqr_low": float(np.percentile(null, 25)) if null.size else float("nan"),
        "null_iqr_high": float(np.percentile(null, 75)) if null.size else float("nan"),
        "margin_median": float(np.median(marg)) if marg.size else float("nan"),
        "margin_iqr_low": float(np.percentile(marg, 25)) if marg.size else float("nan"),
        "margin_iqr_high": float(np.percentile(marg, 75)) if marg.size else float("nan"),
        "n_data_gt_null": int(np.sum(data > null)),
        "wilcoxon_one_sided_data_gt_null_p": gg.paired_one_sided_wilcoxon_greater(data, null),
        "two_sided_subject_sign_flip_p": float(paired_sign_flip_p(marg, n_perm=100000, seed=BASE_SEED)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=N_PERM_FINAL)
    ap.add_argument("--seed", type=int, default=BASE_SEED)
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--grids", type=str, default=f"{GRID_PRIMARY},{GRID_SENS}")
    ap.add_argument("--band-omnibus-perm", type=int, default=100000)
    ap.add_argument("--smoothing-policy", choices=["subject_fixed", "frozen_per_model"],
                    default="subject_fixed")
    ap.add_argument("--axis", choices=["gradient", "endpoint"], default="gradient",
                    help="gradient = frozen shared-else-own planes (primary); "
                         "endpoint = source->sink core axis, per-template A/B (axis-only sensitivity)")
    ap.add_argument("--score-bands-only", action="store_true",
                    help="keep the BB150 anchor in the common mask (mask unchanged) but score only "
                         "the 7 primary bands; skip anchor scoring and parent pooled/strict/gamma groups")
    args = ap.parse_args()

    grids = [int(x) for x in args.grids.split(",")]
    bands = load_primary_bands()
    events = load_parent_events()
    print(f"[lock] {events.subject.nunique()} subjects / {len(events)} events "
          f"(strict={int((events.phenotype=='strict').sum())}, "
          f"gamma={int((events.phenotype=='gamma').sum())}) axis={args.axis}", flush=True)

    # routing + sigma inventory (validate-only stops here after mask check)
    subjects = list(dict.fromkeys(events.subject.tolist()))
    fields = {s: SubjectField(s, axis=args.axis) for s in subjects}
    if args.axis == "endpoint":
        assert all(fields[s].route == "endpoint" for s in subjects)
        print(f"[route] endpoint per-template A/B for all {len(subjects)} subjects "
              f"(axis-only; vs gradient shared-else-own)", flush=True)
    else:
        shared = {s for s in subjects if fields[s].route == "shared"}
        if shared != EXPECTED_SHARED:
            raise SystemExit(f"C3: shared routing {sorted(shared)} != expected {sorted(EXPECTED_SHARED)}")
        print(f"[route] shared={len(shared)} own={len(subjects)-len(shared)}", flush=True)

    if args.outdir is None and not args.validate_only:
        raise SystemExit("--outdir required unless --validate-only")
    outdir = Path(args.outdir) if args.outdir else None

    input_paths = [PARENT_EVENT_CSV, CONFIG] + \
        [FIELD_ROOT / f"{s}.json" for s in subjects] + \
        [V2_BAND_CACHE / f"{s}.npz" for s in subjects] + \
        [BB150_CACHE / f"{s}.npz" for s in subjects]
    hashes_before = {str(p.relative_to(REPO)): sha256_file(p) for p in input_paths}

    # ---- validation pass (common mask + fail-closed) ---------------------
    caches = {s: ActivationCache(s, bands) for s in subjects}
    common_rows = []
    per_event = {}   # (subject, idx) -> dict
    for _, ev in events.iterrows():
        s, idx = ev.subject, int(ev.seizure_idx)
        finite, band_acts, anchor, drop = build_common_mask(fields[s], caches[s], bands, idx)
        if drop is not None:
            raise SystemExit(f"STOP (§9): {s} seizure {idx} {drop}")
        n_common = int(finite.sum())
        if n_common < MIN_CONTACTS:
            raise SystemExit(f"STOP (§9): {s} seizure {idx} common contacts {n_common} < {MIN_CONTACTS}")
        common_rows.append({"subject": s, "seizure_idx": idx, "phenotype": ev.phenotype,
                            "route": fields[s].route, "sigma_common": fields[s].sigma_common,
                            "n_common_contacts": n_common, "n_contact_order": len(fields[s].contact_order)})
        per_event[(s, idx)] = {"finite": finite, "activations": {b["name"]: band_acts[b["name"]] for b in bands},
                               "anchor": anchor, "phenotype": ev.phenotype}
    common_df = pd.DataFrame(common_rows)
    ncc = common_df.n_common_contacts
    print(f"[mask] common contacts min={ncc.min()} median={int(ncc.median())} max={ncc.max()} "
          f"(all events present in 7 bands + anchor: {len(common_df)}/{EXPECTED_EVENTS})", flush=True)
    if len(common_df) != EXPECTED_EVENTS:
        raise SystemExit(f"C2: {len(common_df)} events with valid mask != {EXPECTED_EVENTS}")

    if args.validate_only:
        print("[validate-only] cohort + routing + common-mask contract verified.")
        print(common_df.groupby("route").n_common_contacts.describe()[["min", "50%", "max"]])
        return

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "per_subject").mkdir(exist_ok=True)
    n_perm = int(args.n_perm)
    print(f"[run] n_perm={n_perm} seed={args.seed} grids={grids} outdir={outdir}", flush=True)

    # ---- score every event ----------------------------------------------
    band_names = [b["name"] for b in bands]
    all_keys = band_names + ["bb150_anchor"]
    perm_audit = []
    scored = {}   # (subject, idx) -> score_event output
    ws_meta = {}  # (subject, idx) -> within-shaft eligibility
    t0 = time.time()
    for i, ((s, idx), info) in enumerate(per_event.items()):
        finite = info["finite"]
        acts = dict(info["activations"])
        if not args.score_bands_only:      # anchor stays IN the mask, only its scoring is skipped
            acts["bb150_anchor"] = info["anchor"]
        seed = event_seed(s, idx, args.seed)
        perms_all = make_contact_permutations(fields[s].contact_order, finite, n_perm, seed, mode="all_contact")
        ws = gg.within_shaft_permutations(fields[s].contact_order, finite, n_perm=n_perm, seed=seed, min_group=4)
        ws_perms = ws["permutations"] if ws["eligible"] else None
        ws_meta[(s, idx)] = ws
        perm_audit.append({"subject": s, "seizure_idx": idx, "seed": seed,
                           "n_perm": n_perm, "mapping_sha256": gg.permutation_mapping_hash(perms_all),
                           "within_shaft_eligible": bool(ws["eligible"])})
        scored[(s, idx)] = score_event(fields[s], finite, acts, perms_all, ws_perms, grids,
                                       policy=args.smoothing_policy)
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  scored {i+1}/{len(per_event)} events  ({time.time()-t0:.0f}s)", flush=True)
    print(f"[score] done {len(scored)} events in {time.time()-t0:.0f}s", flush=True)

    _aggregate_and_write(events, fields, common_df, per_event, scored, ws_meta,
                         perm_audit, bands, band_names, grids, n_perm, args,
                         hashes_before, input_paths, outdir)


def _subject_fold(scored, events, subjects, method, key):
    """Return per-subject D, Nmed, margin, and the (subject, n_perm) null-draw matrix."""
    D, Nmed, margin, subj_order, null_draw_rows = [], [], [], [], []
    for s in subjects:
        idxs = events[events.subject == s].seizure_idx.astype(int).tolist()
        obs = [scored[(s, i)]["observed"].get((method, key), np.nan) for i in idxs]
        nulls = [scored[(s, i)]["null"].get((method, key)) for i in idxs]
        nulls = [n for n in nulls if n is not None]
        d = median_event(obs)
        folded = fold_null(nulls) if nulls else np.array([np.nan])
        nm = float(np.median(folded))
        D.append(d); Nmed.append(nm); margin.append(d - nm); subj_order.append(s)
        null_draw_rows.append(folded)
    return (np.array(D), np.array(Nmed), np.array(margin), subj_order,
            np.array(null_draw_rows))


def _aggregate_and_write(events, fields, common_df, per_event, scored, ws_meta,
                         perm_audit, bands, band_names, grids, n_perm, args,
                         hashes_before, input_paths, outdir):
    subjects = list(dict.fromkeys(events.subject.tolist()))
    prim = f"R3_{grids[0]}"
    sens = f"R3_{grids[1]}" if len(grids) > 1 else None

    # ---- C10 observed primary/sensitivity convergence (event×band) -------
    # Column names track the ACTUAL grids (grids[0]=primary, grids[1]=sensitivity)
    # so they never flip when the grid order is changed on the CLI.
    prim_n, sens_n = grids[0], (grids[1] if len(grids) > 1 else None)
    r_prim_col, r_sens_col = f"r{prim_n}", f"r{sens_n}"
    conv_rows = []
    if sens is not None:
        for (s, idx), sc in scored.items():
            for key in band_names:
                r_prim = sc["observed"].get((prim, key), np.nan)
                r_sens = sc["observed"].get((sens, key), np.nan)
                if np.isfinite(r_prim) and np.isfinite(r_sens):
                    conv_rows.append({"subject": s, "seizure_idx": idx, "band": key,
                                      r_prim_col: r_prim, r_sens_col: r_sens,
                                      "abs_diff": abs(r_prim - r_sens)})
    conv_df = pd.DataFrame(conv_rows)
    conv_p95 = float(np.percentile(conv_df.abs_diff, 95)) if len(conv_df) else float("nan")

    # ---- seven-band inheritance + maxT (C15) -----------------------------
    D_mat = np.full((len(subjects), len(band_names)), np.nan)
    N_tensor = np.full((len(subjects), len(band_names), n_perm), np.nan)
    D_mat_r2 = np.full((len(subjects), len(band_names)), np.nan)     # R2 cell (§六)
    N_tensor_r2 = np.full((len(subjects), len(band_names), n_perm), np.nan)
    band_subject_rows, band_null_conv = [], []
    for bi, key in enumerate(band_names):
        D, Nmed, margin, order, null_draws = _subject_fold(scored, events, subjects, prim, key)
        D_mat[:, bi] = D
        N_tensor[:, bi, :] = null_draws
        D2, _, _, _, nd2 = _subject_fold(scored, events, subjects, "R2", key)
        D_mat_r2[:, bi] = D2
        N_tensor_r2[:, bi, :] = nd2
        for si, s in enumerate(order):
            band_subject_rows.append({"band": key, "subject": s, "route": fields[s].route,
                                      "D": D[si], "Nmed": Nmed[si], "delta": margin[si]})
        if sens is not None:
            Ds, Nmeds, _, _, _ = _subject_fold(scored, events, subjects, sens, key)
            band_null_conv.append({"band": key,
                                   "max_subject_data_diff": float(np.nanmax(np.abs(D - Ds))),
                                   "max_subject_null_diff": float(np.nanmax(np.abs(Nmed - Nmeds)))})
    maxt = gg.seven_band_maxt_pfwer(D_mat, N_tensor)
    band_cohort_rows = []
    for bi, key in enumerate(band_names):
        col_D = D_mat[:, bi]
        col_N = N_tensor[:, bi, :]
        margins = maxt["per_subject_delta"][:, bi]
        band_cohort_rows.append({
            "band": key, "n_subjects": int(np.isfinite(col_D).sum()),
            "data_median": float(np.nanmedian(col_D)),
            "null_median": float(maxt["Cnull_median"][bi]),
            "delta_cohort_median": float(maxt["cohort_delta_median"][bi]),
            "delta_iqr_low": float(np.nanpercentile(margins, 25)),
            "delta_iqr_high": float(np.nanpercentile(margins, 75)),
            "n_positive": int(maxt["n_positive"][bi]),
            "wilcoxon_one_sided_p": gg.paired_one_sided_wilcoxon_greater(col_D, np.nanmedian(col_N, axis=1)),
            "coherent_cohort_spatial_null_p": cohort_spatial_null_p(col_D, col_N),
            "seven_band_maxt_pfwer": float(maxt["pFWER"][bi]),
        })

    # ---- direct band specificity (C16) -----------------------------------
    delta_matrix = maxt["per_subject_delta"]
    omnibus = gg.direct_band_omnibus(delta_matrix, n_perm=args.band_omnibus_perm, seed=args.seed)
    contrasts = gg.direct_band_contrasts(delta_matrix, band_labels=band_names)

    sbo = getattr(args, "score_bands_only", False)   # seven-band-only (endpoint axis-only run)

    # ---- parent cohort: pooled / broadband / gamma (C14) -----------------
    def phenotype_activation_key(s, idx):
        return "bb150_anchor" if per_event[(s, idx)]["phenotype"] == "strict" else "gamma_LVFA"

    def parent_group(method, selector):
        data, null, marg, order, draws = [], [], [], [], []
        for s in subjects:
            idxs = [i for i in events[events.subject == s].seizure_idx.astype(int)
                    if selector(s, i)]
            if not idxs:
                continue
            obs = [scored[(s, i)]["observed"].get((method, phenotype_activation_key(s, i)), np.nan) for i in idxs]
            nulls = [scored[(s, i)]["null"].get((method, phenotype_activation_key(s, i))) for i in idxs]
            nulls = [n for n in nulls if n is not None]
            d = median_event(obs)
            folded = fold_null(nulls) if nulls else np.full(n_perm, np.nan)
            nm = float(np.median(folded))
            data.append(d); null.append(nm); marg.append(d - nm); order.append(s); draws.append(folded)
        return data, null, marg, order, np.asarray(draws, float)

    selectors = {
        "all_phenotype_matched": lambda s, i: True,
        "strict_broadband": lambda s, i: per_event[(s, i)]["phenotype"] == "strict",
        "gamma_nonbroadband": lambda s, i: per_event[(s, i)]["phenotype"] == "gamma",
    }
    parent_cohort_rows, parent_subject_rows = [], []
    parent_null_draws = {}   # (method, group) -> (subject, n_perm)
    if not sbo:   # parent groups use the BB150 anchor, which is not scored in bands-only runs
        for method in ("R3_%d" % grids[0], "R2"):
            for gid, sel in selectors.items():
                data, null, marg, order, draws = parent_group(method, sel)
                stat = cohort_from_margins(data, null, marg, null_draws=draws)
                stat.update({"group_id": gid, "method": method,
                             "n_seizures": int(sum(sel(s, int(i)) for s, i in
                                                   zip(events.subject, events.seizure_idx)))})
                parent_cohort_rows.append(stat)
                parent_null_draws[f"{method}__{gid}"] = draws
                for s, d, nm, mg in zip(order, data, null, marg):
                    parent_subject_rows.append({"group_id": gid, "method": method, "subject": s,
                                                "data": d, "null_median": nm, "margin": mg})

    # ---- R2 vs R3 paired diagnostic (C9) ---------------------------------
    r2r3_rows = []
    for s in subjects:
        idxs = events[events.subject == s].seizure_idx.astype(int).tolist()
        for key in band_names + ([] if sbo else ["bb150_anchor"]):
            r3 = median_event([scored[(s, i)]["observed"].get((prim, key), np.nan) for i in idxs])
            r2 = median_event([scored[(s, i)]["observed"].get(("R2", key), np.nan) for i in idxs])
            r2r3_rows.append({"subject": s, "band": key, "r3_data": r3, "r2_data": r2,
                              "r3_minus_r2": r3 - r2})

    # ---- pure within-shaft secondary (C12, min_group=4). PHENOTYPE-MATCHED
    # activation (strict->BB150, gamma->gamma30), same as the parent cohort — NOT
    # BB150-uniform (F3a fix). Separate eligible denominator. Skipped in bands-only
    # runs (its strict events read the unscored BB150 anchor).
    ws_eligible_events = [] if sbo else [(s, i) for (s, i), m in ws_meta.items() if m["eligible"]]
    ws_subjects = sorted({s for s, _ in ws_eligible_events})
    ws_subject_rows, ws_cohort_data, ws_cohort_null, ws_cohort_margin, ws_draws = [], [], [], [], []
    for s in ws_subjects:
        idxs = [i for (ss, i) in ws_eligible_events if ss == s]
        obs = [scored[(s, i)]["observed"].get((prim, phenotype_activation_key(s, i)), np.nan) for i in idxs]
        nulls = [scored[(s, i)]["within_shaft_null"].get((prim, phenotype_activation_key(s, i))) for i in idxs]
        nulls = [n for n in nulls if n is not None]
        if not nulls:
            continue
        d = median_event(obs)
        folded = fold_null(nulls)
        nm = float(np.median(folded))
        ws_subject_rows.append({"subject": s, "n_eligible_seizures": len(idxs),
                                "data": d, "within_shaft_null_median": nm, "margin": d - nm})
        ws_cohort_data.append(d); ws_cohort_null.append(nm); ws_cohort_margin.append(d - nm)
        ws_draws.append(folded)
    ws_cohort = cohort_from_margins(ws_cohort_data, ws_cohort_null, ws_cohort_margin,
                                    null_draws=np.asarray(ws_draws, float) if ws_draws else None)
    ws_cohort.update({"eligible_subjects": len(ws_subjects), "eligible_events": len(ws_eligible_events),
                      "activation": "phenotype_matched (strict->BB150, gamma->gamma30)"})

    # per-band within-shaft (secondary anatomical sensitivity, §4.2). Observed is
    # the same D as all-contact; only the null model differs. Separate denominator.
    ws_band_subject_rows, ws_band_cohort_rows = [], []
    for key in band_names:
        dvals, nvals, mvals = [], [], []
        for s in ws_subjects:
            idxs = [i for (ss, i) in ws_eligible_events if ss == s]
            obs = [scored[(s, i)]["observed"].get((prim, key), np.nan) for i in idxs]
            nulls = [scored[(s, i)]["within_shaft_null"].get((prim, key)) for i in idxs]
            nulls = [n for n in nulls if n is not None]
            if not nulls:
                continue
            d = median_event(obs)
            nm = float(np.median(fold_null(nulls)))
            dvals.append(d); nvals.append(nm); mvals.append(d - nm)
            ws_band_subject_rows.append({"band": key, "subject": s, "D": d,
                                         "within_shaft_null_median": nm, "delta": d - nm})
        ws_band_cohort_rows.append({
            "band": key, "n_subjects": len(dvals),
            "delta_cohort_median": float(np.median(mvals)) if mvals else float("nan"),
            "n_positive": int(sum(1 for m in mvals if m > 0)),
            "wilcoxon_one_sided_p": gg.paired_one_sided_wilcoxon_greater(dvals, nvals)
            if len(dvals) >= 1 else float("nan")})

    # ---- fs-edge sensitivity (C17): ripple_high with/without E139,E253 ---
    fs_rows = []
    rh = "ripple_high"
    if rh in band_names:
        bi = band_names.index(rh)
        keep_full = np.array([True] * len(subjects))
        excl = np.array([s not in FS_EDGE_SUBJECTS for s in subjects])
        for label, mask in (("primary_keeps_E139_E253", keep_full),
                            ("exclude_E139_E253", excl)):
            col_D = D_mat[mask, bi]
            col_N = np.nanmedian(N_tensor[mask, bi, :], axis=1)
            marg = maxt["per_subject_delta"][mask, bi]
            fs_rows.append({"sensitivity": label, "band": rh,
                            "n_subjects": int(np.isfinite(col_D).sum()),
                            "delta_median": float(np.nanmedian(marg)),
                            "wilcoxon_one_sided_p": gg.paired_one_sided_wilcoxon_greater(col_D, col_N)})

    # ==== write artifacts =================================================
    def W(name, df):
        Path(outdir / name).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(outdir / name, index=False)

    # support/overlap + observed detail inventory (§7)
    overlap_rows = []
    for (s, idx), sc in scored.items():
        for key in band_names + ["bb150_anchor"]:
            det = sc["detail"].get((prim, key))
            if det is None:
                continue
            overlap_rows.append({
                "subject": s, "seizure_idx": idx, "band": key,
                "abs_a": det["abs_a"], "abs_b": det["abs_b"], "maxab": det["maxab"],
                "best_template": det["best_template"],
                "mirror_a": det["mirror_a"], "mirror_b": det["mirror_b"],
                "overlap_a": det["overlap_a"], "overlap_b": det["overlap_b"]})
    ws_event_rows = [{"subject": a["subject"], "seizure_idx": a["seizure_idx"],
                      "within_shaft_eligible": a["within_shaft_eligible"]} for a in perm_audit]
    # parent-anchor event-level (phenotype activation) + R2-sensitivity event-level (F5)
    parent_event_rows, r2_event_rows = [], []
    for (s, idx) in per_event:
        pheno = per_event[(s, idx)]["phenotype"]
        pkey = "bb150_anchor" if pheno == "strict" else "gamma_LVFA"
        if not sbo:
            r3n = scored[(s, idx)]["null"].get((prim, pkey))
            parent_event_rows.append({
                "subject": s, "seizure_idx": idx, "phenotype": pheno, "activation_key": pkey,
                "r3_observed": scored[(s, idx)]["observed"].get((prim, pkey), np.nan),
                "r2_observed": scored[(s, idx)]["observed"].get(("R2", pkey), np.nan),
                "r3_null_median": float(np.median(r3n)) if r3n is not None else np.nan})
        for key in band_names + ([] if sbo else ["bb150_anchor"]):
            r2n = scored[(s, idx)]["null"].get(("R2", key))
            r2_event_rows.append({
                "subject": s, "seizure_idx": idx, "band": key,
                "r2_observed": scored[(s, idx)]["observed"].get(("R2", key), np.nan),
                "r2_null_median": float(np.median(r2n)) if r2n is not None else np.nan})
    # routing + sigma policy + full grid bounds/spacing/hash inventory (F5)
    routing_rows = []
    for s in subjects:
        sf = fields[s]
        sa, sb = sf.sigmas(args.smoothing_policy)
        gev = sf.build_event_scorers(np.ones(len(sf.contact_order), bool), grids[0], args.smoothing_policy)
        ga, gb = gev["grid_a"], gev["grid_b"]
        routing_rows.append({
            "subject": s, "axis": args.axis, "route": sf.route, "smoothing_policy": args.smoothing_policy,
            "sigma_a": sa, "sigma_b": sb, "sigma_common": sf.sigma_common, "sigma_own_b": sf.sigma_own_b,
            "n_contact_order": len(sf.contact_order), "fingerprint_sha256": sf.fingerprint,
            "grid_n": ga["n"], "grid_a_x_lo": ga["x_lo"], "grid_a_x_hi": ga["x_hi"],
            "grid_a_y_ext": ga["y_ext"], "grid_a_spacing_x": ga["spacing_x"],
            "grid_a_spacing_y": ga["spacing_y"], "grid_a_support_budget": ga["support_budget"],
            "grid_a_sha256": ga["sha256"], "grid_b_sha256": gb["sha256"]})

    W("cohort_event_inventory.csv", events)
    W("common_contact_inventory.csv", common_df)
    W("support_overlap_inventory.csv", pd.DataFrame(overlap_rows))
    W("within_shaft_event_inventory.csv", pd.DataFrame(ws_event_rows))
    W("parent_anchor_event.csv", pd.DataFrame(parent_event_rows))
    W("r2_sensitivity_event.csv", pd.DataFrame(r2_event_rows))
    W("drop_inventory.csv", pd.DataFrame(columns=["subject", "seizure_idx", "band", "reason"]))
    W("field_routing_sigma_grid_inventory.csv", pd.DataFrame(routing_rows))
    W("permutation_mapping_audit_summary.csv", pd.DataFrame(perm_audit))
    W("parent_anchor_cohort.csv", pd.DataFrame(parent_cohort_rows))
    W("parent_anchor_subject.csv", pd.DataFrame(parent_subject_rows))
    W("multiband_subject.csv", pd.DataFrame(band_subject_rows))
    W("multiband_cohort.csv", pd.DataFrame(band_cohort_rows))
    W("multiband_band_contrasts.csv", pd.DataFrame(contrasts))
    W("r2_r3_subject_comparison.csv", pd.DataFrame(r2r3_rows))
    W("within_shaft_subject.csv", pd.DataFrame(ws_subject_rows))
    W("within_shaft_cohort.csv", pd.DataFrame([ws_cohort]))
    W("within_shaft_multiband_subject.csv", pd.DataFrame(ws_band_subject_rows))
    W("within_shaft_multiband_cohort.csv", pd.DataFrame(ws_band_cohort_rows))
    W("fs_edge_sensitivity.csv", pd.DataFrame(fs_rows))
    if len(conv_df):
        W("r2_r3_grid_convergence.csv", conv_df)
    W("r2_sensitivity_cohort.csv",
      pd.DataFrame([r for r in parent_cohort_rows if r["method"] == "R2"]))
    (outdir / "multiband_band_omnibus.json").write_text(json.dumps(omnibus, indent=2))
    np.savez_compressed(outdir / "multiband_subject_null_draws.npz",
                        D=D_mat, N=N_tensor, D_r2=D_mat_r2, N_r2=N_tensor_r2,
                        subjects=np.array(subjects), bands=np.array(band_names),
                        smoothing_policy=np.asarray(args.smoothing_policy))
    # parent-anchor subject null draws per (method, group) for the coherent cohort p (F5)
    np.savez_compressed(outdir / "parent_anchor_subject_null_draws.npz",
                        subjects=np.array(subjects), **parent_null_draws)
    np.savez_compressed(outdir / "within_shaft_subject_null_draws.npz",
                        subjects=np.array(ws_subjects, dtype=object),
                        draws=np.asarray(ws_draws, float) if ws_draws else np.zeros((0, n_perm)))

    hashes_after = {str(p.relative_to(REPO)): sha256_file(p) for p in input_paths}
    immutable = hashes_before == hashes_after
    (outdir / "input_hashes_before_after.json").write_text(json.dumps(
        {"before": hashes_before, "after": hashes_after, "unchanged": immutable}, indent=2))

    try:
        commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"]).decode().strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"]).decode().strip())
    except Exception:
        commit, dirty = None, None
    prim_sigma_rule = ("one_sigma_common_per_subject (shared or own_a)"
                       if args.smoothing_policy == "subject_fixed"
                       else "frozen_per_model (shared->shared; own->A=own_a,B=own_b)")
    manifest = {
        "contract": CONTRACT, "git_commit": commit,
        "git_worktree_dirty": dirty,
        "reproducibility_note": ("code (runner/module/scripts) is committed on branch "
                                 "topic5-fig3-r3-grid-rebuild; results/ and figures are gitignored so "
                                 "they are NOT committed — regenerate from the committed code at this "
                                 "git_commit with the recorded seed/args."),
        "numpy": np.__version__, "pandas": pd.__version__,
        "seed": args.seed, "n_perm": n_perm, "grids": grids,
        "primary_grid": grids[0], "resolution_sensitivity_grid": grids[1] if len(grids) > 1 else None,
        "smoothing_policy": args.smoothing_policy,
        "axis": args.axis,
        "axis_note": ("endpoint = source->sink core axis (build_endpoint_cores k=3), per-template A/B "
                      "for all subjects; NOT the shared-else-own gradient routing (axis+routing confound)"
                      if args.axis == "endpoint" else "gradient shared-else-own (primary)"),
        "score_bands_only": bool(getattr(args, "score_bands_only", False)),
        "r3_formula_version": gg.R3_FORMULA_VERSION,
        "s_thresh": gg.S_THRESH, "overlap_min": {str(n): gg.overlap_min_for_n(n) for n in grids},
        "band_definitions": bands,
        "routing_rule": ("endpoint_per_template_A_B_all_subjects" if args.axis == "endpoint"
                         else "complete_shared_else_own_fallback"),
        "sigma_rule": prim_sigma_rule,
        "null_modes": ["all_contact_shuffle", "within_shaft_min_group_4"],
        "within_shaft_activation": "phenotype_matched (strict->BB150, gamma->gamma30)",
        "cohort_inference": "coherent_cohort_spatial_null_p (permutation) is the pre-registered "
                            "cohort test; wilcoxon/sign-flip are sidecars",
        "fold": "seizure_median_within_subject_before_cohort",
        "input_hashes": hashes_before, "input_hashes_unchanged": immutable,
        "expected": {"subjects": EXPECTED_SUBJECTS, "events": EXPECTED_EVENTS,
                     "strict": EXPECTED_STRICT, "gamma": EXPECTED_GAMMA},
    }
    (outdir / "contract_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    summary = {
        "contract": CONTRACT,
        "cohort": {"subjects": len(subjects), "events": len(events),
                   "shared": len({s for s in subjects if fields[s].route == 'shared'}),
                   "own": len({s for s in subjects if fields[s].route != 'shared'}),
                   "min_common_contacts": int(common_df.n_common_contacts.min()),
                   "median_common_contacts": int(common_df.n_common_contacts.median()),
                   "max_common_contacts": int(common_df.n_common_contacts.max())},
        "parent_cohort": [r for r in parent_cohort_rows if r["method"].startswith("R3")],
        "seven_band": band_cohort_rows,
        "direct_band_omnibus": omnibus,
        "resolution_convergence": {"event_band_abs_diff_p95": conv_p95,
                                   "subject_convergence": band_null_conv,
                                   "gate_p95_le_0p02": bool(conv_p95 <= 0.02) if np.isfinite(conv_p95) else None},
        "within_shaft": ws_cohort,
        "fs_edge": fs_rows,
        "input_hashes_unchanged": immutable,
        "outputs_root": str(outdir),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[write] artifacts -> {outdir}", flush=True)
    pooled = next((r for r in parent_cohort_rows
                   if r['group_id'] == 'all_phenotype_matched' and r['method'].startswith('R3')), None)
    if pooled is not None:
        print(f"[summary] pooled R3: " + json.dumps(
            {k: round(v, 4) if isinstance(v, float) else v for k, v in pooled.items()
             if k in ("data_median", "null_median", "margin_median",
                      "coherent_cohort_spatial_null_p", "n_data_gt_null")}))
    else:
        print(f"[summary] bands-only ({args.axis}); seven-band delta medians: " + json.dumps(
            [round(float(r["delta_cohort_median"]), 4) for r in band_cohort_rows]))
    print(f"[summary] resolution p95 |r161-r81| = {conv_p95:.4f}", flush=True)


if __name__ == "__main__":
    main()
