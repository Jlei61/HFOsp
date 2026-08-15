#!/usr/bin/env python
"""Multimode propagation-grammar audit over the 40-subject masked adaptive-K cohort.

Question
--------
For the subjects whose frozen ``adaptive_cluster.chosen_k`` exceeds 2, what are
the extra modes?  Four candidate readings:

  H1  several genuinely different propagation directions
  H2  one direction, but different recruitment extent / participation structure
  H3  genuinely independent propagation grammars
  H4  pseudo-multimodality driven by recording block / event count (or, more
      generally, by the size of the discrete feature space the clusterer sees)

This script produces the evidence; it does not fit any model and starts no
simulation.

CONTRACT CLAUSES (CLAUDE.md §6 / hfosp-deep-contract-verify ritual).
Each clause is asserted at runtime by the function named after it.

  C1 boundary        transitions are counted only between events adjacent
                     *inside one recording block*; ``block_ids`` is a REQUIRED
                     argument of every transition helper (no ``=None``).
  C2 frozen K        ``chosen_k`` is read from the artifact and never re-picked;
                     asserted equal to the number of distinct labels.
  C3 masked features every rank quantity flows through
                     ``src.lagpat_rank_audit.mask_phantom_ranks`` /
                     ``build_masked_kmeans_features``.  Raw ``lagPatRank`` is
                     never used as a value (Topic 0 §3.1 phantom pseudo-rank).
  C4 valid mask      per-cluster participation mask is derived from raw
                     ``eventsBool``; prototype entries for non-participating
                     channels are NaN and pairwise statistics are restricted to
                     common-valid contacts.  ``adaptive_cluster.clusters[*]
                     .template_rank`` is carried for provenance only (its
                     ``_legacy_hist_mean_rank`` fallback assigns a rank to
                     non-participating channels — cross-PR contract).
  C5 channel order   loader ``channel_names`` is asserted identical to the JSON
                     ``channel_names`` before any indexing.
  C6 label length    ``len(labels) == n_valid_events == |recomputed valid set|``.
  C7 null contract   within-block, occupancy-preserving label permutation;
                     per-block independent; >= 4096 draws; fixed seed; no
                     cross-block adjacency ever enters observed or null.
  C8 reported-along  recording-block bootstrap CI for every pairwise prototype
                     correlation and recruitment difference is a first-pass
                     requirement, not a follow-up.
  C9 provenance      commit, per-subject input hashes, seeds and every
                     exclusion reason are written to the output JSON.

Outputs (all under this directory):
  engineering_audit.json      C1-C6 replay checks, cohort-wide
  cohort_summary.csv/.json    one row per subject
  mode_pairs.csv              one row per (subject, mode_a, mode_b)
  per_subject/<sid>.json      full per-subject payload
"""
from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.interictal_propagation import (  # noqa: E402
    load_subject_propagation_events,
    _valid_event_indices,
)
from src.lagpat_rank_audit import (  # noqa: E402
    build_masked_kmeans_features,
    mask_phantom_ranks,
)
import scripts.run_interictal_propagation as RUNNER  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
PER_SUBJECT_IN = REPO / "results/interictal_propagation_masked/per_subject"

# Frozen upstream producer settings (scripts/run_interictal_propagation.py).
MIN_SHARED_CHANNELS = 3          # _valid_event_indices(min_participating=...)
IMPUTE = "event_median"          # build_masked_kmeans_features(impute=...)

# This audit's own knobs.
SEED = 20260815
N_PERM_WITHIN_BLOCK = 4096       # C7
N_PERM_GLOBAL = 4096
N_BOOT_BLOCKS = 2000             # C8
MIN_PROTO_COUNT = 20             # channel valid in a mode: >= 20 participations
MIN_PROTO_FRAC = 0.05            # ... and >= 5% of that mode's events
MIN_COMMON_VALID = 3             # Spearman needs >= 3 common-valid contacts

CHANNEL_RE = re.compile(r"([A-Za-z][A-Za-z']*)(\d+)")


# ---------------------------------------------------------------------------
# provenance (C9)
# ---------------------------------------------------------------------------
def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True
        ).strip()
    except Exception:
        return "UNKNOWN"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _array_sha256(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# C1-C6 engineering audit
# ---------------------------------------------------------------------------
def replay_and_audit(sid: str) -> Dict[str, Any]:
    """Reload the raw lagPat tree and replay every index the artifact asserts.

    Honours C2 (K frozen), C5 (channel order), C6 (label length).  Raises on
    any mismatch: a silent index drift here would map cluster labels onto the
    wrong events or the wrong channels.
    """
    dataset, subject = sid.split("_", 1)
    json_path = PER_SUBJECT_IN / f"{sid}.json"
    doc = json.load(open(json_path))
    root = RUNNER.YUQUAN_ROOT if dataset == "yuquan" else RUNNER.EPILEPSIAE_ROOT
    subject_dir = RUNNER._subject_dir(dataset, root, subject)
    loaded = load_subject_propagation_events(subject_dir)

    ranks = loaded["ranks"]
    bools = loaded["bools"]
    ch_names = list(loaded["channel_names"])
    ac = doc["adaptive_cluster"]
    labels = np.asarray(ac["labels"], dtype=int)
    chosen_k = int(ac["chosen_k"])            # C2: read, never re-picked

    valid_events = _valid_event_indices(bools, min_participating=MIN_SHARED_CHANNELS)
    boundaries = doc["event_metadata"]["block_boundaries"]

    checks: Dict[str, Any] = {}
    checks["c5_channel_order_matches_json"] = ch_names == list(doc["channel_names"])
    checks["c6_n_valid_replayed_equals_json"] = (
        int(valid_events.size) == int(ac["n_valid_events"]) == int(labels.size)
    )
    checks["c2_distinct_labels_equals_chosen_k"] = int(np.unique(labels).size) == chosen_k
    checks["c2_labels_are_0_to_k_minus_1"] = bool(
        np.array_equal(np.unique(labels), np.arange(chosen_k))
    )
    checks["blocks_contiguous_and_cover_all_events"] = (
        sum(b["n_events"] for b in boundaries) == ranks.shape[1]
        and all(
            boundaries[i]["end_event_idx"] == boundaries[i + 1]["start_event_idx"]
            for i in range(len(boundaries) - 1)
        )
        and boundaries[0]["start_event_idx"] == 0
    )
    block_ids_all = np.concatenate(
        [np.full(b["n_events"], b["block_id"], dtype=int) for b in boundaries]
    )
    checks["loader_block_ids_match_json_boundaries"] = bool(
        np.array_equal(loaded["block_ids"], block_ids_all)
    )
    t = loaded["event_abs_times"]
    checks["event_times_nondecreasing_within_block"] = bool(
        all(
            np.all(np.diff(t[b["start_event_idx"]: b["end_event_idx"]]) >= 0)
            for b in boundaries
        )
    )
    checks["c6_valid_events_sorted_ascending"] = bool(
        np.all(np.diff(valid_events) > 0)
    )

    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise AssertionError(f"{sid}: engineering audit failed: {failed}")

    return {
        "subject_id": sid,
        "dataset": dataset,
        "subject": subject,
        "subject_dir": str(subject_dir),
        "checks": checks,
        "chosen_k": chosen_k,
        "stable_k": int(ac["stable_k"]) if ac.get("stable_k") is not None else None,
        "chosen_reason": ac.get("chosen_reason"),
        "k_range": list(ac.get("k_range", [])),
        "n_channels": int(ranks.shape[0]),
        "n_events_total": int(ranks.shape[1]),
        "n_valid_events": int(valid_events.size),
        "n_blocks_used": int(len(boundaries)),
        "channel_names": ch_names,
        "input_json_sha256": _file_sha256(json_path),
        "raw_ranks_bools_sha256": _array_sha256(ranks, bools),
        "labels_sha256": _array_sha256(labels),
        "_loaded": loaded,
        "_doc": doc,
        "_labels": labels,
        "_valid_events": valid_events,
        "_block_ids_valid": block_ids_all[valid_events],   # C1 carrier
    }


# ---------------------------------------------------------------------------
# Analysis 1 - occupancy and within-block temporal transitions
# ---------------------------------------------------------------------------
def _within_block_adjacent_pairs(block_ids: np.ndarray) -> np.ndarray:
    """C1: boolean over positions i marking that (i, i+1) lies in one block."""
    if block_ids is None:
        raise ValueError("C1 violation: block_ids is required, never None")
    return block_ids[:-1] == block_ids[1:]


def _transition_counts(labels: np.ndarray, block_ids: np.ndarray, k: int) -> np.ndarray:
    """C1: (k, k) adjacent-event transition counts, never crossing a block."""
    same = _within_block_adjacent_pairs(block_ids)
    a = labels[:-1][same]
    b = labels[1:][same]
    return np.bincount(a * k + b, minlength=k * k).reshape(k, k).astype(float)


def _switch_rate_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return float("nan")
    return float((total - np.trace(counts)) / total)


def _within_block_permutation_null(
    labels: np.ndarray,
    block_ids: np.ndarray,
    k: int,
    n_perm: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """C7: occupancy-preserving within-block label permutation null.

    Clause by clause:
      - occupancy-preserving : each block's label multiset is permuted, so the
        per-block (and hence subject-level) occupancy vector is byte-identical
        to the observed one in every draw.
      - per-block independent: a fresh permutation is drawn for every block.
      - no cross-block adjacency: the switch count is accumulated inside each
        block only; the (last event of block b, first event of block b+1) pair
        is never counted, in the observed statistic or in the null.
      - fixed seed              : ``rng`` is seeded by the caller.
    """
    if block_ids is None:
        raise ValueError("C1 violation: block_ids is required, never None")
    n_pairs_total = 0
    switch_counts = np.zeros(n_perm, dtype=np.int64)
    self_counts = np.zeros((n_perm, k), dtype=np.int64)
    order = np.argsort(block_ids, kind="stable")
    if not np.array_equal(order, np.arange(block_ids.size)):
        raise AssertionError("C1 violation: block ids are not contiguous/sorted")

    starts = np.flatnonzero(np.r_[True, block_ids[1:] != block_ids[:-1]])
    ends = np.r_[starts[1:], block_ids.size]
    for s, e in zip(starts, ends):
        lab_b = labels[s:e]
        n_b = lab_b.size
        if n_b < 2:
            continue
        n_pairs_total += n_b - 1
        tiled = np.tile(lab_b.astype(np.int16), (n_perm, 1))
        perm = rng.permuted(tiled, axis=1)
        neq = perm[:, 1:] != perm[:, :-1]
        switch_counts += neq.sum(axis=1)
        stay = ~neq
        for ci in range(k):
            self_counts[:, ci] += (stay & (perm[:, :-1] == ci)).sum(axis=1)
        del tiled, perm, neq, stay

    if n_pairs_total == 0:
        return {"n_pairs": 0, "error": "no within-block adjacent pairs"}
    null_switch = switch_counts / float(n_pairs_total)
    return {
        "n_pairs": int(n_pairs_total),
        "n_perm": int(n_perm),
        "null_switch_mean": float(null_switch.mean()),
        "null_switch_sd": float(null_switch.std(ddof=1)),
        "null_switch_p2_5": float(np.percentile(null_switch, 2.5)),
        "null_switch_p97_5": float(np.percentile(null_switch, 97.5)),
        "_null_switch": null_switch,
    }


def analysis1_occupancy_and_transitions(rec: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    labels = rec["_labels"]
    block_ids = rec["_block_ids_valid"]     # C1
    k = rec["chosen_k"]                     # C2
    n = labels.size

    counts_occ = np.bincount(labels, minlength=k).astype(float)
    pi = counts_occ / counts_occ.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = float(-np.nansum(np.where(pi > 0, pi * np.log(pi), 0.0)))
    norm_ent = ent / np.log(k) if k > 1 else float("nan")

    counts = _transition_counts(labels, block_ids, k)
    n_pairs = float(counts.sum())
    obs_switch = _switch_rate_from_counts(counts)
    row_sums = counts.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        trans_prob = np.where(row_sums[:, None] > 0, counts / row_sums[:, None], np.nan)
    self_trans = np.diag(trans_prob).tolist()
    exit_prob = [1.0 - s if np.isfinite(s) else float("nan") for s in self_trans]
    # entry probability into j given the previous event was NOT j
    col_from_other = counts.sum(axis=0) - np.diag(counts)
    denom_other = n_pairs - row_sums  # pairs whose source is not j
    entry_prob = [
        float(col_from_other[j] / denom_other[j]) if denom_other[j] > 0 else float("nan")
        for j in range(k)
    ]

    null = _within_block_permutation_null(labels, block_ids, k, N_PERM_WITHIN_BLOCK, rng)
    if "_null_switch" in null:
        ns = null.pop("_null_switch")
        # two-sided empirical p with +1 correction
        n_ge = int(np.sum(ns >= obs_switch))
        n_le = int(np.sum(ns <= obs_switch))
        p_two = min(1.0, 2.0 * (min(n_ge, n_le) + 1) / (ns.size + 1))
        excess = float(obs_switch - null["null_switch_mean"])
        z = float(excess / null["null_switch_sd"]) if null["null_switch_sd"] > 0 else float("nan")
    else:
        p_two, excess, z = float("nan"), float("nan"), float("nan")

    # per-block mode expression
    per_block = []
    for b in np.unique(block_ids):
        m = block_ids == b
        lab_b = labels[m]
        cnt_b = np.bincount(lab_b, minlength=k)
        per_block.append(
            {
                "block_id": int(b),
                "n_events": int(lab_b.size),
                "n_modes_present": int((cnt_b > 0).sum()),
                "counts": cnt_b.tolist(),
            }
        )
    n_blocks_ev = len(per_block)
    frac_blocks_multimode = float(np.mean([b["n_modes_present"] >= 2 for b in per_block]))
    frac_blocks_all_modes = float(np.mean([b["n_modes_present"] == k for b in per_block]))

    # H4 probe: mode x block association vs a GLOBAL label permutation null.
    # This null destroys the block association entirely; it is a DIFFERENT null
    # from the within-block one above and is labelled as such everywhere.
    # Note on interpretation: with 10^4-10^5 events the p-value of this test is
    # near-degenerate; the effect size (Cramer's V) against the null mean is the
    # quantity to read, not the p.
    table = np.array([b["counts"] for b in per_block], dtype=float)
    v_obs = _cramers_v(table)
    # dense block index over the valid-event axis (blocks with events only)
    uniq_blocks = np.array([b["block_id"] for b in per_block])
    blk_dense = np.searchsorted(uniq_blocks, block_ids)
    n_blk = uniq_blocks.size
    v_null = np.empty(N_PERM_GLOBAL)
    for i in range(N_PERM_GLOBAL):
        perm = rng.permutation(labels)
        tab = np.bincount(blk_dense * k + perm, minlength=n_blk * k).reshape(n_blk, k)
        v_null[i] = _cramers_v(tab.astype(float))
    p_v = float((np.sum(v_null >= v_obs) + 1) / (N_PERM_GLOBAL + 1))

    return {
        "n_valid_events": int(n),
        "occupancy": pi.tolist(),
        "occupancy_counts": counts_occ.astype(int).tolist(),
        "entropy_nats": ent,
        "normalized_entropy": norm_ent,
        "transition_counts": counts.astype(int).tolist(),
        "transition_prob": np.where(np.isfinite(trans_prob), trans_prob, None).tolist(),
        "n_within_block_pairs": int(n_pairs),
        "n_cross_block_pairs_excluded": int(n - 1 - n_pairs),
        "observed_switch_rate": obs_switch,
        "self_transition": self_trans,
        "exit_prob": exit_prob,
        "entry_prob": entry_prob,
        "within_block_permutation_null": null,
        "excess_switch_rate": excess,
        "switch_rate_z_vs_null": z,
        "switch_rate_p_two_sided": p_two,
        "n_blocks_with_events": n_blocks_ev,
        "frac_blocks_expressing_ge2_modes": frac_blocks_multimode,
        "frac_blocks_expressing_all_modes": frac_blocks_all_modes,
        "per_block": per_block,
        "mode_block_cramers_v": float(v_obs),
        "mode_block_cramers_v_null_mean": float(v_null.mean()),
        "mode_block_cramers_v_p": p_v,
    }


def _cramers_v(table: np.ndarray) -> float:
    total = table.sum()
    if total <= 0:
        return float("nan")
    row = table.sum(axis=1, keepdims=True)
    col = table.sum(axis=0, keepdims=True)
    expected = row @ col / total
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum(np.where(expected > 0, (table - expected) ** 2 / expected, 0.0))
    r, c = table.shape
    denom = total * (min(r, c) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else float("nan")


# ---------------------------------------------------------------------------
# Analysis 2 - direction and recruitment decomposition
# ---------------------------------------------------------------------------
def _mode_prototype(
    masked: np.ndarray,
    bools_valid: np.ndarray,
    sel: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """C3 + C4: masked-rank prototype, participation count, per-mode valid mask.

    ``masked`` is the output of ``mask_phantom_ranks`` (NaN where the channel
    did not participate), so the prototype averages only real participations.
    A channel that never participates in this mode gets NaN, and its
    ``valid`` entry is False -- it can never be picked as an endpoint or enter
    a pairwise correlation.
    """
    sub = masked[:, sel]
    part = bools_valid[:, sel]
    count = part.sum(axis=1).astype(float)
    frac = count / max(int(sel.sum()), 1)
    with np.errstate(invalid="ignore"):
        proto = np.nanmean(np.where(part, sub, np.nan), axis=1)
    valid = (count >= MIN_PROTO_COUNT) & (frac >= MIN_PROTO_FRAC)
    proto = np.where(valid, proto, np.nan)
    return proto, count, valid


def _pair_stats(
    proto_a: np.ndarray, valid_a: np.ndarray,
    proto_b: np.ndarray, valid_b: np.ndarray,
) -> Tuple[float, int]:
    common = valid_a & valid_b & np.isfinite(proto_a) & np.isfinite(proto_b)
    n_common = int(common.sum())
    if n_common < MIN_COMMON_VALID:
        return float("nan"), n_common
    a, b = proto_a[common], proto_b[common]
    if np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan"), n_common
    rho = float(stats.spearmanr(a, b).statistic)
    return rho, n_common


def _axial_score(proto: np.ndarray, valid: np.ndarray, contact_idx: np.ndarray,
                 shaft_id: np.ndarray) -> Dict[str, Any]:
    """Signed Spearman between the mode prototype and along-shaft contact index.

    Only meaningful when the valid contacts of the mode lie on one shaft (a
    single spatial axis).  For multi-shaft modes we report the score of the
    shaft carrying the most valid contacts, plus the number of shafts spanned.
    """
    ok = valid & np.isfinite(proto)
    if ok.sum() < MIN_COMMON_VALID:
        return {"axial_rho": None, "axial_shaft": None, "n_shafts_valid": int(np.unique(shaft_id[ok]).size)}
    shafts, counts = np.unique(shaft_id[ok], return_counts=True)
    best = shafts[int(np.argmax(counts))]
    sel = ok & (shaft_id == best)
    if sel.sum() < MIN_COMMON_VALID:
        return {"axial_rho": None, "axial_shaft": str(best), "n_shafts_valid": int(shafts.size)}
    rho = float(stats.spearmanr(contact_idx[sel], proto[sel]).statistic)
    return {"axial_rho": rho, "axial_shaft": str(best), "n_shafts_valid": int(shafts.size),
            "n_contacts_on_axial_shaft": int(sel.sum())}


def analysis2_direction_and_extent(
    rec: Dict[str, Any], rng: np.random.Generator
) -> Dict[str, Any]:
    loaded = rec["_loaded"]
    labels = rec["_labels"]
    ve = rec["_valid_events"]
    k = rec["chosen_k"]
    ch_names = rec["channel_names"]
    n_ch = len(ch_names)

    ranks_v = loaded["ranks"][:, ve]
    bools_v = loaded["bools"][:, ve]
    masked = mask_phantom_ranks(ranks_v, bools_v, normalize=True)   # C3
    npart = bools_v.sum(axis=0).astype(float)
    block_ids = rec["_block_ids_valid"]

    shaft_id = np.array([CHANNEL_RE.fullmatch(c).group(1) for c in ch_names])
    contact_idx = np.array([int(CHANNEL_RE.fullmatch(c).group(2)) for c in ch_names], dtype=float)
    n_shafts = int(np.unique(shaft_id).size)
    shaft_parsable = True   # verified cohort-wide upstream; re-asserted here
    for c in ch_names:
        if CHANNEL_RE.fullmatch(c) is None:
            shaft_parsable = False

    modes: List[Dict[str, Any]] = []
    protos = np.full((k, n_ch), np.nan)
    valids = np.zeros((k, n_ch), dtype=bool)
    part_rates = np.full((k, n_ch), np.nan)
    for ci in range(k):
        sel = labels == ci
        proto, count, valid = _mode_prototype(masked, bools_v, sel)
        protos[ci] = proto
        valids[ci] = valid
        part_rates[ci] = count / max(int(sel.sum()), 1)
        rec_frac = npart[sel] / n_ch
        ax = _axial_score(proto, valid, contact_idx, shaft_id)
        # shaft participation: fraction of the mode's events with >=1 contact
        # of that shaft participating
        shaft_part = {}
        for s in np.unique(shaft_id):
            rows = shaft_id == s
            shaft_part[str(s)] = float(np.mean(bools_v[rows][:, sel].any(axis=0)))
        modes.append(
            {
                "mode_id": ci,
                "n_events": int(sel.sum()),
                "fraction": float(sel.mean()),
                "prototype_masked_rank": [None if not np.isfinite(v) else float(v) for v in proto],
                "valid_mask": valid.tolist(),
                "n_valid_contacts": int(valid.sum()),
                "participation_rate": part_rates[ci].tolist(),
                "median_recruited_contacts": float(np.median(npart[sel])),
                "median_recruited_fraction": float(np.median(rec_frac)),
                "mean_recruited_fraction": float(np.mean(rec_frac)),
                "shaft_participation": shaft_part,
                **ax,
                # provenance only (C4): the frozen artifact's template_rank
                "artifact_template_rank": rec["_doc"]["adaptive_cluster"]["clusters"][ci].get("template_rank"),
            }
        )

    # ---- C8 recording-block bootstrap via per-block sufficient statistics ----
    # A block bootstrap resamples whole recording blocks, so every prototype is
    # an additive function of per-block sums.  Precomputing those sums makes each
    # of the N_BOOT_BLOCKS draws an O(n_blocks) tensor contraction instead of a
    # full recomputation, and is *exactly* equivalent to recomputing on the
    # resampled events (verified by `verify_bootstrap_equivalence.py`).
    uniq_blocks = np.unique(block_ids)
    n_blk = uniq_blocks.size
    blk_dense = np.searchsorted(uniq_blocks, block_ids)
    p_vals = np.arange(MIN_SHARED_CHANNELS, n_ch + 1)
    sum_rank = np.zeros((n_blk, k, n_ch))
    cnt_part = np.zeros((n_blk, k, n_ch))
    cnt_ev = np.zeros((n_blk, k))
    hist_np = np.zeros((n_blk, k, p_vals.size))
    masked0 = np.where(bools_v, np.nan_to_num(masked, nan=0.0), 0.0)
    for ci in range(k):
        sel = labels == ci
        if not sel.any():
            continue
        bd, m0, bl = blk_dense[sel], masked0[:, sel], bools_v[:, sel]
        for ch in range(n_ch):
            sum_rank[:, ci, ch] = np.bincount(bd, weights=m0[ch], minlength=n_blk)
            cnt_part[:, ci, ch] = np.bincount(bd, weights=bl[ch].astype(float), minlength=n_blk)
        cnt_ev[:, ci] = np.bincount(bd, minlength=n_blk)
        npi = npart[sel].astype(int) - MIN_SHARED_CHANNELS
        hist_np[:, ci, :] = np.bincount(
            bd * p_vals.size + npi, minlength=n_blk * p_vals.size
        ).reshape(n_blk, p_vals.size)

    def _protos_from_weights(w: np.ndarray):
        """(k, n_ch) prototypes + valid masks + median recruited fraction."""
        sr = np.tensordot(w, sum_rank, axes=(0, 0))
        cp = np.tensordot(w, cnt_part, axes=(0, 0))
        ne = w @ cnt_ev
        hn = np.tensordot(w, hist_np, axes=(0, 0))
        with np.errstate(invalid="ignore", divide="ignore"):
            pr = np.where(cp > 0, sr / np.maximum(cp, 1e-12), np.nan)
            frac = np.where(ne[:, None] > 0, cp / np.maximum(ne[:, None], 1e-12), 0.0)
        vd = (cp >= MIN_PROTO_COUNT) & (frac >= MIN_PROTO_FRAC)
        pr = np.where(vd, pr, np.nan)
        med = np.full(k, np.nan)
        for ci in range(k):
            tot = int(round(hn[ci].sum()))
            if tot > 0:
                # np.median semantics from a histogram: mean of the two central
                # order statistics (identical to np.median on the raw values).
                cum = np.cumsum(hn[ci])
                j1, j2 = (tot - 1) // 2, tot // 2
                v1 = p_vals[int(np.searchsorted(cum, j1, side="right"))]
                v2 = p_vals[int(np.searchsorted(cum, j2, side="right"))]
                med[ci] = 0.5 * (v1 + v2) / n_ch
        return pr, vd, med, ne

    w_obs = np.ones(n_blk)
    _pr_chk, _vd_chk, med_obs, _ = _protos_from_weights(w_obs)
    boot_w = rng.multinomial(n_blk, np.full(n_blk, 1.0 / n_blk), size=N_BOOT_BLOCKS).astype(float)
    boot_pr, boot_vd, boot_med, boot_ne = [], [], [], []
    for i in range(N_BOOT_BLOCKS):
        pr_i, vd_i, med_i, ne_i = _protos_from_weights(boot_w[i])
        boot_pr.append(pr_i); boot_vd.append(vd_i); boot_med.append(med_i); boot_ne.append(ne_i)

    pair_rows: List[Dict[str, Any]] = []
    for a in range(k):
        for b in range(a + 1, k):
            rho, n_common = _pair_stats(protos[a], valids[a], protos[b], valids[b])
            d_rec = modes[a]["median_recruited_fraction"] - modes[b]["median_recruited_fraction"]
            part_diff = np.abs(part_rates[a] - part_rates[b])
            sp_a, sp_b = modes[a]["shaft_participation"], modes[b]["shaft_participation"]
            shaft_diff = (
                float(max(abs(sp_a[s] - sp_b[s]) for s in sp_a))
                if shaft_parsable and n_shafts >= 2 else None
            )
            br, bd_ = [], []
            for i in range(N_BOOT_BLOCKS):
                if boot_ne[i][a] < MIN_PROTO_COUNT or boot_ne[i][b] < MIN_PROTO_COUNT:
                    continue
                r_i, _ = _pair_stats(boot_pr[i][a], boot_vd[i][a], boot_pr[i][b], boot_vd[i][b])
                if np.isfinite(r_i):
                    br.append(r_i)
                if np.isfinite(boot_med[i][a]) and np.isfinite(boot_med[i][b]):
                    bd_.append(float(boot_med[i][a] - boot_med[i][b]))
            pair_rows.append(
                {
                    "mode_a": a,
                    "mode_b": b,
                    "spearman_rho": rho,
                    "prototype_distance_1_minus_rho": (None if not np.isfinite(rho) else float(1.0 - rho)),
                    "n_common_valid_contacts": n_common,
                    "rho_undefined_reason": (
                        None if np.isfinite(rho)
                        else ("too_few_common_valid_contacts" if n_common < MIN_COMMON_VALID
                              else "constant_prototype_on_common_contacts")
                    ),
                    "median_recruited_fraction_a": modes[a]["median_recruited_fraction"],
                    "median_recruited_fraction_b": modes[b]["median_recruited_fraction"],
                    "recruited_fraction_diff": float(d_rec),
                    "abs_recruited_fraction_diff": float(abs(d_rec)),
                    "max_participation_rate_diff": float(np.nanmax(part_diff)),
                    "mean_participation_rate_diff": float(np.nanmean(part_diff)),
                    "max_shaft_participation_diff": shaft_diff,
                    "rho_boot_ci": (
                        [float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))]
                        if len(br) >= 50 else None),
                    "rho_boot_n": len(br),
                    "recruited_fraction_diff_boot_ci": (
                        [float(np.percentile(bd_, 2.5)), float(np.percentile(bd_, 97.5))]
                        if len(bd_) >= 50 else None),
                    "recruited_fraction_diff_boot_n": len(bd_),
                }
            )

    # secondary: do the prototypes fall into two direction superfamilies?
    superfamily = _direction_superfamily(protos, valids, k)

    # ---- exact information decomposition of what the mode label encodes ----
    # The frozen label is a deterministic function of the masked feature vector,
    # and the feature vector factorises as (participating SET, ORDER within set).
    # Hence  H(mode) = I(mode; SET) + I(mode; ORDER | SET), an exact split of
    # "which contacts took part" (recruitment / participation structure) against
    # "in what order they fired" (propagation direction).
    set_code = np.packbits(bools_v.T.astype(np.uint8), axis=1)
    set_id = np.asarray(np.unique(set_code, axis=0, return_inverse=True)[1]).ravel()
    X = build_masked_kmeans_features(ranks_v, bools_v, impute=IMPUTE)   # C3
    uniq, inv, cnt = np.unique(np.round(X, 9), axis=0, return_inverse=True, return_counts=True)
    inv = np.asarray(inv).ravel()
    n_distinct = int(uniq.shape[0])
    n_possible = _lattice_cardinality(n_ch, np.unique(npart.astype(int)))
    h_mode = _entropy(np.bincount(labels, minlength=k))
    mi_set = _mutual_information(labels, set_id)
    mi_npart = _mutual_information(labels, npart.astype(int))
    mi_full = _mutual_information(labels, inv)
    per_mode_vec = [int(np.unique(inv[labels == ci]).size) for ci in range(k)]

    return {
        "n_channels": n_ch,
        "n_shafts": n_shafts,
        "shaft_parsable": shaft_parsable,
        "shaft_of_channel": shaft_id.tolist(),
        "contact_index": contact_idx.tolist(),
        "modes": modes,
        "mode_pairs": pair_rows,
        "direction_superfamily": superfamily,
        "feature_space": {
            "n_distinct_masked_feature_vectors": n_distinct,
            "n_possible_lattice_points": n_possible,
            "lattice_saturation": (float(n_distinct / n_possible) if n_possible else None),
            "events_per_distinct_vector": float(X.shape[0] / n_distinct),
            "top1_vector_share": float(cnt.max() / cnt.sum()),
            "n_distinct_vectors_per_mode": per_mode_vec,
            "median_distinct_vectors_per_mode": float(np.median(per_mode_vec)),
            # SANITY CHECK ONLY, not evidence: identical feature vectors must
            # receive identical KMeans labels by construction, so this is 1.0
            # whenever the replay is correct.  It verifies that no information
            # outside the feature matrix entered the frozen labelling.
            "sanity_label_is_function_of_feature_vector": bool(
                mi_full >= h_mode - 1e-9),
        },
        "information_decomposition": {
            "H_mode_nats": float(h_mode),
            "H_mode_normalized": float(h_mode / np.log(k)) if k > 1 else None,
            "I_mode_participation_set": float(mi_set),
            "I_mode_n_participating": float(mi_npart),
            "frac_mode_explained_by_participation_set": float(mi_set / h_mode) if h_mode > 0 else None,
            "frac_mode_explained_by_n_participating": float(mi_npart / h_mode) if h_mode > 0 else None,
            "frac_mode_explained_by_order_within_set": float((h_mode - mi_set) / h_mode) if h_mode > 0 else None,
        },
    }


def _entropy(counts: np.ndarray) -> float:
    p = np.asarray(counts, dtype=float)
    p = p[p > 0] / p.sum()
    return float(-np.sum(p * np.log(p)))


def _mutual_information(a: np.ndarray, b: np.ndarray) -> float:
    """Plug-in mutual information in nats between two integer labellings."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    ua, ai = np.unique(a, return_inverse=True)
    ub, bi = np.unique(b, return_inverse=True)
    joint = np.bincount(ai * ub.size + bi, minlength=ua.size * ub.size).reshape(ua.size, ub.size)
    return float(_entropy(joint.sum(1)) + _entropy(joint.sum(0)) - _entropy(joint.ravel()))


def _lattice_cardinality(n_ch: int, npart_values: Sequence[int]) -> Optional[int]:
    """Number of distinct masked feature vectors reachable with these n_part.

    For p participating channels out of n, the normalised masked rank places
    values ``j/(p-1)`` on the participating channels and the constant 0.5 on
    the rest.  For p == 3 the middle value equals the fill value 0.5, so only
    the (first, last) pair is identifiable: C(n,2) * 2 vectors.  For p >= 4
    every (subset, ordering) is distinct: C(n,p) * p!.
    """
    from math import comb, factorial
    total = 0
    for p in npart_values:
        p = int(p)
        if p < 3 or p > n_ch:
            continue
        if p == 3:
            total += comb(n_ch, 2) * 2
        else:
            total += comb(n_ch, p) * factorial(p)
    return int(total) if total else None


def _direction_superfamily(protos: np.ndarray, valids: np.ndarray, k: int) -> Dict[str, Any]:
    """Secondary check: split modes into two groups by prototype correlation.

    Reports the achieved within-group and between-group mean rho for the best
    2-way split.  A genuine forward/reverse pair of superfamilies needs
    within > 0 and between < 0; anything else is not a two-direction structure.
    """
    if k < 3:
        return {"applicable": False, "reason": "k < 3"}
    rho = np.full((k, k), np.nan)
    for a in range(k):
        for b in range(k):
            if a == b:
                rho[a, b] = 1.0
            elif a < b:
                r, _ = _pair_stats(protos[a], valids[a], protos[b], valids[b])
                rho[a, b] = rho[b, a] = r
    if not np.any(np.isfinite(rho[~np.eye(k, dtype=bool)])):
        return {"applicable": False, "reason": "no finite pairwise rho"}
    best = None
    for mask in range(1, 2 ** (k - 1)):
        assign = np.array([(mask >> i) & 1 for i in range(k)], dtype=bool)
        if assign.sum() == 0 or assign.sum() == k:
            continue
        w, btw = [], []
        for a in range(k):
            for b in range(a + 1, k):
                if not np.isfinite(rho[a, b]):
                    continue
                (w if assign[a] == assign[b] else btw).append(rho[a, b])

        if not w or not btw:
            continue
        score = float(np.mean(w) - np.mean(btw))
        if best is None or score > best["separation"]:
            best = {
                "assignment": assign.astype(int).tolist(),
                "within_mean_rho": float(np.mean(w)),
                "between_mean_rho": float(np.mean(btw)),
                "separation": score,
                "group_sizes": [int((~assign).sum()), int(assign.sum())],
            }
    if best is None:
        return {"applicable": False, "reason": "no admissible split with finite rho"}
    best["applicable"] = True
    best["two_opposite_families"] = bool(
        best["within_mean_rho"] > 0 and best["between_mean_rho"] < 0
    )
    best["rho_matrix"] = np.where(np.isfinite(rho), rho, None).tolist()
    return best


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def main() -> None:
    subject_ids = sorted(p.stem for p in PER_SUBJECT_IN.glob("*.json"))
    provenance = {
        "git_commit": _git_commit(),
        "git_branch": subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(REPO), text=True
        ).strip(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "seed": SEED,
        "n_perm_within_block": N_PERM_WITHIN_BLOCK,
        "n_perm_global": N_PERM_GLOBAL,
        "n_boot_blocks": N_BOOT_BLOCKS,
        "min_shared_channels_replayed": MIN_SHARED_CHANNELS,
        "impute": IMPUTE,
        "min_proto_count": MIN_PROTO_COUNT,
        "min_proto_frac": MIN_PROTO_FRAC,
        "min_common_valid": MIN_COMMON_VALID,
        "input_dir": str(PER_SUBJECT_IN),
        "n_subject_json_found": len(subject_ids),
    }

    audits, cohort, pair_rows_all, exclusions = [], [], [], []
    for i, sid in enumerate(subject_ids):
        rng = np.random.default_rng(SEED + i)   # C9: deterministic per subject
        try:
            rec = replay_and_audit(sid)
        except Exception as exc:  # C9: every exclusion is recorded with a reason
            exclusions.append({"subject_id": sid, "stage": "replay_and_audit", "reason": repr(exc)})
            print(f"[EXCLUDE] {sid}: {exc}", flush=True)
            continue
        print(f"[{i+1}/{len(subject_ids)}] {sid} K={rec['chosen_k']} "
              f"nch={rec['n_channels']} nvalid={rec['n_valid_events']}", flush=True)

        a1 = analysis1_occupancy_and_transitions(rec, rng)
        a2 = analysis2_direction_and_extent(rec, rng)

        payload = {
            "provenance": provenance,
            "subject_id": sid,
            "dataset": rec["dataset"],
            "subject": rec["subject"],
            "engineering_audit": {kk: rec[kk] for kk in
                                  ("checks", "chosen_k", "stable_k", "chosen_reason", "k_range",
                                   "n_channels", "n_events_total", "n_valid_events",
                                   "n_blocks_used", "channel_names", "input_json_sha256",
                                   "raw_ranks_bools_sha256", "labels_sha256", "subject_dir")},
            "analysis1_occupancy_transitions": a1,
            "analysis2_direction_extent": a2,
        }
        with open(OUT_DIR / "per_subject" / f"{sid}.json", "w") as f:
            json.dump(payload, f, indent=2, allow_nan=True)

        audits.append({kk: rec[kk] for kk in
                       ("subject_id", "dataset", "subject", "checks", "chosen_k", "stable_k",
                        "chosen_reason", "n_channels", "n_events_total", "n_valid_events",
                        "n_blocks_used", "input_json_sha256", "raw_ranks_bools_sha256",
                        "labels_sha256", "subject_dir")})

        fs = a2["feature_space"]
        cohort.append({
            "subject_id": sid,
            "dataset": rec["dataset"],
            "chosen_k": rec["chosen_k"],
            "n_channels": rec["n_channels"],
            "n_shafts": a2["n_shafts"],
            "single_shaft": int(a2["n_shafts"] == 1),
            "n_events_total": rec["n_events_total"],
            "n_valid_events": rec["n_valid_events"],
            "n_blocks_used": rec["n_blocks_used"],
            "n_blocks_with_events": a1["n_blocks_with_events"],
            "normalized_entropy": a1["normalized_entropy"],
            "dominant_occupancy": float(max(a1["occupancy"])),
            "observed_switch_rate": a1["observed_switch_rate"],
            "null_switch_mean": a1["within_block_permutation_null"].get("null_switch_mean"),
            "excess_switch_rate": a1["excess_switch_rate"],
            "switch_rate_z": a1["switch_rate_z_vs_null"],
            "switch_rate_p": a1["switch_rate_p_two_sided"],
            "n_within_block_pairs": a1["n_within_block_pairs"],
            "n_cross_block_pairs_excluded": a1["n_cross_block_pairs_excluded"],
            "frac_blocks_ge2_modes": a1["frac_blocks_expressing_ge2_modes"],
            "frac_blocks_all_modes": a1["frac_blocks_expressing_all_modes"],
            "mode_block_cramers_v": a1["mode_block_cramers_v"],
            "mode_block_cramers_v_null_mean": a1["mode_block_cramers_v_null_mean"],
            "mode_block_cramers_v_p": a1["mode_block_cramers_v_p"],
            "n_distinct_feature_vectors": fs["n_distinct_masked_feature_vectors"],
            "n_possible_lattice_points": fs["n_possible_lattice_points"],
            "lattice_saturation": fs["lattice_saturation"],
            "events_per_distinct_vector": fs["events_per_distinct_vector"],
            "median_distinct_vectors_per_mode": fs["median_distinct_vectors_per_mode"],
            "sanity_label_is_function_of_feature_vector": fs["sanity_label_is_function_of_feature_vector"],
            "frac_mode_by_participation_set": a2["information_decomposition"]["frac_mode_explained_by_participation_set"],
            "frac_mode_by_n_participating": a2["information_decomposition"]["frac_mode_explained_by_n_participating"],
            "frac_mode_by_order_within_set": a2["information_decomposition"]["frac_mode_explained_by_order_within_set"],
            "median_recruited_fraction_overall": float(np.median(
                [m["median_recruited_fraction"] for m in a2["modes"]])),
            "superfamily_two_opposite": (
                a2["direction_superfamily"].get("two_opposite_families")
                if a2["direction_superfamily"].get("applicable") else None),
            "superfamily_within_rho": a2["direction_superfamily"].get("within_mean_rho"),
            "superfamily_between_rho": a2["direction_superfamily"].get("between_mean_rho"),
            "n_modes_with_axial_rho": int(sum(m.get("axial_rho") is not None for m in a2["modes"])),
        })
        for pr in a2["mode_pairs"]:
            pair_rows_all.append({"subject_id": sid, "chosen_k": rec["chosen_k"],
                                  "n_channels": rec["n_channels"], **pr})

    with open(OUT_DIR / "engineering_audit.json", "w") as f:
        json.dump({"provenance": provenance, "per_subject": audits,
                   "exclusions": exclusions,
                   "all_checks_passed": all(all(a["checks"].values()) for a in audits)}, f, indent=2)

    import csv
    with open(OUT_DIR / "cohort_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cohort[0].keys()))
        w.writeheader()
        w.writerows(cohort)
    with open(OUT_DIR / "cohort_summary.json", "w") as f:
        json.dump({"provenance": provenance, "rows": cohort, "exclusions": exclusions}, f, indent=2)
    with open(OUT_DIR / "mode_pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows_all[0].keys()))
        w.writeheader()
        for r in pair_rows_all:
            w.writerow({kk: (json.dumps(v) if isinstance(v, list) else v) for kk, v in r.items()})

    print(f"\nDone. {len(cohort)} subjects, {len(pair_rows_all)} mode pairs, "
          f"{len(exclusions)} exclusions.")


if __name__ == "__main__":
    main()
