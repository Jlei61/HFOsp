#!/usr/bin/env python3
"""Build the Axis-A A5-real per-subject feature-RELIABILITY audit (plan §A5).

This is the reliability gate that decides which subjects/features are stable
enough under BOTH cross-validation folds to enter the eventual A5 real-data
placement. It is the reliability companion to real_feature_coverage.{csv,json}:
coverage answers "is this feature available?"; this script answers "is the
available feature stable under split-half AND odd-even?".

Descriptive only: this carries NO mechanism label and makes NO scientific
claim. It is a reliability inventory that gates placement, nothing else.

--- 朴素话（plan §A5 reliability audit）---
- 测了什么 —— 对每个真实被试，看它的"传播指纹"主特征在把数据切两半之后还稳不稳：
  前半 vs 后半（split-half）、奇数块 vs 偶数块（odd-even），两种切法都看。稳的才放行进 A5
  摆位，不稳的（一种切法过、另一种切法不过）被挡在门外。
- 怎么测的 —— 方向特征（axis_dir）用 rank-space 模板的两半一致度：把前/后半各自的簇 rank 模板
  对上，看 mean_match_corr（=cluster_rank_a 与 cluster_rank_b_matched_to_a 的一致度），两折分别取一个数；
  另带 mm-axis 的 split_half_validation.spearman_rho 作为次要折叠核对（只报，不当门）。
  入口抖动（onset_jitter）当前读出层没存"每折最早入口概率"，所以诚实地用同一套 rank 两折一致度
  当代理（proxy），并明写"per-fold earliest-prob recompute is deferred"，不另造一个数。
  通路宽度（pathway_width）只有全量 perp_spread、没有逐折版本，所以判 deferred、不造假，
  并列出哪些被试将来能补这一折（perp_width_measurable 且有坐标）。
- 揭示了什么 —— 不下 PASS/NULL 的科学结论，只给"这个被试这个特征在两折下看起来稳/不稳/还没法判"。
  门规则比模板级 forward_reverse_reproduced 的 OR 规则更严：连续特征要两折都 >= 0.6（AND），
  不是只要一折过就放行。

Inputs (read-only):
  fingerprint/real_feature_coverage.json
      -> the availability mask per subject/feature. We only audit reliability
         where the feature is available (axis_available / coverage rows).
  results/interictal_propagation_masked/per_subject/<ds>_<sub>.json
      -> time_split_reproducibility.splits.{first_half_second_half,
         odd_even_block}.mean_match_corr (rank-space, coord-free fold material).
  results/spatial_modulation/propagation_geometry/per_subject/<ds>_<sub>.json
      -> path_axis.split_half_validation.spearman_rho (secondary mm-axis fold),
         weak_axis, axis_length_mm (None => degenerate hard-fail),
         perp_spread, perp_width_measurable.

Outputs (the artifact):
  fingerprint/real_feature_reliability.csv
  fingerprint/real_feature_reliability.json

Gate rule (plan §A5, stricter than the template-level OR-rule):
  A CONTINUOUS feature (axis_dir, onset_jitter) is admitted only if
  split_half >= 0.6 AND odd_even >= 0.6 (BOTH folds). For pathway_width,
  reliability_pass = 'deferred' (cannot decide this round; no per-fold
  perp_spread exists).

mm discipline: spearman_rho is reported per subject, never pooled across
datasets (epi=mni152_1mm vs yuquan=fs_native_ras_mm).
"""

import csv
import json
import os
from collections import OrderedDict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = os.path.join(
    REPO,
    "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/fingerprint",
)
COVERAGE_JSON = os.path.join(OUT_DIR, "real_feature_coverage.json")
MASKED_DIR = os.path.join(
    REPO, "results/interictal_propagation_masked/per_subject"
)
GEO_DIR = os.path.join(
    REPO, "results/spatial_modulation/propagation_geometry/per_subject"
)

# --- INTERIM broad-lagPat patch (2026-06-15) -------------------------------
# huangwanling + zhaojinrui: 4ch single-shaft narrow montage -> degenerate
# axis. Their broad-lagPat re-derivation (top_n=20, multi-shaft; built by
# scripts/build_broad_lagpat_patch.py) gives a usable axis. Overlay the broad
# masked propagation JSON (for the time_split folds) and the broad path_axis
# COMPONENT rec (for the mm-axis fold + hard-fail signals) for these two
# subjects only. Remove when the full-cohort broad re-derivation lands.
# NOTE the broad path_axis component file IS the path_axis dict (its fields sit
# at top level, not nested under a "path_axis" key) -> load_geo_axis branches.
BROAD_PATCH = {("yuquan", "huangwanling"), ("yuquan", "zhaojinrui")}
BROAD_MASKED_DIR = os.path.join(
    REPO, "results/interictal_propagation_masked_broad/per_subject"
)
BROAD_GEO_COMPONENT_DIR = os.path.join(
    REPO, "results/spatial_modulation/propagation_geometry_broad/"
    "components/path_axis/per_subject"
)

# Both-fold continuous-feature gate (plan §A5). Stricter than the template
# forward_reverse_reproduced OR-rule: this requires AND across both folds.
RELIABILITY_THRESHOLD = 0.6

# The continuous features audited this round, in report order.
FEATURES = ("axis_dir", "onset_jitter", "pathway_width")


def load_coverage():
    """Return {(dataset, subject): coverage_row} keyed for availability lookup."""
    with open(COVERAGE_JSON) as f:
        cov = json.load(f)
    out = OrderedDict()
    for r in cov["per_subject"]:
        out[(r["dataset"], r["subject"])] = r
    return out


def load_fold_corr(dataset, subject):
    """Pull both folds' rank-space mean_match_corr from the masked JSON.

    Returns (split_half, odd_even, n_events_min) where each corr is the
    concordance of cluster_rank_a vs cluster_rank_b_matched_to_a for that fold;
    any of split_half / odd_even may be None if the fold or the field is absent.
    n_events_min is the smaller per-fold event count usable for reconciliation.
    """
    base = (BROAD_MASKED_DIR if (dataset, subject) in BROAD_PATCH
            else MASKED_DIR)
    path = os.path.join(base, f"{dataset}_{subject}.json")
    if not os.path.isfile(path):
        return None, None, None
    with open(path) as f:
        d = json.load(f)
    tsr = d.get("time_split_reproducibility") or {}
    splits = tsr.get("splits") or {}
    sh = splits.get("first_half_second_half") or {}
    oe = splits.get("odd_even_block") or {}
    split_half = sh.get("mean_match_corr")
    odd_even = oe.get("mean_match_corr")
    n_events = [
        v for v in (
            sh.get("n_events_a"), sh.get("n_events_b"),
            oe.get("n_events_a"), oe.get("n_events_b"),
        ) if isinstance(v, (int, float))
    ]
    n_events_min = min(n_events) if n_events else None
    return split_half, odd_even, n_events_min


def load_geo_axis(dataset, subject):
    """Pull mm-axis fold material + hard-fail signals from propagation_geometry.

    Returns dict with spearman_rho (secondary mm-axis fold check), weak_axis,
    axis_length_mm, degenerate (axis_length_mm is None on an ok-status axis),
    perp_width_measurable. Missing file => all-None / degenerate-unknown.
    """
    is_broad = (dataset, subject) in BROAD_PATCH
    base = BROAD_GEO_COMPONENT_DIR if is_broad else GEO_DIR
    path = os.path.join(base, f"{dataset}_{subject}.json")
    if not os.path.isfile(path):
        return {
            "spearman_rho": None, "weak_axis": None, "axis_length_mm": None,
            "degenerate": None, "perp_width_measurable": None,
            "status": None,
        }
    with open(path) as f:
        d = json.load(f)
    # The broad component file IS the path_axis dict (flat); the canonical
    # per_subject file nests it under "path_axis".
    pa = d if is_broad else (d.get("path_axis", {}) or {})
    shv = pa.get("split_half_validation") or {}
    status = pa.get("status")
    axis_len = pa.get("axis_length_mm")
    # degenerate = ok status but no usable mm vector (axis_length_mm is None),
    # matching the coverage artifact's degenerate_axis tier exactly.
    degenerate = (status == "ok") and (axis_len is None)
    return {
        "spearman_rho": shv.get("spearman_rho"),
        "weak_axis": pa.get("weak_axis") is True,
        "axis_length_mm": axis_len,
        "degenerate": degenerate,
        "perp_width_measurable": pa.get("perp_width_measurable") is True,
        "status": status,
    }


def both_fold_pass(split_half, odd_even):
    """The §A5 AND-gate: both folds present AND each >= threshold."""
    if split_half is None or odd_even is None:
        return False
    return (split_half >= RELIABILITY_THRESHOLD
            and odd_even >= RELIABILITY_THRESHOLD)


def build_rows():
    coverage = load_coverage()
    rows = []
    for (dataset, subject), cov in coverage.items():
        split_half, odd_even, _ = load_fold_corr(dataset, subject)
        geo = load_geo_axis(dataset, subject)
        spearman_rho = geo["spearman_rho"]
        weak_axis = geo["weak_axis"] is True
        degenerate = geo["degenerate"] is True

        # ---- axis_dir (rank-space DIRECTION) ----
        # HARD FAIL if weak_axis OR degenerate (axis_length_mm is None),
        # regardless of the rank-space corr values.
        axis_hard_fail = weak_axis or degenerate
        axis_available = bool(cov.get("axis_available"))
        if not axis_available:
            axis_pass = False
            axis_note = (
                "axis layer unavailable in coverage "
                f"(exclusion_reason={cov.get('exclusion_reason') or 'n/a'}); "
                "rank-space folds not gated for axis_dir"
            )
        elif axis_hard_fail:
            axis_pass = False
            why = "weak_axis" if weak_axis else "degenerate_axis"
            axis_note = (
                f"HARD FAIL ({why}): gated out regardless of fold corr "
                f"(split_half={_fmt(split_half)}, odd_even={_fmt(odd_even)})"
            )
        else:
            axis_pass = both_fold_pass(split_half, odd_even)
            axis_note = (
                "rank-space DIRECTION via mean_match_corr both folds; "
                "mm-axis spearman_rho carried as secondary fold check"
            )
        rows.append(_row(
            dataset, subject, "axis_dir", split_half, odd_even,
            spearman_rho, weak_axis, degenerate, axis_pass, axis_note,
        ))

        # ---- onset_jitter (rank/entry layer) — PROXY ----
        # No per-fold earliest-entry statistic exists in the masked JSON;
        # use the same rank-space fold concordance as the honest proxy.
        oj_note = (
            "onset_jitter reliability uses cluster_rank fold concordance "
            "(mean_match_corr both folds) as a proxy; a per-fold earliest-prob "
            "recompute is deferred (not stored in current readout)"
        )
        oj_pass = both_fold_pass(split_half, odd_even)
        rows.append(_row(
            dataset, subject, "onset_jitter", split_half, odd_even,
            spearman_rho, weak_axis, degenerate, oj_pass, oj_note,
        ))

        # ---- pathway_width — DEFERRED ----
        # path_axis carries only FULL-DATA perp_spread, not per-fold; cannot
        # decide reliability this round. Record which subjects COULD support a
        # later fold recompute (perp_width_measurable AND coords present).
        pw_available = bool(cov.get("pathway_width_available"))
        could_later = (
            geo["perp_width_measurable"] is True
            and cov.get("coord_space") not in (None, "missing")
        )
        pw_note = (
            "requires_fold_recompute (deferred): path_axis stores only "
            "full-data perp_spread, no per-fold split; "
            + ("CAN support fold recompute later (perp_width_measurable AND "
               "coords present)" if could_later
               else "cannot support fold recompute (perp width unmeasurable "
                    "or coords missing)")
        )
        rows.append(_row(
            dataset, subject, "pathway_width",
            # No per-fold continuous corr for width this round.
            None, None, spearman_rho, weak_axis, degenerate,
            "deferred", pw_note,
            extra_pw_available=pw_available,
            extra_could_recompute=could_later,
        ))

    rows.sort(key=lambda r: (r["dataset"], r["subject"], _feat_order(r["feature"])))
    return rows


def _feat_order(feat):
    return FEATURES.index(feat) if feat in FEATURES else 99


def _fmt(v):
    return "None" if v is None else f"{v:.4f}"


def _row(dataset, subject, feature, split_half, odd_even, spearman_rho,
         weak_axis, degenerate, reliability_pass, note,
         extra_pw_available=None, extra_could_recompute=None):
    r = OrderedDict([
        ("dataset", dataset),
        ("subject", subject),
        ("feature", feature),
        ("split_half", split_half),
        ("odd_even", odd_even),
        ("mm_spearman_rho", spearman_rho),
        ("weak_axis", weak_axis),
        ("degenerate", degenerate),
        ("reliability_pass", reliability_pass),
        ("note", note),
    ])
    # JSON-only provenance for the deferred pathway_width layer.
    if extra_pw_available is not None:
        r["_pathway_width_available"] = extra_pw_available
    if extra_could_recompute is not None:
        r["_could_recompute_later"] = extra_could_recompute
    return r


def feature_rollup(rows, feature):
    """Per-feature reliability rollup, split by dataset.

    For continuous features: count of reliability_pass == True (and the
    complementary fail count), each split by dataset, reported on the rows
    where the feature was audited. For pathway_width: count of 'deferred'.
    """
    feat_rows = [r for r in rows if r["feature"] == feature]
    pass_by_dataset = {}
    fail_by_dataset = {}
    deferred_by_dataset = {}
    n_with_fold_data = 0
    for r in feat_rows:
        ds = r["dataset"]
        rp = r["reliability_pass"]
        has_folds = r["split_half"] is not None and r["odd_even"] is not None
        if has_folds:
            n_with_fold_data += 1
        if rp is True:
            pass_by_dataset[ds] = pass_by_dataset.get(ds, 0) + 1
        elif rp is False:
            fail_by_dataset[ds] = fail_by_dataset.get(ds, 0) + 1
        elif rp == "deferred":
            deferred_by_dataset[ds] = deferred_by_dataset.get(ds, 0) + 1

    roll = OrderedDict([
        ("n_subjects_audited", len(feat_rows)),
        ("n_with_usable_fold_data", n_with_fold_data),
        ("n_pass", sum(pass_by_dataset.values())),
        ("n_fail", sum(fail_by_dataset.values())),
        ("n_deferred", sum(deferred_by_dataset.values())),
        ("pass_by_dataset", pass_by_dataset),
        ("fail_by_dataset", fail_by_dataset),
        ("deferred_by_dataset", deferred_by_dataset),
    ])
    return roll


def find_disagreements(rows):
    """Subjects where the two folds disagree strongly (one >=0.6, other <0.6).

    These are the interesting borderline cases the both-fold AND-gate catches
    that a single-fold rule would miss. Reported only for continuous features
    where both folds are present (pathway_width has no folds this round).
    """
    out = []
    for r in rows:
        if r["feature"] == "pathway_width":
            continue
        sh, oe = r["split_half"], r["odd_even"]
        if sh is None or oe is None:
            continue
        hi = max(sh, oe)
        lo = min(sh, oe)
        if hi >= RELIABILITY_THRESHOLD and lo < RELIABILITY_THRESHOLD:
            out.append(OrderedDict([
                ("dataset", r["dataset"]),
                ("subject", r["subject"]),
                ("feature", r["feature"]),
                ("split_half", sh),
                ("odd_even", oe),
            ]))
    return out


def build_reconciliation(rows):
    """How many subjects had usable fold data per feature (for traceability)."""
    recon = OrderedDict()
    for feat in FEATURES:
        feat_rows = [r for r in rows if r["feature"] == feat]
        n_audited = len(feat_rows)
        n_folds = sum(
            1 for r in feat_rows
            if r["split_half"] is not None and r["odd_even"] is not None
        )
        recon[feat] = OrderedDict([
            ("n_subjects_audited", n_audited),
            ("n_with_usable_fold_data", n_folds),
        ])
    recon["note"] = (
        "n_subjects_audited == 40 for every feature because the audit walks "
        "every coverage row; n_with_usable_fold_data is how many of those have "
        "both rank-space folds present in the masked JSON. pathway_width has 0 "
        "fold data this round (no per-fold perp_spread exists) -> all deferred."
    )
    return recon


def main():
    rows = build_rows()
    os.makedirs(OUT_DIR, exist_ok=True)

    # CSV (drop the JSON-only underscore provenance fields).
    csv_path = os.path.join(OUT_DIR, "real_feature_reliability.csv")
    csv_fields = [
        "dataset", "subject", "feature", "split_half", "odd_even",
        "mm_spearman_rho", "weak_axis", "degenerate", "reliability_pass",
        "note",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for r in rows:
            out = {}
            for k in csv_fields:
                v = r.get(k)
                if isinstance(v, bool):
                    out[k] = "true" if v else "false"
                elif v is None:
                    out[k] = ""
                else:
                    out[k] = v
            w.writerow(out)

    disagreements = find_disagreements(rows)

    payload = OrderedDict([
        ("schema_version", "topic4_sef_hfo_axisA_real_feature_reliability_v1"),
        ("topic", "topic4_sef_hfo"),
        ("scope",
         "Axis-A A5-real per-subject feature-RELIABILITY audit (plan §A5). "
         "Split-half AND odd-even reliability of the primary real-data "
         "fingerprint features; gates which subjects/features enter A5 "
         "placement. Descriptive only: NO mechanism label, NO scientific "
         "claim. mm spearman_rho reported per subject, never pooled across "
         "datasets."),
        ("gate_rule", OrderedDict([
            ("threshold", RELIABILITY_THRESHOLD),
            ("rule",
             "A continuous feature (axis_dir, onset_jitter) is admitted only "
             "if split_half >= 0.6 AND odd_even >= 0.6 (BOTH folds)."),
            ("distinction_from_template_or_rule",
             "Stricter than the template-level forward_reverse_reproduced "
             "OR-rule (split-half OR odd-even). This A5 placement gate uses "
             "AND across both folds, not OR."),
            ("axis_dir_hard_fail",
             "axis_dir additionally HARD FAILs (gated out) if weak_axis is "
             "True OR axis_length_mm is None (degenerate), regardless of the "
             "fold corr values."),
        ])),
        ("pathway_width_deferred_note",
         "pathway_width reliability is DEFERRED this round: path_axis stores "
         "only full-data perp_spread, not a per-fold split, so a split-half / "
         "odd-even reliability cannot be computed without a fold recompute. "
         "This honest gap is the correct outcome this round. Subjects that "
         "COULD support a later fold recompute (perp_width_measurable AND "
         "coords present) are flagged via _could_recompute_later in "
         "per_subject."),
        ("onset_jitter_proxy_note",
         "onset_jitter reliability uses the cluster_rank fold concordance "
         "(mean_match_corr in both folds) as a proxy; a per-fold earliest-prob "
         "recompute is deferred (the current readout stores rank only, not "
         "per-fold earliest-entry probability). No separate number is "
         "fabricated."),
        ("mm_coverage_discipline",
         "mm-axis spearman_rho is reported per subject and per dataset "
         "(epilepsiae=mni152_1mm, yuquan=fs_native_ras_mm); mm values are "
         "NEVER pooled across datasets."),
        ("n_subjects", len({(r["dataset"], r["subject"]) for r in rows})),
        ("per_feature_reliability", OrderedDict([
            (feat, feature_rollup(rows, feat)) for feat in FEATURES
        ])),
        ("fold_disagreements",
         OrderedDict([
             ("description",
              "subjects where the two folds disagree strongly (one >= 0.6, "
              "other < 0.6) for a continuous feature; these borderline cases "
              "are caught by the both-fold AND-gate but would pass a "
              "single-fold rule"),
             ("n", len(disagreements)),
             ("rows", disagreements),
         ])),
        ("reconciliation", build_reconciliation(rows)),
        ("per_subject", rows),
    ])
    json_path = os.path.join(OUT_DIR, "real_feature_reliability.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("wrote", csv_path)
    print("wrote", json_path)
    print("n_subjects:", payload["n_subjects"])
    for feat in FEATURES:
        roll = payload["per_feature_reliability"][feat]
        print(f"  {feat}: pass={roll['n_pass']} fail={roll['n_fail']} "
              f"deferred={roll['n_deferred']} "
              f"(fold_data={roll['n_with_usable_fold_data']}/"
              f"{roll['n_subjects_audited']}) "
              f"pass_by_dataset={dict(roll['pass_by_dataset'])}")
    print("fold_disagreements:", len(disagreements))
    for d in disagreements:
        print(f"  {d['dataset']}_{d['subject']} {d['feature']}: "
              f"split_half={d['split_half']:.4f} odd_even={d['odd_even']:.4f}")


if __name__ == "__main__":
    main()
