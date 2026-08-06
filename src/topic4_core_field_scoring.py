"""Frozen-support scoring for the data-driven core field (spec section 5).

Reimplements the definitions in
scripts/paper_figures/plot_fig_subject_snn_realvsmodel.py (templates from event
sign; 2x2 Spearman against the patient's two templates) and adds the frozen
scoring support that file cannot provide. That file is NOT modified -- it carries
published numbers.
"""
from __future__ import annotations

import itertools
import json
import os

import numpy as np
from scipy.stats import spearmanr

GRADIENT_ROOT = "results/interictal_propagation_masked/template_gradient_fields/per_subject"
GEOMETRY_ROOT = "results/spatial_modulation/propagation_geometry/observation_readout/real_subjects"
PART_MIN = 5   # 2*k_dir+1 with k_dir=2; endpoint_centroid_axis returns None below this,
               # so a post-hoc gate of 4 admits nothing and is not an axis.


def load_patient_templates(subject, source, root="."):
    """Patient TA/TB contact ranks. Two non-identical sources, both reported."""
    if source == "gradient":
        field = json.load(open(os.path.join(
            root, GRADIENT_ROOT, f"{subject}.json")))["interictal_field"]
        names = [str(x) for x in field["contact_order"]]
        return {tpl: {n: float(v) for n, v in zip(names, np.asarray(field[key], float))
                      if np.isfinite(v)}
                for key, tpl in (("rank_a", "t_a"), ("rank_b", "t_b"))}
    if source == "geometry":
        out = {}
        for tpl in ("t_a", "t_b"):
            g = json.load(open(os.path.join(root, GEOMETRY_ROOT, f"{subject}_{tpl}.json")))
            out[tpl] = {c["name"]: float(c["typical_rank"]) for c in g["channels"]
                        if c.get("typical_rank") is not None}
        return out
    raise ValueError(f"unknown template source {source!r}")


def model_templates(events, support, part_min=PART_MIN):
    """Forward/reverse mean within-event rank on the frozen support.

    Templates come from event SIGN, not cluster labels: for a one-direction
    readout a cluster->direction mapping would be fabricated.
    """
    support = list(support)
    idx = {n: i for i, n in enumerate(support)}
    usable = [e for e in events
              if e.get("sign") is not None and int(e.get("n_part", 0)) >= int(part_min)]
    acc = {+1: [[] for _ in support], -1: [[] for _ in support]}
    for e in usable:
        key = +1 if e["sign"] > 0 else -1
        for name, v in (e.get("ranks") or {}).items():
            if v is not None and name in idx:
                acc[key][idx[name]].append(float(v))
    out = {}
    for key, label in ((+1, "forward"), (-1, "reverse")):
        tpl = {support[i]: float(np.mean(vals)) for i, vals in enumerate(acc[key]) if vals}
        out[label] = tpl
        out[f"coverage_{label}"] = len(tpl) / len(support) if support else 0.0
    out["n_dir"] = int(bool(out["forward"])) + int(bool(out["reverse"]))
    union = set(out["forward"]) | set(out["reverse"])
    out["coverage_union"] = len(union) / len(support) if support else 0.0
    out["mean_n_part"] = float(np.mean([e["n_part"] for e in usable])) if usable else 0.0
    return out


def _aligned(model_tpl, target_tpl, support, missing_rule):
    """Vectors on the FULL frozen support.

    'mean_rank' fills a contact the candidate never recruited with that
    direction's mean -- an explicit modelling assumption, which is why
    balanced_pair_score and adversarial_gain are reported alongside.
    'common_only' is the legacy candidate-dependent support: regression and
    sensitivity only, never load-bearing.
    """
    names = [n for n in support if n in target_tpl]
    if missing_rule == "common_only":
        names = [n for n in names if n in model_tpl]
        if len(names) < 4:
            return None, None
        return (np.array([model_tpl[n] for n in names]),
                np.array([target_tpl[n] for n in names]))
    if missing_rule != "mean_rank":
        raise ValueError(missing_rule)
    if not model_tpl or len(names) < 4:
        return None, None
    fill = float(np.mean(list(model_tpl.values())))
    return (np.array([model_tpl.get(n, fill) for n in names]),
            np.array([target_tpl[n] for n in names]))


def sim_matrix(model, target, support, missing_rule):
    """2x2 Spearman: rows model forward/reverse, cols patient t_a/t_b."""
    M = np.full((2, 2), np.nan)
    for i, row in enumerate(("forward", "reverse")):
        for j, col in enumerate(("t_a", "t_b")):
            a, b = _aligned(model.get(row, {}), target[col], support, missing_rule)
            if a is None or np.ptp(a) == 0 or np.ptp(b) == 0:
                continue
            M[i, j] = float(spearmanr(a, b).correlation)
    return M


def assignment_invariant_S(M):
    """max over the two TA/TB assignments of the diagonal mean.

    NaN when no full assignment exists. Deliberately NOT a best-single-cell
    fallback: such a value would invite differencing a one-direction arm against
    a two-direction one, which spec 5.3 forbids.
    """
    opts = [0.5 * (M[i, j] + M[k, l])
            for (i, j), (k, l) in (((0, 0), (1, 1)), ((0, 1), (1, 0)))
            if np.isfinite(M[i, j]) and np.isfinite(M[k, l])]
    return float(max(opts)) if opts else float("nan")


def _directed_pair(model_tpl, target_tpl, support):
    """Pairwise concordance with a FIXED denominator: every pair of the frozen
    support counts, and a pair touching an unrecruited contact contributes 0."""
    pairs = list(itertools.combinations(support, 2))
    if not pairs:
        return float("nan")
    tot = sum(np.sign((model_tpl[a] - model_tpl[b]) * (target_tpl[a] - target_tpl[b]))
              for a, b in pairs
              if a in model_tpl and b in model_tpl and a in target_tpl and b in target_tpl)
    return float(tot / len(pairs))


def balanced_pair_score(model, target, support):
    """Bidirectional, assignment-invariant, fixed-denominator pair score.

    NaN unless both directions exist, for the same reason as
    assignment_invariant_S.
    """
    support = list(support)
    if model.get("n_dir", 0) < 2:
        return float("nan")
    opts = [0.5 * (_directed_pair(model["forward"], target[a_col], support)
                   + _directed_pair(model["reverse"], target[b_col], support))
            for a_col, b_col in (("t_a", "t_b"), ("t_b", "t_a"))]
    return float(max(opts))


def axis_only_templates(names, coords, center, u_axis):
    """A model with NO pathology field: contacts ordered by axial projection.

    Pure geometry already scores 0.696 against this patient's templates, so this
    is the reference every claim has to beat (spec 2.4 / 5.4).
    """
    u = np.asarray(u_axis, float)
    u = u / np.linalg.norm(u)
    proj = (np.asarray(coords, float) - np.asarray(center, float)[None, :]) @ u
    return {"forward": {n: float(p) for n, p in zip(names, proj)},
            "reverse": {n: float(-p) for n, p in zip(names, proj)},
            "n_dir": 2, "coverage_forward": 1.0, "coverage_reverse": 1.0,
            "coverage_union": 1.0, "mean_n_part": float(len(names))}


def adversarial_gain(model, target, support, missing_rule):
    """How much could this candidate gain by dropping its worst-matching contact?

    Reported, not asserted. Under mean-rank filling a badly ranked contact can be
    worth dropping, so the size of that incentive has to be visible rather than
    assumed away (third-review P0-4).
    """
    base = assignment_invariant_S(sim_matrix(model, target, support, missing_rule))
    best_gain, worst = 0.0, None
    for name in support:
        trimmed = dict(model)
        trimmed["forward"] = {k: v for k, v in model.get("forward", {}).items() if k != name}
        trimmed["reverse"] = {k: v for k, v in model.get("reverse", {}).items() if k != name}
        s = assignment_invariant_S(sim_matrix(trimmed, target, support, missing_rule))
        if np.isfinite(s) and np.isfinite(base) and s - base > best_gain:
            best_gain, worst = float(s - base), name
    return dict(base=float(base), gain=float(best_gain), worst_contact=worst)


def candidate_key(n_dir, s_rank):
    """Lexicographic fitness key: (n_dir, S_rank), larger is better.

    CMA-ES consumes candidate ORDER, so the tiers separate without inventing a
    rate-loss weight. Never compare S_rank across tiers (spec 5.3).
    """
    s = float(s_rank)
    return (int(n_dir), s if np.isfinite(s) else -np.inf)


def coverage_matched_axis_only(model, axial_projection, support=None):
    """axis-only templates restricted to EXACTLY the contacts `model` recruited.

    The unmatched reference (axis_only_templates) covers every contact by
    construction, while a simulated arm recruits 50-70% of them. Under
    frozen-support mean-rank filling that difference dominates, so the two are not
    comparable: the arm is penalised for sparse recruitment rather than for the
    quality of its ordering. Matching the coverage isolates the ordering.

    Returns None when the model lacks a direction -- there is nothing to match.
    """
    if model.get("n_dir", 0) < 2:
        return None
    fwd = {n: float(axial_projection[n]) for n in model["forward"] if n in axial_projection}
    rev = {n: -float(axial_projection[n]) for n in model["reverse"] if n in axial_projection}
    out = {"forward": fwd, "reverse": rev, "n_dir": 2}
    n_sup = len(support) if support else max(len(fwd), len(rev), 1)
    out["coverage_forward"] = len(fwd) / n_sup
    out["coverage_reverse"] = len(rev) / n_sup
    out["coverage_union"] = len(set(fwd) | set(rev)) / n_sup
    out["mean_n_part"] = float(np.mean([len(fwd), len(rev)]))
    return out
