"""Stage 1 report (spec 7.5).

Exploratory posture: this produces NUMBERS and a recommendation, not an automatic
gate. Only data integrity fails closed -- that is a correctness guard, not a
science gate.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import binomtest

SEEDS = tuple(range(1, 13))
SIM_ARMS = ("manual_hard", "manual_projected", "manual_smooth", "uniform_axial",
            "width_wide", "width_narrow", "transverse_plus", "transverse_minus")
# Arms the optimiser will navigate: same pipeline, real noise.
PROJECTED_ARMS = ("manual_smooth", "uniform_axial", "width_wide",
                  "width_narrow", "transverse_mean")
# No event-gate axis: gate=4 admits zero signed events.
SCORE_KEYS = tuple(itertools.product(("gradient", "geometry"),
                                     ("mean_rank", "common_only"),
                                     ("spearman", "pair")))
PRIMARY_KEY = ("gradient", "mean_rank", "spearman")

COMPARISONS = (
    dict(name="A", a="manual_projected", b="manual_hard", group="equivalence",
         purpose="sampling contract"),
    dict(name="A2", a="manual_smooth", b="manual_projected", group="equivalence",
         purpose="hard mask vs smoothed field"),
    dict(name="B1", a="manual_smooth", b="uniform_axial", group="shape",
         purpose="longitudinal shape"),
    dict(name="B2", a="manual_smooth", b="width_wide", group="shape",
         purpose="transverse width, flattened"),
    dict(name="B3", a="manual_smooth", b="width_narrow", group="shape",
         purpose="transverse width, elongated"),
    dict(name="B4", a="manual_smooth", b="transverse_mean", group="shape",
         purpose="transverse position"),
    dict(name="C", a="manual_smooth", b="axis_only", group="geometry",
         purpose="pure-geometry reference"),
)
COVERAGE_MARGIN = 0.10


def arm_value(runs, arm, seed, key, field):
    if arm == "transverse_mean":
        return float(np.mean([runs[("transverse_plus", seed) + key][field],
                              runs[("transverse_minus", seed) + key][field]]))
    return runs[(arm, seed) + key][field]


def _arm_n_dir(runs, arm, seed, key):
    if arm == "transverse_mean":
        return min(runs[("transverse_plus", seed) + key]["n_dir"],
                   runs[("transverse_minus", seed) + key]["n_dir"])
    return runs[(arm, seed) + key]["n_dir"]


def tiered_paired_stats(pairs):
    """pairs: iterable of (n_dir_a, S_a, n_dir_b, S_b).

    Only seeds where both arms sit in the SAME direction tier contribute a
    numeric difference. Seeds where the tiers differ are counted as wins and
    losses: differencing across tiers is what spec 5.3 forbids.
    """
    deltas, wins, losses = [], 0, 0
    for nda, sa, ndb, sb in pairs:
        if nda == ndb:
            if np.isfinite(sa) and np.isfinite(sb):
                deltas.append(float(sa) - float(sb))
        elif nda > ndb:
            wins += 1
        else:
            losses += 1
    d = np.asarray(deltas, float)
    out = dict(n_same_tier=int(d.size), tier_wins=int(wins), tier_losses=int(losses),
               mean=float(d.mean()) if d.size else float("nan"),
               sd=float(d.std(ddof=1)) if d.size > 1 else float("nan"))
    if d.size > 1:
        se = out["sd"] / np.sqrt(d.size)
        out["ci_low"], out["ci_high"] = out["mean"] - 1.96 * se, out["mean"] + 1.96 * se
    else:
        out["ci_low"] = out["ci_high"] = float("nan")
    nz = d[d != 0.0]
    if nz.size:
        sign = np.sign(out["mean"]) if out["mean"] != 0 else 1.0
        n_same = int((np.sign(nz) == sign).sum())
        out["n_same"] = n_same
        out["p_uncorrected"] = float(
            binomtest(n_same, int(nz.size), 0.5, alternative="two-sided").pvalue)
    else:
        out["n_same"], out["p_uncorrected"] = 0, float("nan")
    return out


def concordance(runs, key):
    """Cross-seed ordering consistency among the projected arms. Diagnostic only:
    how well a single seed would order candidates for CMA-ES."""
    hits = []
    for a, b in itertools.combinations(PROJECTED_ARMS, 2):
        deltas = {}
        for s in SEEDS:
            if _arm_n_dir(runs, a, s, key) != _arm_n_dir(runs, b, s, key):
                continue
            va = arm_value(runs, a, s, key, "S_rank")
            vb = arm_value(runs, b, s, key, "S_rank")
            if np.isfinite(va) and np.isfinite(vb):
                deltas[s] = va - vb
        if len(deltas) < 2:
            continue
        pooled = np.sign(np.mean(list(deltas.values()))) or 1.0
        hits.extend(1.0 if (np.sign(v) or 1.0) == pooled else 0.0 for v in deltas.values())
    return float(np.mean(hits)) if hits else float("nan")


def stage1_report(runs, config):
    # --- integrity: the ONLY fail-closed path -----------------------------
    for arm in SIM_ARMS:
        for seed in SEEDS:
            for key in SCORE_KEYS:
                cell = runs.get((arm, seed) + key)
                if cell is None:
                    return dict(integrity=dict(status="FAIL_CLOSED",
                                               reason=f"missing cell {(arm, seed) + key}"))
                if cell["n_dir"] >= 2 and not np.isfinite(cell["S_rank"]):
                    return dict(integrity=dict(
                        status="FAIL_CLOSED",
                        reason=f"non-finite S_rank with n_dir=2 at {(arm, seed) + key}"))
    integrity = dict(status="ok", checksum=config.get("checksum"))

    uninformative = [s for s in SEEDS
                     if sum(1 for a in SIM_ARMS
                            if runs[(a, s) + PRIMARY_KEY]["n_dir"] == 0) >= 2]

    comparisons = {}
    for key in SCORE_KEYS:
        for comp in COMPARISONS:
            if comp["b"] == "axis_only":
                continue                       # filled by the analysis script
            pairs = [(_arm_n_dir(runs, comp["a"], s, key),
                      arm_value(runs, comp["a"], s, key, "S_rank"),
                      _arm_n_dir(runs, comp["b"], s, key),
                      arm_value(runs, comp["b"], s, key, "S_rank")) for s in SEEDS]
            comparisons[(comp["name"],) + key] = dict(
                group=comp["group"], purpose=comp["purpose"], **tiered_paired_stats(pairs))

    delta_eq = float(config.get("delta_eq", 0.05))
    equivalence = {}
    for name in ("A", "A2"):
        st = comparisons[(name,) + PRIMARY_KEY]
        inside = (np.isfinite(st["ci_low"]) and np.isfinite(st["ci_high"])
                  and st["ci_low"] > -delta_eq and st["ci_high"] < delta_eq)
        equivalence[name] = dict(delta_eq=delta_eq, equivalent=bool(inside), **st)

    shape = {n: comparisons[(n,) + PRIMARY_KEY] for n in ("B1", "B2", "B3", "B4")}
    separates = [n for n, st in shape.items()
                 if np.isfinite(st["p_uncorrected"]) and st["p_uncorrected"] < 0.05]

    cov, low = {}, []
    ref = float(np.mean([min(runs[("manual_smooth", s) + PRIMARY_KEY]["coverage_forward"],
                             runs[("manual_smooth", s) + PRIMARY_KEY]["coverage_reverse"])
                         for s in SEEDS]))
    for arm in SIM_ARMS:
        per_dir = float(np.mean([min(runs[(arm, s) + PRIMARY_KEY]["coverage_forward"],
                                     runs[(arm, s) + PRIMARY_KEY]["coverage_reverse"])
                                 for s in SEEDS]))
        cov[arm] = per_dir
        if per_dir < ref - COVERAGE_MARGIN:
            low.append(arm)

    return dict(
        integrity=integrity,
        scorable=dict(uninformative_seeds=uninformative, n_seeds=len(SEEDS)),
        equivalence=equivalence,
        shape=shape,
        comparisons={"|".join(map(str, k)): v for k, v in comparisons.items()},
        concordance={"|".join(k): concordance(runs, k) for k in SCORE_KEYS},
        coverage=dict(per_arm=cov, reference=ref, low_coverage_arms=low,
                      margin=COVERAGE_MARGIN),
        sensitivities_reported_separately=dict(
            TEMPLATE_SOURCE=sorted({k[0] for k in SCORE_KEYS}),
            SCORER_SENSITIVITY=sorted({k[1] for k in SCORE_KEYS}),
            SCORE_DEFINITION=sorted({k[2] for k in SCORE_KEYS}),
        ),
        recommendation=dict(
            shape_separates=bool(separates),
            separating_dimensions=separates,
            note=("uncorrected p across 4 shape comparisons; exploratory posture, "
                  "no multiplicity correction and no automatic gate"),
        ),
    )
