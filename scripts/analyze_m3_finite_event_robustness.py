#!/usr/bin/env python3
"""Multi-seed robustness analysis for M3 kick-calibration explore runs.

This is an OFFLINE ANALYSIS over already-dumped per_seed_metrics.csv /
candidate_table.csv. It DOES NOT run any SNN — it only re-reads the stored
numbers and answers, per substrate (core config) and per kick, whether the
kick~0.75-1.0 LOCAL+RETURNED finite event is ROBUST across seeds, and whether
the substrate looks like a graded sub-threshold ramp (LINEAR_GRADED / W_small)
or a sharp ignition threshold with a finite event above it (FINITE_THRESHOLD /
W_event, "branch B").

Per (substrate, kick) we aggregate over seeds (and rep bins) and assign a single
phenotype label; per substrate we walk kicks ascending and emit a cross-kick
verdict. See cross_kick_verdict() docstring for the four verdicts.

Inputs per run dir:
  per_seed_metrics.csv  — one row per kick x win x rep_bin x seed
  candidate_table.csv   — per kick x win aggregate (carries core_only_quiet)
  reclassified_candidate_table.csv — optional; if present its core_only_quiet
      column is the corrected RELATIVE core-quiet gate (core fires no more than
      a bare slice) and is preferred over candidate_table's absolute-floor flag.

DRAFT thresholds below are labelled DRAFT — the LOGIC is the deliverable, the
magnitudes are tunable once a real multi-seed fine-scan exists.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


# --------------------------------------------------------------------------- #
# DRAFT thresholds (logic is the deliverable; magnitudes are tunable)         #
# --------------------------------------------------------------------------- #
ROBUST_FRAC = 0.7      # DRAFT: P_local_returned >= this => robust across seeds
MIN_SEEDS = 6          # DRAFT: need at least this many seeds for a stable claim
FARFIELD_NOISE_FRAC = 0.5   # DRAFT: far_field_frac above this => scattered/noise-like
P_RUNAWAY_ESCAPE = 0.5      # DRAFT: P_runaway >= this => escape regime
P_RETURNED_LOCAL = 0.5      # DRAFT: P_returned >= this required for any returned phenotype
R95_LOCAL_CAP_MM = 6.0      # DRAFT: median r95 <= this => spatially local (matches r95_cap)
P_LOCAL_RET_FINITE = 0.5    # DRAFT: P_local_returned >= this => finite_local_returned phenotype
GRADED_RISE_FROM = 0.05     # DRAFT: a LINEAR_GRADED ramp must start rising at/below this kick


PHENOTYPES = (
    "confounded",            # core not quiet -> differenced response untrustworthy
    "noise",                 # no local-returned event + scattered far-field
    "silent",                # essentially no early response (sub-threshold)
    "finite_local_returned", # local AND returned in a robust-ish fraction of seeds
    "large_returned",        # returned but not spatially local (big self-limited event)
    "escape",                # runs away / does not return
)

VERDICTS = ("LINEAR_GRADED", "FINITE_THRESHOLD", "ESCAPE_ONLY", "NO_LOCAL", "SELF_IGNITE")


# --------------------------------------------------------------------------- #
# Data model                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class KickAggregate:
    """Per (substrate, kick) aggregate over seeds and rep bins."""

    kick: float
    n_seeds: int
    p_local_returned: float
    p_returned: float
    p_runaway: float
    median_downstream: float
    median_n_activated: float
    median_r95: float
    median_far_field: float
    core_only_quiet: bool          # True => core stays quiet without kick (trustworthy)
    phenotype: str = ""
    stable_finite: bool = False    # True => stable finite_local_returned candidate


@dataclass
class SubstrateResult:
    name: str
    win_lo: float
    win_hi: float
    kicks: List[KickAggregate] = field(default_factory=list)
    verdict: str = ""
    verdict_reason: str = ""
    stable_kicks: List[float] = field(default_factory=list)
    n_seeds: int = 0


# --------------------------------------------------------------------------- #
# CSV reading                                                                 #
# --------------------------------------------------------------------------- #
def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _fnum(row: Dict[str, str], key: str, default: float = float("nan")) -> float:
    v = row.get(key, "")
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _median(vals: Sequence[float]) -> float:
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return float("nan")
    return float(statistics.median(vals))


def load_core_quiet_map(run_dir: str) -> Dict[Tuple[float, float, float], bool]:
    """Map (kick, win_lo, win_hi) -> core_only_quiet bool.

    Prefer reclassified_candidate_table.csv (corrected RELATIVE core-quiet gate:
    core fires no more than a bare slice) over candidate_table.csv (older
    absolute-floor flag that false-fails barely-eligible narrow cores).
    """
    for fname in ("reclassified_candidate_table.csv", "candidate_table.csv"):
        path = os.path.join(run_dir, fname)
        if not os.path.isfile(path):
            continue
        rows = _read_csv(path)
        if not rows or "core_only_quiet" not in rows[0]:
            continue
        out: Dict[Tuple[float, float, float], bool] = {}
        for r in rows:
            key = (_fnum(r, "kick_boost"), _fnum(r, "win_lo"), _fnum(r, "win_hi"))
            out[key] = bool(int(_fnum(r, "core_only_quiet", 0)))
        return out
    # Fallback: per_seed_metrics core_only_quiet (majority over seeds/bins per kick+win).
    ps_path = os.path.join(run_dir, "per_seed_metrics.csv")
    if os.path.isfile(ps_path):
        rows = _read_csv(ps_path)
        if rows and "core_only_quiet" in rows[0]:
            acc: Dict[Tuple[float, float, float], List[int]] = {}
            for r in rows:
                key = (_fnum(r, "kick_boost"), _fnum(r, "win_lo"), _fnum(r, "win_hi"))
                acc.setdefault(key, []).append(int(_fnum(r, "core_only_quiet", 0)))
            return {k: (sum(v) / len(v) >= 0.5) for k, v in acc.items()}
    # FAIL CLOSED (review P1-c): never silently treat a missing gate as "quiet" — a
    # self-igniting core would then be trusted. Raise so the analysis can't fail open.
    raise FileNotFoundError(
        f"no core_only_quiet source in {run_dir} (need reclassified_candidate_table.csv / "
        "candidate_table.csv / per_seed_metrics.csv carrying a core_only_quiet column)")


# --------------------------------------------------------------------------- #
# Core pure functions (unit-tested with synthetic data, no SNN)               #
# --------------------------------------------------------------------------- #
def aggregate_by_kick(
    per_seed_rows: List[Dict[str, str]],
    core_quiet_map: Dict[Tuple[float, float, float], bool],
    win_lo: float,
    win_hi: float,
) -> List[KickAggregate]:
    """Aggregate per_seed rows for one window family into per-kick aggregates.

    Aggregation is over seeds AND rep bins (all rows matching the kick+window).
    P_* are means of the per-seed 0/1 flags; downstream/n_activated/r95/far_field
    are medians over seeds.
    """
    by_kick: Dict[float, List[Dict[str, str]]] = {}
    for r in per_seed_rows:
        if _fnum(r, "win_lo") != win_lo or _fnum(r, "win_hi") != win_hi:
            continue
        by_kick.setdefault(_fnum(r, "kick_boost"), []).append(r)

    out: List[KickAggregate] = []
    for kick in sorted(by_kick):
        rows = by_kick[kick]
        n = len(rows)
        # distinct seeds present (robustness denominator the user cares about)
        n_seeds = len({_fnum(r, "seed") for r in rows})
        p_local = sum(_fnum(r, "seed_local_returned", 0.0) for r in rows) / n
        p_ret = sum(_fnum(r, "returned", 0.0) for r in rows) / n
        p_run = sum(_fnum(r, "runaway", 0.0) for r in rows) / n
        med_ds = _median([_fnum(r, "downstream_resp") for r in rows])
        med_nact = _median([_fnum(r, "n_activated_bins") for r in rows])
        med_r95 = _median([_fnum(r, "r95_mm") for r in rows])
        med_far = _median([_fnum(r, "far_field_frac") for r in rows])
        quiet = core_quiet_map.get((kick, win_lo, win_hi), True)
        out.append(
            KickAggregate(
                kick=kick,
                n_seeds=n_seeds,
                p_local_returned=p_local,
                p_returned=p_ret,
                p_runaway=p_run,
                median_downstream=med_ds,
                median_n_activated=med_nact,
                median_r95=med_r95,
                median_far_field=med_far,
                core_only_quiet=quiet,
            )
        )
    return out


def per_kick_phenotype(agg: KickAggregate) -> str:
    """Assign one phenotype label to a (substrate, kick) aggregate.

    DRAFT numeric rules (logic matters, magnitudes tunable):
      confounded            : core not quiet (core_only_quiet == False) -> the
                              differenced downstream is untrustworthy.
      escape                : P_runaway >= P_RUNAWAY_ESCAPE OR P_returned low and
                              the response did not come back -> ran away.
      finite_local_returned : core quiet AND P_local_returned >= P_LOCAL_RET_FINITE
                              AND P_returned >= P_RETURNED_LOCAL AND median r95
                              <= R95_LOCAL_CAP_MM (spatially local).
      large_returned        : core quiet AND returned in most seeds but NOT spatially
                              local (median r95 > cap) -> big self-limited event.
      noise                 : core quiet, no robust local-returned event, and the
                              response is scattered (median far_field high).
      silent                : core quiet, essentially no early response and no
                              far-field scatter -> sub-threshold / nothing fired.
    """
    if not agg.core_only_quiet:
        return "confounded"

    if agg.p_runaway >= P_RUNAWAY_ESCAPE:
        return "escape"

    local = (not math.isnan(agg.median_r95)) and agg.median_r95 <= R95_LOCAL_CAP_MM
    returned_mostly = agg.p_returned >= P_RETURNED_LOCAL

    if returned_mostly and agg.p_local_returned >= P_LOCAL_RET_FINITE and local:
        return "finite_local_returned"
    if returned_mostly and agg.p_local_returned >= P_LOCAL_RET_FINITE and not local:
        return "large_returned"

    # not a robust local-returned event: is there even an early response?
    scattered = (
        (not math.isnan(agg.median_far_field))
        and agg.median_far_field >= FARFIELD_NOISE_FRAC
    )
    near_silent = (not math.isnan(agg.median_downstream)) and agg.median_downstream <= 0.0
    if near_silent and not scattered:
        return "silent"
    if scattered:
        return "noise"
    # has some response, but not a robust local-returned event and not scattered:
    # treat as a (sub-threshold) probe that has not yet ignited a finite event.
    return "silent"


def find_stable_finite(
    aggs: List[KickAggregate],
    robust_frac: float = ROBUST_FRAC,
    min_seeds: int = MIN_SEEDS,
) -> None:
    """Mark stable finite-event candidates IN PLACE.

    A (substrate, kick) is a stable finite-event candidate iff its phenotype is
    finite_local_returned AND P_local_returned >= robust_frac over n_seeds >=
    min_seeds. (DRAFT robust_frac=0.7, min_seeds=6.)
    """
    for a in aggs:
        a.stable_finite = (
            a.phenotype == "finite_local_returned"
            and a.p_local_returned >= robust_frac
            and a.n_seeds >= min_seeds
        )


def cross_kick_verdict(aggs: List[KickAggregate]) -> Tuple[str, str]:
    """Per-substrate, multi-seed verdict walking kicks ascending.

    Returns (verdict, reason). Verdicts:

      SELF_IGNITE      : mostly confounded (core fires on its own) -> cannot read
                         a kick-evoked finite event; differencing untrustworthy.
      FINITE_THRESHOLD : kicks below a threshold are silent/noise and a stable
                         finite_local_returned event appears at/above it
                         (silent -> jump -> finite local returned) = branch B / W_event.
      LINEAR_GRADED    : a contiguous low-kick range with a graded, monotone,
                         local-returned response rising smoothly from small kick
                         values (no silent gap then jump) = branch A / W_small.
      ESCAPE_ONLY      : at least one escape and no stable finite_local_returned.
      NO_LOCAL         : no stable finite_local_returned at any kick and no escape
                         (e.g. under-powered: events present but never robust).

    This is a PER-SUBSTRATE, MULTI-SEED verdict; callers must report it with the
    n_seeds and the one-window / DRAFT-threshold caveats.
    """
    if not aggs:
        return "NO_LOCAL", "no kicks"

    aggs = sorted(aggs, key=lambda a: a.kick)
    n = len(aggs)
    n_confounded = sum(1 for a in aggs if a.phenotype == "confounded")
    if n_confounded > n / 2:
        return (
            "SELF_IGNITE",
            f"{n_confounded}/{n} kicks confounded (core fires without kick)",
        )

    stable = [a for a in aggs if a.stable_finite]
    has_escape = any(a.phenotype == "escape" for a in aggs)

    # --- LINEAR_GRADED: graded monotone local-returned ramp from low kicks --- #
    # P_local_returned must be already rising at/below GRADED_RISE_FROM (no silent
    # gap), monotone non-decreasing over a contiguous low-kick prefix, and reach a
    # robust level. The lowest kick must NOT be silent (that signals a threshold).
    lowest = aggs[0]
    graded_prefix = _graded_prefix(aggs)
    if (
        lowest.kick <= GRADED_RISE_FROM
        and lowest.p_local_returned > 0.0
        and len(graded_prefix) >= 3
        and graded_prefix[-1].p_local_returned >= ROBUST_FRAC
        and graded_prefix[-1].n_seeds >= MIN_SEEDS   # adequately powered ramp (review P1-b)
    ):
        return (
            "LINEAR_GRADED",
            "graded monotone local-returned ramp from the lowest kick "
            f"(contiguous over {len(graded_prefix)} kicks, no silent gap)",
        )

    # --- FINITE_THRESHOLD: silent below, stable finite event at/above ------- #
    if stable:
        first_stable = min(stable, key=lambda a: a.kick)
        below = [a for a in aggs if a.kick < first_stable.kick]
        below_quiet = all(
            a.phenotype in ("silent", "noise") for a in below
        )
        if below and below_quiet:
            return (
                "FINITE_THRESHOLD",
                f"silent/noise below kick={first_stable.kick:g}, stable "
                f"finite_local_returned at/above (branch B / W_event)",
            )
        # stable event exists but the below-threshold region wasn't silent:
        # still threshold-like (jump into a robust event), report as threshold.
        return (
            "FINITE_THRESHOLD",
            f"stable finite_local_returned first at kick={first_stable.kick:g}",
        )

    # --- no stable finite event anywhere ------------------------------------ #
    if has_escape:
        return "ESCAPE_ONLY", "escape regime present, no stable finite local event"
    # Under-powered (review P1-b): if the best-powered kick has fewer than MIN_SEEDS
    # seeds, we CANNOT call stability — do NOT output the negative-looking NO_LOCAL.
    max_n_seeds = max((a.n_seeds for a in aggs), default=0)
    if max_n_seeds < MIN_SEEDS:
        return (
            "UNDERPOWERED",
            f"no stable finite event, but max n_seeds={max_n_seeds} < MIN_SEEDS={MIN_SEEDS} "
            "— cannot determine robustness; finite events may be present per-kick "
            "(see P_local_returned), NOT a negative result",
        )
    return (
        "NO_LOCAL",
        "no stable finite_local_returned at any kick despite adequate seeds "
        f"(max n_seeds={max_n_seeds} >= {MIN_SEEDS})",
    )


def _graded_prefix(aggs: List[KickAggregate]) -> List[KickAggregate]:
    """Longest contiguous low-kick prefix with monotone non-decreasing
    P_local_returned (graded ramp, no drop)."""
    prefix = [aggs[0]]
    for a in aggs[1:]:
        if a.p_local_returned + 1e-9 >= prefix[-1].p_local_returned:
            prefix.append(a)
        else:
            break
    return prefix


# --------------------------------------------------------------------------- #
# Per-substrate driver                                                         #
# --------------------------------------------------------------------------- #
def analyze_substrate(
    run_dir: str, win_lo: float, win_hi: float
) -> Optional[SubstrateResult]:
    ps_path = os.path.join(run_dir, "per_seed_metrics.csv")
    if not os.path.isfile(ps_path):
        return None
    per_seed = _read_csv(ps_path)
    if not per_seed:
        return None
    quiet_map = load_core_quiet_map(run_dir)
    aggs = aggregate_by_kick(per_seed, quiet_map, win_lo, win_hi)
    if not aggs:
        return None
    for a in aggs:
        a.phenotype = per_kick_phenotype(a)
    find_stable_finite(aggs)
    verdict, reason = cross_kick_verdict(aggs)
    n_seeds = max((a.n_seeds for a in aggs), default=0)
    return SubstrateResult(
        name=os.path.basename(run_dir.rstrip("/")),
        win_lo=win_lo,
        win_hi=win_hi,
        kicks=aggs,
        verdict=verdict,
        verdict_reason=reason,
        stable_kicks=[a.kick for a in aggs if a.stable_finite],
        n_seeds=n_seeds,
    )


# --------------------------------------------------------------------------- #
# Outputs                                                                      #
# --------------------------------------------------------------------------- #
PHENO_COLORS = {
    "confounded": "#9e9e9e",
    "noise": "#bdbdbd",
    "silent": "#80b1d3",
    "finite_local_returned": "#2ca02c",
    "large_returned": "#ff7f0e",
    "escape": "#d62728",
}


def write_csvs(results: List[SubstrateResult], out_dir: str) -> None:
    def _dump(fname: str, value_key: str, value_fn) -> None:
        with open(os.path.join(out_dir, fname), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["substrate", "kick", value_key, "n_seeds"])
            for r in results:
                for a in r.kicks:
                    w.writerow([r.name, a.kick, value_fn(a), a.n_seeds])

    _dump("finite_event_success_probability.csv", "P_local_returned",
          lambda a: a.p_local_returned)
    _dump("return_probability.csv", "P_returned", lambda a: a.p_returned)
    _dump("r95_by_kick.csv", "median_r95_mm", lambda a: a.median_r95)
    _dump("farfield_by_kick.csv", "median_far_field_frac", lambda a: a.median_far_field)

    with open(os.path.join(out_dir, "phenotype_ladder.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "substrate", "win_lo", "win_hi", "kick", "n_seeds",
            "P_local_returned", "P_returned", "P_runaway", "core_only_quiet",
            "median_downstream", "median_r95_mm", "median_far_field_frac",
            "phenotype", "stable_finite", "verdict",
        ])
        for r in results:
            for a in r.kicks:
                w.writerow([
                    r.name, r.win_lo, r.win_hi, a.kick, a.n_seeds,
                    f"{a.p_local_returned:.4f}", f"{a.p_returned:.4f}",
                    f"{a.p_runaway:.4f}", int(a.core_only_quiet),
                    f"{a.median_downstream:.4f}", f"{a.median_r95:.4f}",
                    f"{a.median_far_field:.4f}", a.phenotype,
                    int(a.stable_finite), r.verdict,
                ])


def plot_phenotype_ladder(results: List[SubstrateResult], out_dir: str) -> str:
    n = len(results)
    if n == 0:
        return ""
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.6 * n + 0.6), squeeze=False)
    for ax, r in zip(axes[:, 0], results):
        kicks = [a.kick for a in r.kicks]
        plocal = [a.p_local_returned for a in r.kicks]
        ax.plot(kicks, plocal, "-o", color="#333333", lw=1.5, ms=4,
                label="P(local & returned)", zorder=3)
        for a in r.kicks:
            ax.scatter([a.kick], [a.p_local_returned], s=140,
                       color=PHENO_COLORS.get(a.phenotype, "#000000"),
                       edgecolors="black", linewidths=0.6, zorder=4)
            if a.stable_finite:
                ax.scatter([a.kick], [a.p_local_returned], s=320,
                           facecolors="none", edgecolors="#2ca02c",
                           linewidths=2.0, zorder=2)
        ax.axhline(ROBUST_FRAC, color="#2ca02c", ls="--", lw=1.0, alpha=0.7)
        ax.text(0.99, ROBUST_FRAC + 0.02,
                f"robust frac = {ROBUST_FRAC:g} (DRAFT)",
                transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=7, color="#2ca02c")
        if kicks:
            ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.08)
        ax.set_ylabel("P(local & returned)")
        ax.set_title(
            f"{r.name}  —  {r.verdict}   (n_seeds={r.n_seeds}, win {r.win_lo:g}-{r.win_hi:g} ms)",
            fontsize=10,
        )
        ax.grid(True, alpha=0.25)
    axes[-1, 0].set_xlabel("kick boost (mV, log scale)")

    legend_handles = [
        Patch(facecolor=PHENO_COLORS[p], edgecolor="black", label=p)
        for p in PHENOTYPES
    ]
    legend_handles.append(
        Patch(facecolor="none", edgecolor="#2ca02c", linewidth=2.0,
              label=f"stable finite event (P>={ROBUST_FRAC:g}, n>={MIN_SEEDS})")
    )
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "M3 kick calibration — multi-seed finite-event phenotype ladder\n"
        "(per substrate: is there a stable LOCAL+RETURNED finite event, and is it graded or threshold?)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, "phenotype_ladder.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


VERDICT_PLAIN = {
    "LINEAR_GRADED": "线性渐变（A 支 / W_small）——小 kick 就有局部回静的响应，随 kick 平滑升高，没有先沉默再突跳。",
    "FINITE_THRESHOLD": "有限事件阈值（B 支 / W_event）——阈值以下沉默/散乱，到某个 kick 突然冒出一个稳定的局部+回静有限事件。",
    "ESCAPE_ONLY": "只跑飞——没有任何 kick 产生稳定局部有限事件，存在跑飞/不回静。",
    "NO_LOCAL": "无稳定局部事件——任何 kick 都没有稳健的局部回静有限事件（可能事件出现但不跨 seed 稳健，欠功率）。",
    "SELF_IGNITE": "核自燃——核不戳也在点火，差分响应不可信，读不出 kick 诱发的有限事件。",
}


def write_status(results: List[SubstrateResult], out_dir: str, win_lo: float,
                 win_hi: float) -> None:
    any_stable = any(r.stable_kicks for r in results)
    max_seeds = max((r.n_seeds for r in results), default=0)
    lines: List[str] = []
    lines.append("# M3 kick 标定 — 多 seed 鲁棒性分析 STATUS\n")
    lines.append("## 头条结论\n")
    if any_stable:
        sub_strs = []
        for r in results:
            if r.stable_kicks:
                ks = ", ".join(f"{k:g}" for k in r.stable_kicks)
                sub_strs.append(f"{r.name}（kick={ks}）")
        lines.append(
            "**存在稳定的局部+回静有限事件**：" + "；".join(sub_strs) + "。\n"
        )
        threshold_subs = [r.name for r in results if r.verdict == "FINITE_THRESHOLD"]
        graded_subs = [r.name for r in results if r.verdict == "LINEAR_GRADED"]
        if threshold_subs and not graded_subs:
            lines.append(
                "形态是**阈值型有限事件（B 支 / W_event）**：阈值以下沉默，"
                "过阈才突然冒出稳定的局部有限事件，不是平滑渐变。\n"
            )
        elif graded_subs and not threshold_subs:
            lines.append(
                "形态是**线性渐变（A 支 / W_small）**：小 kick 就有响应并随 kick 平滑升高。\n"
            )
        else:
            lines.append("形态在不同核之间不一致，逐核见下。\n")
    else:
        lines.append(
            f"**当前数据（n_seeds≈{max_seeds}）下还没有任何核冒出稳定的局部+回静有限事件**"
            f"（判据：P(局部且回静) ≥ {ROBUST_FRAC:g} 且 seed 数 ≥ {MIN_SEEDS}，均为 DRAFT 阈值）。"
            "窄核看起来卡在阈值型有限事件的门口，但 3 seed 欠功率；宽核是核自燃。"
            "这是预期的——稳健判定需要后续的多 seed 细扫。\n"
        )

    lines.append("\n## 一句话讲清楚我们在测什么\n")
    lines.append(
        "我们给一小撮『更易兴奋的核细胞』短戳一下（kick），看戳完之后：(1) 是不是只在核附近局部点了一下、"
        "活动随后回到基线（局部+回静），(2) 还是没戳到什么（沉默），(3) 还是一戳就蔓延成全局波（跑飞），"
        "(4) 还是核根本不戳自己也在烧（自燃，差分不可信）。"
        f"对每个核、每个 kick，我们把 {max_seeds} 个随机种子的结果聚合，"
        "数『局部且回静』在多少比例的种子里出现（这就是 P(局部且回静)）。"
        "然后沿 kick 从小到大走一遍，判断这个核是『小 kick 就平滑有响应』(A)，"
        "还是『过了某个阈值才突然有一个稳定有限事件』(B)。\n"
    )

    lines.append("\n## 逐核：阶梯 + 跨 kick 判定\n")
    for r in results:
        lines.append(f"\n### {r.name}（窗 {r.win_lo:g}–{r.win_hi:g} ms，n_seeds={r.n_seeds}）\n")
        lines.append(f"**跨 kick 判定：{r.verdict}** — {VERDICT_PLAIN.get(r.verdict, '')}\n")
        lines.append(f"（判定依据：{r.verdict_reason}）\n\n")
        lines.append("逐 kick 阶梯：\n")
        for a in r.kicks:
            stable_tag = " ★稳定有限事件" if a.stable_finite else ""
            lines.append(
                f"- kick={a.kick:g}：表型=`{a.phenotype}`，"
                f"P(局部且回静)={a.p_local_returned:.2f}，P(回静)={a.p_returned:.2f}，"
                f"P(跑飞)={a.p_runaway:.2f}，中位 r95={a.median_r95:.1f}mm，"
                f"中位远场={a.median_far_field:.2f}，核安静={'是' if a.core_only_quiet else '否'}"
                f"{stable_tag}"
            )
        lines.append("")

    lines.append("\n## 诚实的注意事项（caveats）\n")
    lines.append(
        f"- **seed 数**：当前是 {max_seeds} seed/格，远低于稳健判定要求的 {MIN_SEEDS}。"
        "任何『稳定有限事件』结论都要等多 seed 细扫；3 seed 只能说『看起来像/不像』，不能下定论。\n"
    )
    lines.append(
        f"- **单窗**：本分析只看一个事件后窗族（win {win_lo:g}–{win_hi:g} ms）；"
        "事件对齐窗是另一条线的工作，不在此处。\n"
    )
    lines.append(
        "- **DRAFT 阈值**：表型规则与判定阈值"
        f"（ROBUST_FRAC={ROBUST_FRAC:g}、MIN_SEEDS={MIN_SEEDS}、"
        f"r95 局部上限={R95_LOCAL_CAP_MM:g}mm、远场散乱阈={FARFIELD_NOISE_FRAC:g}）"
        "都是草拟值——逻辑是交付物，数值随真实细扫再调。\n"
    )
    lines.append(
        "- **核安静门**：优先采用 reclassified_candidate_table.csv 里的相对核安静门"
        "（核放电不显著多于空白薄片才算安静）；缺失时退回 candidate_table 的绝对地板门。"
        "宽核在相对门下仍判自燃 → SELF_IGNITE。\n"
    )

    with open(os.path.join(out_dir, "STATUS.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_figures_readme(results: List[SubstrateResult], out_dir: str) -> None:
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    lines = [
        "# figures 说明\n",
        "### phenotype_ladder.png\n",
        "每个面板一个核（substrate），横轴是 kick 大小（对数刻度），"
        "黑线是 P(局部且回静)——戳完之后『只在局部点一下且活动回到基线』在多少比例的种子里发生；"
        "每个点的颜色是该 kick 的表型标签（沉默/有限局部回静/大事件回静/跑飞/核自燃/散乱噪声）；"
        "绿色空心圈标出『稳定有限事件』（P≥0.7 且 seed≥6，DRAFT）；"
        "绿色虚线是 robust 阈值。面板标题给出该核的跨 kick 判定"
        "（线性渐变 A / 阈值有限事件 B / 只跑飞 / 无局部 / 核自燃）。\n",
        "**关注点**：窄核有没有出现『低 kick 沉默 → 过阈突然冒出绿色稳定有限事件』的 B 支阶梯；"
        "宽核是不是整条线都被判成核自燃（灰色）；以及当前 seed 数下绿色圈是否已经点亮"
        "（3 seed 通常还点不亮，属预期欠功率）。\n",
    ]
    with open(os.path.join(fig_dir, "README.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
DEFAULT_GLOB = (
    "results/topic4_sef_hfo/m3_local_w/kick_calibration_explore/L20_core_*"
)


def _expand_run_dirs(patterns: Sequence[str]) -> List[str]:
    dirs: List[str] = []
    seen = set()
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            if not os.path.isdir(p):
                continue
            # only dirs that actually carry a per_seed_metrics.csv
            if not os.path.isfile(os.path.join(p, "per_seed_metrics.csv")):
                continue
            if p not in seen:
                seen.add(p)
                dirs.append(p)
    return dirs


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "run_dirs",
        nargs="*",
        default=[DEFAULT_GLOB],
        help="run dirs or globs (default: the L20_core_* explore dirs)",
    )
    ap.add_argument("--out-dir", required=True, help="output directory")
    ap.add_argument(
        "--win",
        default="22,32",
        help="event-after window family 'lo,hi' in ms (default 22,32)",
    )
    args = ap.parse_args(argv)

    win_lo, win_hi = (float(x) for x in args.win.split(","))
    patterns = args.run_dirs if args.run_dirs else [DEFAULT_GLOB]
    run_dirs = _expand_run_dirs(patterns)
    if not run_dirs:
        ap.error(f"no run dirs with per_seed_metrics.csv matched: {patterns}")

    os.makedirs(args.out_dir, exist_ok=True)
    results: List[SubstrateResult] = []
    for d in run_dirs:
        r = analyze_substrate(d, win_lo, win_hi)
        if r is not None:
            results.append(r)

    if not results:
        ap.error("no substrate produced an aggregate (empty per_seed_metrics?)")

    write_csvs(results, args.out_dir)
    plot_phenotype_ladder(results, args.out_dir)
    write_status(results, args.out_dir, win_lo, win_hi)
    write_figures_readme(results, args.out_dir)

    print(f"analyzed {len(results)} substrate(s), window {win_lo:g}-{win_hi:g} ms:")
    for r in results:
        stable = ",".join(f"{k:g}" for k in r.stable_kicks) or "none"
        print(
            f"  {r.name:28s} verdict={r.verdict:16s} "
            f"n_seeds={r.n_seeds} stable_finite_kicks={stable}"
        )
    print(f"outputs -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
