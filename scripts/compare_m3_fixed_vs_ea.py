#!/usr/bin/env python3
"""Fixed-window vs event-aligned (EA) comparison for M3 kick-calibration runs.

OFFLINE ANALYSIS over already-dumped per_seed_metrics.csv / candidate_table.csv.
Runs NO SNN. Answers ONE question the Lane A per-window verdict cannot: does the
spatial-locality reading depend on WHERE you place the fixed observation window?

Key fact about the dumped fields: the EA columns (downstream_resp_ea, r95_mm_ea,
far_field_frac_ea, t0_ms, event_detected) are t0-ANCHORED — for a given (kick,
seed) they are IDENTICAL across the three fixed window families. The fixed
columns (downstream_resp, r95_mm, far_field_frac) DO vary per window. So for each
(substrate, kick, window) we compare the window's fixed reading to the single
window-independent EA reading, and flag whether the LOCAL/non-local verdict agrees.

The locality caps mirror Lane A exactly (imported, not re-defined) so the two
analyses cannot drift.

Per the user's review contract (2026-06-22) the comparison table carries, per
substrate x kick x window:
  P_local_returned, qualifies, fixed r95/far/downstream, event_detected_frac,
  t0_ms, EA r95/far/downstream  (+ is_local_fixed / is_local_ea / locality_agree).

Outputs (into --out-dir):
  fixed_vs_ea_comparison.csv   — every substrate x kick x window row
  fixed_vs_ea.png              — 2x2 diagnostic (r95 agree / far agree / t0 / detect)
  COMPARISON_SUMMARY.md        — per-substrate agreement + bare-vs-core contrast
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Reuse Lane A loaders + locality caps so the two analyses share one contract.
from analyze_m3_finite_event_robustness import (  # noqa: E402
    _read_csv, _fnum, _median, load_core_quiet_map,
    R95_LOCAL_CAP_MM, FARFIELD_NOISE_FRAC, ROBUST_FRAC, MIN_SEEDS,
)

WINDOWS: Tuple[Tuple[float, float], ...] = ((18.0, 24.0), (20.0, 28.0), (22.0, 32.0))


@dataclass
class Row:
    substrate: str
    kick: float
    win_lo: float
    win_hi: float
    n_seeds: int
    p_local_returned: float
    qualifies: Optional[int]          # from candidate_table; None if absent
    core_only_quiet: bool
    # fixed window (varies per window)
    fix_r95: float
    fix_far: float
    fix_downstream: float
    is_local_fixed: bool
    # event-aligned (t0-anchored; window-independent)
    event_detected_frac: float
    t0_ms: float
    ea_r95: float
    ea_far: float
    ea_downstream: float
    is_local_ea: bool
    locality_agree: bool


def _is_local(r95: float, far: float) -> bool:
    """Same spatial-locality test Lane A uses for finite_local_returned."""
    if r95 != r95 or far != far:        # NaN -> not classifiable as local
        return False
    return (r95 <= R95_LOCAL_CAP_MM) and (far <= FARFIELD_NOISE_FRAC)


def _qualifies_map(run_dir: str) -> Dict[Tuple[float, float, float], int]:
    """(kick, win_lo, win_hi) -> qualifies flag from candidate_table.csv (if any)."""
    for fname in ("reclassified_candidate_table.csv", "candidate_table.csv"):
        path = os.path.join(run_dir, fname)
        if not os.path.isfile(path):
            continue
        rows = _read_csv(path)
        if not rows or "qualifies" not in rows[0]:
            continue
        out: Dict[Tuple[float, float, float], int] = {}
        for r in rows:
            key = (_fnum(r, "kick_boost"), _fnum(r, "win_lo"), _fnum(r, "win_hi"))
            out[key] = int(_fnum(r, "qualifies", 0))
        return out
    return {}


def build_rows(run_dir: str) -> List[Row]:
    name = os.path.basename(run_dir.rstrip("/")).replace("finescan_", "")
    per_seed = _read_csv(os.path.join(run_dir, "per_seed_metrics.csv"))
    quiet_map = load_core_quiet_map(run_dir)        # fail-closed if no source
    qual_map = _qualifies_map(run_dir)

    # group rows by (kick, win)
    grouped: Dict[Tuple[float, float, float], List[Dict[str, str]]] = {}
    for r in per_seed:
        key = (_fnum(r, "kick_boost"), _fnum(r, "win_lo"), _fnum(r, "win_hi"))
        grouped.setdefault(key, []).append(r)

    out: List[Row] = []
    for (kick, wlo, whi) in sorted(grouped):
        rows = grouped[(kick, wlo, whi)]
        n = len(rows)
        n_seeds = len({_fnum(r, "seed") for r in rows})
        p_local = sum(_fnum(r, "seed_local_returned", 0.0) for r in rows) / n
        fix_r95 = _median([_fnum(r, "r95_mm") for r in rows])
        fix_far = _median([_fnum(r, "far_field_frac") for r in rows])
        fix_ds = _median([_fnum(r, "downstream_resp") for r in rows])
        ea_r95 = _median([_fnum(r, "r95_mm_ea") for r in rows])
        ea_far = _median([_fnum(r, "far_field_frac_ea") for r in rows])
        ea_ds = _median([_fnum(r, "downstream_resp_ea") for r in rows])
        det_frac = sum(_fnum(r, "event_detected", 0.0) for r in rows) / n
        t0 = _median([_fnum(r, "t0_ms") for r in rows])
        loc_fixed = _is_local(fix_r95, fix_far)
        loc_ea = _is_local(ea_r95, ea_far)
        out.append(Row(
            substrate=name, kick=kick, win_lo=wlo, win_hi=whi,
            n_seeds=n_seeds, p_local_returned=p_local,
            qualifies=qual_map.get((kick, wlo, whi)),
            core_only_quiet=quiet_map.get((kick, wlo, whi), True),
            fix_r95=fix_r95, fix_far=fix_far, fix_downstream=fix_ds,
            is_local_fixed=loc_fixed,
            event_detected_frac=det_frac, t0_ms=t0,
            ea_r95=ea_r95, ea_far=ea_far, ea_downstream=ea_ds,
            is_local_ea=loc_ea,
            locality_agree=(loc_fixed == loc_ea),
        ))
    return out


def write_csv(rows: List[Row], out_dir: str) -> str:
    path = os.path.join(out_dir, "fixed_vs_ea_comparison.csv")
    cols = [
        "substrate", "kick", "win_lo", "win_hi", "n_seeds", "p_local_returned",
        "qualifies", "core_only_quiet",
        "fix_r95", "fix_far", "fix_downstream", "is_local_fixed",
        "event_detected_frac", "t0_ms", "ea_r95", "ea_far", "ea_downstream",
        "is_local_ea", "locality_agree",
    ]
    import csv as _csv
    with open(path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([
                r.substrate, f"{r.kick:g}", f"{r.win_lo:g}", f"{r.win_hi:g}",
                r.n_seeds, f"{r.p_local_returned:.3f}",
                "" if r.qualifies is None else r.qualifies, int(r.core_only_quiet),
                f"{r.fix_r95:.3f}", f"{r.fix_far:.3f}", f"{r.fix_downstream:.3f}",
                int(r.is_local_fixed),
                f"{r.event_detected_frac:.3f}", f"{r.t0_ms:.2f}",
                f"{r.ea_r95:.3f}", f"{r.ea_far:.3f}", f"{r.ea_downstream:.3f}",
                int(r.is_local_ea), int(r.locality_agree),
            ])
    return path


_SUB_COLORS = {
    "bare": "#888888", "n17.6": "#1f77b4", "n17.8": "#2ca02c",
    "n18.0": "#ff7f0e", "w18.0": "#d62728",
}


def _color(name: str) -> str:
    return _SUB_COLORS.get(name, "#9467bd")


def plot_fig(rows: List[Row], out_dir: str) -> str:
    subs = sorted({r.substrate for r in rows})
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # Panel A: fixed r95 vs EA r95 (window placement effect on spatial extent)
    axA = axes[0][0]
    for s in subs:
        xs = [r.fix_r95 for r in rows if r.substrate == s]
        ys = [r.ea_r95 for r in rows if r.substrate == s]
        axA.scatter(xs, ys, s=22, alpha=0.7, color=_color(s), label=s)
    lim = max([1.0] + [r.fix_r95 for r in rows] + [r.ea_r95 for r in rows])
    axA.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.6)
    axA.axhline(R95_LOCAL_CAP_MM, color="gray", lw=0.6, ls=":")
    axA.axvline(R95_LOCAL_CAP_MM, color="gray", lw=0.6, ls=":")
    axA.set_xlabel("fixed-window r95 (mm)")
    axA.set_ylabel("event-aligned r95 (mm)")
    axA.set_title("A. spatial extent: fixed window vs event-aligned")
    axA.legend(fontsize=8, loc="upper left")

    # Panel B: fixed far vs EA far (window placement effect on far-field leak)
    axB = axes[0][1]
    for s in subs:
        xs = [r.fix_far for r in rows if r.substrate == s]
        ys = [r.ea_far for r in rows if r.substrate == s]
        axB.scatter(xs, ys, s=22, alpha=0.7, color=_color(s), label=s)
    axB.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6)
    axB.axhline(FARFIELD_NOISE_FRAC, color="gray", lw=0.6, ls=":")
    axB.axvline(FARFIELD_NOISE_FRAC, color="gray", lw=0.6, ls=":")
    axB.set_xlabel("fixed-window far-field frac")
    axB.set_ylabel("event-aligned far-field frac")
    axB.set_title("B. far-field leak: fixed window vs event-aligned")

    # Panel C: t0_ms by substrate x kick (does the event onset land in the windows?)
    axC = axes[1][0]
    for s in subs:
        xs = [r.kick for r in rows if r.substrate == s]
        ys = [r.t0_ms for r in rows if r.substrate == s]
        axC.scatter(xs, ys, s=22, alpha=0.7, color=_color(s), label=s)
    # shade the three fixed observation windows (abs ms = t_kick(100) + offset)
    for (wlo, whi) in WINDOWS:
        axC.axhspan(100 + wlo, 100 + whi, color="gray", alpha=0.06)
    axC.set_xlabel("kick_boost")
    axC.set_ylabel("event onset t0 (ms; kick at 100 ms)")
    axC.set_title("C. event-onset timing vs the 3 fixed windows (shaded)")

    # Panel D: event_detected_frac by substrate x kick
    axD = axes[1][1]
    for s in subs:
        xs = [r.kick for r in rows if r.substrate == s]
        ys = [r.event_detected_frac for r in rows if r.substrate == s]
        axD.plot(xs, ys, "o-", ms=4, alpha=0.7, color=_color(s), label=s)
    axD.set_ylim(-0.05, 1.05)
    axD.set_xlabel("kick_boost")
    axD.set_ylabel("event_detected_frac (over seeds x windows)")
    axD.set_title("D. fraction of seeds with a detected event")
    axD.legend(fontsize=8, loc="lower right")

    fig.suptitle("M3 fine-scan: fixed-window vs event-aligned readout", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = os.path.join(out_dir, "fixed_vs_ea.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def write_summary(rows: List[Row], out_dir: str) -> str:
    subs = sorted({r.substrate for r in rows})
    lines: List[str] = ["# Fixed-window vs event-aligned comparison\n"]
    lines.append(
        "局部性判据沿用 Lane A：r95 ≤ "
        f"{R95_LOCAL_CAP_MM:g} mm 且 far ≤ {FARFIELD_NOISE_FRAC:g} 记为 local。"
        "EA 列是事件对齐（t0 锚定，与固定窗无关）；fixed 列随窗变化。\n")
    for s in subs:
        sr = [r for r in rows if r.substrate == s]
        n_agree = sum(1 for r in sr if r.locality_agree)
        # disagreements: where the fixed window flips the local verdict vs EA
        disagree = [r for r in sr if not r.locality_agree]
        # kicks where EA says local but at least one fixed window says non-local
        ea_local_kicks = sorted({r.kick for r in sr if r.is_local_ea})
        fixed_local_kicks = sorted({r.kick for r in sr if r.is_local_fixed})
        qual_kicks = sorted({r.kick for r in sr if r.qualifies == 1})
        quiet_all = all(r.core_only_quiet for r in sr)
        lines.append(f"\n## {s}\n")
        lines.append(f"- core_only_quiet（全部 kick×窗）: {'是' if quiet_all else '否（有自燃）'}")
        lines.append(f"- locality 一致行: {n_agree}/{len(sr)}（fixed 与 EA 同判 local/非 local）")
        lines.append(f"- EA 判 local 的 kick: {ea_local_kicks or '无'}")
        lines.append(f"- fixed 判 local 的 kick（任一窗）: {fixed_local_kicks or '无'}")
        lines.append(f"- selector qualifies 的 kick: {qual_kicks or '无'}")
        if disagree:
            lines.append("- 不一致行（窗口翻转 local 判定）:")
            for r in disagree:
                lines.append(
                    f"    kick={r.kick:g} win={r.win_lo:g}-{r.win_hi:g}: "
                    f"fixed(r95={r.fix_r95:.2f},far={r.fix_far:.2f})→"
                    f"{'local' if r.is_local_fixed else '非'} vs "
                    f"EA(r95={r.ea_r95:.2f},far={r.ea_far:.2f})→"
                    f"{'local' if r.is_local_ea else '非'}")
    path = os.path.join(out_dir, "COMPARISON_SUMMARY.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def _expand(patterns: Sequence[str]) -> List[str]:
    import glob
    out: List[str] = []
    for p in patterns:
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "per_seed_metrics.csv")):
            out.append(p)
        else:
            for g in sorted(glob.glob(p)):
                if os.path.isfile(os.path.join(g, "per_seed_metrics.csv")):
                    out.append(g)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dirs", nargs="+", help="finescan_* run dirs")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    dirs = _expand(args.run_dirs)
    if not dirs:
        ap.error(f"no run dirs with per_seed_metrics.csv: {args.run_dirs}")
    os.makedirs(args.out_dir, exist_ok=True)

    rows: List[Row] = []
    for d in dirs:
        rows.extend(build_rows(d))

    csv_path = write_csv(rows, args.out_dir)
    fig_path = plot_fig(rows, args.out_dir)
    sum_path = write_summary(rows, args.out_dir)

    subs = sorted({r.substrate for r in rows})
    print(f"compared {len(subs)} substrate(s), {len(rows)} rows:")
    for s in subs:
        sr = [r for r in rows if r.substrate == s]
        n_agree = sum(1 for r in sr if r.locality_agree)
        print(f"  {s:8s} locality_agree {n_agree}/{len(sr)}  "
              f"EA-local kicks {sorted({r.kick for r in sr if r.is_local_ea})}")
    print(f"outputs -> {csv_path}\n            {fig_path}\n            {sum_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
