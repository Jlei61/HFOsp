#!/usr/bin/env python
"""Topic 5 V3a single-subject case panel.

Default case is epilepsiae_1125 in the narrow cohort. The figure is intentionally
case-level and exploratory: it visualizes why E1125 is internally consistent
without promoting it to a cohort-level mode-transition claim.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._topic5_v3_io import classify_subject_contacts, load_subject_phase_envelopes  # noqa: E402
from scripts.run_topic5_v3_avalanche import run_subject as run_h3b_subject  # noqa: E402
from scripts.run_topic5_v3_dynamics import run_subject as run_h3c_subject  # noqa: E402
from scripts.run_topic5_v3_susceptibility import (  # noqa: E402
    _abs_beta_sz,
    _line_length_rate,
    run_subject as run_h3a_subject,
)
from src.topic5_v2_criticality import activations_from_z  # noqa: E402
from src.topic5_v3_mode_transition import (  # noqa: E402
    atm_offdiag,
    dominant_right_singular_vector,
    load_v3_config,
    lowrank_var,
    map_lowrank_vector_to_contacts,
    net_offaxis_flux,
    rank_forward,
    sliding_windows,
    subspace_mode_shift,
    subspace_projectors,
)

DEFAULT_OUTDIR = _ROOT / "results/topic5_ictal_recruitment/v3_mode_transition/case_figures"
PHASES = ["P0", "P1", "P2", "P3", "O", "I1"]
PHASE_TIME = {"P0": -105.0, "P1": -75.0, "P2": -45.0, "P3": -20.0, "O": 0.0, "I1": 20.0}
BASELINE_PHASES = ["P0", "P1", "P2"]

AXIS_COLOR = "#d1791f"
OFFAXIS_COLOR = "#2a9d8f"
MODE_COLOR = "#7b5aa6"
AMBIG_COLOR = "#a8a8a8"


def _as_bool(v) -> bool:
    return bool(v) if isinstance(v, bool) else str(v) == "True"


def _as_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _row_from_csv(path: Path, subject: str) -> dict | None:
    if not path.exists():
        return None
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("subject") == subject:
                return row
    return None


def _write_single_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _load_or_run_endpoint_rows(subject: str, cohort: str, cfg: dict, n_perm: int, table_dir: Path, force: bool) -> dict:
    """Return H3a/H3b/H3c endpoint rows and persist one-row CSVs."""
    table_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "h3b": table_dir / "v3_avalanche_subject.csv",
        "h3c": table_dir / "v3_dynamics_subject.csv",
        "h3a": table_dir / "v3_susceptibility_subject.csv",
    }
    rows = {} if force else {k: _row_from_csv(p, subject) for k, p in paths.items()}

    if rows.get("h3b") is None:
        rows["h3b"] = run_h3b_subject(subject, cohort, cfg, n_perm)
        _write_single_csv(paths["h3b"], rows["h3b"])
    if rows.get("h3c") is None:
        rows["h3c"] = run_h3c_subject(subject, cohort, cfg, n_perm)
        _write_single_csv(paths["h3c"], rows["h3c"])
    if rows.get("h3a") is None:
        rows["h3a"] = run_h3a_subject(subject, cohort, cfg, n_perm)
        _write_single_csv(paths["h3a"], rows["h3a"])
    return rows


def _rank_forward_for_subject(cc: dict) -> dict:
    axis_set = set(cc["is_axis"])
    typical_rank: dict[str, float] = {}
    for rec in (cc["ctx"]["ta"], cc["ctx"]["tb"]):
        for ch in rec["channels"]:
            name = ch["name"]
            rank = ch.get("typical_rank", np.nan)
            if name in axis_set and np.isfinite(rank):
                typical_rank.setdefault(name, float(rank))
    return rank_forward(typical_rank)


def _windows_of(n_t: int, hop: float, win_sec: float, step_sec: float) -> list[tuple[int, int]]:
    relt_syn = np.arange(n_t) * hop
    return sliding_windows(relt_syn, 0, n_t, win_sec, step_sec)


def _mode_vector(Xw: np.ndarray, rank: int, alpha: float, kstar: int) -> np.ndarray:
    B_r, U_r = lowrank_var(Xw, rank, alpha)
    return map_lowrank_vector_to_contacts(dominant_right_singular_vector(B_r, kstar), U_r)


def _density_parts(u: np.ndarray, P_A: np.ndarray, P_N: np.ndarray) -> tuple[float, float, float]:
    rank_a = float(np.trace(P_A))
    rank_n = float(np.trace(P_N))
    axis = float(np.linalg.norm(P_A @ u) ** 2 / rank_a) if rank_a > 0 else float("nan")
    nonaxis = float(np.linalg.norm(P_N @ u) ** 2 / rank_n) if rank_n > 0 else float("nan")
    return axis, nonaxis, nonaxis - axis


def _median_finite(vals: list[float]) -> float:
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    return float(np.median(arr)) if arr.size else float("nan")


def _baseline_z(by_phase: dict[str, list[float]]) -> dict[str, float]:
    base = [v for p in BASELINE_PHASES for v in by_phase.get(p, []) if np.isfinite(v)]
    if len(base) < 2:
        return {p: float("nan") for p in PHASES}
    mu = float(np.mean(base))
    sd = float(np.std(base))
    if not np.isfinite(sd) or sd <= 0:
        return {p: float("nan") for p in PHASES}
    return {p: _median_finite([(v - mu) / sd for v in by_phase.get(p, [])]) for p in PHASES}


def _compute_case_metrics(subject: str, cohort: str, cfg: dict, cc: dict) -> dict:
    z_thr = float(cfg["avalanche"]["z_threshold"])
    rank = int(cfg["dynamics"]["lowrank"])
    alpha = float(cfg["dynamics"]["var_ridge_alpha"])
    kstar = int(cfg["dynamics"]["finite_horizon_k"])
    hop = float(cfg["phases"]["hop_sec"])
    win_sec = float(cfg["phases"]["window_sec"])
    step_sec = float(cfg["phases"]["step_sec"])

    all_clean = cc["all_clean"]
    rf = _rank_forward_for_subject(cc)
    P_A, P_N = subspace_projectors(all_clean, cc["is_axis"], cc["is_nonaxis_strict"])
    env = load_subject_phase_envelopes(subject, cohort, cfg, PHASES, onset_shift=0.0, cls=cc)
    axis_idx = env["axis_idx"]
    nonaxis_idx = env["nonaxis_idx"]

    phase_values = {m: {p: [] for p in PHASES} for m in ("axis_strength", "offaxis_flux", "mode_shift")}
    mode_parts = {p: {"axis": [], "nonaxis": [], "diff": []} for p in ("P3", "I1")}

    for sz in env["seizures"]:
        for phase in PHASES:
            Xp = sz["phases"].get(phase)
            if Xp is None:
                continue
            llr = dict(zip(all_clean, _line_length_rate(Xp)))
            phase_values["axis_strength"][phase].append(_abs_beta_sz(llr, cc["is_axis"], rf))
            phase_values["offaxis_flux"][phase].append(
                net_offaxis_flux(atm_offdiag(activations_from_z(Xp, z_thr)), axis_idx, nonaxis_idx, "source_mean")
            )

            ms_vals = []
            a_vals = []
            n_vals = []
            for ws, we in _windows_of(Xp.shape[1], hop, win_sec, step_sec):
                u = _mode_vector(Xp[:, ws:we], rank, alpha, kstar)
                a, n, d = _density_parts(u, P_A, P_N)
                ms_vals.append(d)
                a_vals.append(a)
                n_vals.append(n)
            phase_values["mode_shift"][phase].append(_median_finite(ms_vals))
            if phase in mode_parts:
                mode_parts[phase]["axis"].append(_median_finite(a_vals))
                mode_parts[phase]["nonaxis"].append(_median_finite(n_vals))
                mode_parts[phase]["diff"].append(_median_finite(ms_vals))

    return {
        "phase_raw": {
            metric: {phase: _median_finite(values) for phase, values in by_phase.items()}
            for metric, by_phase in phase_values.items()
        },
        "phase_z": {metric: _baseline_z(by_phase) for metric, by_phase in phase_values.items()},
        "mode_density_parts": {
            phase: {k: _median_finite(v) for k, v in parts.items()}
            for phase, parts in mode_parts.items()
        },
    }


def _contact_sort_key(name: str) -> tuple:
    m = re.search(r"(\D+)(\d+)$", name)
    if not m:
        return (name, 0)
    return (m.group(1), int(m.group(2)))


def _plot_geometry(ax, cc: dict) -> None:
    shafts = sorted({cc["shaft_by_name"][n] for n in cc["all_clean"]})
    x_by_shaft = {s: i for i, s in enumerate(shafts)}
    axis_set = set(cc["is_axis"])
    nonaxis_set = set(cc["is_nonaxis_strict"])
    for shaft in shafts:
        names = sorted([n for n in cc["all_clean"] if cc["shaft_by_name"][n] == shaft], key=_contact_sort_key)
        for y, name in enumerate(names):
            if name in axis_set:
                color, marker, size = AXIS_COLOR, "o", 34
            elif name in nonaxis_set:
                color, marker, size = OFFAXIS_COLOR, "o", 24
            else:
                color, marker, size = AMBIG_COLOR, "s", 20
            ax.scatter(x_by_shaft[shaft], y, s=size, marker=marker, color=color, edgecolor="white", linewidth=0.35)
    ax.set_title("A. contact classes", loc="left", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(shafts)))
    ax.set_xticklabels(shafts, rotation=90, fontsize=6)
    ax.set_yticks([])
    ax.set_ylabel("contacts along shaft")
    ax.text(
        0.01,
        0.98,
        f"axis {cc['n_axis']}  non-axis {cc['n_nonaxis']}  ambiguous {cc['n_ambiguous']}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="0.25",
    )
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)


def _plot_trajectory(ax, metrics: dict) -> None:
    xs = [PHASE_TIME[p] for p in PHASES]
    series = [
        ("axis organization |beta|", metrics["phase_z"]["axis_strength"], AXIS_COLOR),
        ("off-axis flux", metrics["phase_z"]["offaxis_flux"], OFFAXIS_COLOR),
        ("mode non-axis minus axis", metrics["phase_z"]["mode_shift"], MODE_COLOR),
    ]
    ax.axhline(0, color="0.55", lw=1, ls="--")
    ax.axvspan(-10, 10, color="0.90", zorder=0)
    ax.axvline(0, color="0.35", lw=1.2)
    for label, by_phase, color in series:
        ys = [by_phase.get(p, float("nan")) for p in PHASES]
        ax.plot(xs, ys, "-o", color=color, lw=2.0, ms=5, label=label)
    ax.set_title("B. observed trajectory", loc="left", fontsize=11, fontweight="bold")
    ax.set_xlabel("time relative to EEG onset (s)")
    ax.set_ylabel("change vs P0-P2 baseline (SD)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:+.0f}".replace("+0", "0") for x in xs])
    ax.legend(loc="upper right", fontsize=7.6, frameon=True, framealpha=0.92)


def _plot_mode_parts(ax, metrics: dict) -> None:
    width = 0.34
    phases = ["P3", "I1"]
    xpos = np.arange(len(phases))
    axis_vals = [metrics["mode_density_parts"][p]["axis"] for p in phases]
    nonaxis_vals = [metrics["mode_density_parts"][p]["nonaxis"] for p in phases]
    ax.bar(xpos - width / 2, axis_vals, width=width, color=AXIS_COLOR, label="axis density")
    ax.bar(xpos + width / 2, nonaxis_vals, width=width, color=OFFAXIS_COLOR, label="non-axis density")
    ax.set_xticks(xpos)
    ax.set_xticklabels(phases)
    ax.set_title("C. H3c mode density", loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel("per-contact mode energy")
    ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.92)
    ymax = max([v for v in axis_vals + nonaxis_vals if np.isfinite(v)] or [1.0])
    ax.set_ylim(0, ymax * 1.22)
    for i, p in enumerate(phases):
        d = metrics["mode_density_parts"][p]["diff"]
        ax.text(i, max(axis_vals[i], nonaxis_vals[i]) * 1.04, f"N-A={d:+.4f}", ha="center", va="bottom", fontsize=8)


def _support_driver(rows: dict) -> str:
    h3b = rows["h3b"]
    h3c = rows["h3c"]
    h3b_path = (
        _as_bool(h3b.get("module_support_flag"))
        and _as_bool(h3b.get("onset_jitter_pass"))
        and _as_bool(h3b.get("leave_one_contact_pass"))
        and _as_bool(h3b.get("axis_only_control_pass"))
        and not _as_bool(h3b.get("common_drive_sensitive"))
    )
    h3c_path = (
        _as_bool(h3c.get("module_support_flag"))
        and _as_bool(h3c.get("onset_jitter_pass"))
        and _as_bool(h3c.get("leave_one_contact_mode_shift_pass"))
        and _as_bool(h3c.get("axis_only_control_pass"))
        and not _as_bool(h3c.get("single_contact_driven"))
    )
    if h3b_path and h3c_path:
        return "H3b+H3c"
    if h3b_path:
        return "H3b"
    if h3c_path:
        return "H3c"
    return "none"


def _plot_endpoint_bars(ax, rows: dict, n_perm: int) -> None:
    h3a, h3b, h3c = rows["h3a"], rows["h3b"], rows["h3c"]
    labels = ["H3a axis\nweakening", "H3b flux\nraw", "H3b flux\nsurplus", "H3c mode\nshift"]
    values = [
        _as_float(h3a.get("delta_beta_axis_strength")),
        _as_float(h3b.get("delta_net_offaxis_flux_raw")),
        _as_float(h3b.get("delta_net_offaxis_flux_surplus")),
        _as_float(h3c.get("delta_mode_shift_density")),
    ]
    colors = [AXIS_COLOR, "0.55", OFFAXIS_COLOR, MODE_COLOR]
    x = np.arange(len(labels))
    ax.axhline(0, color="0.35", lw=1)
    bars = ax.bar(x, values, color=colors, width=0.64)
    for bar, val in zip(bars, values):
        y = val + (0.015 if val >= 0 else -0.015)
        va = "bottom" if val >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{val:+.4f}", ha="center", va=va, fontsize=8)
    finite = [v for v in values if np.isfinite(v)]
    ymin = min(finite + [0.0])
    ymax = max(finite + [0.0])
    pad = max(0.04, 0.18 * (ymax - ymin if ymax > ymin else 1.0))
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("D. paired P3 to I1 endpoints", loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel("native metric units")
    ax.text(
        0.02,
        0.98,
        (
            f"n_perm={n_perm}; support driver={_support_driver(rows)}\n"
            f"H3b p_rate={_as_float(h3b.get('p_rate_delta')):.3f}, p_label={_as_float(h3b.get('p_label_delta')):.3f}; "
            f"H3c p_phase={_as_float(h3c.get('p_phase')):.3f}, p_block={_as_float(h3c.get('p_block')):.3f}, "
            f"p_label={_as_float(h3c.get('p_label')):.3f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color="0.25",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.85", "alpha": 0.92},
    )


def _write_readme(outdir: Path, subject: str, n_perm: int) -> None:
    text = f"""### {subject}_v3a_case_panel.png

这张图是 V3a 的单被试个案图，不是 cohort 主结论图。它展示 {subject} 为什么被标成最一致的探索性个案：H3a 轴向组织度下降，H3b 的 raw flux 下降但 rate-null surplus 为正，H3c 模态密度从轴向相对转向非轴向。图中 `support driver` 来自正式 V3a hard-gate 逻辑，当前图运行使用 n_perm={n_perm}。

**关注点**：看 D 面板里 raw flux 与 surplus 的反向关系，以及 C 面板里 I1 的 non-axis mode density 是否高于 P3；这只能支持 case-level hypothesis-generating wording，不能写成 cohort 级轴向到非轴向模态转移。
"""
    (outdir / "README.md").write_text(text, encoding="utf-8")


def main() -> Path:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", default="epilepsiae_1125")
    ap.add_argument("--cohort", choices=["narrow", "broad"], default="narrow")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--force-recompute", action="store_true")
    args = ap.parse_args()

    cfg = load_v3_config()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    table_dir = outdir / "tables" / args.subject

    cc = classify_subject_contacts(args.subject, args.cohort, cfg)
    if not cc["geometry_sufficient"]:
        raise RuntimeError(f"{args.subject} geometry insufficient: {cc['geometry_reason']}")

    rows = _load_or_run_endpoint_rows(args.subject, args.cohort, cfg, args.n_perm, table_dir, args.force_recompute)
    metrics = _compute_case_metrics(args.subject, args.cohort, cfg, cc)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    _plot_geometry(axes[0, 0], cc)
    _plot_trajectory(axes[0, 1], metrics)
    _plot_mode_parts(axes[1, 0], metrics)
    _plot_endpoint_bars(axes[1, 1], rows, args.n_perm)
    fig.suptitle(
        f"{args.subject} V3a case panel: internally consistent non-axis candidate",
        fontsize=14,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    fig.text(
        0.02,
        0.012,
        "Case-level exploratory figure. It visualizes internal consistency; it does not establish cohort-level mode transition.",
        fontsize=9,
        color="0.30",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_png = outdir / f"{args.subject}_v3a_case_panel.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "subject": args.subject,
        "cohort": args.cohort,
        "n_perm": args.n_perm,
        "support_driver": _support_driver(rows),
        "geometry": {
            "n_axis": cc["n_axis"],
            "n_nonaxis": cc["n_nonaxis"],
            "n_ambiguous": cc["n_ambiguous"],
        },
        "endpoints": rows,
        "trajectory": metrics,
    }
    (outdir / f"{args.subject}_v3a_case_metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_readme(outdir, args.subject, args.n_perm)
    print(f"[fig] -> {out_png}", flush=True)
    return out_png


if __name__ == "__main__":
    main()
