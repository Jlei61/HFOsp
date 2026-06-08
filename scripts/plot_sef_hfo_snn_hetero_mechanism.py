"""Three figure classes from the heterogeneity grid (spec 2026-06-08 §5):
  (1) grid_overview.png   — core effect over the (kick × core) grid
  (2) mechanism_<tag>.png — representative cells, reuse two_electrode_readout
                            (panel-a coloured by LOCAL V_th spread + patch outline)
  (3) baseline_compare.png— baseline vs core population read-out for reps
Representative pick rule is PRE-REGISTERED (spec §4): max effect / near-zero /
kick-outside-wave-through-core / matched-vs-unmatched.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.sef_hfo_plot import two_electrode_readout
from src.sef_hfo_heterogeneity import local_vth_spread

OUT = Path("results/topic4_sef_hfo/snn_heterogeneity")
FIG = OUT / "figures"


def _effect(c):
    """Primary grid effect = |Δ core synchrony| (fall back to whole-sheet Δpaf)."""
    v = c["d_core_paf"] if c["d_core_paf"] is not None else c["d_peak_active_frac"]
    return abs(v) if v is not None else 0.0


def _snn_electrode(lfp, contacts, names, t, win, u_shaft, panel_title):
    """Copy of plot_sef_hfo_two_electrode_readout._snn_electrode (same read-out)."""
    pre = t < win[0]; inwin = (t >= win[0]) & (t <= win[1])
    base = lfp[:, pre].mean(axis=1); bstd = lfp[:, pre].std(axis=1) + 1e-9
    pz = (lfp[:, inwin].max(axis=1) - base) / bstd
    part = pz > 30.0
    c = np.asarray(contacts, float)
    return dict(contacts=c, part=part, names=list(names),
                s=(c - c.mean(0)) @ u_shaft, signal=lfp, panel_title=panel_title)


def _pick_representatives(cells):
    """Distinct reps (spec §4), prioritising the HEADLINE mid-patch matched-vs-
    unmatched dissociation (the 3-seed pair: variance-null vs mean-effect), plus
    the max-effect and near-zero cells. Off-patch preferred (mechanism evidence)."""
    picked, used = {}, set()

    def take(tag, c):
        if c is not None and c["idx"] not in used:
            picked[tag] = c; used.add(c["idx"])

    def find(pname, cond, off=True, sweep=None):
        cand = [c for c in cells if c["pname"] == pname and c["cond"] == cond
                and (not off or not c["kick_on_patch"])
                and (sweep is None or c["sweep"] == sweep)]
        return max(cand, key=_effect) if cand else None

    # mid pair pinned to sweep-2 (kick=end) so they share the same kick == the 3-seed pair
    take("mid_matched", find("mid", "matched", sweep=2))    # variance-only (mean held) ~ null
    take("mid_unmatched", find("mid", "unmatched", sweep=2))  # + mean shift (hyperexcitable)
    take("maxeffect", max(cells, key=_effect))
    take("nearzero", min(cells, key=_effect))
    return picked


def _overview(cells):
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = [f"s{c['sweep']}\n{c['kname']}/{c['pname']}\n{c['cond'][:4]}" for c in cells]
    vals = [(c["d_core_paf"] if c["d_core_paf"] is not None else np.nan) for c in cells]
    colors = ["crimson" if c["kick_on_patch"] else "steelblue" for c in cells]
    ax.bar(range(len(cells)), vals, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(cells))); ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("Δ core synchrony (core − baseline)")
    ax.set_title("Heterogeneity grid: pathology-core effect on in-core synchrony\n"
                 "(red = kick-on-patch — NOT mechanism evidence, spec §1)")
    fig.tight_layout(); fig.savefig(FIG / "grid_overview.png", dpi=130); plt.close(fig)


def _mechanism(cell, tag):
    z = np.load(OUT / "per_cell" / f"cell{cell['idx']:02d}.npz", allow_pickle=True)
    posE = z["posE"]; vth = z["vth"]; NE = len(posE)
    spread = local_vth_spread(posE, vth[:NE], np.ones(NE, bool), 0.3)
    L = float(z["L"]); patch = z["patch"]; pr = float(z["patch_r"]); theta = float(z["theta"])
    nc = int(z["nc"])
    t = np.asarray(z["times"]); lfp = np.asarray(z["lfp"]).T          # (n_contact, nt)
    contacts = z["contacts"]; names = [str(s) for s in z["names"]]
    ev_t = float(z["event_peak_t"]) if "event_peak_t" in z.files else 176.0
    win = (ev_t - 25.0, ev_t + 30.0)                                 # event-locked read-out
    u_par = np.array([np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))])
    u_perp = np.array([np.cos(np.deg2rad(theta + 90)), np.sin(np.deg2rad(theta + 90))])
    par = _snn_electrode(lfp[:nc], contacts[:nc], names[:nc], t, win, u_par,
                         "∥ axis — peaks sweep (reads direction)")
    perp = _snn_electrode(lfp[nc:2 * nc], contacts[nc:2 * nc], names[nc:2 * nc], t, win,
                          u_perp, "⊥ axis — peaks aligned (no direction)")
    two_electrode_readout(
        str(FIG / f"mechanism_{tag}.png"),
        field_xy=posE, field_c=spread, field_clabel="local V_th spread (mV)",
        E_xy=None, kick_xy=z["kick"], axis_deg=theta, extent=(0, L, 0, L),
        par=par, perp=perp, t=t, event_window=win,
        signal_ylabel="current-LFP (|I_E|+|I_I|)", name_fs=8, label_endpoints_only=True,
        substrate_label=f"Spiking · pathology core ({cell['cond']}) · {tag}",
        contact_note="contacts SCALED to model sheet — NOT real SEEG spacing; "
                     "firing-density read-out (NOT LFP); dashed = pathology core",
        patch_circle=(float(patch[0]), float(patch[1]), pr),
        title="Pathology core in the spiking network — heterogeneity map + electrode read-out")


def _baseline_compare(reps):
    n = len(reps)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.2), squeeze=False)
    for ax, (tag, c) in zip(axes[0], reps.items()):
        z = np.load(OUT / "per_cell" / f"cell{c['idx']:02d}.npz", allow_pickle=True)
        ax.plot(z["base_times"], np.asarray(z["base_lfp"]).mean(1), color="0.6",
                lw=1.2, label="baseline (wide)")
        ax.plot(z["times"], np.asarray(z["lfp"]).mean(1), color="C3", lw=1.2, label="core")
        dc = c["d_core_paf"]
        ax.set_title(f"{tag}\nΔcore_sync={dc:+.3f}" if dc is not None else tag, fontsize=8)
        ax.set_xlabel("time (ms)"); ax.legend(fontsize=6)
    axes[0, 0].set_ylabel("mean current-LFP (over contacts)")
    fig.suptitle("Network state: baseline vs pathology core (mean over contacts)")
    fig.tight_layout(); fig.savefig(FIG / "baseline_compare.png", dpi=130); plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    cells = json.loads((OUT / "grid_metrics.json").read_text())["cells"]
    _overview(cells)
    reps = _pick_representatives(cells)
    for tag, c in reps.items():
        _mechanism(c, tag)
    _baseline_compare(reps)
    (OUT / "cohort_summary.json").write_text(json.dumps(
        {"representatives": {k: v["idx"] for k, v in reps.items()},
         "n_cells": len(cells)}, indent=2))
    print("representatives:", {k: v["idx"] for k, v in reps.items()})
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
