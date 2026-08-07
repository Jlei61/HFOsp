#!/usr/bin/env python
"""Adjudicate the per-cell adaptation arms, and build the two slow-variable pictures.

Three things come out of this, one per question the run was launched to answer.

**Did anything terminate.**  A bout whose end is the end of the record has not terminated; the
brake adjudication needed the same guard, and without it every arm -- including the control that
never stops -- reads as "terminated and silenced".  Each arm gets one of four verdicts, and the
one that matters is separated from the three ways of missing it.

**The temporal route** (the phase plane): mean adaptation against mean disinhibition, D = 1 - z,
both read from the same 250 ms snapshots, with the entry marked.  A trajectory that turns a
corner is a different object from one that runs to a wall, and the two are told apart by looking,
which is why this is a picture and not a number.

**The spatial route** (the modes): the same snapshots carry the per-cell adaptation and
disinhibition fields binned to a grid.  Subtracting the pre-entry mean field from the field
around entry gives what changed, and its leading singular vectors give the shape of the change --
flat, one lobe, or two.

The adjudication is what the run decides.  The two pictures are how it decides it, and neither is
a claim about the patient: this is a model substrate with a virtual montage.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN = ROOT / "results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/percell_adaptation"
FIGS = RUN / "figures"

A_COL = "#b2182b"       # adaptation
D_COL = "#2166ac"       # disinhibition
ENTRY_COL = "#111111"


def _verdict(rec):
    """Four outcomes, kept apart because they call for different next steps."""
    onset, offset = rec.get("onset_ms"), rec.get("offset_ms")
    run_ms = float(rec["run_ms"])
    if onset is None:
        return "never entered", "no bout in the record"
    if not rec.get("terminated"):
        return ("never stopped",
                "entered and did not terminate inside the window (the bout runs to the end of "
                "the record; no autonomous termination observed)")
    check = rec.get("return_check") or {}
    if check.get("returned"):
        return ("closed the loop",
                f"entered at {onset / 1000:.1f} s, terminated at {offset / 1000:.1f} s, and "
                f"returning events came back inside the reference band")
    return ("stopped but did not recover",
            f"terminated at {offset / 1000:.1f} s of {run_ms / 1000:.0f} s, but returning events "
            f"did not come back inside the reference band ({check.get('reason', 'no reason')})")


def _entry_intact(rec, baseline_n=12):
    """Whether the pre-entry train still looks like the one the frozen run produced.

    An arm that terminates by never letting the tissue build up has not solved the problem it was
    given, so this is reported next to the verdict rather than under it.
    """
    n = rec.get("n_returning_before_onset")
    return dict(n_returning_before_onset=n, reference_n=baseline_n,
                entry_class=rec.get("entry_class"),
                looks_intact=(n is not None and n >= 0.5 * baseline_n))


def _phase_plane(ax, arms):
    for name, rec, z in arms:
        if z is None:
            continue
        t = np.asarray(z["snapshot_t_ms"], float) / 1000.0
        m = np.asarray(z["m_mean"], float)
        d = 1.0 - np.nanmean(np.asarray(z["z_grid"], float), axis=(1, 2))
        ax.plot(d, m, lw=1.4, alpha=0.9, label=f"{name} (eta_m={rec['eta_m']:g})")
        on = rec.get("onset_ms")
        if on is not None:
            i = int(np.argmin(np.abs(t - on / 1000.0)))
            ax.scatter([d[i]], [m[i]], s=34, c=ENTRY_COL, zorder=6)
        off = rec.get("offset_ms")
        if off is not None and rec.get("terminated"):
            j = int(np.argmin(np.abs(t - off / 1000.0)))
            ax.scatter([d[j]], [m[j]], s=44, marker="s", facecolor="none",
                       edgecolor=ENTRY_COL, lw=1.4, zorder=6)
    ax.set_xlabel(r"disinhibition  $D = 1 - \bar{z}$", fontsize=9)
    ax.set_ylabel(r"adaptation  $\bar{m}$", fontsize=9)
    ax.set_title("temporal route", fontsize=9.5, fontweight="bold")
    ax.tick_params(labelsize=7.5)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left")


def _timecourse(ax, rec, z):
    t = np.asarray(z["snapshot_t_ms"], float) / 1000.0
    m = np.asarray(z["m_mean"], float)
    d = 1.0 - np.nanmean(np.asarray(z["z_grid"], float), axis=(1, 2))
    ax.plot(t, m / max(m.max(), 1e-12), color=A_COL, lw=1.3, label=r"adaptation $\bar{m}$")
    ax.plot(t, d / max(d.max(), 1e-12), color=D_COL, lw=1.3, label=r"disinhibition $D$")
    for key, style in (("onset_ms", "-"), ("offset_ms", "--")):
        v = rec.get(key)
        if v is not None and (key == "onset_ms" or rec.get("terminated")):
            ax.axvline(v / 1000.0, color=ENTRY_COL, lw=1.2, ls=style, zorder=5)
    ax.set_xlabel("time (s)", fontsize=9)
    ax.set_ylabel("each scaled to its own max", fontsize=8)
    ax.set_title("the two slow variables", fontsize=9.5, fontweight="bold")
    ax.tick_params(labelsize=7.5)
    ax.legend(frameon=False, fontsize=7.2, loc="lower right")


def _modes(z, rec, pre_s=(1.0, 4.0), around_s=1.5):
    """What the adaptation field looks like before entry, and what entry added to it.

    The change field is the entry-window mean minus the pre-entry mean; its leading singular
    vectors are the shape of the change.  Reporting the singular values alongside is what keeps
    "two lobes" from being read into a field that has none.
    """
    t = np.asarray(z["snapshot_t_ms"], float) / 1000.0
    g = np.asarray(z["m_grid"], float)
    on = rec.get("onset_ms")
    pre = (t >= pre_s[0]) & (t <= pre_s[1])
    if on is None or not pre.any():
        return None
    win = (t >= on / 1000.0 - around_s) & (t <= on / 1000.0 + around_s)
    if not win.any():
        return None
    base = np.nanmean(g[pre], axis=0)
    peri = np.nanmean(g[win], axis=0)
    delta = peri - base
    clean = np.nan_to_num(delta, nan=0.0)
    u, s, vt = np.linalg.svd(clean, full_matrices=False)
    return dict(baseline=base, peri=peri, delta=delta,
                singular=s, mode1=np.outer(u[:, 0], vt[0]) * s[0],
                share1=float(s[0] ** 2 / max(np.sum(s ** 2), 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="")
    args = ap.parse_args()

    arms = []
    for path in sorted(RUN.glob("arm_*.json")):
        rec = json.load(open(path))
        if rec.get("status") != "COMPLETE":
            continue
        if args.arms and rec["arm"] not in set(args.arms.split(",")):
            continue
        npz = Path(rec["output_npz"])
        z = np.load(npz) if npz.is_file() else None
        arms.append((rec["arm"], rec, z))
    if not arms:
        raise SystemExit(f"no completed arms under {RUN}")

    rows = []
    for name, rec, z in arms:
        v, why = _verdict(rec)
        md = _modes(z, rec) if z is not None else None
        rows.append(dict(arm=name, eta_m=rec["eta_m"], mean_field=rec["m_mean_field"],
                         verdict=v, why=why, stage=rec["stage"],
                         onset_ms=rec.get("onset_ms"), offset_ms=rec.get("offset_ms"),
                         terminated=rec.get("terminated"),
                         entry=_entry_intact(rec),
                         m_end_mean=rec.get("m_end_mean"), m_end_max=rec.get("m_end_max"),
                         wear_end=rec.get("wear_end"),
                         mode1_share=(None if md is None else md["share1"])))
        kind = "meanfield" if rec["m_mean_field"] else "per-cell"
        print(f"  {name:>18}  eta_m={rec['eta_m']:<5g} {kind:>9}  {v:<28} "
              f"pre-entry events {rec.get('n_returning_before_onset')}  "
              f"m_end {rec.get('m_end_mean'):.1f}")

    FIGS.mkdir(parents=True, exist_ok=True)
    with_z = [a for a in arms if a[2] is not None]
    if with_z:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), facecolor="white")
        _phase_plane(axes[0], with_z)
        _timecourse(axes[1], with_z[-1][1], with_z[-1][2])
        fig.tight_layout()
        fig.savefig(FIGS / "percell_adaptation_temporal_route.png", dpi=200)
        fig.savefig(FIGS / "percell_adaptation_temporal_route.pdf")
        plt.close(fig)

        for name, rec, z in with_z:
            md = _modes(z, rec)
            if md is None:
                continue
            fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5), facecolor="white")
            for ax, (field, title) in zip(axes, ((md["baseline"], "baseline field"),
                                                 (md["peri"], "around entry"),
                                                 (md["mode1"], "leading mode of the change"))):
                im = ax.imshow(field, origin="lower", cmap="magma", aspect="equal")
                ax.set_title(title, fontsize=9.5, fontweight="bold")
                ax.set_xticks([]), ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.046)
            axes[2].set_xlabel(f"leading mode carries {md['share1'] * 100:.0f}% of the change",
                               fontsize=8)
            fig.suptitle(f"{name} — spatial route (adaptation field)", fontsize=10)
            fig.tight_layout()
            fig.savefig(FIGS / f"percell_adaptation_spatial_{name}.png", dpi=200)
            plt.close(fig)

    out = RUN / "adjudication.json"
    out.write_text(json.dumps(dict(schema="fcxr-lc3-percell-adaptation-1.0", rows=rows), indent=2))
    print(f"\n  wrote {out.relative_to(ROOT)} and figures under {FIGS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
