"""M3B Round-1 Task 1 pilot — model → virtual-SEEG record adapter + 45° sanity gate.

Plan-of-record: docs/superpowers/plans/2026-06-21-sef-hfo-m3-local-w-propagation-operator-plan.md
(commit 19c3398; the rewritten "model↔SEEG bridge" version — NOT the working-tree copy, which is
the superseded a1213ee version).

Round 1 = the bridge. Task 1 is the load-bearing reuse: run the model through the SAME virtual-SEEG
observation pipeline the real `path_axis` records used (cluster rank → source/sink cores →
compute_axis_frame), build a `compare_model_to_cohort`-schema record, and pass the §4 INSTRUMENT
sanity gate — the virtual readout must recover the known 45° E→E axis within ~25°. If it fails, the
bridge is resolution-untestable and we STOP (plan §8). This pilot does the gate + the adapter; it
does NOT run the Task-2 bridge against the real cohort.

SUBSTRATE / EVENT-SOURCE — surfaced as a Task-2 fork, NOT silently resolved (CLAUDE.md §1/§5):
the validated 45° instrument is the KICK-DRIVEN LIF RATE FIELD (four-contrast C1, axis err 3.3°,
readability 0.977 — `increment3a_rate_parity_2026-06-07`). The plan's literal "collect SPONTANEOUS
events" on "one accepted Stage-3/M3 (cm-SNN) substrate" is UNVALIDATED: spontaneous readout is
untested at any scale, and the cm-SNN does not ignite at density=100. The §4 sanity gate asks "can
the montage RESOLVE the axis" — an instrument-resolution question that is event-source-independent,
so the validated kick-driven rate field is the legitimate instrument for it. Which model field to
BRIDGE FROM in Task 2 (validated kick-driven rate field as a labeled instrument probe, vs an
unvalidated spontaneous-cm-SNN run) is left to user review.

Run from the worktree root:  python scripts/run_m3b_task1_bridge_pilot.py
"""
import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
from src.sef_hfo_lif import mean_field                                            # noqa: E402
from src.lagpat_rank_audit import mask_phantom_ranks                              # noqa: E402
from src import propagation_skeleton_geometry as G                               # noqa: E402
from scripts.run_sef_hfo_obs_increment3a import _montage, _read, RATIO           # noqa: E402
from scripts.run_contact_plane_readout import build_record_from_events           # noqa: E402

OUT = "results/topic4_sef_hfo/m3b_bridge/task1_pilot"
GATE_DEG = 25.0          # plan §4 internal sanity gate
THETA_EE_DEG = 45.0      # exact, free: rho_EE=0.6 ellipse major axis (plan §0/§4)
_SCALAR_KEYS = ["axis_length_mm", "transverse_width_mm", "early_zone_spread",
                "late_zone_spread", "early_late_centroid_distance_norm",
                "rank_vs_xnorm_spearman"]


def _angle_mod180(vec):
    return float(np.degrees(np.arctan2(vec[1], vec[0])) % 180.0)


def _err_vs_45(angle_deg):
    d = abs((angle_deg - THETA_EE_DEG) % 180.0)
    return float(min(d, 180.0 - d))


def run():
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)

    # --- (1) the VALIDATED 45° rate-field instrument (increment3a C1 setup, verbatim) ---
    L, n, pitch, n_contacts = 24.0, 96, 4.0, 6
    shafts = (15.0, 75.0, 135.0)
    center = np.zeros(2)
    half = L / 2.0
    op = mean_field(RATIO)
    montage = _montage(center, pitch, n_contacts, shafts)
    # kick at the NEGATIVE θ_EE end (the C1 convention)
    end = center - 0.6 * half * np.array([np.cos(np.deg2rad(THETA_EE_DEG)),
                                          np.sin(np.deg2rad(THETA_EE_DEG))])
    r = _read(op, np.deg2rad(THETA_EE_DEG), 2.0, end, montage, np.deg2rad(THETA_EE_DEG),
              n, L, pitch, save_diag=True)

    if "_diag" not in r:
        raise SystemExit(f"[STOP] no event detected on the validated instrument: {r}. "
                         "Sanity gate cannot run -> bridge resolution-untestable (plan §8).")

    obs_axis_err = r["axis_err"]
    recovered_axis = r["_diag"]["recovered_axis"]
    art = r["_diag"]["artifact"]
    gate_pass = (obs_axis_err is not None) and (obs_axis_err < GATE_DEG)

    # --- (2) build the compare_model_to_cohort-schema record (THE adapter) ---
    coords2d = np.asarray(art.contact_coords, float)
    coords3d = np.column_stack([coords2d, np.zeros(len(coords2d))])     # z=0 (2D model frame)
    names = list(art.names)
    ranks = np.asarray(art.ranks, float)         # (n_contact, 1 event)
    bools = np.asarray(art.bools, bool)
    lag_raw = np.asarray(art.lag_raw, float)
    n_ch = len(names)
    rec = build_record_from_events(
        dataset="model", subject="lif_rate_45deg", template_id="t_a",
        names=names, ranks=ranks, bools=bools, lag_raw=lag_raw,
        coords=coords3d, mapped=np.ones(n_ch, bool), soz_core=set(),
        montage="single", lag_time_unit="ms", spacing_mm=pitch)

    rec_status = rec.get("status", "ok")
    schema_ok = (rec_status != "descriptive_only"
                 and isinstance(rec.get("channels"), list) and len(rec["channels"]) > 0
                 and isinstance(rec.get("scalars"), dict)
                 and all(k in rec["scalars"] for k in _SCALAR_KEYS))

    # --- (3) cross-check: the RECORD's own source/sink-centroid axis (what the bridge uses) ---
    masked = mask_phantom_ranks(ranks, bools, normalize=True)
    taxis = np.array([np.nanmean(rr) if np.any(~np.isnan(rr)) else np.nan for rr in masked])
    eligible = (~np.isnan(taxis)) & np.ones(n_ch, bool)
    cores = G.build_endpoint_cores(taxis, eligible, k_primary=3)
    rec_axis_deg = rec_axis_err = None
    if cores["tier"] != "descriptive_only":
        fr = G.compute_axis_frame(coords3d, cores["source_idx"], cores["sink_idx"])
        av = np.array(fr["sink_centroid"][:2]) - np.array(fr["source_centroid"][:2])
        rec_axis_deg = _angle_mod180(av)
        rec_axis_err = _err_vs_45(rec_axis_deg)

    n_part = int(bools[:, 0].sum())
    summary = {
        "task": "M3B round-1 Task 1 (model->virtual-SEEG record adapter + 45deg sanity gate)",
        "plan_commit": "19c3398",
        "substrate": "lif_rate_field",
        "event_source": "kick_driven",
        "event_source_note": ("VALIDATED instrument (increment3a C1, 3.3deg). Plan's literal "
                              "'spontaneous cm-SNN' is unvalidated; bridge-field choice = Task-2 fork."),
        "theta_EE_deg": THETA_EE_DEG,
        "montage": {"shafts_deg": list(shafts), "pitch_mm": pitch,
                    "n_contacts_per_shaft": n_contacts, "n_contacts_total": n_ch},
        "n_participating": n_part,
        "observation_axis_err_deg": obs_axis_err,
        "readability": r.get("readability"),
        "record_axis_deg": rec_axis_deg,
        "record_axis_err_deg": rec_axis_err,
        "sanity_gate_deg": GATE_DEG,
        "sanity_gate_pass": bool(gate_pass),
        "record_status": rec_status,
        "record_schema_ok": bool(schema_ok),
        "record_n_channels": len(rec.get("channels", [])),
        "record_scalars": rec.get("scalars"),
        "verdict": ("SANITY-PASS (instrument resolves 45deg AND adapter builds a valid "
                    "compare_model_to_cohort record)"
                    if (gate_pass and schema_ok) else
                    "STOP (gate or schema failed) — see plan §8"),
    }

    with open(os.path.join(OUT, "task1_pilot_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(OUT, "model_record_lif_rate_45deg.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else o)

    _figure(montage, art, recovered_axis, rec_axis_deg, obs_axis_err,
            os.path.join(OUT, "figures", "task1_axis_recovery.png"))

    print(json.dumps(summary, indent=2))
    if not (gate_pass and schema_ok):
        raise SystemExit("[STOP] Task-1 gate/schema failed — recap before Task 2 (plan §8).")
    return summary


def _figure(montage, art, recovered_axis, rec_axis_deg, obs_err, path):
    coords = np.asarray(art.contact_coords, float)
    ranks = np.asarray(art.ranks, float)[:, 0]
    bools = np.asarray(art.bools, bool)[:, 0]
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    part = bools
    sc = ax.scatter(coords[part, 0], coords[part, 1], c=ranks[part], cmap="viridis",
                    s=140, edgecolor="k", zorder=3, label="participating (rank)")
    ax.scatter(coords[~part, 0], coords[~part, 1], c="lightgray", s=90,
               edgecolor="gray", zorder=2, label="non-participating")
    plt.colorbar(sc, ax=ax, label="recruitment rank (early→late)")
    span = float(np.abs(coords).max()) * 1.05
    # known 45° E→E axis (dashed)
    t = np.linspace(-span, span, 2)
    ax.plot(t * np.cos(np.deg2rad(45)), t * np.sin(np.deg2rad(45)), "--",
            color="crimson", lw=2, label="known E→E axis (45°)", zorder=1)
    # recovered observation axis (endpoint-centroid)
    if recovered_axis is not None:
        v = np.asarray(recovered_axis, float)
        ax.annotate("", xy=(span * 0.7 * v[0], span * 0.7 * v[1]),
                    xytext=(-span * 0.7 * v[0], -span * 0.7 * v[1]),
                    arrowprops=dict(arrowstyle="->", color="navy", lw=2.5), zorder=4)
        ax.plot([], [], color="navy", lw=2.5, label="recovered readout axis")
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title(f"M3B Task-1 sanity gate — readout recovers 45° axis (err={obs_err}°)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    run()
