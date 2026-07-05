#!/usr/bin/env python3
"""A1c PILOT analysis (spec §5/§6 + P1 review). MECHANISM-SCREEN go/no-go on the Abbott termination
prediction — reported as "global feedback CAN/CANNOT quench the MEASURED runaway corner under this
substrate / these magnitudes", NEVER "inhibitory exhaustion validated" (that = A2).

Per run (cell × gain × tau × seed) reads readout_*.json (activity.tail_to_baseline_ratio etc.) + the
fb_*.npz (I_global + rate_E traces, gain>0 only). Termination uses the ABSOLUTE tail-to-baseline ratio
(NOT the per-event-relative `returned`, which reads a plateau as terminated). A self-terminating run
must (a) IGNITE (not be suppressed-from-the-start), (b) drop to tail≤1.5×baseline sustained, AND
(c) have I_global LEAD the rate decay (dynamic, not a static DC brake). Until a matched-static control
is run, a positive read is "consistent with dynamic feedback", NOT "static brake ruled out" (P1-4).

TWO CELLS (P1-1 semantics):
  l2g1.0 = the runaway ANCHOR (strong local loop). gain=0 baseline tail ~570-1101x => runaway.
  l1g1.0 = a WORKING/ACTIVE-STATE PRESERVATION control (milder local loop). NOTE: at gain=0 it is
           ALSO elevated (tail ~110-338x) -- it is NOT a clean "able-to-return seizure-like" state.
           A1c asks of it only: does the feedback OVER-SUPPRESS the working state to silent?
JOINT WINDOW (P1-2, machine-readable, not eyeballed): the central question is whether ONE (gain,tau)
  simultaneously (i) terminates the anchor (all seeds ignite+return), (ii) keeps the preservation
  control NOT silent, AND (iii) has I_global lead the decay. status.joint_window_by_gain_tau records
  each leg per (gain,tau) so "no window does both" is derived, not asserted.
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = "results/topic4_sef_hfo/m3a_slowvars/a1c_pilot"
TAIL_GATE = 1.5            # absolute tail-to-baseline ratio for "returned to baseline"
IGNITE_PEAK = 0.05         # active_E_fraction_peak above this = the event provably ignited


def _parse_tag(tag):
    # l2g1.0_g0_s1 | l2g1.0_g16_t150_s2 | l2g1.0_g16_t2000_s3
    cell = tag.split("_g")[0]                       # l2g1.0 / l1g1.0
    seed = int(tag.rsplit("_s", 1)[1])
    body = tag[len(cell) + 1:].rsplit("_s", 1)[0]   # g0 | g16_t150
    gain = float(body.split("_")[0][1:])
    tau = float(body.split("_t")[1]) if "_t" in body else 0.0
    return cell, gain, tau, seed


def _leads_decay(igb, rate):
    """I_global LEADS the rate decay if I_global rises BEFORE -d(rate)/dt peaks. Cross-correlate
    I_global(t) with -d(rate)/dt; positive lead lag (I_global earlier) => dynamic, not bystander."""
    igb = np.asarray(igb, float); rate = np.asarray(rate, float)
    n = min(len(igb), len(rate))
    if n < 20 or igb.std() < 1e-9:
        return None, None
    igb, rate = igb[:n], rate[:n]
    drate = -np.gradient(rate)                       # rate DECAY signal
    a = (igb - igb.mean()) / (igb.std() + 1e-12)
    b = (drate - drate.mean()) / (drate.std() + 1e-12)
    lags = np.arange(-n + 1, n)
    xc = np.correlate(a, b, mode="full") / n
    best = lags[int(np.argmax(xc))]                  # >0 => I_global leads the decay (in bins)
    return int(best), round(float(xc.max()), 3)


def _load(base):
    runs = []
    for f in glob.glob(os.path.join(base, "readout_*.json")):
        if os.path.basename(f).startswith("readout_ctrl_"):
            continue                                     # P1-3 --fb-control re-runs (own fbctrl_*.json) — not pilot cells
        s = json.load(open(f)); c = s["config"]; act = s["activity"]
        cell, gain, tau, seed = _parse_tag(s["tag"])
        lead = leadcorr = None
        fb = os.path.join(base, f"fb_{s['tag']}.npz")
        if gain > 0 and os.path.exists(fb):
            z = np.load(fb); lead, leadcorr = _leads_decay(z["I_global_bin"], z["global_E_rate_bin"])
        runs.append(dict(
            tag=s["tag"], cell=cell, gain=gain, tau=tau, seed=seed,
            tail=act["tail_to_baseline_ratio"], peakE=act["peak_E_rate_hz"], coreR=act["core_E_rate_mean_hz"],
            gR=act["global_E_rate_mean_hz"], active_peak=act["active_E_fraction_peak"],
            n_events=s["n_events"], a1c=s.get("a1c"),
            lead_bins=lead, lead_corr=leadcorr))
    return runs


def _label(r, control_tail):
    """silent_suppressed / interictal / seizure_like / terminated / runaway (descriptive)."""
    ignited = (r["active_peak"] >= IGNITE_PEAK) or (r["peakE"] >= 3.0)
    returned = r["tail"] is not None and r["tail"] <= TAIL_GATE
    if not ignited:
        return "silent_suppressed"                   # never crossed event level (over-suppression)
    if returned:
        # ignited AND returned. terminated only if the gain=0 control of this cell/seed was elevated.
        return "terminated" if (control_tail is not None and control_tail > TAIL_GATE) else "seizure_like"
    return "runaway"                                 # ignited but stays elevated to T_end


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    base = os.path.join(ROOT, argv[0]) if argv else os.path.join(ROOT, DEFAULT_DIR)
    runs = _load(base)
    if not runs:
        print(f"[a1c] no readouts in {base}"); return 1
    # gain=0 control tail per (cell, seed)
    ctrl = {(r["cell"], r["seed"]): r["tail"] for r in runs if r["gain"] == 0}
    for r in runs:
        r["state"] = _label(r, ctrl.get((r["cell"], r["seed"])))

    # verdict per (cell, gain, tau): sign-consistent across seeds
    from collections import defaultdict
    cells = defaultdict(list)
    for r in runs:
        cells[(r["cell"], r["gain"], r["tau"])].append(r)
    cellagg = {}
    for k, rs in sorted(cells.items()):
        states = [r["state"] for r in rs]
        n_term = sum(1 for r in rs if r["state"] == "terminated")
        leads = [r["lead_bins"] for r in rs if r["lead_bins"] is not None]
        cellagg["_".join(str(x) for x in k)] = dict(
            n_seeds=len(rs), states=states, n_terminated=n_term,
            tail_med=round(float(np.median([r["tail"] for r in rs])), 2),
            peakE_med=round(float(np.median([r["peakE"] for r in rs])), 2),
            all_terminated=bool(n_term == len(rs)),
            i_global_leads_all=bool(leads and all(l > 0 for l in leads)),
            i_global_ratio=(rs[0]["a1c"]["I_global_to_I_I_ratio"] if rs[0].get("a1c") else None))

    # the central question, per cell + tau: is there an intermediate gain where all seeds TERMINATE?
    verdict = {}
    for cell in ("l2g1.0", "l1g1.0"):
        for tau in sorted(set(r["tau"] for r in runs if r["cell"] == cell and r["gain"] > 0)):
            gains = sorted(set(r["gain"] for r in runs if r["cell"] == cell and r["tau"] == tau))
            term_gains = [g for g in gains if cellagg.get(f"{cell}_{g}_{tau}", {}).get("all_terminated")]
            leads_ok = all(cellagg.get(f"{cell}_{g}_{tau}", {}).get("i_global_leads_all") for g in term_gains) if term_gains else False
            if term_gains and leads_ok:
                v = "CAN_QUENCH (consistent with dynamic feedback; static brake NOT yet ruled out)"
            elif term_gains:
                v = "QUENCHES but I_global does not lead decay -> looks static/bystander (INCONCLUSIVE)"
            else:
                states_by_gain = {g: cellagg.get(f"{cell}_{g}_{tau}", {}).get("states") for g in gains}
                v = f"CANNOT_QUENCH (no all-seed terminating gain). states_by_gain={states_by_gain}"
            verdict[f"{cell}_tau{tau}"] = v

    # P1-2 MACHINE-READABLE JOINT WINDOW: per (gain,tau), does ONE operating point simultaneously
    #   (i) terminate the anchor l2g1.0 (all seeds ignite+return),
    #   (ii) keep the preservation control l1g1.0 NOT silent (working state survives), AND
    #   (iii) have the anchor's I_global LEAD its rate decay (dynamic, not a static over-brake)?
    # Recorded per leg so "no window terminates AND preserves" is derived, not read off a table by eye.
    joint_window = {}
    j_gains = sorted(set(r["gain"] for r in runs if r["gain"] > 0))
    j_taus = sorted(set(r["tau"] for r in runs if r["tau"] > 0))
    for tau in j_taus:
        for g in j_gains:
            l2 = cellagg.get(f"l2g1.0_{g}_{tau}")
            l1 = cellagg.get(f"l1g1.0_{g}_{tau}")
            if l2 is None and l1 is None:
                continue
            anchor_terminated = bool(l2 and l2["all_terminated"])
            preservation_not_silent = bool(l1 and "silent_suppressed" not in l1["states"])
            leads = bool(l2 and l2["i_global_leads_all"])
            fail = []
            if not anchor_terminated:
                fail.append("anchor_not_all_terminated" if l2 else "no_anchor_run")
            if not preservation_not_silent:
                fail.append("preservation_silenced" if l1 else "no_preservation_run")
            if not leads:
                fail.append("i_global_not_leading_decay")
            joint_window[f"g{g:g}_tau{tau:g}"] = {
                "joint_pass": bool(anchor_terminated and preservation_not_silent and leads),
                "anchor_l2g1.0_all_terminated": anchor_terminated,
                "preservation_l1g1.0_not_silent": preservation_not_silent,
                "i_global_leads_decay": leads,
                "anchor_states": (l2["states"] if l2 else None),
                "preservation_states": (l1["states"] if l1 else None),
                "fails": fail}
    joint_exists = any(v["joint_pass"] for v in joint_window.values())

    status = {"base": os.path.relpath(base, ROOT), "tier": "MECHANISM-SCREEN (NOT exhaustion validation)",
              "tail_gate": TAIL_GATE, "control_tail_by_cell_seed": {f"{k[0]}_s{k[1]}": v for k, v in ctrl.items()},
              "per_cell_gain_tau": cellagg, "verdict": verdict,
              "joint_window_exists": joint_exists, "joint_window_by_gain_tau": joint_window,
              "caveat": ("A1c = dynamic global feedback RESTRAINT screen. Allowed: feedback CAN/CANNOT quench the "
                         "measured runaway. Forbidden: inhibitory-exhaustion-mechanism validated (=A2). 'CAN_QUENCH' "
                         "is consistent with dynamic feedback; ruling out a matched static DC brake needs the P1-4 control.")}
    json.dump(status, open(os.path.join(base, "status_a1c_pilot.json"), "w"), indent=1)

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = os.path.join(base, "figures"); os.makedirs(fig_dir, exist_ok=True)
    STATE_C = {"silent_suppressed": 0, "interictal": 1, "seizure_like": 2, "terminated": 3, "runaway": 4}
    CMAP = plt.matplotlib.colors.ListedColormap(["#cfe8ff", "#9ad27f", "#e8a33d", "#2e8b57", "#c0392b"])
    # Fig 1: state surface gain x tau, one panel per cell (gain=0 column shown as the control)
    cells_l = ["l2g1.0", "l1g1.0"]
    gains = sorted(set(r["gain"] for r in runs)); taus = sorted(set(r["tau"] for r in runs if r["tau"] > 0)) or [150.0]
    fig, axes = plt.subplots(1, len(cells_l), figsize=(7 * len(cells_l), 4.2), squeeze=False)
    for ci, cell in enumerate(cells_l):
        M = np.full((len(taus), len(gains)), np.nan)
        for i, tau in enumerate(taus):
            for j, g in enumerate(gains):
                key = f"{cell}_{g}_{0.0 if g == 0 else tau}"
                a = cellagg.get(key)
                if a and a["states"]:
                    from collections import Counter
                    M[i, j] = STATE_C.get(Counter(a["states"]).most_common(1)[0][0], np.nan)
        ax = axes[0][ci]
        im = ax.imshow(M, origin="lower", aspect="auto", cmap=CMAP, vmin=0, vmax=4)
        ax.set_xticks(range(len(gains))); ax.set_xticklabels([f"g{g:g}" for g in gains])
        ax.set_yticks(range(len(taus))); ax.set_yticklabels([f"tau{t:g}" for t in taus])
        ax.set_title(f"{cell} ({'runaway anchor' if cell=='l2g1.0' else 'working-state preservation control'})")
        for i in range(len(taus)):
            for j in range(len(gains)):
                if not np.isnan(M[i, j]):
                    lbl = ["silent", "inter", "seiz", "TERM", "run"][int(M[i, j])]
                    ax.text(j, i, lbl, ha="center", va="center", fontsize=8,
                            color="white" if M[i, j] in (3, 4) else "black")
    fig.suptitle("A1c dynamic global feedback RESTRAINT screen — gain × tau state (NOT exhaustion validation)", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "a1c_state_surface.png"), dpi=130); plt.close(fig)

    # Fig 2: allow-then-quench money figure — the clearest terminating run (or closest), rate + I_global
    term = [r for r in runs if r["state"] == "terminated" and r["gain"] > 0]
    pick = (min(term, key=lambda r: r["tail"]) if term
            else min((r for r in runs if r["gain"] > 0), key=lambda r: r["tail"], default=None))
    if pick is not None:
        fb = os.path.join(base, f"fb_{pick['tag']}.npz")
        if os.path.exists(fb):
            z = np.load(fb); rate = z["rate_E_hz"]; igb = z["I_global_bin"]
            t_r = np.arange(len(rate)) * DT_PLOT; t_i = np.arange(len(igb)) * 1.0
            fig, ax1 = plt.subplots(figsize=(11, 4))
            ax1.plot(t_r, rate, color="tab:red", lw=0.8, label="global E rate (Hz)")
            ax1.set_xlabel("time (ms)"); ax1.set_ylabel("global E rate (Hz)", color="tab:red")
            ax2 = ax1.twinx(); ax2.plot(t_i, igb, color="tab:blue", lw=1.0, label="I_global")
            ax2.set_ylabel("I_global", color="tab:blue")
            ax1.set_title(f"A1c allow-then-quench — {pick['tag']} (state={pick['state']}, "
                          f"tail={pick['tail']}, I_global leads decay by {pick['lead_bins']} bins)")
            fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "a1c_allow_then_quench.png"), dpi=130); plt.close(fig)

    print("[a1c] control runaway tails:", {f"{k[0]}_s{k[1]}": v for k, v in ctrl.items()})
    print(f"[a1c] JOINT WINDOW (terminate anchor AND preserve working state AND I_global leads) exists: {joint_exists}")
    for k, v in joint_window.items():
        print(f"[a1c]   {k}: joint={v['joint_pass']} fails={v['fails']}")
    for k, v in verdict.items():
        print(f"[a1c] {k}: {v}")
    print(f"[a1c] wrote {base}/status_a1c_pilot.json + figures/")
    return 0


DT_PLOT = 0.1
if __name__ == "__main__":
    sys.exit(main())
