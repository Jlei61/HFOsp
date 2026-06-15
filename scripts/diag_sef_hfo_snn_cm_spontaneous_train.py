"""cm-SNN SPONTANEOUS-TRAIN diagnostic (user 2026-06-10: spontaneous-ignition route to the
virtual-SEEG template read-out — option B, the "还没站稳" path the cm archive flagged).

The continuation goal needs MANY events to cluster (masked PR-2 wants >=~20). The validated
cm read-out so far used ONE forced kick = ONE event. Before building the multi-event synthetic
record, this diagnostic answers the viability questions the spontaneous path hinges on — and it
does NOT assume the answer:

  Q1 (TRAIN vs SUSTAINED): does a self-igniting wide core, left to run with NO kick, produce a
     TRAIN of separable, self-terminating events (ignite -> propagate -> return-to-quiet ->
     re-ignite), or one runaway/sustained burst? Only a train is clusterable.
  Q2 (DIRECTIONAL): is each spontaneous event a directional FRONT (onset time monotonic along the
     connectivity axis theta_EE), or a symmetric/flash blob (no readable direction)?
  Q3 (BIDIRECTIONAL): across events, do fronts break symmetry to go BOTH ways along the axis
     (forward AND reverse) — the spontaneous route to the two opposite templates in the data —
     or always the same way?
  Q4 (SEED-ROBUST): are Q1-Q3 stable across noise seeds (the archive's single-realization risk)?

Event detection reuses the LOCKED src.sef_hfo_events.detect_events, but with the participation
bar calibrated to the SNN active-NEURON fraction baseline (the locked EVENT_ON_FRAC=0.05 was set
for the rate field's coherence-active-fraction, a different observable). Per-event direction =
sign+strength of the correlation between each E neuron's first-spike time and its projection on
the axis (forward = front advances toward +axis end). NO virtual electrodes / NO LFP / NO figure
here — this is the cheap viability gate before the full read-out build.
"""
import sys
import os
import json
import argparse
import numpy as np

ENG = os.path.join("src", "snn_engine")
sys.path.insert(0, ENG)
from params import Params, compute_nu_theta                # noqa: E402
from connectivity import place_neurons                      # noqa: E402
from connectivity_rot import build_connectivity_rot         # noqa: E402
from kick_probe import simulate_kick                        # noqa: E402

sys.path.insert(0, os.getcwd())
from src.sef_hfo_heterogeneity import sample_core_field     # noqa: E402
from src.sef_hfo_events import detect_events                 # noqa: E402
from src.sef_hfo_snn_engine_guard import assert_versions     # noqa: E402

OUT = "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous"
ENGINE_VERSIONS = os.path.join("results", "topic4_sef_hfo", "snn_heterogeneity", "engine_versions.json")
DT, DRIVE = 0.1, 0.6
BIN_MS = 1.0                       # active-fraction bin for event detection
BASELINE_MS = (5.0, 50.0)         # pre-ignition quiet window to calibrate the floor
CAL_FRAC = 0.5                     # bar = floor + CAL_FRAC*(peak-floor), same shape as calibrate_detector


def _engine_guard():
    if not os.path.exists(ENGINE_VERSIONS):
        raise RuntimeError(f"engine baseline missing: {ENGINE_VERSIONS}")
    assert_versions(json.loads(open(ENGINE_VERSIONS).read()))


def active_fraction(E_spk_bool, dt, bin_ms):
    """E active-neuron fraction per bin_ms bin (fraction of E that spiked at least once)."""
    bs = max(1, int(round(bin_ms / dt)))
    nb = E_spk_bool.shape[0] // bs
    binned = E_spk_bool[:nb * bs].reshape(nb, bs, -1).any(axis=1)   # (nb, NE)
    return binned.mean(axis=1), bs * dt                              # (nb,), bin width ms


def event_direction(E_spk_bool, posE, axis_unit, t_on, t_off, dt, min_n=20):
    """Per-event front direction: correlate each participating E neuron's first-spike time
    (inside the window) with its projection on axis_unit. Returns (corr, n, ratio, oracle_deg).
    corr>0 = front advances toward +axis end (forward); <0 = reverse; ~0 = symmetric/flash."""
    s, e = int(round(t_on / dt)), int(round(t_off / dt))
    seg = E_spk_bool[s:e]                          # (win, NE)
    fired = seg.any(axis=0)
    if fired.sum() < min_n:
        return None
    idx = np.flatnonzero(fired)
    first = np.argmax(seg[:, idx], axis=0).astype(float) * dt        # first-spike time in window
    proj = posE[idx] @ axis_unit
    if np.ptp(first) == 0 or np.ptp(proj) == 0:
        corr = 0.0
    else:
        corr = float(np.corrcoef(first, proj)[0, 1])
    c = posE[idx] - posE[idx].mean(axis=0)
    w, _ = np.linalg.eigh((c.T @ c) / len(c))
    ratio = float(np.sqrt(w.max() / max(w.min(), 1e-12)))
    cov = (c.T @ c) / len(c)
    wv, V = np.linalg.eigh(cov)
    mj = V[:, np.argmax(wv)]
    oracle = float(np.degrees(np.arctan2(mj[1], mj[0])) % 180.0)
    return dict(corr=round(corr, 3), n=int(fired.sum()), ratio=round(ratio, 2),
                oracle_deg=round(oracle, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, default=20.0)
    ap.add_argument("--density", type=float, default=100.0)
    ap.add_argument("--theta", type=float, default=45.0)
    ap.add_argument("--AR", type=float, default=2.0)
    ap.add_argument("--T", type=float, default=1500.0)         # long: room for a train
    ap.add_argument("--core-mean", type=float, default=17.0)   # wide self-igniting (cm fine scan)
    ap.add_argument("--core-std", type=float, default=1.5)
    ap.add_argument("--core-r", type=float, default=1.5)
    ap.add_argument("--core-at", choices=["center", "negend", "posend", "both"], default="center")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="spont_L20d100")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    _engine_guard()

    L, theta_rad = a.L, np.deg2rad(a.theta)
    axis_unit = np.array([np.cos(theta_rad), np.sin(theta_rad)])
    p = Params(g=3.6, L=L, density=a.density, T=a.T, dt=DT, nu_ext_ratio=DRIVE, seed=a.seed)
    rng = np.random.default_rng(a.seed)
    print(f"[{a.tag}] N~{int(a.density*L*L)} seed={a.seed} T={a.T} core_at={a.core_at} "
          f"m={a.core_mean} s={a.core_std} ...", flush=True)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=theta_rad, AR=a.AR, verbose=False)
    posE = net["pos"][:NE]
    center = np.array([L / 2, L / 2])
    half = L / 2
    neg_xy = center - 0.6 * half * axis_unit
    pos_xy = center + 0.6 * half * axis_unit
    is_E = np.zeros(len(net["pos"]), bool); is_E[:NE] = True
    if a.core_at == "both":
        # TWO foci, one near each end of the axis = the spontaneous route to bidirectionality
        # (each focus launches a front in opposite directions). Cores are ~12mm apart >> 2*core_r
        # so they are disjoint; element-wise min keeps each focus's lowered draw, surround stays base.
        cf1 = sample_core_field(net["pos"], is_E, neg_xy, a.core_r, np.random.default_rng(a.seed + 7),
                                core_mean=a.core_mean, core_std=a.core_std, base_mean=18.0)
        cf2 = sample_core_field(net["pos"], is_E, pos_xy, a.core_r, np.random.default_rng(a.seed + 8),
                                core_mean=a.core_mean, core_std=a.core_std, base_mean=18.0)
        vth = np.minimum(cf1["vth"], cf2["vth"])
        core_mask = cf1["core_mask"] | cf2["core_mask"]
        core_xy = center                       # report center as nominal; both foci recorded below
    else:
        core_xy = {"center": center, "negend": neg_xy, "posend": pos_xy}[a.core_at]
        cf = sample_core_field(net["pos"], is_E, core_xy, a.core_r, np.random.default_rng(a.seed + 7),
                               core_mean=a.core_mean, core_std=a.core_std, base_mean=18.0)
        vth = cf["vth"]
        core_mask = cf["core_mask"]

    net["rng"] = np.random.default_rng(a.seed)
    res = simulate_kick(p, net, KICK_BOOST=0.0, kick_center=list(core_xy), r_kick=a.core_r,
                        t_kick=1e9, V_th_per_neuron=vth)         # t_kick beyond T => pure spontaneous
    spk = res["E_spk_bool"]

    af, bin_w = active_fraction(spk, DT, BIN_MS)
    nb0, nb1 = int(BASELINE_MS[0] / bin_w), int(BASELINE_MS[1] / bin_w)
    floor = float(np.percentile(af[nb0:nb1], 95)) if nb1 > nb0 else float(af.min())
    peak = float(af.max())
    bar = floor + CAL_FRAC * (peak - floor)
    events = detect_events(af, bin_w, event_on_frac=bar)

    ev_out, corrs = [], []
    for ev in events:
        d = event_direction(spk, posE, axis_unit, ev["t_on"], ev["t_off"], DT)
        rec = dict(t_on=round(ev["t_on"], 1), t_off=round(ev["t_off"], 1),
                   dur_ms=round(ev["dur_ms"], 1), returned=bool(ev["returned"]),
                   peak_ext=round(float(ev["peak_ext"]), 4), direction=d)
        ev_out.append(rec)
        if d is not None:
            corrs.append(d["corr"])
    ieis = [round(ev_out[i + 1]["t_on"] - ev_out[i]["t_on"], 1) for i in range(len(ev_out) - 1)]
    n_ret = sum(e["returned"] for e in ev_out)
    n_fwd = sum(1 for c in corrs if c > 0.2)
    n_rev = sum(1 for c in corrs if c < -0.2)
    n_flat = sum(1 for c in corrs if abs(c) <= 0.2)

    summary = dict(
        tag=a.tag,
        config=dict(L=L, density=a.density, theta=a.theta, AR=a.AR, T=a.T, NE=int(NE),
                    core_mean=a.core_mean, core_std=a.core_std, core_r=a.core_r,
                    core_at=a.core_at, core_xy=[round(float(core_xy[0]), 2), round(float(core_xy[1]), 2)],
                    seed=a.seed, bin_ms=bin_w, n_core=int(core_mask.sum())),
        detector=dict(floor=round(floor, 4), peak=round(peak, 4), bar=round(bar, 4),
                      cal_frac=CAL_FRAC, baseline_ms=list(BASELINE_MS)),
        n_events=len(ev_out), n_returned=n_ret,
        direction_summary=dict(n_forward=n_fwd, n_reverse=n_rev, n_flat=n_flat,
                               corrs=[round(c, 2) for c in corrs]),
        iei_ms=ieis, events=ev_out)
    json.dump(summary, open(os.path.join(OUT, f"spont_{a.tag}.json"), "w"), indent=2)
    np.save(os.path.join(OUT, f"active_fraction_{a.tag}.npy"), af)

    print(f"[{a.tag}] events={len(ev_out)} returned={n_ret}/{len(ev_out)} "
          f"| dir fwd/rev/flat={n_fwd}/{n_rev}/{n_flat} corrs={[round(c,2) for c in corrs][:12]} "
          f"| IEI(ms)={ieis[:8]} | bar={bar:.4f} peak={peak:.4f}", flush=True)
    if len(ev_out) == 0:
        print("  -> NO events above bar: core may be sub-ignition or sustained (check active_fraction npy).")
    elif n_ret < len(ev_out):
        print(f"  -> {len(ev_out)-n_ret} event(s) did NOT self-terminate (sustained/runaway tail).")


if __name__ == "__main__":
    main()
