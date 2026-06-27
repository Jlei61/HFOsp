"""Subject-specific SNN spontaneous read-out (field-swap plan §3C).

Places low-V_th cores at the patient's swap source/sink centroids on the registered
L=20 sheet (blessed s3_brakeoff substrate: twoend_equal, core_mean 17.5, core_std 1.0;
cores at the registered centroids), with virtual electrodes at the patient contact
coords + LFP read-out, E->E axis along source->sink. Spontaneous (no kick).

Workflow (user 2026-06-26): FIRST check forward/reverse appears (this script's
fwd/rev counts), THEN cluster (separate KMeans step). One-core modes (source/sink
only) are the §3C.1 control. Saves a per-event rank matrix + an LFP/onset npz for
the core_model_s3-style A/B figure.

Reuses the blessed engine (no engine edit -> no re-bless) + cm_spontaneous read-out glue.
"""
import sys
import os
import json
import argparse
import importlib.util
import numpy as np

ENG = os.path.join("src", "snn_engine")
sys.path.insert(0, ENG)
sys.path.insert(0, os.getcwd())
from params import Params                                    # noqa: E402
from connectivity import place_neurons                       # noqa: E402
from connectivity_rot import build_connectivity_rot          # noqa: E402
from kick_probe import simulate_kick                         # noqa: E402
from lfp import LFPRecorder                                  # noqa: E402
from src.sef_hfo_snn_adapter import snn_event_envelope       # noqa: E402
from src.sef_hfo_events import detect_events                 # noqa: E402
from src.sef_hfo_heterogeneity import sample_core_field      # noqa: E402
from src.sef_hfo_subject_placement import (                  # noqa: E402
    load_swap_endpoints, load_subject_montage, register_to_sheet, template_source_foci)

_spec = importlib.util.spec_from_file_location(
    "cmrun", os.path.join("scripts", "run_sef_hfo_snn_cm_spontaneous_readout.py"))
cmrun = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmrun)
read_event, valid_mask, active_fraction = cmrun.read_event, cmrun.valid_mask, cmrun.active_fraction
per_neuron_onset = cmrun.per_neuron_onset
DT, BIN_MS, BASELINE_MS, CAL_FRAC = cmrun.DT, cmrun.BIN_MS, cmrun.BASELINE_MS, cmrun.CAL_FRAC
# PART_MIN/KDIR live as module globals in cmrun and are read inside read_event; we override
# them for sparse patient electrodes (see --k-dir). Strict clean gate uses 2*k_dir.

OUT = "results/topic4_sef_hfo/field_swap_subject_snn"


def subject_run(subject, montage_name, lesion, L, density, drive, T,
                core_mean, core_std, core_r, seed, target_inter_core, k_dir=2,
                placement="template_source", k_early=3, manual_source=None, manual_sink=None):
    # adapt the read-out estimator to sparse patient electrodes: k_dir=2 lets a 4-5 contact
    # event get a direction sign (k_dir=3 needs >=6 participating). Strict clean gate = 2*k_dir.
    cmrun.KDIR = int(k_dir)
    cmrun.PART_MIN = 2 * int(k_dir)
    PART_MIN = 2 * int(k_dir)
    # core placement: 'template_source' = earliest-k of each template = the two template SOURCES at
    # the two true ends (user-corrected 2026-06-26); 'swap' = rank-displacement source/sink centroids.
    if placement == "template_source":
        m_real, src_names, snk_names = template_source_foci(subject, montage_name, k_early)
        swap_class, decision_k = "template_source", k_early
    elif placement == "manual":
        # user-specified core channels (e.g. left=C6,C7 / right=F5,F6); positions from geometry.
        m_real = load_subject_montage(subject, montage_name, "t_a")
        src_names, snk_names = list(manual_source), list(manual_sink)
        swap_class, decision_k = "manual", len(src_names)
    else:
        sw = load_swap_endpoints(subject, montage_name)
        m_real = load_subject_montage(subject, montage_name, "t_a")
        src_names, snk_names = sw["source"], sw["sink"]
        swap_class, decision_k = sw["swap_class"], sw["decision_k"]
    reg = register_to_sheet(m_real, src_names, snk_names, L=L, target_inter_core_mm=target_inter_core)
    msheet = reg["montage_sheet"]
    src_xy, snk_xy = reg["source_centroid"], reg["sink_centroid"]
    axis_unit = (snk_xy - src_xy) / np.linalg.norm(snk_xy - src_xy)   # forward = source->sink
    theta_rad = np.deg2rad(reg["theta_deg"])

    p = Params(g=3.6, L=L, density=density, T=T, dt=DT, nu_ext_ratio=drive, seed=seed)
    rng = np.random.default_rng(seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=theta_rad, AR=2.0, verbose=False)
    is_E = np.zeros(len(net["pos"]), bool); is_E[:NE] = True

    def core(xy, s):
        return sample_core_field(net["pos"], is_E, xy, core_r, np.random.default_rng(s),
                                 core_mean=core_mean, core_std=core_std, base_mean=18.0)
    if lesion == "twoend_equal":
        cf1, cf2 = core(src_xy, seed + 7), core(snk_xy, seed + 8)
        vth = np.minimum(cf1["vth"], cf2["vth"])
    elif lesion == "source":
        vth = core(src_xy, seed + 7)["vth"]
    elif lesion == "sink":
        vth = core(snk_xy, seed + 8)["vth"]
    else:
        raise ValueError(lesion)

    posE = net["pos"][:NE]
    valid = valid_mask(msheet, posE, L, p.Rr)
    rec = LFPRecorder(p, net["pos"], net["labels"], sites=msheet.contacts)
    net["rng"] = np.random.default_rng(seed)
    res = simulate_kick(p, net, KICK_BOOST=0.0, kick_center=list(reg["center"]), r_kick=core_r,
                        t_kick=1e9, V_th_per_neuron=vth, lfp_recorder=rec)
    spk, lfp_trace, times = res["E_spk_bool"], res["lfp_trace"], res["times"]

    af, bin_w = active_fraction(spk, DT, BIN_MS)
    nb0, nb1 = int(BASELINE_MS[0] / bin_w), int(BASELINE_MS[1] / bin_w)
    floor = float(np.percentile(af[nb0:nb1], 95)) if nb1 > nb0 else float(af.min())
    peak = float(af.max()); bar = floor + CAL_FRAC * (peak - floor)
    events = detect_events(af, bin_w, event_on_frac=bar)
    env_f, fdt, _ = snn_event_envelope(spk, posE, msheet, DT)

    recs = []
    for ev in events:
        rd = read_event(env_f, fdt, msheet, valid, (ev["t_on"], ev["t_off"]), axis_unit)
        s, e = int(ev["t_on"] / bin_w), int(ev["t_off"] / bin_w)
        ep = (s + int(np.argmax(af[s:e]))) * bin_w if e > s else ev["t_on"]
        recs.append(dict(t_on=round(ev["t_on"], 1), t_off=round(ev["t_off"], 1),
                         event_peak_t=round(ep, 1), returned=bool(ev["returned"]),
                         n_part=rd["n_part"], sign=rd["sign"], axis_err=rd["axis_err"],
                         readability=rd["readability"], ranks=rd["ranks"]))
    clean = [r for r in recs if r["n_part"] >= PART_MIN and r["sign"] is not None and r["readability"] is not None]
    fwd = [r for r in clean if r["sign"] > 0]; rev = [r for r in clean if r["sign"] < 0]
    # softer "directional" screen: any event with a computable axis sign (needs n_part >= 2*k_dir = 6),
    # below the strict clean gate -- captures whether forward/reverse exists even when events are sparse.
    directional = [r for r in recs if r["sign"] is not None]
    dir_fwd = [r for r in directional if r["sign"] > 0]; dir_rev = [r for r in directional if r["sign"] < 0]
    max_n_part = max((r["n_part"] for r in recs), default=0)

    out = dict(subject=subject, montage=montage_name, lesion=lesion, seed=seed,
               swap_class=swap_class, decision_k=decision_k, placement=placement,
               anchor=reg["anchor"], target_inter_core=target_inter_core, k_dir=int(k_dir),
               inter_core_sheet=round(reg["inter_core_mm_sheet"], 2), theta_deg=round(reg["theta_deg"], 1),
               n_contacts=len(msheet.names), valid_contacts=int(valid.sum()),
               n_contacts_offsheet=reg["n_contacts_offsheet"],
               n_events=len(recs), n_clean=len(clean), max_n_part=max_n_part,
               clean_forward=len(fwd), clean_reverse=len(rev),
               n_directional=len(directional), dir_forward=len(dir_fwd), dir_reverse=len(dir_rev),
               dominant=("forward" if len(fwd) > len(rev) else "reverse" if len(rev) > len(fwd) else "tie"),
               bidirectional=(len(dir_fwd) > 0 and len(dir_rev) > 0),
               events=recs)
    # representative forward + reverse events (for the core_model_s3-style A/B panels):
    # pick the directional event with the most participating contacts (ties -> readability),
    # save its per-neuron onset field (spatial propagation gradient) + window.
    def pick(evs):
        if not evs:
            return None
        best = max(evs, key=lambda r: (r["n_part"], r["readability"] or 0))
        on = per_neuron_onset(spk, best["t_on"], best["t_off"], DT)
        return dict(meta=best, onset=on.astype(np.float32))
    rep_fwd, rep_rev = pick(dir_fwd), pick(dir_rev)

    # figure/cluster sidecar: rep forward + reverse events + LFP + onset fields
    fig = dict(reg=dict(source_centroid=src_xy.tolist(), sink_centroid=snk_xy.tolist(),
                        center=reg["center"].tolist(), theta_deg=reg["theta_deg"], L=L,
                        axis_unit=axis_unit.tolist(),
                        source_names=src_names, sink_names=snk_names),
               contacts=np.asarray(msheet.contacts), names=np.array(msheet.names, dtype=object),
               valid=valid, lfp_trace=lfp_trace, times=times, bin_w=bin_w,
               posE=posE.astype(np.float32),
               vth=vth[:NE].astype(np.float32),          # per-E-neuron threshold (mechanism heterogeneity map)
               foci=np.array([src_xy, snk_xy], float),   # two core centroids (sheet coords)
               core_r=float(core_r), core_mean=float(core_mean), theta_deg=float(reg["theta_deg"]), L=float(L),
               rep_fwd=np.array(rep_fwd, dtype=object), rep_rev=np.array(rep_rev, dtype=object))
    return out, fig, spk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="epilepsiae_958")
    ap.add_argument("--montage", default="narrow")
    ap.add_argument("--lesion", default="twoend_equal", choices=["twoend_equal", "source", "sink"])
    ap.add_argument("--L", type=float, default=20.0)
    ap.add_argument("--density", type=float, default=100.0)
    ap.add_argument("--drive", type=float, default=cmrun.DRIVE)
    ap.add_argument("--T", type=float, default=8000.0)
    ap.add_argument("--core-mean", type=float, default=17.5)
    ap.add_argument("--core-std", type=float, default=1.0)     # blessed s3_brakeoff value
    ap.add_argument("--core-r", type=float, default=1.5)
    ap.add_argument("--target-inter-core", type=float, default=None,
                    help="None=plane-fit (all electrodes, cores closer); else core-anchored mm "
                         "(blessed separation = sep_frac*L = 14; keeps dynamics, drops far contacts)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--k-dir", type=int, default=2, help="endpoint-centroid k_dir (2 for sparse patient electrodes)")
    ap.add_argument("--placement", default="template_source", choices=["template_source", "swap", "manual"],
                    help="template_source = earliest-k of each template; swap = rank-displacement source/sink; manual = user --source-core/--sink-core")
    ap.add_argument("--k-early", type=int, default=3, help="template_source: # earliest electrodes per template core")
    ap.add_argument("--source-core", default=None, help="manual: comma-separated left-core channels, e.g. C6,C7")
    ap.add_argument("--sink-core", default=None, help="manual: comma-separated right-core channels, e.g. F5,F6")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cmrun._engine_guard()
    tag = a.tag or f"{a.subject}_{a.lesion}_{a.placement}_s{a.seed}"

    msrc = a.source_core.split(",") if a.source_core else None
    msnk = a.sink_core.split(",") if a.sink_core else None
    out, fig, _ = subject_run(a.subject, a.montage, a.lesion, a.L, a.density, a.drive, a.T,
                              a.core_mean, a.core_std, a.core_r, a.seed, a.target_inter_core, a.k_dir,
                              a.placement, a.k_early, msrc, msnk)
    json.dump(out, open(os.path.join(a.out, f"readout_{tag}.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(a.out, f"figdata_{tag}.npz"), **fig)
    print(f"[{tag}] events={out['n_events']} clean(n>=7)={out['n_clean']} fwd/rev={out['clean_forward']}/{out['clean_reverse']} "
          f"| directional(n>=6)={out['n_directional']} dir_fwd/rev={out['dir_forward']}/{out['dir_reverse']} "
          f"bidir={out['bidirectional']} max_n_part={out['max_n_part']} "
          f"| valid={out['valid_contacts']}/{out['n_contacts']} (offsheet {out['n_contacts_offsheet']}) "
          f"inter={out['inter_core_sheet']}mm theta={out['theta_deg']}", flush=True)
    print(f"[written] readout_{tag}.json + figdata_{tag}.npz", flush=True)


if __name__ == "__main__":
    main()
