"""MZ early-field bridge runner (design 2026-07-19).

*** RUNS SIMULATIONS. *** Every path that simulates is gated by --confirm-run (import-safe).
Per seed: slow-off run -> frozen event bar + interictal timing templates; native zA_q75_tz5000 run
-> operational-runaway onset + early energy fields -> template/energy association + spatial nulls.
Writes resumable per-seed artifacts under results/topic4_sef_hfo/mz_early_field_bridge/per_seed/seedN/.

Reuse (design §0/§5): PP.build_substrate, run_topic4_mz_slowvars.{run_mz_cell,build_core_masks},
run_sef_hfo_snn_cm_spontaneous_readout.{active_fraction,BIN_MS,BASELINE_MS,CAL_FRAC},
src.sef_hfo_events.detect_events, run_m4_dynamic_qi.{_smooth,_first_sustained} (exact 120Hz/100ms t120),
snn_engine.lfp.LFPRecorder, src.topic4_mz_early_field_bridge (pure readout), src.early_recruitment_readout.
Edits NONE of the 6 guarded engine files -> no engine re-bless.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse                                              # noqa: E402
import dataclasses                                          # noqa: E402
import hashlib                                              # noqa: E402
import json                                                 # noqa: E402
import re                                                   # noqa: E402
import resource                                             # noqa: E402
import shutil                                               # noqa: E402
import subprocess                                           # noqa: E402
import sys                                                  # noqa: E402
import time                                                 # noqa: E402
import traceback                                            # noqa: E402

import numpy as np                                          # noqa: E402
import yaml                                                 # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

import run_m4_phaseplane as PP                              # noqa: E402
import run_sef_hfo_snn_cm_spontaneous_readout as C         # noqa: E402
import run_m4_dynamic_qi as M4                              # noqa: E402
import run_topic4_mz_slowvars as MZR                        # noqa: E402  (run_mz_cell, build_core_masks)
from lfp import LFPRecorder                                 # noqa: E402
from mz_slow_vars import MZSlowVarsConfig                   # noqa: E402
from src.sef_hfo_events import detect_events                # noqa: E402
import src.topic4_mz_early_field_bridge as B                # noqa: E402

DT = 0.1
OUT_DIR = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_early_field_bridge")
CONFIG_PATH = os.path.join(ROOT, "config", "topic4_mz_early_field_bridge.yaml")
_GUARDED_ENGINE = ("kick_probe.py", "params.py", "model.py", "connectivity.py", "connectivity_rot.py", "lfp.py")


# ============================================================ provenance
def _git_sha():
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def _engine_shas():
    d = {}
    eng = os.path.join(ROOT, "src", "snn_engine")
    for f in _GUARDED_ENGINE:
        try:
            d[f] = hashlib.sha256(open(os.path.join(eng, f), "rb").read()).hexdigest()[:12]
        except Exception:
            d[f] = None
    return d


def _provenance(seed, cfg, extra=None):
    prov = dict(git_sha=_git_sha(), engine_shas=_engine_shas(), subject=PP.SUBJECT, montage=PP.MONTAGE,
                dt=DT, seed=seed, candidate=cfg["candidate"], T_ms=cfg["T_ms"], argv=sys.argv)
    if extra:
        prov.update(extra)
    return prov


def _fingerprint(seed, cfg):
    """Stable per-seed provenance fingerprint for --resume (git sha + engine shas + candidate + T + seed)."""
    payload = dict(git_sha=_git_sha(), engine_shas=_engine_shas(), seed=seed,
                   candidate=cfg["candidate"], T_ms=cfg["T_ms"],
                   schema=cfg.get("schema_version"))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _shaft(name):
    m = re.match(r"[A-Za-z]+", str(name))
    return m.group(0) if m else str(name)


def _dump(path, obj):
    tmp = path + ".tmp"
    json.dump(B.to_jsonable(obj), open(tmp, "w"), indent=2)
    os.replace(tmp, path)                                    # atomic per-seed write


# ============================================================ per-window readout helpers
def _contact_readout(rank_a, rank_b, energy, shafts, keep, cfg):
    """All-support + direct-core-excluded contact readout for one window (design §9)."""
    ncfg = cfg["nulls"]
    n_perm, max_exact, seed = ncfg["n_permutations"], ncfg["max_exact_permutations"], ncfg["seed"]
    min_support = cfg["core_excluded"]["min_support"]

    def _one(ra, rb, e, grp):
        obs = B.maxab_observed(ra, rb, e, support_a=np.isfinite(ra), support_b=np.isfinite(rb))
        within = B.maxab_permutation_null(ra, rb, e, support_a=np.isfinite(ra), support_b=np.isfinite(rb),
                                          groups=grp, n_permutations=n_perm, seed=seed,
                                          max_exact_permutations=max_exact)
        unrestricted = B.maxab_permutation_null(ra, rb, e, support_a=np.isfinite(ra),
                                                support_b=np.isfinite(rb), groups=None,
                                                n_permutations=n_perm, seed=seed,
                                                max_exact_permutations=max_exact)
        assoc_a = B.associate(ra, e) if ra is not None else None
        assoc_b = B.associate(rb, e) if rb is not None else None
        return {"maxab": obs, "within_shaft_null": within, "unrestricted_null": unrestricted,
                "assoc_A_to_B": assoc_a, "assoc_B_to_A": assoc_b}

    all_support = _one(rank_a, rank_b, energy, shafts)
    # core-excluded: permute ONLY within the kept subset (never mix excluded energy into kept positions)
    k = np.asarray(keep, bool)
    if int(k.sum()) < min_support:
        core_excluded = {"status": "insufficient_core_excluded_support", "n_kept": int(k.sum())}
    else:
        core_excluded = _one(rank_a[k], rank_b[k], energy[k], np.asarray(shafts)[k])
        core_excluded["status"] = "eligible"
        core_excluded["n_kept"] = int(k.sum())
    return {"all_support": all_support, "core_excluded": core_excluded}


def _source_readout(rank_a_s, rank_b_s, energy_s, core_bins, cfg, n):
    """All-support + core-excluded source-grid readout with toroidal-shift maxAB null (design §9)."""
    occupied = np.isfinite(energy_s)

    def _one(support_extra):
        sa = np.isfinite(rank_a_s) & support_extra
        sb = np.isfinite(rank_b_s) & support_extra
        obs = B.maxab_observed(rank_a_s, rank_b_s, energy_s, support_a=sa, support_b=sb)
        tor = B.toroidal_maxab_null(rank_a_s, rank_b_s, energy_s, support_a_grid=sa,
                                    support_b_grid=sb, n=n)
        return {"maxab": obs, "toroidal_null": tor,
                "assoc_A_to_B": B.associate(rank_a_s, energy_s, support=sa),
                "assoc_B_to_A": B.associate(rank_b_s, energy_s, support=sb)}

    all_support = _one(occupied)
    keep = occupied & ~np.asarray(core_bins, bool)
    if int(keep.sum()) < cfg["core_excluded"]["min_support"]:
        core_excluded = {"status": "insufficient_core_excluded_support", "n_kept": int(keep.sum())}
    else:
        core_excluded = _one(keep)
        core_excluded["status"] = "eligible"
        core_excluded["n_kept"] = int(keep.sum())
    return {"all_support": all_support, "core_excluded": core_excluded}


def _wkey(w):
    return f"early_{float(w[0]):g}_{float(w[1]):g}_ms"


# ============================================================ slow-off reuse (design §8; task §8/§12)
def _substrate_geometry(S):
    """The immutable slow-off geometry that MUST match for a reused slow-off bundle to be valid."""
    msheet = S["reg"]["montage_sheet"]
    return dict(names=[str(x) for x in msheet.names],
                contacts=np.asarray(msheet.contacts, float), dt=float(DT))


def verify_slowoff_reuse(reuse_root, seed, S, *, expected_shas=None):
    """FAIL-CLOSED check that a reused slow-off bundle is bit-compatible with THIS run (design §8, task §8/§12).

    Reuse is valid ONLY when the reused slow-off was produced by the same 6 guarded engine files, the same
    dt, the same montage geometry (contact NAMES + ORDER + COORDINATES), and its held-out timing templates
    were BOTH-DIRECTION eligible. ANY mismatch or missing artifact RAISES (fail-closed). There is no silent
    fallback: the caller decides whether to re-run a fresh slow-off. Returns the verified reuse seed_dir.
    """
    sd = os.path.join(reuse_root, "per_seed", f"seed{seed}")
    need = ("slowoff.npz", "templates.npz", "slowoff.json", "templates.json", "bridge_metrics.json")
    missing = [f for f in need if not os.path.exists(os.path.join(sd, f))]
    if missing:
        raise FileNotFoundError(f"[reuse seed{seed}] FAIL-CLOSED: missing slow-off artifacts {missing} under {sd}")
    bm = json.load(open(os.path.join(sd, "bridge_metrics.json")))
    if bm.get("status") != "complete":
        raise ValueError(f"[reuse seed{seed}] FAIL-CLOSED: source status={bm.get('status')} != complete")
    prov = bm.get("provenance", {})
    exp = expected_shas if expected_shas is not None else _engine_shas()
    if prov.get("engine_shas") != exp:
        raise ValueError(f"[reuse seed{seed}] FAIL-CLOSED: engine SHA mismatch\n reused={prov.get('engine_shas')}\n current={exp}")
    if float(prov.get("dt", -1.0)) != float(DT):
        raise ValueError(f"[reuse seed{seed}] FAIL-CLOSED: dt mismatch reused={prov.get('dt')} current={DT}")
    geom = _substrate_geometry(S)
    so = np.load(os.path.join(sd, "slowoff.npz"), allow_pickle=True)
    names_reused = [str(x) for x in so["names"]]
    if names_reused != list(geom["names"]):
        raise ValueError(f"[reuse seed{seed}] FAIL-CLOSED: contact NAME/order mismatch\n reused={names_reused}\n current={geom['names']}")
    contacts_reused = np.asarray(so["contacts"], float)
    if contacts_reused.shape != geom["contacts"].shape or not np.allclose(contacts_reused, geom["contacts"], atol=1e-9):
        raise ValueError(f"[reuse seed{seed}] FAIL-CLOSED: contact COORDINATE mismatch")
    tj = json.load(open(os.path.join(sd, "templates.json")))["templates"]
    elig = {d: bool(tj[d]["contact"]["eligible"]) for d in ("A_to_B", "B_to_A")}
    if not (elig["A_to_B"] and elig["B_to_A"]):
        raise ValueError(f"[reuse seed{seed}] FAIL-CLOSED: reused templates not both-direction eligible {elig}")
    return sd


def _reconstruct_templates(sd, n_grid):
    """Rebuild tmpl_c (DirectionTemplate) + tmpl_s (dict) from a verified reused templates.{npz,json}."""
    tm = np.load(os.path.join(sd, "templates.npz"), allow_pickle=True)
    tj = json.load(open(os.path.join(sd, "templates.json")))["templates"]
    tmpl_c, tmpl_s = {}, {}
    spec = {"A_to_B": ("contact_A", "contact_A_train", "source_A"),
            "B_to_A": ("contact_B", "contact_B_train", "source_B")}
    for d, (fc, tc, fs) in spec.items():
        cj, sj = tj[d]["contact"], tj[d]["source"]
        tmpl_c[d] = B.DirectionTemplate(
            direction=d, full_template=np.asarray(tm[fc], float), train_template=np.asarray(tm[tc], float),
            heldout_scores=list(cj.get("heldout_scores") or []), n_train=int(cj["n_train"]),
            n_heldout=int(cj["n_heldout"]), n_shared_contacts=int(cj["n_shared"]),
            template_variance_ok=bool(cj.get("variance_ok", True)), eligible=bool(cj["eligible"]))
        tmpl_s[d] = {"full_template": np.asarray(tm[fs], float), "eligible": bool(sj["eligible"]),
                     "n_train": int(sj["n_train"]), "n_heldout": int(sj["n_heldout"]),
                     "n_shared": int(sj["n_shared"]), "heldout_scores": []}
    return tmpl_c, tmpl_s


def _load_verified_slowoff(reuse_root, seed, S, cfg, band, order, n_grid):
    """Verify + load a reused slow-off bundle so _run_seed_body downstream is identical to the fresh path."""
    sd = verify_slowoff_reuse(reuse_root, seed, S)          # RAISES fail-closed on any mismatch
    so = np.load(os.path.join(sd, "slowoff.npz"), allow_pickle=True)
    times = np.asarray(so["times"], float)
    lfp_so = np.asarray(so["lfp_trace"], np.float32)
    qmed = np.asarray(so["qmed"], float)
    qmad = np.asarray(so["qmad"], float)
    src_quiet_ref = np.asarray(so["src_quiet_ref"], float)
    bin_w = float(so["bin_w"])
    af_so = np.asarray(so["af"], float)
    ebar = B.EventBar(float(so["floor"]), float(so["bar"]), float(so["af_max"]),
                      tuple(cfg["event_detector"]["baseline_ms"]), float(cfg["event_detector"]["cal_frac"]))
    env_so = B.burst_envelope(lfp_so, times, band, order)
    slowoff_energy_floor = np.mean(np.maximum(env_so - qmed[None, :], 0.0) ** 2, axis=0)
    r20_so = np.repeat(np.asarray(so["r20"], float), 10)     # slowoff.npz stores r20 downsampled [::10]
    tmpl_c, tmpl_s = _reconstruct_templates(sd, n_grid)
    n_returning = int(json.load(open(os.path.join(sd, "slowoff.json")))["n_returning"])
    return (af_so, bin_w, ebar, times, qmed, qmad, src_quiet_ref, r20_so,
            slowoff_energy_floor, lfp_so, tmpl_c, tmpl_s, n_returning, sd)


# ============================================================ candidate preflight (design §16; task §6)
def preflight(cfg, seed=1, T=2000.0):
    """Fast z+m candidate preflight (no full run). Confirms (a) use_m/eta_m/tau_adp reach MZSlowVarsConfig,
    (b) the adaptation trace m is genuinely non-zero, (c) attaching the LFP recorder does NOT change the
    network dynamics (rate_E is bit-identical with vs without the recorder). The t120-within-1ms gate needs
    the full native run and is checked there (design §16 stop-rule)."""
    cand = cfg["candidate"]["cfg"]
    assert cand.get("use_m") is True, "preflight: candidate use_m must be True for V2 z+m"
    assert float(cand.get("eta_m", 0.0)) > 0.0, "preflight: eta_m must be > 0"
    assert float(cand.get("tau_adp", 0.0)) > 0.0, "preflight: tau_adp must be > 0"
    # eta_m provenance: recompute from committed calibration and require exact agreement (never hand-copied)
    calib = json.load(open(os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_slowvars", "calibration.json")))
    a_target = float(cfg["candidate"]["A_target"])
    eta_expect = a_target * float(calib["I_EE_scale"]) / float(calib["peak_m"]["tau2000"])
    assert abs(float(cand["eta_m"]) - eta_expect) < 1e-12, (
        f"preflight: eta_m {cand['eta_m']} != calibration-derived {eta_expect} "
        f"(A_target*I_EE_scale/peak_m_tau2000)")
    S = PP.build_substrate(seed)
    contacts = np.asarray(S["reg"]["montage_sheet"].contacts, float)
    # (b) m non-zero
    _, mz = MZR.run_mz_cell(S, MZSlowVarsConfig(**cand), T, early_stop=False)
    adap = np.asarray(mz.trace_adap_current, float)
    m_nonzero = bool(adap.size and np.nanmax(np.abs(adap)) > 0.0)
    # (c) LFP recorder is a non-perturbing observer: rate_E identical with vs without it
    S2 = PP.build_substrate(seed)
    res_nolfp, _ = MZR.run_mz_cell(S2, MZSlowVarsConfig(**cand), T, early_stop=False)
    S3 = PP.build_substrate(seed)
    rec = LFPRecorder(S3["p"], S3["net"]["pos"], S3["net"]["labels"], sites=contacts)
    res_lfp, _ = MZR.run_mz_cell(S3, MZSlowVarsConfig(**cand), T, early_stop=False, lfp_recorder=rec)
    rate_identical = bool(np.array_equal(np.asarray(res_nolfp["rate_E"]), np.asarray(res_lfp["rate_E"])))
    out = {"seed": seed, "T_ms": T, "use_m": True, "eta_m": float(cand["eta_m"]),
           "eta_m_calibration_derived": eta_expect, "tau_adp": float(cand["tau_adp"]),
           "adaptation_trace_nonzero": m_nonzero, "adaptation_trace_absmax": float(np.nanmax(np.abs(adap)) if adap.size else 0.0),
           "lfp_recorder_is_noop_on_rate": rate_identical}
    print(f"[preflight] seed={seed} T={T}ms use_m=True eta_m={cand['eta_m']:.12g} tau_adp={cand['tau_adp']} "
          f"m_nonzero={m_nonzero} lfp_noop={rate_identical}", flush=True)
    if not m_nonzero:
        raise SystemExit("preflight FAILED: adaptation trace m is all-zero — use_m/eta_m not entering the sim")
    if not rate_identical:
        raise SystemExit("preflight FAILED: LFP recorder changed rate_E — recorder is not a pure observer")
    return out


# ============================================================ per-seed pipeline
def run_seed(seed, cfg, out_dir, resume=False, T_override=None, reuse_root=None):
    seed_dir = os.path.join(out_dir, "per_seed", f"seed{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    metrics_path = os.path.join(seed_dir, "bridge_metrics.json")
    fp = _fingerprint(seed, cfg)
    if resume and B.resume_should_skip(metrics_path, fp):
        print(f"[seed {seed}] resume: complete + provenance match -> skip", flush=True)
        return json.load(open(metrics_path))
    t0 = time.time()
    try:
        result = _run_seed_body(seed, cfg, seed_dir, fp, T_override, reuse_root)
        result["wall_s"] = round(time.time() - t0, 1)
        result["status"] = "complete"
        _dump(metrics_path, result)
        print(f"[seed {seed}] complete in {result['wall_s']}s -> {metrics_path}", flush=True)
        return result
    except Exception as ex:
        err = {"seed": seed, "status": "failed", "error": repr(ex),
               "traceback": traceback.format_exc(), "provenance_fingerprint": fp,
               "provenance": _provenance(seed, cfg), "wall_s": round(time.time() - t0, 1)}
        _dump(metrics_path, err)
        print(f"[seed {seed}] FAILED after {err['wall_s']}s: {ex}\n{err['traceback']}", flush=True)
        return err


def _run_seed_body(seed, cfg, seed_dir, fp, T_override, reuse_root=None):
    T = float(T_override) if T_override else float(cfg["T_ms"])       # native trajectory length
    slowoff_T = float(cfg.get("slowoff_T_ms", T))                    # slow-off length (V2: 15000 = match V1 templates)
    band = tuple(cfg["timing"]["band_hz"]); order = int(cfg["timing"]["filter_order"])
    n_grid = int(cfg["source_grid"]["n"]); tbin = float(cfg["source_grid"].get("time_bin_ms", 10.0))
    tk = cfg["timing"]
    timing_kw = dict(event_offset_ms=tk["event_offset_ms"], mad_k=tk["readable_mad_k"],
                     rel_peak=tk["readable_rel_peak"], min_readable=tk["min_readable_contacts"],
                     direction_abs=tk["direction_abs_spearman"])
    sp = cfg["split"]; split_kw = dict(min_train=sp["min_train_events"], min_heldout=sp["min_heldout_events"],
                                       min_shared=sp["min_shared_contacts"])

    S = PP.build_substrate(seed)
    msheet = S["reg"]["montage_sheet"]
    contacts = np.asarray(msheet.contacts, float)
    names = [str(x) for x in msheet.names]
    shafts = np.array([_shaft(nm) for nm in names], object)
    axis_unit = np.asarray(S["axis_unit"], float); center = np.asarray(S["center"], float)
    contact_axis = (contacts - center) @ axis_unit          # + = source->sink (axis_unit); metadata records mapping
    posE = np.asarray(S["posE"], float); L = float(S["L"])
    core_mask_E = MZR.build_core_masks(S)
    cell, counts = B.source_bins(posE, L, n=n_grid)
    core_bins = np.zeros(n_grid * n_grid, bool)
    core_bins[np.unique(cell[core_mask_E])] = True

    # -------------------------------------------------- SLOW-OFF (frozen bar + templates)
    if reuse_root:
        # design §8 / task §8: reuse V1's frozen slow-off (z/m-off => identical templates), fail-closed verified.
        (af_so, bin_w, ebar, times, qmed, qmad, src_quiet_ref, r20_so, slowoff_energy_floor,
         lfp_so, tmpl_c, tmpl_s, _n_returning, reuse_src_dir) = _load_verified_slowoff(
            reuse_root, seed, S, cfg, band, order, n_grid)
        returning = [None] * _n_returning                    # only len() is used downstream (n_returning_events)
        print(f"[seed {seed}] REUSED verified slow-off <- {reuse_src_dir} (design §8)", flush=True)
    else:
        rec = LFPRecorder(S["p"], S["net"]["pos"], S["net"]["labels"], sites=contacts)
        res_so, _ = MZR.run_mz_cell(S, MZSlowVarsConfig(use_z=False, use_m=False), slowoff_T,
                                    early_stop=False, lfp_recorder=rec)
        af_so, bin_w = C.active_fraction(res_so["E_spk_bool"], DT, C.BIN_MS)
        ebar = B.compute_event_bar(af_so, bin_w, tuple(cfg["event_detector"]["baseline_ms"]),
                                   cfg["event_detector"]["cal_frac"])
        events_so = detect_events(af_so, bin_w, event_on_frac=ebar.bar)
        returning = [e for e in events_so if e["returned"]]
        times = np.asarray(res_so["times"], float)
        env_so = B.burst_envelope(res_so["lfp_trace"], times, band, order)
        qmask = B.quiet_mask(times, events_so)
        qmed, qmad = B.quiet_baseline(env_so, qmask)
        src_quiet_ref = B.source_quiet_ref(res_so["E_spk_bool"], cell, counts, qmask, DT, n=n_grid, time_bin_ms=tbin)
        r20_so = M4._smooth(np.asarray(res_so["rate_E"], float), DT)
        event_onsets = sorted(e["t_on"] for e in events_so)

        timings, src_ranks = [], []
        for e in returning:
            nxt = next((t for t in event_onsets if t > e["t_off"]), None)
            ct = B.event_contact_timing(env_so, times, e, next_event_t_on=nxt, record_end_ms=times[-1],
                                        quiet_med=qmed, quiet_mad=qmad, contact_axis=contact_axis, **timing_kw)
            timings.append(ct)
            src_ranks.append(B._ordinal_rank(B.source_timing_field(
                res_so["E_spk_bool"], cell, e, DT, n=n_grid, min_active=cfg["source_grid"]["min_active_e_per_bin"])))
        # §8.5 slow-off energy field (per-contact mean-sq excess over quiet) -> P95 = recruited-count reference
        slowoff_energy_floor = np.mean(np.maximum(env_so - qmed[None, :], 0.0) ** 2, axis=0)
        lfp_so = np.asarray(res_so["lfp_trace"], np.float32)     # keep small LFP for the seed1 figure
        del res_so, env_so                                       # free ~5-10 GB before the native run

        # direction assignment (contact axis-latency) -> per-direction chronological event lists
        dir_ct = {"A_to_B": [], "B_to_A": []}
        dir_sr = {"A_to_B": [], "B_to_A": []}
        for ct, sr in zip(timings, src_ranks):
            if ct.eligible and ct.direction in dir_ct:
                dir_ct[ct.direction].append(ct)
                dir_sr[ct.direction].append(sr)
        tmpl_c = {d: B.build_direction_template(dir_ct[d], d, **split_kw) for d in dir_ct}
        tmpl_s = {d: B.build_template_from_ranks(dir_sr[d], **split_kw) for d in dir_sr}
        reuse_src_dir = None

    # -------------------------------------------------- NATIVE (operational runaway)
    rec2 = LFPRecorder(S["p"], S["net"]["pos"], S["net"]["labels"], sites=contacts)
    res_na, _ = MZR.run_mz_cell(S, MZSlowVarsConfig(**cfg["candidate"]["cfg"]), T,
                                early_stop=False, lfp_recorder=rec2)
    times_na = np.asarray(res_na["times"], float)
    rate_na = np.asarray(res_na["rate_E"], float)
    r20_na = M4._smooth(rate_na, DT)
    t120 = M4._first_sustained(r20_na, DT)
    # design §16 / task §6: native t120 must reproduce the committed operational-runaway onset within tol
    preflight_gate = None
    _ref = (cfg.get("preflight") or {}).get("reference_onset_ms") or {}
    _ref_t = _ref.get(seed, _ref.get(str(seed)))
    if _ref_t is not None:
        _tol = float((cfg.get("preflight") or {}).get("t120_tol_ms", 1.0))
        _delta = abs(float(t120) - float(_ref_t)) if t120 is not None else None
        preflight_gate = {"reference_onset_ms": float(_ref_t), "t120_ms": t120, "delta_ms": _delta,
                          "tol_ms": _tol, "pass": bool(_delta is not None and _delta <= _tol)}
        print(f"[seed {seed}] preflight t120 gate: t120={t120} ref={_ref_t} delta={_delta} "
              f"tol={_tol} pass={preflight_gate['pass']}", flush=True)
    onset = B.compute_t_recruit(r20_na, r20_so, DT, t120,
                                theta_pct=cfg["onset"]["theta_recruit_pct"], gap_ms=cfg["onset"]["recruit_gap_ms"])
    env_na = B.burst_envelope(res_na["lfp_trace"], times_na, band, order)
    lfp_na = np.asarray(res_na["lfp_trace"], np.float32)
    contact_loading = B.contact_core_loading(contacts, posE, core_mask_E,
                                             kernel_width_mm=cfg["core_excluded"]["contact_kernel_width_mm"])
    contact_keep = contact_loading < cfg["core_excluded"]["contact_core_loading_threshold"]

    windows = [tuple(cfg["windows"]["primary_ms"])] + [tuple(w) for w in cfg["windows"]["sensitivity_ms"]]
    ra_c = tmpl_c["A_to_B"].full_template if tmpl_c["A_to_B"].eligible else np.full(len(names), np.nan)
    rb_c = tmpl_c["B_to_A"].full_template if tmpl_c["B_to_A"].eligible else np.full(len(names), np.nan)
    ra_s = tmpl_s["A_to_B"]["full_template"] if tmpl_s["A_to_B"]["eligible"] else np.full(n_grid ** 2, np.nan)
    rb_s = tmpl_s["B_to_A"]["full_template"] if tmpl_s["B_to_A"]["eligible"] else np.full(n_grid ** 2, np.nan)

    by_window = {}
    slowoff_p95_c = (float(np.nanpercentile(slowoff_energy_floor, 95))
                     if np.isfinite(slowoff_energy_floor).any() else None)
    energy_arrays = {}
    if onset["status"] == "eligible":
        t_recruit = onset["t_recruit_ms"]
        for w in windows:
            cef = B.contact_energy_field(env_na, times_na, qmed, t_recruit, w, record_end_ms=times_na[-1])
            sef = B.source_energy_field(res_na["E_spk_bool"], cell, counts, t_recruit, w, DT, n=n_grid,
                                        quiet_ref=src_quiet_ref, time_bin_ms=tbin, record_end_ms=times_na[-1])
            wk = _wkey(w)
            entry = {"window_ms": list(map(float, w)),
                     "contact_energy_status": cef.status, "source_energy_status": sef["status"],
                     "contact_field_diag": B.field_diagnostics(cef.energy, slowoff_p95_c),
                     "source_field_diag": B.field_diagnostics(sef["energy"])}
            if cef.status == "eligible":
                entry["contact"] = _contact_readout(ra_c, rb_c, cef.energy, shafts, contact_keep, cfg)
                s_step = int(round((t_recruit + w[0]) / DT)); e_step = int(round((t_recruit + w[1]) / DT))
                part = B.local_participation(contacts, posE, res_na["E_spk_bool"], (s_step, e_step),
                                             radius_mm=cfg["participation_audit"]["radius_mm"])
                entry["local_participation"] = {"per_contact": part,
                                                "median": float(np.nanmedian(part)) if np.isfinite(part).any() else None}
            if sef["status"] == "eligible":
                entry["source"] = _source_readout(ra_s, rb_s, sef["energy"], core_bins, cfg, n_grid)
            by_window[wk] = entry
            energy_arrays[f"contact_energy__{wk}"] = np.asarray(cef.energy, np.float32)
            energy_arrays[f"source_energy__{wk}"] = np.asarray(sef["energy"], np.float32)

    # -------------------------------------------------- pre-runaway within-trajectory audit (§7.1 secondary)
    af_na, _ = C.active_fraction(res_na["E_spk_bool"], DT, C.BIN_MS)
    events_na_frozen = detect_events(af_na, bin_w, event_on_frac=ebar.bar)     # native events under the FROZEN bar
    pre_runaway = [e for e in events_na_frozen if e["returned"] and (t120 is None or e["t_off"] < t120 - 20.0)]
    within_traj = {"n_pre_runaway_returning": len(pre_runaway),
                   "status": "eligible" if len(pre_runaway) >= 3 else "insufficient_support"}
    del res_na

    # -------------------------------------------------- persist artifacts
    np.savez_compressed(os.path.join(seed_dir, "native.npz"),
                        lfp_trace=lfp_na, times=np.asarray(times_na, np.float32),
                        rate=np.asarray(rate_na, np.float32), r20=np.asarray(r20_na[::10], np.float32),
                        contacts=np.asarray(contacts, np.float32), names=np.array(names, object),
                        contact_loading=np.asarray(contact_loading, np.float32), contact_keep=contact_keep,
                        **energy_arrays)
    if reuse_src_dir is not None:
        # reused slow-off is verified identical -> copy the frozen slow-off/template artifacts (provenance-preserved)
        for f in ("slowoff.npz", "templates.npz", "slowoff.json", "templates.json"):
            shutil.copyfile(os.path.join(reuse_src_dir, f), os.path.join(seed_dir, f))
    else:
        np.savez_compressed(os.path.join(seed_dir, "slowoff.npz"),
                            af=np.asarray(af_so, np.float32), bin_w=bin_w, floor=ebar.floor, bar=ebar.bar,
                            af_max=ebar.af_max, lfp_trace=lfp_so, times=np.asarray(times, np.float32),
                            contacts=np.asarray(contacts, np.float32), names=np.array(names, object),
                            shafts=np.array(shafts, object), contact_axis=np.asarray(contact_axis, np.float32),
                            qmed=np.asarray(qmed, np.float32), qmad=np.asarray(qmad, np.float32),
                            r20=np.asarray(r20_so[::10], np.float32), src_quiet_ref=np.asarray(src_quiet_ref, np.float32),
                            core_bins=core_bins, event_t_on=np.array([e["t_on"] for e in returning], float),
                            event_t_off=np.array([e["t_off"] for e in returning], float),
                            event_dir=np.array([ct.direction for ct in timings], object),
                            event_rank_stack=np.array([ct.rank for ct in timings], float),
                            src_rank_stack=np.array(src_ranks, float))
        np.savez_compressed(os.path.join(seed_dir, "templates.npz"),
                            contact_A=np.asarray(ra_c, np.float32), contact_B=np.asarray(rb_c, np.float32),
                            contact_A_train=np.asarray(tmpl_c["A_to_B"].train_template, np.float32),
                            contact_B_train=np.asarray(tmpl_c["B_to_A"].train_template, np.float32),
                            source_A=np.asarray(ra_s, np.float32), source_B=np.asarray(rb_s, np.float32))

        templates_summary = {}
        for d in ("A_to_B", "B_to_A"):
            tc, ts = tmpl_c[d], tmpl_s[d]
            templates_summary[d] = {
                "contact": {"n_events": len(dir_ct[d]), "n_train": tc.n_train, "n_heldout": tc.n_heldout,
                            "n_shared": tc.n_shared_contacts, "eligible": tc.eligible,
                            "heldout_scores": tc.heldout_scores,
                            "heldout_median": float(np.nanmedian(tc.heldout_scores)) if tc.heldout_scores else None,
                            "variance_ok": tc.template_variance_ok},
                "source": {"n_events": len(dir_sr[d]), "n_train": ts["n_train"], "n_heldout": ts["n_heldout"],
                           "n_shared": ts["n_shared"], "eligible": ts["eligible"],
                           "heldout_median": float(np.nanmedian(ts["heldout_scores"])) if ts["heldout_scores"] else None},
            }
        _dump(os.path.join(seed_dir, "templates.json"),
              {"seed": seed, "direction_axis_mapping":
                  {"A_to_B": "low-axis(source-side) -> high-axis(sink-side); axis_unit = src_xy->snk_xy",
                   "src_xy": np.asarray(S["src_xy"]).tolist(), "snk_xy": np.asarray(S["snk_xy"]).tolist(),
                   "axis_unit": axis_unit.tolist(), "center": center.tolist()},
               "n_returning_events": len(returning), "n_eligible_events": len(dir_ct["A_to_B"]) + len(dir_ct["B_to_A"]),
               "templates": templates_summary})
        _dump(os.path.join(seed_dir, "slowoff.json"),
              {"seed": seed, "event_bar": dataclasses.asdict(ebar), "n_events": len(events_so),
               "n_returning": len(returning),
               "n_eligible_timing_events": int(sum(ct.eligible for ct in timings)),
               "direction_counts": {"A_to_B": len(dir_ct["A_to_B"]), "B_to_A": len(dir_ct["B_to_A"]),
                                    "unresolved": int(sum(ct.direction == "unresolved" for ct in timings))}})
    _dump(os.path.join(seed_dir, "native.json"),
          {"seed": seed, "t120_ms": t120, "onset": onset,
           "record_end_ms": float(times_na[-1]), "within_trajectory_audit": within_traj})

    return {"seed": seed, "provenance_fingerprint": fp, "provenance": _provenance(seed, cfg),
            "onset": onset, "t120_ms": t120, "preflight_gate": preflight_gate,
            "reused_slowoff_from": (os.path.abspath(reuse_src_dir) if reuse_src_dir else None),
            "template_eligibility": {d: {"contact": tmpl_c[d].eligible, "source": tmpl_s[d]["eligible"]}
                                     for d in ("A_to_B", "B_to_A")},
            "maxab_eligible": bool(tmpl_c["A_to_B"].eligible and tmpl_c["B_to_A"].eligible),
            "n_returning_events": len(returning), "within_trajectory_audit": within_traj,
            "by_window": by_window}


# ============================================================ readout-only patch (contact float-window fix)
def readout_only(seed, cfg, out_dir):
    """Recompute the CONTACT readout (energy / association / nulls / core-excluded / field-diag) from the
    SAVED LFP artifacts and patch bridge_metrics.json + native.npz in place — no 2.4 h sim re-run. Fixes
    the float-window contact-energy bug (source results, integer-step and already correct, are preserved).
    Local participation needs the native raster (not persisted) -> marked not_recomputed_readout_patch."""
    sd = os.path.join(out_dir, "per_seed", f"seed{seed}")
    need = ("bridge_metrics.json", "slowoff.npz", "native.npz", "templates.npz", "native.json")
    if not all(os.path.exists(os.path.join(sd, f)) for f in need):
        print(f"[readout-only seed{seed}] artifacts missing; skip", flush=True)
        return None
    bm = json.load(open(os.path.join(sd, "bridge_metrics.json")))
    if bm.get("status") != "complete":
        print(f"[readout-only seed{seed}] status={bm.get('status')} not complete; skip", flush=True)
        return None
    so = np.load(os.path.join(sd, "slowoff.npz"), allow_pickle=True)
    na = np.load(os.path.join(sd, "native.npz"), allow_pickle=True)
    tm = np.load(os.path.join(sd, "templates.npz"), allow_pickle=True)
    nj = json.load(open(os.path.join(sd, "native.json")))
    band = tuple(cfg["timing"]["band_hz"]); order = int(cfg["timing"]["filter_order"])
    names = [str(x) for x in so["names"]]
    shafts = np.array([_shaft(nm) for nm in names], object)
    qmed = np.asarray(so["qmed"], float)
    env_so = B.burst_envelope(so["lfp_trace"], np.asarray(so["times"], float), band, order)
    floor = np.mean(np.maximum(env_so - qmed[None, :], 0.0) ** 2, axis=0)
    p95 = float(np.nanpercentile(floor, 95)) if np.isfinite(floor).any() else None
    env_na = B.burst_envelope(na["lfp_trace"], np.asarray(na["times"], float), band, order)
    times_na = np.asarray(na["times"], float)
    ra_c = np.asarray(tm["contact_A"], float); rb_c = np.asarray(tm["contact_B"], float)
    keep = np.asarray(na["contact_keep"], bool)
    onset = nj["onset"]
    windows = [tuple(cfg["windows"]["primary_ms"])] + [tuple(w) for w in cfg["windows"]["sensitivity_ms"]]
    energy_updates = {}
    if onset["status"] == "eligible":
        tr = onset["t_recruit_ms"]
        for w in windows:
            wk = _wkey(w)
            cef = B.contact_energy_field(env_na, times_na, qmed, tr, w, record_end_ms=times_na[-1])
            entry = (bm.setdefault("by_window", {}).get(wk) or {"window_ms": list(map(float, w))})
            entry["contact_energy_status"] = cef.status
            entry["contact_field_diag"] = B.field_diagnostics(cef.energy, p95)
            if cef.status == "eligible":
                entry["contact"] = _contact_readout(ra_c, rb_c, cef.energy, shafts, keep, cfg)
                entry["local_participation"] = {"status": "not_recomputed_readout_patch",
                                                "reason": "native raster not persisted; re-run to recompute"}
            bm["by_window"][wk] = entry
            energy_updates[f"contact_energy__{wk}"] = np.asarray(cef.energy, np.float32)
    bm["readout_patched"] = ("contact energy float-window fix (§8.2/§8.3); source preserved; "
                             "local participation not recomputed (raster not persisted)")
    _dump(os.path.join(sd, "bridge_metrics.json"), bm)
    keys = {k: na[k] for k in na.files}                      # preserve every native.npz key ...
    keys.update(energy_updates)                              # ... overwrite corrected contact energy fields
    np.savez_compressed(os.path.join(sd, "native.npz"), **keys)
    prim = _wkey(tuple(cfg["windows"]["primary_ms"]))
    mx = (((bm.get("by_window", {}).get(prim, {}) or {}).get("contact", {}) or {})
          .get("all_support", {}).get("maxab") or {})
    print(f"[readout-only seed{seed}] contact patched; primary rho_maxab={mx.get('rho_maxab')} "
          f"maxab_eligible={mx.get('maxab_eligible')}", flush=True)
    return bm


# ============================================================ cohort aggregation
def aggregate(out_dir, seeds, cfg):
    rows, per_seed = [], {}
    for seed in seeds:
        p = os.path.join(out_dir, "per_seed", f"seed{seed}", "bridge_metrics.json")
        if not os.path.exists(p):
            per_seed[str(seed)] = {"status": "missing"}
            continue
        d = json.load(open(p))
        per_seed[str(seed)] = d
        if d.get("status") != "complete":
            continue
        wk = _wkey(tuple(cfg["windows"]["primary_ms"]))
        w = d.get("by_window", {}).get(wk, {})
        c = (w.get("contact") or {}).get("all_support", {})
        mx = (c.get("maxab") or {})
        null = (c.get("within_shaft_null") or {})
        rows.append({"seed": seed, "onset_status": d["onset"]["status"],
                     "t120_ms": d.get("t120_ms"), "t_recruit_ms": d["onset"].get("t_recruit_ms"),
                     "maxab_eligible": d.get("maxab_eligible"),
                     "rho_maxab": mx.get("rho_maxab"), "rho_a": mx.get("rho_a"), "rho_b": mx.get("rho_b"),
                     "within_shaft_p": null.get("p_one_sided")})
    finite = lambda xs: [x for x in xs if isinstance(x, (int, float)) and np.isfinite(x)]
    maxabs = finite([r["rho_maxab"] for r in rows])
    cohort = {"experiment": "MZ early-field bridge (seeds 1/3/4; design 2026-07-19)",
              "candidate": cfg["candidate"], "n_seeds_complete": len(rows),
              "primary_window": _wkey(tuple(cfg["windows"]["primary_ms"])),
              "rho_maxab_median": float(np.median(maxabs)) if maxabs else None,
              "rho_maxab_range": [float(np.min(maxabs)), float(np.max(maxabs))] if maxabs else None,
              "n_positive_maxab": int(sum(1 for x in maxabs if x > 0)),
              "note": "n=3: consistency (median/range/sign count) only; NO cohort significance claim (design §9).",
              "per_seed_rows": rows}
    _dump(os.path.join(out_dir, "cohort_summary.json"), cohort)
    # CSV
    keys = ["seed", "onset_status", "t120_ms", "t_recruit_ms", "maxab_eligible", "rho_maxab",
            "rho_a", "rho_b", "within_shaft_p"]
    import csv
    with open(os.path.join(out_dir, "cohort_summary.csv"), "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        wtr.writeheader()
        for r in rows:
            wtr.writerow(r)
    _dump(os.path.join(out_dir, "provenance.json"),
          {"git_sha": _git_sha(), "engine_shas": _engine_shas(), "seeds": seeds,
           "candidate": cfg["candidate"], "T_ms": cfg["T_ms"], "argv": sys.argv})
    print(f"[aggregate] {len(rows)} complete seeds; rho_maxab median="
          f"{cohort['rho_maxab_median']} range={cohort['rho_maxab_range']} -> {out_dir}/cohort_summary.json",
          flush=True)
    return cohort


# ============================================================ smoke
def smoke(cfg, seed=1, T=2000.0):
    """Short end-to-end check: LFP shape, contact order, RAM, wall time (design 8h-prompt P2)."""
    t0 = time.time()
    S = PP.build_substrate(seed)
    msheet = S["reg"]["montage_sheet"]
    contacts = np.asarray(msheet.contacts, float)
    rec = LFPRecorder(S["p"], S["net"]["pos"], S["net"]["labels"], sites=contacts)
    res, _ = MZR.run_mz_cell(S, MZSlowVarsConfig(use_z=False, use_m=False), T, early_stop=False, lfp_recorder=rec)
    af, bin_w = C.active_fraction(res["E_spk_bool"], DT, C.BIN_MS)
    ebar = B.compute_event_bar(af, bin_w, tuple(cfg["event_detector"]["baseline_ms"]), cfg["event_detector"]["cal_frac"])
    events = detect_events(af, bin_w, event_on_frac=ebar.bar)
    env = B.burst_envelope(res["lfp_trace"], np.asarray(res["times"], float), tuple(cfg["timing"]["band_hz"]))
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
    wall = time.time() - t0
    print(f"[smoke] seed={seed} T={T}ms N={S['N']} NE={S['NE']}", flush=True)
    print(f"[smoke] lfp_trace shape={np.asarray(res['lfp_trace']).shape} (expect (nsteps,15)) "
          f"E_spk_bool shape={np.asarray(res['E_spk_bool']).shape}", flush=True)
    print(f"[smoke] contacts={[str(x) for x in msheet.names]}", flush=True)
    print(f"[smoke] envelope shape={env.shape} events={len(events)} returning="
          f"{sum(e['returned'] for e in events)} bar={ebar.bar:.4f}", flush=True)
    print(f"[smoke] peak_RSS={rss_gb:.2f} GB wall={wall:.1f}s -> extrapolated T=15000 wall "
          f"~{wall * 15000.0 / T:.0f}s per run (x2 runs/seed)", flush=True)
    return dict(rss_gb=round(rss_gb, 2), wall_s=round(wall, 1), n_events=len(events))


# ============================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="MZ early-field bridge runner (design 2026-07-19).")
    ap.add_argument("--confirm-run", action="store_true", help="required to run simulations")
    ap.add_argument("--config", default=CONFIG_PATH)
    ap.add_argument("--seeds", default=None, help="comma list; default from config")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="short one-seed pipeline/RAM/wall check")
    ap.add_argument("--T", default=None, help="override T_ms (smoke or debug)")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--readout-only", action="store_true",
                    help="recompute CONTACT readout from saved LFP artifacts (float-window fix); no sim")
    ap.add_argument("--output-dir", default=None,
                    help="output root (default results/topic4_sef_hfo/mz_early_field_bridge); V2 isolates its own dir")
    ap.add_argument("--reuse-slowoff-root", default=None,
                    help="reuse a verified frozen slow-off bundle from this bridge output root (design §8; fail-closed)")
    ap.add_argument("--preflight", action="store_true",
                    help="fast z+m candidate preflight (use_m/eta_m/tau_adp + m non-zero + LFP no-op); no full run")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    seeds = [int(s) for s in (args.seeds.split(",") if args.seeds else cfg["seeds"])]
    out_dir = os.path.abspath(args.output_dir) if args.output_dir else OUT_DIR
    reuse_root = os.path.abspath(args.reuse_slowoff_root) if args.reuse_slowoff_root else None
    os.makedirs(out_dir, exist_ok=True)
    # snapshot the frozen config next to the outputs
    with open(os.path.join(out_dir, "config_snapshot.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    if args.aggregate_only:
        aggregate(out_dir, seeds, cfg)
        return
    if args.readout_only:
        for seed in seeds:
            readout_only(seed, cfg, out_dir)
        aggregate(out_dir, seeds, cfg)
        return
    if not args.confirm_run:
        print("REFUSING: this runs simulations. Pass --confirm-run (import-safe gate).", file=sys.stderr)
        sys.exit(2)
    if args.preflight:
        _dump(os.path.join(out_dir, "preflight.json"),
              preflight(cfg, seed=seeds[0], T=float(args.T) if args.T else 2000.0))
        return
    if args.smoke:
        smoke(cfg, seed=seeds[0], T=float(args.T) if args.T else 2000.0)
        return
    for seed in seeds:
        run_seed(seed, cfg, out_dir, resume=args.resume, T_override=float(args.T) if args.T else None,
                 reuse_root=reuse_root)
    aggregate(out_dir, seeds, cfg)


if __name__ == "__main__":
    main()
