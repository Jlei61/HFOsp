"""M4 Pass-1 scientific runner — q_core x alpha_G phase-plane + §9.1 go/no-go (spec 2026-07-05 rev4 §8.1,
§9.1, §11 step 7-8).  *** THIS RUNS SIMULATIONS — it is the Pass-1 SIM GATE. ***

Assembled for review; DO NOT run until the plan + Pass-1 implementation are approved. Nothing runs on
import — all simulation is inside main() / worker functions invoked only from `__main__`.

Flow:
  1. build_substrate           -> p, net, E positions, V_th, center
  2. derive_core               -> core neuron mask from an arm-0 kick's first-activation map (rev4 §8.1)
  3. calibrate_r50             -> Psi_G half-recruitment from the rE_fast peak scale (the pre-req I flagged)
  4. reference_metrics         -> arm-0 TRIVIAL-A (flood) + TRIVIAL-B (axial-retreat) reference CellMetrics
     calibrate_guards_from_references (pure) -> §9.1 GuardThresholds excluding those references
  5. sweep q_core x alpha_G for arm 0 (baseline) / arm 1 (beta_SG) / arm 2 (alpha_G divisive)
     extract_cell_metrics -> classify_cell -> per-arm go grids
  6. go_plane_verdict(arm2, arm1) -> go / no-go; write results/topic4_m4/phase_plane.{json,png} + README

REVIEW POINTS (science contract — see also sef_hfo_m4_metrics DEFINITIONAL CHOICES):
  - metric-extraction params below (T_MIN_MS / BAND_HALF_MM / THRESH_HZ / RETREAT_FACTOR / SAT_CEILING_FRAC);
  - the reference q_core picks for calibration (Q_FLOOD / Q_AXIAL) and the guard `MARGIN`;
  - the r50 percentile pick (R50_PCTL); K_MIN for the contiguous-area verdict.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")             # memory caveat: parallel numpy nulls MUST OMP=1
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse                                            # noqa: E402
import json                                                # noqa: E402
import multiprocessing as mp                               # noqa: E402
import sys                                                 # noqa: E402
import time                                                # noqa: E402

import numpy as np                                         # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))
DATA_ROOT = os.path.dirname(os.path.dirname(ROOT))         # main checkout: E1146 results/ data (worktree gitignored)

from params import Params                                  # noqa: E402
from connectivity import place_neurons                     # noqa: E402
from connectivity_rot import build_connectivity_rot        # noqa: E402
from kick_probe import simulate_kick                       # noqa: E402
from slow_field import SpatialSlowField, SpatialSlowFieldConfig  # noqa: E402
from src.sef_hfo_heterogeneity import sample_core_field    # noqa: E402
from src.sef_hfo_subject_placement import template_source_foci, register_to_sheet  # noqa: E402
from src.sef_hfo_m4_metrics import extract_cell_metrics    # noqa: E402
from src.sef_hfo_m4_phaseplane import (                    # noqa: E402
    calibrate_guards_from_references, classify_cell, go_plane_verdict,
    CalibrationError, validate_reference_metrics,
)

OUT_DIR = os.path.join(ROOT, "results", "topic4_m4")

# ---- E1146 subject substrate (SNN Stage-5 blessed; run_sef_hfo_subject_snn.py + sef_hfo_subject_placement) ----
# Register the patient contact plane into an L=20 sheet; TWO low-V_th cores at the source/sink centroids
# (twoend_equal, core_mean 17.5 / surround 18.0), E->E axis source->sink. M4 change vs Stage-5: a FOCAL IED
# kick on the SOURCE core (spec §8.1 "IED-like pulse"), NOT the spontaneous probe. q_core is frozen on the
# source core; f_off is measured vs the registered source->sink axis. Data read from the main checkout.
SUBJECT = "epilepsiae_1146"
MONTAGE = "narrow"
PLACEMENT_K_EARLY = 3                                      # template_source: earliest-k electrodes per template core
L = 20.0
DENSITY, G, DRIVE = 100.0, 3.6, 0.6                        # Stage-5 blessed
CORE_MEAN, CORE_STD, CORE_R, BASE_MEAN = 17.5, 1.0, 1.5, 18.0   # Stage-5 blessed twoend_equal cores
TARGET_INTER_CORE = None                                   # plane-fit registration (Stage-5 default; keeps full layout)
T_SIM = 500.0                                              # kick probe (Stage-5 spontaneous used 8000)
T_KICK = 120.0                                             # focal IED kick onset
KICK = 3.0                                                 # focal IED kick boost (REVIEW: Stage-5 was spontaneous)
R_KICK = 0.3                                               # focal IED probe radius on the source core
DT = 0.1

# ---- metric-extraction params (REVIEW POINTS) ----
T_MIN_MS = 60.0            # persist: burst sustained >= this after the kick
BAND_HALF_MM = 1.0         # f_off: perpendicular band half-width around the onset axis
THRESH_HZ = 10.0           # branching: supra-threshold bins (matches pre_kick_ignition rest ref)
RETREAT_FACTOR = 0.5       # spatial self-limit: late extent < this * peak extent
SAT_CEILING_FRAC = 0.5     # saturation ceiling (per-neuron Hz) = this * single-neuron max (1000/tau_ref_E)

# ---- calibration params (REVIEW POINTS) ----
R50_FRAC = 0.3             # Psi_G r50 = this * the rE_fast TIME-peak (half-recruitment inside burst range)
R50_MIN_PEAK = 1e-3        # fail-closed: rE_fast time-peak <= this -> pool sensor never saw activity
Q_FLOOD = None             # reference q_core for TRIVIAL-A flood (default = min of the q grid)
Q_AXIAL = None             # reference q_core for TRIVIAL-B axial-retreat (default = median of the q grid)
GUARD_MARGIN = 0.05        # exclude references by this margin

# ---- phase-plane grid (REVIEW POINTS) ----
Q_GRID = np.round(np.linspace(0.25, 1.0, 8), 3)           # q_core axis
ALPHA_GRID = np.round(np.linspace(0.0, 8.0, 8), 3)        # alpha_G axis
K_MIN = 3                                                  # go(plane): min contiguous go cells
LABELS = ["decay", "blip", "trivial_A", "trivial_B", "runaway", "other_nogo", "go"]   # classify_cell labels


def build_substrate(seed=1, dt=None):
    """E1146 SNN Stage-5 subject substrate on the L=20 sheet (REUSE sef_hfo_subject_placement; blessed
    twoend_equal cores at the source/sink centroids). Data read from the main checkout (worktree results/ is
    gitignored/empty). vth = min of the two sample_core_field patches (two low-V_th cores); E->E axis along
    the registered source->sink direction."""
    m_real, src_names, snk_names = template_source_foci(SUBJECT, MONTAGE, PLACEMENT_K_EARLY, root=DATA_ROOT)
    reg = register_to_sheet(m_real, src_names, snk_names, L=L, target_inter_core_mm=TARGET_INTER_CORE)
    src_xy, snk_xy = reg["source_centroid"], reg["sink_centroid"]
    axis_unit = (snk_xy - src_xy) / np.linalg.norm(snk_xy - src_xy)          # forward = source -> sink
    # ``dt`` is optional so every existing caller remains byte-identical.  The
    # Z/M branch-decision resolution check uses this hook to rebuild the whole
    # substrate (including delay quantisation) natively at dt/2; replacing only
    # ``p.dt`` after connectivity construction would be numerically invalid.
    p = Params(g=G, L=L, density=DENSITY, T=T_SIM, dt=DT if dt is None else float(dt),
               nu_ext_ratio=DRIVE, seed=seed)
    rng = np.random.default_rng(seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=np.deg2rad(reg["theta_deg"]),
                                 AR=2.0, verbose=False)
    is_E = np.zeros(len(net["pos"]), bool); is_E[:NE] = True

    def _core_vth(xy, s):
        return sample_core_field(net["pos"], is_E, xy, CORE_R, np.random.default_rng(s),
                                 core_mean=CORE_MEAN, core_std=CORE_STD, base_mean=BASE_MEAN)["vth"]
    vth = np.minimum(_core_vth(src_xy, seed + 7), _core_vth(snk_xy, seed + 8))   # twoend_equal (two cores)
    return dict(p=p, net=net, NE=NE, NI=NI, N=NE + NI, L=L, posE=net["pos"][:NE], posI=net["pos"][NE:],
                vth=vth, src_xy=src_xy, snk_xy=snk_xy, center=reg["center"], axis_unit=axis_unit,
                reg=reg, seed=seed)


def _run(S, slow=None, seed=None):
    S["net"]["rng"] = np.random.default_rng(S["seed"] if seed is None else seed)
    return simulate_kick(S["p"], S["net"], KICK_BOOST=KICK, slow=slow, kick_center=list(S["src_xy"]),
                         r_kick=R_KICK, t_kick=T_KICK, V_th_per_neuron=S["vth"])


def _make_slow(S, q_core_val, r50, alpha_G=0.0, beta_SG=0.0):
    """SpatialSlowField with the pool ON (use_SG), q_I FROZEN (k_q=0), frozen value set per-cell by
    _freeze_qcore. V_th0=18.0 base (the per-neuron heterogeneous V_th is passed to simulate_kick, not here);
    default n_grid=32 matches the S2 substrate. alpha_G=beta_SG=0 -> neutral pool (arm-0/baseline)."""
    cfg = SpatialSlowFieldConfig(use_SG=True, alpha_G=alpha_G, beta_SG=beta_SG,
                                 r0_psi=0.0, r50_psi=r50, n_psi=2.0, p_pool=3.0, tau_mu=30.0, tau_S=80.0,
                                 S_max=1.0, use_qI=True, k_q=0.0, q_init=1.0)   # k_q=0 -> q_I frozen
    return SpatialSlowField(S["N"], 18.0, S["posE"], S["posI"], S["L"], cfg=cfg)


def _freeze_qcore(slow, core_mask, q_core_val):
    """Set the frozen q_I grid: core neurons' grid cells -> q_core_val, everything else stays 1.0."""
    iy = slow._iyE[core_mask]; ix = slow._ixE[core_mask]
    slow.q_I[iy, ix] = q_core_val
    return slow


def derive_core(S):
    """Source core = E neurons within CORE_R of the registered source centroid (the low-V_th source patch =
    the M4 single-core ignition site: q_core is frozen here and the focal kick lands here)."""
    d = np.linalg.norm(S["posE"] - S["src_xy"], axis=1)
    core = d <= CORE_R
    if not core.any():
        raise RuntimeError("no E neurons within CORE_R of the source centroid; check registration")
    return core


def _r50_from_peak(trace_rEfast_max):
    """Pure fail-closed rule (fix P1-1): r50 = R50_FRAC * time-peak of the rE_fast spatial-max trace.
    Raise CalibrationError on an empty trace or a peak <= R50_MIN_PEAK (pool sensor never saw activity)."""
    if not trace_rEfast_max:
        raise CalibrationError("calibrate_r50: no rE_fast trace recorded (use_SG off?)")
    peak = float(np.max(trace_rEfast_max))                              # time peak of the spatial-max rE_fast
    if not np.isfinite(peak) or peak <= R50_MIN_PEAK:
        raise CalibrationError(f"calibrate_r50: no valid rE_fast peak (peak={peak:.3g} <= {R50_MIN_PEAK}); "
                               "the calibration kick did not drive the pool sensor")
    return peak * R50_FRAC                                               # half-recruitment inside the burst range


def calibrate_r50(S, core):
    """Pick Psi_G r50 from the rE_fast TIME PEAK (fix P1-1: NOT slow.rE_fast at sim end, which is the
    post-burst decay ~0). Uses slow.trace_rEfast_max (per-step spatial-max recorded under use_SG)."""
    slow = _make_slow(S, q_core_val=Q_GRID.min(), r50=1.0, alpha_G=0.0)   # r50 here only affects the unused pool
    _freeze_qcore(slow, core, Q_GRID.min())
    _run(S, slow=slow)
    return _r50_from_peak(slow.trace_rEfast_max)


def _sat_ceiling_hz(S):
    """Saturation ceiling in per-neuron Hz (fix P1-2): SAT_CEILING_FRAC * single-neuron max rate
    (1000/tau_ref_E). rate_E (a per-step count) is converted to Hz inside extract_cell_metrics."""
    return SAT_CEILING_FRAC * (1000.0 / S["p"].tau_ref_E)


def _extract(S, core, res):
    # f_off is measured vs the KNOWN registered source->sink axis (not the derived onset axis), through the
    # source->sink midpoint (reg center) -- the subject's real propagation axis.
    return extract_cell_metrics(
        res, S["posE"], DT, T_KICK, core_neuron_mask=core, center=S["center"], axis_unit=S["axis_unit"],
        T_min=T_MIN_MS, band_half=BAND_HALF_MM, sat_ceiling=_sat_ceiling_hz(S),
        thresh_hz=THRESH_HZ, retreat_factor=RETREAT_FACTOR)


def reference_metrics(S, core, r50):
    """arm-0 TRIVIAL reference instances: flood (low q_core, most excitable) + axial-retreat (mid q_core)."""
    q_flood = Q_FLOOD if Q_FLOOD is not None else float(Q_GRID.min())
    q_axial = Q_AXIAL if Q_AXIAL is not None else float(np.median(Q_GRID))
    flood = _extract(S, core, _run(S, _freeze_qcore(_make_slow(S, q_flood, r50, alpha_G=0.0), core, q_flood)))
    axial = _extract(S, core, _run(S, _freeze_qcore(_make_slow(S, q_axial, r50, alpha_G=0.0), core, q_axial)))
    return flood, axial, q_flood, q_axial


def run_cell(S, core, r50, guards, q_core_val, alpha_G, beta_SG):
    """One phase-plane cell: freeze q_I at q_core_val, run the pool arm, extract metrics, classify."""
    slow = _freeze_qcore(_make_slow(S, q_core_val, r50, alpha_G=alpha_G, beta_SG=beta_SG), core, q_core_val)
    res = _run(S, slow=slow)
    m = _extract(S, core, res)
    v = classify_cell(m, guards)
    return v, m, float(np.max(slow.trace_SG) if slow.trace_SG else 0.0)


_SHARED = {}


def _worker_cell(task):
    name, iq, ia, q, aG, bG = task
    v, m, maxS = run_cell(_SHARED["S"], _SHARED["core"], _SHARED["r50"], _SHARED["guards"], q, aG, bG)
    metrics = {k: round(getattr(m, k), 4) if isinstance(getattr(m, k), float) else getattr(m, k)
               for k in m.__dataclass_fields__}
    return (name, iq, ia, bool(v.go), v.label, round(float(maxS), 4), metrics, q, aG, bG)


def sweep(S, core, r50, guards, workers=16):
    """arm 0 (baseline pool off), arm 1 (beta_SG subtractive), arm 2 (alpha_G divisive). Returns per-arm go
    grids, label grids (int codes into LABELS), and per-cell rows. PARALLEL via a fork Pool: the built S is
    shared read-only across workers by copy-on-write (net matrices shared, NO rebuild); each cell resets its
    own kick RNG (fixed seed) so the result is identical to a serial sweep."""
    arms = {"arm0_baseline": dict(alpha_G=0.0, beta_SG=0.0),
            "arm1_subtractive": dict(alpha_G=0.0, beta_SG=1.0),
            "arm2_divisive": dict(alpha_G=None, beta_SG=0.0)}   # arm2 alpha_G swept per cell
    tasks = []
    for name, arm in arms.items():
        for iq, q in enumerate(Q_GRID):
            for ia, a in enumerate(ALPHA_GRID):
                aG = float(a) if arm["alpha_G"] is None else arm["alpha_G"]
                tasks.append((name, iq, ia, float(q), aG, float(arm["beta_SG"])))
    _SHARED.update(S=S, core=core, r50=r50, guards=guards)      # set BEFORE Pool -> fork inherits (COW share)
    with mp.Pool(min(workers, len(tasks))) as pool:
        results = pool.map(_worker_cell, tasks)
    shape = (len(Q_GRID), len(ALPHA_GRID))
    go_grids = {n: np.zeros(shape, dtype=bool) for n in arms}
    label_grids = {n: np.full(shape, LABELS.index("other_nogo"), dtype=int) for n in arms}
    rows = []
    for (name, iq, ia, go, label, maxS, metrics, q, aG, bG) in results:
        go_grids[name][iq, ia] = go
        label_grids[name][iq, ia] = LABELS.index(label) if label in LABELS else LABELS.index("other_nogo")
        rows.append(dict(arm=name, q_core=q, alpha_G=aG, beta_SG=bG, label=label, go=go,
                         max_S_G=maxS, metrics=metrics))
    return go_grids, label_grids, rows


_LABEL_COLORS = {"decay": "#e8e8e8", "blip": "#b8b8b8", "trivial_A": "#f4a261", "trivial_B": "#a97155",
                 "runaway": "#e63946", "other_nogo": "#6c757d", "go": "#2a9d8f"}


def plot_and_readme(go_grids, label_grids, verdict, label_counts, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    figdir = os.path.join(out_dir, "figures")
    os.makedirs(figdir, exist_ok=True)
    ext = [ALPHA_GRID.min(), ALPHA_GRID.max(), Q_GRID.min(), Q_GRID.max()]
    n = len(go_grids)

    # Figure 1: go cells (green) per arm
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.6), constrained_layout=True)
    for ax, (name, go) in zip(np.atleast_1d(axes), go_grids.items()):
        ax.imshow(go.astype(float), origin="lower", aspect="auto", cmap="Greens", vmin=0, vmax=1, extent=ext)
        ax.set_title(name); ax.set_xlabel("alpha_G"); ax.set_ylabel("q_core")
    fig.suptitle(f"M4 Pass-1 phase-plane — go cells (green).  verdict: {verdict['verdict']}")
    fig.savefig(os.path.join(figdir, "phase_plane_qcore_alpha.png"), dpi=140); plt.close(fig)

    # Figure 2: full label phase diagram per arm (decay/blip/trivial_A/trivial_B/runaway/other_nogo/go)
    cmap = ListedColormap([_LABEL_COLORS[l] for l in LABELS])
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.9))
    for ax, (name, lbl) in zip(np.atleast_1d(axes), label_grids.items()):
        ax.imshow(lbl, origin="lower", aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(LABELS) - 0.5, extent=ext)
        ax.set_title(name); ax.set_xlabel("alpha_G"); ax.set_ylabel("q_core")
    fig.legend(handles=[Patch(color=_LABEL_COLORS[l], label=l) for l in LABELS], loc="lower center",
               ncol=len(LABELS), fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("M4 Pass-1 phase-plane — cell labels")
    fig.tight_layout(rect=[0, 0.09, 1, 0.95])
    fig.savefig(os.path.join(figdir, "phase_plane_labels.png"), dpi=140, bbox_inches="tight"); plt.close(fig)

    counts_md = "".join(f"- {name}: " + ", ".join(f"{k}={v}" for k, v in c.items() if v) + "\n"
                        for name, c in label_counts.items())
    open(os.path.join(figdir, "README.md"), "w").write(
        "### phase_plane_qcore_alpha.png\n\n"
        "M4 Pass-1 go/no-go 相平面（rev4 §9.1）。三张子图=三个臂（arm0 基线 / arm1 纯减法 / arm2 除法），"
        "横轴 alpha_G、纵轴 q_core，绿色格=go(cell)（有界+自维持+大范围+空间结构+非 TRIVIAL-A/B）。\n\n"
        f"**关注点**：arm2（除法）是否开出一片**连通**的绿区（≥K_MIN={K_MIN}）且该绿区在 arm1 里**不存在**——"
        f"go(plane) 判据；当前 verdict=`{verdict['verdict']}`（arm2 最大连通={verdict['arm2_max_contiguous']}, "
        f"arm1={verdict['arm1_max_contiguous']}）。\n\n"
        "### phase_plane_labels.png\n\n"
        "同一扫描的**完整标签**相图：每格分类为 decay / blip / trivial_A（低幅全场 skirt）/ trivial_B（轴向自限）"
        "/ runaway / other_nogo / go，看非 go 格具体属于哪种失败模式。\n\n"
        "**关注点**：go 区是否被 trivial_A / trivial_B 夹住、runaway 是否集中在低 q_core、高 alpha_G 角是否把 "
        "runaway 压回有界。\n\n"
        "### 各臂 label 计数\n\n" + counts_md)


def _m2d(m):
    return {k: (round(getattr(m, k), 4) if isinstance(getattr(m, k), float) else getattr(m, k))
            for k in m.__dataclass_fields__}


def _label_counts(rows, arm_names):
    return {name: {lab: sum(1 for r in rows if r["arm"] == name and r["label"] == lab) for lab in LABELS}
            for name in arm_names}


def _write_json(out_dir, obj):
    json.dump(obj, open(os.path.join(out_dir, "phase_plane.json"), "w"), indent=2)


def _report(meta, flood, axial, guards, val, label_counts, verdict, calibration_valid):
    print("\n===== M4 Pass-1 phase-plane report =====")
    print(f"calibration_valid = {calibration_valid}")
    print(f"n_core={meta['n_core']}  r50_psi={meta.get('r50_psi')}  sat_ceiling_hz={meta['sat_ceiling_hz']}")
    print(f"reference q_flood={meta.get('q_flood')} q_axial={meta.get('q_axial')}")
    print(f"  flood (want TRIVIAL-A): {_m2d(flood)}")
    print(f"  axial (want TRIVIAL-B): {_m2d(axial)}")
    print(f"  reference_validation: {val['reasons'] or 'OK (flood_ok & axial_ok)'}")
    if not calibration_valid:
        print("=> CALIBRATION INVALID: NO sweep, NO verdict written.\n"); return
    print(f"guards = {guards.__dict__}")
    for name, counts in label_counts.items():
        print(f"  {name}: " + ", ".join(f"{k}={v}" for k, v in counts.items() if v))
    print(f"go_plane_verdict = {verdict}")
    print("(M4 Pass-1 bounded-core / nontrivial-intermediate SCREEN — NOT proven seizure mechanism.)\n")


def main():
    ap = argparse.ArgumentParser(description="M4 Pass-1 phase-plane (RUNS SIMULATIONS — sim gate)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=16, help="parallel sweep workers (fork Pool, COW-shared substrate)")
    ap.add_argument("--confirm-run", action="store_true",
                    help="required: this runner executes the Pass-1 scientific simulation sweep")
    a = ap.parse_args()
    if not a.confirm_run:
        print("REFUSED: this is the Pass-1 SIM GATE. Re-run with --confirm-run once plan+impl are approved.")
        return
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    S = build_substrate(a.seed)
    core = derive_core(S)
    meta = dict(spec="2026-07-05 rev4 §8.1/§9.1", seed=a.seed, n_core=int(core.sum()),
                substrate=dict(subject=SUBJECT, montage=MONTAGE, L=L, N=S["N"], NE=S["NE"],
                               placement="template_source_twoend_equal", core_mean=CORE_MEAN, core_r=CORE_R,
                               src_xy=S["src_xy"].tolist(), snk_xy=S["snk_xy"].tolist(),
                               theta_deg=round(S["reg"]["theta_deg"], 1),
                               inter_core_mm=round(S["reg"]["inter_core_mm_sheet"], 2),
                               kick=dict(boost=KICK, r_kick=R_KICK, t_kick=T_KICK, on="source_core")),
                sat_ceiling_hz=round(_sat_ceiling_hz(S), 2), tau_ref_E_ms=S["p"].tau_ref_E,
                rate_hz_conversion="rate_E / NE / dt * 1e3 (per-neuron mean Hz)",
                q_grid=Q_GRID.tolist(), alpha_grid=ALPHA_GRID.tolist(), k_min=K_MIN)
    # ---- calibration (fail-closed) ----
    try:
        r50 = calibrate_r50(S, core)
        flood, axial, q_flood, q_axial = reference_metrics(S, core, r50)
    except CalibrationError as e:
        meta["calibration_error"] = str(e)
        _write_json(a.out, dict(meta=meta, calibration_valid=False, verdict=None,
                                calibration_failed=dict(stage="calibrate_r50/reference", reason=str(e))))
        print(f"CALIBRATION FAILED (no sweep, no verdict): {e}")
        return
    val = validate_reference_metrics(flood, axial)
    meta.update(r50_psi=round(r50, 5), q_flood=q_flood, q_axial=q_axial,
                flood_metrics=_m2d(flood), axial_metrics=_m2d(axial), reference_validation=val)
    if not val["valid"]:
        _write_json(a.out, dict(meta=meta, calibration_valid=False, verdict=None,
                                calibration_failed=dict(stage="validate_reference_metrics",
                                                        reasons=val["reasons"])))
        _report(meta, flood, axial, None, val, None, None, calibration_valid=False)
        return
    guards = calibrate_guards_from_references(flood, axial, margin=GUARD_MARGIN)
    meta["guards"] = guards.__dict__
    # ---- sweep (only after calibration is VALID) ----
    go_grids, label_grids, rows = sweep(S, core, r50, guards, workers=a.workers)
    verdict = go_plane_verdict(go_grids["arm2_divisive"], go_grids["arm1_subtractive"], K_MIN)
    label_counts = _label_counts(rows, list(go_grids))
    meta["wall_s"] = round(time.time() - t0, 1)
    _write_json(a.out, dict(meta=meta, calibration_valid=True, verdict=verdict,
                            label_counts=label_counts, rows=rows))
    plot_and_readme(go_grids, label_grids, verdict, label_counts, a.out)
    _report(meta, flood, axial, guards, val, label_counts, verdict, calibration_valid=True)


if __name__ == "__main__":
    main()
