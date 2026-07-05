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
import sys                                                 # noqa: E402
import time                                                # noqa: E402

import numpy as np                                         # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

from params import Params                                  # noqa: E402
from connectivity import place_neurons                     # noqa: E402
from connectivity_rot import build_connectivity_rot        # noqa: E402
from kick_probe import simulate_kick                       # noqa: E402
from slow_field import SpatialSlowField, SpatialSlowFieldConfig  # noqa: E402
from src.sef_hfo_snn_metrics import onset_times            # noqa: E402
from src.sef_hfo_m4_metrics import extract_cell_metrics    # noqa: E402
from src.sef_hfo_m4_phaseplane import (                    # noqa: E402
    GuardThresholds, calibrate_guards_from_references, classify_cell,
    q_core as q_core_of, go_plane_verdict,
)

OUT_DIR = os.path.join(ROOT, "results", "topic4_m4")

# ---- kick / substrate ----
L = 6.0; DENSITY = 100.0; T_SIM = 400.0; DT = 0.1; NU = 0.6
THETA_EE = np.radians(45); AR = 2.0
KICK_BOOST = 8.0; R_KICK = 1.5; T_KICK = 50.0
V_TH = 16.5                                                # excitable core threshold
CORE_FRAC = 0.10                                           # earliest-activating fraction = core (rev4 §8.1)

# ---- metric-extraction params (REVIEW POINTS) ----
T_MIN_MS = 60.0            # persist: burst sustained >= this after the kick
BAND_HALF_MM = 1.0         # f_off: perpendicular band half-width around the onset axis
THRESH_HZ = 10.0           # branching: supra-threshold bins (matches pre_kick_ignition rest ref)
RETREAT_FACTOR = 0.5       # spatial self-limit: late extent < this * peak extent
SAT_CEILING_FRAC = 0.5     # saturation ceiling = this * NE (per-step count) -> runaway guard

# ---- calibration params (REVIEW POINTS) ----
R50_PCTL = 90.0            # Psi_G r50 = this percentile of the peak-window rE_fast (focal-in-range)
Q_FLOOD = None             # reference q_core for TRIVIAL-A flood (default = min of the q grid)
Q_AXIAL = None             # reference q_core for TRIVIAL-B axial-retreat (default = median of the q grid)
GUARD_MARGIN = 0.05        # exclude references by this margin

# ---- phase-plane grid (REVIEW POINTS) ----
Q_GRID = np.round(np.linspace(0.25, 1.0, 8), 3)           # q_core axis
ALPHA_GRID = np.round(np.linspace(0.0, 8.0, 8), 3)        # alpha_G axis
K_MIN = 3                                                  # go(plane): min contiguous go cells


def build_substrate(seed=1):
    p = Params(L=L, density=DENSITY, T=T_SIM, dt=DT, nu_ext_ratio=NU, seed=seed)
    rng = np.random.default_rng(seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=THETA_EE, AR=AR)
    posE = pos[labels == 0]; posI = pos[labels == 1]
    return dict(p=p, net=net, labels=labels, NE=NE, NI=NI, posE=posE, posI=posI,
                vth=np.full(NE + NI, V_TH), center=np.array([L / 2, L / 2]), seed=seed)


def _run(S, slow=None, seed=None):
    S["net"]["rng"] = np.random.default_rng(S["seed"] if seed is None else seed)
    return simulate_kick(S["p"], S["net"], KICK_BOOST=KICK_BOOST, slow=slow, kick_center=S["center"],
                         r_kick=R_KICK, t_kick=T_KICK, V_th_per_neuron=S["vth"])


def _make_slow(S, q_core_val, r50, alpha_G=0.0, beta_SG=0.0):
    """SpatialSlowField with the pool ON (use_SG), q_I FROZEN (k_q=0) at q_core_val on the core cells / 1
    elsewhere. alpha_G=beta_SG=0 -> neutral pool (arm-0/baseline observation)."""
    cfg = SpatialSlowFieldConfig(n_grid=8, use_SG=True, alpha_G=alpha_G, beta_SG=beta_SG,
                                 r0_psi=0.0, r50_psi=r50, n_psi=2.0, p_pool=3.0, tau_mu=30.0, tau_S=80.0,
                                 S_max=1.0, use_qI=True, k_q=0.0, q_init=1.0)   # k_q=0 -> q_I frozen
    slow = SpatialSlowField(S["NE"] + S["NI"], V_TH, S["posE"], S["posI"], L, cfg=cfg)
    return slow


def _freeze_qcore(slow, core_mask, q_core_val):
    """Set the frozen q_I grid: core neurons' grid cells -> q_core_val, everything else stays 1.0."""
    iy = slow._iyE[core_mask]; ix = slow._ixE[core_mask]
    slow.q_I[iy, ix] = q_core_val
    return slow


def derive_core(S):
    """Core neuron mask from an arm-0 (no pool) kick: the earliest-activating CORE_FRAC of E neurons."""
    res = _run(S, slow=None)
    onset = onset_times(res["E_spk_bool"], DT, T_KICK)
    fin = np.isfinite(onset)
    if not fin.any():
        raise RuntimeError("arm-0 reference kick produced no E activation; cannot derive core")
    thr = np.quantile(onset[fin], CORE_FRAC)
    return fin & (onset <= thr)


def calibrate_r50(S, core):
    """Pick Psi_G r50 from the peak-window rE_fast scale so the pool bites (the flagged pre-req: default
    r50=1.0 is above the fast-EMA sensor scale). Runs a neutral (alpha_G=0) pool kick and reads rE_fast."""
    slow = _make_slow(S, q_core_val=Q_GRID.min(), r50=1.0, alpha_G=0.0)   # r50 here only affects unused pool
    _freeze_qcore(slow, core, Q_GRID.min())
    _run(S, slow=slow)
    peak = float(np.max(slow.rE_fast))
    # r50 = a high percentile of the active rE_fast cells at the trace end is unstable (post-burst decay);
    # use a fraction of the observed peak so Psi_G's half-recruitment sits inside the burst range.
    return max(1e-3, peak * (R50_PCTL / 100.0))


def _extract(S, core, res):
    return extract_cell_metrics(
        res, S["posE"], DT, T_KICK, core_neuron_mask=core, center=_core_centroid(S, core),
        T_min=T_MIN_MS, band_half=BAND_HALF_MM, sat_ceiling=SAT_CEILING_FRAC * S["NE"],
        thresh_hz=THRESH_HZ, retreat_factor=RETREAT_FACTOR)


def _core_centroid(S, core):
    return S["posE"][core].mean(axis=0)


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


def sweep(S, core, r50, guards):
    """arm 0 (baseline q_I+g_K+h_G off, pool off), arm 1 (beta_SG subtractive), arm 2 (alpha_G divisive)."""
    arms = {"arm0_baseline": dict(alpha_G=0.0, beta_SG=0.0),
            "arm1_subtractive": dict(alpha_G=0.0, beta_SG=1.0),
            "arm2_divisive": dict(alpha_G=None, beta_SG=0.0)}   # arm2 alpha_G swept per cell
    grids = {}
    rows = []
    for name, arm in arms.items():
        go = np.zeros((len(Q_GRID), len(ALPHA_GRID)), dtype=bool)
        for iq, q in enumerate(Q_GRID):
            for ia, a in enumerate(ALPHA_GRID):
                aG = a if arm["alpha_G"] is None else arm["alpha_G"]
                bG = arm["beta_SG"]
                v, m, maxS = run_cell(S, core, r50, guards, float(q), float(aG), float(bG))
                go[iq, ia] = v.go
                rows.append(dict(arm=name, q_core=float(q), alpha_G=float(aG), beta_SG=float(bG),
                                 label=v.label, go=v.go, max_S_G=round(maxS, 4),
                                 metrics={k: round(getattr(m, k), 4) if isinstance(getattr(m, k), float)
                                          else getattr(m, k) for k in m.__dataclass_fields__}))
        grids[name] = go
    return grids, rows


def plot_and_readme(grids, rows, verdict, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = os.path.join(out_dir, "figures")
    os.makedirs(figdir, exist_ok=True)
    fig, axes = plt.subplots(1, len(grids), figsize=(4 * len(grids), 3.6), constrained_layout=True)
    if len(grids) == 1:
        axes = [axes]
    for ax, (name, go) in zip(axes, grids.items()):
        ax.imshow(go.astype(float), origin="lower", aspect="auto", cmap="Greens", vmin=0, vmax=1,
                  extent=[ALPHA_GRID.min(), ALPHA_GRID.max(), Q_GRID.min(), Q_GRID.max()])
        ax.set_title(name); ax.set_xlabel("alpha_G"); ax.set_ylabel("q_core")
    fig.suptitle(f"M4 Pass-1 phase-plane — go cells (green).  verdict: {verdict['verdict']}")
    fig.savefig(os.path.join(figdir, "phase_plane_qcore_alpha.png"), dpi=140)
    plt.close(fig)
    with open(os.path.join(figdir, "README.md"), "w") as f:
        f.write("### phase_plane_qcore_alpha.png\n\n"
                "M4 Pass-1 go/no-go 相平面（rev4 §9.1）。每张子图一个臂（arm0 基线 / arm1 纯减法 / arm2 除法），"
                "横轴 alpha_G、纵轴 q_core，绿色格=go(cell)（有界+自维持+大范围+空间结构+非 TRIVIAL-A/B）。\n\n"
                f"**关注点**：arm2（除法）是否开出一片**连通**的绿区（面积≥K_MIN={K_MIN}），且该绿区在 arm1"
                "（纯减法）里**不存在**——这是 go(plane) 的判据；当前 verdict = "
                f"`{verdict['verdict']}`（arm2 最大连通={verdict['arm2_max_contiguous']}, "
                f"arm1={verdict['arm1_max_contiguous']}）。\n")


def main():
    ap = argparse.ArgumentParser(description="M4 Pass-1 phase-plane (RUNS SIMULATIONS — sim gate)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=OUT_DIR)
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
    r50 = calibrate_r50(S, core)
    flood, axial, q_flood, q_axial = reference_metrics(S, core, r50)
    guards = calibrate_guards_from_references(flood, axial, margin=GUARD_MARGIN)
    grids, rows = sweep(S, core, r50, guards)
    verdict = go_plane_verdict(grids["arm2_divisive"], grids["arm1_subtractive"], K_MIN)
    wall = time.time() - t0
    out = dict(meta=dict(spec="2026-07-05 rev4 §8.1/§9.1", seed=a.seed, n_core=int(core.sum()),
                         r50_psi=round(r50, 5), q_flood=q_flood, q_axial=q_axial,
                         guards=guards.__dict__, q_grid=Q_GRID.tolist(), alpha_grid=ALPHA_GRID.tolist(),
                         k_min=K_MIN, wall_s=round(wall, 1)),
               verdict=verdict, rows=rows)
    json.dump(out, open(os.path.join(a.out, "phase_plane.json"), "w"), indent=2)
    plot_and_readme(grids, rows, verdict, a.out)
    print(f"verdict={verdict['verdict']} (arm2={verdict['arm2_max_contiguous']}, "
          f"arm1={verdict['arm1_max_contiguous']}); wrote {a.out}/phase_plane.json in {wall:.0f}s")


if __name__ == "__main__":
    main()
