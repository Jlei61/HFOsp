#!/usr/bin/env python3
"""mini-W_event PILOT orchestrator (Topic 4 M3, Step D).

Design: docs/archive/topic4/sef_hfo/m3_mini_w_event_design_2026-06-23.md.
PILOT SCOPE (§8 step 3): bare + n17.6 x 5 source x kick{0.8..1.6} x 12 seed, produce
  (1) K_min(q) figure (5 sources x 2 substrates), and
  (2) center-source W_shape reproducibility (B1a),
then STOP for review. Does NOT run off-axis B1b/c/d or the mu phase diagram.

This script ONLY orchestrates: it invokes run_m3_kick_calibration.py --emit-ea-bins per
(substrate, source) [engine untouched], then assembles via src.sef_hfo_mini_w_event.
PILOT-FIRST: requires --run to launch the heavy L=20 sweep; --smoke runs a tiny L=8
plumbing check; --dry-run prints the commands.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import sef_hfo_mini_w_event as mwe  # noqa: E402

SOURCE_NAMES = ["center", "+axis", "-axis", "+offaxis", "-offaxis"]
SUBSTRATES = ["bare", "n17.6"]
# Ceiling working point (results/.../kick_ceiling_n17.6/config.json), center = its source.
DEFAULT_CENTER = (10.0, 10.0)
THETA_AXIS_DEG = 45.0       # E->E long axis (matches --theta-ee)
R_SRC_MM = 4.0              # ~1 electrode pitch, design §1
PILOT_KICKS = [0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6]   # design §2 (extends ceiling down to 0.8)
PILOT_WINDOWS = ["18,24", "20,28", "22,32"]          # ceiling windows (EA is window-indep)


def source_xy(name, center=DEFAULT_CENTER, r_src=R_SRC_MM):
    """(x,y) of a named source: center, +/-axis along theta=45, +/-offaxis along 135."""
    cx, cy = center
    if name == "center":
        return np.array([cx, cy], dtype=float)
    th = np.deg2rad(THETA_AXIS_DEG)
    axis = np.array([np.cos(th), np.sin(th)])               # 45 deg unit
    off = np.array([np.cos(th + np.pi / 2), np.sin(th + np.pi / 2)])  # 135 deg unit
    vec = {"+axis": axis, "-axis": -axis, "+offaxis": off, "-offaxis": -off}[name]
    return np.array([cx, cy], dtype=float) + r_src * vec


def build_run_cmd(substrate, xy, out_dir, cfg):
    """Argv for one (substrate, source) runner invocation (core field stays at center)."""
    cmd = [
        sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "run_m3_kick_calibration.py"),
        "--run", "--mode", "explore",
        "--L", str(cfg["L"]), "--density", str(cfg["density"]),
        "--T", str(cfg["T"]), "--t-kick", str(cfg["t_kick"]), "--dt", "0.1",
        "--seeds", str(cfg["seeds"]),
        "--kick-boosts", *[str(k) for k in cfg["kicks"]],
        "--win-ms", *cfg["windows"],
        "--n-bins-per-axis", str(cfg["n_bins_per_axis"]), "--n-rep-bins", "1",
        "--theta-ee", str(THETA_AXIS_DEG), "--AR", "2.0", "--vth0", "18.0",
        "--kick-xy", str(xy[0]), str(xy[1]),
        "--emit-ea-bins",
        "--out-dir", out_dir,
    ]
    if substrate != "bare":          # n17.6 core; bare = no core field
        cmd += ["--core-mean", "17.6", "--core-std", "0.5", "--core-r", "1.5",
                "--core-center-xy", str(cfg["center"][0]), str(cfg["center"][1])]
    return cmd


def run_pilot(out_root, cfg, dry_run=False):
    """Run all (substrate, source) sweeps; return {(substrate, source): run_dir}."""
    run_dirs = {}
    for substrate in SUBSTRATES:
        for name in SOURCE_NAMES:
            xy = source_xy(name, center=cfg["center"], r_src=cfg["r_src"])
            out_dir = os.path.join(out_root, "runs", f"{substrate}_{name}")
            cmd = build_run_cmd(substrate, xy, out_dir, cfg)
            run_dirs[(substrate, name)] = out_dir
            if dry_run:
                print(" ".join(cmd))
                continue
            os.makedirs(out_dir, exist_ok=True)
            print(f"[mini-W pilot] {substrate} / {name} @ ({xy[0]:.2f},{xy[1]:.2f}) ...",
                  flush=True)
            subprocess.run(cmd, check=True)
    return run_dirs


def assemble_kmin(run_dirs):
    """K_min / K50 / P_EA curve per (substrate, source). Missing/failed dir -> None row."""
    rows = []
    for (substrate, name), d in run_dirs.items():
        try:
            r = mwe.load_run_dir(d)
        except (FileNotFoundError, OSError):
            rows.append({"substrate": substrate, "source": name, "status": "missing"})
            continue
        kmin = mwe.extract_kmin(r["kicks"], r["p_ea"], r["n_seeds"])
        k50 = mwe.extract_k50(r["kicks"], r["p_ea"])
        rows.append({
            "substrate": substrate, "source": name, "status": "ok",
            "kicks": r["kicks"], "p_ea": [round(p, 3) for p in r["p_ea"]],
            "n_seeds": r["n_seeds"], "k_min": kmin, "k50": k50,
            "n_spont": len(r["spont_seeds"]),
        })
    return rows


def center_w_shape_repro(run_dirs, n_null=1000, rng_seed=0):
    """B1a at the CENTER source for bare + n17.6: W_shape at near-threshold kick, then
    cross-seed reproducibility vs spatial-bin-shuffle null. Returns per-substrate dict."""
    out = {}
    for substrate in SUBSTRATES:
        d = run_dirs.get((substrate, "center"))
        info = {"substrate": substrate, "status": "no_run"}
        if d and os.path.exists(os.path.join(d, "ea_net_bins.npz")):
            r = mwe.load_run_dir(d)
            k50 = mwe.extract_k50(r["kicks"], r["p_ea"])
            kmin = mwe.extract_kmin(r["kicks"], r["p_ea"], r["n_seeds"])
            # near-threshold kick for W_shape: K50 if finite, else K_min, snapped to grid.
            ref = k50 if np.isfinite(k50) else kmin
            info = {"substrate": substrate, "status": "no_ea_local",
                    "k_min": kmin, "k50": k50, "kick_used": None}
            if np.isfinite(ref):
                ki = int(np.argmin(np.abs(np.asarray(r["npz_kicks"]) - ref)))
                kick = float(r["npz_kicks"][ki])
                recs = r["recs_by_kick"][r["kicks"][min(ki, len(r["kicks"]) - 1)]]
                succ = mwe.success_seeds_at_kick(recs, r["spont_seeds"])
                try:
                    per_seed, mean_w, used = mwe.build_w_shape(
                        r["ea_net_bins"][ki], succ, r["src_bin_idx"])
                    rep = mwe.w_shape_reproducibility(
                        per_seed, n_null=n_null, rng_seed=rng_seed)
                    info = {"substrate": substrate, "status": "ok",
                            "k_min": kmin, "k50": k50, "kick_used": kick,
                            "n_success": len(used), "used_seeds": used,
                            "per_seed_w_shape": per_seed.tolist(),
                            "mean_w_shape": mean_w.tolist(), **rep}
                except ValueError:
                    info["status"] = "no_ea_local"
        out[substrate] = info
    return out


# --------------------------------------------------------------------------- #
# Figures                                                                      #
# --------------------------------------------------------------------------- #
def plot_kmin(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = [r for r in rows if r.get("status") == "ok"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(SOURCE_NAMES))
    for i, sub in enumerate(SUBSTRATES):
        kmins = []
        for name in SOURCE_NAMES:
            r = next((r for r in ok if r["substrate"] == sub and r["source"] == name), None)
            kmins.append(r["k_min"] if r and np.isfinite(r["k_min"]) else np.nan)
        ax.plot(x, kmins, "o-", label=sub)
    ax.set_xticks(x); ax.set_xticklabels(SOURCE_NAMES)
    ax.set_ylabel("K_min  (min kick with P_EA-local-returned >= 0.7)")
    ax.set_xlabel("kick source")
    ax.set_title("mini-W_event K_min(q): recruitment threshold by source")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_center_wshape(repro, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(SUBSTRATES), figsize=(10, 4), squeeze=False)
    for j, sub in enumerate(SUBSTRATES):
        ax = axes[0][j]
        info = repro.get(sub, {})
        if info.get("status") == "ok":
            per_seed = np.asarray(info["per_seed_w_shape"])
            for row in per_seed:
                ax.plot(row, color="0.6", lw=0.8)
            ax.plot(np.asarray(info["mean_w_shape"]), "k-", lw=2, label="mean")
            ax.set_title(f"{sub}  center W_shape (kick={info['kick_used']})\n"
                         f"B1a obs={info['observed']:.2f} null95={info['null_p95']:.2f} "
                         f"{'PASS' if info['pass'] else 'n.s.'}  (n={info['n_success']})")
        else:
            ax.text(0.5, 0.5, f"{sub}\n{info.get('status','?')}", ha="center", va="center")
            ax.set_title(f"{sub}  center W_shape")
        ax.set_xlabel("non-source bin"); ax.set_ylabel("normalized early recruitment")
    fig.suptitle("center-source W_shape reproducibility (B1a)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def write_readme(fig_dir, kmin_rows, repro):
    lines = ["# mini-W_event pilot 图说明（中文）\n"]
    lines.append("### kmin_by_source.png")
    lines.append("每个 kick 源（center / ±轴 / ±off-axis）触发"
                 "“有限幅、局部、回静”事件所需的最小 kick（K_min，P_EA-local-returned≥0.7）。"
                 "bare vs n17.6 两条线：若核在多个源上压低 K_min，说明核效应是空间分布的、不是单点。")
    lines.append("**关注点**：n17.6 是否整体低于 bare；以及是否只在 center 附近低（局域）还是各源都低（全局）。\n")
    lines.append("### center_wshape_repro.png")
    lines.append("中心源在近阈 kick 点着、回静、不失控的成功事件的早期被招募形状 W_shape（按 bin），"
                 "灰线=各 seed，黑线=均值；标题给出 B1a 跨 seed 相似度 observed 与 bin-shuffle 零分布 95 分位。")
    lines.append("**关注点**：observed 是否≥null95（PASS=形状跨 seed 可重复）；bare 与 n17.6 形状是否相像。\n")
    with open(os.path.join(fig_dir, "README.md"), "w") as f:
        f.write("\n".join(lines))


def _cfg(smoke):
    # n_bins_per_axis MUST be 5 (25 bins) to match the ceiling working point: the ceiling
    # used a 5x5 grid (thresholds.json n_bins=25 — NOT in config.sweep_parameters, easy to
    # miss). On 5x5 the sheet center [10,10] is a bin CENTROID (bin 12), so the center kick
    # has a correct radius reference; on 4x4 it is the 4-bin junction (no centroid within
    # ~3.5mm) which inflates r95_ea/far_ea and flips the argmin source bin. bins_cap also
    # scales with n_bins (0.5*25=12.5), matching the ceiling locality cap.
    if smoke:
        return {"L": 8.0, "density": 100.0, "T": 120.0, "t_kick": 50.0, "seeds": 2,
                "kicks": [1.0, 1.4], "windows": ["18,28"], "n_bins_per_axis": 5,
                "center": (4.0, 4.0), "r_src": 1.5}
    return {"L": 20.0, "density": 100.0, "T": 500.0, "t_kick": 100.0, "seeds": 12,
            "kicks": PILOT_KICKS, "windows": PILOT_WINDOWS, "n_bins_per_axis": 5,
            "center": DEFAULT_CENTER, "r_src": R_SRC_MM}


def main(argv=None):
    p = argparse.ArgumentParser(description="mini-W_event pilot orchestrator (Step D)")
    p.add_argument("--out-root", default="results/topic4_sef_hfo/m3_local_w/mini_w_event")
    p.add_argument("--run", action="store_true",
                   help="launch the heavy L=20 sweep (PILOT-FIRST; off by default)")
    p.add_argument("--smoke", action="store_true",
                   help="tiny L=8 plumbing check (fast); overrides --run gate")
    p.add_argument("--dry-run", action="store_true", help="print the runner commands only")
    p.add_argument("--assemble-only", action="store_true",
                   help="skip running; assemble K_min + W_shape from existing runs/ dirs")
    p.add_argument("--n-null", type=int, default=1000)
    args = p.parse_args(argv)

    if not (args.run or args.smoke or args.dry_run or args.assemble_only):
        print("[mini-W pilot] PILOT-FIRST: pass --run (heavy L=20) / --smoke (tiny) / "
              "--dry-run / --assemble-only. Nothing was run.")
        return 0

    cfg = _cfg(args.smoke)
    if args.assemble_only:
        run_dirs = {(s, n): os.path.join(args.out_root, "runs", f"{s}_{n}")
                    for s in SUBSTRATES for n in SOURCE_NAMES}
    else:
        run_dirs = run_pilot(args.out_root, cfg, dry_run=args.dry_run)
        if args.dry_run:
            return 0

    kmin_rows = assemble_kmin(run_dirs)
    repro = center_w_shape_repro(run_dirs, n_null=args.n_null)

    fig_dir = os.path.join(args.out_root, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    with open(os.path.join(args.out_root, "kmin_by_source.json"), "w") as f:
        json.dump({"kmin_rows": kmin_rows, "center_w_shape_repro":
                   {k: {kk: vv for kk, vv in v.items() if kk != "per_seed_w_shape"}
                    for k, v in repro.items()}}, f, indent=1)
    plot_kmin(kmin_rows, os.path.join(fig_dir, "kmin_by_source.png"))
    plot_center_wshape(repro, os.path.join(fig_dir, "center_wshape_repro.png"))
    write_readme(fig_dir, kmin_rows, repro)
    print(f"[mini-W pilot] wrote figures + JSON -> {args.out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
