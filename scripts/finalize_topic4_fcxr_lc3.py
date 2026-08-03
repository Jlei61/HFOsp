#!/usr/bin/env python3
"""Aggregate completed LC3 axes, render diagnostic figures and archive honest claims."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import run_topic4_fcxr_lc3 as E01  # noqa: E402
from src.topic4_fcxr_lc3_xcal import return_brackets  # noqa: E402


OUT = E01.OUT
FIG = os.path.join(OUT, "figures")
ARCHIVE = os.path.join(
    ROOT, "docs", "archive", "topic4", "sef_hfo", "fcxr_lc3_execution_2026-08-03.md")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _load(path):
    with open(path) as f:
        return json.load(f)


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def _write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        f.write(text)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def _existing(path):
    return _load(path) if os.path.isfile(path) else None


def _materialize_index_artifacts():
    geometry = _existing(os.path.join(OUT, "geometry_map.json"))
    targets = _existing(os.path.join(OUT, "d_field_targets.json"))
    dmeans = ({"D_healthy": 0.0, **targets["target_means_D"]}
              if targets is not None else {})
    brackets = return_brackets(geometry["rows"], dmeans) if geometry is not None else []
    boundary = dict(
        status="COMPLETE" if geometry is not None else "GEOMETRY_UNRESOLVED",
        probability_contours_authorized=False, return_brackets=brackets,
        h1_low_start_entry=(geometry.get("h1_low_start_entry", []) if geometry else []),
        h1_high_start_survival=(geometry.get("h1_high_start_survival", []) if geometry else []),
        h1_high_to_low_return=(geometry.get("h1_high_to_low_return", []) if geometry else []),
        source_path=os.path.join(OUT, "geometry_map.json"), completed=_now())
    _write_json(os.path.join(OUT, "boundary_replication.json"), boundary)

    slow_src = os.path.join(OUT, "slow_vector_field", "slow_vector_field.json")
    if os.path.isfile(slow_src):
        slow = _load(slow_src)
        slow["source_path"] = slow_src
        slow["source_sha256"] = _sha(slow_src)
        _write_json(os.path.join(OUT, "slow_vector_field.json"), slow)

    recon_manifest_src = os.path.join(OUT, "dynamic_reconnaissance", "manifest.json")
    recon_agg_src = os.path.join(OUT, "dynamic_reconnaissance", "aggregate.json")
    if os.path.isfile(recon_manifest_src):
        rec = _load(recon_manifest_src); rec["source_path"] = recon_manifest_src
        rec["source_sha256"] = _sha(recon_manifest_src)
        _write_json(os.path.join(OUT, "reconnaissance_manifest.json"), rec)
    if os.path.isfile(recon_agg_src):
        rec = _load(recon_agg_src); rec["source_path"] = recon_agg_src
        rec["source_sha256"] = _sha(recon_agg_src)
        _write_json(os.path.join(OUT, "reconnaissance_verdict.json"), rec)

    sensitivity = dict(
        status="COMPLETE" if geometry is not None else "UNRESOLVED",
        return_brackets=brackets,
        high_start_rows=([dict(row_id=row["row_id"], d_label=row["d_label"],
                               a_x=row["a_x"], label=row["resolved_label"])
                          for row in geometry["rows"]
                          if row["point_id"] == "H1_ts1.25_r025"
                          and row["state_kind"] == "high"] if geometry else []),
        interpretation="frozen availability sensitivity; not dynamic X causality",
        completed=_now())
    _write_json(os.path.join(OUT, "x_field_sensitivity.json"), sensitivity)
    return geometry, dmeans, brackets


def _plot_geometry(geometry, dmeans, brackets):
    if geometry is None:
        return None
    colors = {
        "INTERICTAL_WORKPOINT": "#4C78A8", "FINITE_HIGH_FIXED": "#D55E00",
        "FINITE_HIGH_ORBIT": "#E69F00", "SATURATED_TONIC_BAD_DATA": "#7A5195",
        "UNRESOLVED": "#999999",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), constrained_layout=True)
    for ax, kind, title in zip(axes, ("low", "high"), ("low-state start", "high-state start")):
        rows = [r for r in geometry["rows"] if r["point_id"] == "H1_ts1.25_r025"
                and r["state_kind"] == kind]
        for label in sorted({r["resolved_label"] for r in rows}):
            use = [r for r in rows if r["resolved_label"] == label]
            ax.scatter([dmeans[r["d_label"]] for r in use], [r["a_x"] for r in use],
                       s=42, color=colors.get(label, "#777777"), label=label.replace("_", " ").lower())
        if kind == "high":
            for row in brackets:
                ax.plot([row["mean_D"], row["mean_D"]],
                        [row["a_return_max"], row["a_survive_min"]], color="black", lw=2)
        ax.set(title=title, xlabel="mean inhibitory depletion D", ylabel="frozen availability aX")
        ax.set_ylim(0.05, 1.05)
        ax.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False)
    fig.suptitle("Frozen D–X geometry (single microstate; descriptive brackets)", fontsize=11)
    path = os.path.join(FIG, "geometry_map.png")
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
    return path


def _plot_dynamic():
    agg = _existing(os.path.join(OUT, "dynamic_reconnaissance", "aggregate.json"))
    if agg is None:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)
    palette = {401: "#B2182B", 405: "#2166AC", 406: "#4D9221"}
    primary = None
    for row in agg["rows"]:
        noise = int(row["noise_seed"])
        js = _load(os.path.join(OUT, "dynamic_reconnaissance", f"recon_noise{noise}.json"))
        with np.load(js["output_npz"]) as z:
            t = np.arange(z["rate_E"].size) * float(z["rate_dt_ms"][0]) * 1e-3
            axes[0].plot(t, z["rate_E"], lw=0.8, color=palette[noise], label=f"noise {noise}")
            if noise == 401:
                primary = (js, {key: z[key].copy() for key in z.files})
    axes[0].set(xlabel="time (s)", ylabel="population rate (Hz)", title="no-kick trajectories")
    axes[0].legend(frameon=False, fontsize=8); axes[0].grid(alpha=0.2)
    if primary is not None:
        js, z = primary
        d = z["D_fields"].mean(axis=1); x = z["X_fields"].mean(axis=1)
        axes[1].plot(d, x, "o-", color="#7A5195")
        for i, name in enumerate(z["landmark_names"]):
            axes[1].annotate(str(name), (d[i], x[i]), fontsize=7)
        axes[1].set(xlabel="mean D", ylabel="mean aX", title="primary slow path")
        means = [field.mean(axis=1) for field in (z["D_fields"], z["X_fields"], z["H_fields"])]
        names = ["D", "aX", "H"]
        tt = z["landmark_times_ms"] * 1e-3
        for values, name in zip(means, names):
            axes[2].plot(tt, values, "o-", label=name)
        axes[2].set(xlabel="landmark time (s)", ylabel="spatial mean", title="slow coordinates")
        axes[2].legend(frameon=False); axes[2].grid(alpha=0.2)
    fig.suptitle("Dynamic reconnaissance (not parameter acceptance)", fontsize=11)
    path = os.path.join(FIG, "dynamic_trajectory.png")
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
    return path


def _plot_spatial():
    direct = _existing(os.path.join(OUT, "spatial_direct_response.json"))
    lock = _existing(os.path.join(OUT, "spatial_probe_lock.json"))
    if direct is None or lock is None or not lock.get("states"):
        return None
    state_id = lock["states"][0]["state_id"]
    state = _load(os.path.join(OUT, "spatial_probe_cells", f"{state_id}.json"))
    amps = sorted({row["amplitude"] for row in state["positive_rows"]})
    amp = amps[-1]
    rows = [row for row in state["positive_rows"] if row["amplitude"] == amp]
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.8), constrained_layout=True)
    names = [row["pattern"] for row in rows]
    gains = [row["metrics"]["finite_time_gain"]["300.0"] for row in rows]
    axes[0].bar(np.arange(len(rows)), gains, color="#4C78A8")
    axes[0].set_xticks(np.arange(len(rows)), names, rotation=45, ha="right", fontsize=7)
    axes[0].set(ylabel="finite-time gain", title="positive response at 300 ms")
    for signed in state["signed_response"]:
        s = signed["svd"]["300.0"]["singular_values"]
        axes[1].plot(np.arange(1, len(s) + 1), s, "o-",
                     label=f"A={signed['amplitude_reference']:.3g}")
    axes[1].set(xlabel="projected mode", ylabel="singular value", title="response SVD")
    axes[1].legend(frameon=False, fontsize=8); axes[1].grid(alpha=0.2)
    labels = list(direct["state_labels"])
    values = [direct["state_labels"][key].replace("_", " ").lower() for key in labels]
    axes[2].axis("off")
    axes[2].set_title("state-level interpretation")
    axes[2].text(0.02, 0.95, "\n\n".join(f"{a}\n{b}" for a, b in zip(labels, values)),
                 va="top", fontsize=8)
    fig.suptitle("Exact-state direct response (projected, not an eigenmode fit)", fontsize=11)
    path = os.path.join(FIG, "spatial_direct_response.png")
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
    return path


def _plot_lifecycle_candidate():
    verdict = _existing(os.path.join(OUT, "lifecycle_verdict.json"))
    if verdict is None or not verdict.get("candidate_rows"):
        return None
    row_id = verdict["candidate_rows"][0]
    js = _load(os.path.join(OUT, "lifecycle_cells", f"{row_id}.json"))
    with np.load(js["fields_path"]) as z:
        t = np.arange(z["rate_E"].size) * float(z["rate_dt_ms"][0]) * 1e-3
        fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), constrained_layout=True)
        axes[0].plot(t, z["rate_E"], color="#222222", lw=0.8)
        axes[0].axvline(js["onset_ms"] * 1e-3, color="#B2182B", ls="--")
        axes[0].axvline(js["offset_ms"] * 1e-3, color="#2166AC", ls="--")
        axes[0].set(xlabel="time (s)", ylabel="population rate (Hz)", title="nominal lifecycle")
        tt = z["landmark_times_ms"] * 1e-3
        axes[1].plot(tt, z["D_fields"].mean(axis=1), "o-", label="D")
        axes[1].plot(tt, z["X_fields"].mean(axis=1), "o-", label="aX")
        axes[1].plot(tt, z["H_fields"].mean(axis=1), "o-", label="H")
        axes[1].legend(frameon=False); axes[1].set(title="slow landmarks", xlabel="time (s)")
        ratios = js["statistical_return"].get("ratios", {})
        axes[2].bar(np.arange(len(ratios)), list(ratios.values()), color="#7A5195")
        axes[2].axhline(1.0, color="black", lw=1)
        axes[2].set_xticks(np.arange(len(ratios)), list(ratios), rotation=45, ha="right", fontsize=7)
        axes[2].set(title="post / pre event statistics", ylabel="ratio")
        fig.suptitle("Lifecycle candidate (morphology not yet tested)", fontsize=11)
        path = os.path.join(FIG, "lifecycle_candidate.png")
        fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
    return path


def _verdict_axes(geometry, brackets):
    e0 = _load(os.path.join(OUT, "prepared_state_contract.json"))
    recon = _existing(os.path.join(OUT, "reconnaissance_verdict.json"))
    spatial = _existing(os.path.join(OUT, "spatial_direct_response.json"))
    xcal = _existing(os.path.join(OUT, "x_calibration.json"))
    life = _existing(os.path.join(OUT, "lifecycle_verdict.json"))
    temporal = bool(life and life.get("status") == "TEMPORAL_LIFECYCLE_CANDIDATE")
    axial = bool(spatial and any(label == "AXIAL_LOCAL_DIRECT_RESPONSE"
                                 for label in spatial.get("state_labels", {}).values()))
    if temporal and axial:
        overall = "SPATIOTEMPORAL_LIFECYCLE_CANDIDATE"
    elif temporal:
        overall = "TEMPORAL_LIFECYCLE_POSITIVE_SPATIAL_MECHANISM_NOT_AXIAL"
    else:
        overall = "NO_LIFECYCLE_CANDIDATE_IN_REGISTERED_LC3_PROGRAM"
    payload = dict(
        status="COMPLETE", overall=overall,
        exact_state=dict(status=e0["status"], clauses=e0["clauses"]),
        frozen_geometry=dict(status=(geometry.get("status") if geometry else "UNRESOLVED"),
                             n_return_brackets=len(brackets)),
        dynamic_reconnaissance=recon,
        spatial_direct_response=(spatial if spatial else {"status": "UNRESOLVED"}),
        x_calibration=(xcal if xcal else {"status": "NOT_RUN"}),
        lifecycle=(life if life else {"status": "NOT_RUN"}),
        claim_boundary=("LC3 tests bounded lifecycle core and spatial susceptibility; M/K/A/ELR "
                        "morphology and patient-level seizure reproduction remain outside scope"),
        completed=_now())
    _write_json(os.path.join(OUT, "verdict_axes.json"), payload)
    return payload


def _write_docs(verdict, figures):
    readme_parts = []
    descriptions = {
        "geometry_map.png": ("冻结 D–X 几何：左边从低态出发，右边从高态出发；黑色短线只表示同一 D 下实测的返回/存活括号。",
                             "看是否存在高态能被 X 推回低态的经验边界，不把单微状态格点写成概率分岔图。"),
        "dynamic_trajectory.png": ("三条无 kick 动态侦察及 primary seed 的 D–X/H landmarks。",
                                   "看真实慢轨迹是否进入高态、是否 offset，以及它是否真正穿过冻结几何括号。"),
        "spatial_direct_response.png": ("exact-state 正刺激与 signed 中心差分响应的低维汇总。",
                                        "看 axial、transverse、global 与 shuffled 控制是否可分；SVD 不是 Jacobian eigenmode。"),
        "lifecycle_candidate.png": ("仅在严格 lifecycle gate 通过时生成，展示单条候选及 post/pre 多变量返回。",
                                    "候选仍不包含 M/K/A/ELR morphology，不能写成患者样发作已复现。"),
    }
    for path in figures:
        if path is None:
            continue
        name = os.path.basename(path); body, focus = descriptions[name]
        readme_parts.append(f"### {name}\n\n{body}\n\n**关注点**：{focus}\n")
    _write_text(os.path.join(FIG, "README.md"), "\n".join(readme_parts))

    status = f"""# FCXR-LC3 status

- overall: `{verdict['overall']}`
- exact state: `{verdict['exact_state']['status']}`
- frozen geometry: `{verdict['frozen_geometry']['status']}`; return brackets={verdict['frozen_geometry']['n_return_brackets']}
- X calibration: `{verdict['x_calibration'].get('status')}`
- lifecycle: `{verdict['lifecycle'].get('status')}`
- claim boundary: {verdict['claim_boundary']}

Generated: {verdict['completed']}
"""
    _write_text(os.path.join(OUT, "STATUS.md"), status)
    archive = f"""# FCXR-LC3 execution closeout

## 一句话结论

`{verdict['overall']}`。

## 分层判决

- 工程 exact-state：`{verdict['exact_state']['status']}`；完整膜电位、refractory、突触滤波、delay ring、慢场与 RNG continuation 受 byte-parity 合同约束。
- 冻结几何：`{verdict['frozen_geometry']['status']}`；同一 D 下共得到 {verdict['frozen_geometry']['n_return_brackets']} 个经验 return/survival bracket，单微状态不解释成概率分岔。
- 动态侦察：`{verdict['dynamic_reconnaissance'].get('status') if verdict['dynamic_reconnaissance'] else 'UNRESOLVED'}`。
- 空间直接响应：`{verdict['spatial_direct_response'].get('overall_label')}`；只报告 exact-state finite-time response，不把 SVD 称作 eigenmode。
- X 标定：`{verdict['x_calibration'].get('status')}`。
- lifecycle：`{verdict['lifecycle'].get('status')}`。

## 科学边界

{verdict['claim_boundary']}。任何 kick/reset/parameter step 均不计入 lifecycle；mean-rate 回落也不等于统计恢复。

## 产物

- result root: `{OUT}`
- figures: `{FIG}`
- verdict: `{os.path.join(OUT, 'verdict_axes.json')}`

Generated: {verdict['completed']}
"""
    _write_text(ARCHIVE, archive)


def main():
    os.makedirs(FIG, exist_ok=True)
    done = os.path.join(OUT, "X_LIFECYCLE_AUTOPILOT_DONE.json")
    if not os.path.isfile(done) or _load(done).get("status") != "DONE":
        raise SystemExit("LC3 downstream program is not complete")
    geometry, dmeans, brackets = _materialize_index_artifacts()
    figures = [
        _plot_geometry(geometry, dmeans, brackets),
        _plot_dynamic(),
        _plot_spatial(),
        _plot_lifecycle_candidate(),
    ]
    verdict = _verdict_axes(geometry, brackets)
    _write_docs(verdict, figures)
    _write_json(os.path.join(OUT, "FINALIZE_DONE.json"), dict(
        status="DONE", verdict_axes_sha256=_sha(os.path.join(OUT, "verdict_axes.json")),
        figures=[path for path in figures if path is not None], archive=ARCHIVE,
        finished=_now()))
    print(json.dumps(dict(overall=verdict["overall"],
                          figures=[path for path in figures if path is not None],
                          archive=ARCHIVE), indent=2))


if __name__ == "__main__":
    main()
