"""Audit the two current E1146 TB seizures against frozen shared-plane sources."""
import sys
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from src.topic5_template_axis_field import scorers_from_interictal_record
from scripts.analyze_e1146_source_signed_correspondence import common_scores, classify_source
from scripts.plot_topic5_interictal_event_envelope_field import load_frozen, _event_field

OUT = ROOT / "results/topic5_patient_state_inference/e1146_label_and_forecast_review_20260910"
FIELD = ROOT / "results/interictal_propagation_masked/template_gradient_fields_all_events_timing_plus_space/per_subject/epilepsiae_1146.json"
LABELS = ROOT / "results/topic5_preseizure_template_association_broadband/cohort_20260909/per_subject/epilepsiae_1146/per_seizure"


def main():
    outfig = OUT / "figures"
    outfig.mkdir(parents=True, exist_ok=True)
    rec = json.loads(FIELD.read_text())
    fz = load_frozen("epilepsiae_1146", frozen_root=FIELD.parent)
    sc = scorers_from_interictal_record(rec)
    a, b = sc["shared_a"], sc["shared_b"]
    assert np.array_equal(a["points"], b["points"]) and a["sigma"] == b["sigma"]
    support = np.minimum(a["support"], b["support"])
    display_support = np.minimum(fz["support_a"], fz["support_b"])
    assert rec["names"] == fz["names"]
    old = pd.read_csv(ROOT / "results/topic5_preseizure_template_share/epilepsiae_1146/seizure_labels.csv")
    audit, examples = [], []
    for index in (18, 21):
        path = LABELS / f"seizure_{index:03d}.json"
        r = json.loads(path.read_text())
        assert r["contact_order"] == rec["names"]
        assert hashlib.sha256(FIELD.read_bytes()).hexdigest() == r["field_sha256"]
        scores, _, _ = common_scores(r["activation"], rec["rank_a"], rec["rank_b"], a["points"], support, a["sigma"])
        np.testing.assert_allclose(scores, [r["provisional_r_a"], r["provisional_r_b"]], rtol=0, atol=1e-12)
        label, _ = classify_source(*scores)
        assert label == r["qualified_source_label"] == "TB"
        row = old[old.seizure_idx.eq(index)].iloc[0]
        audit.append(dict(sz=index + 1, seizure_idx=index, seizure_id=r["seizure_id"],
                          label=label, r_a=scores[0], r_b=scores[1],
                          sql_pattern=str(row.get("pattern", "")),
                          sql_classification=str(row.get("classification", "")),
                          label_file=str(path), baseline_clinical_sec=r["baseline_clinical_sec"]))
        examples.append(np.asarray(r["activation"], float))
    pd.DataFrame(audit).to_csv(OUT / "label_audit.csv", index=False)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 4, figsize=(15, 5.3))
    values = [-np.asarray(rec["rank_a"]), -np.asarray(rec["rank_b"]), *examples]
    titles = ["Interictal TA source", "Interictal TB source", "SZ19: early seizure energy", "SZ22: early seizure energy"]
    points = np.asarray(fz["points_mm"])
    for k, (ax, value, title) in enumerate(zip(axes, values, titles)):
        v = (value - value.min()) / np.ptp(value)
        x, y, field, _, _ = _event_field(fz, v, display_support)
        cmap = "Blues"
        ax.imshow(field, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()], cmap=cmap, vmin=0, vmax=1, aspect="equal")
        ax.scatter(points[:, 0], points[:, 1], c=v, cmap=cmap, vmin=0, vmax=1, s=27, edgecolor="white", linewidth=.7)
        for name in ("SCL6", "SCL9", "ICL1", "ICL11"):
            j = rec["names"].index(name)
            ax.annotate(name, points[j], xytext=(-2, 6) if name == "ICL1" else (4, 5),
                        ha="right" if name == "ICL1" else "left", textcoords="offset points", fontsize=8)
        ax.set_title(title, fontsize=11, pad=12)
        ax.set_xlabel("Frozen shared axis (mm)")
        if k == 0:
            ax.set_ylabel("Transverse coordinate (mm)")
        else:
            ax.tick_params(labelleft=False)
        cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1) if k < 2 else Normalize(value.min(), value.max()), cmap=cmap), ax=ax, location="bottom", pad=.19, fraction=.05)
        if k < 2:
            cb.set_ticks([0, 1], labels=["Late", "Early"])
            cb.set_label("Relative template earliness", fontsize=9)
        else:
            cb.set_label("1–150 Hz log-power (baseline robust z)", fontsize=9)
            r = audit[k-2]
            ax.text(.5, -.19, f"r(TA) = {r['r_a']:.3f}; r(TB) = {r['r_b']:.3f}", transform=ax.transAxes, ha="center", fontsize=9)
    fig.suptitle("E1146 | Frozen interictal source fields and clinical-onset [0, 10] s energy", fontsize=14, y=1.055)
    fig.text(.5, .015, "Same geometry and contacts. Darker = earlier template / greater energy. Display kernel: 6 mm; correlations use the frozen scoring kernel.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.06, right=.99, top=.90, bottom=.16, wspace=.18)
    for ext in ("png", "pdf"):
        fig.savefig(outfig / f"tb_source_correspondence.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    (OUT / "label_audit.json").write_text(json.dumps(dict(
        status="CURRENT_LABEL_SCORES_REPRODUCED", cases=audit, field=str(FIELD),
        shared_geometry=True, display_sigma_mm=fz["display_sigma_mm"],
        scoring_sigma=a["sigma"], scoring_coordinate_units="frozen normalized shared plane",
        interpretation="Early ictal source-energy correspondence, not ictal propagation-order validation",
        no_relabelling=True, user_visual_acceptance="pending"), indent=2) + "\n")
    (outfig / "README.md").write_text(
        "### tb_source_correspondence.png / tb_source_correspondence.pdf\n\n"
        "左侧是冻结 TA/TB 传播模板的早晚分布，右侧是 SZ19/SZ22 临床起点后 0–10 s 的 1–150 Hz 能量。"
        "四图保持同一共享平面、接触点和共同支持；显示核为 6 mm，相关系数按原冻结评分核重新计算并与当前标签逐项核对。"
        "能量以 EEG 起点前 −120 至 −90 s 作基线，空间图保持连续强度而未重新排名。\n\n"
        "**关注点**：两例高能量区域是否更接近 TB 早期源区；这不是发作内传播顺序复现证据。待用户目视检查。\n\n"
        "### sz19/fig3-panela.png / sz19/fig3-panela.pdf\n\n"
        "SZ19（零基索引 18）的 Figure 3A 风格波形和固定 ICL2 频谱。全部 15 个冻结接触点均保留，计算沿用 canonical producer。\n\n"
        "**关注点**：原始活动是否与 TB 源区能量标签相容，勿将源区标签视为临床分类。\n\n"
        "### sz22/fig3-panela.png / sz22/fig3-panela.pdf\n\n"
        "SZ22（零基索引 21）的同口径波形和 ICL2 频谱，用于直接与 SZ19 比较。省略的时间以断轴标明。\n\n"
        "**关注点**：核对两次发作的异同及起始空间分布；待用户目视检查。\n", encoding="utf-8")
    print(pd.DataFrame(audit)[["sz", "seizure_id", "label", "r_a", "r_b", "sql_pattern", "sql_classification"]].to_string(index=False))


if __name__ == "__main__":
    main()
