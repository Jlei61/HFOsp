#!/usr/bin/env python3
"""Render the three cohort panels, stack them and write the figure README.

Each panel is produced by its own script and kept as an independent file; this
only aligns them into the lettered layout the style guide asks for, so nothing
is redrawn here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
DEFAULT_CONFIG = ROOT / "config/topic4_data_driven_snn_cohort_formal_v1.json"
PANELS = (
    ("A", "topic4_data_driven_snn_cohort_statistics",
     "plot_topic4_data_driven_snn_cohort.py"),
    ("B", "topic4_cohort_representative_readout",
     "plot_topic4_cohort_representative_readout.py"),
    ("C", "topic4_cohort_representative_kmeans",
     "plot_topic4_cohort_representative_kmeans.py"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _readme(result: dict, config: dict, panels: list[dict]) -> str:
    cohort = result["cohort"]
    primary = cohort["primary_test"]
    sensitivity = cohort.get("sensitivity") or {}
    representative = result["representative_subject"]["subject_id"]
    denominators = result["denominators"]
    null = json.loads(
        (ROOT / config["output_root"] / "cohort_layout_audit.json").read_text()
    )["within_shaft_null"]
    coarse = null["subjects_that_cannot_reach_p_0_05_alone"]
    return f"""# 图说明

### topic4_data_driven_snn_cohort-complete-layout.png

三块拼在一起回答一条主线：一个照着病人自己数据调出来的神经网络仿真，能不能长出这位病人平时那两条固定的传播路线，而且是"每个触点排在第几位"都对得上。

### topic4_data_driven_snn_cohort_statistics.png（A）

队列层统计。四格各答一个独立问题：谁在队列里（{denominators['primary_canonical_layout']} 人按触点名摆位、其中 {denominators['real_geometry_sensitivity']} 人另有真实三维坐标）；每位患者的模型读数是不是比"同一根杆内部把触点标签打乱"更贴近没看过的那半段病人数据；同一张网络里是不是同时长出两簇可复现的事件（不告诉它病人标签）；把触点摆位换成真实几何后方向是否一致。当前裁定是 `{result['status']}`：{cohort['pass_fraction']:.0%} 的患者赢过自己的打乱对照（门槛 {cohort['pass_fraction_min']:.0%}），中位优势 {primary['median_delta']:+.4f}，p = {primary['wilcoxon_p']:.3g}；同网络双模式出现在 {cohort['same_network_k2_fraction']:.0%} 的患者上。

**关注点**：打乱对照的"备选数"因人而异 —— {len(coarse)} 位患者（{', '.join(coarse) if coarse else '无'}）的电极摆放凑不出足够多的杆内重排，单看这个对照永远够不到 p ≤ 0.05，所以他们的结论必须连同"能达到的最小 p 值"一起读；判据用的是"比自己对照的中位数更好"，不是逐人 p 值。

### topic4_cohort_representative_readout.png（B）

代表患者 `{representative}` 的底物与读数，四列依次是：连续场、模型模式 A 的事件起始分布与平均传播方向、模型模式 B 的同样内容、以及同一张网络里两次相隔数秒的事件在电极上的读数。两次事件画在各自独立的时间窗里、各有自己的本地时钟并标出它在整段仿真中的秒数，中间有断带 —— 不能读成一段连续记录。

**关注点**：触点位置是"只看电极名字"的摆位，**不是这位患者的解剖**；读数是发放密度包络的 30–80 Hz 带通结果，不是临床 SEEG 电压。代表患者是按队列中位表现事先选定的，不是挑最好看的。

### topic4_cohort_representative_kmeans.png（C）

同一位代表患者的聚类核验：不告诉模型任何病人标签，直接对它自己的事件做两类聚类，看这两类是不是分别对上病人的两条路线。四块依次是按簇分组的事件热图、逐触点的名次分布、两簇的平均名次曲线（实线是模型、虚线是病人）、以及模型两类与病人两条路线的相关矩阵。

**关注点**：热图里的灰格表示该事件没有招募这个触点，不是"排名为 0"；三块名次面板共用同一套触点顺序；矩阵报的是汇总后的簇-病人剖面相关，图下方常驻簇大小、跨种子聚类一致性、轮廓系数和落在病人事件云之外的事件比例，这些不利限定符不得从图上移除。

### 科学边界

{config['claim_boundary']}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--expected-commit", default=None)
    parser.add_argument("--skip-panels", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output_root = ROOT / config["output_root"]
    result_path = output_root / "cohort_result.json"
    if not result_path.exists():
        print(json.dumps({"status": "COHORT_RESULT_ABSENT_FIGURE_SKIPPED"}))
        return
    figures = output_root / "figures"
    if not args.skip_panels:
        for _, _, script in PANELS:
            subprocess.run(
                [str(PYTHON), str(Path(__file__).parent / script),
                 "--config", str(args.config)],
                cwd=ROOT, check=True,
            )

    panels = []
    images = []
    for letter, stem, _ in PANELS:
        path = figures / f"{stem}.png"
        if not path.exists():
            raise RuntimeError(f"panel {letter} was not rendered: {path}")
        image = mpimg.imread(path)
        images.append((letter, image))
        panels.append({
            "panel": letter, "png": str(path.relative_to(ROOT)),
            "png_sha256": _sha256(path),
        })

    width = max(image.shape[1] for _, image in images)
    heights = [image.shape[0] * width / image.shape[1] for _, image in images]
    total = sum(heights)
    fig = plt.figure(figsize=(width / 240.0, total / 240.0), facecolor="white")
    top = 1.0
    for (letter, image), height in zip(images, heights):
        fraction = height / total
        ax = fig.add_axes([0.0, top - fraction, 1.0, fraction])
        ax.imshow(image)
        ax.axis("off")
        ax.text(0.004, 0.985, letter, transform=ax.transAxes, ha="left",
                va="top", fontsize=19, fontweight="bold", color="#111111")
        top -= fraction
    stem = figures / "topic4_data_driven_snn_cohort-complete-layout"
    fig.savefig(stem.with_suffix(".png"), dpi=240)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)

    result = json.loads(result_path.read_text())
    (figures / "README.md").write_text(_readme(result, config, panels))
    metadata = {
        "schema_version": "topic4_data_driven_snn_cohort_layout_v1",
        "science_status": {"cohort_status": result["status"],
                           "verdict": result["verdict"]},
        "panels": panels,
        "complete_layout": {
            "png": str(stem.with_suffix(".png").relative_to(ROOT)),
            "png_sha256": _sha256(stem.with_suffix(".png")),
            "pdf": str(stem.with_suffix(".pdf").relative_to(ROOT)),
            "pdf_sha256": _sha256(stem.with_suffix(".pdf")),
        },
        "scientific_boundary": config["claim_boundary"],
    }
    (figures / "topic4_data_driven_snn_cohort-complete-layout_metadata.json"
     ).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], **metadata["complete_layout"]},
                     indent=2))


if __name__ == "__main__":
    main()
