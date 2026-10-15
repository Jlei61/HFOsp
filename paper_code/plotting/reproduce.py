"""Build the copied paper layouts and numerical core-count panel."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import numpy as np
from paper_paths import CODE_ROOT, INPUT_ROOT, sha256


def reproduce_core_count(case, output):
    import matplotlib.pyplot as plt
    from plotting.core_count_curves import draw
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3.5), layout="constrained")
    draw(ax, case, show_title=False, show_legend=True, summary_style="mean_std")
    for extension in ("png", "pdf"):
        fig.savefig(output / f"fig4-core-count-recomputed.{extension}", dpi=200)
    plt.close(fig)
    (output / "README.md").write_text("### fig4-core-count-recomputed.png\n\n从冻结逐次评分重新计算4种core数量、各4次重复的最佳训练目标，再用原绘图函数显示共同前31个epoch的均值与样本标准差。\n\n**关注点**：数值与正式Fig4 C一致；此教学排版不是正式拼版的替代版本。\n")
    return output / "fig4-core-count-recomputed.png"


def reproduce_figures(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new empty directory for the complete figure rebuild")
    output.mkdir(parents=True, exist_ok=True)
    packages = INPUT_ROOT / "results/paper-ready-figure"
    from plotting.fig1 import build_current as f1
    from plotting.fig3 import build_current as f3
    from plotting.fig4 import build_current as f4
    from plotting.vendor.scripts.paper_figures import build_fig2_wider_scores as f2
    report = {}
    f1.rebuild(packages / "fig1", output / "fig1")
    report["fig1"] = json.loads((output / "fig1/rebuild_validation.json").read_text())
    report["fig3"] = f3.rebuild(packages / "fig3", output / "fig3", dpi=240)
    report["fig4"] = f4.restore(packages / "fig4", output / "fig4")
    f2.build(output / "fig2_candidate", dpi=240)
    f2.validate(output / "fig2_candidate")
    report["fig2"] = json.loads((output / "fig2_candidate/validation.json").read_text())
    report["fig2"]["author_acceptance"] = "PENDING"
    (output / "fig2_candidate/figures/README.md").write_text(
        "### fig2-complete-layout.png\n\n重画冻结26人得分与null统计面板，并保留原生A/C/D图层，重建宽B轴的A–D候选。\n\n**关注点**：数值和未修改面板均已核对；此候选仍待作者目视确认。\n")
    subprocess.run([sys.executable, "-m", "plotting.fig5.build_reference_layout", "--output",
                    str(output / "fig5_candidate")], cwd=CODE_ROOT, check=True)
    report["fig5"] = json.loads((output / "fig5_candidate/validation.json").read_text())
    report["fig5"]["author_acceptance"] = "PENDING"
    # Exact video restoration uses copied assets only; all frames are decoded.
    from PIL import Image
    video = packages / "supplementary-video-2.gif"
    with Image.open(video) as frames:
        assert frames.n_frames == 125
        for i in range(frames.n_frames):
            frames.seek(i); frames.load()
    target = output / "video2"; target.mkdir()
    shutil.copy2(video, target / video.name)
    assert sha256(video) == sha256(target / video.name)
    (target / "README.md").write_text(
        "### supplementary-video-2.gif\n\n原样恢复作者定稿的高/低拟合误差模型事件动画，已解码检查125帧。\n\n**关注点**：四个事件各自对齐，不表示同时发生，也不新增独立验证。\n")
    report["video2"] = {"status": "PASS", "frames_decoded": 125, "frozen_export_identical": True}
    (output / "reproduction_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report
