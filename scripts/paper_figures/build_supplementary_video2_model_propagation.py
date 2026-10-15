#!/usr/bin/env python3
"""Join the frozen higher/lower error propagation examples without refitting.

The static companion embeds both original, tuned rows unchanged. The movie
preserves their two envelope panels and replaces the six time snapshots with
the two continuous native fields. All four events retain their frozen windows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Ellipse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "results/paper-ready-figure/archive/2026-10-11_fig4_finalization/fig4"
STILL = ARCHIVE / "candidates/complete_aj_completed_density_d_20261009/figures"
ANIMATION = ARCHIVE / "animations/20261009"
OUT = ROOT / "results/paper-ready-figure/supplementary_model_propagation_comparison/paired_rows_20261015"
LABELS = ("Higher error / Poorer fit", "Lower error / Better fit")
COLORS = ("#77582F", "#255E53")
FONT = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def protected_files():
    package = ROOT / "results/paper-ready-figure/fig4"
    files = [package / "current_version.json", package / "figure4_panel_registry.json",
             ROOT / "docs/manuscript/current_version.json",
             ROOT / "docs/manuscript/versions/author_final_20261015/manuscript.tex"]
    return files + sorted((package / "figures").glob("*"))


def stack_rows(rows):
    """Paste at identical x coordinates, with no resampling of either row."""
    assert rows[0].size == rows[1].size
    w, h = rows[0].size
    margin, header, gap = 32, 82, 34
    out = Image.new("RGB", (w + 2*margin, 2*(h+header)+gap+2*margin), "white")
    draw = ImageDraw.Draw(out)
    boxes = []
    for k, row in enumerate(rows):
        y = margin + k*(h+header+gap)
        draw.rectangle((margin, y, w+margin-1, y+63), fill="#F3F4F4")
        draw.rectangle((margin, y, margin+9, y+63), fill=COLORS[k])
        draw.text((margin+26, y+5), LABELS[k], font=ImageFont.truetype(FONT, 43), fill=COLORS[k])
        pos = (margin, y+header)
        out.paste(row, pos)
        box = (*pos, pos[0]+w, pos[1]+h)
        assert np.array_equal(np.asarray(out.crop(box)), np.asarray(row))
        boxes.append(box)
    return out, boxes


def envelope_rects(panel):
    """Read exact original plotting rectangles; use one uniform SVG->PNG scale."""
    root = ET.parse(STILL / f"fig4-panel{panel}.svg").getroot()
    scale = 320 / 72  # Original frozen exports were 320 dpi.
    rects = []
    for clip in root.findall(".//{http://www.w3.org/2000/svg}clipPath"):
        for r in clip:
            a = {k: float(v) for k, v in r.attrib.items()}
            if np.isclose(a.get("width", 0), 68.031496) and np.isclose(a.get("height", 0), 113.385827):
                rects.append(tuple(a[k]*scale for k in ("x", "y", "width", "height")))
    assert len(rects) == 2
    return sorted(rects)


def movie_row_context(output, panel):
    source = Path(output["source"])
    assert sha(source) == output["source_sha256"]
    assert sha(source.with_suffix(".npz")) == output["arrays_sha256"]
    physics_file = source.parents[1] / "applied_physics.json"
    assert sha(physics_file) == output["applied_physics_sha256"]
    physics = read(physics_file)
    run = read(source)
    assert run["status"] == "COMPLETE"
    with np.load(source.with_suffix(".npz"), allow_pickle=False) as a:
        counts = a["sheet_activity_counts"].copy()
        xy = a["contact_xy_mm"].copy()
        assert float(a["sheet_activity_frame_ms"]) == output["native_frame_ms"] == 2
        for e in output["events"]:
            assert run["events"][e["event"]]["window_ms"] == [e["lo"], e["hi"]]
            assert int(a["event_mode"][e["event"]]) == (1 if e["mode"] == "MTA" else 0)
            assert np.isclose(np.nanmin(a["centroid_ms"][e["event"]]), e["zero"], rtol=0, atol=1e-8)

    # The two original tall envelope panels remain pixel-for-pixel unchanged,
    # except for their moving cursors. The native fields occupy one row beside them.
    original = Image.open(STILL / f"fig4-panel{panel}.png").convert("RGB")
    env = original.crop((0, 0, 1025, original.height))
    width, height = 2430, original.height
    fig = plt.figure(figsize=(width/160, height/160), dpi=160)
    field_rects = [(1130, 78, 504, 504), (1740, 78, 504, 504)]
    images, titles = [], []
    for e, (x,y,w,h) in zip(output["events"], field_rects):
        ax = fig.add_axes([x/width, (height-y-h)/height, w/width, h/height])
        im = ax.imshow(np.zeros_like(counts[0]), origin="lower", extent=[0,20,0,20],
                       cmap="inferno", vmin=0, vmax=75, interpolation="nearest")
        candidate, threshold = physics["candidate"], physics["threshold"]
        a,b = threshold["ellipse_A_semiaxes_mm"]
        ax.add_patch(Ellipse(candidate["centers_mm"][0], 2*a, 2*b,
                             angle=threshold["ellipse_A_angle_deg"], fill=False, color="white", lw=1.1))
        ax.add_patch(Circle(candidate["centers_mm"][1], candidate["radii_mm"][1],
                            fill=False, color="white", lw=1.1))
        ax.scatter(xy[:,0],xy[:,1],facecolors="none",edgecolors="cyan",s=18,lw=.65)
        ax.set(xlim=(0,20),ylim=(0,20),xticks=[0,10,20],yticks=[0,10,20],xlabel="x (mm)")
        ax.tick_params(labelsize=16, width=1, length=4)
        ax.set_xlabel("x (mm)", fontsize=22, labelpad=8)
        if e["mode"] == "MTA":
            fig.text(1054/width, (height-y-h/2)/height, "y (mm)",
                     fontsize=22, rotation=90, ha="center", va="center")
        else:
            ax.set_yticklabels([])
        title = fig.text((x+w/2)/width, 1-35/height, "", ha="center", va="center",
                         fontsize=15, weight="bold", color="#C63D3A" if e["mode"]=="MTA" else "#287FA1")
        images.append(im)
        titles.append(title)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,75),cmap="inferno"),
                      cax=fig.add_axes([2280/width,(height-582)/height,16/width,504/height]),ticks=[0,75])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=17,length=3)
    cb.set_label("E cells / mm² / 2 ms",fontsize=19,labelpad=12)
    return dict(fig=fig, env=env, rects=envelope_rects(panel), images=images,
                titles=titles, counts=counts, events=output["events"])


def movie_row(context, frame):
    cursor_x = []
    for im,title,e,rect in zip(context["images"],context["titles"],context["events"],context["rects"]):
        absolute = e["lo"] + 2*frame
        assert absolute < e["hi"]
        offset = absolute-e["zero"]
        im.set_data(context["counts"][round(absolute/2)])
        title.set_text(f'{e["mode"]}    t = {offset:+.1f} ms')
        x,y,w,h = rect
        cursor_x.append((x+(offset+125)/325*w,y,h))
    fig = context["fig"]
    fig.canvas.draw()
    row = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy())
    row.paste(context["env"],(0,0))
    draw = ImageDraw.Draw(row)
    for x,y,h in cursor_x:
        draw.line((round(x),round(y),round(x),round(y+h)-1),fill="white",width=3)
    if frame in (0,50,124):
        renderer = fig.canvas.get_renderer()
        for t in fig.findobj(matplotlib.text.Text):
            if t.get_visible() and t.get_text():
                b = t.get_window_extent(renderer)
                assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.x1 and b.y1<=fig.bbox.y1, t.get_text()
                assert b.x0>=context["env"].width, f"Envelope crop obscures {t.get_text()}"
    return row


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--static-only", action="store_true")
    args=parser.parse_args()
    before={str(p):sha(p) for p in protected_files() if p.is_file()}
    destination=OUT / "figures"
    destination.mkdir(parents=True,exist_ok=True)
    old=read(ANIMATION / "animation_metadata.json")
    with (ARCHIVE / "source/B_visible_workpoints.csv").open() as f:
        scores={r["candidate"]:r for r in csv.DictReader(f)}
    cases=old["outputs"]
    assert float(scores[cases[0]["candidate"]]["J_joint"])>float(scores[cases[1]["candidate"]]["J_joint"])
    stills=[Image.open(STILL / f"fig4-panel{p}.png").convert("RGB") for p in "ef"]
    joined,boxes=stack_rows(stills)
    joined.save(destination / "model-propagation-higher-lower-error.png")
    joined.save(destination / "model-propagation-higher-lower-error.pdf",resolution=320)
    record=dict(producer=str(Path(__file__).resolve()),producer_sha256=sha(__file__),
                status="PENDING_AUTHOR_VISUAL_REVIEW",static_rows_pixel_identical=True,
                static_row_boxes_px=boxes,static_size_px=joined.size,
                source_metadata=str(ANIMATION / "animation_metadata.json"),
                source_metadata_sha256=sha(ANIMATION / "animation_metadata.json"),
                timing=old["timing"],new_simulations=0,events_reselected=False,
                conditions=[dict(label=LABELS[i],source_candidate=c["candidate"],
                                 J_joint=float(scores[c["candidate"]]["J_joint"]),
                                 events=c["events"],source_sha256=c["source_sha256"],
                                 arrays_sha256=c["arrays_sha256"]) for i,c in enumerate(cases)],
                source_static_rows=[dict(path=str(STILL / f"fig4-panel{p}.png"),
                                        sha256=sha(STILL / f"fig4-panel{p}.png")) for p in "ef"])
    print("Static companion complete: original row pixels preserved",flush=True)
    if not args.static_only:
        contexts=[movie_row_context(c,p) for c,p in zip(cases,"ef")]
        frames=[]
        for k in range(125):
            combined,_=stack_rows([movie_row(c,k) for c in contexts])
            if k==50:
                combined.save(destination / "supplementary-video-2-preview.png")
            if k in (0,50,90,124):
                combined.save(destination / f"validation-frame-{k:03d}.png")
            size=(1600,round(combined.height*1600/combined.width))
            frames.append(combined.resize(size,Image.Resampling.LANCZOS).convert("P",palette=Image.Palette.ADAPTIVE,colors=256))
            if k%25==0:
                print(f"Video frame {k+1}/125",flush=True)
        path=destination / "supplementary-video-2.gif"
        frames[0].save(path,save_all=True,append_images=frames[1:],duration=50,loop=0,disposal=2,optimize=False)
        for context in contexts:
            plt.close(context["fig"])
        with Image.open(path) as gif:
            assert gif.n_frames==125
            for k in range(gif.n_frames):
                gif.seek(k);gif.load()
                assert gif.info["duration"]==50
            record["video"]=dict(path=str(path),sha256=sha(path),size_px=gif.size,
                                 n_frames=125,native_frame_ms=2,playback_fps=20,
                                 duration_s=6.25,all_frames_decoded=True,
                                 envelope_panels="Original static image pixels plus cursors; uniformly resized only for GIF delivery",
                                 layout="One condition per row; MTA envelope, MTB envelope, MTA native field, MTB native field",
                                 field_scale=[0,75],envelope_scale=[0,1])
    assert all(sha(p)==digest for p,digest in before.items())
    record["accepted_figure4_and_manuscript_unchanged"]=True
    record["outputs"]={p.name:sha(p) for p in destination.iterdir() if p.suffix in (".png",".pdf",".gif")}
    (OUT / "metadata.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n")
    descriptions=[
        "### model-propagation-higher-lower-error.png\n高误差／较差拟合在上，低误差／较好拟合在下；直接拼接原 Figure 4 的两条横排图，所有原图像素、图块比例和间距保持。每行左侧为 MTA、MTB 包络，右侧为原 0、30、60 ms 场快照；上下对应列严格对齐。较好／较差是同一训练拟合指标下这两个配置的相对比较。\n**关注点**：醒目条件名替代内部实验编号；等待作者目视确认。\n",
        "### model-propagation-higher-lower-error.pdf\n与 PNG 相同的单页合图，以原 320 dpi 图像嵌入；不是新绘制的矢量图。原图内容与排列未重绘。\n**关注点**：用于查看和排版，PNG 保留无损像素。\n"]
    if not args.static_only:
        descriptions.extend([
            "### supplementary-video-2.gif\n模型间期传播的两个拟合条件上下比较，每个条件一整行：MTA 包络、MTB 包络、MTA 动态场、MTB 动态场。复用原包络图，只增加游标；原静态图的 0、30、60 ms 三列快照在动画中由各自的连续场取代。四个事件均完整回放原 250 ms 窗，每步 2 ms，20 fps、125 帧；按各自窗口起点推进，所示时间各自相对最早有效质心，并非同一绝对时刻。\n**关注点**：色标、通道顺序、事件和模型几何不变；不包含 RNN，未重跑模型；这是新 Video 2 候选，不覆盖旧登记的围发作动画。\n",
            "### supplementary-video-2-preview.png\n合并动画第 51 帧的高分辨率预览。上下条件、包络和原生场与动画一致。\n**关注点**：请检查条件标识与上下对应关系；连续传播见 GIF。\n"])
        for k in (0,50,90,124):
            descriptions.append(f"### validation-frame-{k:03d}.png\n动画第 {k+1} 帧的未缩放校验图。用于核查开头、事件进行中和末尾的游标及时间标签。\n**关注点**：仅作布局校验，不是新事件或新结果。\n")
    (destination / "README.md").write_text("\n".join(descriptions))
    print(OUT,flush=True)


if __name__=="__main__":
    main()
