#!/usr/bin/env python3
"""Restore Figure 1B to the original packed-event spectrum / S**3 contract.

Rebuilds the original 200 s preprocessing segments and all their packed events,
then extracts the three already selected Y1 examples. Display zoom is applied
after computation. No peak-component mask, edge guard or channel-wise shifts.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

import mne
import numpy as np
import scipy
from scipy import signal
from scipy.ndimage import gaussian_filter
from scipy.signal import spectrogram, butter, filtfilt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
LEGACY = ROOT / "ReplayIED/inter_events/yuquan_24h_perPatientAnalysis_dropRef"
ORIGINAL = LEGACY / "for523_p16_packGroupEvents_per2h_showSpecs_bipolar_refine_bool.py"
CANON = ROOT / "results/paper-ready-figure/fig1"
OUT = CANON / "revisions/y1_legacy_spectrum_restored_20261010"
DATA = Path("/mnt/yuquan_data/yuquan_24h_edf/zhangkexuan")
PREVIOUS = CANON / "revisions/y1_a7_zoom_final_20261009"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def original_functions():
    """Execute only named pure functions; never run the legacy batch script."""
    ns = dict(np=np, scipy=scipy, signal=signal, butter=butter, filtfilt=filtfilt,
              spectrogram=spectrogram, gaussian_filter=gaussian_filter,
              resample_to=800, highpass_freqband=[80, 250], specWinLen=.05,
              specFR=[50, 300])
    wanted = [
        (LEGACY / "highEvents_yuquan0910_utils.py", {"notch_filt", "band_filt"}),
        (ORIGINAL, {"return_seg_splitContiHigh", "return_specCenter_packed",
                    "return_massCenterPat", "norm_theSpec_toMaxOne"}),
    ]
    for path, names in wanted:
        nodes = [n for n in ast.parse(path.read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name in names]
        assert {n.name for n in nodes} == names
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


def legacy_spectrum(stitched, borders, fs=800.):
    """Numerical copy of legacy plot_perSeg_specCenter, with both centroids."""
    originals = original_functions()
    specs, centers = [], []
    for row in stitched:
        f, t, magnitude = spectrogram(row, fs, window="hamming",
            nperseg=int(.05*fs), noverlap=int(.8*.05*fs), nfft=int(.05*fs), mode="magnitude")
        # This order and strict endpoints are part of the original definition.
        smooth = gaussian_filter(magnitude, sigma=1.5)
        keep = (f > 50) & (f < 300)
        smooth = smooth[keep]
        specs.append(smooth)
        centers.append(originals["return_specCenter_packed"](smooth, t, borders))
    raw = np.concatenate(specs)
    normalized = originals["norm_theSpec_toMaxOne"](raw, f[keep], t, borders)
    centers = np.asarray(centers)
    reference, _ = originals["return_massCenterPat"](stitched, borders, fs)
    np.testing.assert_array_equal(centers[:, :, 0], reference)
    # Independent implementation already used by the main analysis pipeline.
    from src.group_event_analysis import compute_stitched_spectrogram_centroids_legacy
    analysis = compute_stitched_spectrogram_centroids_legacy(stitched, borders, sfreq=fs)
    np.testing.assert_allclose(centers[:, :, 0], analysis, rtol=1e-13, atol=1e-13)
    return normalized, t, f[keep], centers, raw


def build_source():
    (OUT / "source").mkdir(parents=True, exist_ok=True)
    selection = json.loads((PREVIOUS / "source/selection.json").read_text())
    old = original_functions()
    reports = []
    for example in selection["events"]:
        record = example["record"]
        cache = OUT / f"source/{record}_segment.npz"
        report_path = OUT / f"source/{record}_segment.json"
        if cache.exists() and report_path.exists():
            reports.append(json.loads(report_path.read_text()))
            continue
        edf_path = DATA / f"{record}.edf"
        packs = np.load(DATA / f"{record}_packedTimes.npy")
        target = example["event_index"]
        window = packs[target]
        np.testing.assert_allclose(window, example["window"], atol=1e-10)
        start = np.floor(window[0]/200)*200
        stop = start + 200
        raw = mne.io.read_raw_edf(edf_path, preload=False, encoding="latin1", verbose="ERROR")
        fs = float(raw.info["sfreq"])
        names = [re.sub(r"-(Ref|REF)$", "", re.sub(r"^(EEG|POL)\s+", "", n.strip())).replace(" ", "") for n in raw.ch_names]
        channels = selection["channels"]
        contacts = [f"A{i}" for i in range(3, 11)]
        picks = [names.index(n) for n in contacts]
        bounds = raw.time_as_index([start, min(stop, raw.times[-1])])
        values, segtime = raw[picks, bounds[0]:bounds[1]]
        bipolar = values[:-1]-values[1:]
        stitched, borders, out_fs, indices = old["return_seg_splitContiHigh"](bipolar, segtime, fs, packs)
        assert out_fs == 800
        assert abs(stitched.shape[1]/800 - borders[-1]) < 1e-10
        event = int(np.flatnonzero(indices == target)[0])
        raw.close()
        specs, times, freqs, centers, unnormalized = legacy_spectrum(stitched, borders)
        artifact_path = DATA / f"{record}_lagPat.npz"
        artifact = np.load(artifact_path, allow_pickle=True)
        channel_indices = [artifact["chnNames"].tolist().index(n) for n in channels]
        participation = artifact["eventsBool"][channel_indices, target]
        assert np.all(participation == 1)
        stored = artifact["lagPatRaw"][channel_indices, target]
        reconstructed = centers[:, event, 0]
        # Remove the arbitrary segment-local stitched origin for comparison.
        relative_delta = ((reconstructed-reconstructed.mean())-(stored-stored.mean()))*1000
        report = dict(record=record, event_index=target, event_index_in_segment=event,
            patient="Y1", channels=channels, bipolar_channels=selection["bipolar_channels"],
            edf=str(edf_path), edf_sample_bounds=bounds.tolist(), fs_native=fs,
            fs_processed=800, segment_sec=[float(start),float(stop)],
            event_window_sec=window.tolist(), original_event_center_sec=float(window.mean()),
            packed_times=str(DATA/f"{record}_packedTimes.npy"),
            artifact=str(artifact_path), artifact_sha256=sha(artifact_path),
            segment_event_count=len(indices), all_displayed_channels_participate=True,
            original_function_max_abs_difference_sec=0.,
            stored_lagpat_relative_difference_ms=relative_delta.tolist(),
            stored_lagpat_max_abs_relative_difference_ms=float(abs(relative_delta).max()),
            saved_centroid_span_ms=float(np.ptp(stored)*1000),
            restored_centroid_span_ms=float(np.ptp(reconstructed)*1000))
        np.savez_compressed(cache, signals_stitched_V=stitched, split_borders_sec=borders,
            global_event_indices=indices, specs=specs, times=times, freqs=freqs,
            centers=centers, unnormalized_specs=unnormalized, stored_lagpat=stored,
            channel_names=np.asarray(channels))
        write_json(report_path, report)
        reports.append(report)
        print(record, "n_events", len(indices), "lagpat delta ms", report["stored_lagpat_max_abs_relative_difference_ms"], flush=True)
    contract = dict(patient="Y1", shaft="A", channels=selection["channels"],
        examples=reports, method="original segment preprocessing and stitched spectrum; full-window S**3 mass centroid",
        legacy_source=str(ORIGINAL), legacy_source_sha256=sha(ORIGINAL),
        methods=str(ROOT/"docs/paper-draft/methods_revised_draft.md"),
        fs_hz=800, band_hz=[80,250], spec_window="hamming", spec_window_sec=.05,
        overlap_sec=.04, gaussian_sigma=1.5, smooth_before_frequency_crop=True,
        frequency_mask="50 < f < 300 Hz", centroid_power=3,
        centroid_support="all frequency/time cells within the complete original packed window",
        normalization="each channel/event spectrum divided by its maximum for display only",
        no_peak_component_mask=True, no_edge_guard=True, no_channel_time_shifts=True,
        event_center="original packed event window midpoint; no peak-based recentering")
    write_json(OUT/"spectrum_contract.json", contract)
    return contract


def display_events(contract, *, write_metadata=True):
    """Frozen illustrative choices, checked with the restored full-window rule."""
    report = next(r for r in contract["examples"] if r["record"] == "FA134AX6")
    cache = np.load(OUT/"source/FA134AX6_segment.npz")
    artifact = np.load(report["artifact"], allow_pickle=True)
    packs = np.load(report["packed_times"])
    channel_ids = [artifact["chnNames"].tolist().index(n) for n in contract["channels"]]
    events, validation = [], []
    origin = report["edf_sample_bounds"][0]/report["fs_native"]
    batch_times = origin + np.arange(int(round(200*800)))/800
    for global_index in (1559, 1562, 1574):
        local = int(np.flatnonzero(cache["global_event_indices"] == global_index)[0])
        left = 0. if local == 0 else cache["split_borders_sec"][local-1]
        right = cache["split_borders_sec"][local]
        window = packs[global_index]
        first_sample = batch_times[(batch_times >= window[0]) & (batch_times <= window[1])][0]
        center = left + window.mean() - first_sample
        mask = (cache["times"] > left) & (cache["times"] < right)
        times = cache["times"][mask] - center
        specs = cache["specs"][:, mask].reshape(7, 12, -1)
        centroids = cache["centers"][:, local].copy()
        centroids[:, 0] -= center
        stored = artifact["lagPatRaw"][channel_ids, global_index]
        delta = ((centroids[:, 0]-centroids[:, 0].mean())-(stored-stored.mean()))
        assert np.max(abs(delta)) < 1e-10
        assert np.all(artifact["eventsBool"][channel_ids, global_index] == 1)
        support = times[np.any(specs >= .5, axis=(0, 1))]
        assert np.max(abs(support)) < .15
        assert np.max(abs(centroids[:, 0])) < .15
        events.append(dict(specs=specs, times=times, centers=centroids, freqs=cache["freqs"],
                           time_bounds=[left-center, right-center]))
        validation.append(dict(record="FA134AX6", event_index=global_index,
            original_window_sec=window.tolist(), event_center_sec=float(window.mean()),
            relative_lagpat_max_difference_ms=float(np.max(abs(delta))*1000),
            centroid_span_ms=float(np.ptp(centroids[:, 0])*1000),
            halfmax_time_bounds_sec=[float(support.min()), float(support.max())],
            all_channels_participate=True))
    contract["display_events"] = validation
    contract["display_window_sec"] = [-.15, .15]
    contract["full_window_review_sec"] = [-.25, .25]
    contract["illustration_selection"] = (
        "Same Y1 A3-A9 shaft; 1559, 1562, 1574 from one original 200 s segment. "
        "Chosen after restoring the S**3 rule and inspecting complete 0.5 s windows; "
        "all half-maximum components fit the shared +/-150 ms display. "
        "Other sources from the prior figure were also audited, not used for inference.")
    if write_metadata: write_json(OUT/"spectrum_contract.json", contract)
    return events


def audit_hfo_showcase():
    """Check the separate 178-snippet baseline normalization against its source."""
    src = CANON/"candidates/y1_local_rank_peak_profiles_20261010/source/hfo_showcase.npz"
    z = np.load(src)
    specs = []
    for row in z["snippets_V"]:
        f, t, s = signal.spectrogram(row, fs=1000,
            window=signal.get_window("hann", 180), nperseg=180, nfft=180,
            noverlap=160, mode="magnitude")
        s = gaussian_filter(s[(f >= 0) & (f <= 240)], sigma=1.5)
        specs.append(s)
    mean = np.mean(specs, axis=0)
    raw = np.log(mean)
    norm = mean/np.mean(mean[:, t <= .15], axis=1, keepdims=True)-1
    np.testing.assert_array_equal(raw, z["raw_spec"])
    np.testing.assert_array_equal(norm, z["normalized_spec"])
    return dict(n=178, legacy_source=str(LEGACY/"p16_mechan_events_specComp.py"),
                raw_max_abs_difference=0., normalized_max_abs_difference=0.,
                normalization="mean spectrum / pre-event baseline mean - 1; distinct from per-event max normalization")


def draw_spectrum(fig, events, positions, colorbar, title_baseline, *, large=False, halfwidth=.15):
    import matplotlib.patheffects as pe
    from matplotlib.ticker import FuncFormatter
    width, height = fig.get_size_inches()
    axes = []
    for i, (event, pos) in enumerate(zip(events, positions)):
        ax = fig.add_axes([pos[0]/width, pos[1]/height, pos[2]/width, pos[3]/height])
        times = event["times"]
        edges = np.r_[times[0]-.005, (times[:-1]+times[1:])/2, times[-1]+.005]
        # Preserve A3 -> A9 top-to-bottom, but frequency increases upward within
        # each channel, as in the original spectrum coordinate convention.
        values = event["specs"][:, ::-1, :].reshape(84, -1)
        im = ax.pcolormesh(edges*1000, np.arange(85), values, cmap="coolwarm", vmin=0, vmax=1, rasterized=True)
        for row in range(1, 7): ax.axhline(row*12, color=".75", lw=.4, ls="--")
        ax.axvline(0, color="white", lw=.6, alpha=.75)
        x = event["centers"][:, 0]*1000
        y = np.arange(7)*12 + 12-event["centers"][:, 1]-.5
        line, = ax.plot(x, y, color="#151515", lw=1.45, zorder=6)
        line.set_path_effects([pe.Stroke(linewidth=2.65, foreground="white"), pe.Normal()])
        ax.scatter(x, y, s=18, facecolor="white", edgecolor="#151515", lw=.9, zorder=7)
        ax.set(xlim=(-halfwidth*1000, halfwidth*1000), ylim=(84, 0),
               xticks=[-halfwidth*1000, halfwidth*1000] if large else [-halfwidth*1000, 0, halfwidth*1000],
               yticks=(np.arange(7)+.5)*12)
        ax.set_yticklabels([f"A{j}" for j in range(3,10)] if i == 0 else [])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: "0" if abs(value)<1e-9 else f"{value/1000:.2f}"))
        ax.tick_params(axis="y", length=0, labelsize=15 if large else 8.5, pad=3)
        ax.tick_params(axis="x", labelsize=17.5 if large else 9, pad=3, length=4 if large else 3)
        ax.get_xticklabels()[0].set_ha("left"); ax.get_xticklabels()[-1].set_ha("right")
        ax.spines[["top","right"]].set_visible(False)
        if i == 1: ax.set_xlabel("Time (s)", fontsize=21.25 if large else 12, labelpad=6)
        axes.append(ax)
        np.testing.assert_array_equal(line.get_xdata(), event["centers"][:,0]*1000)
    x,y,w,h = colorbar
    cax=fig.add_axes([x/width,y/height,w/width,h/height]);axes.append(cax)
    cb=fig.colorbar(im,cax=cax,ticks=[0,1]);cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=16.25 if large else 9,length=0,pad=3)
    fig.text(positions[0][0]/width,title_baseline/height,"Yuquan Y1",ha="left",va="baseline",
             fontsize=25 if large else 12,fontweight="bold")
    return axes


def compose_patch(source_root, destination, events, *, latest=False):
    """Keep every original figure object outside B's spectrum unchanged."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import Bbox
    from PIL import Image
    sys.path.insert(0,"/tmp/fig1_pdf_deps")
    from pypdf import PdfReader, PdfWriter, Transformation

    figures=destination/"figures";figures.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42,"axes.unicode_minus":False})
    if latest:
        width,height=19.5,15.9
        positions=[[14.82+i*1.36,11.27,1.28,3.85] for i in range(3)]
        colorbar=[18.98,11.27,.10,3.85];baseline=15.52
        erase=[14.28,10.48,19.28,15.87]
    else:
        width,height=16.,13.8
        positions=[[11.64+i*1.27,9.52,1.20,3.50] for i in range(3)]
        colorbar=[15.47,9.52,.10,3.50];baseline=13.161275
        erase=[11.19,9.04,15.85,13.40]
    fig=plt.figure(figsize=(width,height))
    fig.add_artist(Rectangle((erase[0]/width,erase[1]/height),(erase[2]-erase[0])/width,
        (erase[3]-erase[1])/height,transform=fig.transFigure,facecolor="white",edgecolor="none",zorder=-10))
    axes=draw_spectrum(fig,events,positions,colorbar,baseline,large=latest)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    bounds=Bbox.union([ax.get_tightbbox(renderer) for ax in axes]).transformed(fig.dpi_scale_trans.inverted())
    assert bounds.x0 >= erase[0] and bounds.x1 <= erase[2], bounds
    assert bounds.y0 >= erase[1] and bounds.y1 <= erase[3], bounds
    ticks=[t.get_window_extent(renderer) for ax in axes[:3] for t in ax.get_xticklabels()]
    assert all(a.x1+1 < b.x0 for a,b in zip(ticks,ticks[1:])), [b.bounds for b in ticks]
    overlay_png=destination/"spectrum_overlay.png";overlay_pdf=destination/"spectrum_overlay.pdf"
    fig.savefig(overlay_png,dpi=300,transparent=True)
    fig.savefig(overlay_pdf,dpi=300,transparent=True)

    def merge_pdf(base_path, overlay_path, out_path, translation=None):
        reader=PdfReader(base_path);page=reader.pages[0];patch=PdfReader(overlay_path).pages[0]
        if translation is None: page.merge_page(patch)
        else: page.merge_transformed_page(patch,Transformation().translate(*translation))
        writer=PdfWriter();writer.add_page(page)
        with out_path.open("wb") as f:writer.write(f)

    base=Image.open(source_root/"figures/fig1-complete-layout.png").convert("RGBA")
    patch=Image.open(overlay_png).convert("RGBA")
    assert base.size==patch.size
    result=Image.alpha_composite(base,patch)
    result.convert("RGB").save(figures/"fig1-complete-layout.png")
    changed=np.any(np.asarray(result)!=np.asarray(base),axis=2)
    yy,xx=np.where(changed)
    assert np.all((xx>=erase[0]*300-1)&(xx<=erase[2]*300+1))
    assert np.all((yy>=(height-erase[3])*300-1)&(yy<=(height-erase[1])*300+1))
    merge_pdf(source_root/"figures/fig1-complete-layout.pdf",overlay_pdf,figures/"fig1-complete-layout.pdf")
    for letter in "acdef":
        for suffix in ("png","pdf","svg"):
            src=source_root/f"figures/fig1-panel{letter}.{suffix}"
            if src.exists():shutil.copy2(src,figures/src.name)
    if latest:
        meta=json.loads((source_root/"metadata.json").read_text())
        box=Bbox.union([Bbox.from_extents(*meta["visible_bounds_inches"][k]) for k in ["B_hfo","B_tfr"]]).padded(.04)
        cropped_overlay=destination/"panelb_overlay.png"
        fig.savefig(cropped_overlay,dpi=300,transparent=True,bbox_inches=box)
        old=Image.open(source_root/"figures/fig1-panelb.png").convert("RGBA")
        over=Image.open(cropped_overlay).convert("RGBA")
        assert old.size==over.size,(old.size,over.size)
        Image.alpha_composite(old,over).convert("RGB").save(figures/"fig1-panelb.png")
        merge_pdf(source_root/"figures/fig1-panelb.pdf",overlay_pdf,figures/"fig1-panelb.pdf",(-box.x0*72,-box.y0*72))
    else:
        standalone=plt.figure(figsize=(6.65,4.75))
        standalone.add_artist(Rectangle((2.09/6.65,0),4.56/6.65,1,transform=standalone.transFigure,
                                        facecolor="white",edgecolor="none",zorder=-10))
        draw_spectrum(standalone,events,[[2.54+i*1.27,.55,1.2,3.5] for i in range(3)],
                      [6.37,.55,.10,3.5],4.191275)
        opng=destination/"panelb_overlay.png";opdf=destination/"panelb_overlay.pdf"
        standalone.savefig(opng,dpi=300,transparent=True);standalone.savefig(opdf,dpi=300,transparent=True)
        old=Image.open(source_root/"figures/fig1-panelb.png").convert("RGBA")
        Image.alpha_composite(old,Image.open(opng).convert("RGBA")).convert("RGB").save(figures/"fig1-panelb.png")
        merge_pdf(source_root/"figures/fig1-panelb.pdf",opdf,figures/"fig1-panelb.pdf")
        plt.close(standalone)
    plt.close(fig)
    return dict(source_layout=str(source_root),unchanged_A_C_D_E_F=True,
        changed_full_figure_pixels=int(changed.sum()),changes_restricted_to_right_spectrum=True,
        replacement_bounds_inches=erase,display_time_limits_sec=[-.15,.15],
        title_baseline_inches=baseline)


def full_window_review(events):
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(10,5.5))
    axes=draw_spectrum(fig,events,[[.55+i*3.02,.65,2.85,4.2] for i in range(3)],
                       [9.55,.65,.10,4.2],5.12,halfwidth=.25)
    for ax in axes[:3]:
        for t in (-150,150):ax.axvline(t,color=".25",lw=.7,ls="--")
    fig.savefig(OUT/"figures/fig1-spectrum-full-window-check.png",dpi=220,facecolor="white")
    fig.savefig(OUT/"figures/fig1-spectrum-full-window-check.pdf",facecolor="white")
    plt.close(fig)


def write_delivery(contract, audit):
    pointer_path=CANON/"current_revision.json"
    current_pointer=json.loads(pointer_path.read_text())
    shutil.copytree(PREVIOUS/"source/panel_a_snapshot",OUT/"source/panel_a_snapshot",dirs_exist_ok=True)
    # Keep the previous author's accepted A status and layout review states.
    for dest,base,key in [(OUT,PREVIOUS,"current_layout"),
                          (OUT/"latest_layout",CANON/"candidates/y1_local_rank_peak_profiles_20261010","latest_layout")]:
        metadata=json.loads((base/"metadata.json").read_text())
        metadata.update(producer=str(Path(__file__).resolve()),
            status="LEGACY_SPECTRUM_RESTORED_PENDING_VISUAL_REVIEW",
            source_revision=str(base),spectrum_contract=str(OUT/"spectrum_contract.json"),
            panel_b=dict(patient="Y1",channels=contract["channels"],events=contract["display_events"],
                method="full original event S**3 weighted centroid",spectrum_fs_hz=800,
                no_peak_component_mask=True,display_time_limits_sec=[-.15,.15],
                normalization="per-channel/event maximum for display; S**3 for centroid",
                left_showcase_unchanged=True),
            human_visual_acceptance="PENDING_COMPLETE_LAYOUT_REVIEW",
            panel_a_human_visual_acceptance="ACCEPTED",
            changed_panels=["B right spectrogram"],preservation=audit[key],
            validation=str(OUT/"validation.json"),
            input_hashes={str(p):sha(p) for p in [Path(__file__),ORIGINAL,
                LEGACY/"highEvents_yuquan0910_utils.py",base/"metadata.json"]})
        for obsolete in ["panel_b_f_pixels_unchanged","complete_figure_matches_trial_assembly"]:
            metadata.pop(obsolete,None)
        if "summaries" in metadata: metadata["summaries"]["B"]=metadata["panel_b"]
        metadata["outputs"]={str(p.relative_to(dest)):sha(p) for p in (dest/"figures").glob("*") if p.suffix in (".png",".pdf",".svg")}
        write_json(dest/"metadata.json",metadata)
        descriptions={
            "fig1-panela.png / .pdf":"采用该布局来源中的Y1 A7/A9脑图与波形，文件逐字节保留。彩色中点表示相邻双极通道，波形示意不参与本次质心计算。\n\n**关注点**：脑朝向、引线和作者已接受的几何对应不变。",
            "fig1-panelb.png / .pdf":"右侧恢复原始800 Hz处理链、50 ms Hamming窗与40 ms重叠，先平滑完整谱再截50<f<300 Hz。中心为完整事件S³加权质心，显示为S/max(S)；采用Y1同一A杆A3–A9的1559/1562/1574三个实例，显示窗统一为±150 ms。左侧178段HFO谱与老脚本数值完全一致。\n\n**关注点**：多峰时中心允许位于两峰之间；不能改成峰顶或70%峰团。",
            "fig1-panelc.png / .pdf":"逐字节保留来源布局的Y1原序热图与rank分布。全部18,190事件及参与掩码不变。\n\n**关注点**：18通道布局为显示内rank，26通道布局为原rank，不能混作重新聚类。",
            "fig1-paneld.png / .pdf":"逐字节保留来源布局的permutation机制示意及40人MI统计。Null点、括号、统计值和坐标均未改动。\n\n**关注点**：本次不重新计算MI或其null。",
            "fig1-panele.png / .pdf":"逐字节保留来源布局中Y1的TA/TB热图及均值±总体标准差。冻结分组仍为13,160和5,030个事件。\n\n**关注点**：与C共享相同事件全集及显示通道。",
            "fig1-panelf.png / .pdf":"逐字节保留来源布局的40人overall/within-template MI与single/multi配对inset。字体和坐标排版保持。\n\n**关注点**：谱图修复不改变队列结果。",
            "fig1-complete-layout.png / .pdf":"仅替换完整拼版中B右侧谱图区域，其他区域PNG逐像素不变；PDF保留原矢量图层并叠加新谱图。标题和时间轴继续统一。\n\n**关注点**：本次恢复的是计算方法，完整拼版待作者目视检查。",
        }
        if dest==OUT:
            descriptions["fig1-spectrum-full-window-check.png / .pdf"]="显示同一三个事件的完整500 ms谱图及S³质心。虚线仅标出主图±150 ms显示范围，计算始终覆盖完整事件。\n\n**关注点**：主图zoom没有重新计算或裁剪质心支持域，半峰高以上谱成分均保留。"
        lines=["# Figure 1：恢复原始谱图与质心方法", "", "来源布局：`"+str(base)+"`。算法核对见上级spectrum_contract.json和method_consistency_audit.md。", ""]
        for filename,description in descriptions.items():lines += ["### "+filename,"",description,""]
        (dest/"figures/README.md").write_text("\n".join(lines))
    method_report="""# Figure 1 方法一致性核对（2026-10-10）

## 已恢复并验证

- B右侧直接复用ReplayIED原始预处理函数：按EDF原200 s分段、相邻双极、800 Hz重采样、50–250 Hz谐波IIR陷波Q=30、3阶80–250 Hz Butterworth及filtfilt。先处理整段，再截取该段全部既有packed events并拼接，未单独过滤展示用短窗。
- Hamming 50 ms / overlap 40 ms / nfft等于窗长，幅度谱先Gaussian σ=1.5平滑再保留50<f<300 Hz。显示每通道每事件S/max(S)，中心使用完整事件S³/ΣS³；无70%阈值、连通区选择、边缘排除或逐通道移动。
- 同一输入下，新谱与中心逐项匹配原函数；亦通过src.group_event_analysis.compute_stitched_spectrogram_centroids_legacy独立实现验证。展示事件对已存lagPatRaw的相对质心最大偏差小于1e-10 ms，属于浮点舍入。
- 左侧178段HFO的原始平均谱和基线归一化谱，与p16_mechan_events_specComp.py数值逐项完全相同。它的1000 Hz、180 ms Hann窗及先截频段后平滑是该独立展示的原有定义，不能被右侧群体事件算法覆盖。
- A、C、D、E、F的独立PNG/PDF逐字节保留。完整图修改区域仅B右侧；C/E全部18,190事件、冻结TA/TB的13,160/5,030标签及D/F 40人统计未改动。

## Methods中仍需澄清的一处文字

methods_revised_draft.md第21行把线噪处理统称为FIR；实际原始检测脚本有FIR分支，但本次复现并与lagPatRaw完全对齐的质心支路调用highEvents_yuquan0910_utils.py的IIR notch + Butterworth/filtfilt。因此不能声称这一句对所有处理支路都准确。此次未改写Methods或改变原分析来迁就该文字；应在稿件中区分检测与质心计算支路。

## 显示及解释边界

主图±150 ms只是显示范围；另提供完整±250 ms窗口。三例按恢复后的方法检查并选取，仅作展示，不估计发生率或队列传播强度。质心是整段时频分布的代表时间，多峰时可能处于两峰之间，不是精确生物学起始时刻。

最新18通道C/E布局沿用作者另一轮的显示内重排名次；原始26通道rank和聚类未改写。当前修复同时提供26通道正式布局及18通道最新布局，未把后者自动宣称为人工验收通过。
"""
    (OUT/"method_consistency_audit.md").write_text(method_report)
    shutil.copy2(Path(__file__),OUT/"source/restore_fig1_legacy_spectrum.py")
    new_pointer=dict(current_pointer,revision_root=str(OUT),
        status="LEGACY_SPECTRUM_RESTORED_PENDING_VISUAL_REVIEW",
        complete_figure=str(OUT/"figures/fig1-complete-layout.pdf"),
        complete_figure_png=str(OUT/"figures/fig1-complete-layout.png"),
        producer=str(Path(__file__).resolve()),metadata=str(OUT/"metadata.json"),
        previous_revision_retained=str(PREVIOUS),
        panel_b_method="original_full_window_S3_centroid",
        latest_layout_revision=dict(root=str(OUT/"latest_layout"),
            complete_figure=str(OUT/"latest_layout/figures/fig1-complete-layout.pdf"),
            status="PENDING_AUTHOR_VISUAL_REVIEW",source_layout="y1_local_rank_peak_profiles_20261010"))
    write_json(pointer_path,new_pointer)


def main():
    contract=build_source()
    for example in contract["examples"]:
        z=np.load(OUT/f"source/{example['record']}_segment.npz")
        result=legacy_spectrum(z["signals_stitched_V"],z["split_borders_sec"])
        for key,value in zip(("specs","times","freqs","centers","unnormalized_specs"),result):
            np.testing.assert_array_equal(z[key],value)
    events=display_events(contract)
    audit=dict(status="PASS",spectrum_original_functions_exact=True,
               spectrum_analysis_implementation_agrees=True,
               stored_lagpat_max_difference_ms=max(e["relative_lagpat_max_difference_ms"] for e in contract["display_events"]),
               hfo_showcase=audit_hfo_showcase(),human_visual_acceptance="PENDING")
    audit["current_layout"]=compose_patch(PREVIOUS,OUT,events)
    latest=CANON/"candidates/y1_local_rank_peak_profiles_20261010"
    audit["latest_layout"]=compose_patch(latest,OUT/"latest_layout",events,latest=True)
    full_window_review(events)
    write_json(OUT/"validation.json",audit)
    write_delivery(contract,audit)
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    main()
