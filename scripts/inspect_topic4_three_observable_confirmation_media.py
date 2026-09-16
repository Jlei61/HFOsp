"""Extract predeclared real GIF frames for the nominated confirmation review.

This is a viewing aid, not a propagation classifier or a replacement for the
patient spectrogram/native GIF. It cannot mark scientific review as passed.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src/snn_engine')]
from scripts import analyze_topic4_three_observable_bo as a


def refresh_static_metadata(nomination, plan):
    """Fix documented colour-limit metadata and label new graph identities."""
    import numpy as np
    import matplotlib.pyplot as plt
    from scripts.paper_figures import plot_topic4_recovery_review as fr
    from scripts.analyze_topic4_core_connectivity_search import load_unit
    from src.topic4_pdf_font_guard import install
    install()
    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 9, 'pdf.fonttype': 3})
    patient = fr.patient_payloads()
    changes = []
    for path in sorted((a.A / 'native_review').glob('*/*/manifest.json')):
        manifest = a.rt.read(path)
        original = a.rt.read(path)
        source = Path(manifest['source'])
        record = a.rt.read(source)
        item = manifest['items'][0]
        with np.load(source.with_suffix('.npz')) as arrays:
            small = {key: arrays[key] for key in ['sheet_activity_frame_ms', 'sheet_activity_counts']}
        fields = [fr.native_timing(record, small, rep['event'])[0] for rep in item['representatives'].values()]
        valid = np.concatenate([field[np.isfinite(field)] for field in fields]) if fields else np.array([0., 250.])
        lower = float(valid.min())
        limits = [lower, max(float(valid.max()), lower + 1.)]
        old_limits = item['native_shared_color_limits_ms']
        item['native_shared_color_limits_ms'] = limits
        cid, topo, noise = path.parent.parent.name, record['job']['topology_seed'], record['job']['dynamics_seed']
        fresh = (cid in nomination['ids'] and topo in plan['confirmation_seeds']['topology']
                 and noise in plan['confirmation_seeds']['dynamics'])
        spectral_refreshed = False
        if fresh:
            r, arrays, ids = load_unit(source, 1500.)
            c = a.rt.read(a.OUT / 'candidates' / f'{cid}.json')
            c['topology'] = topo
            manifest['items'][1] = fr.spectral_comparison(c, {noise: (r, arrays, ids)}, [noise], path.parent, patient, 'primary')
            spectral_refreshed = True
        if old_limits != limits or spectral_refreshed:
            archive = path.parent / 'metadata_archive_20260915'
            archive.mkdir(exist_ok=True)
            previous = archive / 'manifest_before_display_metadata_fix.json'
            if not previous.exists():
                a.rt.write(previous, original)
            manifest['display_metadata_correction'] = {
                'native_time_limits_ms': 'Restored the limits actually used by the native timing panels; envelope amplitude had overwritten the returned metadata variable. Native timing pixels and simulation are unchanged.',
                'spectral_title': 'Explicit topology seed added to fresh confirmation spectral panels; data and example selection unchanged.'}
            a.rt.write(path, manifest)
            changes.append(dict(path=str(path), old_limits_ms=old_limits, corrected_limits_ms=limits,
                                spectral_title_refreshed=spectral_refreshed, archive=str(previous)))
    a.rt.write(a.A / 'confirmation_review/metadata_refresh.json', dict(changes=changes,
        scientific_or_training_change=False, explanation='Display identity and metadata correction only.'))


def extract():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw, ImageFont

    plan = a.rt.read(a.OUT / 'plan.json')
    nomination = a.rt.read(a.OUT / 'nomination.json')
    for cid in nomination['ids']:
        for topology in plan['confirmation_seeds']['topology']:
            for noise in plan['confirmation_seeds']['dynamics']:
                if not (a.A / 'native_review' / cid / f'{topology}_{noise}' / 'manifest.json').exists():
                    raise RuntimeError('Confirmation media production is not complete yet')
    dest = a.A / 'confirmation_review/native_frames'
    dest.mkdir(parents=True, exist_ok=True)
    refresh_static_metadata(nomination, plan)
    offsets = [16, 48, 80, 112, 144, 192, 240]
    # Exact middle-axis geometry used by render_mean_template_gif.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4),
                             gridspec_kw={'width_ratios': [1, 1, 1.1]})
    fig.subplots_adjust(left=.06, right=.985, bottom=.14, top=.82, wspace=.3)
    box = axes[1].get_position().bounds
    plt.close(fig)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    records = []
    for cid in nomination['ids']:
        for topology in plan['confirmation_seeds']['topology']:
            for noise in plan['confirmation_seeds']['dynamics']:
                folder = a.A / 'native_review' / cid / f'{topology}_{noise}'
                source = folder / 'patient_mean_native_multievent.gif'
                metadata = folder / 'patient_mean_native_multievent.json'
                meta = a.rt.read(metadata)
                events = meta['events']
                counts = [len(np.arange(*event['window_ms'], 4)) for event in events]
                starts = np.r_[0, np.cumsum(counts)]
                selected = []
                with Image.open(source) as gif:
                    if gif.n_frames != sum(counts):
                        raise RuntimeError(f'GIF timeline does not match event windows: {source}')
                    bx, by, bw, bh = box
                    size = min(int(bw * gif.width), int(bh * gif.height))
                    left = int((bx + bw / 2) * gif.width - size / 2)
                    top = int((1 - by - bh / 2) * gif.height - size / 2)
                    crop = (left, top, left + size, top + size)
                    tile, header, row_header = 185, 68, 31
                    canvas = Image.new('RGB', (tile * len(offsets), header + len(events) * (tile + row_header)), 'white')
                    draw = ImageDraw.Draw(canvas)
                    draw.text((8, 5), f'{cid} | topology {topology} | noise {noise}', fill='black', font=font)
                    draw.text((8, 27), 'All native E activity; first 3 events per mode; fixed frame offsets from each window.', fill='black', font=font)
                    draw.text((8, 47), 'Colour normalization is per run; compare geometry, not absolute brightness between runs.', fill='black', font=font)
                    for row, event in enumerate(events):
                        lo, hi = event['window_ms']
                        for col, offset in enumerate(offsets):
                            local = int(round(offset / 4))
                            if local >= counts[row]:
                                raise RuntimeError('Requested frame lies outside the primary event window')
                            index = int(starts[row] + local)
                            gif.seek(index)
                            frame = gif.convert('RGB').crop(crop).resize((tile, tile), Image.Resampling.NEAREST)
                            x, y = col * tile, header + row * (tile + row_header)
                            draw.text((x + 3, y + 3), f'{event["mode"]} #{event["event"]} +{offset}ms', fill='black', font=font)
                            canvas.paste(frame, (x, y + row_header))
                            selected.append(dict(event=event['event'], mode=event['mode'], frame=index,
                                                 offset_ms=offset, absolute_ms=lo + 4 * local))
                output = dest / f'{cid}_{topology}_{noise}.png'
                canvas.save(output)
                records.append(dict(candidate=cid, topology=topology, noise=noise,
                                    source=str(source), source_sha256=a.rt.sha(source),
                                    metadata=str(metadata), metadata_sha256=a.rt.sha(metadata),
                                    output=str(output), output_sha256=a.rt.sha(output),
                                    crop_pixels=list(crop), events=events, sampled_frames=selected))
    a.rt.write(dest / 'extraction.json', dict(records=records, selection_offsets_ms=offsets,
        actual_agent_visual_review='PENDING', scientific_acceptance='NOT_PROVIDED_BY_EXTRACTION',
        scope='Actual sampled frames of the unchanged full native GIF; no patient-based event selection.'))
    with (dest / 'README.md').open('w') as f:
        for record in records:
            f.write(f'### {Path(record["output"]).name}\n'
                    '从该完整确认运行的真实原生GIF中，按每模式最早三个事件及固定窗内时点抽取中间活动场，全部E活动均保留。'
                    '每行是一个事件，列是相对该事件观察窗起点的毫秒；白线为core，触点位置沿用原图，原始GIF及帧号见extraction.json。'
                    '**关注点**：逐例比较源区、招募和重叠过程；色标按运行设定，跨运行亮度不作定量比较，抽帧不代替完整读出与患者频谱审阅。\n\n')
    print(dest)


if __name__ == '__main__':
    extract()
