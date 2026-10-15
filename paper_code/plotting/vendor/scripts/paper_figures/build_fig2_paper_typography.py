"""Paper-size typography and one vertical, in-axis Figure 2B legend.

All scientific inputs and all trajectories are frozen from the preceding
reviewed candidate. Only native graphics placement and text are changed.
"""
from __future__ import annotations
from paper_paths import CODE_ROOT, INPUT_ROOT, OUTPUT_ROOT
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
ROOT = INPUT_ROOT
pass
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator
import numpy as np
import pymupdf as pdf
from plotting.vendor.scripts.paper_figures import build_fig2_refined_layout as base
from plotting.vendor.scripts.paper_figures.build_fig2_sequence_field_layout import export, place
from plotting.vendor.scripts.paper_figures.plot_fig2_exemplar_trajectory import mpl_path, TEMPLATE_COLORS
from plotting.vendor.scripts.plot_topic5_interictal_template_ab_fields import build_interictal_ab_panel_payloads, draw_interictal_rank_field_panel
SOURCE = base.OUT
OUT = SOURCE.parent / 'paper_typography_inset_legends_20261012'
REGULAR = str(INPUT_ROOT / 'fonts' / 'LiberationSans-Regular.ttf')
BOLD = str(INPUT_ROOT / 'fonts' / 'LiberationSans-Bold.ttf')
FONT_FAMILY = 'Liberation Sans'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def export_pdf_png(document, stem, dpi):
    document.save(stem.with_suffix('.pdf'), garbage=4, deflate=True)
    document[0].get_pixmap(dpi=dpi, alpha=False).save(stem.with_suffix('.png'))

def strip_text(document):
    page = document[0]
    spans = []
    for b in page.get_text('dict', flags=pdf.TEXTFLAGS_TEXT)['blocks']:
        for line in b.get('lines', []):
            for span in line['spans']:
                spans.append(span)
                rect = pdf.Rect(span['bbox'])
                if line['dir'] == (1.0, 0.0):
                    middle = (rect.y0 + rect.y1) / 2
                    rect.y0 = middle - 0.15
                    rect.y1 = middle + 0.15
                else:
                    middle = (rect.x0 + rect.x1) / 2
                    rect.x0 = middle - 0.15
                    rect.x1 = middle + 0.15
                page.add_redact_annot(rect, fill=None)
    page.apply_redactions(images=0, graphics=0, text=0)
    return spans

def text_drawer(page):
    fonts = {}
    for name, path in [('PaperText', REGULAR), ('PaperBold', BOLD)]:
        page.insert_font(fontname=name, fontfile=path)
        fonts[name] = pdf.Font(fontfile=path)

    def write(text, x, y, size=14, align='center', bold=False, color=(0, 0, 0), rotate=0):
        key = 'PaperBold' if bold else 'PaperText'
        length = fonts[key].text_length(text, fontsize=size)
        if rotate:
            y += length / 2
        else:
            x -= length * {'left': 0, 'center': 0.5, 'right': 1}[align]
        page.insert_text((x, y), text, fontname=key, fontsize=size, color=color, rotate=rotate)
        return length
    return write

def build_a(out, dpi):
    metadata = json.loads((SOURCE / 'panela_revision/trajectory_metadata.json').read_text())
    oldlayout = metadata['representation_labels']
    oldpos = oldlayout['placements']
    with pdf.open(SOURCE / 'figures/fig2-panela.pdf') as old:
        spans = strip_text(old)
        doc = pdf.open()
        page = doc.new_page(width=700, height=430)
        write = text_drawer(page)
        placements = {}
        axes = {}

        def copy(name, src, dst):
            page.show_pdf_page(pdf.Rect(dst), old, 0, clip=pdf.Rect(src), keep_proportion=False)
            placements[name] = dict(source_rect_pt=list(src), target_rect_pt=list(dst))
        for row, template in enumerate(('TA', 'TB')):
            top = 34 + 200 * row
            side = 150.0
            for kind, x, title in [('sequence', 52.0, f'{template} Rank SEQ'), ('field', 485.0, f'{template} Spatial Trajectory')]:
                key = f'{kind}_{template.lower()}'
                source = pdf.Rect(oldpos[key]['data_axes_rect_pt'])
                target = pdf.Rect(x, top, x + side, top + side)
                scale = side / source.width
                copy(key, source + (-0.8, -0.8, 0.8, 0.8), target + (-0.8 * scale, -0.8 * scale, 0.8 * scale, 0.8 * scale))
                axes[key] = list(target)
                write(title, x + side / 2, top - 11, size=16, bold=True, color=matplotlib.colors.to_rgb(TEMPLATE_COLORS[template]))
                oldtop = source.y0
                candidates = [s for s in spans if s['text'] in ('0', '100', '−20', '20') and abs(s['origin'][1] - (oldtop + 132 + 14)) < 0.1 and (source.x0 - 15 < (s['bbox'][0] + s['bbox'][2]) / 2 < source.x1 + 15)]
                for s in candidates:
                    cx = (s['bbox'][0] + s['bbox'][2]) / 2
                    xx = x + (cx - source.x0) * scale
                    page.draw_line((xx, top + side), (xx, top + side + 3.5), width=0.8)
                    write(s['text'], xx, top + side + 18, size=14)
                if kind == 'sequence':
                    labels = [s['text'] for s in spans if s['text'].startswith(('ICL', 'SCL')) and s['bbox'][0] < 50 and (oldtop - 4 < s['origin'][1] < oldtop + 136)]
                    for i, name in enumerate(labels):
                        write(name, x - 7, top + (i + 0.5) * side / len(labels) + 3.8, size=12, align='right')
                    write('Normalized magnitude', x + side + 51, top + side / 2, size=14, rotate=90)
                else:
                    for s in spans:
                        if s['text'] in ('0', '20') and 429 < s['bbox'][0] < 444 and (oldtop < s['origin'][1] < oldtop + 132):
                            yy = top + (s['origin'][1] - 3.2 - oldtop) * scale
                            page.draw_line((x - 3.5, yy), (x, yy), width=0.8)
                            write(s['text'], x - 8, yy + 4.4, size=14, align='right')
                    write('y (mm)', x - 37, top + side / 2, size=17, rotate=90)
                source_bar = pdf.Rect(oldpos[key + '_colorbar']['placed_rect_pt'])
                bar = pdf.Rect(x + side + 9, top, x + side + 14, top + side)
                copy(key + '_bar', source_bar, bar)
                for value in [0, 1] if kind == 'sequence' else [0, 0.5, 1]:
                    yy = top + (1 - value) * side
                    page.draw_line((bar.x1, yy), (bar.x1 + 2.5, yy), width=0.7)
                    label = str(value) if kind == 'sequence' else {0: '0 early', 0.5: '0.5', 1: '1 late'}[value]
                    write(label, bar.x1 + 5, yy + 4.2, size=13, align='left')
                if kind == 'field':
                    write('ranks', bar.x0, top - 10, size=14, align='left')
        anatomy_old = pdf.Rect(oldpos['anatomy']['placed_rect_pt'])
        anatomy_new = pdf.Rect(273, 35, 428, 35 + anatomy_old.height * 155 / anatomy_old.width)
        with pdf.open(SOURCE / 'panela_revision/figures/E10-embedded-analysis-plane.pdf') as anatomy:
            page.show_pdf_page(anatomy_new, anatomy, 0)
        placements['anatomy'] = dict(source_rect_pt=list(anatomy_old), target_rect_pt=list(anatomy_new), frozen_MRI_reused=True)
        scale = anatomy_new.width / anatomy_old.width
        corners = (np.array(oldpos['anatomy']['plane_corners_pt']) - [anatomy_old.x0, anatomy_old.y0]) * scale + [anatomy_new.x0, anatomy_new.y0]
        frame_old = pdf.Rect(oldlayout['projection_frame']['rect_pt'])
        frame = pdf.Rect(272, 266, 422, 266 + frame_old.height * 150 / frame_old.width)
        copy('projection', frame_old + (-0.6, -0.6, 0.6, 0.6), frame + (-0.7, -0.7, 0.7, 0.7))
        guides = []
        for start, end in zip(corners[:2], [[frame.x0, frame.y0], [frame.x1, frame.y0]]):
            page.draw_line(start.tolist(), end, color=(0.55, 0.67, 0.7), width=0.7, dashes='[2.5 2] 0')
            guides.append([start.tolist(), end])
        write('Early Projection', 347, 253, size=13, color=(0.25, 0.43, 0.48))
        for s in spans:
            if frame_old.contains(pdf.Rect(s['bbox'])):
                point = (np.array(s['origin']) - [frame_old.x0, frame_old.y0]) * 150 / frame_old.width + [frame.x0, frame.y0]
                if s['text'] in ('σ', ' = 6 mm'):
                    continue
                if s['text'] in ('ICL', 'SCL'):
                    point[0] -= 3
                write(s['text'], *point, size=11.5, align='left', color=(0.25, 0.36, 0.39))
        write('σ = 6 mm', frame.x0 + 78, frame.y1 - 28, size=11.5, align='left', color=(0.11, 0.36, 0.4))
        arrow = np.array([[431, 212.2], [463, 212.2], [463, 208], [472, 213], [463, 218], [463, 213.8], [431, 213.8]])
        page.draw_polyline(arrow.tolist(), color=None, fill=(0.42, 0.49, 0.51), closePath=True)
        write('Time (ms)', 127, 420, size=17)
        write('shared TA axis (mm)', 560, 420, size=16)
        export_pdf_png(doc, out / 'figures/fig2-panela', dpi)
        doc.close()
    return dict(source=str(SOURCE / 'figures/fig2-panela.pdf'), axes_rect_pt=axes, placements=placements, projection_guides_pt=guides, conversion_arrow_pt=arrow.tolist(), numerical_content_unchanged=True, native_font_pt=dict(title=16, axis=17, ticks=14, dense_contacts=12, colorbar_ticks=13))

def build_b(out, dpi):
    snap = SOURCE / 'source_snapshot'
    rows = base.scores._load_rows(snap / 'B_scores.csv')
    summary = json.loads((snap / 'B_summary.json').read_text())['cohort_statistics']
    oldmeta = json.loads((snap / 'panelb_source_metadata.json').read_text())
    examples = {e['subject_id']: e['display_label'] for e in oldmeta['rose_examples']}
    with np.load(snap / 'B_null.npz') as cached:
        null = np.median(cached['patient_null_timing_plus_space_score'], axis=0)
    width, height = (520.0, 430.0)
    with plt.rc_context({'font.family': FONT_FAMILY, 'font.size': 13, 'pdf.fonttype': 42, 'axes.linewidth': 0.8}):
        fig = plt.figure(figsize=(width / 72, height / 72), facecolor='white')
        ax = fig.add_axes([260 / width, 45 / height, 247 / width, 363 / height])
        base.scores.draw_absolute_scores_and_null(ax, rows, summary, null, example_labels=examples, bootstrap_in_source_order=True)

        def forward(y):
            y = np.asarray(y)
            return np.where(y <= 2.4, (y + 1.65) * 102 / 4.05, np.where(y <= 30, 102 + (y - 2.4) * 157 / 27.6, 259 + (y - 30) * 104 / 16))

        def backward(y):
            y = np.asarray(y)
            return np.where(y <= 102, y * 4.05 / 102 - 1.65, np.where(y <= 259, (y - 102) * 27.6 / 157 + 2.4, (y - 259) * 16 / 104 + 30))
        ax.set_yscale('function', functions=(forward, backward))
        ax.set_ylim(-1.65, 46)
        ax.set_yticks([])
        for t in list(ax.texts):
            if t.get_text() in ['Direction-\nshuffle null', 'Timing-only', 'Timing + 3D\ndirection'] or 'higher with' in t.get_text():
                t.remove()
            else:
                t.set_fontsize(13 if t.get_text() in examples.values() else 15)
        ax.text(0.975, 0.973, '22/26 higher\nwith 3D', ha='right', va='top', transform=ax.transAxes, fontsize=11)
        ax.set_xlim(-0.1, 0.98)
        ax.set_xticks([0, 0.3, 0.6, 0.9])
        ax.tick_params(axis='x', labelsize=14, pad=3, length=3)
        ax.set_xlabel('Held-out direction score', fontsize=17, labelpad=9)
        ax.set_ylabel('Patients (ordered)', fontsize=16)
        ax.yaxis.set_label_coords(-0.095, 0.495)
        handles = [Patch(facecolor=(0.55, 0.55, 0.55, 0.24), edgecolor='#6F6F6F'), (Line2D([], [], color=TEMPLATE_COLORS['TA'], lw=2), Line2D([], [], color=TEMPLATE_COLORS['TB'], lw=2)), Line2D([], [], color='#202020', ls=(0, (3, 2)), lw=1.5, marker='o', mfc='white', mec=base.scores.TEMPORAL_COLOR, ms=4), Line2D([], [], color='#202020', lw=1.5, marker='o', mfc=base.scores.HYBRID_COLOR, mec='white', ms=4), Patch(facecolor='#BDBDBD', edgecolor='#777777')]
        labels = ['Event directions', 'TA / TB', 'Timing only', 'Timing + 3D', 'Shuffle null']
        legend = ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.025, 0.985), ncol=1, frameon=True, facecolor='white', edgecolor='none', framealpha=1, fancybox=False, borderpad=0.25, fontsize=13, handlelength=1.6, labelspacing=0.36, borderaxespad=0, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.25)})
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        lb = legend.get_window_extent(renderer)
        ab = ax.get_window_extent(renderer)
        assert ab.contains(lb.x0, lb.y0) and ab.contains(lb.x1, lb.y1)
        covered_points = 0
        for collection in ax.collections:
            if isinstance(collection, matplotlib.collections.PathCollection):
                screen = collection.get_offset_transform().transform(collection.get_offsets())
                covered_points += sum((lb.contains(*point) for point in screen))
        assert covered_points == 0, 'Legend must not cover observed patient points'
        legend_bounds = lb.transformed(fig.dpi_scale_trans.inverted()).bounds
        fig.savefig(out / 'source_snapshot/B-score-native.pdf', facecolor='white')
        plt.close(fig)
    with pdf.open(out / 'source_snapshot/B-score-native.pdf') as doc, pdf.open(snap / 'original_panelb.pdf') as roses:
        rose_spans = strip_text(roses)
        page = doc[0]
        write = text_drawer(page)
        placements = []
        for index, (source_y, cy, name) in enumerate([(118.3538, 115, 'E10'), (258.3666, 300, 'E14')]):
            src = pdf.Rect(56.2419, source_y - 57.7674, 171.7767, source_y + 57.7674)
            target = pdf.Rect(50, cy - 75, 200, cy + 75)
            placements.append(place(page, roses, target, src + (-0.5, -0.5, 0.5, 0.5)))
            write(name, 15, cy - 66, size=17, bold=True, align='left')
            for text, x, y, align in [('90°', 125, cy - (83 if index == 0 else 80), 'center'), ('270°', 125, cy + (88 if index == 0 else 93), 'center'), ('180°', 43, cy + 4, 'right'), ('0°', 208, cy + 4, 'left')]:
                write(text, x, y, size=12, align=align)
            scale = 150 / (src.width + 1)
            for s in rose_spans:
                if s['text'] in ('10%', '20%') and source_y - 58 < s['origin'][1] < source_y + 58:
                    xy = (np.array(s['origin']) - [src.x0 - 0.5, src.y0 - 0.5]) * scale + [50, cy - 75]
                    write(s['text'], *xy, size=12, align='left')
        export(doc, out / 'figures/fig2-panelb', dpi)
    return dict(n_score_axes=1, separate_distribution_axis=False, legend_count=1, legend_columns=1, legend_inside_axis=True, legend_labels=labels, legend_bounds_inches=list(legend_bounds), legend_overlapping_patient_points=int(covered_points), cohort_statistics=summary, rose_frozen_graphics_reused=True, rose_placements=placements, bootstrap_in_original_CSV_order=True, native_font_pt=dict(axis=17, ticks=14, legend=13, identity=17))

def build_c(out, source_meta, dpi):
    width, height = (710.0, 350.0)
    records = []
    with plt.rc_context({'font.family': FONT_FAMILY, 'font.size': 12, 'pdf.fonttype': 42, 'axes.linewidth': 0.8}):
        fig = plt.figure(figsize=(width / 72, height / 72), facecolor='white')
        for col, old in enumerate(source_meta['panel_c']):
            name = old['display_id']
            record = json.loads((SOURCE / f'source_snapshot/{name}_fields.json').read_text())
            a, b, mode = build_interictal_ab_panel_payloads(record, display_sigma_mm=old['display_sigma_mm'])
            assert mode == 'shared'
            entry = json.loads(json.dumps(old))
            entry['axes_rect_pt'] = {}
            for row, (template, payload) in enumerate(zip(('TA', 'TB'), (a, b))):
                payload['frame'].update(xlim=old['xlim_mm'], ylim=old['ylim_mm'])
                left, top, side = (80 + 151 * col, 28 + 162 * row, 120)
                ax = fig.add_axes([left / width, 1 - (top + side) / height, side / width, side / height])
                draw_interictal_rank_field_panel(ax, payload, template, compact=True, contact_size=20, contact_outline_lw=0.75, show_template_tag=False)
                ax.set_title(name if row == 0 else '', fontsize=16, fontweight='bold', pad=5)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
                ax.tick_params(axis='both', labelsize=12, length=3, pad=2)
                if col == 0:
                    ax.set_ylabel('y (mm)', fontsize=15, labelpad=3)
                    fig.text(14 / width, 1 - (top + side / 2) / height, f'{template} trajectory', rotation=90, fontsize=14, fontweight='bold', color=TEMPLATE_COLORS[template], ha='center', va='center')
                for item in old['trajectories'][template]:
                    mpl_path(ax, np.array(item['displayed_points_mm']), TEMPLATE_COLORS[template], scale=0.85)
                for collection in ax.collections:
                    collection.set_zorder(9)
                entry['axes_rect_pt'][template] = [left, top, left + side, top + side]
            records.append(entry)
        cax = fig.add_axes([670 / width, 40 / height, 6 / width, 282 / height])
        cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap='viridis'), cax=cax, ticks=[0, 1])
        cb.ax.set_yticklabels(['early', 'late'])
        cb.ax.tick_params(labelsize=12, length=2, pad=2)
        cb.ax.set_title('rank', fontsize=13, pad=5)
        cb.outline.set_linewidth(0.8)
        fig.text(375 / width, 7 / height, 'Shared TA axis (mm)', ha='center', va='bottom', fontsize=15)
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(out / f'figures/fig2-panelc.{ext}', dpi=dpi, facecolor='white')
        plt.close(fig)
    return records

def build(out, dpi):
    for sub in ('figures', 'source_snapshot', 'trajectories'):
        (out / sub).mkdir(parents=True, exist_ok=True)
    source_meta = json.loads((SOURCE / 'figure2_layout_metadata.json').read_text())
    for path in (SOURCE / 'trajectories').glob('*.csv'):
        shutil.copy2(path, out / 'trajectories' / path.name)
    for name in ('figure2_layout_metadata.json', 'validation.json'):
        shutil.copy2(SOURCE / name, out / 'source_snapshot' / ('previous_' + name))
    ameta = build_a(out, dpi)
    bmeta = build_b(out, dpi)
    cmeta = build_c(out, source_meta, dpi)
    for ext in ('png', 'pdf', 'svg'):
        shutil.copy2(SOURCE / f'figures/fig2-paneld.{ext}', out / f'figures/fig2-paneld.{ext}')
    doc = pdf.open()
    page = doc.new_page(width=1240, height=860)
    parts = {k: pdf.open(out / f'figures/fig2-panel{k}.pdf') for k in 'abcd'}
    layout = {'a': place(page, parts['a'], (22, 44, 704, 446), parts['a'][0].rect), 'b': place(page, parts['b'], (715, 44, 1222, 446), parts['b'][0].rect), 'c': place(page, parts['c'], (24, 452, 816, 842), parts['c'][0].rect), 'd': place(page, parts['d'], (843, 461, 1216, 838))}
    for label, x, y in [('A', 12, 32), ('B', 702, 32), ('C', 12, 443), ('D', 824, 443)]:
        page.insert_text((x, y), label, fontsize=27, fontname='hebo')
    export_pdf_png(doc, out / 'figures/fig2-complete-layout', dpi)
    page.get_pixmap(matrix=pdf.Matrix(1800 / 1240, 1800 / 1240), alpha=False).save(out / 'figures/fig2-complete-layout-preview.png')
    for part in parts.values():
        part.close()
    doc.close()
    meta = dict(schema_version='fig2_paper_typography_inset_legends_v1', human_visual_acceptance='PENDING', source_candidate=str(SOURCE), producer=str(Path(__file__).relative_to(CODE_ROOT)), examples=source_meta['examples'], canvas_pt=[1240, 860], complete_layout=layout, panel_a=ameta, panel_b=bmeta, panel_c=cmeta, panel_d_frozen=True, scientific_data_and_trajectories_unchanged=True, display_bandwidth_mm=source_meta['display_bandwidth_mm'], typography_reference='Accepted Figures 1, 3 and 4 PDFs measured at the same 1240 pt display width; Fig3 ticks 12.8 pt and axis labels 15.6 pt.', font_family=FONT_FAMILY, legend_policy='one in-axis vertical legend in B; field panels retain their quantitative colorbars', agent_visual_review='PENDING')
    (out / 'figure2_layout_metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    shutil.copy2(Path(__file__), out / 'source_snapshot' / Path(__file__).name)
    return meta
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()
    build(args.output.resolve(), args.dpi)
    print(args.output)
