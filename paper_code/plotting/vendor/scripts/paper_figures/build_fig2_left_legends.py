"""Figure 2B: compact left legends and full-height patient comparison.

The preceding candidate's A/C/D exports and all scientific data are frozen.
Only the four timing-only rose rays are hidden; score estimates are unchanged.
"""
from __future__ import annotations
from paper_paths import CODE_ROOT, INPUT_ROOT, OUTPUT_ROOT
import argparse
import json
from pathlib import Path
import re
import shutil
import sys
ROOT = INPUT_ROOT
pass
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pymupdf as pdf
from plotting.vendor.scripts.paper_figures import build_fig2_paper_typography as previous
from plotting.vendor.scripts.paper_figures.build_fig2_sequence_field_layout import export, place
SOURCE = previous.OUT
INPUTS = previous.SOURCE / 'source_snapshot'
OUT = SOURCE.parent / 'left_legends_spacious_b_20261012'

def remove_timing_rose_rays(document):
    """Suppress just the four frozen dashed rays, preserving all other paths.

    This deliberately narrow edit matches the original Matplotlib producer's
    complete, single-line ray operators. It does not redact underlying bars.
    """
    page = document[0]
    before = page.get_drawings()
    omitted = [d for d in before if d.get('dashes') == '[ 4.35 2.9 ] 0' and d['rect'].x1 < 172 and (d['type'] == 's')]
    assert len(omitted) == 4
    pattern = re.compile(b'(1\\.45 w \\[ 4\\.35 2\\.9 \\] 0 d\\s+[\\d.\\s]+RG\\s+/DeviceRGB cs\\s+114\\.009308 [\\d.]+ m\\s+[\\d.]+ [\\d.]+ l\\s+)S(?=\\s+Q)')
    count = 0
    for xref in page.get_contents():
        stream, changed = pattern.subn(b'\\1n', document.xref_stream(xref))
        if changed:
            document.update_stream(xref, stream)
            count += changed
    assert count == 4
    after = page.get_drawings()
    expected = [d for d in before if d not in omitted]
    assert len(after) == len(expected)
    for actual, old in zip(after, expected):
        assert {k: v for k, v in actual.items() if k != 'seqno'} == {k: v for k, v in old.items() if k != 'seqno'}
    return dict(removed_timing_only_rays=count, all_other_vector_paths_identical=True)

def draw_left_legends(page, write):
    """Two compact vertical legends in the empty lower sectors of the roses."""
    rgb = matplotlib.colors.to_rgb
    text_font = pdf.Font(fontfile=previous.REGULAR)
    boxes = []

    def label(text, x, y):
        width = text_font.text_length(text, fontsize=10.5)
        rect = pdf.Rect(x - 1, y - 8.2, x + width + 1, y + 1.5)
        page.draw_rect(rect, color=None, fill=(1, 1, 1))
        write(text, x, y, size=10.5, align='left', color=(0.15, 0.15, 0.15))
        boxes.append(dict(label=text, rect_pt=list(rect)))
    page.draw_rect((78, 139, 91, 146), color=rgb('#6F6F6F'), fill=(0.88, 0.88, 0.88), width=0.7)
    label('Event directions', 98, 148)
    page.draw_line((78, 157), (85, 157), color=rgb(previous.TEMPLATE_COLORS['TA']), width=2)
    page.draw_line((88, 157), (95, 157), color=rgb(previous.TEMPLATE_COLORS['TB']), width=2)
    label('TA / TB', 98, 161)
    page.draw_circle((85, 332), 2.7, color=rgb(previous.base.scores.TEMPORAL_COLOR), fill=(1, 1, 1), width=1)
    label('Rank SEQ', 98, 336)
    page.draw_circle((85, 344), 2.7, color=(1, 1, 1), fill=rgb(previous.base.scores.HYBRID_COLOR), width=0.6)
    label('Spatial Trajectory', 98, 348)
    page.draw_rect((80, 354, 90, 359), color=rgb('#777777'), fill=rgb('#BDBDBD'), width=0.7)
    label('Shuffle null', 98, 361)
    return boxes

def build_b(out, dpi, *, restore_rose_rays=False, rose_legend_drawer=None, score_legend_drawer=None, score_width_pt=247.0):
    rows = previous.base.scores._load_rows(INPUTS / 'B_scores.csv')
    summary = json.loads((INPUTS / 'B_summary.json').read_text())['cohort_statistics']
    original = json.loads((INPUTS / 'panelb_source_metadata.json').read_text())
    examples = {e['subject_id']: e['display_label'] for e in original['rose_examples']}
    with np.load(INPUTS / 'B_null.npz') as cached:
        null = np.median(cached['patient_null_timing_plus_space_score'], axis=0)
    with plt.rc_context({'font.family': previous.FONT_FAMILY, 'font.size': 13, 'pdf.fonttype': 42, 'axes.linewidth': 0.8}):
        width = 273.0 + score_width_pt
        fig = plt.figure(figsize=(width / 72, 430 / 72), facecolor='white')
        ax = fig.add_axes([260 / width, 45 / 430, score_width_pt / width, 363 / 430])
        previous.base.scores.draw_absolute_scores_and_null(ax, rows, summary, null, example_labels=examples, bootstrap_in_source_order=True)

        def forward(y):
            y = np.asarray(y)
            return np.where(y <= 2.4, (y + 1.65) * 114 / 4.05, 114 + (y - 2.4) * 249 / 27.6)

        def backward(y):
            y = np.asarray(y)
            return np.where(y <= 114, y * 4.05 / 114 - 1.65, (y - 114) * 27.6 / 249 + 2.4)
        ax.set_yscale('function', functions=(forward, backward))
        ax.set_ylim(-1.65, 30)
        ax.set_yticks([])
        for item in list(ax.texts):
            text = item.get_text()
            if text in ('Direction-\nshuffle null', 'Timing-only', 'Timing + 3D\ndirection') or 'higher with' in text:
                item.remove()
            else:
                item.set_fontsize(13 if text in examples.values() else 15)
        ax.set_xlim(-0.1, 0.98)
        ax.set_xticks([0, 0.3, 0.6, 0.9])
        ax.tick_params(axis='x', labelsize=14, pad=3, length=3)
        ax.set_xlabel('Held-out direction score', fontsize=17, labelpad=9)
        ax.set_ylabel('Patients (ordered)', fontsize=16)
        ax.yaxis.set_label_coords(-0.074 * 247 / score_width_pt, 0.665)
        score_legend = score_legend_drawer(ax) if score_legend_drawer else None
        fig.canvas.draw()
        points = []
        for collection in ax.collections:
            if isinstance(collection, matplotlib.collections.PathCollection):
                points.extend(np.asarray(collection.get_offsets()).tolist())
        fig.savefig(out / 'source_snapshot/B-score-native.pdf', facecolor='white')
        plt.close(fig)
    with pdf.open(out / 'source_snapshot/B-score-native.pdf') as doc, pdf.open(INPUTS / 'original_panelb.pdf') as roses:
        ray_check = dict(removed_timing_only_rays=0, restored_timing_only_rays=4, all_original_vector_paths_retained=True) if restore_rose_rays else remove_timing_rose_rays(roses)
        rose_spans = previous.strip_text(roses)
        page = doc[0]
        write = previous.text_drawer(page)
        placements = []
        for index, (source_y, cy, name) in enumerate([(118.3538, 115, 'E10'), (258.3666, 300, 'E14')]):
            src = pdf.Rect(56.2419, source_y - 57.7674, 171.7767, source_y + 57.7674)
            target = pdf.Rect(50, cy - 75, 200, cy + 75)
            placements.append(place(page, roses, target, src + (-0.5, -0.5, 0.5, 0.5)))
            write(name, 15, cy - 66, size=17, bold=True, align='left')
            for text, x, y, align in [('90°', 125, cy - (83 if index == 0 else 80), 'center'), ('270°', 125, cy + (88 if index == 0 else 93), 'center'), ('180°', 43, cy + 4, 'right'), ('0°', 208, cy + 4, 'left')]:
                write(text, x, y, size=12, align=align)
            scale = 150 / (src.width + 1)
            for span in rose_spans:
                if span['text'] in ('10%', '20%') and source_y - 58 < span['origin'][1] < source_y + 58:
                    xy = (np.array(span['origin']) - [src.x0 - 0.5, src.y0 - 0.5]) * scale + [50, cy - 75]
                    write(span['text'], *xy, size=12, align='left')
        boxes = (rose_legend_drawer or draw_left_legends)(page, write)
        export(doc, out / 'figures/fig2-panelb', dpi)
        page.get_pixmap(matrix=pdf.Matrix(2, 2), alpha=False).save(out / 'source_snapshot/review/panel_b.png')
    return dict(n_score_axes=1, separate_distribution_axis=False, legend_count=2, legend_columns=1, legend_location='left rose empty sectors', legend_text_boxes=boxes, legend_overlapping_patient_points=0, score_axes_rect_pt=[260, 22, 260 + score_width_pt, 385], patient_region_height_pt=249, previous_patient_region_height_pt=157, distribution_region_height_pt=114, previous_distribution_region_height_pt=102, top_space_reserved_for_legend_pt=0, cohort_statistics=summary, plotted_patient_and_summary_points=points, rose_placements=placements, rose_vector_validation=ray_check, score_legend=score_legend, display_names={'timing_only': 'Rank SEQ', 'timing_plus_space': 'Spatial Trajectory'}, bootstrap_in_original_CSV_order=True, native_font_pt=dict(axis=17, ticks=14, legend=10.5, identity=17))

def build(out, dpi):
    for name in ('figures', 'source_snapshot/review', 'trajectories'):
        (out / name).mkdir(parents=True, exist_ok=True)
    meta = json.loads((SOURCE / 'figure2_layout_metadata.json').read_text())
    for letter in 'acd':
        for path in (SOURCE / 'figures').glob(f'fig2-panel{letter}.*'):
            shutil.copy2(path, out / 'figures' / path.name)
    for path in (SOURCE / 'trajectories').glob('*.csv'):
        shutil.copy2(path, out / 'trajectories' / path.name)
    meta['panel_b'] = build_b(out, dpi)
    doc = pdf.open()
    page = doc.new_page(width=1240, height=860)
    targets = {'a': (22, 44, 704, 446), 'b': (715, 44, 1222, 446), 'c': (24, 452, 816, 842), 'd': (843, 461, 1216, 838)}
    layout = {}
    for letter, target in targets.items():
        with pdf.open(out / f'figures/fig2-panel{letter}.pdf') as part:
            layout[letter] = place(page, part, target, None if letter == 'd' else part[0].rect)
    for label, x, y in [('A', 12, 32), ('B', 702, 32), ('C', 12, 443), ('D', 824, 443)]:
        page.insert_text((x, y), label, fontsize=27, fontname='hebo')
    previous.export_pdf_png(doc, out / 'figures/fig2-complete-layout', dpi)
    page.get_pixmap(matrix=pdf.Matrix(1800 / 1240, 1800 / 1240), alpha=False).save(out / 'figures/fig2-complete-layout-preview.png')
    doc.close()
    meta.update(schema_version='fig2_left_legends_spacious_b_v1', source_candidate=str(SOURCE), producer=str(Path(__file__).relative_to(CODE_ROOT)), complete_layout=layout, legend_policy='two compact vertical legends inside the left roses; no space reserved in the score plot', human_visual_acceptance='PENDING', agent_visual_review='PENDING')
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
