"""Figure 2: distinguish rose line styles from patient-score markers."""
from __future__ import annotations
from paper_paths import CODE_ROOT, INPUT_ROOT, OUTPUT_ROOT
import argparse
import json
from pathlib import Path
import shutil
import sys
ROOT = INPUT_ROOT
pass
import matplotlib
matplotlib.use('Agg')
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pymupdf as pdf
from plotting.vendor.scripts.paper_figures import build_fig2_left_legends as base
previous = base.previous
SOURCE = base.OUT
OUT = SOURCE.parent / 'separate_rose_score_legends_20261012'

def rose_legend(page, write):
    font = pdf.Font(fontfile=previous.REGULAR)
    boxes = []
    for text, y in [('Event directions', 148), ('Rank SEQ', 161), ('Spatial TRAJ.', 174)]:
        rect = pdf.Rect(97, y - 8.2, 99 + font.text_length(text, fontsize=10.5), y + 1.5)
        page.draw_rect(rect, color=None, fill=(1, 1, 1))
        write(text, 98, y, size=10.5, align='left', color=(0.15, 0.15, 0.15))
        boxes.append(dict(label=text, rect_pt=list(rect)))
    page.draw_rect((78, 139, 91, 146), color=(0.435, 0.435, 0.435), fill=(0.88, 0.88, 0.88), width=0.7)
    page.draw_line((78, 157), (93, 157), color=(0.2, 0.2, 0.2), width=1.3, dashes='[3 2] 0')
    page.draw_line((78, 170), (93, 170), color=(0.2, 0.2, 0.2), width=1.8)
    return boxes

def score_legend(ax):
    handles = [Line2D([], [], ls='none', marker='o', mfc='white', mec=previous.base.scores.TEMPORAL_COLOR, ms=5, mew=1), Line2D([], [], ls='none', marker='o', mfc=previous.base.scores.HYBRID_COLOR, mec='white', ms=5, mew=0.6), Patch(facecolor='#BDBDBD', edgecolor='#777777', linewidth=0.7)]
    labels = ['Rank SEQ', 'Spatial TRAJ.', 'Shuffle null']
    legend = ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.035, 0.985), fontsize=10.5, ncol=1, handlelength=1.2, labelspacing=0.45, handletextpad=0.6, frameon=True, facecolor='white', edgecolor='none', framealpha=1, fancybox=False, borderpad=0.3, borderaxespad=0)
    ax.figure.canvas.draw()
    lb = legend.get_window_extent(ax.figure.canvas.get_renderer())
    ab = ax.get_window_extent(ax.figure.canvas.get_renderer())
    assert ab.contains(lb.x0, lb.y0) and ab.contains(lb.x1, lb.y1)
    overlap = 0
    for collection in ax.collections:
        if isinstance(collection, matplotlib.collections.PathCollection):
            points = collection.get_offset_transform().transform(collection.get_offsets())
            overlap += sum((lb.contains(*point) for point in points))
    for line in ax.lines:
        xy = np.asarray(line.get_xydata())
        if len(xy) == 2 and xy[0, 1] == xy[1, 1] and (xy[0, 1] >= 3):
            points = ax.transData.transform(xy)
            assert not (min(points[:, 0]) <= lb.x1 and max(points[:, 0]) >= lb.x0 and (lb.y0 <= points[0, 1] <= lb.y1))
    assert overlap == 0
    box = lb.transformed(ax.figure.dpi_scale_trans.inverted())
    return dict(labels=labels, columns=1, font_pt=10.5, location='upper left inside score axes', box_pt=[box.x0 * 72, 430 - box.y1 * 72, box.x1 * 72, 430 - box.y0 * 72], patient_points_and_connectors_covered=0, axes_geometry_unchanged=True)

def abbreviate_panel(letter, out, dpi):
    """Replace only the trajectory text, preserving all scientific graphics."""
    with pdf.open(SOURCE / f'figures/fig2-panel{letter}.pdf') as doc:
        page = doc[0]
        edits = []
        for block in page.get_text('dict', flags=pdf.TEXTFLAGS_TEXT)['blocks']:
            for line in block.get('lines', []):
                for span in line['spans']:
                    if 'trajectory' not in span['text'].lower():
                        continue
                    rect = pdf.Rect(span['bbox'])
                    select = pdf.Rect(rect)
                    if line['dir'] == (1.0, 0.0):
                        select.y0 = (rect.y0 + rect.y1) / 2 - 0.15
                        select.y1 = select.y0 + 0.3
                    else:
                        select.x0 = (rect.x0 + rect.x1) / 2 - 0.15
                        select.x1 = select.x0 + 0.3
                    page.add_redact_annot(select, fill=None)
                    edits.append((span, line['dir']))
        assert len(edits) == 2
        page.apply_redactions(images=0, graphics=0, text=0)
        write = previous.text_drawer(page)
        changed = []
        for span, direction in edits:
            text = span['text'].replace('Trajectory', 'TRAJ.').replace('trajectory', 'TRAJ.')
            rect = pdf.Rect(span['bbox'])
            color = tuple(((span['color'] >> shift & 255) / 255 for shift in (16, 8, 0)))
            if direction == (1.0, 0.0):
                write(text, (rect.x0 + rect.x1) / 2, span['origin'][1], size=span['size'], bold=True, color=color)
            else:
                write(text, span['origin'][0], (rect.y0 + rect.y1) / 2, size=span['size'], bold=True, color=color, rotate=90)
            changed.append(dict(old=span['text'], new=text, old_bbox=list(rect)))
        if letter == 'a':
            previous.export_pdf_png(doc, out / f'figures/fig2-panel{letter}', dpi)
        else:
            base.export(doc, out / f'figures/fig2-panel{letter}', dpi)
    return changed

def build(out, dpi):
    for name in ('figures', 'source_snapshot/review', 'trajectories'):
        (out / name).mkdir(parents=True, exist_ok=True)
    meta = json.loads((SOURCE / 'figure2_layout_metadata.json').read_text())
    edits = {letter: abbreviate_panel(letter, out, dpi) for letter in 'ac'}
    for path in (SOURCE / 'figures').glob('fig2-paneld.*'):
        shutil.copy2(path, out / 'figures' / path.name)
    for path in (SOURCE / 'trajectories').glob('*.csv'):
        shutil.copy2(path, out / 'trajectories' / path.name)
    b = base.build_b(out, dpi, restore_rose_rays=True, rose_legend_drawer=rose_legend, score_legend_drawer=score_legend)
    b.update(legend_location='rose line-style legend on left; score-marker legend at upper left of right axes', display_names={'timing_only': 'Rank SEQ', 'timing_plus_space': 'Spatial TRAJ.'})
    meta['panel_b'] = b
    doc = pdf.open()
    page = doc.new_page(width=1240, height=860)
    targets = {'a': (22, 44, 704, 446), 'b': (715, 44, 1222, 446), 'c': (24, 452, 816, 842), 'd': (843, 461, 1216, 838)}
    layout = {}
    for letter, target in targets.items():
        with pdf.open(out / f'figures/fig2-panel{letter}.pdf') as part:
            layout[letter] = base.place(page, part, target, None if letter == 'd' else part[0].rect)
    for label, x, y in [('A', 12, 32), ('B', 702, 32), ('C', 12, 443), ('D', 824, 443)]:
        page.insert_text((x, y), label, fontsize=27, fontname='hebo')
    previous.export_pdf_png(doc, out / 'figures/fig2-complete-layout', dpi)
    page.get_pixmap(matrix=pdf.Matrix(1800 / 1240, 1800 / 1240), alpha=False).save(out / 'figures/fig2-complete-layout-preview.png')
    doc.close()
    meta.update(schema_version='fig2_separate_rose_score_legends_v1', source_candidate=str(SOURCE), producer=str(Path(__file__).relative_to(CODE_ROOT)), complete_layout=layout, typography_edits=edits, legend_policy='left: event directions and method line styles; right: method markers and shuffle null', human_visual_acceptance='PENDING', agent_visual_review='PENDING')
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
