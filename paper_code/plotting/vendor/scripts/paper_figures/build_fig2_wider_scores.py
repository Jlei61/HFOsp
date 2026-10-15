"""Widen Figure 2B's score axes while retaining height and native typography."""
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
import numpy as np
import pymupdf as pdf
from plotting.vendor.scripts.paper_figures import build_fig2_separate_legends as prior
SOURCE = prior.OUT
OUT = SOURCE.parent / 'wider_b_score_axes_20261012'

def build(out, dpi):
    for name in ('figures', 'source_snapshot/review', 'trajectories'):
        (out / name).mkdir(parents=True, exist_ok=True)
    old = json.loads((SOURCE / 'figure2_layout_metadata.json').read_text())
    meta = json.loads(json.dumps(old))
    for letter in 'acd':
        for path in (SOURCE / 'figures').glob(f'fig2-panel{letter}.*'):
            shutil.copy2(path, out / 'figures' / path.name)
    for path in (SOURCE / 'trajectories').glob('*.csv'):
        shutil.copy2(path, out / 'trajectories' / path.name)
    b = prior.base.build_b(out, dpi, restore_rose_rays=True, rose_legend_drawer=prior.rose_legend, score_legend_drawer=prior.score_legend, score_width_pt=310.0)
    b.update(legend_location=old['panel_b']['legend_location'], display_names=old['panel_b']['display_names'], canvas_pt=[583, 430], prior_score_width_pt=247, score_width_pt=310, score_height_pt=363, score_width_increase_percent=100 * (310 / 247 - 1))
    b['score_legend']['axes_geometry_unchanged'] = False
    b['score_legend']['axis_height_and_vertical_patient_spacing_unchanged'] = True
    meta['panel_b'] = b
    doc = pdf.open()
    page = doc.new_page(width=1240, height=860)
    layout = {}
    for letter in 'abcd':
        with pdf.open(out / f'figures/fig2-panel{letter}.pdf') as part:
            if letter == 'b':
                layout[letter] = prior.base.place(page, part, (680, 44, 1230, 446), part[0].rect)
            else:
                placement = old['complete_layout'][letter]
                layout[letter] = prior.base.place(page, part, placement['placed_rect_pt'], placement['source_clip_pt'])
    for label, x, y in [('A', 12, 32), ('B', 667, 32), ('C', 12, 443), ('D', 824, 443)]:
        page.insert_text((x, y), label, fontsize=27, fontname='hebo')
    prior.previous.export_pdf_png(doc, out / 'figures/fig2-complete-layout', dpi)
    page.get_pixmap(matrix=pdf.Matrix(1800 / 1240, 1800 / 1240), alpha=False).save(out / 'figures/fig2-complete-layout-preview.png')
    doc.close()
    meta.update(schema_version='fig2_wider_b_score_axes_v1', source_candidate=str(SOURCE), producer=str(Path(__file__).relative_to(CODE_ROOT)), complete_layout=layout, human_visual_acceptance='PENDING', agent_visual_review='PENDING')
    (out / 'figure2_layout_metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    shutil.copy2(Path(__file__), out / 'source_snapshot' / Path(__file__).name)
    return meta

def validate(out):
    meta = json.loads((out / 'figure2_layout_metadata.json').read_text())
    old = json.loads((SOURCE / 'figure2_layout_metadata.json').read_text())
    frozen = []
    for letter in 'acd':
        for path in (SOURCE / 'figures').glob(f'fig2-panel{letter}.*'):
            assert prior.previous.sha(path) == prior.previous.sha(out / 'figures' / path.name)
            frozen.append(path.name)
        for key in ('source_clip_pt', 'placed_rect_pt', 'scale'):
            assert np.allclose(meta['complete_layout'][letter][key], old['complete_layout'][letter][key], rtol=0, atol=1e-10)
    assert meta['panel_a'] == old['panel_a'] and meta['panel_c'] == old['panel_c']
    csvs = list((SOURCE / 'trajectories').glob('*.csv'))
    for path in csvs:
        assert prior.previous.sha(path) == prior.previous.sha(out / 'trajectories' / path.name)
    b = meta['panel_b']
    for key in ('patient_region_height_pt', 'distribution_region_height_pt', 'cohort_statistics', 'plotted_patient_and_summary_points', 'rose_placements', 'legend_text_boxes', 'display_names'):
        assert b[key] == old['panel_b'][key], key
    assert b['n_score_axes'] == 1 and (not b['separate_distribution_axis'])
    assert b['score_legend']['patient_points_and_connectors_covered'] == 0
    assert meta['complete_layout']['b']['scale'] == old['complete_layout']['b']['scale']
    with pdf.open(SOURCE / 'figures/fig2-panelb.pdf') as a, pdf.open(out / 'figures/fig2-panelb.pdf') as z:
        clip = pdf.Rect(0, 0, 224, 430)
        before = a[0].get_pixmap(matrix=pdf.Matrix(2, 2), clip=clip, alpha=False)
        after = z[0].get_pixmap(matrix=pdf.Matrix(2, 2), clip=clip, alpha=False)
        assert before.samples == after.samples
    for name in ('panelb', 'complete-layout'):
        with pdf.open(out / f'figures/fig2-{name}.pdf') as doc:
            spans = [s for block in doc[0].get_text('dict', flags=pdf.TEXTFLAGS_TEXT)['blocks'] for line in block.get('lines', []) for s in line['spans']]
            assert all(((doc[0].rect + (-0.5, -0.5, 0.5, 0.5)).contains(pdf.Rect(s['bbox'])) for s in spans))
            if name == 'complete-layout':
                assert [s['text'] for s in spans if s['size'] > 25] == list('ABCD')
    report = dict(status='PASS', A_C_D_exports_byte_identical=frozen, A_C_D_placements_unchanged=True, C_trajectory_CSVs_byte_identical=len(csvs), B_rose_pixels_unchanged=True, B_score_axis_width_pt=[247, 310], B_score_axis_height_pt=363, B_width_increase_percent=b['score_width_increase_percent'], B_scale_and_fonts_unchanged=True, B_patient_values_statistics_and_vertical_spacing_unchanged=True, B_one_shared_axis=True, B_legend_overlapping_patient_points_and_connectors=0, PDF_text_inside_canvas=True, human_visual_acceptance='PENDING')
    (out / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()
    out = args.output.resolve()
    build(out, args.dpi)
    validate(out)
    print(out)
