#!/usr/bin/env python3
"""Enlarge native C maps, align B/C lower edges, and bring D/E upward."""
import json
from pathlib import Path
import shutil

import build_fig6_selected_r08_r09_compact_rows as previous

base, matched, layout = previous.base, previous.matched, previous.layout
np, plt, MM = previous.np, previous.plt, previous.MM
SOURCE = previous.OUT
OUT = SOURCE.with_name('selected_r08_r09_enlarged_c_20261031')
C_WIDTH, C_HEIGHT, C_LEFT, C_TOP = 177., 110., 140., 109.
FIELD_SIZE, COLUMN_GAP = 33., 3.5
BOTTOM_ROW_TOP = 220.
C_DISPLAY = {}


def enlarged_c_export(fig, stem):
    assert stem == 'fig6-panelc' and len(fig.axes) == 10
    source_meta = json.loads((SOURCE/'figure_metadata.json').read_text())
    source_layout = json.loads((SOURCE/'layout_manifest.json').read_text())
    b_box = source_layout['panels']['B']['box_mm']
    # Align field-map bottoms to the projected B coordinate-box bottom.
    bottom_global = b_box[1] + (1-source_meta['B_display']['colorbar_bounds'][1])*124.
    row_gap = source_meta['C_display']['row_gap_mm']
    second_top = bottom_global-C_TOP-FIELD_SIZE
    first_top = second_top-row_gap-FIELD_SIZE
    fig.set_size_inches(C_WIDTH/25.4, C_HEIGHT/25.4)
    positions = []
    for j, ax in enumerate(fig.axes[:8]):
        row, col = divmod(j, 4)
        top = [first_top, second_top][row]
        x = 16. + col*(FIELD_SIZE+COLUMN_GAP)
        ax.set_position([x/C_WIDTH, (C_HEIGHT-top-FIELD_SIZE)/C_HEIGHT,
                         FIELD_SIZE/C_WIDTH, FIELD_SIZE/C_HEIGHT])
        if row == 1 and col > 0:
            # The shared zero is shown on the first column; avoid adjacent 20/0 labels touching.
            ax.set_xticklabels(['', '10', '20'])
        positions.append(ax.get_position().bounds)
    for row, cbax in enumerate(fig.axes[8:]):
        pos = fig.axes[row*4+3].get_position()
        cbax.set_position([160.5/C_WIDTH, pos.y0, 2.076/C_WIDTH, pos.height])
        fig.texts[row].set_position((3.2/C_WIDTH, pos.y0+pos.height/2))
    fig.canvas.draw()
    for row in range(2):
        np.testing.assert_allclose(fig.axes[8+row].get_position().extents[[1,3]],
                                   fig.axes[4*row+3].get_position().extents[[1,3]], atol=1e-12)
    actual_bottom = C_TOP+(1-fig.axes[4].get_position().y0)*C_HEIGHT
    np.testing.assert_allclose(actual_bottom, bottom_global, atol=1e-9)
    C_DISPLAY.update(final_size_mm=[C_WIDTH,C_HEIGHT], field_size_mm=FIELD_SIZE,
        previous_field_size_mm=source_meta['C_display']['field_size_mm'],
        field_linear_enlargement=FIELD_SIZE/source_meta['C_display']['field_size_mm'],
        column_gap_mm=COLUMN_GAP, row_gap_mm=row_gap,
        field_axes_bounds=positions, colorbar_bounds=[a.get_position().bounds for a in fig.axes[8:]],
        B_C_plot_bottom_mm=bottom_global, lower_edges_aligned=True)
    matched.OUT = OUT
    previous.NATIVE_EXPORT(fig, stem)


def compose():
    from matplotlib.font_manager import findfont, FontProperties
    doc = base.fitz.open()
    page = doc.new_page(width=320*MM, height=294*MM)
    font = findfont(FontProperties(family='DejaVu Sans', weight='bold'))
    page.insert_font(fontname='FigureBold', fontfile=font)
    boxes = {'A':(0,1,320,108), 'B':(3,102,142,226),
             'C':(C_LEFT,C_TOP,C_LEFT+C_WIDTH,C_TOP+C_HEIGHT),
             'D':(3,BOTTOM_ROW_TOP,248,BOTTOM_ROW_TOP+71),
             'E':(255,BOTTOM_ROW_TOP,317,BOTTOM_ROW_TOP+71)}
    letters = {'B':(3,106), 'C':(C_LEFT,106), 'D':(3,220), 'E':(254,220)}
    panels = {}
    for label, box in boxes.items():
        source = OUT/'figures'/f'fig6-panel{label.lower()}.pdf'
        with base.fitz.open(source) as src:
            x0,y0,x1,y1 = box
            scale = min((x1-x0)*MM/src[0].rect.width,(y1-y0)*MM/src[0].rect.height)
            rect = base.fitz.Rect(x0*MM,y0*MM,x0*MM+src[0].rect.width*scale,y0*MM+src[0].rect.height*scale)
            page.show_pdf_page(rect,src,0,keep_proportion=True)
        if label in letters:
            page.insert_text(tuple(v*MM for v in letters[label]),label,
                             fontsize=layout.LETTER,fontname='FigureBold')
        if label != 'A':
            np.testing.assert_allclose(scale,1,atol=1e-6)
        panels[label] = dict(source=str(source),box_mm=[v/MM for v in rect],scale=scale)
    path = OUT/'figures/fig6-complete-layout.pdf'
    doc.save(path,garbage=4,deflate=True)
    doc.close()
    with base.fitz.open(path) as doc:
        doc[0].get_pixmap(matrix=base.fitz.Matrix(220/72,220/72),alpha=False).save(path.with_suffix('.png'))
        path.with_suffix('.svg').write_text(doc[0].get_svg_image(text_as_path=False))
    prior = json.loads((SOURCE/'layout_manifest.json').read_text())
    base.write(OUT/'layout_manifest.json',dict(canvas_mm=[320,294],panels=panels,
        bottom_row_shift_up_from_previous_mm=8., B_C_plot_bottom_mm=C_DISPLAY['B_C_plot_bottom_mm'],
        C_field_size_mm=FIELD_SIZE, B_C_plot_lower_edges_aligned=True,
        D_E_plot_height_mm=46.,bottom_row_plot_top_mm=230.,bottom_row_plot_bottom_mm=276.,
        A_E_right_plot_edge_mm=313.6,E_right_panel_edge_mm=317.,
        same_field_and_statistic_plot_height=True,typography=prior['typography']))


def documents():
    caption = (SOURCE/'caption_zh.md').read_text()
    first = caption.split('\n\n')[1]
    caption = caption.replace(first,
        'C 的八幅场图从约 29.74 mm 放大至 33 mm（线性尺寸增加约 11%），字体与阶段时刻保持，'
        '两行地图之间的净间隙保持约 15.26 mm。第二行场图与 B 三维坐标框投影的下边缘对齐。'
        'D/E 整行相对上一版上移 8 mm；其地图、统计及右边界对齐保持。'
        'A/B/D/E 独立面板导出与上一版字节一致，原生数据和统计均未改变。')
    (OUT/'caption_zh.md').write_text(caption)
    readme = (SOURCE/'figures/README.md').read_text()
    readme = readme.replace('完整 A–E 收紧两处行间距，C 两个模型之间的间距也缩小。地图与 E 的尺寸和右侧对齐保持。',
        '完整 A–E 放大 C，并将 C 第二行场图与 B 坐标框的下边缘对齐。D/E 整行上移 8 mm，地图与 E 的尺寸和横向对齐保持。')
    readme = readme.replace('两行 Model Onset 1/2 的地图大小、数据与字号保持。两行之间的空隙缩小，每行色条仍与场图等高。',
        '两行 Model Onset 1/2 的八幅场图边长由约 29.74 mm 放大至 33 mm，数据与字号保持。'
        '两行之间的净间距保持约 15.26 mm，每行色条与场图等高；第二行下边缘与 B 三维坐标框对齐。')
    (OUT/'figures/README.md').write_text(readme)
    (OUT/'README.md').write_text('# Figure 6：放大 C 与对齐下边缘\n\n'
        '[完整图](figures/fig6-complete-layout.png) · [PDF](figures/fig6-complete-layout.pdf)\n\n'
        f'复现：`python {Path(__file__).resolve()}`。原生数据、阶段时刻及统计保持。\n')


def main():
    pointer = base.MAIN/'results/paper-ready-figure/fig6/current_version.json'
    frozen = pointer.read_bytes()
    shutil.copytree(SOURCE,OUT,dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('validation.json','visual_review.json'))
    plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans',
                         'pdf.fonttype':42,'svg.fonttype':'none','path.simplify':False})
    meta = json.loads((SOURCE/'figure_metadata.json').read_text())
    matched.SOURCE = SOURCE
    matched.export = enlarged_c_export
    try:
        cmeta = matched.draw_c(meta)
    finally:
        matched.export = previous.NATIVE_EXPORT
    cmeta.update(C_DISPLAY)
    cmeta['individual_row_colorbars'] = C_DISPLAY['colorbar_bounds']
    meta.update(producer=str(Path(__file__).resolve()),previous_candidate=str(SOURCE),
        C_display=cmeta,revision_scope='Enlarge C fields, align B/C plot bottoms, and shift D/E up 8 mm.',
        status='PENDING_AUTHOR_VISUAL_REVIEW')
    base.write(OUT/'figure_metadata.json',meta)
    compose()
    unchanged = []
    for name in ['a','b','d','e','b-interictal','b-full-trajectory']:
        for ext in ['png','pdf','svg']:
            filename = f'fig6-panel{name}.{ext}'
            assert (OUT/'figures'/filename).read_bytes() == (SOURCE/'figures'/filename).read_bytes()
            unchanged.append(filename)
    caches = ['trajectory_data.npz','stage_readouts.npz','field_comparisons.json','A_C_stage_correspondence.csv',
              'model_selected_patient_matches.json','model_selected_patient_arrays.npz',
              'all_patient_seizure_matches.csv','cohort_statistics_selected.json']
    for name in caches:
        assert (OUT/name).read_bytes() == (SOURCE/name).read_bytes()
    text_checks = {}
    for stem in ['fig6-panelc','fig6-complete-layout']:
        with base.fitz.open(OUT/'figures'/f'{stem}.pdf') as doc:
            text_checks[stem] = all(page.rect.contains(base.fitz.Rect(span['bbox'])) for page in doc
                for block in page.get_text('dict')['blocks'] if block['type']==0
                for line in block['lines'] for span in line['spans'])
    assert all(text_checks.values()),text_checks
    assert pointer.read_bytes() == frozen
    base.write(OUT/'validation.json',dict(unchanged_panel_files=unchanged,
        all_native_data_times_fields_and_statistics_unchanged=True,
        C_field_size_mm=FIELD_SIZE,B_C_plot_lower_edges_aligned=True,
        PDF_text_inside_canvas=text_checks,formal_pointer_unchanged=True,human_visual_acceptance='PENDING'))
    documents()
    shutil.copy2(__file__,OUT/'source'/Path(__file__).name)
    print(json.dumps(dict(output=str(OUT),C=C_DISPLAY,checks=text_checks),indent=2))


if __name__ == '__main__':
    main()
