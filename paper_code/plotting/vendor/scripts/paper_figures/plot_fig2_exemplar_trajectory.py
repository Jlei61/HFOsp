"""Overlay data-derived directions on the E10 Figure 2A candidate only."""
from __future__ import annotations
from paper_paths import CODE_ROOT, INPUT_ROOT, OUTPUT_ROOT
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pymupdf as pdf
from scipy.spatial import cKDTree
ROOT = INPUT_ROOT
pass
from plotting.vendor.scripts.paper_figures.build_fig2_sequence_field_layout import export
from plotting.vendor.scripts.paper_figures.fig2_embedded_plane import build_embedded_plane
from plotting.vendor.scripts.plot_contact_plane_static import VIS_MASK_REL, _smooth_rank_field_mm
from plotting.vendor.scripts.plot_topic5_interictal_template_ab_fields import build_interictal_ab_panel_payloads, draw_interictal_rank_field_panel
from plotting.vendor.src.topic5_field_trajectory import gaussian_rank_value_gradient, rank_gradient_trajectory
BASE = ROOT / 'results/paper-ready-figure/fig2/candidates/sequence_to_spatial_field_20261009'
OUT = ROOT / 'results/paper-ready-figure/fig2/candidates/exemplar_rank_trajectory_20261009'
FIELD = ROOT / 'results/interictal_propagation_masked/template_gradient_fields/per_subject/epilepsiae_1146.json'
CACHE = ROOT / 'results/interictal_propagation_masked/event_envelope_fields/epilepsiae_1146_event_envelope_field_cache.npz'
TEMPLATE_COLORS = {'TA': '#B2182B', 'TB': '#2166AC'}
READOUT_ARROW_SHIFT_MS = -45.0
TRAJECTORY_CONTACT_CLEARANCE_MM = 2.0
TRAJECTORY_ARROW_FRACTIONS = (1.0,)
SEQUENCE_ARROW_COLOR = '#EF8FB3'
SEQUENCE_ARROW_OUTLINE = '#FFFFFF'
TRAJECTORY_COLORS = {'TA': '#F07862', 'TB': '#55ADE9'}
TRAJECTORY_HALO = '#FFFFFF'
SEQUENCE_TITLE_SUFFIX = 'seq.'
TRAJECTORY_TITLE_SUFFIX = 'traj.'
SUBPLOT_TITLE_SIZE_PT = 13
COMPACT_GEOMETRY_FLOW = False

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)

def freeze():
    snapshot = OUT / 'source_snapshot'
    snapshot.mkdir(parents=True, exist_ok=True)
    sources = {'template_fields.json': FIELD, 'event_cache.npz': CACHE, 'figure2_layout_metadata.json': BASE / 'figure2_layout_metadata.json', 'fig2-panela.pdf': BASE / 'figures/fig2-panela.pdf', 'fig2_panelc_metadata.json': BASE / 'source_snapshot/fig2_panelc_metadata.json', 'original-panelc.pdf': BASE / 'source_snapshot/figures/fig2-panelc.pdf', 'original-paneld.pdf': BASE / 'source_snapshot/figures/fig2-paneld.pdf', 'geometry_source_metadata.json': BASE / 'source_snapshot/fig2_panela_metadata.json'}
    manifest_path = snapshot / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, item in manifest.items():
            assert sha(snapshot / name) == item['sha256'], name
        for name, source in sources.items():
            if name not in manifest:
                shutil.copy2(source, snapshot / name)
                manifest[name] = {'source': str(source), 'sha256': sha(snapshot / name)}
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    else:
        manifest = {}
        for name, source in sources.items():
            shutil.copy2(source, snapshot / name)
            manifest[name] = {'source': str(source), 'sha256': sha(snapshot / name)}
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return snapshot

def in_composite(points, placement):
    src = np.array(placement['source_clip_pt'][:2])
    dst = np.array(placement['placed_rect_pt'][:2])
    return (np.asarray(points) - src) * placement['scale'] + dst

def spatial_pdf_points(points, axes_rect, xlim, ylim):
    r = pdf.Rect(axes_rect)
    points = np.asarray(points)
    x = r.x0 + (points[:, 0] - xlim[0]) / np.ptp(xlim) * r.width
    y = r.y1 - (points[:, 1] - ylim[0]) / np.ptp(ylim) * r.height
    return np.c_[x, y]

def display_trajectory(trajectory, contacts):
    """Trim only overlapping displayed ends; preserve the computed trajectory."""
    points = np.asarray(trajectory['points_mm'])
    arc = np.asarray(trajectory['arc_length_mm'])
    distances = cKDTree(contacts).query(points)[0]
    clear = np.flatnonzero(distances >= TRAJECTORY_CONTACT_CLEARANCE_MM)
    if len(clear) < 3:
        raise ValueError('No resolved trajectory with contact-symbol clearance')
    first, last = (int(clear[0]), int(clear[-1]))
    return (points[first:last + 1], {'display_only': True, 'contact_symbol_clearance_mm': TRAJECTORY_CONTACT_CLEARANCE_MM, 'first_computed_point_index': first, 'last_computed_point_index': last, 'trimmed_early_arc_mm': float(arc[first]), 'trimmed_late_arc_mm': float(arc[-1] - arc[last]), 'computed_late_endpoint_mm': points[-1].tolist(), 'displayed_late_endpoint_mm': points[last].tolist(), 'computed_late_contact_distance_mm': float(distances[-1]), 'displayed_late_contact_distance_mm': float(distances[last])})

def trajectory_arrow_layers(points, scale=1.0):
    """Curved ribbon joined to one rigid, flat terminal triangle per layer.

    Both layers share the same base, tangent, and normal. Only the shaft follows
    changing curve normals; the head never twists or bends with the curve.
    """
    points = np.asarray(points, dtype=float)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    assert np.all(np.diff(arc) > 0)
    tangent = np.gradient(points, arc, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    base_s = arc[-1] - 7.0 * scale
    assert base_s > 3.3 * scale
    base = np.array([np.interp(base_s, arc, points[:, k]) for k in (0, 1)])
    tip = points[-1]
    direction = (tip - base) / np.linalg.norm(tip - base)
    head_normal = np.array([-direction[1], direction[0]])
    shaft_s = np.r_[arc[arc < base_s], base_s]
    shaft = np.column_stack([np.interp(shaft_s, arc, points[:, k]) for k in (0, 1)])
    local_tangent = np.column_stack([np.interp(shaft_s, arc, tangent[:, k]) for k in (0, 1)])
    local_tangent /= np.linalg.norm(local_tangent, axis=1)[:, None]
    normals = np.c_[-local_tangent[:, 1], local_tangent[:, 0]]
    blend = np.clip((shaft_s - (base_s - 2.0 * scale)) / (2.0 * scale), 0, 1)
    normals = normals * (1 - blend[:, None]) + head_normal * blend[:, None]
    normals /= np.linalg.norm(normals, axis=1)[:, None]

    def contour(shaft_width, head_width, tip_inset):
        head_tip = tip - direction * tip_inset
        left = np.vstack([shaft + normals * shaft_width / 2, base + head_normal * head_width / 2, head_tip])
        right = np.vstack([shaft - normals * shaft_width / 2, base - head_normal * head_width / 2, head_tip])
        theta = np.linspace(0, np.pi, 18)
        tail = points[0] + shaft_width / 2 * (-np.cos(theta)[:, None] * normals[0] - np.sin(theta)[:, None] * tangent[0])
        polygon = np.vstack([left, right[-2::-1], tail[1:]])
        assert np.isfinite(polygon).all()
        return polygon
    return (contour(3.3 * scale, 5.4 * scale, 0.0), contour(1.3 * scale, 3.0 * scale, 0.8 * scale))

def colored_path(page, points, color):
    outer, inner = trajectory_arrow_layers(points)
    for polygon, fill in [(outer, TRAJECTORY_HALO), (inner, color)]:
        page.draw_polyline(polygon.tolist(), color=None, fill=matplotlib.colors.to_rgb(fill), closePath=True)

def straight_direction_arrow(page, start, tip, *, shaft_width=1.35, head_length=5.0, head_width=4.6):
    """Draw a rose-pink, white-edged polygon with no line through the tip."""
    start, tip = (np.asarray(start), np.asarray(tip))
    tangent = (tip - start) / np.linalg.norm(tip - start)
    normal = np.array([-tangent[1], tangent[0]])
    base = tip - head_length * tangent
    polygon = np.array([start + normal * shaft_width / 2, base + normal * shaft_width / 2, base + normal * head_width / 2, tip, base - normal * head_width / 2, base - normal * shaft_width / 2, start - normal * shaft_width / 2])
    page.draw_polyline(polygon.tolist(), color=matplotlib.colors.to_rgb(SEQUENCE_ARROW_OUTLINE), fill=matplotlib.colors.to_rgb(SEQUENCE_ARROW_COLOR), width=0.22, closePath=True)
    return {'tail_pt': start.tolist(), 'tip_pt': tip.tolist(), 'head_base_pt': base.tolist(), 'polygon_pt': polygon.tolist(), 'shaft_head_single_polygon': True}

def mpl_path(ax, points, color, *, scale=1.25):
    ax.apply_aspect()
    points_pt = ax.transData.transform(points) * 72 / ax.figure.dpi
    outer, inner = trajectory_arrow_layers(points_pt, scale=scale)

    def to_data(points_pt):
        return ax.transData.inverted().transform(points_pt * ax.figure.dpi / 72)
    ax.fill(*to_data(outer).T, color=TRAJECTORY_HALO, edgecolor='none', zorder=5)
    ax.fill(*to_data(inner).T, color=color, edgecolor='none', zorder=6)

def representation_layout(source, layout, field_rects, readout_rects, static, embedded_plane):
    """Use identical data frames, with independently sized text and colorbars."""
    page = source[0]
    removed = {'Electrode-sequence template (1D)', 'Local electrode geometry', 'Spatial propagation field (2D)', 'TA samples', 'TB samples', 'TA', 'TB', 'TA field', 'TB field'}
    found = set()
    channel_labels = {'TA': [], 'TB': []}
    colorbar_rects = [d['rect'] for d in page.get_drawings() if 2 < d['rect'].width < 6 and 80 < d['rect'].height < 110]
    for block in page.get_text('dict')['blocks']:
        for line in block.get('lines', []):
            for span in line['spans']:
                text = span['text']
                if span['bbox'][0] < 80 and text.startswith(('ICL', 'SCL')):
                    lab = 'TA' if span['bbox'][1] < 190 else 'TB'
                    channel_labels[lab].append(text)
                if text in removed or span['bbox'][0] < 180 or span['bbox'][0] > 440:
                    if text in removed:
                        found.add(text)
                    rect = pdf.Rect(span['bbox'])
                    if line['dir'] == (1.0, 0.0):
                        center = (rect.y0 + rect.y1) / 2
                        rect.y0, rect.y1 = (center - 0.2, center + 0.2)
                    else:
                        center = (rect.x0 + rect.x1) / 2
                        rect.x0, rect.x1 = (center - 0.2, center + 0.2)
                    page.add_redact_annot(rect, fill=None)
    assert found == removed, found
    page.apply_redactions(images=0, graphics=0, text=0)
    result = pdf.open()
    target = result.new_page(width=660, height=390)
    placements = {}

    def place(name, clip, dest):
        clip = pdf.Rect(clip)
        dest = pdf.Rect(dest)
        target.show_pdf_page(dest, source, 0, clip=clip, keep_proportion=False)
        placements[name] = {'source_clip_pt': list(clip), 'placed_rect_pt': list(dest), 'scale_xy': [dest.width / clip.width, dest.height / clip.height]}
    fonts = {}
    for name, filename in (('Fig2ShortLabels', 'DejaVuSans-Bold.ttf'), ('Fig2Text', 'DejaVuSans.ttf')):
        fontfile = f'/usr/share/fonts/truetype/dejavu/{filename}'
        target.insert_font(fontname=name, fontfile=fontfile)
        fonts[name] = pdf.Font(fontfile=fontfile)

    def label(text, x, baseline, size=10, align='center', bold=False, color=(0, 0, 0)):
        name = 'Fig2ShortLabels' if bold else 'Fig2Text'
        width = fonts[name].text_length(text, fontsize=size)
        x -= width * {'left': 0, 'center': 0.5, 'right': 1}[align]
        target.insert_text((x, baseline), text, fontname=name, fontsize=size, color=color)
        return width

    def vertical_label(text, x, center_y, size=12):
        length = fonts['Fig2Text'].text_length(text, fontsize=size)
        target.insert_text((x, center_y + length / 2), text, rotate=90, fontname='Fig2Text', fontsize=size)
    titles = []
    side = 132.0
    for i, lab in enumerate(('TA', 'TB')):
        top = 34 + 180 * i
        for kind, source_rect, x, title in (('sequence', readout_rects[i], 50, f'{lab} {SEQUENCE_TITLE_SUFFIX}'), ('field', field_rects[i], 450, f'{lab} {TRAJECTORY_TITLE_SUFFIX}')):
            name = f'{kind}_{lab.lower()}'
            axes = in_composite([[source_rect.x0, source_rect.y0], [source_rect.x1, source_rect.y1]], layout[name])
            original_axes = pdf.Rect(axes.ravel().tolist())
            final_axes = pdf.Rect(x, top, x + side, top + side)
            scale_x, scale_y = (side / original_axes.width, side / original_axes.height)
            place(name, original_axes + (-0.45, -0.45, 0.45, 0.45), final_axes + (-0.45 * scale_x, -0.45 * scale_y, 0.45 * scale_x, 0.45 * scale_y))
            placements[name]['data_axes_rect_pt'] = list(final_axes)
            placements[name]['source_data_axes_rect_pt'] = list(original_axes)
            titles.append(title)
            label(title, x + side / 2, top - 10, size=SUBPLOT_TITLE_SIZE_PT, bold=True, color=matplotlib.colors.to_rgb(TEMPLATE_COLORS[lab]))
            limits = np.array(static['readout_xlim_ms'] if kind == 'sequence' else static['display_xlim_mm'])
            for value in [0, 100] if kind == 'sequence' else [-20, 0, 20]:
                tick_x = x + (value - limits[0]) / np.ptp(limits) * side
                target.draw_line((tick_x, top + side), (tick_x, top + side + 3), width=0.8)
                label(str(value).replace('-', '−'), tick_x, top + side + 14)
            if kind == 'sequence':
                names = channel_labels[lab]
                for row, text in enumerate(names):
                    label(text, x - 5, top + (row + 0.5) / len(names) * side + 3.0, size=9.6, align='right')
            else:
                ylim = np.array(static['display_ylim_mm'])
                for value in [0, 20]:
                    tick_y = top + side - (value - ylim[0]) / np.ptp(ylim) * side
                    target.draw_line((x - 3, tick_y), (x, tick_y), width=0.8)
                    label(str(value), x - 7, tick_y + 3.2, align='right')
                vertical_label('y (mm)', x - 33, top + side / 2)
            bars = [r for r in colorbar_rects if 1 < r.x0 - original_axes.x1 < 5]
            bar = min(bars, key=lambda r: abs(r.y0 + r.y1 - (original_axes.y0 + original_axes.y1)))
            bar_dest = pdf.Rect(x + side + 8, top, x + side + 12, top + side)
            place(f'{name}_colorbar', bar + (-0.1, -0.1, 0.1, 0.1), bar_dest + (-0.1, -0.1, 0.1, 0.1))
            values = [0, 1] if kind == 'sequence' else [0, 0.5, 1]
            for value in values:
                tick_y = top + (1 - value) * side
                target.draw_line((bar_dest.x1, tick_y), (bar_dest.x1 + 2.5, tick_y), width=0.7)
                text = str(value) if kind == 'sequence' else {0: '0 early', 0.5: '0.5', 1: '1 late'}[value]
                label(text, bar_dest.x1 + 5, tick_y + 3.2, align='left')
            if kind == 'sequence':
                vertical_label('Normalized magnitude', x + side + 38, top + side / 2, size=10)
            else:
                label('ranks', bar_dest.x0 - 1, top - 8, size=11, align='left')
    for name, scale, x, y in (('projection', 0.66, 241, 240),):
        clip = pdf.Rect(layout[name]['placed_rect_pt'])
        clip.y1 += 2.0
        place(name, clip, [x, y, x + clip.width * scale, y + clip.height * scale])
    projection_rect = pdf.Rect(placements['projection']['placed_rect_pt'])
    frame_color = (126 / 255, 156 / 255, 162 / 255)
    spines = [d for d in target.get_drawings() if d['color'] is not None and np.allclose(d['color'], frame_color, atol=1e-06) and projection_rect.contains(d['rect']) and (d['rect'].width < 0.01) and (d['rect'].height > 80)]
    assert len(spines) == 2, len(spines)
    frame = pdf.Rect(min((d['rect'].x0 for d in spines)), min((d['rect'].y0 for d in spines)), max((d['rect'].x1 for d in spines)), max((d['rect'].y1 for d in spines)))
    assert frame.width > 100 and frame.height > 80
    frame_width = spines[0]['width']
    assert np.isclose(frame_width, spines[1]['width'])
    target.draw_rect(frame, color=frame_color, fill=None, width=frame_width)
    with pdf.open(OUT / 'figures/E10-embedded-analysis-plane.pdf') as plane_doc:
        plane_dest = pdf.Rect(237, 24, 398, 169.4193548387)
        target.show_pdf_page(plane_dest, plane_doc, 0)
        plane_scale = plane_dest.width / plane_doc[0].rect.width
        corners = np.array(embedded_plane['plane_corners_pdf_pt']) * plane_scale + [plane_dest.x0, plane_dest.y0]
        placements['anatomy'] = {'source': 'E10-embedded-analysis-plane.pdf', 'source_clip_pt': list(plane_doc[0].rect), 'placed_rect_pt': list(plane_dest), 'scale_xy': [plane_scale, plane_scale], 'plane_corners_pt': corners.tolist()}
    projection_border = [242.9, 242.1, 369.2, 333.2]
    plane_links = []
    endpoints = [[frame.x0, frame.y0], [frame.x1, frame.y0]] if COMPACT_GEOMETRY_FLOW else [[projection_border[0], projection_border[3]], [projection_border[2], projection_border[3]]]
    for point, endpoint in zip(corners[:2], endpoints):
        endpoint = np.array(endpoint)
        intervals = [(point[1], endpoint[1])] if COMPACT_GEOMETRY_FLOW else [(point[1], 175.0), (211.0, 218.0), (234.0, 239.0)]
        for ya, yb in intervals:

            def on_line(y):
                return point + (endpoint - point) * ((y - point[1]) / (endpoint[1] - point[1]))
            start, end = (on_line(ya), on_line(yb))
            target.draw_line(start.tolist(), end.tolist(), color=(0.55, 0.67, 0.7), width=0.65, dashes='[2.5 2] 0')
            plane_links.append([start.tolist(), end.tolist()])
    label('Early Projection', 303, 230, size=9, color=(0.25, 0.43, 0.48))
    xlabel_baseline = 380
    xlabel_widths = {}
    for text, x in (('Time (ms)', 116), ('shared TA axis (mm)', 516)):
        xlabel_widths[text] = label(text, x, xlabel_baseline, size=12)
        assert xlabel_widths[text] <= side
    arrow = [[200, 196.2], [386, 196.2], [386, 191], [396, 197], [386, 203], [386, 197.8], [200, 197.8]]
    if COMPACT_GEOMETRY_FLOW:
        arrow = [[377, 196.2], [419, 196.2], [419, 192], [427, 197], [419, 202], [419, 197.8], [377, 197.8]]
    target.draw_polyline(arrow, color=None, fill=(0.42, 0.49, 0.51), closePath=True)
    return (result, {'subplot_titles': titles, 'removed_labels': sorted(removed), 'placements': placements, 'data_axes_size_pt': [side, side], 'geometry_linear_scales': [plane_scale, 0.66], 'embedded_plane_links_pt': plane_links, 'projection_label': 'Early Projection', 'projection_bottom_clip_padding_source_pt': 2.0, 'projection_frame': {'rect_pt': list(frame), 'width_pt': frame_width, 'color': list(frame_color), 'complete_rectangle': True}, 'shared_xlabel_baseline_pt': xlabel_baseline, 'xlabel_widths_pt': xlabel_widths, 'font_sizes_pt': {'subplot_title': SUBPLOT_TITLE_SIZE_PT, 'axis_label': 12, 'ticks': 10, 'rank_label': 11, 'contact_label': 9.6, 'magnitude_label': 10}, 'conversion_annotation': None, 'conversion_annotation_removed': True, 'conversion_arrow_polygon_pt': arrow, 'geometry_flow_layout': 'short_arrow_right_of_unbroken_projection_guides' if COMPACT_GEOMETRY_FLOW else 'original', 'physical_data_geometry_changed': False})

def build():
    snapshot = freeze()
    figures = OUT / 'figures'
    figures.mkdir(exist_ok=True)
    record = json.loads((snapshot / 'template_fields.json').read_text())
    embedded_plane = build_embedded_plane(record, json.loads((snapshot / 'geometry_source_metadata.json').read_text()), figures / 'E10-embedded-analysis-plane')
    cmeta = json.loads((snapshot / 'fig2_panelc_metadata.json').read_text())
    layout = json.loads((snapshot / 'figure2_layout_metadata.json').read_text())
    assert record['interictal_field']['fingerprint_sha256'] == cmeta['frozen_fingerprint']
    a, b, mode = build_interictal_ab_panel_payloads(record)
    assert mode == 'shared'
    cache = np.load(snapshot / 'event_cache.npz', allow_pickle=False)
    assert list(cache['contact_order']) == a['names'] == b['names']
    np.testing.assert_allclose(cache['points_mm'], np.c_[a['xs'], a['ys']])
    xlim = np.array(cmeta['static']['display_xlim_mm'])
    ylim = np.array(cmeta['static']['display_ylim_mm'])
    doc = pdf.open(snapshot / 'fig2-panela.pdf')
    original_c = pdf.open(snapshot / 'original-panelc.pdf')
    original_d = pdf.open(snapshot / 'original-paneld.pdf')
    field_rects = sorted([pdf.Rect(im['bbox']) for im in original_d[0].get_image_info() if pdf.Rect(im['bbox']).x0 > 0 and abs(pdf.Rect(im['bbox']).width - pdf.Rect(im['bbox']).height) < 0.1], key=lambda r: r.y0)
    readout_rects = sorted([d['rect'] for d in original_c[0].get_drawings() if 45 < d['rect'].x0 < 55 and 65 < d['rect'].width < 83 and (75 < d['rect'].height < 110)], key=lambda r: r.y0)
    assert len(field_rects) == len(readout_rects) == 2
    trajectories = {}
    display_paths = {}
    display_path_metadata = {}
    fits = {}
    validation = {}
    for i, (lab, payload) in enumerate((('TA', a), ('TB', b))):
        points = np.c_[payload['xs'], payload['ys']]
        payload['frame'].update(xlim=xlim, ylim=ylim)
        X, Y, T, S, mask = _smooth_rank_field_mm(*points.T, payload['vals'], payload['sup'], xlim, ylim, 6.0)
        floor = VIS_MASK_REL * float(np.max(S))
        kwargs = dict(sigma_mm=6.0, support_floor=floor)
        trajectory = rank_gradient_trajectory(points, payload['vals'], payload['sup'], **kwargs)
        refined = rank_gradient_trajectory(points, payload['vals'], payload['sup'], step_mm=0.04, **kwargs)
        error = max(cKDTree(refined['points_mm']).query(trajectory['points_mm'])[0].max(), cKDTree(trajectory['points_mm']).query(refined['points_mm'])[0].max())
        sample = [(j, k) for j in range(0, T.shape[0], 11) for k in range(0, T.shape[1], 11) if mask[j, k]]
        field_error = max((abs(gaussian_rank_value_gradient([X[j, k], Y[j, k]], points, payload['vals'], payload['sup'], 6)[0] - T[j, k]) for j, k in sample))
        sensitivity = []
        for delta in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
            alternative = rank_gradient_trajectory(points, payload['vals'], payload['sup'], seed=trajectory['seed_mm'] + delta, **kwargs)
            displacement = alternative['points_mm'][-1] - alternative['points_mm'][0]
            sensitivity.append({'seed_offset_mm': delta, 'endpoint_displacement_mm': displacement.tolist(), 'max_departure_from_chord_mm': alternative['max_departure_from_endpoint_chord_mm']})
        validation[lab] = {'n_contacts': len(points), 'rank_strictly_increases': bool(np.all(np.diff(trajectory['rank01']) > 0)), 'all_points_in_display_support': bool(np.all(trajectory['support_mass'] >= floor)), 'integrator_half_step_max_distance_mm': float(error), 'canonical_field_max_abs_error': float(field_error), 'geometry_seed_sensitivity_1mm': sensitivity}
        assert validation[lab]['rank_strictly_increases'] and field_error < 1e-12 and (error < 0.09)
        trajectories[lab] = trajectory
        display_paths[lab], display_path_metadata[lab] = display_trajectory(trajectory, points)
        mapped = spatial_pdf_points(display_paths[lab], field_rects[i], xlim, ylim)
        colored_path(doc[0], in_composite(mapped, layout['panel_a_layout'][f'field_{lab.lower()}']), TRAJECTORY_COLORS[lab])
        assert int(cache[f'{lab}_event_pos']) == cmeta['exemplar'][lab]['event_pos']
        time = cache[f'{lab}_fig1a_centroid_ms']
        usable = cache[f'{lab}_participant'] & np.isfinite(time)
        ordered = np.argsort(points[:, 0])
        selected = ordered[usable[ordered]]
        rows = np.arange(len(selected)) + 0.5
        slope, intercept = np.polyfit(rows, time[selected], 1)
        ends = np.array([1.5, len(selected) - 1.5])
        if slope < 0:
            ends = ends[::-1]
        rr = np.linspace(*ends, 80)
        fitted = intercept + slope * rr + READOUT_ARROW_SHIFT_MS
        r = readout_rects[i]
        time_range = cmeta['static']['readout_xlim_ms']
        source_line = np.c_[r.x0 + (fitted - time_range[0]) / np.ptp(time_range) * r.width, r.y0 + rr / len(selected) * r.height]
        mapped_line = in_composite(source_line, layout['panel_a_layout'][f'sequence_{lab.lower()}'])
        arrow_geometry = straight_direction_arrow(doc[0], mapped_line[0], mapped_line[-1])
        residual = time[selected] - (intercept + slope * rows)
        fits[lab] = {'event_pos': int(cache[f'{lab}_event_pos']), 'contact_names_in_display_order': [a['names'][j] for j in selected], 'centroid_ms': time[selected].tolist(), 'slope_ms_per_display_contact': float(slope), 'intercept_ms': float(intercept), 'r_squared': float(1 - residual @ residual / np.sum((time[selected] - time[selected].mean()) ** 2)), 'arrow_direction': 'increasing time', 'fit_level': 'displayed individual event', 'display_only_horizontal_shift_ms': READOUT_ARROW_SHIFT_MS, 'inset_from_first_last_contact_centers_rows': 1.0, 'arrow_geometry': arrow_geometry}
        with (OUT / f'E10_{lab.lower()}_trajectory.csv').open('w') as handle:
            writer = csv.writer(handle)
            writer.writerow(['arc_length_mm', 'x_mm', 'y_mm', 'interpolated_rank01', 'support_mass'])
            writer.writerows(zip(trajectory['arc_length_mm'], *trajectory['points_mm'].T, trajectory['rank01'], trajectory['support_mass']))
    doc, labeling = representation_layout(doc, layout['panel_a_layout'], field_rects, readout_rects, cmeta['static'], embedded_plane)
    exports = export(doc, figures / 'fig2-panela-trajectories', 600)
    doc[0].get_pixmap(matrix=pdf.Matrix(2.5, 2.5), alpha=False).save(figures / 'fig2-panela-trajectories-preview.png')
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42}):
        fig = plt.figure(figsize=(7.6, 3.8))
        ax_a = fig.add_axes([0.095, 0.19, 0.32, 0.71])
        ax_b = fig.add_axes([0.565, 0.19, 0.32, 0.71])
        for ax, lab, payload in [(ax_a, 'TA', a), (ax_b, 'TB', b)]:
            draw_interictal_rank_field_panel(ax, payload, lab, contact_size=34, contact_outline_lw=1)
            mpl_path(ax, display_paths[lab], TRAJECTORY_COLORS[lab])
            ax.set_title(f'{lab} {TRAJECTORY_TITLE_SUFFIX}', fontsize=SUBPLOT_TITLE_SIZE_PT, color=TEMPLATE_COLORS[lab], fontweight='bold', pad=10)
            ax.set_xlabel('')
            ax.set_ylabel('y (mm)', fontsize=12)
            ax.set_xticks([-20, 0, 20])
            ax.set_yticks([-10, 0, 10, 20])
            ax.tick_params(labelsize=10)
        pos = ax_b.get_position()
        cax = fig.add_axes([0.92, pos.y0, 0.018, pos.height])
        cb = fig.colorbar(ax_b.images[0], cax=cax)
        cb.set_ticks([0, 0.5, 1], labels=['0 early', '0.5', '1 late'])
        cb.ax.set_title('rank', fontsize=11, loc='left')
        fig.text(0.49, 0.075, 'shared TA axis (mm)', ha='center', fontsize=12)
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(figures / f'E10-spatial-rank-trajectories.{ext}', dpi=450, facecolor='white', bbox_inches='tight', pad_inches=0.08)
        plt.close(fig)
    metadata = {'schema_version': 'fig2_e10_rank_gradient_trajectory_v18', 'status': 'CANDIDATE_PENDING_AUTHOR_VISUAL_REVIEW', 'subject': 'E10', 'frozen_fingerprint': cmeta['frozen_fingerprint'], 'definition': 'd position / d arc_length = gradient(interpolated rank) / norm(gradient)', 'seed_rule': 'same rank-independent sampling-convex-hull area centroid for both templates', 'display_sigma_mm': 6, 'sampling_hull_constrained': True, 'interpretation': 'One descriptive gradient streamline per frozen template; not a unique physical propagation track, velocity, or measured tissue wave.', 'left_arrow_definition': 'OLS of displayed event centroid time on displayed contact-row index; arrow points towards increasing time.', 'arrow_style': {'sequence_color': SEQUENCE_ARROW_COLOR, 'sequence_outline': SEQUENCE_ARROW_OUTLINE, 'sequence_single_filled_polygon': True, 'sequence_horizontal_shift_ms_display_only': READOUT_ARROW_SHIFT_MS, 'sequence_inset_contact_rows': 1.0, 'trajectory_colors': TRAJECTORY_COLORS, 'trajectory_dark_outline': False, 'trajectory_shape': 'curved_ribbon_joined_to_rigid_flat_terminal_triangle', 'trajectory_head_uses_one_fixed_normal': True, 'trajectory_core_and_halo_each_single_polygon': True, 'trajectory_separate_head_patches': False, 'trajectory_pdf_source_linewidth_pt': 1.3, 'trajectory_outer_halo': TRAJECTORY_HALO, 'trajectory_pdf_source_halo_extra_width_pt': 2.0, 'trajectory_pdf_source_head_base_arc_from_end_pt': 7.0, 'trajectory_pdf_source_outer_head_width_pt': 5.4, 'trajectory_pdf_source_core_head_width_pt': 3.0, 'trajectory_pdf_source_core_tip_inset_pt': 0.8, 'trajectory_zoom_style_scale': 1.25, 'trajectory_arrow_fractions': TRAJECTORY_ARROW_FRACTIONS, 'trajectory_outer_tip_at_displayed_endpoint': True}, 'trajectory_display': display_path_metadata, 'spatial_palette': {'name': 'viridis', 'frozen_source_field_and_colorbar_restored': True, 'contact_colors_restored': True, 'rank_normalization_changed': False}, 'representation_labels': labeling, 'embedded_analysis_plane': embedded_plane, 'left_and_right_statistical_units': {'left': 'one existing event per template', 'right': 'frozen population template'}, 'trajectories': trajectories, 'sequence_fits': fits, 'validation': validation, 'canonical_and_complete_figure_replaced': False, 'other_patients_changed': False, 'outputs': exports}
    (OUT / 'trajectory_metadata.json').write_text(json.dumps(metadata, default=plain, ensure_ascii=False, indent=2) + '\n')
    (figures / 'README.md').write_text('# Figure 2A：E10 方向轨迹候选\n\n### fig2-panela-trajectories.png / .pdf / .svg\n\n四张数据图使用相同的132 × 132 pt轴框，上下两行分别严格对齐，短标签为TA/TB seq.与TA/TB traj.；Time (ms)与shared TA axis (mm)共用底部文字基线。中间上方使用同一患者T1与真实电极坐标，在三维MRI中嵌入冻结的二维分析平面，平面上两条白色箭头表示原三维TA/TB方向，浅虚线提示下方平面的展开关系，下方标签为Early Projection；下边框补足原先被裁掉的线宽，使四边完整显示。二维场、触点填色与色标保留原viridis顺序色系；左侧方向箭头改为带白色细边的浅玫红（#EF8FB3），将TA／TB空间流线改为明亮珊瑚红／晴蓝线芯和加粗的纯白外沿，移除深色描边，曲线线身接入固定朝向的平面尖端，尖端使用同一组方向与法向量，不随曲线逐点旋转，去掉中段箭头；彩色线芯和白色外沿各自为一个完整轮廓，末端不再叠加独立三角箭头；为避免流线看起来接在电极上，显示端点沿原曲线截短至与触点中心至少相距2 mm。\n**关注点**：截短仅用于避让电极符号，完整积分曲线及其凸包边界终止原因仍保存在CSV和metadata中，不能把显示末端解释为生物学传播终点。\n\n### E10-embedded-analysis-plane.png / .pdf / .svg\n\n展示E10配准后MNI152 1 mm空间中的局部MRI、ICL/SCL电极及真实斜切面，切面纹理直接采样同一患者T1，采用与下方展开图呼应的浅青色，保留30%不透明度；三张正交MRI切片统一在冻结分析平面原点相交，按原始LAS（左、前、上）方向使用同一正交相机显示，三张MRI组织切片完全不透明；仅分析平面保留30%不透明度，完整青色矢量边框加粗至2.1 pt，取消额外交线。物理RAS坐标到LAS仅作显示轴转换，原始影像和电极坐标不变。原三视图交点相对分析原点偏离约3.69 mm的问题已修正；电极、分析平面与TA/TB方向未平移或重拟合；移除R/A/S坐标标记与中间“+ spatial information”文字，保留向右转换箭头。切面的位置与基向量和右侧二维场完全相同。两条白箭头来自冻结三维early→late方向，分别从各自模板的p_early出发，不再共用起点；末端仅为避让触点沿原轴截短至至少2.5 mm距离。p_early为拟合轴上的模板早端摘要，不是已定位的生物学传播源。\n**关注点**：该图不把左侧时间—通道斜率当作三维角度；不重新拟合平面或改变电极坐标，沿用原bundle的grid_confirmed_warp_type_unverified边界。\n\n### E10-spatial-rank-trajectories.png / .pdf / .svg\n\n并排放大二维场，两图上方分别写TA traj.、TB traj.，纵轴仅写y (mm)，使用与A相同的显示曲线段及一体化细长尖端样式。底图、触点与色标保留原viridis顺序映射，仅将TA／TB轨迹换为明亮珊瑚红／晴蓝线芯及加粗纯白外沿，取消深色描边与中段箭头，线身连续接到固定朝向的平面尖端，取消尖部沿流线逐点旋转的绘制方式；曲线仍由原6 mm高斯平滑rank场的归一化解析梯度积分得到。\n**关注点**：曲线为同一采样凸包几何中心出发的描述性gradient streamline，不能解释为唯一轨迹、传播速度或直接测得的组织波前。\n\n### fig2-panela-trajectories-preview.png\n\nA 面板的快速预览。完整导出保留原矢量内容及新增矢量箭头，候选尚待作者目视验收。\n**关注点**：检查浅玫红拟合箭头、红蓝曲线的平面尖端和纯白外沿在整图缩小后的可读性，并与原通道质心、早晚色标一起读。\n\n数据与验收：同级 `embedded_plane_metadata.json` 保存三维切面、T1来源和白箭头坐标；`trajectory_metadata.json` 保存定义、全部曲线点、1D 拟合参数、fingerprint、步长检查及几何 seed ±1 mm 敏感性；`E10_ta_trajectory.csv` / `E10_tb_trajectory.csv` 保存物理毫米坐标及沿线 rank。原始输入冻结在 `source_snapshot/`，producer 为 `scripts/paper_figures/plot_fig2_exemplar_trajectory.py`，三维切面 helper 为 `scripts/paper_figures/fig2_embedded_plane.py`，依赖 NumPy、SciPy、Matplotlib、PyMuPDF、NiBabel。原完整 Figure 2 与其他患者图不受影响。\n', encoding='utf-8')
    print(json.dumps({'output': str(figures), 'validation': validation, 'sequence_fits': fits}, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    build()
