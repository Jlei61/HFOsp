"""Show the frozen E10 analysis plane in its registered 3-D T1 geometry."""
from __future__ import annotations
from paper_paths import CODE_ROOT, INPUT_ROOT, OUTPUT_ROOT
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
import nibabel as nib
import numpy as np
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plotting.vendor.scripts.paper_figures.plot_fig2_e1146_template_projection_composite import MNI_BUNDLE, T1_PATH, _basis, _canonical_transverse_sign, _load_t1, _local_geometry, _limits_2d, _sample_t1, _shaft_indices, _t1_display_range

def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
VIEW_ELEVATION = 22
VIEW_AZIMUTH = 45
DISPLAY_AXIS_SIGNS = np.array([-1.0, 1.0, 1.0])
PLANE_EDGE_COLOR = '#4F8697'
PLANE_EDGE_WIDTH_PT = 2.1

def _add_depth_sorted_t1_scene(ax, image, data, basis, corners, xlim, ylim, limits_min, limits_max, display_range, dpi=450):
    """Ray-sample four real planes and alpha-composite by depth at each pixel."""
    center = basis['origin']
    planes = []
    for k in range(3):
        ij = [i for i in range(3) if i != k]
        planes.append((f'mri_{k}', np.eye(3)[ij[0]], np.eye(3)[ij[1]], (limits_min[ij[0]] - center[ij[0]], limits_max[ij[0]] - center[ij[0]]), (limits_min[ij[1]] - center[ij[1]], limits_max[ij[1]] - center[ij[1]])))
    planes.append(('analysis', basis['u'], basis['w'], xlim, ylim))
    screen = ax.bbox.get_points()
    q = ax.transData.inverted().transform(screen)
    extent = [q[0, 0], q[1, 0], q[0, 1], q[1, 1]]
    width, height = np.ceil(np.ptp(screen, axis=0) * dpi / ax.figure.dpi).astype(int)
    px = extent[0] + (np.arange(width) + 0.5) / width * (extent[1] - extent[0])
    py = extent[3] - (np.arange(height) + 0.5) / height * (extent[3] - extent[2])
    xx, yy = np.meshgrid(px, py)
    rays = np.stack([xx, yy, np.ones_like(xx)], -1)
    matrix = ax.get_proj()
    colors, alphas, depths, corner_errors, mri_alpha_values = ([], [], [], [], [])
    for name, u, w, xbounds, ybounds in planes:
        basis_h = np.vstack([np.column_stack([u, w, center]), [0.0, 0.0, 1.0]])
        camera_h = matrix @ basis_h
        homography = camera_h[[0, 1, 3]]
        local_h = rays @ np.linalg.inv(homography).T
        local = local_h[..., :2] / local_h[..., 2, None]
        edge_distance = np.minimum.reduce([local[..., 0] - xbounds[0], xbounds[1] - local[..., 0], local[..., 1] - ybounds[0], ybounds[1] - local[..., 1]])
        inside = edge_distance >= 0
        world = center + local[..., 0, None] * u + local[..., 1, None] * w
        values = _sample_t1(image, data, world)
        intensity = np.clip((values - display_range[0]) / np.ptp(display_range), 0, 1) ** 0.78
        projected = np.concatenate([local, np.ones((*local.shape[:2], 1))], -1) @ camera_h.T
        depth = projected[..., 2] / projected[..., 3]
        if name == 'analysis':
            low = np.array(matplotlib.colors.to_rgb('#BCD3D7'))
            high = np.array(matplotlib.colors.to_rgb('#F6FAF9'))
            color = low + intensity[..., None] * (high - low)
            alpha = np.where(inside, 0.3, 0.0)
        else:
            color = np.repeat(intensity[..., None], 3, axis=-1)
            alpha = np.where(inside & (values > 5), 1.0, 0.0)
            mri_alpha_values.append(np.unique(alpha).tolist())
        colors.append(color.astype(np.float32))
        alphas.append(alpha.astype(np.float32))
        depths.append(np.where(alpha > 0, depth, np.inf).astype(np.float32))
        local_corners = np.array([[xbounds[0], ybounds[0]], [xbounds[1], ybounds[0]], [xbounds[1], ybounds[1]], [xbounds[0], ybounds[1]]])
        world_corners = center + local_corners[:, :1] * u + local_corners[:, 1:] * w
        qx, qy, _ = proj3d.proj_transform(*world_corners.T, matrix)
        check = np.c_[local_corners, np.ones(4)] @ homography.T
        error = float(np.max(np.abs(check[:, :2] / check[:, 2, None] - np.c_[qx, qy])))
        assert error < 1e-12
        corner_errors.append(error)
    elev, azim = np.deg2rad([VIEW_ELEVATION, VIEW_AZIMUTH])
    toward_camera = DISPLAY_AXIS_SIGNS * np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
    _, _, near_far_z = proj3d.proj_transform(*(center + np.array([5.0, -5.0])[:, None] * toward_camera).T, matrix)
    assert near_far_z[0] < near_far_z[1]
    colors, alphas, depths = (np.array(colors), np.array(alphas), np.array(depths))
    order = np.argsort(depths, axis=0)[::-1]
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    alpha_out = np.zeros((height, width), dtype=np.float32)
    for layer in order:
        alpha = np.take_along_axis(alphas, layer[None], axis=0)[0]
        color = np.take_along_axis(colors, np.broadcast_to(layer[None, ..., None], (1, height, width, 3)), axis=0)[0]
        rgb = rgb * (1 - alpha[..., None]) + color * alpha[..., None]
        alpha_out = alpha_out * (1 - alpha) + alpha
    rgba = np.dstack([np.divide(rgb, alpha_out[..., None], out=np.zeros_like(rgb), where=alpha_out[..., None] > 0), alpha_out])
    texture = AxesImage(ax, origin='upper', extent=extent, interpolation='bilinear', zorder=1)
    texture.set_data(rgba)
    ax.add_image(texture)
    return {'opacity': 0.3, 'mri_opacity': 1.0, 'texture_size_px': [int(width), int(height)], 'texture_dpi': dpi, 'compositing': 'per_pixel_back_to_front_physical_depth', 'mri_layer_alpha_values': mri_alpha_values, 'analysis_border_full_vector_outline': True, 'analysis_border_width_pt': PLANE_EDGE_WIDTH_PT, 'all_planes_share_camera_and_origin': True, 'true_analysis_mri_intersection_traces': False, 'camera_projection_corner_max_abs_error': max(corner_errors), 'projection_matrix_world_to_clip': matrix.tolist(), 'near_and_far_projected_depth_check': near_far_z.tolist()}

def build_embedded_plane(record: dict, source_meta: dict, output: Path) -> dict:
    """Display-only addition; use the exact existing plane, axes, and contacts."""
    sign = _canonical_transverse_sign(record['interictal_field']['planes']['shared']['w'])
    basis = _basis(record, sign)
    previous = source_meta['projection_plane']
    for key, old in [('u', 'u'), ('w', 'w_after_display_sign'), ('normal', 'normal'), ('origin', 'origin_mm')]:
        np.testing.assert_allclose(basis[key], previous[old], atol=1e-12)
    coords_file = MNI_BUNDLE / 'epilepsiae_1146_electrodes_mni152_1mm.csv'
    with coords_file.open() as handle:
        rows = list(csv.DictReader(handle))
    names = [row['electrode_name'] for row in rows]
    coords = np.array([[float(row[k]) for k in ['mni_x_mm', 'mni_y_mm', 'mni_z_mm']] for row in rows])
    geometry = _local_geometry(record, basis, names, coords)
    for name, xyz in zip(record['names'], record['coords'], strict=True):
        np.testing.assert_allclose(coords[names.index(name)], xyz, atol=1e-12)
    image, data, manifest = _load_t1()
    assert _sha(T1_PATH) == source_meta['anatomical_context']['source_t1_sha256']
    measured = np.asarray(geometry['measured'])
    world = np.asarray(geometry['world'])
    selected = np.asarray(geometry['selected'])
    shaft_names = np.asarray(geometry['shafts'])
    contact_names = np.asarray(geometry['names'])
    xlim, ylim = _limits_2d(measured[:, :2], pad_fraction=0.24)

    def to_world(xy):
        xy = np.asarray(xy)
        return basis['origin'] + xy[..., 0, None] * basis['u'] + xy[..., 1, None] * basis['w']
    corners_xy = np.array([[xlim[0], ylim[0]], [xlim[1], ylim[0]], [xlim[1], ylim[1]], [xlim[0], ylim[1]]])
    corners = to_world(corners_xy)
    center = basis['origin'].copy()
    half_extent = np.max(np.abs(np.vstack([world, corners]) - center), axis=0) + [7, 11, 7]
    limits_min, limits_max = (center - half_extent, center + half_extent)
    display_range = _t1_display_range(data)
    directions = {}
    with plt.rc_context({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42}):
        fig = plt.figure(figsize=(3.1, 2.8), facecolor='white')
        ax = fig.add_axes([0, 0, 1, 1], projection='3d', computed_zorder=False)
        ax.plot(*np.vstack([corners, corners[0]]).T, color=PLANE_EDGE_COLOR, lw=PLANE_EDGE_WIDTH_PT, solid_capstyle='round', zorder=4)
        for lab, key in [('TA', 'a'), ('TB', 'b')]:
            axis = record['axis_pair'][f'axis_{key}']
            vector = np.array(axis['u'])
            normal_component = float(vector @ basis['normal'])
            assert abs(normal_component) < 1e-10
            direction = vector @ np.column_stack([basis['u'], basis['w']])
            direction /= np.linalg.norm(direction)
            normal = np.array([-direction[1], direction[0]])
            poles_world = np.array([axis['p_early'], axis['p_late']])
            poles_xy = (poles_world - basis['origin']) @ np.c_[basis['u'], basis['w']]
            np.testing.assert_allclose(to_world(poles_xy), poles_world, atol=1e-10)
            glyph_origin = poles_xy[0]
            length = np.linalg.norm(poles_xy[1] - poles_xy[0])
            np.testing.assert_allclose((poles_xy[1] - poles_xy[0]) / length, direction, atol=1e-10)
            positions = np.linspace(0, length, 500)
            segment = glyph_origin + positions[:, None] * direction
            clearance = np.linalg.norm(segment[:, None, :] - measured[None, :, :2], axis=-1).min(axis=1)
            last = np.flatnonzero(clearance >= 2.5)[-1]
            tip = segment[last]
            neck = tip - 3.5 * direction
            polygon = np.array([glyph_origin + 0.34 * normal, neck + 0.34 * normal, neck + 1.25 * normal, tip, neck - 1.25 * normal, neck - 0.34 * normal, glyph_origin - 0.34 * normal])
            ax.add_collection3d(Poly3DCollection([to_world(polygon)], facecolors='white', edgecolors='#6D929A', linewidths=0.4, zorder=8))
            label_xy = glyph_origin + 0.18 * (tip - glyph_origin) + np.array([0, 3.5])
            ax.text(*to_world(label_xy), lab, color='#386875', fontsize=10, fontweight='bold', ha='center', va='center', zorder=10)
            directions[lab] = {'frozen_early_to_late_unit_vector_ras': vector.tolist(), 'in_plane_unit_vector': direction.tolist(), 'normal_component': normal_component, 'glyph_origin_xy_mm': glyph_origin.tolist(), 'glyph_tip_xy_mm': tip.tolist(), 'frozen_p_early_ras_mm': poles_world[0].tolist(), 'frozen_p_late_ras_mm': poles_world[1].tolist(), 'origin_source': f'axis_pair.axis_{key}.p_early', 'late_end_display_trim_mm': float(length - positions[last]), 'displayed_tip_contact_clearance_mm': float(clearance[last])}
        for idx in _shaft_indices(contact_names, shaft_names):
            fitted = geometry['world_fitted'][idx]
            ax.plot(*fitted.T, color='#24444C', lw=1.2, zorder=9)
            anchor = idx[np.argmin(measured[idx, 0])]
            shaft = str(shaft_names[anchor])
            offset = [1.0, 3.8] if shaft == 'ICL' else [-2.1, 0.7]
            label_xy = measured[anchor, :2] + offset
            ax.text(*to_world(label_xy), shaft_names[anchor], fontsize=8.5, color='#244F59', ha='left' if shaft == 'ICL' else 'right', va='center', fontweight='bold', zorder=11)
        ax.scatter(*world[~selected].T, s=17, facecolor='#D9DEE0', edgecolor='#526C74', linewidth=0.6, depthshade=False, zorder=10)
        ax.scatter(*world[selected].T, s=25, facecolor='#25818E', edgecolor='white', linewidth=0.8, depthshade=False, zorder=10)
        ax.set(xlim=(limits_min[0], limits_max[0]), ylim=(limits_min[1], limits_max[1]), zlim=(limits_min[2], limits_max[2]))
        ax.set_box_aspect(limits_max - limits_min, zoom=1.28)
        ax.invert_xaxis()
        ax.set_proj_type('ortho')
        ax.view_init(elev=VIEW_ELEVATION, azim=VIEW_AZIMUTH, roll=0, vertical_axis='z')
        ax.set_axis_off()
        fig.canvas.draw()
        transparency = _add_depth_sorted_t1_scene(ax, image, data, basis, corners, xlim, ylim, limits_min, limits_max, display_range)
        qx, qy, _ = proj3d.proj_transform(*corners.T, ax.get_proj())
        corners_px = ax.transData.transform(np.c_[qx, qy])
        corners_pt = corners_px * 72 / fig.dpi
        corners_pt[:, 1] = fig.get_figheight() * 72 - corners_pt[:, 1]
        cx, cy, _ = proj3d.proj_transform(*center, ax.get_proj())
        center_pt = ax.transData.transform([cx, cy]) * 72 / fig.dpi
        center_pt[1] = fig.get_figheight() * 72 - center_pt[1]
        for ext in ['png', 'pdf', 'svg']:
            fig.savefig(output.with_suffix(f'.{ext}'), dpi=450, facecolor='white')
        plt.close(fig)
    metadata = {'subject': 'E10', 'coordinate_space': 'mni152_1mm', 'world_convention': 'RAS+', 'input_fingerprint_sha256': record['interictal_field']['fingerprint_sha256'], 'source_t1_sha256': _sha(T1_PATH), 'source_coordinates_sha256': _sha(coords_file), 'normalization_certainty': manifest['normalization_certainty'], 'plane_basis': {k: v.tolist() for k, v in basis.items()}, 'plane_xlim_mm': xlim, 'plane_ylim_mm': ylim, 'plane_corners_xy_mm': corners_xy.tolist(), 'plane_corners_world_mm': corners.tolist(), 'plane_corners_pdf_pt': corners_pt.tolist(), 'canvas_pt': [223.2, 201.6], 'direction_glyphs': directions, 'plane_directions_are_frozen_3d_axes': True, 'plane_normal_offset_mm': 0, 'normal_exaggeration': 1, 'view_elevation_deg': VIEW_ELEVATION, 'view_azimuth_deg': VIEW_AZIMUTH, 'camera_projection': 'orthographic', 'view_roll_deg': 0, 'vertical_display_axis': 'S', 'view_side': 'left_anterior_superior_in_LAS', 'native_voxel_axis_codes': list(nib.aff2axcodes(image.affine)), 'display_axis_codes': ['L', 'A', 'S'], 'world_ras_to_display_las_axis_signs': DISPLAY_AXIS_SIGNS.tolist(), 'anatomy_slice_center_ras_mm': center.tolist(), 'anatomy_center_matches_analysis_origin': True, 'anatomy_slice_normals_ras': np.eye(3).tolist(), 'shared_four_plane_intersection_pdf_pt': center_pt.tolist(), 'previous_anatomy_center_offset_mm': float(np.linalg.norm(world.mean(axis=0) - center)), 'scene_bounds_ras_mm': [limits_min.tolist(), limits_max.tolist()], 'arrow_origins_are_template_specific_frozen_early_poles': True, 'arrow_origins_are_localized_biological_sources': False, 'previous_direction_display_code': 'scripts/paper_figures/plot_fig2_template_projection_schematic.py:240-251', 'oblique_slice_samples_real_t1': True, 'display_style': {'orientation_triad_shown': False, 'anatomy_uses_native_t1_intensity': True, 'anatomy_opacity': 1.0, 'plane_transparency': transparency, 't1_display_range': list(display_range), 't1_display_gamma': 0.78, 'plane_tint_low': '#BCD3D7', 'plane_tint_high': '#F6FAF9', 'plane_edge': PLANE_EDGE_COLOR, 'plane_edge_width_pt': PLANE_EDGE_WIDTH_PT, 'shaft_label_offsets_local_mm': {'ICL': [1.0, 3.8], 'SCL': [-2.1, 0.7]}, 'white_arrow_edge': '#6D929A', 'white_arrow_edge_width_pt': 0.4, 'direction_label_color': '#386875'}, 'selected_contact_plane_residual_max_abs_mm': float(np.max(np.abs(measured[selected, 2]))), 'meaning': 'Frozen shared analysis plane embedded in registered 3-D T1; white arrows are frozen 3-D template directions, not the 1-D time/contact slopes or measured tissue trajectories.'}
    (output.parent.parent / 'embedded_plane_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata
