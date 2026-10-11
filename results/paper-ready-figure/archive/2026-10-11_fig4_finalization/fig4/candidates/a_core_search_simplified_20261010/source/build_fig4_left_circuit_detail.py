#!/usr/bin/env python3
"""Render the two requested circuit details through the frozen native producer."""
from pathlib import Path
import importlib.util
import io
import json
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/paper-ready-figure/fig4/candidates'
OUT = BASE / 'a_left_circuit_spacing_20261010'
SOURCE = OUT / 'source'


def render_circuit(native, **detail):
    # Match the archived native export, crop, placement and scale bar exactly.
    # This rerenders plotting primitives; no existing artwork is painted over.
    with plt.rc_context(plt.rcParamsDefault):
        fig, ax = plt.subplots(figsize=(5.2, 4.15), facecolor='white')
        fig.subplots_adjust(left=.02, right=.98, bottom=.01, top=.92)
        native._draw_integrated_mechanism(ax, **detail)
        for text in ax.texts:
            if text.get_text() == 'z↓':
                text.set_position((5.61, 2.82)); text.set_fontsize(13.)
            else:
                text.set_fontsize(max(14., 1.35 * text.get_fontsize()))
        ax.set_title('Local E/I circuit', fontsize=20, fontweight='bold', pad=-2)
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=600, facecolor='white',
                    bbox_inches='tight', pad_inches=.045)
        plt.close(fig)
    original = Image.open(buffer).convert('RGB')
    circuit = original.crop((187, 0, 2773, 2218))
    ImageDraw.Draw(circuit).rectangle((0, 0, circuit.width, 250), fill='white')
    scale = min(2540 / circuit.width, 2070 / circuit.height)
    size = (round(circuit.width * scale), round(circuit.height * scale))
    origin = (120 + (2540 - size[0]) // 2, 470 + (2070 - size[1]) // 2)
    canvas = Image.new('RGB', (6000, 3000), 'white')
    canvas.paste(circuit.resize(size, Image.Resampling.LANCZOS), origin)
    frame = (origin[0] - 30, origin[1] + 160,
             origin[0] + size[0] + 30, origin[1] + size[1] + 30)
    # Frozen combined-panel scale calibration and typography.
    diameter_mm = 2 * .380 * np.sqrt(2.)
    bar_px = round(.5 / diameter_mm * 6 * (4.720625 * 600 / 10) * size[0] / circuit.width)
    x1, y = frame[2] - 95, frame[3] - 105
    x0 = x1 - bar_px
    draw = ImageDraw.Draw(canvas)
    draw.line((x0, y, x1, y), fill='black', width=13)
    for x in [x0, x1]:
        draw.line((x, y - 17, x, y + 17), fill='black', width=13)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 82)
    draw.text(((x0 + x1) // 2, y - 58), '0.5 mm', font=font, fill='black', anchor='mm')
    return np.asarray(canvas)[636:2564, 159:2620].copy()


def main():
    spec = importlib.util.spec_from_file_location('frozen_local_circuit', SOURCE / 'local_circuit_native_source.py')
    native = importlib.util.module_from_spec(spec)
    previous_bytecode_setting = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(native)
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
    previous = BASE / 'a_local_sampling_sigma_20261010/source/legacy_a_components.npz'
    with np.load(previous) as data:
        old = {key: data[key] for key in data.files}
    baseline = render_circuit(native)
    np.testing.assert_array_equal(baseline, old['image_0'])
    params = json.loads((SOURCE / 'mechanism_source.json').read_text())['left_circuit_detail_revision']
    revised = render_circuit(native,
        central_offset=params['central_offset_schematic_units'],
        z_linewidth=params['z_border_linewidth_pt'],
        z_dashes=tuple(params['z_border_dash_pattern']))
    changed = np.any(revised != baseline, axis=2)
    ys, xs = np.where(changed)
    bounds = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
    # Native E/I connection and z node occupy only this small interior region.
    allowed = np.zeros(changed.shape, bool)
    allowed[800:1520, 1050:1500] = True
    assert np.any(changed) and not np.any(changed & ~allowed), bounds
    old['image_0'] = revised
    np.savez_compressed(SOURCE / 'legacy_a_components.npz', **old)
    record = dict(native_baseline_pixel_identical_to_v9=True,
        changed_pixels=int(changed.sum()), changed_bounds_pixels=bounds,
        allowed_change_bounds_pixels=[1050, 800, 1500, 1520],
        unchanged_pixels_outside_requested_detail_region=True,
        previous_shaft_separation_schematic_units=.08,
        new_shaft_separation_schematic_units=2 * params['central_offset_schematic_units'],
        parameters=params)
    (SOURCE / 'left_circuit_detail_validation.json').write_text(json.dumps(record, indent=2) + '\n')
    shutil.copy2(__file__, SOURCE / Path(__file__).name)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
