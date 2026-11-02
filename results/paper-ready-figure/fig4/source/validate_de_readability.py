"""Check D/E source conservation and preservation of all other Fig4 panels."""
from pathlib import Path
import hashlib
import importlib.util
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path('/home/honglab/leijiaxin/HFOsp')
OUT = Path(__file__).resolve().parents[1]
PARENT = OUT.parent / 'snn_layout_order_label_20261027'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(package, name):
    spec = importlib.util.spec_from_file_location(name, package/'source/render_native_de.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fig = plt.figure(figsize=(320/25.4, 232.52/25.4))
    result = module.render_de(fig, (9/320, 71.05/232.52, 144/320, 78/232.52),
                              (171/320, 71.05/232.52, 144/320, 78/232.52))
    fig.canvas.draw()
    return fig, result


before = json.loads((OUT/'qa/input_hashes_before.json').read_text())
changed_inputs = [name for name, digest in before.items() if sha(ROOT/name) != digest]
assert not changed_inputs, changed_inputs
numeric_files = [p for p in (PARENT/'source').rglob('*') if p.suffix in ('.npz', '.csv')]
assert all(sha(p) == sha(OUT/'source'/p.relative_to(PARENT/'source')) for p in numeric_files)

old_fig, old = render(PARENT, 'old_de')
new_fig, new = render(OUT, 'new_de')
assert old['raster_neuron_ids'] == new['raster_neuron_ids']
assert old['window_ms'] == new['window_ms'] == [1500., 2900.]
assert old['lfp_ylim'] == new['lfp_ylim']
samples = {}
for arm in 'DE':
    old_axes, new_axes = old['axes'][arm], new['axes'][arm]
    for kind in ('field', 'lfp', 'raster'):
        a, b = old_axes[kind], new_axes[kind]
        assert a.get_xlim() == b.get_xlim() and a.get_ylim() == b.get_ylim()
        assert len(a.collections) == len(b.collections)
        for left, right in zip(a.collections, b.collections):
            np.testing.assert_array_equal(left.get_offsets(), right.get_offsets())
        assert len(a.lines) == len(b.lines)
        for left, right in zip(a.lines, b.lines):
            np.testing.assert_array_equal(left.get_xdata(), right.get_xdata())
            np.testing.assert_array_equal(left.get_ydata(), right.get_ydata())
    ax = new_axes['lfp']
    ticks = [float(t) for t in ax.get_yticks() if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
    assert len(ticks) <= 3 and all(t.is_integer() for t in ticks)
    assert ax.get_ylabel() == 'LFP (a.u.)' and ax.get_title(loc='left') == ''
    assert new_axes['raster'].get_ylabel() == 'Neuron ID'
    assert [t.get_text() for t in new_axes['raster'].get_yticklabels()] == ['E', 'I']
    samples[arm] = dict(lfp_samples=len(ax.lines[0].get_ydata()), lfp_visible_ticks=ticks,
                       spike_count=sum(len(c.get_offsets()) for c in new_axes['raster'].collections))
plt.close(old_fig)
plt.close(new_fig)

unchanged_panels = {p: sha(PARENT/f'figures/fig4-panel{p.lower()}.png') ==
                   sha(OUT/f'figures/fig4-panel{p.lower()}.png') for p in 'ABCFGHI'}
assert all(unchanged_panels.values()), unchanged_panels
old_pixels = np.asarray(Image.open(PARENT/'figures/fig4-complete-layout.png'))
new_pixels = np.asarray(Image.open(OUT/'figures/fig4-complete-layout.png'))
assert old_pixels.shape == new_pixels.shape
diff = np.any(old_pixels != new_pixels, axis=2)
layout = json.loads((OUT/'row_layout_audit.json').read_text())
height_mm = layout['size_mm'][1]
_, bottom, _, top = layout['row_bounds'][1]['bounds_mm']
lo = int(np.floor((height_mm-top-1)/height_mm*diff.shape[0]))
hi = int(np.ceil((height_mm-bottom+1)/height_mm*diff.shape[0]))
outside = diff.copy()
outside[lo:hi] = False
assert not outside.any()
text_layout = json.loads((OUT/'text_layout_audit.json').read_text())
assert not text_layout['outside_canvas']
assert min(layout['clear_gaps_mm']) > 1.8
report = dict(status='AGENT_CHECKS_PASS_AUTHOR_REVIEW_PENDING', changed_panels=['D', 'E'],
              scientific_change=None, preserved_parent_and_formal_files=len(before),
              unchanged_numeric_files=len(numeric_files), unchanged_panel_pngs=unchanged_panels,
              native_samples_and_spike_coordinates_exactly_preserved=samples,
              neuron_order_and_time_window_unchanged=True,
              changed_pixels=int(diff.sum()), changed_pixels_outside_DE_row=int(outside.sum()),
              text_outside_canvas=[], clear_row_gaps_mm=layout['clear_gaps_mm'],
              D_circle='Same fixed readout ROI; no core',
              E_fill='Categorical core identity; not a quantitative threshold field',
              human_visual_acceptance='PENDING')
(OUT/'visual_qa.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(report, ensure_ascii=False, indent=2))
