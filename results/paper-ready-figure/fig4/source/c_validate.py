"""Numerical and embedding checks for the candidate, without old figure imports."""
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.transforms import Bbox

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'src/group_event_analysis.py').exists())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    spec = importlib.util.spec_from_file_location('fig4_new_c', HERE/'c_panel.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checks = {}
    for key, size, rect in [('standalone',[151,73],[0,0,1,1]),
                            ('embedded',[320,245],[.50,.68,151/320,73/245])]:
        fig = plt.figure(figsize=np.array(size)/25.4, dpi=160)
        rc = dict(matplotlib.rcParams)
        result = module.render_c(fig, rect)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        box = Bbox.from_bounds(rect[0]*fig.bbox.width, rect[1]*fig.bbox.height,
                               rect[2]*fig.bbox.width, rect[3]*fig.bbox.height)
        outside, sizes = [], []
        for t in fig.findobj(Text):
            if t.get_visible() and t.get_text():
                bb = t.get_window_extent(renderer)
                sizes.append(t.get_fontsize())
                if not (box.contains(*bb.p0) and box.contains(*bb.p1)):
                    outside.append(t.get_text())
        checks[key] = dict(text_outside_rect=outside, minimum_font_pt=min(sizes),
                           axes_count=len(result['axes']), rc_unchanged=dict(matplotlib.rcParams)==rc)
        assert not outside, (key, outside)
        assert checks[key]['rc_unchanged']
        plt.close(fig)
    z = np.load(HERE/'c_panel_data.npz')
    meta = json.loads((HERE/'c_panel_metadata.json').read_text())
    for side in ['model','patient']:
        keep=(z[side+'_tfr_frequency_hz']>=20) if side=='model' else np.ones(len(z[side+'_tfr_frequency_hz']),bool)
        ww = z[side+'_tfr_magnitude'][:,keep]**3
        mass = ww.sum((1,2))
        t, f = z[side+'_tfr_time_ms'], z[side+'_tfr_frequency_hz'][keep]
        np.testing.assert_allclose(np.einsum('cft,t->c',ww,t)/mass,
                                   z[side+'_tfr_centroid_ms'], atol=1e-10)
        np.testing.assert_allclose(np.einsum('cft,f->c',ww,f)/mass,
                                   z[side+'_tfr_centroid_hz'], atol=1e-10)
        np.testing.assert_allclose(z[side+'_tfr_relative'].max((1,2)), np.ones(len(z['contact_names'])))
        ix=[z['native_all_contact_names'].tolist().index(n) for n in z['contact_names']]
        np.testing.assert_array_equal(z[side+'_native_rank'],z[side+'_native_rank_all'][ix])
        assert np.all(abs(z[side+'_native_centroid_ms'])<125)
    checks['centroid_direct_sum_parity'] = True
    checks['data_digest_matches_metadata'] = digest(HERE/'c_panel_data.npz')==meta['data_sha256']
    assert checks['data_digest_matches_metadata']
    formal = ROOT/'results/paper-ready-figure/fig4'
    manifest = json.loads((formal/'release_manifest.json').read_text())
    changed = [name for name,v in manifest['files'].items()
               if digest(formal/name)!=v['sha256']]
    checks['formal_fig4_release_files_checked'] = len(manifest['files'])
    checks['formal_fig4_files_not_matching_frozen_manifest'] = changed
    assert not changed
    fig=plt.figure(figsize=(151/25.4,73/25.4),dpi=180)
    module.render_c(fig,[0,0,1,1])
    for suffix in ['png','pdf']:fig.savefig(HERE/('c_panel_preview.'+suffix),dpi=240)
    plt.close(fig)
    info = subprocess.check_output(['pdfinfo', str(HERE/'c_panel_preview.pdf')], text=True)
    checks['pdf_page_size'] = next(line for line in info.splitlines() if line.startswith('Page size:'))
    checks['human_visual_acceptance'] = 'PENDING'
    checks['agent_visual_review'] = 'PENDING final full-figure inspection'
    checks['file_digests'] = {name:digest(HERE/name) for name in
        ['c_panel.py','c_panel_data.npz','c_panel_metadata.json','c_panel_preview.png','c_panel_preview.pdf']}
    (HERE/'c_panel_validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__ == '__main__':
    main()
