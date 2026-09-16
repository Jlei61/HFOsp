"""Fixed-time stills of the preselected multi-event GIFs, with a shared scale."""
import json
import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_topic4_first_refinement_response as first

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,choices=[847101,847102],default=847101)
    parser.add_argument('--review-dir',type=Path,default=None)
    args=parser.parse_args()
    out = first.OUT / 'native_review'
    if args.seed!=847101:
        out=first.OUT/f'noise_{args.seed}'/'native_review'
    if args.review_dir is not None:
        out=args.review_dir
    plan = json.loads((out / 'plan.json').read_text())
    cases = plan['candidates']
    units = []
    vmax = 1.
    for case in cases:
        path = first.analysis.path_for(case, args.seed)
        result = json.loads(path.read_text())
        with np.load(path.with_suffix('.npz')) as source:
            arrays = {key: source[key] for key in ['sheet_activity_counts', 'contact_xy_mm', 'event_mode']}
        selection = json.loads((out / (case['id'] + '_visual_selection.json')).read_text())
        picks = selection['gif_events']
        for i in picks:
            lo, hi = result['events'][i]['window_ms']
            vmax = max(vmax, float(arrays['sheet_activity_counts'][round(lo/2):round(hi/2)].max()))
        units.append((case, result, arrays, picks))
    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 8, 'pdf.fonttype': 42})
    offsets = list(range(0, 241, 30))
    records = []
    for case, result, arrays, picks in units:
        fig, axes = plt.subplots(len(picks), len(offsets), figsize=(16, 11), layout='constrained')
        for row, i in enumerate(picks):
            lo, hi = result['events'][i]['window_ms']
            for col, offset in enumerate(offsets):
                frame = round((lo + offset)/2)
                ax = axes[row, col]
                im = ax.imshow(arrays['sheet_activity_counts'][frame], origin='lower', extent=[0,20,0,20],
                    cmap='inferno', vmin=0, vmax=vmax, interpolation='nearest')
                ax.scatter(*arrays['contact_xy_mm'].T, s=7, facecolors='none', edgecolors='cyan', linewidths=.4)
                for center, radius in zip(case['centers_mm'], case['radii_mm']):
                    ax.add_patch(Circle(center, radius, fill=False, edgecolor='white', linewidth=.4))
                ax.set(xticks=[], yticks=[], aspect='equal', xlim=(0,20), ylim=(0,20))
                if row == 0:
                    ax.set_title(f'窗口内 {offset} ms')
                if col == 0:
                    label = 'TA' if arrays['event_mode'][i] == 1 else 'TB'
                    ax.set_ylabel(f'{label} · 事件 {i}\n{lo/1000:.3f} s 起', fontsize=9)
            records.append(dict(candidate=case['id'], event_index=i, window_ms=[lo,hi], offsets_ms=offsets))
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.5, label='每个 1 mm 网格内的原生 2 ms 发放数；两条件共用色标')
        label = first.analysis.label(case)
        fig.suptitle(label + '：同一网络／噪声的六个事件\n与 GIF 使用相同事件；每类最早三个合格事件，按时间排列；定时截帧不代表帧间无活动')
        name = case['id'] + '_native_fixed_times'
        for ext in ['png','pdf']:
            fig.savefig(out/'figures'/f'{name}.{ext}', dpi=140)
        plt.close(fig)
    (out/'native_still_selection.json').write_text(json.dumps(dict(shared_max_count=vmax, events=records), indent=2))

if __name__ == '__main__':
    main()
