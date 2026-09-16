"""Update the historical response layout from frozen, completed run summaries.

This is an offline display extension: no observer, fitting, or simulation changes.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.topic4_pdf_font_guard import install

BASE = Path('/data/hfosp/topic4_sef_hfo')
SOURCES = dict(followup=BASE/'core_recruitment_tradeoff_followup_20260912',
               multiseed=BASE/'core_multiseed_response_curves_20260913')
OLD = BASE/'core_extent_long_propagation_20260909/interim_review_18runs/draw_review.py'
PAIRS = [('SCL9', 'SCL8'), ('SCL8', 'SCL7'), ('SCL7', 'SCL6')]
GROUPS = dict(
    original=[('participation_mae', '触点参与概率误差 ↓'),
              ('rank_correlation', '平均质心顺序相关 ↑'),
              ('no_SCL', '完全缺少 SCL 的事件比例')],
    time=[('SCL_minus_ICL_lag_median_ms', 'SCL − ICL 质心时差 (ms)'),
          ('recruitment_span_ms_median', '各触点 t10 招募跨度 (ms)'),
          ('local_width_ms_median', '单触点局部活动宽度 (ms)')],
    scl=[(f'pair_{i}', f'{a} 质心早于 {b} 的概率') for i, (a, b) in enumerate(PAIRS)],
    scl_lag=[(f'scl_lag_{i}', f'{b} − {a} 质心时差 (ms)') for i, (a, b) in enumerate(PAIRS)],
    support=[('SCL_upper_participation', 'SCL9 / SCL8 参与概率'),
             ('mode_fraction', '本类事件占全部合格事件比例'),
             ('n', '本类合格事件数')])
LIMITS = dict(participation_mae=(-.02, .42), rank_correlation=(-1.06, 1.06),
              no_SCL=(-.035, 1.035), SCL_minus_ICL_lag_median_ms=(-45, 130),
              recruitment_span_ms_median=(0, 140), local_width_ms_median=(0, 25),
              SCL_upper_participation=(-.035, 1.035), mode_fraction=(-.035, 1.035),
              **{f'pair_{i}': (-.035, 1.035) for i in range(3)})
COLORS = {2511: '#666666', 2711: '#277da8', 2712: '#cd7c24'}
GROUP_NAMES = dict(original='原图三项观测', time='窗内毫秒时间尺度',
                   scl='SCL 内部相邻顺序', scl_lag='SCL 内部毫秒时差', support='参与与事件支持')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))


def snapshot(out):
    """Freeze complete summary files; never mutate running observer outputs."""
    dest = out/'snapshots'
    dest.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, root in SOURCES.items():
        files = ['plan.json', 'status.json'] + ['analysis/'+f for f in
            ['counts.csv', 'observations.csv', 'pairs.csv', 'contacts.csv',
             'events.csv', 'patient_reference.json', 'status.json']]
        for relative in files:
            src = root/relative
            if not src.exists():
                continue
            target = dest/name/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                before = src.stat()
                shutil.copyfile(src, target)
                after = src.stat()
                if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
                    target.unlink()
                    raise RuntimeError(f'Source changed while snapshotting: {src}; rerun')
            manifest.append(dict(source=str(src), snapshot=str(target), sha256=sha(target)))
    for src, label in [(OLD, 'historical_figure_producer.py'),
                       (BASE/'scl_internal_template_review_20260914/pair_differences.csv', 'patient_scl_pairs.csv')]:
        target = dest/label
        if not target.exists():
            shutil.copyfile(src, target)
        manifest.append(dict(source=str(src), snapshot=str(target), sha256=sha(target)))
    return manifest


class Data:
    def __init__(self, out, label):
        self.label = label
        root = out/'snapshots'/label
        self.plan = json.loads((root/'plan.json').read_text())
        self.candidates = {c['id']: c for c in self.plan['candidates']}
        self.counts = pd.read_csv(root/'analysis/counts.csv')
        self.obs = pd.read_csv(root/'analysis/observations.csv')
        self.obs = self.obs[self.obs.layer == 'primary'].copy()
        self.pairs = pd.read_csv(root/'analysis/pairs.csv')
        self.pairs = self.pairs[self.pairs.layer == 'primary'].copy()
        self.ref = json.loads((root/'analysis/patient_reference.json').read_text())
        self.lookup = self.obs.set_index(['candidate', 'topology', 'noise', 'mode'])
        self.clook = self.counts.set_index(['candidate', 'topology', 'noise'])
        self.plook = self.pairs.set_index(['candidate', 'topology', 'noise', 'mode', 'contact_i', 'contact_j'])
        assert self.lookup.index.is_unique and self.clook.index.is_unique and self.plook.index.is_unique
        self.names = self.ref['contact_order']
        p = pd.read_csv(out/'snapshots/patient_scl_pairs.csv')
        self.patient_pairs = p[(p.id == 'patient_FIT') & (p.scope == 'pairwise_coparticipating')].set_index(['mode', 'earlier_test', 'later_test'])
        for r in self.counts.itertuples():
            assert r.physical_status == 'COMPLETE_NO_RUNAWAY' and r.duration_ms == 60000
            for mode, n in [('TA', r.TA), ('TB', r.TB), ('ALL', r.primary)]:
                assert self.lookup.loc[(r.candidate, r.topology, r.noise, mode), 'n'] == n
        self.obs['no_SCL'] = 1 - self.obs.SCL_any
        self.obs.to_csv(out/(label+'_primary_observations.csv'), index=False)

    def value(self, cid, topo, noise, mode, key):
        ix = (cid, topo, noise, mode)
        if ix not in self.lookup.index:
            return np.nan, 0
        row = self.lookup.loc[ix]
        n = int(row['n'])
        if n == 0:
            return (0., 0) if key == 'n' else (np.nan, 0)
        if key.startswith('pair_'):
            pair = self.plook.loc[ix + PAIRS[int(key[-1])]]
            return float(pair.model_i_precedes_j), int(pair.model_joint_n)
        if key.startswith('scl_lag_'):
            pair = self.plook.loc[ix + PAIRS[int(key[-1])]]
            return float(pair.lag_median), int(pair.model_joint_n)
        if key == 'no_SCL':
            return 1-float(row.SCL_any), n
        if key == 'mode_fraction':
            return n/float(self.clook.loc[ix[:3], 'primary']), n
        return float(row[key]), int(row.lag_joint_n) if key == 'SCL_minus_ICL_lag_median_ms' else n

    def reference(self, mode, key):
        ref = self.ref['modes'][mode]
        if key.startswith('scl_lag_'):
            return float(self.patient_pairs.loc[(mode,) + PAIRS[int(key[-1])], 'median'])
        if key.startswith('pair_'):
            a, b = PAIRS[int(key[-1])]
            ia, ib = self.names.index(a), self.names.index(b)
            return next(p['p'] for p in ref['pairs'] if (p['i'], p['j']) == (ia, ib))
        if key == 'no_SCL':
            return 1-ref['SCL_any']
        if key == 'mode_fraction':
            return ref['n']/self.ref['fit_n']
        if key == 'n':
            return None
        return ref.get(key)


def families(datasets):
    f, n = datasets['followup'], datasets['multiseed']
    result = []
    for shape, text, anchor in [
        ('circle', '圆核', 'up3__circle__EE_core_to_out_scale_1.25'),
        ('ellipse4', '左核椭圆 4∶1', 'up3__ellipse4__EI_same_core_scale_0.75')]:
        lines = []
        for ei, color in [(1., '#666666'), (.875, '#277da8'), (.75, '#b54d79')]:
            ids = [next(c['id'] for c in f.plan['candidates'] if c.get('factorial') ==
                        dict(shape=shape, EE_out=ee, EI=ei)) for ee in [1., 1.125, 1.25]]
            for noise, ls, marker in [(847101, '-', 'o'), (847102, '--', '^')]:
                lines.append(dict(ids=ids, topology=2511, noise=noise, color=color,
                                  ls=ls, marker=marker, legend=f'核内 E→I ×{ei:g}'))
        result.append(dict(id=f'{shape}_EE_EI', data='followup', title=f'{text}：向外兴奋 × 核内抑制',
            x=[1., 1.125, 1.25], ticks=['1', '1.125', '1.25'], xlabel='core E → 核外 E 权重倍数',
            lines=lines, color='颜色＝核内 E→I 倍数；同一条线只改变向外 E→E',
            context='固定基础网络 2511、原中心；每个组合两条配对噪声；完整 3 × 3 参数组合。'))
        suffix = ['', '__x_minus075', '__x_plus075', '__y_minus10', '__y_plus10', '__radius205', '__radius235']
        ids = [anchor+s for s in suffix]
        lines = [dict(ids=ids, topology=2511, noise=noise, color=color,
                      ls='-', marker='o', legend=f'噪声 {noise}')
                 for noise, color in [(847101, '#277da8'), (847102, '#d47e24')]]
        result.append(dict(id=f'{shape}_geometry', data='followup', title=f'{text}：左核位置与范围',
            x=list(range(7)), ticks=['原位\nr≈1.75', 'x −0.75', 'x +0.75', 'y −1', 'y +1', 'r→2.05', 'r→2.35'],
            xlabel='独立几何条件 (mm)；每项均相对原位，其余保持该工作点设置',
            lines=lines, color='颜色＝两条噪声；固定基础网络 2511',
            context=('向外 EE×1.25、核内 E→I×1。' if shape == 'circle' else '向外 EE×1、核内 E→I×0.75；r为等面积半径。')+
                    '扩大左核时匹配降阈值总量；平移仍改变core成员。'))
    topo_lines = lambda ids: [dict(ids=ids, topology=t, noise=s, color=COLORS[t],
                                  ls='-' if s == 847401 else '--', marker='o' if s == 847401 else '^', legend=f'网络 {t}')
                            for t in [2511, 2711, 2712] for s in [847401, 847402]]
    for family in n.plan['families']:
        result.append(dict(id='multiseed_'+family['id'], data='multiseed', title=family['label'],
            x=family['values'], ticks=[f'{v:g}' for v in family['values']],
            xlabel='核内 E→I 倍数' if family['axis'] == 'EI' else 'core E → 核外 E 权重倍数',
            lines=topo_lines(family['candidates']), color='颜色＝网络；实线圆点 / 虚线三角＝噪声 847401 / 847402',
            context=family['fixed']+'；未完成点留空，缺口两侧不连线。'))
    for key, title, ids, ticks, context in [
        ('position', '左核左移：三张网络的配对响应', ['bridge_circle_out125', 'bridge_circle_out125_xminus075'],
         ['原位置', '左移 0.75 mm'], '圆核；向外EE×1.25、核内E→I×1；平移改变core成员。'),
        ('EI_endpoints', '降低核内 E→I：三张网络的端点比较', ['curve_circle_out1_EI1', 'curve_circle_out1_EI0.75'],
         ['核内 E→I ×1', '核内 E→I ×0.75'], '圆核；向外EE×1；此页只比较两个完整端点，中间细扫另页列出。'),
        ('edge_vs_weight', '离核输出：增加边数与增加权重', ['bridge_circle_weight150', 'bridge_circle_degree150'],
         ['向外边权 ×1.5', '向外边数 ×1.5'], '圆核、核内E→I×1；名义剂量接近，不是实际总权重精确匹配。')]:
        result.append(dict(id='multiseed_'+key, data='multiseed', title=title, x=[0, 1], ticks=ticks,
            xlabel='两个离散干预条件', lines=topo_lines(ids),
            color='颜色＝网络；实线圆点 / 虚线三角＝噪声 847401 / 847402', context=context))
    return result


def key_effects(out, datasets):
    contrasts = [
        ('multiseed', 'bridge_circle_out125', 'bridge_circle_out125_xminus075', '左核左移0.75mm'),
        ('multiseed', 'curve_circle_out1_EI1', 'curve_circle_out1_EI0.75', '核内EI从1降至0.75'),
        ('followup', 'up3__circle__EE_core_to_out_scale_1.25',
         'up3__circle__EE_core_to_out_scale_1.25__radius235', '左核半径1.75增至2.35mm')]
    rows = []
    for source, ref, cid, name in contrasts:
        data = datasets[source]
        for mode in ['TA', 'TB']:
            cells = [(t, n) for c, t, n, m in data.lookup.index if c == ref and m == mode
                     and (cid, t, n, mode) in data.lookup.index and (source != 'followup' or t == 2511)]
            for metric in ['n', 'participation_mae', 'rank_correlation', 'SCL_any',
                           'SCL_upper_participation', 'SCL_minus_ICL_lag_median_ms', 'local_width_ms_median']:
                a = np.array([data.value(ref, t, n, mode, metric)[0] for t, n in cells])
                b = np.array([data.value(cid, t, n, mode, metric)[0] for t, n in cells])
                valid = np.isfinite(a) & np.isfinite(b)
                a, b = a[valid], b[valid]
                rows.append(dict(source=source, contrast=name, mode=mode, metric=metric,
                    paired_runs=len(a), topologies=len({cells[i][0] for i in np.flatnonzero(valid)}),
                    reference_mean=a.mean(), candidate_mean=b.mean(), candidate_min=b.min(), candidate_max=b.max(),
                    negative_pairs=int((b < a).sum()), positive_pairs=int((b > a).sum())))
    pd.DataFrame(rows).to_csv(out/'key_paired_effects.csv', index=False)


def draw(out, fam, data, group, pdf, plotted, captions):
    modes = ['TB', 'TA']  # Exact row correspondence to the supplied M0/M1 figure.
    metrics = GROUPS[group]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.7))
    fig.subplots_adjust(left=.065, right=.985, top=.79, bottom=.19, hspace=.50, wspace=.29)
    observed = set()
    for row, mode in enumerate(modes):
        for col, (key, label) in enumerate(metrics):
            ax = axes[row, col]
            for line in fam['lines']:
                ys, ns = zip(*[data.value(cid, line['topology'], line['noise'], mode, key) for cid in line['ids']])
                ax.plot(fam['x'], ys, color=line['color'], ls=line['ls'], lw=1.4, alpha=.85)
                for x, cid, y, support in zip(fam['x'], line['ids'], ys, ns):
                    if not np.isfinite(y):
                        continue
                    outside = key.startswith('scl_lag_') and abs(y) > 20
                    ax.plot(x, np.sign(y)*19 if outside else y,
                            marker=('^' if y > 0 else 'v') if outside else line['marker'],
                            ms=6 if outside else 5, color=line['color'],
                            mfc='white' if support < 16 else line['color'], linestyle='none')
                    if outside:
                        ax.annotate(f'{y:.1f}', (x, np.sign(y)*17), color=line['color'],
                                    fontsize=8, ha='left' if x == min(fam['x']) else 'right', va='center')
                    observed.add((cid, line['topology'], line['noise']))
                    plotted.append(dict(figure=fam['id']+'_'+group, source=data.label, candidate=cid,
                        topology=line['topology'], noise=line['noise'], mode=mode, metric=key,
                        x=x, value=y, support_n=support))
            ref = data.reference(mode, key)
            if ref is not None:
                ax.axhline(ref, color='black', ls=':', lw=1.5)
            if key in LIMITS:
                ax.set_ylim(LIMITS[key])
            else:
                ax.set_ylim(0, 200)
            ax.set(title=f'{mode} ({"M0" if mode == "TB" else "M1"}) · {label}',
                   xticks=fam['x'], xticklabels=fam['ticks'])
            ax.set_xlim(min(fam['x'])-(max(fam['x'])-min(fam['x']))*.06,
                        max(fam['x'])+(max(fam['x'])-min(fam['x']))*.06)
            ax.tick_params(labelsize=10)
            if len(fam['x']) == 2:
                ax.get_xticklabels()[0].set_ha('left')
                ax.get_xticklabels()[-1].set_ha('right')
            ax.grid(alpha=.16)
            if row == 1:
                ax.set_xlabel(fam['xlabel'], fontsize=9)
    handles = []
    seen = set()
    for line in fam['lines']:
        if line['legend'] not in seen:
            handles.append(Line2D([], [], color=line['color'], label=line['legend']))
            seen.add(line['legend'])
    if len({line['ls'] for line in fam['lines']}) > 1:
        seeds = sorted({line['noise'] for line in fam['lines']})
        handles += [Line2D([], [], color='#333', ls='-', marker='o', label=f'噪声 {seeds[0]}'),
                    Line2D([], [], color='#333', ls='--', marker='^', label=f'噪声 {seeds[1]}')]
    handles += [Line2D([], [], color='black', ls=':', label='患者 FIT 参考')]
    fig.suptitle(fam['title']+'｜'+GROUP_NAMES[group], fontsize=17, y=.985)
    fig.text(.5, .934, fam['context'], ha='center', fontsize=10)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .913), ncol=len(handles), frameon=False, fontsize=10)
    info = {
        'original': '参与误差：15触点参与概率的平均绝对差；顺序相关：参与内归一化rank的逐触点均值与患者的Spearman相关。\n整杆缺失只要求四个SCL触点均未参与；它接近0不代表整杆完整参与或内部顺序恢复。误差0 / 相关1是参照值，不是验收门槛。',
        'time': '杆间差：每事件参与SCL质心中位数 − 参与ICL质心中位数，再取事件中位数；负值表示SCL较早。\nt10跨度：参与触点累计包络质量10%时刻的极差；局部宽度：事件内各触点t90−t10中位数，再跨事件取中位数。后两项仅模型包络，无患者HFO等价线。',
        'scl': '每个概率只使用对应两个触点共同参与的事件，质心完全并列计0.5；不要求四个SCL触点全部参与。\n先后概率是时序观测，不等于因果传播路径；黑点线保留患者自身的顺序变异，不强制0或1。',
        'scl_lag': '每点为共同参与事件的质心时差中位数；正值表示编号较小触点更晚。固定显示±20ms，越界以边缘三角与实际数值标注。\n患者与模型使用相同触点对；质心时差不等于起始时差或局部持续时间。逐运行均值、方差、5–95%范围另保留于pairs.csv。',
        'support': '上部SCL参与＝SCL9与SCL8参与概率的平均；模式占比以该运行全部合格事件为分母。\nTA/TB由冻结分类器组织比较，标签也用于既有分布训练目标；两类标签存在不等于患者传播恢复。'
    }[group]
    planned = len({(cid, l['topology'], l['noise']) for l in fam['lines'] for cid in l['ids']})
    fig.text(.065, .105, fam['color']+f'。本页已完成 {len(observed)}/{planned} 个运行。', fontsize=10)
    fig.text(.065, .058, info, fontsize=9, linespacing=1.5)
    fig.text(.065, .014, '每点一条60秒运行，排除前1.5秒；线仅连接已测参数点。空心点＝相关事件数<16，未删除或改loss；不是显著性检验。各页同一指标固定Y轴。', fontsize=9)
    name = fam['id']+'_'+group
    for ext in ['png', 'pdf']:
        fig.savefig(out/'figures'/(name+'.'+ext), dpi=150)
    pdf.savefig(fig)
    plt.close(fig)
    captions.append(f'### {name}.png\n{fam["title"]}，展示{GROUP_NAMES[group]}；上排TB=M0、下排TA=M1，沿用用户旧图的类别行序。{fam["color"]}，每点为60秒运行，未完成及不可估计点不补零。\n{info.replace(chr(10), " ")}\n**关注点**：参数改变对两类事件是否有取舍，以及响应是否在同网噪声重演和不同网络中保留。\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=BASE/'parameter_window_response_update_20260914')
    args = ap.parse_args()
    out = args.out
    (out/'figures').mkdir(parents=True, exist_ok=True)
    install()
    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 11})
    sources = snapshot(out)
    datasets = {name: Data(out, name) for name in SOURCES}
    assert datasets['followup'].ref == datasets['multiseed'].ref
    key_effects(out, datasets)
    # Keep clinically relevant small lags readable. Out-of-view values remain
    # in the data and are explicitly annotated at the shared display boundary.
    LIMITS.update({f'scl_lag_{i}': (-20, 20) for i in range(3)})
    fams = families(datasets)
    plotted, captions, skipped = [], [], []
    with PdfPages(out/'parameter_window_response_atlas.pdf') as pdf:
        for fam in fams:
            data = datasets[fam['data']]
            if not any(cid in set(data.counts.candidate) for l in fam['lines'] for cid in l['ids']):
                skipped.append(fam['id'])
                continue
            for group in GROUPS:
                draw(out, fam, data, group, pdf, plotted, captions)
    frame = pd.DataFrame(plotted)
    frame.to_csv(out/'plotted_points.csv', index=False)
    # All pooled-label observables remain separate from conditional comparisons.
    for label, data in datasets.items():
        data.obs[data.obs['mode'] == 'ALL'].to_csv(out/(label+'_ALL_observations.csv'), index=False)
    checks = []
    for p in sorted((out/'figures').glob('*.png')):
        with Image.open(p) as im:
            im.load()
            checks.append(dict(file=p.name, dimensions=list(im.size), sha256=sha(p)))
    # Quantitative invariants, including the exact legacy metric transformation.
    for r in frame.itertuples():
        expected, n = datasets[r.source].value(r.candidate, r.topology, r.noise, r.mode, r.metric)
        assert np.isclose(expected, r.value) and n == r.support_n
        if r.metric in LIMITS and not r.metric.startswith('scl_lag_'):
            lo, hi = LIMITS[r.metric]
            assert lo <= r.value <= hi, (r.metric, r.value)
    dump(out/'manifest.json', dict(created_unix=time.time(), producer=str(Path(__file__).resolve()),
        sources=sources, completed_summary_runs={k:len(v.counts) for k,v in datasets.items()},
        skipped_unstarted_families=skipped, new_simulations=0, loss_changes=0,
        figure_row_order=['TB=M0', 'TA=M1'], snapshot_reuse='Rerunning this output path reuses its immutable snapshots.'))
    dump(out/'artifact_qa.json', dict(status='NUMERICAL_AND_DECODE_PASS_VISUAL_REVIEW_PENDING',
        plotted_points_checked=len(frame), png_count=len(checks), figures=checks,
        explicitly_annotated_scl_lag_values_outside_display=int(((frame.metric.str.startswith('scl_lag_')) & (frame.value.abs()>20)).sum())))
    (out/'figures/README.md').write_text('\n'.join(captions))
    print(json.dumps(dict(output=str(out), figures=len(checks), plotted_points=len(frame), skipped=skipped)))


if __name__ == '__main__':
    main()
