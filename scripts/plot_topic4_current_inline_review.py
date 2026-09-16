"""Small, immutable display packet from completed runs; no simulation or selection changes."""
from pathlib import Path
import csv, json, hashlib, shutil, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.topic4_pdf_font_guard import install

BASE = Path('/data/hfosp/topic4_sef_hfo')
OUT = BASE / 'current_best_inline_review_20260913'
N = BASE / 'core_multiseed_response_curves_20260913/analysis'
F = BASE / 'core_recruitment_tradeoff_followup_20260912/analysis'


def main():
    install()
    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 11})
    (OUT / 'figures').mkdir(parents=True, exist_ok=True)
    sources = []
    for label, folder in [('multiseed', N), ('followup', F)]:
        for name in ['counts.csv', 'observations.csv', 'pairs.csv', 'patient_reference.json']:
            src = folder / name
            dest = OUT / (label + '_' + name)
            shutil.copyfile(src, dest)
            sources.append(dict(source=str(src), snapshot=str(dest), sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    def rows(label, kind):
        return list(csv.DictReader((OUT / f'{label}_{kind}.csv').open()))
    def lookup(label):
        return {(r['candidate'], int(r['topology']), int(r['noise']), r['mode']): r
                for r in rows(label, 'observations') if r['layer'] == 'primary'}
    nr, fr = lookup('multiseed'), lookup('followup')
    patient = json.loads((OUT / 'multiseed_patient_reference.json').read_text())
    metrics = [('TA', 'SCL_upper_participation', 'TA：上部SCL参与概率'),
               ('TA', 'SCL_minus_ICL_lag_median_ms', 'TA：SCL−ICL杆间时差 (ms)'),
               ('TB', 'SCL_minus_ICL_lag_median_ms', 'TB：SCL−ICL杆间时差 (ms)')]
    colors = {2511: '#666666', 2711: '#2878b5', 2712: '#d47e18'}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.subplots_adjust(left=.085, right=.985, bottom=.19, top=.79, hspace=.64, wspace=.30)
    designs = [(['bridge_circle_out125', 'bridge_circle_out125_xminus075'], [0, 1],
                ['原位置', '左移0.75 mm'], '左核位置；向外EE×1.25、核内EI×1'),
               (['curve_circle_out1_EI1', 'curve_circle_out1_EI0.75'], [0, 1],
                ['核内EI×1', '核内EI×0.75'], '核内E→I强度；原位置、向外EE×1')]
    for row, (ids, xs, labels, context) in enumerate(designs):
        for col, (mode, metric, title) in enumerate(metrics):
            ax = axes[row, col]
            for topo, color in colors.items():
                for seed in [847401, 847402]:
                    vals = [float(nr[(cid, topo, seed, mode)][metric]) for cid in ids]
                    ax.plot(xs, vals, color=color, marker='o' if seed == 847401 else '^',
                            ls='-' if seed == 847401 else '--', lw=1.5, ms=5)
            ax.axhline(patient['modes'][mode][metric], color='black', ls=':', lw=1.5)
            ax.set(title=title, xticks=xs, xticklabels=labels, xlim=(-.12, 1.12))
            ax.grid(alpha=.18)
            ax.set_ylim((0, 1.04) if col == 0 else (-30, 110) if col == 1 else (-5, 42))
            if col == 0:
                ax.annotate(context, xy=(0, 1.26), xycoords='axes fraction', fontsize=11, fontweight='bold')
    handles = [Line2D([], [], color=c, label=f'网络 {t}') for t, c in colors.items()]
    handles += [Line2D([], [], color='black', marker='o', label='噪声847401'),
                Line2D([], [], color='black', marker='^', ls='--', label='噪声847402'),
                Line2D([], [], color='black', ls=':', label='患者FIT参考')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .925), ncol=6, frameon=False, fontsize=10)
    fig.suptitle('位置与局部抑制：哪些作用能跨网络、跨噪声保留？', y=.98, fontsize=17)
    fig.text(.075, .035, '每条线固定同一基础网络和同一噪声；每点一条完整60秒运行，不把事件当作独立网络。两端是实际离散条件，连线不代表已测中间值。\n'
             '上部SCL = SCL9/8参与概率平均；杆间差 = 每事件参与SCL质心中位数 − 参与ICL质心中位数，再取事件中位数。负值为SCL较早。\n'
             '每项改变均与其直接对照比较；位置改变也改变core成员。患者参考不是恢复阈值；两类标签齐全不代表路径恢复。\n'
             '下排TA样本数：EI×1为8–46，EI×0.75为57–93；上排TA为62–90。完整参与、时差散布与事件支持量保留在快照CSV。', fontsize=9)
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / 'figures' / ('position_and_EI_multiseed.' + ext), dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.subplots_adjust(left=.085, right=.985, bottom=.18, top=.83, hspace=.43, wspace=.28)
    ei_colors = {1.: '#666666', .875: '#2878b5', .75: '#b64971'}
    fs = [('SCL_upper_participation', '上部SCL参与概率'),
          ('SCL_minus_ICL_lag_median_ms', 'SCL−ICL杆间时差 (ms)'),
          ('pair_order_probability_mae', '接触对先后概率误差 ↓')]
    for row, mode in enumerate(['TA', 'TB']):
        for col, (metric, title) in enumerate(fs):
            ax = axes[row, col]
            for ei, color in ei_colors.items():
                for seed in [847101, 847102]:
                    rs = []
                    for ee in [1., 1.125, 1.25]:
                        # Read candidate factorial identity rather than infer old IDs.
                        matches = []
                        for key, obs in fr.items():
                            cid, topo, noise, mode0 = key
                            if (topo, noise, mode0) != (2511, seed, mode):
                                continue
                            c = candidates[cid]
                            fac = c.get('factorial', {})
                            if fac.get('shape') == 'circle' and fac.get('EE_out') == ee and fac.get('EI') == ei:
                                matches.append(obs)
                        assert len(matches) == 1, (ee, ei, seed, mode, len(matches))
                        rs.append(float(matches[0][metric]))
                    ax.plot([1, 1.125, 1.25], rs, color=color,
                            marker='o' if seed == 847101 else '^', ls='-' if seed == 847101 else '--', lw=1.5)
            ref = 0 if metric == 'pair_order_probability_mae' else patient['modes'][mode][metric]
            ax.axhline(ref, color='black', ls=':', lw=1.5)
            ax.set(title=f'{mode}：{title}', xlabel='core向外E→E权重倍数', xticks=[1, 1.125, 1.25])
            ax.grid(alpha=.18)
            ax.set_ylim((0, 1.04) if col == 0 else (-30, 110) if col == 1 else (0, .55))
    handles = [Line2D([], [], color=c, label=f'核内EI×{v:g}') for v, c in ei_colors.items()]
    handles += [Line2D([], [], color='black', marker='o', label='噪声847101'),
                Line2D([], [], color='black', marker='^', ls='--', label='噪声847102')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .93), ncol=5, frameon=False)
    fig.suptitle('向外兴奋与核内抑制：参与增加，是否同时带来正确时序？', y=.99, fontsize=17)
    fig.text(.075, .035, '圆核原位置、固定基础网络2511；颜色代表核内E→I强度，圆实线／三角虚线代表两条噪声。同一条线只改变向外E→E权重。\n'
             '黑点线为患者FIT参考；成对误差为共同支持触点对中，模型与患者“谁较早”的概率差的平均绝对值，0表示这些概率相同。\n'
             '这张图是一个网络内的3×3参数组合响应，不代表三张网络；每点完整60秒，历史直接对照复用，不重复计算。\n'
             '固定Y轴便于前后比较；上部SCL参与和杆间差定义同上一图。完整事件分布、原生场与模式数量仍须同时审阅。', fontsize=9)
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / 'figures' / ('EE_EI_joint_response.' + ext), dpi=160)
    plt.close(fig)
    manifest = dict(created_unix=time.time(), producer=str(Path(__file__)), sources=sources,
                    physical_changes=0, selected_candidate='bridge_circle_out125_xminus075',
                    selection='lowest equal-run mean frozen L_search among completed 3-network by 2-noise conditions; not mechanism acceptance')
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    (OUT / 'figures/README.md').write_text('### position_and_EI_multiseed.png\n两组独立的单参数配对，各含3张网络和2条噪声；颜色是网络身份。比较TA参与、TA时差与TB时差，不能以均值改善替代完整分布。\n**关注点**：核内EI端点变化的TA响应是否重复，以及TB是否仍有固定残差。\n\n### EE_EI_joint_response.png\n圆核原位置、同网络2511的EE×EI九组合，各2条噪声。颜色是EI倍数，同线只改变EE权重，黑点线是患者参考。\n**关注点**：参与与顺序是否同时改善，以及参数之间是否存在不同响应。\n\n### gif_six_events_2511.png\n当前提名候选的6个动画事件固定中间帧，用于Agent图件核查；不是按患者相似度选例。\n**关注点**：完整GIF中的运动与固定患者参考应同时检查。\n\n### gif_six_events_2711.png\n同参数另一网络的6个事件固定中间帧，供核查；不是独立患者验证。\n**关注点**：网络改变后参与缺口及传播顺序是否保留。\n')
    print(OUT)


if __name__ == '__main__':
    plan = json.loads((F.parent / 'plan.json').read_text())
    candidates = {c['id']: c for c in plan['candidates']}
    main()
