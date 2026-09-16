"""Describe a completed BO batch without changing its objective or proposals."""
from pathlib import Path
import argparse
import datetime
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src/snn_engine')]
import numpy as np
from scripts import analyze_topic4_three_observable_bo as analysis
from scripts import control_topic4_three_observable_bo as ctl
from scripts.topic4_three_observable_plot_labels import plot_label, parameter_table


def review(batch):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.topic4_pdf_font_guard import install
    install()
    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 10, 'pdf.fonttype': 3})
    rt, out = analysis.rt, analysis.OUT
    proposal = rt.read(out / f'proposals/adaptive_{batch:02}.json')
    plan = rt.read(out / 'plan.json')
    rows = analysis.records()
    lookup = {(r['candidate'], r['noise']): r for r in rows
              if r['stage'] in ['initial', 'adaptive'] and r['topology'] == 2511}
    units = [lookup[cid, noise] for cid in proposal['ids'] for noise in plan['seeds']]
    assert len(units) == 8 and all(r['J'] is not None for r in units)
    data = ctl.training_data()
    before = [r for r in proposal['training_data'] if r['scorable']]
    previous = min(before, key=lambda r: r['J'])['candidate']
    ids = list(dict.fromkeys([plan['reference_id'], previous] + proposal['ids']))
    dest = out / f'overnight_20260914/batch{batch:02}_review'
    figures = dest / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    patient = rt.read(out / 'overnight_20260914/condition_summary.json')['patient']
    summary = []
    for cid in ids:
        q = [lookup[cid, n] for n in plan['seeds']]
        summary.append(dict(candidate=cid, label=plot_label(cid),
                            J=float(np.mean([r['J'] for r in q])),
                            components=np.mean([r['components'] for r in q], axis=0).tolist(),
                            runs=[{k: r[k] for k in ['noise', 'N', 'J', 'mode_counts', 'raw', 'groups']}
                                  for r in q]))
    # The comparison retains both noise realizations. These are not confidence intervals.
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for row, mode in enumerate(['TA', 'TB']):
        for col, (key, title) in enumerate([
            ('rank_correlation', '平均 rank 模板相关 ↑'),
            ('both_rods', '两杆同时参与比例'),
            ('SCL_minus_ICL_lag_median_ms', 'SCL−ICL 质心中位差 (ms)')]):
            ax = axes[row, col]
            for j, cid in enumerate(ids):
                for offset, noise, color, marker in [(-.08, 847401, '#3276a8', 'o'),
                                                       (.08, 847402, '#db7c36', '^')]:
                    val = lookup[cid, noise]['raw'][mode].get(key)
                    if val is not None:
                        ax.scatter(j + offset, val, color=color, marker=marker, s=40,
                                   label=f'噪声 {noise}' if j == 0 else None)
            target = (1.0 if key == 'rank_correlation' else patient[mode]['both_rods']
                      if key == 'both_rods' else patient[mode]['rod_lag_ms']['median'])
            ax.axhline(target, color='black', ls='--', lw=1.1,
                       label='患者自身模板' if key == 'rank_correlation' else '患者 FIT')
            if key in ['rank_correlation', 'both_rods']:
                ax.set_ylim(-1.05 if key == 'rank_correlation' else -.05, 1.05)
            ax.set(title=f'{mode} · {title}', xticks=range(len(ids)),
                   xticklabels=[plot_label(cid) for cid in ids])
            ax.tick_params(axis='x', rotation=20, labelsize=8)
            ax.grid(alpha=.18)
            if row == 0:
                ax.legend(fontsize=8, loc='best')
    fig.suptitle(f'第 {batch} 批完整轨迹：共用拓扑种子，两次噪声分别展示；方向改变会重建 EE 边和时延\n'
                 '横轴为参数条件；联合点同时改变多个参数，不能据此归因单个参数；时差仅在两杆参与事件中计算', fontsize=12)
    fig.tight_layout(rect=(0, .24, 1, .93))
    parameter_table(fig, ids, height=.19)
    for ext in ['png', 'pdf']:
        fig.savefig(figures / f'batch{batch:02}_mode_tradeoffs.{ext}', dpi=150)
    plt.close(fig)
    raw = rt.read(analysis.A / 'raw/observables.json')
    # Scoring may finish before the asynchronous raw-observable report. Read
    # newly completed units directly instead of relying on that report's clock.
    from scripts.report_topic4_three_observable_raw import observables
    for cid in ids:
        for noise in plan['seeds']:
            r = lookup[cid, noise]
            key = f"{r['stage']}/{cid}/2511_{noise}"
            if key not in raw['runs']:
                _, times, labels, event_ids, names = analysis.load_small(Path(r['source']))
                raw['runs'][key] = {
                    mode: observables(times[event_ids[labels[event_ids] == k]], names)[0]
                    for mode, k in [('TA', 1), ('TB', 0)]}
    fig, axes = plt.subplots(2, len(ids) + 1, figsize=(19, 10), sharex=True, sharey=True)
    display = analysis.DISPLAY
    yy = np.arange(len(display))
    for row, mode in enumerate(['TA', 'TB']):
        target = raw['patient'][mode]
        means = np.array([target['contacts'][n]['mean'] for n in display])
        for col, cid in enumerate([None] + ids):
            ax = axes[row, col]
            if cid is None:
                lower = [target['contacts'][n]['q05'] for n in display]
                upper = [target['contacts'][n]['q95'] for n in display]
                ax.fill_betweenx(yy, lower, upper, color='#929aa2', alpha=.25)
            for ix in [slice(0, 4), slice(4, 15)]:
                ax.plot(means[ix], yy[ix], 'o-', color='black', lw=1, ms=2.7,
                        label='患者平均' if ix.start == 0 else None)
            if cid is not None:
                for noise, color, marker in [(847401, '#3276a8', 'o'), (847402, '#db7c36', '^')]:
                    r = lookup[cid, noise]
                    key = f"{r['stage']}/{cid}/2511_{noise}"
                    obs = raw['runs'][key][mode]
                    value = np.array([obs['contacts'][n]['mean'] for n in display], dtype=float)
                    for ix in [slice(0, 4), slice(4, 15)]:
                        ax.plot(value[ix], yy[ix], marker=marker, color=color, lw=1, ms=2.7,
                                label=f'{noise}' if ix.start == 0 else None)
            ax.set(xlim=(-.03, 1.03), ylim=(14.5, -.5), yticks=yy, yticklabels=display,
                   xlabel='归一化 rank', title=f'{mode} · ' + ('患者 FIT' if cid is None else plot_label(cid)))
            ax.axhline(3.5, color='#888888', lw=.8)
            ax.grid(alpha=.15)
            ax.tick_params(labelsize=8)
            if col == 1 and row == 0:
                ax.legend(fontsize=7, loc='lower left')
    fig.suptitle(f'第 {batch} 批：患者 TA/TB 平均模板与每个条件的模型平均模板\n'
                 '固定 SCL9–6 / ICL11–1 行序；逐事件在参与触点内归一化 rank，未参与不填零；患者阴影为事件间5–95%范围', fontsize=12)
    fig.tight_layout(rect=(0, .24, 1, .92))
    parameter_table(fig, ids, height=.19)
    for ext in ['png', 'pdf']:
        fig.savefig(figures / f'batch{batch:02}_patient_model_rank_templates.{ext}', dpi=150)
    plt.close(fig)
    later_path = out / f'proposals/adaptive_{batch+1:02}.json'
    later = rt.read(later_path) if later_path.exists() else None
    feedback = None
    if later:
        following = {r['candidate']: r for r in later['training_data']}
        for cid in proposal['ids']:
            exact = float(np.mean([lookup[cid, n]['J'] for n in plan['seeds']]))
            np.testing.assert_allclose(following[cid]['J'], exact, atol=1e-12)
        feedback = dict(next_batch=batch+1, training_conditions=len(following),
                        includes_exact_current_scores=True, trust_before=later['trust_before'],
                        trust_after=later['trust_after'])
    rt.write(dest / 'audit.json', dict(time=datetime.datetime.now().astimezone().isoformat(),
        status='COMPLETE_BATCH_SCORES_REVIEWED', batch=batch, completed_units=8,
        eligible_events=sum(r['N'] for r in units), previous_best=previous,
        batch_best=min(summary[-4:], key=lambda r: r['J'])['candidate'],
        next_proposal_feedback=feedback, conditions=summary,
        statistical_unit='One fixed topology and one noise realization; equal run weights, not event pooling.',
        scientific_acceptance='Not established by this descriptive score review; actual native and fresh topology review remain separate.'))
    (figures / 'README.md').write_text(
        f'### batch{batch:02}_mode_tradeoffs.png\n'
        '本批四个联合条件、固定参考及进入本批前的最优条件，分别展示 TA/TB 的平均 rank 模板相关、两杆参与比例和杆间质心中位差。'
        '蓝圆、橙三角是同条件、同图的两次噪声重演；各条件共用拓扑种子，但方向改变会按模型规则重建EE边和时延，不能当成完全相同连接图。黑虚线是患者参考；时间差仅对两杆均参与的事件定义，表格列出实际参数。'
        '**关注点**：时间差缩小是否伴随参与丢失或另一模式变差；本图不是单参数因果效应，也不是跨网络重复性结论。\n\n'
        f'### batch{batch:02}_mode_tradeoffs.pdf\n'
        '上一图的矢量版本，数据、坐标和比较定义相同。'
        '**关注点**：结合患者模板及原生多事件 GIF 阅读，不能将高 rank 相关单独解释为完整传播恢复。\n\n'
        f'### batch{batch:02}_patient_model_rank_templates.png\n'
        '左列为患者 TA/TB 的平均归一化 rank 与事件间5–95%范围，右列依次为各模型条件；每个模型面板保留黑色患者平均线。'
        '蓝、橙分别为两噪声运行的模式均值，只对参与接触点计算，所有面板保留固定电极行序、SCL/ICL分组和断线。'
        '**关注点**：模板形状是否相似，以及平均模板如何隐藏参与组合、逐事件变化及实际毫秒时差。\n\n'
        f'### batch{batch:02}_patient_model_rank_templates.pdf\n'
        '上一模板图的矢量版本，统计定义与数据相同。'
        '**关注点**：与原量对照图、患者真实 STFT 及模型原生动画联合阅读。\n')
    return dest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('batch', type=int)
    args = parser.parse_args()
    print(review(args.batch))
