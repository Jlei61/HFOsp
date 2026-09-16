"""Report the actual nominated 2-topology x 2-noise confirmation, without selection."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src/snn_engine')]
import numpy as np
from scripts import analyze_topic4_three_observable_bo as a
from scripts.topic4_three_observable_plot_labels import plot_label, parameter_table


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from src.topic4_pdf_font_guard import install
    install()
    plt.rcParams.update({'font.family':'Noto Sans CJK JP', 'font.size':10, 'pdf.fonttype':3})
    plan = a.rt.read(a.OUT/'plan.json')
    nomination = a.rt.read(a.OUT/'nomination.json')
    ids = nomination['ids']
    topologies = plan['confirmation_seeds']['topology']
    noises = plan['confirmation_seeds']['dynamics']
    pairs = [(t,n) for t in topologies for n in noises]
    records = a.records()
    lookup = {(r['candidate'],r['topology'],r['noise']):r for r in records if r['stage']=='confirmation'}
    expected = {(cid,t,n) for cid in ids for t,n in pairs}
    if not expected.issubset(lookup):
        raise RuntimeError(f'Confirmation incomplete: {len(expected.intersection(lookup))}/{len(expected)} scored units')
    static_checks = []
    for cid in ids:
        for topology in topologies:
            identities = [lookup[cid,topology,noise]['static_identity'] for noise in noises]
            same = all(identity == identities[0] for identity in identities[1:])
            if not same:
                raise RuntimeError(f'Static network changed between noise replays: {cid}, {topology}')
            static_checks.append(dict(candidate=cid,topology=topology,same_static_identity=same))
    from scripts.report_topic4_three_observable_raw import observables
    ev,names,_ = a.patient()
    patient = {mode:observables(ev.fit[ev.fit_labels==k],names)[0] for mode,k in [('TA',1),('TB',0)]}
    raw = {}
    for cid,t,n in sorted(expected):
        row = lookup[cid,t,n]
        _,times,labels,event_ids,_ = a.load_small(Path(row['source']))
        raw[cid,t,n] = {mode:observables(times[event_ids[labels[event_ids]==k]],names)[0]
                        for mode,k in [('TA',1),('TB',0)]}
    def pair_error(m,mode,shaft):
        values = [abs(v['order_probability']-patient[mode]['pairs'][key]['order_probability'])
                  for key,v in m['pairs'].items() if v['shaft']==shaft
                  and v['order_probability'] is not None
                  and patient[mode]['pairs'][key]['order_probability'] is not None]
        return float(np.mean(values)) if values else None
    dest = a.A/'confirmation_review'
    figures = dest/'figures'
    figures.mkdir(exist_ok=True,parents=True)
    colors = ['#3276a8','#d78035','#4b9b75','#9464a6']
    markers = ['o','^','s','D']
    labels = [f'图种子{t} / 噪声{n}' for t,n in pairs]
    def panel(ax, value, title, target=None, limits=None):
        for j,cid in enumerate(ids):
            for q,((t,n),color,marker,label) in enumerate(zip(pairs,colors,markers,labels)):
                val = value(cid,t,n) if lookup[cid,t,n]['physical_status']!='RUNAWAY' else None
                if val is not None and np.isfinite(val):
                    ax.scatter(j+(q-1.5)*.065,val,color=color,marker=marker,s=38,label=label if j==0 else None)
        if target is not None:
            ax.axhline(target,color='black',ls='--',lw=1,label='患者FIT参考')
        ax.set(title=title,xticks=range(len(ids)),xticklabels=[plot_label(c) for c in ids])
        if limits:ax.set_ylim(*limits)
        ax.tick_params(axis='x',rotation=15,labelsize=8)
        ax.grid(alpha=.15)
    fig,axes = plt.subplots(1,4,figsize=(17,6.5))
    panel(axes[0],lambda c,t,n:lookup[c,t,n]['J'],'三组联合训练目标 J ↓')
    for j,label in enumerate(['整体rank分布','杆内/杆间时序分布','参与结构分布']):
        panel(axes[j+1],lambda c,t,n,j=j:None if lookup[c,t,n]['components'] is None else lookup[c,t,n]['components'][j],label+'误差 ↓')
    axes[0].legend(fontsize=7,loc='best')
    fig.suptitle('提名后的新网络确认：每个点是一条完整运行；患者标签仍属于训练表征\n'
                 '新拓扑/噪声用于检验模型随机性的可重复性，不冒称新的患者外部验证',fontsize=12)
    fig.tight_layout(rect=(0,.25,1,.89));parameter_table(fig,ids,height=.16)
    for ext in ['png','pdf']:fig.savefig(figures/f'confirmation_three_objectives.{ext}',dpi=150)
    plt.close(fig)
    fig,axes = plt.subplots(2,5,figsize=(20,10))
    for i,mode in enumerate(['TA','TB']):
        panel(axes[i,0],lambda c,t,n:lookup[c,t,n]['raw'][mode].get('rank_correlation'),f'{mode} 平均rank相关 ↑',1,(-1.05,1.05))
        panel(axes[i,1],lambda c,t,n:pair_error(raw[c,t,n][mode],mode,'SCL'),f'{mode} SCL先后概率误差 ↓',0,(-.03,1.03))
        panel(axes[i,2],lambda c,t,n:pair_error(raw[c,t,n][mode],mode,'ICL'),f'{mode} ICL先后概率误差 ↓',0,(-.03,1.03))
        panel(axes[i,3],lambda c,t,n:raw[c,t,n][mode]['rod_lag_ms']['median'],f'{mode} SCL−ICL时差 (ms)',patient[mode]['rod_lag_ms']['median'])
        panel(axes[i,4],lambda c,t,n:raw[c,t,n][mode]['both_rods'],f'{mode} 两杆参与比例',patient[mode]['both_rods'],(-.03,1.03))
    axes[0,0].legend(fontsize=7,loc='best')
    fig.suptitle('新网络确认：平均模式、杆内顺序、杆间时差与参与分别展示\n'
                 '先后误差仅在共同可读触点对上定义；未出现或不可读不填零，实际支持量保存于表',fontsize=12)
    fig.tight_layout(rect=(0,.23,1,.91));parameter_table(fig,ids,height=.15)
    for ext in ['png','pdf']:fig.savefig(figures/f'confirmation_raw_observables.{ext}',dpi=150)
    plt.close(fig)
    for topology in topologies:
        fig,axes = plt.subplots(2,len(ids)+1,figsize=(4.2*(len(ids)+1),10),sharex=True,sharey=True,squeeze=False)
        yy = np.arange(15)
        for row,mode in enumerate(['TA','TB']):
            mean = np.array([patient[mode]['contacts'][name]['mean'] for name in a.DISPLAY],float)
            for col,cid in enumerate([None]+ids):
                ax = axes[row,col]
                if cid is None:
                    ax.fill_betweenx(yy,[patient[mode]['contacts'][name]['q05'] for name in a.DISPLAY],
                                     [patient[mode]['contacts'][name]['q95'] for name in a.DISPLAY],color='#929aa2',alpha=.25)
                for ix in [slice(0,4),slice(4,15)]:
                    ax.plot(mean[ix],yy[ix],'ko-',lw=1,ms=3,label='患者平均' if ix.start==0 else None)
                if cid is not None:
                    for noise in noises:
                        if lookup[cid,topology,noise]['physical_status']=='RUNAWAY':continue
                        q = pairs.index((topology,noise))
                        value = np.array([raw[cid,topology,noise][mode]['contacts'][name]['mean'] for name in a.DISPLAY],float)
                        for ix in [slice(0,4),slice(4,15)]:
                            ax.plot(value[ix],yy[ix],color=colors[q],marker=markers[q],lw=1,ms=3,label=str(noise) if ix.start==0 else None)
                ax.axhline(3.5,color='#888888',lw=.8)
                ax.set(xlim=(-.03,1.03),ylim=(14.5,-.5),yticks=yy,yticklabels=a.DISPLAY,xlabel='归一化rank',
                       title=f'{mode} · '+('患者FIT' if cid is None else plot_label(cid)))
                ax.grid(alpha=.15)
                if row==0 and col==1:ax.legend(fontsize=8,loc='best')
        fig.suptitle(f'新拓扑种子 {topology}：患者模板与每次噪声重演\n'
                     '未参与触点不填零；SCL/ICL分杆断线且固定15行；患者阴影为事件间5–95%范围',fontsize=12)
        fig.tight_layout(rect=(0,.2,1,.92));parameter_table(fig,ids,height=.13)
        for ext in ['png','pdf']:fig.savefig(figures/f'confirmation_templates_topology_{topology}.{ext}',dpi=150)
        plt.close(fig)
    rows = []
    for cid,t,n in sorted(expected):
        r = lookup[cid,t,n];baseline = lookup[plan['reference_id'],t,n]
        rows.append(dict(candidate=cid,topology=t,noise=n,N=r['N'],mode_counts=r['mode_counts'],J=r['J'],components=r['components'],
                         score_status=r['status'],physical_status=r['physical_status'],
                         paired_delta_J=None if r['J'] is None or baseline['J'] is None else r['J']-baseline['J'],
                         raw=raw[cid,t,n],source=r['source'],source_sha256=r['source_sha256'],static_identity=r['static_identity']))
    a.rt.write(dest/'confirmation_summary.json',dict(nomination=nomination,expected_units=len(expected),runs=rows,patient=patient,
        static_replication_checks=static_checks,
        statistical_unit='One condition-specific graph and noise realization; paired comparison across identical seed namespaces.',
        native_review='NOT_REPLACED_BY_THIS_REPORT',human_review='PENDING'))
    with PdfPages(figures/'confirmation_report.pdf') as pdf:
        from PIL import Image
        for path in sorted(figures.glob('*.png')):
            with Image.open(path) as im:
                fig,ax = plt.subplots(figsize=(17,11));ax.imshow(im);ax.axis('off');fig.tight_layout();pdf.savefig(fig);plt.close(fig)
    descriptions = {'confirmation_three_objectives':'新拓扑与新噪声确认的联合分数及三个分项，每条完整运行分别显示。',
                    'confirmation_raw_observables':'TA/TB的平均rank、SCL/ICL成对先后概率误差、杆间毫秒差和两杆参与分别与患者参考比较。'}
    for t in topologies:descriptions[f'confirmation_templates_topology_{t}']=f'拓扑种子{t}的两次噪声重演与患者TA/TB模板并列，保留固定分杆行序和缺失。'
    with (figures/'README.md').open('w') as f:
        for name,description in descriptions.items():
            for ext in ['png','pdf']:
                f.write(f'### {name}.{ext}\n{description}不同颜色/符号表示明确的拓扑与噪声组合；同一条件内两噪声固定图，方向不同的条件可重建EE边和时延。**关注点**：重复性、参与支持与条件分布分开看；该图不替代原生传播验图。\n\n')
        f.write('### confirmation_report.pdf\n汇集本次实际确认结果与患者模板，原始统计、实际事件数及物理状态见上级JSON。Runaway不进入正式比较图，事件不足不制造数值loss；新网络确认没有使用新的患者数据集。**关注点**：数值改善是否在多张图上保留，以及是否仍存在传播残差。\n')
    print(dest)


if __name__=='__main__':report()
