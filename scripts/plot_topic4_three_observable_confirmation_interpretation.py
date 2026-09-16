"""Show what changed in the frozen G3 scores, without changing those scores."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src/snn_engine')]
import numpy as np
from scripts import analyze_topic4_three_observable_bo as a
from scripts.topic4_three_observable_plot_labels import plot_label, parameter_table


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image
    from src.topic4_pdf_font_guard import install
    install()
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'pdf.fonttype':3})
    dest = a.A/'confirmation_review';figures = dest/'figures'
    ids = a.rt.read(a.OUT/'nomination.json')['ids']
    plan = a.rt.read(a.OUT/'plan.json')
    units = [(t,n) for t in plan['confirmation_seeds']['topology'] for n in plan['confirmation_seeds']['dynamics']]
    rows = [a.rt.read(p) for p in (a.OUT/'overnight_20260914/score_decomposition').glob('*.json')]
    lookup = {(r['candidate'],r['topology'],r['noise']):r for r in rows}
    summary = a.rt.read(dest/'score_interpretation.json')['conditions']
    colors = ['#3276a8','#d78035','#4b9b75','#9464a6'];markers = ['o','^','s','D']
    fig,axes = plt.subplots(2,3,figsize=(15,10))
    groups = [('rank_pattern','整体rank分布'),('local_and_interrod_timing','杆内/杆间时序分布'),('participation','参与结构分布')]
    for i,mode in enumerate(['TA','TB']):
        for j,(group,title) in enumerate(groups):
            ax = axes[i,j]
            for x,cid in enumerate(ids):
                for q,((t,n),color,marker) in enumerate(zip(units,colors,markers)):
                    val = lookup[cid,t,n]['conditional_feature_distances'][group][mode]['D_off']
                    ax.scatter(x+(q-1.5)*.06,val,color=color,marker=marker,s=40,label=f'{t} / {n}' if x==0 else None)
            ax.set(title=f'{mode} · {title}误差 ↓',xticks=range(len(ids)),xticklabels=[plot_label(c) for c in ids],ylabel='条件特征 Doff（未截断）')
            ax.grid(alpha=.15)
            if i==0 and j==0:ax.legend(title='拓扑 / 噪声',fontsize=8,title_fontsize=8)
    fig.suptitle('去除模型模式频率权重后，分别比较 TA、TB 自身的条件分布\n同一冻结特征映射和患者目标；仅作解释，不更改本轮提名或优化目标',fontsize=12)
    fig.tight_layout(rect=(0,.23,1,.91));parameter_table(fig,ids,height=.15)
    for ext in ['png','pdf']:fig.savefig(figures/f'conditional_distribution_distances.{ext}',dpi=150)
    plt.close(fig)
    fig,axes = plt.subplots(1,3,figsize=(16,7))
    patient_fraction = a.rt.read(a.A/'patient_reference.json')['proportions']['TB']
    for x,cid in enumerate(ids):
        for q,((t,n),color,marker) in enumerate(zip(units,colors,markers)):
            axes[0].scatter(x+(q-1.5)*.06,summary[cid]['TB_fractions'][q],color=color,marker=marker,s=42,label=f'{t} / {n}' if x==0 else None)
    axes[0].axhline(patient_fraction,color='black',ls='--',label='患者FIT自然比例')
    axes[0].set(title='TB事件比例：每条完整运行',ylim=(0,1));axes[0].legend(fontsize=7)
    bottom = np.zeros(len(ids))
    parts = [('global_features','全事件特征'),('mode_frequency','模式比例常数'),('TB_frequency_weighted_features','TB特征（含比例权重）'),('TA_frequency_weighted_features','TA特征（含比例权重）')]
    for (key,label),color in zip(parts,['#638aaa','#deb65c','#9b78ac','#65a38c']):
        values = [summary[cid]['J_contributions'][key] for cid in ids]
        axes[1].bar(range(len(ids)),values,bottom=bottom,color=color,label=label);bottom += values
    axes[1].set(title='四运行等权平均：对J的代数贡献');axes[1].legend(fontsize=7)
    x = np.arange(len(ids))
    axes[2].bar(x-.16,[summary[c]['A_mean'] for c in ids],width=.32,color='#608daf',label='A：均值嵌入差')
    axes[2].bar(x+.16,[summary[c]['B_mean'] for c in ids],width=.32,color='#db9867',label='B：有限事件减项')
    axes[2].set(title='J = A − B：改善来自哪里');axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set(xticks=range(len(ids)),xticklabels=[plot_label(c) for c in ids]);ax.grid(alpha=.15,axis='y')
    fig.suptitle('新网络确认：模式比例、联合目标构成与有限事件减项分开看\n代数贡献不能解释为患者方差比例；两个模式特征块仍混有频率权重，纯条件分布见另一图',fontsize=12)
    fig.tight_layout(rect=(0,.25,1,.87));parameter_table(fig,ids,height=.16)
    for ext in ['png','pdf']:fig.savefig(figures/f'mode_frequency_and_score_decomposition.{ext}',dpi=150)
    plt.close(fig)
    with (figures/'README.md').open('a') as f:
        for stem,description in [('conditional_distribution_distances','按TA/TB分别计算冻结特征的条件Doff，去掉模型模式频率因子；每点是一条完整确认运行。'),('mode_frequency_and_score_decomposition','分开显示自然模式比例、对联合J的代数贡献，以及A和B；患者FIT比例作为参考。')]:
            for ext in ['png','pdf']:
                f.write(f'\n### {stem}.{ext}\n{description}本图只解释已冻结结果，不加入新训练项或验收门槛。**关注点**：总分下降是否伴随两类各自的分布改善；分数贡献不是患者方差解释比例。\n')
    with PdfPages(figures/'confirmation_report.pdf') as pdf:
        for path in sorted(figures.glob('*.png')):
            with Image.open(path) as im:
                fig,ax=plt.subplots(figsize=(17,11));ax.imshow(im);ax.axis('off');fig.tight_layout();pdf.savefig(fig);plt.close(fig)
    print(figures)


if __name__=='__main__':plot()
