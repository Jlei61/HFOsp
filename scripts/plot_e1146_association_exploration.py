#!/usr/bin/env python3
"""Figures and scientific interpretation for the bounded 22-readout screen."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.explore_e1146_seizure_interictal_association import OUT,META
from scripts.analyze_e1146_source_signed_correspondence import RED,BLUE
WINDOW_NAMES={'whole':'Whole interval','pre60':'Last 60 min','pre15':'Last 15 min','halves':'Second half − first half','last60_change':'Last full hour − earlier'}
ZH={'whole':'整段','pre60':'末60分钟可用部分','pre15':'末15分钟可用部分','halves':'后半减前半','last60_change':'完整末小时减较早部分'}
LABELS_ZH={'TA share':'TA比例','TA event rate':'TA事件率','TB event rate':'TB事件率','Total event rate':'总事件率','Same-label adjacency excess':'同型相邻超额概率','TA share in last 20 events':'最后20事件的TA比例','TA-relative event recency':'TA相对近期性','TA share change':'TA比例变化','TA rate change':'TA率变化','TB rate change':'TB率变化'}


def save(fig,name):
    for ext in ('png','pdf'):fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def plot(f,s,w):
    (OUT/'figures').mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(14,5.3),layout='constrained')
    for ax,key,title in zip(axes,('whole_p_ta','whole_log_rate_tb','whole_log_rate_all'),('TA proportion','TB event rate','Total event rate')):
        is_rate='rate' in key
        for x,lab,color in [(0,'TA',RED),(1,'TB',BLUE)]:
            g=f[(f.label==lab)&np.isfinite(f[key])].copy();values=np.expm1(g[key]) if is_rate else g[key]*100
            offsets=np.linspace(-.22,.22,len(g));np.random.default_rng(20260909).shuffle(offsets)
            for j,(_,r),value in zip(offsets,g.iterrows(),values):
                ax.scatter(x+j,value,c=color,s=40,zorder=3)
                if r.sz in (11,13,14,15,16,17,19,20,21,25,26):ax.annotate(str(r.sz),(x+j,value),xytext=(3,2),textcoords='offset points',fontsize=8)
            med=np.median(values);ax.plot([x-.3,x+.3],[med,med],c=color,lw=3)
        row=s[s.feature==key].iloc[0]
        ax.set(xticks=[0,1],xticklabels=[f'TA-source\nn={row.n_ta}',f'TB-source\nn={row.n_tb}'],xlim=(-.45,1.45),
            title=f'{title}\np={row.p:.3g}; BH q={row.q_bh22:.3g}',ylabel='Events / observed hour' if is_rate else 'TA share (%)')
        if is_rate:
            ax.set_yscale('symlog',linthresh=1);ax.set_ylim(-.12,6000);ax.set_yticks([0,1,10,100,1000]);ax.set_yticklabels(['0','1','10','100','1,000'])
        else:ax.set_ylim(0,105);ax.axhline(67.6866,c='#777777',ls=':',lw=.8)
    fig.supxlabel('Whole interval; each point is one seizure, bars are group medians. Rates retain two observed zero-event intervals.',fontsize=10)
    save(fig,'composition_vs_event_rate')
    fig,axes=plt.subplots(1,2,figsize=(14,5.5),layout='constrained')
    ax=axes[0]
    for _,r in f.iterrows():
        x,y=np.expm1(r.whole_log_rate_ta),np.expm1(r.whole_log_rate_tb)
        ax.scatter(x,y,c=RED if r.label=='TA' else BLUE,s=45)
        if r.sz in (5,14,15,16,17,19,20,21,24,25,26):ax.annotate(str(r.sz),(x,y),xytext=(3,3),textcoords='offset points',fontsize=9)
    grid=np.linspace(0,3500,200);ax.plot(grid,grid*(1-.676866)/.676866,c='#777777',ls=':',lw=1,label='Patient background ratio')
    ax.set_xscale('symlog',linthresh=1);ax.set_yscale('symlog',linthresh=1)
    ax.set(xlim=(-.15,4000),ylim=(-.15,1600),xlabel='TA events / observed hour',ylabel='TB events / observed hour',title='TA and TB event rates covary')
    for axis in (ax.xaxis,ax.yaxis):axis.set_ticks([0,1,10,100,1000]);axis.set_ticklabels(['0','1','10','100','1,000'])
    ax=axes[1]
    for _,r in f.iterrows():
        ax.scatter(r.time_hours,np.expm1(r.whole_log_rate_all),c=RED if r.label=='TA' else BLUE,s=45)
        if r.sz in (5,14,15,16,17,19,20,21,24,25,26):ax.annotate(str(r.sz),(r.time_hours,np.expm1(r.whole_log_rate_all)),xytext=(3,3),textcoords='offset points',fontsize=9)
    ax.set_yscale('symlog',linthresh=1);ax.set_yticks([0,1,10,100,1000]);ax.set_yticklabels(['0','1','10','100','1,000'])
    ax.set(xlabel='Hours since first observed seizure',ylabel='Total events / observed hour',ylim=(-.15,6000),title='Seizure type also clusters in time')
    fig.legend(handles=[Line2D([],[],marker='o',lw=0,c=RED,label='TA-source seizure'),Line2D([],[],marker='o',lw=0,c=BLUE,label='TB-source seizure'),Line2D([],[],ls=':',c='#777777',label='Patient background TA/TB ratio')],loc='outside lower center',ncol=3,frameon=False,fontsize=11)
    save(fig,'joint_rates_and_chronology')
    fig,axes=plt.subplots(1,2,figsize=(14,10.5),sharey=True,gridspec_kw={'width_ratios':[1,1.1]},layout='constrained')
    s=s.set_index('feature').loc[list(META)].reset_index()
    ax=axes[0]
    for i,r in s.iterrows():
        color=BLUE if r.rank_biserial>0 else RED
        ax.plot([r.leave_one_out_min,r.leave_one_out_max],[i,i],c=color,lw=1.5)
        ax.scatter(r.rank_biserial,i,c=color,s=28)
    ax.set(yticks=np.arange(len(s)),yticklabels=[f'{WINDOW_NAMES[r.window]}: {r.readout}' for _,r in s.iterrows()],
           ylim=(len(s)-.5,-.6),xlim=(-1.02,1.02),xlabel='Rank effect: higher before TA  ←  0  →  higher before TB',title='All 22 planned readouts\nWhiskers: leave-one-seizure-out range (not CI)')
    ax.axvline(0,c='#999999',lw=.8);ax.tick_params(axis='y',labelsize=9)
    ax=axes[1];ax.set_xlim(0,1);ax.axis('off')
    ax.set_title('n (TA/TB)       raw p       BH q       6h-stratified p',fontsize=11)
    for i,r in s.iterrows():
        if i%2==0:
            for a in axes:a.axhspan(i-.5,i+.5,color='#f4f4f4',zorder=-1)
        for x,txt in zip([.1,.37,.61,.86],[f'{r.n_ta}/{r.n_tb}',f'{r.p:.3f}',f'{r.q_bh22:.3f}',f'{r.block6_p:.3f}']):ax.text(x,i,txt,ha='center',va='center',fontsize=10)
    save(fig,'all_candidate_associations')
    # One clock stratum's constrained null makes the time-confounding limit concrete.
    (OUT/'figures/README.md').write_text('''# 发作标签与发作前间期事件：22项有界探索

### composition_vs_event_rate.png / .pdf
同一患者中比较两种source发作前的整段TA比例、TB事件率和总事件率，每点为一次发作，短横线为组中位数。事件率按有标签资产的实际观察小时数计算；有覆盖却0事件的两段进入率统计，不能进入比例统计；率轴在0附近线性、其余对数，数字为发作编号。
**关注点**：TB事件率比模式比例更能区分当前两组，但总事件率也升高，不能直接归因于TB特异招募；p为探索性秩置换，q在22项内校正。

### joint_rates_and_chronology.png / .pdf
左图为每次发作前TA与TB的事件率联合分布，点线为患者背景比例；右图按真实小时数显示总事件率与source标签。红/蓝表示发作的TA/TB source标签，不表示点本身是某个间期事件。
**关注点**：两种模式常共同增加，TB型发作又集中在中间记录时段，总体活动状态与时间趋势均可能影响联系。

### all_candidate_associations.png / .pdf
完整展示预先列出的22项统计，左侧为秩效应及删去单次发作的效应范围，右侧为样本数、未校正p、22项BH q及6小时时间层内置换p。正效应表示TB-source组的该读数较高；区间不是置信区间。
**关注点**：不只展示最小p；没有指标达到本探索池q<0.05，时间层内比较的信息更有限。6小时层只约束统计标签置换，完全不恢复整小时事件排除规则。图待用户目视检查。
''',encoding='utf-8')


def report(f,s,w):
    index=s.set_index('feature');whole=w[w.window=='whole'];main=index.loc['whole_log_rate_tb']
    lines=['# E1146：发作类型与发作前间期事件统计的有界探索','','本轮按 `PLAN.md` 在查看新特征结果前固定22项候选。最值得继续验证的线索是事件率，尤其TB事件率；模式比例、近期模式组成、序列连续性和临近发作变化未给出更强联系。当前没有指标在22项BH校正后达到q<0.05，且时间相近条件下比较的证据不足，因此不能宣称已建立模式特异联系或预测能力。','','## 1. 科学问题与设计','','保留冻结空间事件标签与已确认的signed source发作标签。检验的是下一次TA-source或TB-source发作之前，间期事件的构成、强度、顺序或变化是否不同。每次发作是一个样本，事件数不是统计重复；首发排除，SZ4不明确、SZ6无正相关、SZ18无能量基线不强制归类，主比较为15次TA-source与7次TB-source。两段有观察时间但无事件的TA-source发作可进入事件率统计，因此率n=22，比例n=20。','','使用逐事件实际起止，不能跨上一发作边界，不再整块排除，也不因为覆盖率低于50%丢弃间隔。率的分母是实际标签资产覆盖小时数，而不是整段日历时长；80个packed与80个lagPat文件对应，未发现额外空事件资产可补充覆盖。SZ25仍仅29%时间覆盖，其率只代表已观测部分。','','22项包含：三个时间窗各自的TA比例、TA率、TB率、总率（12项）；两窗的成分校正同型相邻概率（2项）；最后20事件组成与两类事件近期性（2项）；整段前后半、末完整小时相对较早部分的比例与两模式率变化（6项）。总率是活动强度对照，同型连续性是顺序结构对照；它们有助于判断是否真是模式特异偏向。','','## 2. 事件率是当前最强候选','','|整段统计|TA-source组中位数|TB-source组中位数|探索性秩置换p|22项BH q|6小时内置换p|','|---|---:|---:|---:|---:|---:|']
    for key in ('whole_p_ta','whole_log_rate_ta','whole_log_rate_tb','whole_log_rate_all'):
        r=index.loc[key];g=f.groupby('label')[key]
        if 'rate' in key:
            va=np.median(np.expm1(g.get_group('TA')));vb=np.median(np.expm1(g.get_group('TB')));a=f'{va:.1f}次/小时';b=f'{vb:.1f}次/小时'
        else:a=f"{g.get_group('TA').median():.1%}";b=f"{g.get_group('TB').median():.1%}"
        lines.append(f'|{LABELS_ZH[r.readout]}|{a}|{b}|{r.p:.4f}|{r.q_bh22:.3f}|{r.block6_p:.3f}|')
    lines += ['',f"整段TB率的描述性AUC为{main.auc_tb:.3f}（正类TB-source；含义为随机取两类各一次发作，TB型此前TB率更高的概率，平局计半），不是交叉验证后的预测AUC。删去任意单次发作，秩效应保持{main.leave_one_out_min:.3f}–{main.leave_one_out_max:.3f}，因此方向不靠单一离群发作。与连续早期能量偏向 r(TA)−r(TB) 的Spearman相关为{main.spearman_source_score:.3f}，只作描述，不开启另一套p值搜索。",'', '末15分钟的TB率也有同向候选信号（未校正p=0.0080，22项q=0.088），末60分钟更弱。各窗高度重叠，不是独立复制。不能把q=0.088在看到结果后改称通过q<0.1。','','## 3. 为什么还不能说是TB特异联系','','TA-source之前的TA率中位数164.7次/小时、TB-source之前950.4次/小时；总率也由195.6升至1113.2次/小时。TA与TB在两类发作之间往往同向偏高；这是组间活动水平差，不是临近同一次发作逐渐升高的证据。TB绝对率分离更强，但仅此不足以排除总体活动状态共同升高。固定总事件数后，TB数的信息就是模式比例；本轮比例统计较弱，因此没有证明总强度之外的TB特异成分。','','例如，SZ15（TB-source）前TA≈2808、TB≈927次/小时，而SZ16（TA-source）前TA≈692、TB≈271次/小时；SZ20虽为TA-source，TB率≈374次/小时，高于SZ21–23三个TB-source发作前的TB率。这些反例表明不存在一个简单的TB率阈值把两类稳定分开。','','## 4. 时间、事件数与统计方法的限制','',f"TB-source集中在首发后约14.6–25.2小时。整段TB率的6小时层内置换p={main.block6_p:.3f}，12小时层内p={main.block12_p:.3f}，循环标签平移p={main.circular_shift_p:.3f}。6小时层只有{int(main.block6_n_mixed_time_blocks)}个同时包含两类的时间层，共{int(main.block6_n_permutations)}种标签分配；12小时有{int(main.block12_n_permutations)}种。这个结果表明目前无法把联系与时间状态清楚分开，也可能受层内样本少限制，不能反过来证明完全由时间造成。时间层只限制统计标签置换，不删除任何来源小时块。",'',f"在与整段比例相同的20次有事件发作上，TB率秩效应仍为{main.common_event_cohort_rank_biserial:.3f}、未校正p={main.common_event_cohort_p:.4f}；至少20事件的18次发作上效应为{main.n20_rank_biserial:.3f}、p={main.n20_p:.4f}、本敏感性22项q={main.n20_q_bh22:.3f}。排除零事件后方向保留，但低计数和样本数影响证据强度。",'', '上一轮比例检验针对等权均值差（p=0.376）；本轮预先选择秩统计，整段比例p=0.643。它们是不同统计量，不应混为一次检验或择较小p报告。本轮秩效应适合偏斜率数据和极端值，但舍弃了实际幅度，故同时报告原单位组中位数。', '', '22项各自精确置换，保留该特征可观测样本内的标签组数；多重校正在本轮候选池内进行，不能消除历史探索选择。时间依赖会影响普通标签置换的交换性，因此未校正及BH结果都只是探索线索。留一发作范围不是置信区间；不训练小样本多变量分类器、不声称前瞻预测。','','## 5. 已探索的其他联系与下一步判断','','较近期事件的TA/TB组成、哪类事件离发作更近、扣除模式数量后是否更易同型连续，以及同一间隔末期是否出现模式转向，本轮都未提供明确证据。它们仍是不同科学问题，不能仅因某一率指标p较小就把所有问题合并为“传播模式匹配”。','','建议保留“活动强度与发作source标签可能有关”作为待验证候选，暂不接受“同型间期模式特异地预示同型发作”。最合适的下一层模型应同时分开总强度 λA+λB 与模式构成 λB/(λA+λB)，并考虑发作后时间与记录时间状态；否则TB率高仍可能只是两种模式一起增多。当前只有7次TB-source且时间集中，复杂回归容易把时间段记住。需要不同时间段都有两类发作、或扩展到有相同来源合同的更多患者，再用按患者/连续记录段留出的验证判断增量信息。此为后续设计建议，本轮未追加模型拟合或扩大队列。','','## 6. 完整22项结果','','正秩效应表示TB-source组该读数更高；对TA比例/TA近期性这种方向量，负效应才符合TB偏向。','', '|窗口|统计|n(TA/TB)|秩效应|原始p|BH q|6小时内p|','|---|---|---:|---:|---:|---:|---:|']
    for key in META:
        r=index.loc[key];lines.append(f'|{ZH[r.window]}|{LABELS_ZH[r.readout]}|{r.n_ta}/{r.n_tb}|{r.rank_biserial:+.3f}|{r.p:.4f}|{r.q_bh22:.3f}|{r.block6_p:.3f}|')
    lines += ['', '## 7. 复现、图与方法来源','','代码：`scripts/explore_e1146_seizure_interictal_association.py` 计算特征和精确检验；`scripts/plot_e1146_association_exploration.py` 生成报告与3张PNG/PDF。使用nd2环境，OMP/OPENBLAS/MKL线程数均为1。`seizure_features.csv`、`window_observations.csv`、`feature_support_counts.csv`为可核对分母；`association_screen.csv`包含全部效应、p、q、时间限制与敏感性。', '', '5项针对性测试验证已知秩置换、纯时间混杂、混合时间层、缺口与计数组成校正的顺序统计、BH校正。46,683个原始事件索引和SQL起止对齐；整段计数与前轮逐事件分析逐一相等。图已Agent目视检查，用户目视验收待定。', '', '交换性和受限置换依据：[Winkler et al., 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4010955/)。时间依赖数据的验证边界参考：[Roberts et al., 2017](https://www.wsl.ch/lud/biodiversity_events/papers/Roberts_et_al-2017-Ecography.pdf)。这些文献支持方法边界，不是E1146生物学结果的外部证据。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    f=pd.read_csv(OUT/'seizure_features.csv');s=pd.read_csv(OUT/'association_screen.csv');w=pd.read_csv(OUT/'window_observations.csv')
    plot(f,s,w);report(f,s,w)

if __name__=='__main__':main()
