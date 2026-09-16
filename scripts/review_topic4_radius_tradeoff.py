"""Read completed, paired radius probes; do not change scoring or simulations."""
from pathlib import Path
import sys, json, hashlib, time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.topic4_pdf_font_guard import install

BASE = Path('/data/hfosp/topic4_sef_hfo')
F = BASE/'core_recruitment_tradeoff_followup_20260912'
OUT = BASE/'overnight_exploration_20260913/radius_tradeoff_review'
ANCHORS = [('circle', 'up3__circle__EE_core_to_out_scale_1.25', '圆核／向外EE×1.25', '#3478b8'),
           ('ellipse', 'up3__ellipse4__EI_same_core_scale_0.75', '椭圆／核内EI×0.75', '#bd7939')]
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    (OUT/'figures').mkdir(parents=True, exist_ok=True)
    lookup = {}
    for p in (F/'analysis/units').glob('*/result.json'):
        r = read(p); c = r['counts']
        lookup[(c['candidate'], c['topology'], c['noise'])] = (r, p)
    rows = []; sources = []; missing = []
    for kind, anchor, label, color in ANCHORS:
        for suffix in ['', '__radius205', '__radius235']:
            cid = anchor+suffix
            for noise in [847101, 847102]:
                key = (cid, 2511, noise)
                if key not in lookup:
                    missing.append(key); continue
                r, path = lookup[key]; c = r['counts']; trajectory = Path(r['source'])
                applied_path = trajectory.parent.parent/'applied_physics.json'
                applied = read(applied_path); th = applied['threshold']; counts = applied['group_counts']
                assert th['n_raised'] == 0
                modes = {o['mode']: o for o in r['observations'] if o['layer']=='primary'}
                pair = next(o for o in r['pairs'] if o['layer']=='primary' and o['mode']=='TB' and o['contact_i']=='ICL11' and o['contact_j']=='ICL9')
                rows.append(dict(kind=kind, candidate=cid, topology=2511, noise=noise,
                    radius_mm=th['radii_mm'][0], primary=c['primary'], TA=c['TA'], TB=c['TB'],
                    TA_fraction=c['TA']/c['primary'] if c['primary'] else np.nan,
                    TA_upper=modes['TA']['SCL_upper_participation'],
                    TA_order=modes['TA']['pair_order_probability_mae'],
                    TA_rod=modes['TA']['SCL_minus_ICL_lag_median_ms'],
                    TB_return=pair['model_i_precedes_j'], TB_pair_n=pair['model_joint_n'],
                    TB_rod=modes['TB']['SCL_minus_ICL_lag_median_ms'],
                    L_search=c['L_search'], A_cells=counts['coreAE'],
                    A_mean_lowering_mV=th['mean_lowering_per_member_mV'][0],
                    A_total_lowering_mV=th['total_lowering_per_core_mV'][0],
                    A_rate_per_cell_hz=c['coreAE_rate_hz'],
                    A_population_spikes_per_s=c['coreAE_rate_hz']*counts['coreAE'],
                    A_semiaxes_mm=th.get('ellipse_A_semiaxes_mm'),
                    local_width_ms=modes['ALL']['local_width_ms_median']))
                sources.append(dict(analysis=str(path), analysis_sha256=sha(path),
                    trajectory=str(trajectory), trajectory_sha256=sha(trajectory),
                    arrays_sha256=r['source_sha256'], applied=str(applied_path), applied_sha256=sha(applied_path)))
    d = pd.DataFrame(rows); d.to_csv(OUT/'per_run_observations.csv', index=False)
    for kind in d.kind.unique():
        z = d[d.kind==kind]
        assert np.ptp(z.A_total_lowering_mV.to_numpy()) < 1e-8
    install(); plt.rcParams.update({'font.family':'Noto Sans CJK JP', 'font.size':10, 'pdf.fonttype':3})
    fig, axes = plt.subplots(2, 3, figsize=(15,9))
    fig.subplots_adjust(left=.075, right=.98, top=.79, bottom=.21, hspace=.39, wspace=.30)
    metrics = [('TA_fraction','TA 标签占比',13165/19770,(0,1.02)),
               ('TA_upper','TA：上部SCL参与率',.84823395,(0,1.02)),
               ('TA_rod','TA：杆间质心差中位数 (ms)',-9.28,None),
               ('A_population_spikes_per_s','左核群体发放量 (spikes/s)',None,None),
               ('TB_return','TB：P(ICL11早于ICL9)',.693789711,(0,1.02)),
               ('TB_rod','TB：杆间质心差中位数 (ms)',1.188,None)]
    for ax,(metric,title,reference,ylim) in zip(axes.flat,metrics):
        for kind,anchor,label,color in ANCHORS:
            for noise,marker in [(847101,'o'),(847102,'^')]:
                z=d[(d.kind==kind)&(d.noise==noise)].sort_values('radius_mm')
                ax.plot(z.radius_mm,z[metric],c=color,marker=marker,mfc=color if noise==847101 else 'white',lw=1)
        if reference is not None: ax.axhline(reference,c='black',ls=':',lw=1)
        ax.set(title=title, xlabel='左核范围参数 r (mm)', xticks=[1.75257829056,2.05,2.35], xticklabels=['1.75','2.05','2.35'], xlim=(1.71,2.39))
        if ylim: ax.set_ylim(*ylim)
        ax.grid(alpha=.14)
    handles=[Line2D([],[],c=color,label=label) for _,_,label,color in ANCHORS]+[
        Line2D([],[],c='#555',marker='o',ls='none',label='噪声847101'),
        Line2D([],[],c='#555',marker='^',mfc='white',ls='none',label='噪声847102'),
        Line2D([],[],c='black',ls=':',label='患者FIT参考（有对应量时）')]
    fig.legend(handles=handles,ncol=3,loc='upper center',bbox_to_anchor=(.52,.93),frameon=False)
    fig.suptitle(f'扩大左核：招募、模式出现与原生活动的取舍｜已完成 {len(d)}/12 条',fontsize=17,y=.99)
    fig.text(.045,.055,'每点是一条完整60秒轨迹，固定基础网络2511；同色同点形配对，不是独立事件重复。连接已测离散点，不表示连续拟合。\n'
        'r 是成员范围参数；椭圆按4:1形状匹配参考圆的E成员数，实际半轴见CSV。两种颜色的EE/EI背景不同，不作纯形状因果比较。\n'
        '各范围匹配左核降阈值总量，但平均单细胞降幅、随机输入支持人数、I成员及连接分块会改变。群体发放量=逐细胞平均Hz×实际E成员数。\n'
        '杆间差为参与SCL质心中位数减参与ICL质心中位数，再取该类事件中位数。TA/TB为冻结标签；比例、参与、路径分别判断。\n'
        '缺失条件不填0；各类事件数、实际阈值和来源见CSV。原生群体发放量不是患者HFO能量或因果传播强度。',fontsize=9)
    for ext in ['png','pdf']: fig.savefig(OUT/'figures'/('radius_participation_and_activity.'+ext),dpi=155)
    plt.close(fig)
    (OUT/'manifest.json').write_text(json.dumps(dict(created_unix=time.time(),complete=len(d),planned=12,missing=missing,sources=sources,threshold_total_matched=True,physical_runs_added=0,producer=str(Path(__file__)),producer_sha256=sha(Path(__file__))),ensure_ascii=False,indent=2))
    table=['|工作点|r mm|噪声|TA/TB数|TA上部SCL|左核E数|平均降阈值 mV|左核群体发放 /s|', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for _,row in d.iterrows():
        table.append(f'|{"圆核" if row.kind=="circle" else "椭圆"}|{row.radius_mm:.2f}|{row.noise}|{row.TA}/{row.TB}|{row.TA_upper:.1%}|{row.A_cells}|{row.A_mean_lowering_mV:.3f}|{row.A_population_spikes_per_s:.0f}|')
    note=f'# 扩大左核的传播取舍\n\n已完成{len(d)}/12条运行；缺失列表见manifest.json。固定基础网络2511，两次开发噪声、每条60秒，排除前1.5秒。每条运行是实验单位，事件是运行内观测。圆核背景为向外EE×1.25／核内EI×1，椭圆背景为向外EE×1／核内EI×0.75，不能把两颜色之差当纯形状效应。\n\n'
    note+='\n'.join(table)+'\n\n'
    note+='各范围的左核总降阈值均为约529.953mV（逐细胞降幅之和），E成员数从720到1018、1358，平均降幅相应下降。输入仍按同样每细胞规律限制在核内，故随机输入支持、I成员与连接分块也改变；该设计不分离纯面积、全部输入剂量和全部局部连接效应。群体发放量是全分析时段的左核总spikes/s，既不是患者HFO能量，也不是一次传播的因果强度。\n\n'
    note+='已完成的较大范围对照显示：TA上部SCL招募上升可以伴TA标签数量／比例下降，核心群体总发放量却增加。因此不能把TA减少简单解释为左核总体活动变弱。TB杆间差和ICL局部顺序仍有明显残差，不能只按SCL参与率接受工作点。该现象仍可能涉及模式内过程组成、核心间时序、参与触点及核外传播，没有通过本次联合变化唯一确定机制。\n\n'
    note+='[六项响应图](figures/radius_participation_and_activity.png)保持原始量纲；全部逐运行参与／顺序／时差、标签支持及实际物理见CSV与manifest。患者FIT参考未改变，未新增损失、门槛、提名或物理运行。\n'
    (OUT/'scientific_note.md').write_text(note)
    (OUT/'figures/README.md').write_text('### radius_participation_and_activity.png\n完整半径探针在同一基础网络、相同噪声下配对，分别显示TA出现、SCL招募、杆间时差、核心群体发放量与TB局部顺序。范围改变时匹配降阈值总量，但不匹配所有外部输入与连接支持。\n**关注点**：更多核心活动或更多接触参与，是否同时带来患者相容的模式分布；不能由总发放量直接推断因果传播。\n')
    print(d.to_string(index=False),flush=True)

if __name__=='__main__': main()
