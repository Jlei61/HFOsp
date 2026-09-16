"""Eight existing shape confirmations: core phase versus observed rod timing."""
from pathlib import Path
import hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.topic4_pdf_font_guard import install

P=Path('/data/hfosp/topic4_sef_hfo/core_shape_output_response_20260911')
OUT=P.parent/'overnight_exploration_20260913/geometry_confirmation_effects/core_contact_timing'
CASES={'up3__circle':'上移后圆核','up3__ellipse4':'上移后等面积椭圆4:1'}
COLORS={2611:'#3675a9',2612:'#c67535'}

def main():
    F=OUT/'figures';F.mkdir(parents=True,exist_ok=True);rows=[];sources=[]
    for p in sorted((P/'analysis/units').glob('*/result.json')):
        r=json.loads(p.read_text());c=r['counts']
        if c['stage']!='confirmation' or c['candidate'] not in CASES:continue
        sources.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),trajectory=r['source'],arrays_sha256=r['source_sha256']))
        for e in r['events']:
            if not e['primary']:continue
            rows.append(dict(candidate=c['candidate'],topology=c['topology'],noise=c['noise'],mode=e['mode'],event=e['event'],core_dt_ms=e['B_minus_A_t10_ms'],rod_dt_ms=e['centroid_SCL_minus_ICL_ms']))
    assert len(sources)==8
    d=pd.DataFrame(rows);d.to_csv(OUT/'events.csv',index=False);summ=[]
    for keys,z in d.groupby(['candidate','topology','noise','mode']):
        q=dict(zip(['candidate','topology','noise','mode'],keys));q['n']=len(z)
        for col in ['core_dt_ms','rod_dt_ms']:
            v=z[col].dropna();q[col+'_valid_n']=len(v)
            for name,value in zip(['q05','median','q95'],np.quantile(v,[.05,.5,.95])):q[col+'_'+name]=float(value)
        v=z.core_dt_ms.dropna();q['left_core_t10_earlier_fraction']=float((v>0).mean());summ.append(q)
    s=pd.DataFrame(summ);s.to_csv(OUT/'per_run_summary.csv',index=False)
    assert len(s)==16 and (s.loc[s['mode']=='TA','left_core_t10_earlier_fraction']==1).all()
    ref=json.loads((P/'analysis/patient_reference.json').read_text())['modes']['TA']
    install();plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'pdf.fonttype':3})
    fig,axes=plt.subplots(2,2,figsize=(12,10),sharex=True,sharey=True)
    fig.subplots_adjust(left=.1,right=.98,bottom=.19,top=.87,wspace=.2,hspace=.22)
    for i,topo in enumerate([2611,2612]):
      for j,(cid,title) in enumerate(CASES.items()):
        ax=axes[i,j];ax.axhspan(ref['SCL_minus_ICL_lag_q05_ms'],ref['SCL_minus_ICL_lag_q95_ms'],color='.93')
        ax.axhline(ref['SCL_minus_ICL_lag_median_ms'],c='black',ls=':',lw=1);ax.axhline(0,c='.65',lw=.8)
        for noise,marker,filled in [(847201,'o',True),(847202,'^',False)]:
            z=d[(d.candidate==cid)&(d.topology==topo)&(d.noise==noise)&(d['mode']=='TA')]
            valid=z.dropna(subset=['core_dt_ms','rod_dt_ms'])
            ax.scatter(valid.core_dt_ms,valid.rod_dt_ms,marker=marker,facecolors=COLORS[topo] if filled else 'none',edgecolors=COLORS[topo],s=35,alpha=.75,label=f'{noise}：可测{len(valid)}／TA{len(z)}')
        ax.set(title=f'{title}｜网络{topo}',xlabel='右核t10 − 左核t10 (ms)',ylabel='SCL − ICL 质心时差 (ms)',xlim=(0,160),ylim=(-85,125));ax.grid(alpha=.12);ax.legend(frameon=False,fontsize=8,loc='upper right')
    fig.suptitle('核先后保持，电极先后却可能明显改变',fontsize=18,y=.98)
    fig.text(.1,.91,'TA条件分布；2种形状 × 2张基础网络 × 2噪声，全部既有确认轨迹',fontsize=11)
    fig.text(.07,.055,'每点是单条记录内的TA事件；颜色表示网络，实心圆／空心三角表示噪声。横轴正值表示左核窗口内累计10%活动较早。\n八条运行的TA均为左核较早；这不等于左核是唯一因果起源。纵轴仍受参与触点集合和原生传播共同影响。\n黑虚线／灰区为患者TA中位数／5–95%事件范围，患者没有可直接对应的模型core时间。核心时序不能单独证明路径恢复。\n均为核内EI=1、离核EE=1；形状改变core成员、输入支持与局部连接分块。事件不是独立网络复制，未改变loss或仿真。',fontsize=9)
    for ext in ['png','pdf']:fig.savefig(F/f'core_phase_and_rod_timing.{ext}',dpi=155)
    plt.close(fig)
    note='''# 核先后能否解释几何效应的跨网络反向？

重新读取原确认中的圆核／等面积椭圆、两张基础网络2611/2612、两条噪声847201/847202，共8条完整60秒记录。只用原primary事件与原窗口累计10%核心活动时间，不重新检测、不修改标签或损失。统计单位为运行，图中点为运行内事件。

这8条运行所有TA事件的左核t10都早于右核。网络2611的核间t10差中位数在圆核约65ms、椭圆约69ms，而TA杆间时差由约88ms变为−26/−17ms。网络2612的核间差约84/79ms变为71/72ms，杆间时差却由−4.5/+1.0ms变为+4.2/+10.2ms。因此不能将跨网络几何响应的反向，简单解释为TA中领先core换了一边。

这仍不是因果定位：t10是事件窗内后验累计质量时间，不是真正起燃时间；形状也改变被标为TA的事件集合。杆间质心受实际参与接触点与核外传播共同影响。证据把问题指向核心活动如何被传播和采样成电极时序，但尚未区分连接结构、局部招募或采样各自的贡献。TB逐运行数据也保留在表中，不筛掉较差事件。

同噪声下实际两核OU调制率相同的检查见上级ou_input_interpretation.md；它不涵盖逐细胞Poisson创新。这8条不能替代新EE/EI工作点自己的多网络确认。
'''
    (OUT/'scientific_note.md').write_text(note)
    (F/'README.md').write_text('### core_phase_and_rod_timing.png\n固定TA标签，按网络分行、形状分列展示核心累计活动先后与杆间质心差；两个噪声分别保留。所有点来自8条既有完整确认运行。\n**关注点**：领先core未交换，接触时序仍大幅变化；窗口累计时间不等于因果起源。\n')
    (OUT/'manifest.json').write_text(json.dumps(dict(status='COMPLETE_PENDING_VISUAL_REVIEW',created_unix=time.time(),sources=sources,analysis_units=8,events=len(d),TA_events=int((d['mode']=='TA').sum()),all_TA_left_core_t10_earlier=True,new_simulations=0,producer=str(Path(__file__)),producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    print(s.to_string(index=False));print(OUT)

if __name__=='__main__':main()
