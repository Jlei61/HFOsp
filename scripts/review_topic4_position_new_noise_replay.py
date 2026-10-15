"""Keep development and new noise identities separate for a fixed XY contrast."""
from pathlib import Path
import sys,json,hashlib,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.topic4_pdf_font_guard import install
B=Path('/data/hfosp/topic4_sef_hfo');F=B/'core_recruitment_tradeoff_followup_20260912';N=B/'core_multiseed_response_curves_20260913'
OUT=B/'overnight_exploration_20260913/position_new_noise_replay'
OLD=['up3__circle__EE_core_to_out_scale_1.25','up3__circle__EE_core_to_out_scale_1.25__x_minus075']
NEW=['bridge_circle_out125','bridge_circle_out125_xminus075']

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    D=OUT/'figures';D.mkdir(parents=True,exist_ok=True);lookup={};sources=[];rows=[];statics={}
    for root,ids,seeds,role in [(F,OLD,[847101,847102],'development'),(N,NEW,[847401,847402],'new_noise')]:
        found={}
        for p in (root/'analysis/units').glob('*/result.json'):
            r=read(p);c=r['counts']
            if c['candidate'] in ids and c['topology']==2511 and c['noise'] in seeds:found[(c['candidate'],c['noise'])]=(r,p)
        for seed in seeds:
            if not all((cid,seed) in found for cid in ids):continue
            for shifted,cid in enumerate(ids):
                r,p=found[(cid,seed)];c=r['counts'];path=Path(r['source']);rr=read(path)
                assert rr['actual_duration_ms']==60000 and rr['physical_status']=='COMPLETE_NO_RUNAWAY'
                if shifted in statics:assert rr['static_array_identity']==statics[shifted]
                else:statics[shifted]=rr['static_array_identity']
                modes={x['mode']:x for x in r['observations'] if x['layer']=='primary'}
                pairs={x['mode']:x for x in r['pairs'] if x['layer']=='primary' and x['contact_i']=='ICL11' and x['contact_j']=='ICL9'}
                z=dict(role=role,noise=seed,topology=2511,candidate=cid,shifted=shifted,n=c['primary'],TA_n=c['TA'],TB_n=c['TB'],TA_fraction=c['TA']/c['primary'],L_search=c['L_search'])
                for mode in ['ALL','TA','TB']:
                    for key in ['SCL_upper_participation','participation_mae','pair_order_probability_mae','SCL_minus_ICL_lag_median_ms','local_width_ms_median']:z[mode+'_'+key]=modes[mode][key]
                    z[mode+'_return_probability']=pairs[mode]['model_i_precedes_j'];z[mode+'_pair_n']=pairs[mode]['model_joint_n']
                rows.append(z);lookup[(seed,shifted)]=z;sources.append(dict(analysis=str(p),analysis_sha256=sha(p),trajectory=str(path),trajectory_sha256=sha(path),arrays_sha256=rr['arrays_sha256']))
    d=pd.DataFrame(rows);assert len(d)>=6;d.to_csv(OUT/'per_run_observations.csv',index=False)
    effects=[]
    for seed in sorted(d.noise.unique()):
        a,b=lookup[(seed,0)],lookup[(seed,1)];z=dict(noise=int(seed),role=a['role'],topology=2511)
        for key in [k for k in a if k.startswith(('TA_','TB_','ALL_'))]+['L_search']:z[key]=b[key]-a[key]
        effects.append(z)
    pd.DataFrame(effects).to_csv(OUT/'paired_differences.csv',index=False)
    ref=read(N/'analysis/patient_reference.json');patient_pairs=read(next((N/'analysis/units').glob('*/result.json')))['pairs']
    p_return=next(x['patient_i_precedes_j'] for x in patient_pairs if x['mode']=='TB' and x['layer']=='primary' and x['contact_i']=='ICL11' and x['contact_j']=='ICL9')
    metrics=[('TA_SCL_upper_participation','TA：上部SCL参与',ref['modes']['TA']['SCL_upper_participation']),('TA_pair_order_probability_mae','TA：成对顺序概率误差',0.),('TA_SCL_minus_ICL_lag_median_ms','TA：杆间质心差中位数 (ms)',ref['modes']['TA']['SCL_minus_ICL_lag_median_ms']),('TB_return_probability','TB：P(ICL11早于ICL9)',p_return),('TB_pair_order_probability_mae','TB：成对顺序概率误差',0.),('TB_SCL_minus_ICL_lag_median_ms','TB：杆间质心差中位数 (ms)',ref['modes']['TB']['SCL_minus_ICL_lag_median_ms'])]
    colors={847101:'#a6a6a6',847102:'#747474',847401:'#1976a2',847402:'#bc6b28'};markers={847101:'o',847102:'^',847401:'s',847402:'D'}
    install();plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'pdf.fonttype':3})
    fig,axes=plt.subplots(2,3,figsize=(15,9));fig.subplots_adjust(left=.075,right=.99,bottom=.2,top=.82,wspace=.29,hspace=.36)
    for ax,(key,label,pv) in zip(axes.flat,metrics):
        ax.axhline(pv,c='black',ls=':',lw=1)
        for seed in sorted(d.noise.unique()):
            vals=[lookup[(seed,k)][key] for k in [0,1]];ax.plot([0,1],vals,c=colors[seed],marker=markers[seed],ls='--' if seed<847400 else '-',lw=1 if seed<847400 else 2,ms=5)
        ax.set(xticks=[0,1],xticklabels=['原位置\nx=4.20mm','左移0.75mm\nx=3.45mm'],ylabel=label,xlim=(-.18,1.18));ax.grid(alpha=.13)
        if 'participation' in key or 'probability' in key and 'mae' not in key:ax.set_ylim(-.02,1.02)
        if 'mae' in key:ax.set_ylim(0,.45)
    handles=[Line2D([],[],c=colors[s],marker=markers[s],ls='--' if s<847400 else '-',label=('开发噪声' if s<847400 else '新噪声')+str(s)) for s in sorted(d.noise.unique())]+[Line2D([],[],c='black',ls=':',label='患者FIT参考／误差0')]
    fig.legend(handles=handles,ncol=3,loc='upper center',bbox_to_anchor=(.52,.935),frameon=False)
    fig.suptitle('左核位置的作用，在新噪声下是否保留？',fontsize=18,y=.99)
    fig.text(.075,.055,'固定基础网络2511、圆核、向外EE×1.25，其余参数固定；位置改变实际core成员与连接分块。每个点是一条完整60秒运行的摘要。\n灰线为两次开发噪声，彩线为已完成的新噪声直接配对；同一位置在不同噪声中的全部静态数组身份已核对一致。不是不同网络。\nTA/TB由冻结分类器组织；杆间差=SCL参与触点质心中位数−ICL参与触点质心中位数，再取该类事件中位数。\n每条运行的实际事件数、模式比例、全部不分类观测与训练分数见CSV；连线不是连续参数曲线或置信区间。缺失新噪声不补值。',fontsize=9)
    for ext in ['png','pdf']:fig.savefig(D/f'position_effect_old_and_new_noise.{ext}',dpi=155)
    plt.close(fig)
    note='''# 位置响应的新噪声重演

问题是已经提名的圆核向外EE×1.25工作点，左核x减少0.75mm的作用能否在新噪声下保留。固定基础网络2511、60秒时长、1.5秒burn-in和原损失／资格。两次开发噪声只作开发参照；新噪声没有用于重选本次位置。每条运行是实验单位，事件不是独立网络复制。

'''+'|噪声|合格事件 原→左移|TA数 原→左移|TA上部SCL 原→左移|TA杆间差(ms) 原→左移|TB杆间差(ms) 原→左移|\n|---|---:|---:|---:|---:|---:|\n'
    for seed in sorted(d.noise.unique()):
        a,b=lookup[(seed,0)],lookup[(seed,1)];note+=f'|{seed}|{a["n"]}→{b["n"]}|{a["TA_n"]}→{b["TA_n"]}|{a["TA_SCL_upper_participation"]:.1%}→{b["TA_SCL_upper_participation"]:.1%}|{a["TA_SCL_minus_ICL_lag_median_ms"]:.2f}→{b["TA_SCL_minus_ICL_lag_median_ms"]:.2f}|{a["TB_SCL_minus_ICL_lag_median_ms"]:.2f}→{b["TB_SCL_minus_ICL_lag_median_ms"]:.2f}|\n'
    note+='''
首组新噪声847401中，TA上部SCL参与39.6%→61.7%，杆间差−15.62→−12.48ms，顺序误差0.167→0.156，与开发方向一致。TA标签比例48.2%→64.7%，原／改后合格事件都为139。该比例更接近患者66.6%，但不能把单次新噪声的有利模式构成当作稳定机制结论。

TB的ICL11早于ICL9由10/72变为12/49，仍低于患者约69.4%；TB杆间差35.15→34.94ms，患者约1.19ms。训练分数3.200→1.952，不能据此接受TB传播恢复或使用本次新噪声重新提名。灰色开发轨迹与新轨迹保持各自身份，不计作三张网络。

该图对参与、成对顺序与时差分别评价，没有新的Overall恢复率。ALL、TA、TB各自观测及事件支持均保留在per_run_observations.csv；N批次的不分类曲线见其analysis/figures。患者仍是FIT参考，不是新的患者外推集合。其他基础网络和第二次新噪声的确认按原计划继续。
'''
    (OUT/'scientific_note.md').write_text(note)
    (D/'README.md').write_text('### position_effect_old_and_new_noise.png\n固定网络和同一位置对照，灰线为开发噪声、彩线为完整新噪声配对；六个观测分别保留量纲。每点代表一条运行，旧数据不计为本次新增仿真。\n**关注点**：TA招募与时差作用是否保留，TB回返和绝对时差是否仍有残差；不能把换噪声当换网络。\n')
    (OUT/'manifest.json').write_text(json.dumps(dict(created_unix=time.time(),sources=sources,complete_pairs=len(d)//2,topologies=[2511],noise_ids=sorted(map(int,d.noise.unique())),static_identity_same_by_position_across_noises=True,new_physical_runs_added_by_analysis=0,nomination_changes=0,producer=str(Path(__file__)),producer_sha256=sha(Path(__file__))),indent=2))
    print(pd.DataFrame(effects).to_string(index=False));print(OUT)

if __name__=='__main__':main()
