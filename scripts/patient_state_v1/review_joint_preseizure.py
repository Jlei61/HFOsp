"""Separate the time-weighted state from event-weighted mode observations."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
from scipy.special import expit
from numpy.polynomial.hermite import hermgauss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.renewal import prepare
from scripts.patient_state_v1.joint_two_state import unpack
from scripts.patient_state_v1.preseizure import covered
from scripts.patient_state_v1.forward_preseizure_review import permutation
OUT=RUN/'joint_preseizure_v1_22'

def query(times,theta,coupled,d,f,inv):
    b,a,c,ts,ss,tr,sr=unpack(theta,coupled);offsets=np.array([v['offset'] for v in inv]);epoch=np.searchsorted(offsets,times,side='right');idx=np.searchsorted(d['hi'],times,side='right')-1;valid=idx>=0;safe=np.maximum(idx,0);valid&=d['epoch'][safe]==epoch;m=np.zeros((len(times),2));v=np.tile([ss*ss,sr*sr,0.],(len(times),1))
    ix=idx[valid];assert np.all(d['hi'][ix]<=times[valid]);dt=(times[valid]-(d['lo'][ix]+d['hi'][ix])/2)/3600;assert np.all(dt>=0);rs=np.exp(-dt/ts);rr=np.exp(-dt/tr);m[valid,0]=rs*f['mean'][ix,0];m[valid,1]=rr*f['mean'][ix,1];v[valid,0]=rs*rs*f['variance'][ix,0]+ss*ss*(-np.expm1(-2*dt/ts));v[valid,1]=rr*rr*f['variance'][ix,1]+sr*sr*(-np.expm1(-2*dt/tr));v[valid,2]=rs*rr*f['variance'][ix,2]
    nodes,w=hermgauss(40);w/=np.sqrt(np.pi);uniform=np.empty(len(times));total=np.empty(len(times));tbrate=np.empty(len(times))
    for i in range(len(times)):
        ps=max(v[i,0],1e-14);pr=max(v[i,1]-v[i,2]**2/ps,1e-14);s=m[i,0]+np.sqrt(2*ps)*nodes[:,None];r=m[i,1]+np.sqrt(2)*v[i,2]/np.sqrt(ps)*nodes[:,None]+np.sqrt(2*pr)*nodes[None,:];p=expit(b+s);hazard=np.exp(np.clip(a+c*s+r,-700,60));rate=hazard/(1+hazard*.25/3600);ww=w[:,None]*w[None,:];uniform[i]=np.sum(w*p[:,0]);total[i]=np.sum(ww*rate);tbrate[i]=np.sum(ww*rate*p)
    return dict(mode_state=b+m[:,0],activity_state=m[:,1],mode_state_sd=np.sqrt(v[:,0]),uniform_tb_probability=uniform,expected_total_rate=total,expected_tb_rate=tbrate,event_weighted_tb_probability=tbrate/total,expected_mode_drift=-m[:,0]/ts,last_completed_bin=idx)

def window(lo,hi,theta,coupled,d,f,inv,ex):
    if hi<=lo:return dict(observed_seconds=0,mode_state=np.nan,uniform_tb_probability=np.nan,event_weighted_tb_share=np.nan,expected_total_rate=np.nan,expected_tb_rate=np.nan,expected_mode_drift=np.nan)
    edges=np.arange(lo,hi,5.);duration=np.minimum(edges+5,hi)-edges;times=edges+duration/2;ok=covered(times,ex)
    if not ok.any():return window(hi,hi,theta,coupled,d,f,inv,ex)
    r=query(times[ok],theta,coupled,d,f,inv);w=duration[ok];out={key:float(np.average(r[key],weights=w)) for key in ['mode_state','uniform_tb_probability','expected_total_rate','expected_tb_rate','expected_mode_drift']};out.update(observed_seconds=float(w.sum()),event_weighted_tb_share=float(np.sum(w*r['expected_tb_rate'])/np.sum(w*r['expected_total_rate'])));return out

def main():
    status=json.loads((OUT/'status.json').read_text());assert status['status']=='COMPLETE' and status['n_jobs']==48;assert len(list((OUT/'fits').glob('*.json')))==48
    d=prepare(5,.25);ex=pd.read_csv(RUN/'exposure.csv');inv=json.loads((RUN/'seizures.json').read_text());wins=pd.read_csv(RUN/'frozen_seizure_windows.csv');wins=wins[wins.window=='pre15'].sort_values('sz');best={}
    for p in (OUT/'fits').glob('*.json'):
        r=json.loads(p.read_text());assert r['status']=='COMPLETE';j=r['job'];key=(j['sz'],j['coupled'])
        if key not in best or r['loglik']>best[key][1]['loglik']:best[key]=(p,r)
    assert len(best)==24;rows=[];traces={};numerical=[]
    for r in wins.itertuples():
        onset=inv[r.sz-1]['onset'];previous=inv[r.sz-2]['offset'];latest=max(previous,onset-900);earlier=max(previous,onset-1800)
        for coupled in [False,True]:
            path,fit=best[(r.sz,coupled)];theta=fit['theta'];f=dict(np.load(path.with_suffix('.npz')));assert fit['job']['cutoff_epoch']<=earlier;a=window(latest,onset,theta,coupled,d,f,inv,ex);old=window(earlier,latest,theta,coupled,d,f,inv,ex);row=dict(sz=r.sz,label=r.label,coupled=coupled,fit_source=str(path),optimizer_success=fit['success'],train_cutoff=fit['job']['cutoff_epoch'],pre15_observed_tb_rate=r.n_tb/r.observed_hours,pre15_observed_tb_share=r.n_tb/r.n_events if r.n_events else np.nan)
            for key in a:row['pre15_'+key]=a[key];row['prior15_'+key]=old[key];row['delta_'+key]=a[key]-old[key]
            rows.append(row);edges=np.arange(max(previous,onset-3600),onset,5.);duration=np.minimum(edges+5,onset)-edges;times=edges+duration/2;z=query(times,theta,coupled,d,f,inv);mask=covered(times,ex);prefix=f'sz{r.sz}_c{int(coupled)}';traces[prefix+'_minutes']=(times-onset)/60
            for key,value in z.items():
                if key=='last_completed_bin':continue
                value=value.copy();value[~mask]=np.nan;traces[prefix+'_'+key]=value
            # Alter unavailable future posterior arrays; query must not change.
            test_t=np.array([times[len(times)//2]]);ref=query(test_t,theta,coupled,d,f,inv);future=d['hi']>test_t[0];changed={k:v.copy() for k,v in f.items()};changed['mean'][future]=100.;changed['variance'][future]=np.array([100.,100.,0.]);check=query(test_t,theta,coupled,d,changed,inv);err=max(float(np.max(abs(ref[k]-check[k]))) for k in ref);assert err==0.;numerical.append(dict(sz=r.sz,coupled=coupled,future_posterior_perturbation_error=err))
    df=pd.DataFrame(rows);df.to_csv(OUT/'state_readouts.csv',index=False);np.savez_compressed(OUT/'state_traces.npz',**traces);write_json(OUT/'causal_query_audit.json',dict(status='PASS',checks=numerical,rule='Only completed bins in current interval may affect query; future posterior entries have exactly zero effect'))
    tests=[]
    for coupled in [False,True]:
        part=df[df.coupled==coupled]
        for measure in ['pre15_mode_state','delta_mode_state','pre15_uniform_tb_probability','delta_uniform_tb_probability','pre15_event_weighted_tb_share','delta_event_weighted_tb_share']:
            tests.append(dict(coupled=coupled,measure=measure,**permutation(part[measure].to_numpy(),part.label.to_numpy()=='TB')))
    write_json(OUT/'readout_audit.json',dict(status='COMPLETE',tests=tests,limits='Unadjusted exploratory seizure-level permutations; only two TB-source cases and record-time confounding remain',probabilities='Uniform-time probability integrates sigmoid(b+s). Event-weighted share integrates effective retained rate times mode probability; uses a slow-state deadtime approximation, not exact packing inversion.',causality='Parameter fit excludes current interval. Current-interval observations update state; no current seizure type is used.',not_a_forecast='These are event-driven filtered reconstructions, not a prediction of the seizure or of the whole preictal interval from its start'))
    fig,axs=plt.subplots(3,4,figsize=(15,9),sharex=True,sharey=True)
    for ax,r in zip(axs.flat,wins.itertuples()):
        for coupled,color in [(False,'#777777'),(True,'#2166ac')]:
            key=f'sz{r.sz}_c{int(coupled)}';ax.plot(traces[key+'_minutes'],traces[key+'_mode_state'],c=color,lw=1,label='Joint observation' if coupled else 'Independent states')
            if coupled:ax.fill_between(traces[key+'_minutes'],traces[key+'_mode_state']-1.96*traces[key+'_mode_state_sd'],traces[key+'_mode_state']+1.96*traces[key+'_mode_state_sd'],color=color,alpha=.1)
        ax.axvline(-15,c='gray',ls=':',lw=.7);ax.set_title(f'SZ{r.sz} | {r.label}-source',color='#2166ac' if r.label=='TB' else '#b2182b');ax.set_xlim(-60,0)
    axs[0,0].legend(fontsize=7)
    for ax in axs[-1]:ax.set_xlabel('Minutes before seizure')
    for ax in axs[:,0]:ax.set_ylabel('Filtered mode state (TB log-odds)')
    fig.suptitle('Does the mode-state trend survive a joint event-rate observation model?\nAll 12 seizures; preinterval-only parameters; shading: fixed-parameter Gaussian state intervals',fontsize=12);fig.tight_layout(rect=(0,0,1,.94))
    for ext in ['png','pdf']:fig.savefig(RUN/'figures'/f'joint_preseizure_state_all12.{ext}',dpi=180)
    plt.close(fig)
    fig,axs=plt.subplots(3,2,figsize=(11,9),sharex=True)
    for col,sz in enumerate([19,22]):
        for coupled,color in [(False,'#777777'),(True,'#2166ac')]:
            key=f'sz{sz}_c{int(coupled)}';x=traces[key+'_minutes'];label='Joint observation' if coupled else 'Independent states';axs[0,col].plot(x,traces[key+'_mode_state'],c=color,label=label);axs[1,col].plot(x,traces[key+'_uniform_tb_probability'],c=color,label=label);axs[2,col].plot(x,traces[key+'_expected_total_rate'],c=color,label=label)
            if coupled:axs[0,col].fill_between(x,traces[key+'_mode_state']-1.96*traces[key+'_mode_state_sd'],traces[key+'_mode_state']+1.96*traces[key+'_mode_state_sd'],color=color,alpha=.1)
        axs[0,col].set_title(f'SZ{sz} | TB-source');axs[2,col].set_yscale('log');axs[2,col].set_xlabel('Minutes before seizure')
        for ax in axs[:,col]:ax.axvline(-15,c='gray',ls=':',lw=.7);ax.spines[['top','right']].set_visible(False);ax.set_xlim(-60,0)
    axs[0,0].set_ylabel('Mode state (TB log-odds)');axs[1,0].set_ylabel('Uniform-time TB probability');axs[2,0].set_ylabel('Approx. retained events / hour');axs[0,0].legend(fontsize=8);fig.suptitle('The two TB-source seizures: state level, state change and total activity are separate\nForward-parameter filtering; expected retained rate uses a slow-state 250-ms support approximation',fontsize=11);fig.tight_layout(rect=(0,0,1,.94))
    for ext in ['png','pdf']:fig.savefig(RUN/'figures'/f'joint_preseizure_tb_cases.{ext}',dpi=180)
    plt.close(fig)
    readme=RUN/'figures/README.md'
    if '### joint_preseizure_state_all12.png' not in readme.read_text():
        with readme.open('a') as f:f.write('\n### joint_preseizure_state_all12.png\n对全部12个冻结合格发作，比较独立模式/活动状态与允许模式状态影响事件率时的事件前状态轨迹。参数仅由上一发作结束前的数据拟合，轨迹只消费查询时已经完成的观察分箱。\n**关注点**：模式状态的水平和变化是否依赖观察模型；这是过滤重建，不是发作预测。\n\n### joint_preseizure_tb_cases.png\n分别放大仅有的两次TB型发作，分开显示状态log-odds、按实际时间平均的模式概率以及预期保留事件率。事件率曲线采用慢状态下250ms有效支持近似，缺失数据不补零。\n**关注点**：TB事件率较高是否由相同的状态上升过程解释；两例的共同趋势不能预设。\n')
    print(df[df.label=='TB'][['sz','coupled','pre15_mode_state','prior15_mode_state','delta_mode_state','pre15_uniform_tb_probability','delta_uniform_tb_probability','pre15_event_weighted_tb_share']].to_string(index=False))
if __name__=='__main__':main()
