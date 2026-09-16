#!/usr/bin/env python3
"""Per-contact distributions and pairwise covariance; no new ranking metric."""
from pathlib import Path
import csv,json,sys,pickle
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.topic4_contact_event_objective_v2 import patient_event,stack_events,worker_events
from src.topic4_interictal_repaired_evaluation import rank_features


def features(table):
    mask=table['participation'].astype(bool);cent=np.asarray(table['centroid']);rank=np.full(cent.shape,np.nan)
    for i,m in enumerate(mask):
        if m.sum()>1:rank[i,m]=(rankdata(cent[i,m],method='average')-1)/(m.sum()-1)
    return dict(participation=mask.astype(float),rank=rank,
        recruitment_ms=np.where(mask,table['recruitment'][:,:,0],np.nan),
        local_width_ms=np.where(mask,table['local_shape'][:,:,17]-table['local_shape'][:,:,1],np.nan))


def summarize(x,w):
    valid=np.isfinite(x);v=x[valid];ww=w[valid]
    if not len(v) or ww.sum()==0:return dict(n=0,mean=None,median=None,variance=None,q05=None,q95=None)
    ww=ww/ww.sum();mu=float(ww@v);ix=np.argsort(v);cdf=np.cumsum(ww[ix])
    # Weighted empirical inverse CDF; no interpolation through missing contacts.
    q=(np.quantile(v,[.05,.5,.95]).tolist() if np.allclose(ww,ww[0]) else [float(v[ix[min(np.searchsorted(cdf,p),len(v)-1)]]) for p in [.05,.5,.95]])
    return dict(n=len(v),mean=mu,median=q[1],variance=float(ww@((v-mu)**2)),q05=q[0],q95=q[2])


def covariance(x,w):
    c=x.shape[1];cov=np.full((c,c),np.nan);counts=np.zeros((c,c),int)
    for i in range(c):
        for j in range(c):
            good=np.isfinite(x[:,i])&np.isfinite(x[:,j]);counts[i,j]=good.sum()
            if counts[i,j]<2 or w[good].sum()==0:continue
            a=x[good,i];b=x[good,j];ww=w[good]/w[good].sum()
            cov[i,j]=ww@((a-ww@a)*(b-ww@b))
    return cov,counts


def run(folder,obj,old):
    folder=Path(folder);manifest=json.load(open(ROOT/'results/topic4_sef_hfo/contact_event_objective_revision_v2/manifest.json'))
    patient=stack_events([patient_event(e['arrays_path'],obj.names) for e in manifest['patient_training']])
    groups=[('patient_TRAIN','reference',patient,np.array([e['mode'] for e in manifest['patient_training']]),obj.weights)]
    evaluator=pickle.load(open(manifest['full_fit_source'],'rb'));fit=np.asarray(evaluator.fit,float)
    fit_labels=old.km.predict(rank_features(fit))
    if not np.array_equal(fit_labels,evaluator.fit_labels):raise RuntimeError('FIT mode identities disagree with frozen patient labels')
    fit_table=dict(participation=np.isfinite(fit).astype(float),centroid=fit,recruitment=np.full((*fit.shape,1),np.nan),local_shape=np.full((*fit.shape,19),np.nan))
    groups.append(('patient_FIT','reference',fit_table,fit_labels,np.ones(len(fit))/len(fit)))
    candidates=json.load(open(folder/'scores.json'))['candidates']
    for r in candidates:
        for uid,u in r['units'].items():
            tab,details,names,w=worker_events(u['worker_path'])
            if tab is None:continue
            labels=old.km.predict(rank_features(tab['centroid']))
            groups.append((r['candidate_id'],uid,tab,labels,np.ones(len(labels))/len(labels)))
    rows=[];arrays={};index=[];lookup={}
    for gi,(cid,uid,tab,labels,w) in enumerate(groups):
        for mode,name in [(None,'all'),(1,'TA'),(0,'TB')]:
            select=np.ones(len(labels),bool) if mode is None else labels==mode
            feat=features(tab)
            for key,val in feat.items():
                if cid=='patient_FIT' and key not in ['participation','rank']:continue
                x=val[select];ww=np.asarray(w)[select]
                for c,n in enumerate(obj.names):
                    record=dict(candidate_id=cid,unit=uid,mode=name,feature=key,contact=n,**summarize(x[:,c],ww))
                    rows.append(record);lookup[cid,uid,name,key,n]=record
                cv,n=covariance(x,ww);prefix=f'g{gi}_{name}_{key}'
                arrays[prefix+'_covariance']=cv;arrays[prefix+'_pair_counts']=n
                index.append(dict(prefix=prefix,candidate_id=cid,unit=uid,mode=name,feature=key))
    with open(folder/'per_contact_distributions.csv','w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    np.savez_compressed(folder/'per_contact_covariances.npz',contact_names=np.array(obj.names),**arrays)
    (folder/'per_contact_covariance_metadata.json').write_text(json.dumps(dict(index=index,units={'participation':'binary','rank':'normalized within-event participating rank, 0..1','recruitment_ms':'contact t50 minus within-event median contact t50','local_width_ms':'contact t90-t10'},
        statistics='Descriptive weighted empirical distribution; patient uses frozen natural-mode weights, model equal event weights. Uniform-weight quantiles use ordinary interpolated sample quantiles; otherwise weighted empirical inverse CDF.',
        missingness='Timing/rank/width are undefined at nonparticipating contacts; participation itself includes all events. Covariance uses joint observed pairs with at least 2 events and is not necessarily positive semidefinite; pair counts retained.',
        labels='Old fixed rank classifier organizes model TA/TB; labels do not establish physiological mode recovery.',training_objective_changed=False),indent=2)+'\n')
    order=sorted(obj.names,key=lambda n:(not n.startswith('SCL'),-int(''.join(filter(str.isdigit,n)))))
    keys=['participation','rank','recruitment_ms','local_width_ms'];titles=['Participation probability','Participating-contact rank','Recruitment time (ms)','Local width (ms)']
    for r in candidates:
        fig,axs=plt.subplots(3,4,figsize=(16,9),squeeze=False)
        for i,mode in enumerate(['all','TA','TB']):
            for j,key in enumerate(keys):
                ax=axs[i,j];source='patient_FIT' if key in ['participation','rank'] else 'patient_TRAIN';p=[lookup[source,'reference',mode,key,n] for n in order]
                vals=lambda a,k:np.array([np.nan if z[k] is None else z[k] for z in a])
                center='mean' if key=='participation' else 'median'
                if key!='participation':ax.fill_between(range(15),vals(p,'q05'),vals(p,'q95'),color='#aaaaaa',alpha=.25,label='Patient 5-95%')
                ax.plot(vals(p,center),color='black',label='Patient FIT' if source=='patient_FIT' else 'Patient waveform TRAIN',lw=1.8)
                for idx,(uid,u) in enumerate(r['units'].items()):
                    if (r['candidate_id'],uid,mode,key,order[0]) not in lookup:continue
                    v=[lookup[r['candidate_id'],uid,mode,key,n] for n in order]
                    color=['#b35e48','#4288a8'][idx%2]
                    if key!='participation':ax.fill_between(range(15),vals(v,'q05'),vals(v,'q95'),color=color,alpha=.12)
                    ax.plot(vals(v,center),'.-',label=uid,color=color,lw=.9,ms=3)
                ax.set(xticks=range(15),xticklabels=order,ylabel=mode,title=titles[j] if i==0 else '')
                ax.tick_params(axis='x',labelrotation=90,labelsize=6);ax.axvline(3.5,color='#5f9da6',lw=.6);ax.grid(alpha=.1)
                if key in ['participation','rank']:ax.set_ylim(-.05,1.05)
        axs[0,0].legend(fontsize=6);fig.suptitle(r['candidate_id']+' | conditional contact distributions',fontsize=11);fig.tight_layout(rect=(0,0,1,.96))
        for ext in ['png','pdf']:fig.savefig(folder/'figures'/f'{r["candidate_id"]}_per_contact.{ext}',dpi=145,bbox_inches='tight')
        plt.close(fig)
    with (folder/'figures/README.md').open('a') as f:f.write('\n### *_per_contact.png / .pdf\n\n逐接触点比较参与概率、参与者 rank、相对招募时间与局部宽度；分别显示全部事件、TA 和 TB，参与和 rank 的黑线来自完整 19,770 个 FIT 事件；招募与宽度黑线来自 46 个 waveform TRAIN 事件，彩线为各网络。时间量灰带为患者 5–95% 范围，彩色带为各模型网络 5–95% 范围，方差与均值等完整统计在 per_contact_distributions.csv 中保留。**关注点**：缺失不补零，不把某杆缺乏参与误判为传播时间相符；模型标签仅组织比较。跨接触点协方差与配对事件数另存 NPZ。\n')
    return len(rows)
