"""Separate participation, normalized rank and millisecond lag distributions.

Run-level empirical quantiles, not confidence intervals or an overall recovery
percentage. Patient FIT is the frozen development reference. All-detection and
original-primary views remain separate. No observer or training changes.
"""
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts import analyze_topic4_propagation_recovery_night as review
an=review.an;rt=review.rt


def profile(x):
    p=np.isfinite(x).mean(0) if len(x) else np.full(x.shape[1],np.nan)
    rank=an.ranks(x);lag=x-np.nanmin(x,axis=1)[:,None] if len(x) else x
    def q(table):
        out=np.full((3,table.shape[1]),np.nan)
        for j in range(table.shape[1]):
            v=table[:,j];v=v[np.isfinite(v)]
            if len(v):out[:,j]=np.quantile(v,[.05,.5,.95])
        return out
    return p,q(rank),q(lag)


def pair_matrix(x):
    c=x.shape[1];p=np.full((c,c),np.nan);n=np.zeros((c,c),int)
    for i in range(c):
      for j in range(c):
        if i==j:continue
        ok=np.isfinite(x[:,i])&np.isfinite(x[:,j]);d=x[ok,j]-x[ok,i];n[i,j]=len(d)
        if len(d):p[i,j]=np.mean((d>0)+.5*(d==0))
    return p,n


def contact_moments(x):
    rank=an.ranks(x);lag=x-np.nanmin(x,axis=1)[:,None] if len(x) else x;out=[]
    for j in range(x.shape[1]):
        a=rank[:,j];b=lag[:,j];a=a[np.isfinite(a)];b=b[np.isfinite(b)]
        out.append(dict(n_participating=len(a),rank_mean=float(a.mean()) if len(a) else None,rank_variance=float(a.var(ddof=1)) if len(a)>1 else None,
            lag_mean_ms=float(b.mean()) if len(b) else None,lag_variance_ms2=float(b.var(ddof=1)) if len(b)>1 else None))
    return out


def main(phase):
    old,plan,spec,cases=review.stage_cases(phase);out=review.night.OUT/('distribution_review_'+phase);F=out/'figures';F.mkdir(parents=True,exist_ok=True)
    parent=rt.read(an.run.PARENT);ev=rt.load_evaluator(parent);patient=np.asarray(ev.fit);pl=np.asarray(ev.fit_labels)
    names=list(rt.load_observation_contract(parent)['contact_names']);order=[names.index(x) for x in an.DISPLAY]
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':8,'pdf.fonttype':42});rows=[];pairs=[];manifest=[]
    modes=[('ALL',None),('TA',1),('TB',0)];refs={lab:patient if label is None else patient[pl==label] for lab,label in modes}
    reference_profiles={lab:profile(x[:,order]) for lab,x in refs.items()}
    reference_pairs={lab:pair_matrix(x[:,order]) for lab,x in refs.items()}
    patient_rows=[]
    for lab,x in refs.items():
        pp=reference_profiles[lab]
        for j,moment in enumerate(contact_moments(x[:,order])):
            patient_rows.append(dict(mode=lab,n=len(x),contact=an.DISPLAY[j],participation=float(pp[0][j]),rank_q05=float(pp[1][0,j]),rank_median=float(pp[1][1,j]),rank_q95=float(pp[1][2,j]),
                lag_q05_ms=float(pp[2][0,j]),lag_median_ms=float(pp[2][1,j]),lag_q95_ms=float(pp[2][2,j]),**moment))
    an.writecsv(out/'patient_contact_reference.csv',patient_rows)
    for c in cases:
      seeds=sorted({int(s) for cid,t,s in spec['units'] if cid==c['base_id'] and int(t)==c['topology']}) or old['seeds'];units={}
      for seed in seeds:
        path=an.run.result_path(c['output_stage'],c['base_id'],c['topology'],seed);u=an.load_unit(path,old['analysis']['burnin_ms'])
        if u is not None:units[seed]=u
      if not units:continue
      for layer in ['primary','all_detected']:
        fig,axes=plt.subplots(3,3,figsize=(15,9),layout='constrained');data={};nn={}
        for col,(lab,mode) in enumerate(modes):
          ref=refs[lab];profiles=reference_profiles[lab];xx=np.arange(15)
          for row,(values,title) in enumerate(zip(profiles,['触点参与概率','事件内归一化顺序','距最早参与质心 (ms)'])):
            ax=axes[row,col]
            if row==0:ax.plot(xx,values,'k-',lw=1.5,label='患者 FIT')
            else:
                ax.fill_between(xx,values[0],values[2],color='#bdbdbd',alpha=.45,label='患者 5–95%');ax.plot(xx,values[1],'k-',lw=1.5,label='患者中位数')
            ax.set(xticks=xx,xticklabels=an.DISPLAY,ylabel=title);ax.tick_params(axis='x',rotation=65,labelsize=7);ax.grid(alpha=.12)
            if row<2:ax.set_ylim(-.03,1.03)
          sizes=[]
          for si,seed in enumerate(seeds):
            if seed not in units:continue
            r,a,primary=units[seed];ids=primary if layer=='primary' else an.all_detected_ids(r,old['analysis']['burnin_ms'])
            ids=ids if mode is None else ids[a['event_mode'][ids]==mode];x=a['centroid_ms'][ids];data[(lab,seed)]=x;nn[(lab,seed)]=len(x)
            sizes.append(f'{seed}: {len(x)}');color=['#138b80','#9b59a3'][si];style=['-','--'][si];marker=['o','s'][si]
            pp=profile(x[:,order]);moments=contact_moments(x[:,order])
            for row,values in enumerate(pp):
                ax=axes[row,col]
                if row==0:ax.plot(xx,values,color=color,ls=style,marker=marker,ms=3,lw=1,label=f'噪声 {seed}')
                else:
                    ax.plot(xx,values[1],color=color,ls=style,marker=marker,ms=3,lw=1,label=f'噪声 {seed}')
                    ax.vlines(xx+(si-.5)*.15,values[0],values[2],color=color,alpha=.6,lw=.6)
            for j,name in enumerate(an.DISPLAY):
                rows.append(dict(candidate=c['id'],topology=c['topology'],seed=seed,layer=layer,mode=lab,n=len(x),contact=name,
                    participation=float(pp[0][j]),rank_q05=float(pp[1][0,j]),rank_median=float(pp[1][1,j]),rank_q95=float(pp[1][2,j]),
                    lag_q05_ms=float(pp[2][0,j]),lag_median_ms=float(pp[2][1,j]),lag_q95_ms=float(pp[2][2,j]),**moments[j]))
          axes[0,col].set_title(lab+f'｜患者 n={len(ref)}\n模型 '+', '.join(sizes),fontsize=10,color=an.MODE_COLOR[lab])
        axes[0,0].legend(fontsize=7);axes[1,0].legend(fontsize=7,loc='upper left')
        population='原primary，保留患者匹配的孤立窗口规则' if layer=='primary' else '全部检测：资格不同，仅开发诊断'
        fig.suptitle(review.display(c)+f'｜拓扑 {c["topology"]}\n{population}；后两行：点/曲线为中位数，竖线与灰带为事件 5–95%范围，不是置信区间',fontsize=12)
        stem=f'{c["id"]}_{layer}_contact_distributions'
        for ext in ['png','pdf']:fig.savefig(F/f'{stem}.{ext}',dpi=160,bbox_inches='tight')
        plt.close(fig)
        fig,axes=plt.subplots(3,3,figsize=(13,11),layout='constrained')
        for row,(lab,mode) in enumerate(modes):
          for col,seed in enumerate([None]+seeds):
            x=refs[lab] if seed is None else data.get((lab,seed),np.empty((0,15)))
            p,n=reference_pairs[lab] if seed is None else pair_matrix(x[:,order]);ax=axes[row,col];cmap=plt.get_cmap('coolwarm').copy();cmap.set_bad('#aaaaaa')
            im=ax.imshow(p,cmap=cmap,vmin=0,vmax=1,interpolation='nearest')
            yy,xx=np.where((n>0)&(n<5));ax.scatter(xx,yy,marker='x',c='black',s=5,lw=.4)
            ax.set(xticks=range(15),yticks=range(15),xticklabels=an.DISPLAY,yticklabels=an.DISPLAY,
                title=f'{lab} · '+('患者 FIT' if seed is None else f'噪声 {seed}')+f' · n={len(x)}');ax.tick_params(axis='x',rotation=90,labelsize=6);ax.tick_params(axis='y',labelsize=6)
            if seed is not None:
                for i in range(15):
                  for j in range(i+1,15):pairs.append(dict(candidate=c['id'],topology=c['topology'],seed=seed,layer=layer,mode=lab,row_contact=an.DISPLAY[i],column_contact=an.DISPLAY[j],model_joint_n=int(n[i,j]),model_row_precedes_column=float(p[i,j]),patient_joint_n=int(reference_pairs[lab][1][i,j]),patient_row_precedes_column=float(reference_pairs[lab][0][i,j])))
        fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.6,label='P(行触点早于列触点 | 两触点均参与)；同刻贡献 0.5')
        fig.suptitle(review.display(c)+'\n'+population+'；灰色为无共同事件/对角线；×为共同事件少于5，仅标支持量，不删点',fontsize=11)
        stem2=f'{c["id"]}_{layer}_pair_order'
        for ext in ['png','pdf']:fig.savefig(F/f'{stem2}.{ext}',dpi=160,bbox_inches='tight')
        plt.close(fig);manifest.append(dict(candidate=c['id'],layer=layer,profile=stem,pair_order=stem2,n={f'{k[0]}_{k[1]}':v for k,v in nn.items()}))
    an.writecsv(out/'contact_quantiles.csv',rows);an.writecsv(out/'pair_probabilities.csv',pairs)
    rt.write(out/'manifest.json',dict(status='COMPLETE_PENDING_SCIENTIFIC_REVIEW',phase=phase,records=manifest,producer=__file__,producer_sha256=rt.sha(__file__),patient_reference='frozen FIT development events',
        scope='conditional participation/rank/lag observations, not a variance-weighted overall recovery statistic; model centroid and patient HFO centroid are different signal constructions',
        moments='Means and sample variances (ddof1; n<2 not estimable) per participating contact saved separately. No across-feature weighting or overall variance percentage.'))
    texts=[]
    for f in sorted(F.glob('*.png')):
        body=('每列分别为不分标签、TA、TB，三行分别显示触点参与概率、事件内归一化rank及距该事件最早参与质心的毫秒时差。黑线/灰带为患者FIT中位数和5–95%事件分布，两种颜色/线型保留两条噪声；无事件保持缺失，散布不是置信区间。**关注点**：恢复了哪些触点的均值、散布和参与，不能将逐维相容替代联合分布。' if '_contact_distributions' in f.name else '三行分别为全部、TA、TB，每行比较患者与两条模型噪声；每格表示两触点共同参与时，行早于列的概率。灰格不估计，×仅标共同事件少于5；资格见文件名和标题。**关注点**：平均rank相似时是否仍有特定触点对顺序偏差；患者本身有方向变异，不能要求每一次都相同。')
        texts.append(f'### {f.name}\n\n{body}')
    (F/'README.md').write_text('\n\n'.join(texts)+'\n');print(json.dumps(dict(output=str(out),panels=len(manifest)),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',default='wave1');args=parser.parse_args();main(args.phase)
