#!/usr/bin/env python3
"""Explain real E1146 abs/signed/mirror scores using all-event Timing+Space.

Diagnostic outputs only. No seizure boundaries, clustering or paper panels are
changed. Energy vectors are the independently raw-recomputed 15-contact values.
"""
from pathlib import Path
import json
import sys
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.topic5_template_axis_field import scorers_from_interictal_record, score_field, _smooth_from_weights
from scripts.analyze_e1146_preseizure_template_share import write_json

BASE=ROOT/'results/topic5_preseizure_template_share/epilepsiae_1146'
OUT=BASE/'template_label_audit'
FIELD=ROOT/'results/interictal_propagation_masked/template_gradient_fields_all_events_timing_plus_space/per_subject/epilepsiae_1146.json'
OLD=ROOT/'results/interictal_propagation_masked/template_gradient_fields/per_subject/epilepsiae_1146.json'
RED,BLUE='#B2182B','#2166AC'
EXAMPLES=(3,5,23,26)


def save(fig,name):
    fig.savefig(OUT/'figures'/f'{name}.png',dpi=200,bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/'figures'/f'{name}.pdf',bbox_inches='tight',facecolor='white')
    plt.close(fig)


def standard(x):
    x=np.asarray(x,float)
    return (x-x.mean())/x.std()


def run():
    (OUT/'figures').mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    record=json.loads(FIELD.read_text());old=json.loads(OLD.read_text())
    scorers=scorers_from_interictal_record(record)
    assert record['interictal_field']['contact_order']==old['interictal_field']['contact_order']
    discovery=record['template_discovery']
    idx=np.asarray(discovery['sampled_event_indices']);new_labels=np.asarray(discovery['event_labels'])
    event_index=np.load(BASE/'event_index.npz')
    assert np.array_equal(idx,event_index['source_event_index'])
    old_labels=event_index['template_label']
    confusion=np.zeros((2,2),int)
    np.add.at(confusion,(old_labels,new_labels),1)
    rows=[];events={};score_vectors=[];max_checkpoint_error=0.
    for p in sorted((BASE/'per_seizure').glob('*.json')):
        e=json.loads(p.read_text())
        if e['status']!='ok':continue
        assert e['field_sha256']['fig3c']==hashlib.sha256(OLD.read_bytes()).hexdigest()
        activation=np.asarray(e['activation'])
        assert len(activation)==15 and np.isfinite(activation).all()
        a,b=[score_field(scorers['shared_'+k],activation) for k in ('a','b')]
        ra,rb=a['signed_r'],b['signed_r']
        row=dict(sz=e['sz'],seizure_idx=e['seizure_idx'],r_a=ra,r_b=rb,
                 abs_a=abs(ra),abs_b=abs(rb),abs_label='TA' if abs(ra)>abs(rb) else 'TB',
                 signed_label='TA' if ra>rb else 'TB',identity_label='TA' if a['r_identity']>b['r_identity'] else 'TB',
                 abs_margin=abs(abs(ra)-abs(rb)),identity_a=a['r_identity'],identity_b=b['r_identity'],
                 mirror_a=a['r_mirror'],mirror_b=b['r_mirror'],choice_a=a['mirror_choice'],choice_b=b['mirror_choice'])
        rows.append(row);events[e['sz']]=dict(activation=activation,scores={'a':a,'b':b})
        cp=ROOT/f'results/topic5_ictal_recruitment/tspectral_field_concordance_all_events_timing_plus_space/per_subject/clinical_onset_shared_field/epilepsiae_1146/seizure_{e["seizure_idx"]:03d}.json'
        prior=json.loads(cp.read_text())['event']
        error=max(abs(ra-prior['shared_a_signed']),abs(rb-prior['shared_b_signed']))
        assert error<1e-10
        max_checkpoint_error=max(max_checkpoint_error,error)
        for k,result in [('a',a),('b',b)]:
            scorer=scorers['shared_'+k]
            key='weight_mirror' if result['mirror_choice']=='mirror' else 'weight_id'
            smoothed=_smooth_from_weights(activation,scorer[key]);tpl=scorer['template_field']
            assert abs(np.corrcoef(tpl,smoothed)[0,1]-result['signed_r'])<1e-12
            for name,t,v in zip(record['names'],tpl,smoothed):
                score_vectors.append(dict(sz=e['sz'],template=k,contact=name,template_value=t,
                                          smoothed_energy=v,orientation=result['mirror_choice']))
    df=pd.DataFrame(rows);df.to_csv(OUT/'seizure_scores.csv',index=False)
    pd.DataFrame(score_vectors).to_csv(OUT/'exact_scoring_vectors.csv',index=False)
    original=pd.read_csv(BASE/'seizure_labels.csv').query("status=='ok'")
    assert np.array_equal(df.sz,original.sz)
    # Correct the original strict-block proportions with the actual space-informed labels.
    counts=pd.read_csv(BASE/'interval_template_shares.csv'); corrected=[]
    times=event_index['event_abs_time'];eligible=event_index['strict_eligible']
    for _,r in counts.iterrows():
        mask=eligible&(times>=r.start_epoch)&(times<r.end_epoch)
        aa=int(np.sum(new_labels[mask]==0));bb=int(np.sum(new_labels[mask]==1))
        assert aa+bb==r.n_events
        q=r.to_dict();q.update(n_ta=aa,n_tb=bb,ta_share=aa/(aa+bb) if aa+bb else np.nan)
        for name in list(q):
            if name.startswith(('fig3c_','timing_space_')):del q[name]
        sr=df[df.sz.eq(r.sz)]
        if len(sr):
            s=sr.iloc[0]
            q['all_event_space_abs_label']=s.abs_label;q['all_event_space_signed_label']=s.signed_label
            q['matched_share']=(aa if s.abs_label=='TA' else bb)/(aa+bb) if aa+bb else np.nan
        corrected.append(q)
    pd.DataFrame(corrected).to_csv(OUT/'corrected_strict_interval_shares.csv',index=False)
    summary=dict(field=str(FIELD),field_sha256=hashlib.sha256(FIELD.read_bytes()).hexdigest(),
                 field_fingerprint=record['interictal_field']['fingerprint_sha256'],
                 event_counts_old=np.bincount(old_labels),event_counts_space=np.bincount(new_labels),
                 event_label_confusion_old_rows_space_columns=confusion,event_labels_changed=int(np.sum(old_labels!=new_labels)),
                 ranks_a_identical=np.array_equal(record['rank_a'],old['rank_a']),ranks_b_identical=np.array_equal(record['rank_b'],old['rank_b']),
                 n=25,abs_labels=df.abs_label.value_counts().to_dict(),signed_labels=df.signed_label.value_counts().to_dict(),
                 abs_signed_disagreements=int(np.sum(df.abs_label!=df.signed_label)),
                 seizure_abs_labels_changed_from_previous=int(np.sum(df.abs_label!='T'+original.fig3c_abs_label.to_numpy())),
                 both_abs_above_05=int(np.sum(np.minimum(df.abs_a,df.abs_b)>.5)),both_abs_above_07=int(np.sum(np.minimum(df.abs_a,df.abs_b)>.7)),
                 median_abs_a=df.abs_a.median(),median_abs_b=df.abs_b.median(),opposite_sign_count=int(np.sum(df.r_a*df.r_b<0)),
                 across_seizure_corr_r_a_r_b=np.corrcoef(df.r_a,df.r_b)[0,1],
                 template_field_corr=np.corrcoef(scorers['shared_a']['template_field'],scorers['shared_b']['template_field'])[0,1],
                 mirror_a=int(np.sum(df.choice_a=='mirror')),mirror_b=int(np.sum(df.choice_b=='mirror')),
                 differing_mirror_choices=int(np.sum(df.choice_a!=df.choice_b)),
                 signed_vs_identity_labels_changed=int(np.sum(df.signed_label!=df.identity_label)),
                 abs_margin_below_005=int(np.sum(df.abs_margin<.05)),checkpoint_score_max_error=max_checkpoint_error,
                 checkpoint_identity_note='Historical checkpoint file hash and fingerprint differ from current field; scores independently recomputed and match numerically; no claim of identical source artifacts.',
                 example_selection='SZ3 clear positive TA; SZ5 opposing high scores; SZ23 near absolute tie; SZ26 mirror-dependent signed label. Illustrative, not independent evidence.',
                 human_visual_acceptance='pending')
    write_json(OUT/'summary.json',summary)
    print(json.dumps(summary,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else bool(x),indent=2))

    # Joint plots expose information lost by taking absolute values.
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.4),layout='constrained')
    changed=df.abs_label!=df.signed_label
    for ax,absolute in zip(axes,[False,True]):
        xa=df.abs_a if absolute else df.r_a;yb=df.abs_b if absolute else df.r_b
        ax.scatter(xa,yb,s=52,c=np.where(changed,'#8c510a','#555555'),alpha=.85,zorder=3)
        low=0 if absolute else -1
        ax.plot([low,1],[low,1],color='#999999',ls='--',lw=1)
        if not absolute:
            ax.plot([-1,1],[1,-1],color='#bbbbbb',ls=':',lw=1)
            ax.axhline(0,color='#cccccc',lw=.7);ax.axvline(0,color='#cccccc',lw=.7)
        for sz in EXAMPLES:
            i=df.index[df.sz.eq(sz)][0]
            offsets=({3:(6,8),5:(10,-23),23:(-44,19),26:(8,8)} if absolute
                     else {3:(6,8),5:(-38,12),23:(8,8),26:(8,8)})
            ax.annotate(f'SZ{sz}',(xa[i],yb[i]),xytext=offsets[sz],textcoords='offset points',fontsize=11,
                        arrowprops={'arrowstyle':'-','color':'#555555','lw':.7})
        ax.set(xlim=(low-.035,1.035),ylim=(low-.035,1.035),
               xlabel='TA |r|' if absolute else 'TA signed r',ylabel='TB |r|' if absolute else 'TB signed r')
        ax.set_aspect('equal')
        ax.set_title('Absolute values discard polarity' if absolute else 'All 25 seizures: opposite signs',fontsize=15)
        if absolute:
            ax.text(.035,.98,'Above diagonal: |r| selects TB\nBelow diagonal: |r| selects TA',transform=ax.transAxes,va='top',fontsize=10)
        else:
            ax.text(.56,.94,'13/25 labels disagree\n(brown points)',transform=ax.transAxes,ha='left',va='top',fontsize=10)
    save(fig,'signed_and_absolute_scores')

    # Exact vectors used by the score; each dot is a contact evaluation site.
    fig,axes=plt.subplots(2,2,figsize=(11,9),layout='constrained')
    for i,sz in enumerate((5,23)):
        for j,k in enumerate(('a','b')):
            ax=axes[i,j];ev=events[sz];s=ev['scores'][k];sc=scorers['shared_'+k]
            w=sc['weight_mirror'] if s['mirror_choice']=='mirror' else sc['weight_id']
            x=standard(sc['template_field']);y=standard(_smooth_from_weights(ev['activation'],w))
            color=RED if k=='a' else BLUE
            ax.scatter(x,y,c=color,s=48,alpha=.85)
            slope=np.corrcoef(x,y)[0,1];xx=np.array([min(x)-.15,max(x)+.15]);ax.plot(xx,slope*xx,color=color,lw=1.4)
            ax.axhline(0,color='#bbbbbb',lw=.6);ax.axvline(0,color='#bbbbbb',lw=.6)
            ax.set_title(f'SZ{sz} versus T{k.upper()}    r = {slope:+.3f}\nEnergy orientation: {s["mirror_choice"]}',fontsize=14)
            ax.set_xlabel(f'Frozen T{k.upper()} earliness field (spatial z)')
            ax.set_ylabel('Smoothed ictal energy (spatial z)')
    save(fig,'real_examples_exact_correlations')

    # Actual unmirrored spatial maps, using the existing 6-mm display renderer.
    from scripts.plot_topic5_interictal_event_envelope_field import load_frozen,_event_field
    fz=load_frozen('epilepsiae_1146',frozen_root=FIELD.parent)
    fig,axes=plt.subplots(2,3,figsize=(14.5,9.3),layout='constrained')
    specs=[('TA',None),('TB',None),('energy',3),('energy',5),('energy',23),('energy',26)]
    common_support=np.minimum(fz['support_a'],fz['support_b'])
    for ax,(kind,sz) in zip(axes.flat,specs):
        if kind in ('TA','TB'):
            k=kind[-1].lower();value=fz['rank_'+k]/14;support=fz['support_'+k];cmap='viridis'
            title=f'{kind} timing template';norm=Normalize(0,1);clabel='Rank: 0 early → 1 late'
        else:
            raw=events[sz]['activation'];vmin,vmax=float(min(raw)),float(max(raw));value=(raw-vmin)/(vmax-vmin)
            support=common_support;cmap='Blues';norm=Normalize(vmin,vmax);clabel='Early energy (baseline robust z)'
            s=df[df.sz.eq(sz)].iloc[0]
            title=f'SZ{sz} energy: original orientation\n|r| → {s.abs_label}; signed r → {s.signed_label}'
        xg,yg,field,_,_=_event_field(fz,value,support)
        ax.imshow(field,origin='lower',extent=[xg.min(),xg.max(),yg.min(),yg.max()],cmap=cmap,vmin=0,vmax=1,aspect='equal')
        pts=fz['points_mm'];ax.scatter(pts[:,0],pts[:,1],c=value,cmap=cmap,vmin=0,vmax=1,s=32,edgecolors='#888888',linewidths=.5,zorder=3)
        for name in ('SCL9','ICL1','ICL11'):
            n=fz['names'].index(name);ax.annotate(name,pts[n],xytext=(4,3),textcoords='offset points',fontsize=9)
        ax.set(xlim=fz['display_xlim_mm'],ylim=fz['display_ylim_mm'],xlabel='Shared TA axis (mm)',ylabel='Y (mm)')
        ax.set_title(title,fontsize=12)
        cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=ax,fraction=.04,pad=.02,shrink=.8);cb.set_label(clabel,fontsize=10);cb.ax.tick_params(labelsize=9)
    save(fig,'templates_and_real_energy_maps')

    # The optional mirror can even change polarity; signed r alone is not a
    # fixed-geometry readout.
    fig,ax=plt.subplots(figsize=(8.5,4.6),layout='constrained')
    s=df[df.sz.eq(26)].iloc[0];xx=np.arange(2)
    ax.plot(xx,[s.identity_a,s.r_a],'-o',color=RED,lw=2,ms=7,label='TA')
    ax.plot(xx,[s.identity_b,s.r_b],'-o',color=BLUE,lw=2,ms=7,label='TB')
    for vals,col in [([s.identity_a,s.r_a],RED),([s.identity_b,s.r_b],BLUE)]:
        for x,y in zip(xx,vals):ax.annotate(f'{y:+.3f}',(x,y),xytext=(10,3),textcoords='offset points',color=col)
    ax.axhline(0,color='#999999',lw=.8);ax.set(xlim=(-.2,1.45),ylim=(-1,1),ylabel='Signed correlation',xticks=xx,
          xticklabels=['Original orientation\nSigned winner: TA','After per-template mirror selection\nSigned winner: TB'])
    ax.set_title('SZ26: the mirror choice changes the signed label',fontsize=14);ax.legend(frameon=False,loc='upper right')
    save(fig,'mirror_changes_signed_label')
    (OUT/'figures/README.md').write_text('''# E1146 模板标签诊断图（全部事件 Timing+Space）

### signed_and_absolute_scores.png / .pdf
25次发作在有符号坐标下全部落在异号象限，取绝对值后反向相关被折叠成高相似度。棕色点表示两种规则的标签不一致，编号示例仅用于解释。
**关注点**：高绝对值表示可正可反的空间结构关联，不能直接充当方向相同的类别证据。

### real_examples_exact_correlations.png / .pdf
第5和23次发作与TA/TB实际评分向量的散点，每点是一个触点评估位置。能量向量按原算法进行模板特异support平滑及镜像选择，横纵轴仅为解释相关而作空间标准化，斜率与实际r精确一致。
**关注点**：负斜率不是弱相似，而是强反向关系；两个模板的能量变换不一定相同。

### templates_and_real_energy_maps.png / .pdf
上行TA/TB为加入空间信息后全部事件聚类得到的时序模板；其余为第3、5、23、26次真实发作能量场。所有地图保留共同原始几何，使用既有6mm显示renderer，模板深色=早，能量深色=高；能量色条分别保留真实robust-z范围。
**关注点**：展示图不镜像，不从像素算r；共同support用于能量显示，正式评分仍用各模板support和评分kernel。不同发作色条范围不同，不能据深浅比较绝对幅度。

### mirror_changes_signed_label.png / .pdf
第26次发作在原方位与镜像选择后的TA/TB相关并列展示。A和B各自选择绝对相关更大的方位，因此保留signed r仍不等于固定几何的相似度。
**关注点**：原方位的signed winner是TA，镜像选择后变为TB；所有图为诊断候选，待用户目视检查。
''',encoding='utf-8')


if __name__=='__main__':run()
