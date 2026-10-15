#!/usr/bin/env python3
"""Exact Fig2C exemplars and existing model examples as contact-time maps.

Diagnostic sidecar only. No canonical figure, training input, or simulation
is changed. The full-contact view retains nonparticipant signal explicitly.
"""
from pathlib import Path
import csv, json, hashlib, sys, subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.topic4_envelope_joint_pilot import envelope_descriptor
MAIN=Path('/home/honglab/leijiaxin/HFOsp')
R=ROOT/'results/topic4_sef_hfo/multievent_distribution_search_v2_1'
OUT=R/'fig2c_contact_envelope_comparison';F=OUT/'figures'
META=MAIN/'results/paper-ready-figure/fig2/fig2_panelc_metadata.json'
CACHE=MAIN/'results/interictal_propagation_masked/event_envelope_fields/epilepsiae_1146_event_envelope_field_cache.npz'
NAMES=[f'SCL{i}' for i in range(9,5,-1)]+[f'ICL{i}' for i in range(11,0,-1)]
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})

def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def reorder(names):
    assert len(names)==len(set(names)) and set(names)==set(NAMES)
    return [names.index(n) for n in NAMES]

def load_data():
    meta=json.loads(META.read_text());data={};sources=[META,CACHE]
    with np.load(CACHE,allow_pickle=False) as z:
        order=reorder(z['contact_order'].astype(str).tolist())
        for lab in ['TA','TB']:
            ex=meta['exemplar'][lab];pos=int(z[lab+'_event_pos']);assert pos==ex['event_pos']=={'TA':6344,'TB':937}[lab]
            assert str(z[lab+'_block'])==ex['block']
            part=z[lab+'_participant'].astype(bool)
            cs=ex['fig1a_centroid_alignment']['centroids']
            # Exact canonical t0 is the earliest usable participating Fig1a
            # centroid. Undo it to recover the original 250ms packed window.
            offset=min(c['time_within_event_sec'] for c in cs if part[c['channel_index']])*1000
            time=z[lab+'_envelope_time_from_first_centroid_ms']+offset
            keep=(time>=-1e-6)&(time<ex['packed_window_ms']-1e-6)
            assert keep.sum()==256 and ex['packed_window_ms']==250.
            mass=np.maximum(z[lab+'_envelope_robust_z'][order][:,keep],0)
            data[(lab,0)]=dict(source='Fig. 2C patient',event_id=pos,block=ex['block'],mode=lab,
                mass=mass,time=time[keep],mask=part[order],canonical_fig2c_zero_ms=offset,
                quantity='positive baseline-robust-z 80-250Hz Hilbert amplitude envelope',
                sample_dt_ms=1000/ex['fs_hz'],selection='locked direction-qualified illustrative exemplar')
    examples=json.loads((R/'native_activity_shortcut_audit/example_selection.json').read_text())
    gs={c['candidate_id']:c for c in json.loads((R/'g3_scores.json').read_text())['candidates']}
    ids=['v2_1_pop1_de_b_002','v2_anchor_old_joint__baseline']
    for ci,cid in enumerate(ids,1):
        for lab,mode in [('TA',1),('TB',0)]:
            e=next(e for e in examples if e['candidate_id']==cid and e['mode']==mode)
            wp=Path(gs[cid]['units'][e['unit']]['worker_path']);op=wp.parent.parent/'repaired_observation'/wp.name
            obs=json.loads(op.read_text());evt=obs['events'][e['event_id']]
            assert not obs['worker_lineage_onsets_used_for_training']
            sources.extend([wp.with_suffix('.npz'),op,op.with_suffix('.npz')])
            with np.load(op.with_suffix('.npz')) as z:mask=np.isfinite(z['centroid_ms'][e['event_id']])
            with np.load(wp.with_suffix('.npz')) as z:
                names=z['contact_names'].astype(str).tolist();order=reorder(names);dt=float(z['contact_envelope_dt_ms'])
                lo,hi=np.rint(np.array(evt['window_ms'])/dt).astype(int)
                mass=np.maximum(z['contact_envelope'][:,lo:hi].astype(float)-np.array(evt['local_baseline'])[:,None],0)[order]
            assert hi-lo==125 and dt==2.
            data[(lab,ci)]=dict(source=['','Best feature-fit model','Reference placement A'][ci],candidate_id=cid,
                unit=e['unit'],event_id=e['event_id'],mode=lab,mass=mass,time=(np.arange(hi-lo)+.5)*dt,
                mask=mask[order],window_ms=evt['window_ms'],sample_dt_ms=dt,
                quantity='full E firing-density contact envelope after frozen local baseline subtraction',
                selection='same first chronological label example as native activity audit; no new visual selection')
    return data,sources

def main():
    F.mkdir(parents=True,exist_ok=True);data,sources=load_data();rows=[];meta=[];arrays={}
    for (lab,ci),d in data.items():
        norm=d['mass'].max(1);assert np.all(norm>0)
        d['normalized']=d['mass']/norm[:,None]
        desc=envelope_descriptor(d['mass'],d['time'],d['mask']);d['stats']=desc['statistics']
        d['mu']=desc['centroid'];d['q']=desc['quantiles']
        for i,n in enumerate(NAMES):
            q=d['q'][i]
            rows.append(dict(source=d['source'],mode=lab,event_id=d['event_id'],contact=n,
                participates=bool(d['mask'][i]),normalizing_peak=float(norm[i]),
                centroid_ms=float(d['mu'][i]) if d['mask'][i] else None,
                t10_ms=float(q[0]) if d['mask'][i] else None,t50_ms=float(q[1]) if d['mask'][i] else None,
                t90_ms=float(q[2]) if d['mask'][i] else None))
        info={k:v for k,v in d.items() if k not in ['mass','time','mask','normalized','mu','q']}
        info.update(participating_contacts=[n for n,b in zip(NAMES,d['mask']) if b],
                    nonparticipating_contacts=[n for n,b in zip(NAMES,d['mask']) if not b],
                    normalizing_peak=norm.tolist())
        meta.append(info)
        for k in ['mass','time','mask','normalized','mu','q']:arrays[f'{lab}_{ci}_{k}']=d[k]
    arrays['contact_order']=np.array(NAMES);np.savez_compressed(OUT/'displayed_arrays.npz',**arrays)
    with (OUT/'contact_times.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig_metadata=[]
    for mask_view in [False,True]:
        fig,axs=plt.subplots(2,3,figsize=(13,7.7),sharex=True)
        fig.subplots_adjust(left=.095,right=.90,top=.87,bottom=.235,hspace=.75,wspace=.36)
        for ri,lab in enumerate(['TA','TB']):
            for ci in range(3):
                d=data[(lab,ci)];ax=axs[ri,ci];a=d['normalized'].copy()
                if mask_view:a[~d['mask']]=np.nan
                cmap=plt.get_cmap('magma').copy();cmap.set_bad('#dedede')
                im=ax.imshow(a,origin='upper',aspect='auto',extent=[0,250,14.5,-.5],
                    vmin=0,vmax=1,cmap=cmap,interpolation='none')
                labels=[n+(' *' if not b else '') for n,b in zip(NAMES,d['mask'])]
                ax.set(yticks=range(15),yticklabels=labels,xlim=(0,250),xticks=range(0,251,50),xlabel='Time in event window (ms)')
                ax.tick_params(axis='y',labelsize=8);ax.tick_params(axis='x',labelbottom=True);ax.axhline(3.5,c='#6faeb8',lw=.8)
                if ri==0:ax.set_title(d['source'],fontweight='bold',pad=13)
                ax.text(0,1.02,f'Event {d["event_id"]} | {int(d["mask"].sum())}/15 participating',transform=ax.transAxes,fontsize=8)
                if ci==0:
                    # Canonical Fig2C frame times marked on patient only, with
                    # the coordinate conversion explicit rather than warping.
                    for ti in [0,16,32,48]:ax.plot(d['canonical_fig2c_zero_ms']+ti,-.5,'v',c='#5bc7c7',ms=4,clip_on=False)
                ax.text(.5,-.32,f'Median local width: {d["stats"]["contact_width_ms"]:.1f} ms',ha='center',transform=ax.transAxes,fontsize=9)
            axs[ri,0].set_ylabel(lab,color=['#b2182b','#2166ac'][ri],fontweight='bold',fontsize=13)
        cax=fig.add_axes([.925,.27,.012,.55]);cb=fig.colorbar(im,cax=cax,ticks=[0,.5,1]);cb.set_label('Envelope / own contact peak')
        fig.suptitle('Fig. 2C examples and model contact envelopes'+(' — event participation mask applied' if mask_view else ' — all recorded contacts'),fontsize=14)
        fig.text(.5,.045,'Same contact order and 250 ms windows; no time stretching. Each row has its own fixed peak scale.\n'
            '* = not participating in this event; '+('gray = excluded by event mask.' if mask_view else 'signal is still displayed; normalization can amplify weak activity.')+'\n'
            'Cyan triangles: original Fig. 2C frames (0, 16, 32, 48 ms). Patient HFO and model firing density have different amplitude units.',ha='center',fontsize=9)
        name='contact_envelopes_participation_mask' if mask_view else 'contact_envelopes_all_contacts'
        for ext in ['png','pdf']:fig.savefig(F/f'{name}.{ext}',dpi=170)
        plt.close(fig);fig_metadata.append(name)
    # A common event peak preserves relative amplitudes across contacts; it
    # checks whether own-contact normalization hides weak background.
    fig,axs=plt.subplots(2,3,figsize=(13,6.7));fig.subplots_adjust(left=.095,right=.90,top=.88,bottom=.14,hspace=.40,wspace=.36)
    for ri,lab in enumerate(['TA','TB']):
        for ci in range(3):
            d=data[(lab,ci)];ax=axs[ri,ci];norm=d['mass'].max()
            im=ax.imshow(d['mass']/norm,origin='upper',aspect='auto',extent=[0,250,14.5,-.5],vmin=0,vmax=1,cmap='magma',interpolation='none')
            ax.set(yticks=range(15),yticklabels=[n+(' *' if not b else '') for n,b in zip(NAMES,d['mask'])],xlim=(0,250),xticks=range(0,251,50),xlabel='Time in event window (ms)');ax.tick_params(axis='y',labelsize=8);ax.axhline(3.5,c='#6faeb8',lw=.8)
            if ri==0:ax.set_title(d['source'],fontweight='bold')
            if ci==0:ax.set_ylabel(lab,color=['#b2182b','#2166ac'][ri],fontweight='bold',fontsize=13)
    cax=fig.add_axes([.925,.20,.012,.62]);cb=fig.colorbar(im,cax=cax,ticks=[0,.5,1]);cb.set_label('Envelope / whole-event peak')
    fig.suptitle('All contacts — one fixed amplitude scale per event',fontsize=14)
    fig.text(.5,.025,'Within each event, every contact uses the same denominator. * = not participating; signal is retained.\nRelative amplitudes across contacts are visible; absolute patient/model amplitudes remain incomparable.',ha='center',fontsize=9)
    for ext in ['png','pdf']:fig.savefig(F/f'contact_envelopes_event_scale.{ext}',dpi=170)
    plt.close(fig);fig_metadata.append('contact_envelopes_event_scale')
    write(OUT/'metadata.json',dict(status='DIAGNOSTIC_FIGURES_COMPLETE',canonical_fig2c_unchanged=True,
        patient_ids_exactly_match_fig2c=True,patient_source='canonical Fig2C cache rather than generic wavepacket exemplar selection',model_examples_same_as_previous_audit=True,
        contact_order=NAMES,contact_order_rule='SCL9 to SCL6, then ICL11 to ICL1; fixed in every panel, no event-based sorting',
        time_rule='original 250ms event-window start = 0; no centroid alignment or time warping',
        model_source='FULL contact_envelope, never lineage-restricted onsets',examples=meta,
        source_sha256={str(p):sha(p) for p in sources+[Path(__file__)]}))
    verification=[]
    for name in fig_metadata:
        with Image.open(F/f'{name}.png') as im:im.load();verification.append(dict(figure=name,size=list(im.size)))
        subprocess.run(['pdfinfo',str(F/f'{name}.pdf')],check=True,capture_output=True)
    write(OUT/'verification.json',dict(status='PASS',contact_identity_exact=True,patient_exemplar_identity_exact=True,
        common_250_ms_duration=True,patient_samples_per_event=256,model_bins_per_event=125,figures=verification,human_visual_review='PENDING'))
    (F/'README.md').write_text('''### contact_envelopes_all_contacts.png / .pdf
左列为 Fig. 2C 原始 TA 6344、TB 937 的正部 robust-z HFO 包络，中列和右列为上一次原生场审计中的模型相同事件。所有面板固定 SCL9→6、ICL11→1 顺序及 250 ms 真实窗口，每通道以自己的完整窗口峰值归一化；星号表示未参与该事件但完整保留其信号，青色小三角对应原 Fig. 2C 四帧时刻。
**关注点**：颜色比较每通道时间形状，不能比较通道间或患者/模型绝对幅度。局部宽度为参与通道 t90−t10 的中位数，只描述这一次事件；星号行可能因独立归一化放大弱活动。

### contact_envelopes_participation_mask.png / .pdf
与完整图使用相同数值、顺序、色尺和窗口，仅将未参与通道显示为灰色，保持原接触点位置。患者掩膜来自 Fig. 2C 缓存，模型掩膜来自 frozen repaired observer，不将两套参与规则说成相同算法。
**关注点**：比较参与筛选如何改变视觉印象；灰色表示未纳入此事件，不代表真实活动为零。

### contact_envelopes_event_scale.png / .pdf
每个事件的所有接触点改用同一个完整窗口峰值作分母，仍显示全部信号和参与标记。它补充检查逐通道归一化是否放大弱背景，并显示该事件内通道相对幅度。
**关注点**：TB 的未参与通道也有包络，不能从被掩膜的 Fig. 2C 场图推断其他通道无活动；患者 HFO 与模型发放密度有不同物理含义。
''')
    print(json.dumps(dict(status='COMPLETE',output=str(OUT),examples=[dict(source=d['source'],mode=lab,**d['stats']) for (lab,ci),d in sorted(data.items())])))

if __name__=='__main__':main()
