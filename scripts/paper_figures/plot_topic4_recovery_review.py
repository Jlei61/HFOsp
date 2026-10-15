"""Main-layout review from frozen trajectories, with the actual Fig2C spectra.

No simulation, new event selection, spatial interpolation, or HFO model conversion.
Model exemplars minimize distance to their own run-mode feature mean. All-contact
and native activity remain visible independently of participant-conditioned views.
"""
from pathlib import Path
import argparse,importlib.util,json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Ellipse
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from PIL import Image
from scripts import analyze_topic4_propagation_recovery_night as review
from scripts.analyze_topic4_geometry_threshold_refinement import CACHE,META,MAIN
an=review.an;rt=review.rt
_display_spec=importlib.util.spec_from_file_location('fixed_snn_contact_display',MAIN/'src/snn_contact_display.py')
display=importlib.util.module_from_spec(_display_spec);_display_spec.loader.exec_module(display)


def patient_payloads():
    # This old worktree deliberately reads the current, read-only patient renderer.
    # Extend already-imported package paths for its current figure-only helpers.
    import src,scripts,scripts.paper_figures
    src.__path__=[str(MAIN/'src')]+list(src.__path__)
    scripts.__path__=[str(MAIN/'scripts')]+list(scripts.__path__)
    scripts.paper_figures.__path__=[str(MAIN/'scripts/paper_figures')]+list(scripts.paper_figures.__path__)
    spec=importlib.util.spec_from_file_location('canonical_patient_envelope',MAIN/'scripts/plot_topic5_interictal_event_envelope_field.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    meta=rt.read(META);payload={}
    with np.load(CACHE) as z:
      names=z['contact_order'].astype(str);order=display.contact_indices(names)
      for lab in ['TA','TB']:
        part=z[lab+'_participant'].astype(bool);ex=meta['exemplar'][lab]
        offset=min(c['time_within_event_sec'] for c in ex['fig1a_centroid_alignment']['centroids'] if part[c['channel_index']])*1000
        assert int(z[lab+'_event_pos'])==ex['event_pos']
        centroid=z[lab+'_fig1a_centroid_ms']
        payload[lab]=dict(usable=part&np.isfinite(centroid),spec_freq_hz=z[lab+'_fig1a_spec_freq_hz'],
            spec=z[lab+'_fig1a_spec_norm'],spec_t_ms=z[lab+'_fig1a_spec_time_from_first_centroid_ms'],
            tile_lo_ms=-offset,tile_hi_ms=ex['packed_window_ms']-offset,centroid_ms=centroid,
            centroid_freq_index=z[lab+'_fig1a_centroid_freq_index'],event=ex['event_pos'])
    return module,meta,payload,names,order


def representatives(a,ids):
    result={}
    for mode,lab in [(1,'TA'),(0,'TB')]:
        ix=ids[a['event_mode'][ids]==mode]
        if len(ix):
            x=a['event_phi'][ix]
            result[lab]=dict(event=int(ix[np.argmin(((x-x.mean(0))**2).sum(1))]),n=int(len(ix)))
    return result


def geometry(ax,c,a,contact_labels=False):
    xy=a['contact_xy_mm'];names=a['contact_names']
    for shaft,col in [('ICL','#dc8722'),('SCL','#42a9b5')]:
        ix=[i for i,n in enumerate(names) if str(n).startswith(shaft)]
        ax.plot(xy[ix,0],xy[ix,1],color=col,lw=1.1,zorder=5)
        ax.scatter(*xy[ix].T,s=13,facecolors='white',edgecolors=col,lw=.8,zorder=6)
        if contact_labels:
            for i in [ix[0],ix[-1]]:ax.annotate(names[i],xy[i],xytext=(3,4),textcoords='offset points',fontsize=6,color=col,zorder=8)
    for k,(xy0,rad) in enumerate(zip(c['centers_mm'],c['radii_mm'])):
        ax.add_patch(Circle(xy0,rad,fill=False,ec='#c94a44',lw=.95,zorder=7))
        if contact_labels:ax.text(xy0[0],xy0[1],str(k+1),color='#b22d27',ha='center',va='center',fontsize=8,zorder=8)
    ax.set(xlim=(0,20),ylim=(0,20),aspect='equal',xlabel='x (mm)',ylabel='y (mm)',xticks=[0,10,20],yticks=[0,10,20])


def native_timing(r,a,i):
    lo,hi=r['events'][i]['window_ms'];dt=float(a['sheet_activity_frame_ms'])
    movie=a['sheet_activity_counts'][round(lo/dt):round(hi/dt)].astype(float)
    mass=movie.sum(0);q=np.argmax(np.cumsum(movie,axis=0)>=.1*mass,axis=0)*dt
    q=q.astype(float);q[mass==0]=np.nan
    return q,mass


def four_panel(c,seed,r,a,ids,out,physics,population='all'):
    rep=representatives(a,ids);fig=plt.figure(figsize=(17.5,4.1),layout='constrained')
    grid=fig.add_gridspec(1,4,width_ratios=[1,1,1,2.25]);axes=[fig.add_subplot(grid[0,i]) for i in range(4)]
    delta=18-a['vtheta'][:len(a['positions_E'])]
    assert delta.min()>=-1e-6
    axes[0].scatter(*a['positions_E'][::3].T,c=delta[::3],s=1.2,cmap='plasma',vmin=0,vmax=max(delta.max(),1e-8),rasterized=True,linewidths=0)
    geometry(axes[0],c,a,True);axes[0].set_title('局部易激双核',fontsize=11)
    kernel=physics['graph']['kernel']
    axes[0].add_patch(Ellipse((16,16),2*kernel['l_par'],2*kernel['l_perp'],angle=kernel['theta_deg'],facecolor='white',edgecolor='black',lw=.7,zorder=10))
    axes[0].text(16,18,'EE 1/e 概率核',fontsize=6,ha='center',bbox=dict(fc='white',ec='none',alpha=.8),zorder=11)
    fig.colorbar(ScalarMappable(Normalize(0,max(delta.max(),1e-8)),cmap='plasma'),ax=axes[0],location='bottom',shrink=.75,pad=.14,label='E 阈值降低量 (mV)')
    fields={lab:native_timing(r,a,x['event'])[0] for lab,x in rep.items()}
    valid=np.concatenate([q[np.isfinite(q)] for q in fields.values()]) if fields else np.array([0.,250.])
    vmin=float(valid.min());vmax=max(float(valid.max()),vmin+1.)
    for ax,lab in zip(axes[1:3],['TA','TB']):
        if lab in rep:
            i=rep[lab]['event'];q=fields[lab]
            cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#eeeeee')
            ax.imshow(q,origin='lower',extent=[0,20,0,20],interpolation='nearest',cmap=cmap,vmin=vmin,vmax=vmax)
            ax.set_title(f'模型 {lab} · 事件 {i}\n该类观测 n={rep[lab]["n"]}',fontsize=10,color=an.MODE_COLOR[lab])
        else:
            ax.set_facecolor('#eeeeee');ax.text(.5,.5,'本运行未观测到该类',ha='center',va='center',transform=ax.transAxes,fontsize=9);ax.set_title('模型 '+lab)
        geometry(ax,c,a)
    fig.colorbar(ScalarMappable(Normalize(vmin,vmax),cmap='viridis'),ax=axes[1:3],location='bottom',pad=.14,shrink=.85,label='距窗口起点的 10% 累计活动时间 (ms)')
    names=list(a['contact_names']);order=[names.index(x) for x in an.DISPLAY];env=a['contact_envelope'][:,order].T
    envelope_vmax=max(float(np.quantile(env[env>0],.99)),1e-8) if (env>0).any() else 1
    ax=axes[3];ax.imshow(env,aspect='auto',extent=[0,r['actual_duration_ms']/1000,14.5,-.5],cmap='magma',vmin=0,vmax=envelope_vmax,interpolation='nearest',rasterized=True)
    for lab,item in rep.items():
        lo,hi=r['events'][item['event']]['window_ms'];color=an.MODE_COLOR[lab]
        ax.axvline((lo+hi)/2000,color=color,lw=.8);ax.text((lo+hi)/2000,-.8,lab,color=color,ha='center',fontsize=7,clip_on=False)
    ax.set(yticks=range(15),yticklabels=an.DISPLAY,xlabel='实际时间 (s)',title='同一网络：完整连续电极包络');ax.tick_params(axis='y',labelsize=7)
    population_title='原合格孤立窗' if population=='primary' else '全部检测，仅作开发诊断'
    fig.suptitle(review.display(c)+f'｜网络 {c["topology"]}，噪声 {seed}\n'+population_title+'：各类自身均值附近示例；连续读出保留全记录',fontsize=11)
    stem=c['id']+f'_{seed}_same_network'
    for ext in ['png','pdf']:fig.savefig(out/f'{stem}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    return dict(file=stem,candidate=c['id'],seed=seed,representatives=rep,arrays_sha256=r['arrays_sha256'],
        native_resolution_mm=1,native_frame_ms=float(a['sheet_activity_frame_ms']),native_shared_color_limits_ms=[vmin,vmax],applied_EE_kernel=kernel,native_time_definition='10 percent total bin mass within full 250ms window; no causal-source inference',
        display_population='frozen primary events' if population=='primary' else 'all detected developmental events; no change to primary scorer',raster_population='all contacts and all recorded time')


def spectral_comparison(c,units,seeds,out,patient,population='all'):
    canonical,meta,pat,names,porder=patient
    ncols=1+len(seeds)
    fig,axes=plt.subplots(2,ncols,figsize=(13*ncols/3,8),layout='constrained',squeeze=False)
    provenance=[]
    # Shared real-time limits include complete patient and model windows. No time dilation.
    xlo=min(p['tile_lo_ms'] for p in pat.values());xhi=max(p['tile_hi_ms'] for p in pat.values());models={}
    for col,seed in enumerate(seeds,1):
      if seed not in units:continue
      r,a,ids=units[seed];rep=representatives(a,ids)
      for lab,item in rep.items():
        i=item['event'];lo,hi=r['events'][i]['window_ms'];t=a['centroid_ms'][i];part=np.isfinite(t)
        zero=float(np.min(t[part]));dt=float(a['contact_envelope_dt_ms'])
        model_order=[list(a['contact_names']).index(names[k]) for k in porder]
        sel=model_order
        mass=a['contact_envelope'][round(lo/dt):round(hi/dt),sel].T
        models[(lab,col)]=dict(mass=mass,usable=part[sel],times=t[sel]-zero,labels=a['contact_names'][sel],lo=lo-zero,hi=hi-zero,event=i,n=item['n'],seed=seed)
        xlo=min(xlo,lo-zero);xhi=max(xhi,hi-zero)
        provenance.append(dict(candidate=c['id'],seed=seed,mode=lab,event=i,model_zero_absolute_ms=zero,window_ms=[lo,hi],participating_contacts=a['contact_names'][[j for j in sel if part[j]]].tolist(),display_contact_order=list(display.CONTACT_ORDER),display_ylim=list(display.YLIM)))
    for row,lab in enumerate(['TA','TB']):
      e=pat[lab];ax=axes[row,0]
      # Canonical function retains true STFT cell edges and dominant-enhancement centroids.
      display.patient_readout(ax,e,names,canonical,(xlo,xhi),an.MODE_COLOR[lab])
      ax.set_box_aspect(None);ax.set_facecolor('#b6b6b6');ax.set_xlabel('相对最早参与质心 (ms)')
      ax.set_title(f'患者 {lab} · Fig2C 事件 {e["event"]}\n真实 STFT 幅度',fontsize=11)
      for col,seed in enumerate(seeds,1):
        ax=axes[row,col];m=models.get((lab,col))
        if m is None:
            state='该运行尚无完整输出' if seed not in units else '完整运行中未观测到该类'
            ax.set_facecolor('#eeeeee');ax.text(.5,.5,state,ha='center',va='center',transform=ax.transAxes)
            display.contact_axis(ax);ax.set(xlim=(xlo,xhi),title=f'模型 {lab} · 噪声 {seed}');continue
        env=m['mass']/np.maximum(m['mass'].max(1,keepdims=True),1e-20);n=len(env)
        env[~m['usable']]=np.nan
        cmap=plt.get_cmap('magma').copy();cmap.set_bad('#777777')
        ax.imshow(env,aspect='auto',extent=[m['lo'],m['hi'],n-.5,-.5],cmap=cmap,vmin=0,vmax=1,interpolation='nearest')
        display.centroid_lines(ax,m['times'],color=an.MODE_COLOR[lab])
        for j in np.flatnonzero(~m['usable']):
            ax.text((xlo+xhi)/2,j,'未参与 / 无有效质心',ha='center',va='center',fontsize=6,color='white')
        display.contact_axis(ax)
        ax.set(yticks=range(n),yticklabels=m['labels'],xlim=(xlo,xhi),xlabel='相对最早参与质心 (ms)',title=f'模型 {lab} · 噪声 {seed}\n发放密度包络；事件 {m["event"]}，n={m["n"]}')
        ax.set_facecolor('#b6b6b6');ax.tick_params(axis='y',labelsize=8)
      for ax in axes[row]:ax.axvline(0,color='black',lw=.7,ls='--')
    population_title='模型示例来自原合格孤立窗' if population=='primary' else '模型示例来自全部检测，仅作开发诊断'
    fig.suptitle(review.display(c)+f'｜网络 {c["topology"]}\n'+population_title+'；固定15行：SCL9–6，然后ICL11–1，未参与触点保留原位\n各例仅平移时间零点；浅灰=记录窗外，深灰=未参与；无时间拉伸',fontsize=11)
    stem=c['id']+'_patient_spectra_model_envelopes'
    fig.savefig(out/f'{stem}.png',dpi=180,bbox_inches='tight')
    with Image.open(out/f'{stem}.png') as im:im.convert('RGB').save(out/f'{stem}.pdf',resolution=180.)
    plt.close(fig)
    return dict(file=stem,topology_seed=c['topology'],display_contract='E1146_fixed_shaft_rows_v1',display_ylim=list(display.YLIM),absent_rows_preserved=True,model_examples=provenance,completed_seeds=sorted(units),pending_seeds=[s for s in seeds if s not in units],patient_cache=str(CACHE),patient_cache_sha256=rt.sha(CACHE),
        patient_example_rule='frozen Fig2C direction-qualified illustrations, not unconditional mode medoids',
        patient_quantity='per-contact normalized STFT magnitude; exact frozen Fig1a arrays and cell edges, fixed contact slots',model_quantity='per-contact normalized spike-density envelope; not model HFO spectrum',
        model_population=population,common_contact_order=names[porder].tolist(),common_xlim_ms=[xlo,xhi])


def main(phase,population='all',candidates=None,spectra_only=False):
    old,plan,spec,cases=review.stage_cases(phase);out=review.night.OUT/('main_review_'+phase+('_primary' if population=='primary' else '')+'_shaft_fixed');F=out/'figures';F.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':9,'pdf.fonttype':42})
    patient=patient_payloads();records=[]
    for c in cases:
        if candidates and c['base_id'] not in candidates:continue
        seeds=sorted({int(s) for cid,t,s in spec['units'] if cid==c['base_id'] and int(t)==c['topology']}) or old['seeds']
        units={}
        for seed in seeds:
            path=an.run.result_path(c['output_stage'],c['base_id'],c['topology'],seed);unit=an.load_unit(path,old['analysis']['burnin_ms'])
            if unit is None:continue
            r,a,primary=unit;ids=primary if population=='primary' else an.all_detected_ids(r,old['analysis']['burnin_ms']);units[seed]=(r,a,ids)
            if not spectra_only:records.append(four_panel(c,seed,r,a,ids,F,rt.read(path.parent.parent/'applied_physics.json'),population))
        if units:records.append(spectral_comparison(c,units,seeds,F,patient,population))
    rt.write(out/'manifest.json',dict(status='RENDERED_PENDING_SCIENTIFIC_REVIEW',phase=phase,model_population=population,producer=str(Path(__file__)),producer_sha256=rt.sha(Path(__file__)),
        patient_renderer_sha256=rt.sha(MAIN/'scripts/plot_topic5_interictal_event_envelope_field.py'),display_helper_sha256=rt.sha(MAIN/'src/snn_contact_display.py'),records=records,scientific_acceptance=False))
    notes=[]
    for path in sorted(F.glob('*.png')):
        with Image.open(path) as im:im.load()
        body=('同一实际网络的阈值降低量、TA/TB各自均值附近的单事件原生时间场和完整连续电极包络。原生时间为1 mm网格在完整250 ms窗内达到10%累计发放的时间；空白为无发放，未作插值，不代表因果起源；标签来自冻结分类器而非预设source。**关注点**：完整核外活动、两个模式的实际时序及连续事件支持；罕见模式的一例不代表稳定恢复。' if '_same_network' in path.name else '左列直接复用Fig2C缓存的真实STFT幅度、cell edges和质心，右列是两条噪声的模型发放密度包络；两种物理量不能当作同一频谱。全部面板固定SCL9–6、ICL11–1这15行，未参与触点以深灰保留位置，杆间不连质心线；保留完整记录时窗，只平移到最早参与质心，全触点信号另见QC；患者示例经过原方向筛选。**关注点**：已参与触点的顺序、局部持续和杆间时间，需结合全分布及原生动画。')
        notes.append(f'### {path.name}\n\n模型选例集合：'+('原primary合格孤立窗，缺少模式时不从其他窗口补样例。' if population=='primary' else '全部检测的开发诊断，未替换原primary评分。')+body)
    (F/'README.md').write_text('\n\n'.join(notes)+'\n')
    print(json.dumps(dict(output=str(out),records=len(records)),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',default='wave1');parser.add_argument('--population',choices=['all','primary'],default='all');parser.add_argument('--candidates',nargs='*');parser.add_argument('--spectra-only',action='store_true');args=parser.parse_args();main(args.phase,args.population,args.candidates,args.spectra_only)
