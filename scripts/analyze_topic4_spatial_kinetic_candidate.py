"""Native/kinetic temporal and full-sheet correspondence, no bifurcation claims."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from run_topic4_spatial_kinetic_candidate import OUT,SOURCE,ROOT,write
from audit_topic4_fig5_native_reduction_correspondence import finite_events,first,compare_arrivals

AUDIT=ROOT/'results/topic4_sef_hfo/fig5_native_reduction_correspondence_20260916'
GEO=dict(np.load(SOURCE/'approx/coarse_20/geometry.npz'))
COUNTS=GEO['count_e']
CENTERS=GEO['centers_mm']
NATIVE={}


def native(seed):
    if seed in NATIVE:return NATIVE[seed]
    data=np.load(AUDIT/f'paired_fields_{seed}.npz')
    cells=[];regions=[]
    for p in sorted((SOURCE/f'replay/runs/eta0.0005_s{seed}/chunks').glob('*.npz')):
        with np.load(p) as z:
            if int(z['end_step'])<=80000:continue
            cells.append(z['field_1ms'].astype(float));regions.append(z['regions_1ms'][:,:3].astype(float))
    field1=np.concatenate(cells)/COUNTS/.001
    rc=np.bincount(GEO['g175'],minlength=3)
    region1=np.concatenate(regions)/rc/.001
    # This audit requires exact matching counts, not an unweighted spatial mean.
    assert np.max(abs(field1.reshape(-1,10,400).mean(1)-data['native_rate10_hz']))<1e-9
    assert np.max(abs(np.average(region1,weights=rc,axis=1)-np.average(field1,weights=COUNTS,axis=1)))<1e-9
    NATIVE[seed]=(field1,region1,data['reduced_rate10_hz'])
    return NATIVE[seed]


def event_stats(field1,region1,end_s=9.42):
    n=min(int(round((end_s-8)*1000)),len(field1));n=n//10*10
    field10=field1[:n].reshape(-1,10,400).mean(1)
    r=np.average(field10,weights=COUNTS,axis=1)
    _,ev=finite_events(r)
    # Five-ms trailing population-rate window, arrival sustained for five ms.
    cs=np.vstack([np.zeros((1,400)),np.cumsum(field1,axis=0)])
    i=np.arange(len(field1))+1;left=np.maximum(i-5,0)
    sm=(cs[i]-cs[left])/(i-left)[:,None]
    rows=[];maps=[]
    for e in ev:
        a,b=e['start_bin']*10,e['end_bin']*10
        arrival=np.full(400,np.nan);censored=np.any(sm[max(0,a-10):a]>=50.,axis=0) if a else np.ones(400,bool)
        for c in range(400):
            if censored[c]:continue
            on=sm[a:b,c]>=50.;sums=np.convolve(on.astype(int),np.ones(5,int),mode='valid') if len(on)>=5 else np.array([])
            hits=np.flatnonzero(sums>=5)
            if len(hits):arrival[c]=hits[0]
        valid=np.isfinite(arrival)
        early=region1[a:min(a+30,b),:2].mean(0)
        core='A' if early[0]>2*early[1] else ('B' if early[1]>2*early[0] else 'both')
        peak=int(np.argmax(np.average(field1[a:b],weights=COUNTS,axis=1)))+a
        span=float(np.quantile(arrival[valid],.9)-np.quantile(arrival[valid],.1)) if valid.sum()>1 else None
        row=dict(**e,early_core_activity=core,early_core_A_hz=float(early[0]),early_core_B_hz=float(early[1]),
             peak_time_s=8.+peak*.001,recruited_E_fraction=float(COUNTS[valid].sum()/COUNTS.sum()),
             left_censored_E_fraction=float(COUNTS[censored].sum()/COUNTS.sum()),arrival_10_90_ms=span,
             peak_active_E_fraction=float(np.max(np.average(field10[e['start_bin']:e['end_bin']]>=50.,weights=COUNTS,axis=1))))
        rows.append(row);maps.append(arrival)
    return rows,maps


def summarize(field1,region1):
    n=len(field1)//10*10;field10=field1[:n].reshape(-1,10,400).mean(1)
    r=np.average(field10,weights=COUNTS,axis=1);windows={}
    for label,lo,hi in [('pilot',8.,9.),('pre_entry',8.,9.42),('entry',9.42,10.37),('high_tail',10.37,12.5)]:
        a,b=round((lo-8)*100),round((hi-8)*100)
        if b>len(r):continue
        _,e=finite_events(r[a:b],start=lo)
        windows[label]=dict(mean_global_E_hz=float(r[a:b].mean()),quiet_fraction=float((r[a:b]<5).mean()),
                  finite_events=len(e),mean_core_hz=region1[a*10:b*10,:2].mean(0).tolist())
    ev,maps=event_stats(field1,region1,min(9.42,8.+n*.001))
    return dict(windows=windows,high_onset_s=first(r>=200,20),events=ev),maps


def event_comparison(nrows,nmaps,krows,kmaps):
    pairs=[]
    for i,(nr,nm) in enumerate(zip(nrows,nmaps)):
        for j,(kr,km) in enumerate(zip(krows,kmaps)):
            if nr['early_core_activity']!=kr['early_core_activity']:continue
            a=np.isfinite(nm);b=np.isfinite(km);joint=a&b;union=a|b
            rho=None;error=None
            if joint.sum()>=5:
                d=km[joint]-nm[joint];error=float(np.median(abs(d-np.median(d))))
                if len(np.unique(nm[joint]))>1 and len(np.unique(km[joint]))>1:rho=float(spearmanr(nm[joint],km[joint]).statistic)
            pairs.append(dict(native_event=i,candidate_event=j,early_core_activity=nr['early_core_activity'],
               valid_joint_cells=int(joint.sum()),recruited_E_iou=float(COUNTS[joint].sum()/COUNTS[union].sum()) if union.any() else None,
               arrival_order_spearman=rho,arrival_offset_removed_mae_ms=error))
    def med(key):
        x=[p[key] for p in pairs if p[key] is not None];return float(np.median(x)) if x else None
    return dict(native_events=len(nrows),candidate_events=len(krows),conditioned_pairs=len(pairs),
          median_recruited_E_iou=med('recruited_E_iou'),median_arrival_order_spearman=med('arrival_order_spearman'),
          median_offset_removed_mae_ms=med('arrival_offset_removed_mae_ms'),pairs=pairs,
          scope='All cross-event pairs conditioned on early core activity, no best-match selection. Pairs/cells are not independent samples. Not exact wave identity.')


def spatial(ax,field,label=''):
    im=ax.imshow(field.reshape(20,20),origin='lower',extent=(0,20,0,20),cmap='magma',vmin=0,vmax=500,interpolation='nearest')
    for j,xy in enumerate(CENTERS):
        ax.add_patch(Circle(xy,1.5,fc='none',ec='#1cd3d5',lw=1))
        ax.text(xy[0],xy[1]+2.1,'AB'[j],ha='center',fontsize=8,color='#117d81',bbox=dict(fc='white',ec='none',pad=.2))
    ax.set(xticks=(0,10,20),yticks=(0,10,20))
    if label:ax.set_ylabel(label+'\ny (mm)')
    return im


def main():
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    all_results={};full={};resources={}
    for folder in sorted((OUT/'runs').iterdir()):
        if not (folder/'fields.npz').exists():continue
        status=json.loads((folder/'status.json').read_text())
        if status['status']!='COMPLETE':continue
        cfg=status['config'];seed=cfg['seed'];z=np.load(folder/'fields.npz')
        if cfg['start_ms']!=8000:continue
        if 'field_e20_1ms' in z:
            assert np.array_equal(z['count_e20'],COUNTS)
            f=z['field_e20_1ms']/COUNTS/.001
        else:
            assert cfg['grid']==20 and np.array_equal(z['counts'][:400],COUNTS)
            f=z['spikes_1ms'][:,:400]/COUNTS/.001
        if 'regions_1ms' in z:
            reg=z['regions_1ms']/z['region_counts']/.001
            assert np.max(abs(np.average(reg,weights=z['region_counts'],axis=1)-np.average(f,weights=COUNTS,axis=1)))<1e-9
            region_scope='exact native membership'
        else:
            weights=np.array([np.bincount(GEO['cell_e'][GEO['g175']==k],minlength=400) for k in range(3)])
            reg=np.array([np.average(f,weights=w,axis=1) for w in weights]).T
            region_scope='cell-overlap proxy; pilot did not record per-neuron regions'
        nf,nr,_=native(seed);nat,nmaps=summarize(nf[:len(f)],nr[:len(f)]);cand,cmaps=summarize(f,reg)
        result=dict(config=cfg,native=nat,candidate=cand,region_scope=region_scope,
              event_propagation=event_comparison(nat['events'],nmaps,cand['events'],cmaps))
        if cand['high_onset_s'] is not None:
            onset_index=int(round((cand['high_onset_s']-8)*100))
            # Use last saved pre-bin state, not a future state inside the onset bin.
            before=max(0,onset_index-1);nc=cfg['grid']**2;cc=z['counts'][:nc]
            zm=z['slow_10ms'][before,:nc];mm=z['slow_10ms'][before,2*nc:3*nc]
            cp=np.load(SOURCE/f'replay/runs/eta0.0005_s{seed}/checkpoints/t9870ms.npz')
            result['resource_at_high_onset']=dict(candidate_D=float(1-np.average(zm,weights=cc)),
                    native_D=float(1-cp['slow__z'][:32000].mean()),candidate_M=float(np.average(mm,weights=cc)),
                    native_M=float(cp['slow__m'][:32000].mean()),candidate_onset_s=cand['high_onset_s'],
                    native_onset_s=9.87,temporal_offset_s=float(cand['high_onset_s']-9.87),
                    scope='Operational sustained >=200Hz threshold; state sampled before onset bin. Not a bifurcation parameter estimate.')
            grid_path=SOURCE/f'approx/coarse_{cfg["grid"]}' if cfg['grid'] in (10,20) else OUT/f'coarse_{cfg["grid"]}'
            fine_cells=np.load(grid_path/'geometry.npz')['cell_e']
            projected=np.bincount(GEO['cell_e'],weights=zm[fine_cells],minlength=400)/COUNTS
            native_z=np.bincount(GEO['cell_e'],weights=cp['slow__z'][:32000],minlength=400)/COUNTS
            result['resource_at_high_onset']['Z_field_weighted_rmse']=float(np.sqrt(np.average((projected-native_z)**2,weights=COUNTS)))
            result['resource_at_high_onset']['Z_field_spearman']=float(spearmanr(projected,native_z).statistic)
            if cfg['grid']==20 and cfg['closure']=='shot':resources[seed]=(native_z,projected)
        if len(f)>=2370:
            result['transition_recruitment']=compare_arrivals(nf.reshape(-1,10,400).mean(1),f.reshape(-1,10,400).mean(1),COUNTS,50.)[0]
            full[(seed,cfg['closure'],cfg['grid'])]=(f,reg,z['slow_10ms'],z['counts'])
        write(folder/'comparison.json',result);all_results[folder.name]=result
    baselines=[]
    for seed in (9108401,9108402):
        f,r,_=native(seed);res,maps=summarize(f,r);baselines.append((res,maps))
    cross=event_comparison(baselines[0][0]['events'],baselines[0][1],baselines[1][0]['events'],baselines[1][1])
    write(OUT/'comparison.json',dict(runs=all_results,native_noise_event_baseline=cross,
           equivalence_gate='NOT_ESTABLISHED',reason='Development references only; spatial fidelity and frozen-Z intervention correspondence must qualify before bifurcation.'))
    if len(resources)==2:
        fig,axs=plt.subplots(2,3,figsize=(9.4,5.5),layout='constrained',sharex=True,sharey=True)
        archive={}
        for row,seed in enumerate((9108401,9108402)):
            nz,kz=resources[seed];dn,dk=1-nz,1-kz
            archive[f'native_D_s{seed}']=dn;archive[f'candidate_D_s{seed}']=dk
            for col,(v,label) in enumerate(((dn,'Native SNN'),(dk,'Kinetic candidate'),(dk-dn,'Candidate − native'))):
                ax=axs[row,col]
                im=ax.imshow(v.reshape(20,20),origin='lower',extent=(0,20,0,20),cmap='magma' if col<2 else 'RdBu_r',vmin=0 if col<2 else -.1,vmax=.6 if col<2 else .1,interpolation='nearest')
                if col<2:baseim=im
                else:diffim=im
                for xy in CENTERS:ax.add_patch(Circle(xy,1.5,fill=False,ec='#12caca',lw=.9))
                if row==0:ax.text(.5,1.03,label,ha='center',transform=ax.transAxes)
                if col==0:ax.set_ylabel(f'Stream {row+1}\ny (mm)')
                if row==1:ax.set_xlabel('x (mm)')
                ax.set(xticks=(0,10,20),yticks=(0,10,20))
        fig.colorbar(baseim,ax=axs[:,:2].ravel().tolist(),label='Local depletion 1 − Z',fraction=.025,pad=.025)
        fig.colorbar(diffim,ax=axs[:,2].ravel().tolist(),label='Depletion difference',fraction=.045,pad=.03)
        for ext in ('png','pdf','svg'):fig.savefig(figdir/f'fig_entry_resource_fields_g20.{ext}',dpi=220,bbox_inches='tight')
        plt.close(fig);np.savez_compressed(OUT/'entry_resource_fields_g20.npz',**archive,count_e=COUNTS)
    # Short pilot in its entirety, before subsequent selections.
    fig,ax=plt.subplots(2,1,figsize=(9,4.6),sharex=True,layout='constrained')
    nf,_,old=native(9108401);native10=nf[:1000].reshape(-1,10,400).mean(1)
    time=8.+(np.arange(100)+.5)*.01
    ax[0].plot(time,np.average(native10,weights=COUNTS,axis=1),color='black',lw=1.1,label='Native SNN')
    ax[0].plot(time,np.average(old[:100],weights=COUNTS,axis=1),color='#ba5149',lw=1,label='Previous rate model')
    ax[0].set_ylabel('Global E rate (Hz)');ax[0].legend(frameon=False,ncol=2)
    ax[1].plot(time,np.average(native10,weights=COUNTS,axis=1),color='black',lw=1.1,label='Native SNN')
    for closure,color in [('shot','#078571'),('mean','#be8615')]:
        folder=OUT/'runs'/f'g20_{closure}_s9108401_8000_1000ms'
        if not (folder/'fields.npz').exists():continue
        zz=np.load(folder/'fields.npz');r=zz['spikes_1ms'][:,:400].sum(1).reshape(-1,10).sum(1)/320
        ax[1].plot(time,r,color=color,lw=1,label='Kinetic '+closure)
    ax[1].set(xlabel='Time (s)',ylabel='Global E rate (Hz)');ax[1].legend(frameon=False,ncol=3)
    for a in ax:a.set_ylim(0,250);a.set_xlim(8,9)
    for ext in ('png','pdf','svg'):fig.savefig(figdir/f'fig_kinetic_pilot.{ext}',dpi=220)
    plt.close(fig)
    for grid,closure in ((20,'shot'),(40,'shot'),(40,'mean')):
        if not any(k[2]==grid and k[1]==closure for k in full):continue
        suffix='' if closure=='shot' else '_mean'
        fig,axs=plt.subplots(2,2,figsize=(10,5),sharex='col',layout='constrained')
        for j,seed in enumerate((9108401,9108402)):
            nf,_,old=native(seed);t=8.+(np.arange(450)+.5)*.01
            axs[0,j].plot(t,np.average(nf.reshape(-1,10,400).mean(1),weights=COUNTS,axis=1),color='black',lw=.8,label='Native SNN')
            if (seed,closure,grid) not in full:continue
            f,_,slow,counts=full[(seed,closure,grid)];nt=len(f)//10;n=grid**2
            axs[0,j].plot(t[:nt],np.average(f.reshape(-1,10,400).mean(1),weights=COUNTS,axis=1),color='#078571',lw=.8,label='Kinetic candidate')
            axs[1,j].plot(t[:nt],1-np.average(slow[:,:n],weights=counts[:n],axis=1),color='#078571',label='Kinetic candidate')
            tt=[ms for ms in [8000,9000,9300,9420,9870,10370,12500] if ms<=8000+len(f)];dd=[]
            for ms in tt:
                with np.load(SOURCE/f'replay/runs/eta0.0005_s{seed}/checkpoints/t{ms}ms.npz') as zz:dd.append(1-float(zz['slow__z'][:32000].mean()))
            axs[1,j].plot(np.array(tt)/1000,dd,'o-',color='black',ms=3,label='Native SNN')
            axs[0,j].text(.03,.95,f'Stream {j+1}',transform=axs[0,j].transAxes,va='top')
            axs[1,j].set_xlabel('Time (s)')
            axs[1,j].set_xlim(8,8+len(f)/1000)
        axs[0,0].set_ylabel('Global E rate (Hz)');axs[1,0].set_ylabel(r'$D=1-\langle Z_E\rangle$')
        axs[0,1].legend(frameon=False,loc='lower right');axs[1,1].legend(frameon=False)
        for ext in ('png','pdf','svg'):fig.savefig(figdir/f'fig_autonomous_dynamics_g{grid}{suffix}.{ext}',dpi=220)
        plt.close(fig)
        for seed in (9108401,9108402):
            if (seed,closure,grid) not in full:continue
            nf,_,_=native(seed);kf=full[(seed,closure,grid)][0]
            fig,axs=plt.subplots(2,3,figsize=(8.8,5.3),layout='constrained',sharex=True,sharey=True)
            for j,t in enumerate((9.42,9.87,10.37)):
                a,b=round((t-8)*1000)-25,round((t-8)*1000)+25
                for i,(f,label) in enumerate([(nf,'Native SNN'),(kf,'Kinetic candidate')]):
                    im=spatial(axs[i,j],f[a:b].mean(0),label if j==0 else '')
                    if i==0:axs[i,j].text(.5,1.03,f'{t:.3f} s',transform=axs[i,j].transAxes,ha='center')
                    if i==1:axs[i,j].set_xlabel('x (mm)')
            fig.colorbar(im,ax=axs.ravel().tolist(),label='E rate (Hz)',fraction=.025,pad=.025)
            for ext in ('png','pdf','svg'):fig.savefig(figdir/f'fig_spatial_correspondence_g{grid}_s{seed}{suffix}.{ext}',dpi=220)
            plt.close(fig)
            key=f'g{grid}_{closure}_s{seed}_8000_{len(kf)}ms'
            own_onset=all_results[key]['candidate']['high_onset_s']
            if own_onset is not None and own_onset+.225<=8+len(kf)/1000:
                fig,axs=plt.subplots(2,3,figsize=(8.8,5.3),layout='constrained',sharex=True,sharey=True)
                for j,offset in enumerate((-.45,0.,.2)):
                    for i,(f,t0,label) in enumerate([(nf,9.87,'Native SNN'),(kf,own_onset,'Kinetic candidate')]):
                        time=t0+offset;a,b=round((time-8)*1000)-25,round((time-8)*1000)+25
                        im=spatial(axs[i,j],f[a:b].mean(0),label if j==0 else '')
                        axs[i,j].text(.5,1.03,f'{time:.3f} s',transform=axs[i,j].transAxes,ha='center')
                        if i==1:axs[i,j].set_xlabel('x (mm)')
                fig.colorbar(im,ax=axs.ravel().tolist(),label='E rate (Hz)',fraction=.025,pad=.025)
                for ext in ('png','pdf','svg'):fig.savefig(figdir/f'fig_entry_aligned_g{grid}_s{seed}{suffix}.{ext}',dpi=220)
                plt.close(fig)
    names=[p.name for p in figdir.glob('*.png') if not p.name.startswith('fig_frozen_Z')]
    text=[]
    for name in names:
        if 'pilot' in name:desc='同一8–9s窗口比较原SNN、旧率模型与两种空间动理学闭合。纵轴是每神经元平均放电率，不是振荡频率；两个子图使用相同尺度。**关注点**：新模型是否保留会结束的事件以及事件之间的低活动。'
        elif 'first_event' in name:desc='首个完整自限事件的同一物理时钟下10ms中心窗口，比较原SNN与1mm动理学shot候选。完整20×20mm视野、同一色标和固定A/B核，不按图像相似度平移时间；这是单次事件示例。**关注点**：局部向二维场扩展、峰值后衰退是否相容；单例不能取代多事件与发作进入验收。'
        elif 'entry_resource' in name:desc='1mm候选与原SNN各自在操作性高率进入时的局部耗减1−Z场，两行为两条输入；第三列为候选减原生。候选使用进入10ms率窗之前的资源帧，原生使用9.87s完整检查点。**关注点**：全局D接近时，整张Z场是否也接近；这里的行为进入不是已经证明的分岔。'
        elif 'entry_aligned' in name:desc='分别以各模型连续200ms达到全局200Hz的首次进入点对齐，显示进入前450ms、进入时和后200ms的50ms中心窗口。上下排的实际时间明确标出；没有用空间图像相似度挑选时间。**关注点**：区分到达时间偏差与到达状态的空间差异；必须与同一绝对时钟的空间对照并看，不能用对齐掩盖时钟偏差。'
        elif 'autonomous' in name:desc='两条配对外源输入下的自主续接，1mm版覆盖8–12.5s，0.5mm版覆盖8–10.5s；Z和M都动态更新。下排D为实际模型Z读出，原生圆点来自完整状态检查点，连线仅供读图。**关注点**：恢复短程burst后是否也恢复全局高率进入与资源变化，不能只看早期片段。'
        else:desc='同一物理时刻、同一50ms中心窗口的完整20×20mm活动场，上排原生SNN，下排新候选。各图色标均为0–500Hz，A/B圆标示原核位置。**关注点**：局部活动、两核协同和向全场招募是否一致；配对外源输入不保证随机闭合后逐事件同相。'
        text.append('### '+name+'\n\n'+desc+'\n')
    (figdir/'README.md').write_text('\n'.join(text))
    frozen_results={}
    fig,axs=plt.subplots(2,1,figsize=(9,4.6),sharex=True,layout='constrained')
    for row,zms in enumerate((8000,9870)):
        tag=f'g20_shot_s9108401_10370_2000ms_z{zms}_h8000';folder=OUT/'runs'/tag
        if not (folder/'qa.json').exists():continue
        z=np.load(folder/'fields.npz');n=2000
        kr=z['spikes_1ms'][:,:400].sum(1).reshape(-1,10).sum(1)/320
        native_folder=SOURCE/f'native/runs/z{zms}_h8000_W1'
        pieces=[];field_pieces=[]
        for path in sorted((native_folder/'chunks').glob('*.npz')):
            with np.load(path) as zz:
                pieces.append(zz['spikes_1ms'][:,0].astype(float))
                field_pieces.append(zz['field_1ms'].astype(float))
                if sum(map(len,pieces))>=n:break
        nr=np.concatenate(pieces)[:n].reshape(-1,10).sum(1)/320
        nf=np.concatenate(field_pieces)[:n]/COUNTS/.001
        kf=z['field_e20_1ms']/COUNTS/.001
        stats={}
        for label,r in [('native',nr),('candidate',kr)]:
            q,e=finite_events(r,start=0.)
            stats[label]=dict(mean_global_E_hz=float(r.mean()),quiet_fraction=float((r<5).mean()),finite_events=len(e),
              sustained200Hz_onset_s=first(r>=200,20,start=0.),tail500ms_mean_hz=float(r[-50:].mean()))
        frozen_results[tag]=dict(z_field_ms=zms,history_ms=8000,clock_start_ms=10370,observation_ms=2000,**stats,
                   scope='Matched first 2s only; not the original 10-20s classification.')
        nm=nf[-500:].mean(0);km=kf[-500:].mean(0)
        dx=nm-np.average(nm,weights=COUNTS);dy=km-np.average(km,weights=COUNTS)
        rho=np.average(dx*dy,weights=COUNTS)/np.sqrt(np.average(dx**2,weights=COUNTS)*np.average(dy**2,weights=COUNTS))
        union=(nm>=50)|(km>=50);joint=(nm>=50)&(km>=50)
        frozen_results[tag]['tail_spatial_field']=dict(window_after_transplant_s=[1.5,2.],
            weighted_rmse_hz=float(np.sqrt(np.average((nm-km)**2,weights=COUNTS))),weighted_correlation=float(rho),
            active_E_iou=float(COUNTS[joint].sum()/COUNTS[union].sum()) if union.any() else None)
        t=(np.arange(200)+.5)*.01
        axs[row].plot(t,nr,color='black',lw=1,label='Native SNN')
        axs[row].plot(t,kr,color='#078571',lw=1,label='Kinetic candidate')
        axs[row].set_ylabel('Global E rate (Hz)')
        axs[row].text(.02,.94,f'Z field: {zms/1000:.2f} s',transform=axs[row].transAxes,va='top')
    if frozen_results:
        write(OUT/'frozen_comparison.json',frozen_results)
        axs[0].legend(frameon=False,loc='upper right');axs[1].set_xlabel('Time after transplant (s)')
        for ext in ('png','pdf','svg'):fig.savefig(figdir/f'fig_frozen_Z_comparison.{ext}',dpi=220)
        with (figdir/'README.md').open('a') as fp:
            fp.write('\n### fig_frozen_Z_comparison.png\n\n相同8s快速历史/M移植到10.37s外部时钟，分别固定8s和9.87s的逐细胞Z，M保持动态。黑线原SNN、绿线新候选，均只取共同前2s。**关注点**：相同Z场下两侧的动力学是否恢复，不能将2s诊断替代长期边界验收。\n')
    plt.close(fig)
    if len(frozen_results)==2:
        fig,axes=plt.subplots(2,2,figsize=(6.7,5.9),layout='constrained')
        for row,zms in enumerate((8000,9870)):
            d=OUT/'runs'/f'g20_shot_s9108401_10370_2000ms_z{zms}_h8000'
            kf=np.load(d/'fields.npz')['field_e20_1ms']/COUNTS/.001
            fields=[]
            for p in sorted((SOURCE/f'native/runs/z{zms}_h8000_W1/chunks').glob('*.npz')):
                with np.load(p) as zz:fields.append(zz['field_1ms'].astype(float))
                if sum(map(len,fields))>=2000:break
            nf=np.concatenate(fields)[:2000]/COUNTS/.001
            for col,(f,label) in enumerate([(nf,'Native SNN'),(kf,'Kinetic candidate')]):
                im=spatial(axes[row,col],f[1500:2000].mean(0),f'Z: {zms/1000:.2f} s' if col==0 else '')
                if row==0:axes[row,col].text(.5,1.04,label,transform=axes[row,col].transAxes,ha='center')
                if row==1:axes[row,col].set_xlabel('x (mm)')
        fig.colorbar(im,ax=axes.ravel().tolist(),label='E rate (Hz)',fraction=.035,pad=.03)
        for ext in ('png','pdf','svg'):fig.savefig(figdir/f'fig_frozen_Z_spatial.{ext}',dpi=220,bbox_inches='tight',pad_inches=.12)
        plt.close(fig)
        with (figdir/'README.md').open('a') as fp:
            fp.write('\n### fig_frozen_Z_spatial.png\n\n固定相同Z场的原SNN与新候选，在移植后1.5–2.0s取相同500ms平均率场。上排8s的Z、下排9.87s的Z，M均动态；共享0–500Hz色标。**关注点**：持续态的空间支持和量值是否恢复；上排时间平均会淡化短暂传播，需结合事件帧和时序，不能当成瞬时状态。\n')
    print(json.dumps({k:{'native':v['native']['windows'],'candidate':v['candidate']['windows'],'onset':v['candidate']['high_onset_s'],'propagation':{x:y for x,y in v['event_propagation'].items() if x!='pairs'}} for k,v in all_results.items()},indent=2))


if __name__=='__main__':main()
