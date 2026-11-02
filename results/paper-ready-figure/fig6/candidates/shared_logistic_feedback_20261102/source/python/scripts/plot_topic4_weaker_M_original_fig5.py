#!/usr/bin/env python3
"""Primary original Fig5 layout for every completed weaker-M condition."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import plot_topic4_m_parameter_modes as f
import plot_topic4_fig5_resting_recovery as layout
import analyze_topic4_weaker_M_onset_pilot as pilot

OUT=pilot.OUT/'original_fig5'
ETA=[0.,.001,.0025,.005,.01,.02,.04]
TAU=[1.,2.,4.,8.,20.]


def measured_grid():
    """Original 40 first endpoints plus six new settings; never count reused controls twice."""
    inherited=f.grid_summary()
    original=f.read(f.OUT/'protocol.json');new=f.read(pilot.OUT/'protocol.json')
    assert original['identity']==new['identity'] and original['tau_Z_s']==new['tau_Z_s']==5.
    assert original['I_th']==new['I_th']
    times=np.full((7,5,2),np.nan);entered=times.copy();sources=[]
    for row in inherited['endpoint_rows']:
        if row['status']!='ESTABLISHED':continue
        i,j,k=ETA.index(row['eta_M']),TAU.index(row['tau_M_s']),pilot.SEEDS.index(row['seed'])
        times[i,j,k]=row['restricted_time_s'];entered[i,j,k]=row['observed'];sources.append(row)
    for job in new['jobs']:
        endpoint=f.committed_first_endpoint(pilot.OUT/'runs'/job['name'])
        if endpoint['status']!='ESTABLISHED':continue
        i,j,k=ETA.index(job['eta_m']),TAU.index(job['tau_M_s']),pilot.SEEDS.index(job['seed'])
        assert not np.isfinite(times[i,j,k]),'Duplicated physical trial'
        times[i,j,k]=endpoint['restricted_time_s'];entered[i,j,k]=endpoint['observed']
        sources.append(dict(name=job['name'],eta_M=job['eta_m'],tau_M_s=job['tau_M_s'],seed=job['seed'],**endpoint))
    count=np.isfinite(times).sum(-1);complete=count==2
    mean=np.full(count.shape,np.nan);prob=mean.copy()
    mean[complete]=times[complete].mean(-1);prob[complete]=entered[complete].mean(-1)
    result=dict(mean=mean,prob=prob,count=count,eta_M=ETA,tau_M_s=TAU,sources=sources,
        original_first_endpoints=inherited['established_first_endpoints'],additional_first_endpoints=len(sources)-inherited['established_first_endpoints'],
        baseline_reuse_adds_zero_samples=True,statistical_unit='One setting/noise seed on the same fixed topology; two seeds per measured cell.',
        endpoint='First all-E >=200Hz continuously200ms, confirmation time; restricted at180s.',
        grid_coordinates='Discrete tested settings, including eta=0; untested cells are grey, not interpolated.',
        scope='Early-refill and original late-refill protocols share the same pre-first-entry dynamics. This F estimates first entry only, not recovery or recurrence.')
    f.write(OUT/'F_measured_grid.json',f.safe(result))
    return result


def draw_grid(fig,spec,g,job):
    ax=fig.add_subplot(spec);cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#eeeeee')
    im=ax.pcolormesh(np.arange(6),np.arange(8),np.ma.masked_invalid(g['mean']),cmap=cmap,
                    vmin=0,vmax=180,edgecolors='#ffffff77',linewidth=.4,shading='flat')
    ax.set(xticks=np.arange(5)+.5,xticklabels=['1','2','4','8','20'],
           yticks=np.arange(7)+.5,yticklabels=['0','.001','.0025','.005','.01','.02','.04'],
           xlabel=r'$\tau_M$ (s)',ylabel=r'$\eta_M$')
    ax.set_title('F  M kinetics',loc='left',weight='bold')
    ax.text(1,1.025,'Entered by 180 s: n/2 · grey: untested',transform=ax.transAxes,ha='right',fontsize=10)
    for y,x in zip(*np.where(g['count']==2)):
        value=float(g['mean'][y,x]);n=int(round(2*g['prob'][y,x]))
        if n<2:ax.add_patch(Rectangle((x,y),1,1,fc='none',ec='#55555588',lw=0,hatch='///'))
        ax.text(x+.5,y+.5,f'{value:.1f}\n{n}/2',ha='center',va='center',fontsize=10,linespacing=.95,
                color='white' if value<90 else '#111111',bbox=dict(fc=cmap(value/180),ec='none',pad=.2))
    i,j=ETA.index(job['eta_m']),TAU.index(job['tau_M_s'])
    ax.add_patch(Rectangle((j,i),1,1,fc='none',ec='#111111',lw=1.8))
    cb=fig.colorbar(im,ax=ax,pad=.035);cb.set_label('Restricted mean first-entry time (s)')
    cb.set_ticks([0,45,90,135,180])


def render_original_package():
    OUT.mkdir(parents=True,exist_ok=True)
    grid=measured_grid();records=[]
    for row in pilot.sources():
        if not (row['source']/'result.json').exists():continue
        a,r,m=pilot.load_case(row);event=pilot.states_and_zooms(a,m,r)
        a.pop('AB_zoom_windows',None)
        a['separate_trajectory_labels']=True
        a['F_renderer']=draw_grid;a['F_complete_cells']=int((grid['count']==2).sum())
        a['F_semantics']=grid['scope']+' Original40 plus newly measured6 endpoints; discrete grid with untested cells masked.'
        if event:
            try:
                a['event_order']=layout.model_event_order(a,m,event)
                if f.early(a,m)['status']=='MEASURED':
                    a['E2_renderer']=layout.draw_e2
                    a['E2_semantics']='Model interictal contact rank versus early energy, with unchanged canonical Fig3C below; measured decreases retained.'
                    a['E2_display']=dict(top_left='Actual masked contact order of displayed finite event',
                        top_right='Signed CAR 1–150Hz log-power robust z',bottom='Unchanged original Fig3C',
                        native_field='Existing separate0.1ms diagnostic',primary_contact_endpoint_changed=False)
            except AssertionError as exc:
                a['E2_semantics']='Event order not estimable: '+str(exc)
        a['display_contract']=dict(primary_delivery='Original full Fig5 A/B/C/D/E1/E2/F layout',
            full_continuous_A_B_C=True,zoom_windows='Separate companion only',
            states=[s['label'] for s in a['display_snapshots']],Z_on=True,M_observed=True,
            effective_M_feedback=row['eta_m']>0,M_never_reset=True,second_high_not_numbered=True,
            display_end_after_second_confirmation_s=2,source_duration_s=r['end_s'],
            clinical_HFO_and_ictal_oscillation_not_established=True)
        dest=OUT/row['name'];f.render(a,r,m,dest/'figures',grid)
        meta=f.read(dest/'fig5_metadata.json')
        # Verify this is the same physical observation as the detailed candidate.
        detailed=pilot.OUT/'candidates'/row['name']/'fig5_metadata.json'
        old=f.read(detailed) if detailed.exists() else None
        if old:
            assert np.array_equal(old['readout']['scales'],meta['readout']['scales'])
            assert old['display_time_window_s']==meta['display_time_window_s']
            assert old['patient_reference_sha256']==meta['patient_reference_sha256']
            assert old['E2']['contact_robust_z']==meta['E2']['contact_robust_z']
        meta.update(source_duration_s=r['end_s'],source_run=str(row['source']),
            original_layout_producer=str(Path(__file__).resolve()),original_layout_producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            same_data_scales_and_patient_reference_as_detailed_candidate=old is not None,
            primary_full_fig5=True,paired_timing_diagnostic=str(pilot.OUT/'candidates'/row['name']/'figures/fig5.png'))
        f.write(dest/'fig5_metadata.json',meta)
        (dest/'figures/README.md').write_text('### fig5.png / .pdf\n'
            '原版Fig5完整布局：左侧连续电极读出、固定80神经元raster、同轴Z/M及Resting/Interictal HFO/Entry/Ictal/Recovery五个原生空间观察位置。第二次进入保留在时间线上，确认后2秒结束显示；本版不嵌入局部放大窗。\n'
            '右侧保留实际三维轨迹、模型传播与早期能量对照、原Fig3C患者图，以及二维M参数首次进入时间图。F合并原40个首次终点和本轮6个新终点，基线复用不重复计数；灰格未测，不能解释为未进入。\n'
            '**关注点**：原版整图为主交付，放大图和配对时间曲线为配套诊断；高率和外部补Z后恢复不证明临床振荡或自主终止，等待用户目视验收。\n')
        records.append(dict(name=row['name'],eta_m=row['eta_m'],seed=row['seed'],
            first_onset_s=m['entries'][0]['onset_s'] if m['entries'] else None,
            second_onset_s=m['entries'][1]['onset_s'] if len(m['entries'])>1 else None,
            display_end_s=m['duration_s'],figure=str(dest/'figures/fig5.png'),pdf=str(dest/'figures/fig5.pdf'),
            raw_data_consistency='PASS',agent_visual_review='PENDING',human_review='PENDING'))
        print(row['name'],m['duration_s'],flush=True)
        del a
    f.write(OUT/'delivery_manifest.json',dict(primary='Original complete Fig5',versions=records,
        original_first_endpoints=grid['original_first_endpoints'],new_first_endpoints=grid['additional_first_endpoints'],
        complete_measured_F_cells=int((grid['count']==2).sum()),new_simulations_added_for_this_layout=0,
        human_review='PENDING'))
    lines=['# 原版Fig5完整候选','','每个参数、每个种子均保留独立整图；未冻结任何模型。A/B局部放大及配对时间曲线只作为配套诊断。','',
           '| ηM | seed | 首次onset(s) | 再次onset(s) | 原版整图 |','|---|---|---|---|---|']
    for v in records:
        lines.append(f"| {v['eta_m']:g} | {v['seed']} | {v['first_onset_s']} | {v['second_onset_s']} | [PNG]({v['figure']}) · [PDF]({v['pdf']}) |")
    lines+=['','F为原20格M扫描加3个低M补测格，每格2种子，合计46个首次终点；0.005两条早回填基线不额外计数。没有把未测的低M×其他τM格补成结果。',
            'E2沿用原始数据，模型频带功率降低仍然保留，不作为患者早期能量增强已复现的证据。']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':render_original_package()
