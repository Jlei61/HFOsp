"""Conditional Z/M bifurcation and same-model spatial activity, no SNN trace."""
from periodic_zm import *
from periodic_stability_zm import refresh
from figures_and_report import spatial,save
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import uniform_filter1d


def orbit_field(s,path):
    z=np.load(path);r=resample(z['r'],4096,axis=0)*1000;T=float(z['T'])
    whole=r[:,s.E]@s.mean_weights;n=max(1,round(50/T*len(r)))
    sm=uniform_filter1d(whole,n,mode='wrap');idx=int(sm.argmax())
    ids=(np.arange(n)+idx-n//2)%len(r);v=r[ids].mean(0)
    cells=s.geo['group_cell'];den=np.bincount(cells[s.E],weights=s.sizes[s.E],minlength=s.grid**2)
    field=np.bincount(cells[s.E],weights=v[s.E]*s.sizes[s.E],minlength=s.grid**2)/den
    return field,dict(source=str(path),phase_fraction=idx/len(r),window_ms=50.,D=float(z['D']))


def main():
    s=ZMSpatialRate();verified=refresh();root=PERIODIC_OUT/'orbits'
    low=[read(p) for p in root.glob('physicalLow*.json')]+[read(root/'burst_D0.180000000_N512.json'),read(root/'burst_D0.181000000_N512.json')]
    low=sorted(low,key=lambda q:q['D']);arc=[read(p) for p in sorted(root.glob('burstUp*.json'))]
    fold=read(root/'LPC_burst_N1024.json');split=int(np.argmax([q['D'] for q in arc]))
    before=low+arc[:split+1]+[fold];after=[fold,read(root/'LPC_burst_after_N1024.json')]+arc[split+1:]
    fig=plt.figure(figsize=(13,8.4));gs=fig.add_gridspec(3,2,width_ratios=[2.7,1],hspace=.45,wspace=.37)
    ax=fig.add_subplot(gs[:,0]);orange='#ca8325';green='#208975'
    eqnames=['D_arclength_lower','D_arclength_upper_focused','D_arclength_upper_onset_range',
             'D_arclength_upper_onset_range_v2','D_arclength_upper_onset_range_v3','D_arclength_upper_onset_range_v4',
             'D_gap_lower_guarded','D_gap_middle_down','D_gap_middle_up','D_gap_middle_to_low']
    for name in eqnames:
        rows=[q for q in read(OLD/'g20'/name/'result.json')['rows'] if q.get('converged',True)]
        ax.plot([q['D'] for q in rows],[q['global_E_hz'] for q in rows],'.',color='#a5a5a5',ms=1.4,zorder=1)
        check=PERIODIC_OUT/f'equilibrium_classification/{name}.json'
        if check.exists():
            classified=read(check)['rows']
            for left,right in zip(classified[:-1],classified[1:]):
                if left['status']==right['status']=='UNSTABLE':
                    section=[q for q in rows if left['index']<=q['index']<=right['index']]
                    ax.plot([q['D'] for q in section],[q['global_E_hz'] for q in section],'--',color='#333333',lw=.85,zorder=2)
    tail=[q for q in read(OLD/'g20/conditional_dynamic_M/result.json')['rows'] if q['converged'] and q['direction']=='decreasing' and q['D']>=.4]
    ax.plot([q['D'] for q in tail],[q['global_E_hz'] for q in tail],color='#333333',lw=1.25)
    for p in sorted((PERIODIC_OUT/'equilibrium_counts').glob('*.json')):
        q=read(p)
        if q['status']=='RESOLVED' and q['unstable_roots']==0:ax.plot(q['D'],q['global_E_hz'],'o',ms=3.5,color='#333333')
    eqcrit=read(DEST/'critical_spectra/result.json')['rows']
    for q in eqcrit:
        if q['label'].startswith('SN'):
            ax.plot(q['D'],q['global_E_hz'],'D',ms=4,mfc='white',mec='#494949',mew=.9)
            if q['label']=='SN7':ax.annotate('SN7',(q['D'],q['global_E_hz']),xytext=(13,15),textcoords='offset points',fontsize=10,arrowprops=dict(arrowstyle='-',lw=.7))
            if q['label'] in ['SN1','SN2']:
                ax.annotate(q['label'],(q['D'],q['global_E_hz']),xytext=(18,-9 if q['label']=='SN1' else 7),textcoords='offset points',fontsize=9,arrowprops=dict(arrowstyle='-',lw=.6))
            if q['label']=='SN6':ax.annotate('SN3–6',(q['D'],q['global_E_hz']),xytext=(12,-20),textcoords='offset points',fontsize=9,arrowprops=dict(arrowstyle='-',lw=.6))
    hopf=read(PERIODIC_OUT/'hopf_high.json');nf=read(PERIODIC_OUT/'normal_form_high.json')
    ax.plot(hopf['D'],hopf['global_E_hz'],'^',color='#8355a4',ms=7,zorder=8)
    ax.annotate('H',(hopf['D'],hopf['global_E_hz']),xytext=(-24,-25),textcoords='offset points',fontsize=11,color='#8355a4',arrowprops=dict(arrowstyle='-',lw=.7,color='#8355a4'))
    for rows,style in [(before,'-'),(after,'--')]:
        ax.plot([q['D'] for q in rows],[q['global_mean_hz'] for q in rows],style,color=orange,lw=1.7,zorder=3)
    high=sorted([read(p) for p in root.glob('highHopf_A*_N64.json')],key=lambda q:q['D'])
    if any(Path(q['source']).stem=='highHopf_A16_N64' and q['status']=='STABLE' for q in verified):
        ax.plot([q['D'] for q in high]+[hopf['D']],[q['global_mean_hz'] for q in high]+[hopf['global_E_hz']],color=orange,lw=1.4,zorder=4)
    # Max/min symbols are reserved for actual computed Floquet classifications.
    selected={}
    for q in verified:
        if q['status'] not in ['STABLE','UNSTABLE']:continue
        key=Path(q['source']).stem
        if key not in selected or q['dt_ms']<selected[key]['dt_ms']:selected[key]=q
    if (PERIODIC_OUT/'spectral_floquet/after_turn_N512.json').exists():
        q=read(PERIODIC_OUT/'spectral_floquet/after_turn_N512.json')
        if q['residual']<2e-9 and q['lambda_per_ms'][0]>0:selected[Path(q['orbit']).stem]=dict(status='UNSTABLE',source=q['orbit'],dt_ms=0.)
    for key,q in selected.items():
        row=read(root/f'{key}.json');fill=green if q['status']=='STABLE' else 'white'
        ax.plot([row['D']]*2,[row['global_min_hz'],row['global_max_hz']],'s',mfc=fill,mec=green,mew=1.,ms=4.5,zorder=4)
    ax.plot(fold['D'],fold['global_mean_hz'],'*',color='#ad3748',ms=13,zorder=7)
    ax.annotate('2  LPC',(fold['D'],fold['global_mean_hz']),xytext=(22,-20),textcoords='offset points',color='#ad3748',fontsize=11,arrowprops=dict(arrowstyle='-',color='#ad3748',lw=.8))
    q=read(root/'burst_D0.180000000_N512.json')
    ax.annotate('1',(q['D'],q['global_mean_hz']),xytext=(-18,-32),textcoords='offset points',color=orange,fontsize=11,arrowprops=dict(arrowstyle='-',color=orange,lw=.7))
    cross=read(PERIODIC_OUT/'crossing_runs/D0.188700000/result.json')
    post=next(q for q in read(PERIODIC_OUT/'crossing_summary.json')['rows'] if q['D']==cross['D'])
    ax.plot(cross['D'],post['last_2000ms_mean_hz'],'d',color='#7654a0',ms=6,zorder=5)
    ax.annotate('3',(cross['D'],post['last_2000ms_mean_hz']),xytext=(13,0),textcoords='offset points',color='#7654a0')
    for path in sorted((DEST/'runs').glob('main_D*/result.json')):
        q=read(path)
        if q['D_initial']>=.2:ax.plot(q['D_initial'],q['dynamics'][0]['mean_hz'],'d',color='#7654a0',ms=4,zorder=4)
    ax.set(xlim=(0,1),ylim=(.045,550),yscale='log',xlabel=r'$D=1-\langle Z_E\rangle$',ylabel='Global E rate (Hz / neuron)')
    ax.set_yticks([.05,.1,1,10,100,500]);ax.set_yticklabels(['0.05','0.1','1','10','100','500'])
    ax.text(-.11,1.015,'A',transform=ax.transAxes,fontsize=18,fontweight='bold')
    ins=ax.inset_axes([.49,.10,.47,.25])
    for rows,style in [(before,'-'),(after,'--')]:ins.plot([q['D'] for q in rows],[q['global_mean_hz'] for q in rows],style,color=orange,lw=1.3)
    ins.plot(fold['D'],fold['global_mean_hz'],'*',color='#ad3748',ms=10)
    ins.set(xlim=(.188575,.18865),ylim=(16.76,17.005),xticks=[.18858,.18864],yticks=[16.8,16.9,17.0],xlabel='D')
    ins.set_ylabel('Period mean (Hz)',fontsize=9);ins.tick_params(labelsize=9);ins.xaxis.get_offset_text().set_visible(False)
    ins.ticklabel_format(axis='x',style='plain',useOffset=False)
    handles=[Line2D([],[],color='#333333',label='Stable equilibrium'),
        Line2D([],[],color='#333333',ls='--',label='Unstable equilibrium'),
        Line2D([],[],marker='.',ls='none',color='#a5a5a5',label='Equilibrium: unclassified'),
        Line2D([],[],color=orange,label='Period mean: stable branch'),
        Line2D([],[],color=orange,ls='--',label='Period mean: unstable branch'),
        Line2D([],[],marker='s',ls='none',color=green,label='Periodic extrema: stable'),
        Line2D([],[],marker='s',ls='none',mfc='white',mec=green,label='Periodic extrema: unstable'),
        Line2D([],[],marker='D',ls='none',mfc='white',mec='#494949',label='Equilibrium fold (SN)'),
        Line2D([],[],marker='*',ls='none',color='#ad3748',ms=10,label='Fold of cycles (LPC)'),
        Line2D([],[],marker='^',ls='none',color='#8355a4',label='Supercritical Hopf (H)'),
        Line2D([],[],marker='d',ls='none',color='#7654a0',label='Post-fold finite-time mean')]
    ax.legend(handles=handles,loc='center right',bbox_to_anchor=(.99,.63),frameon=False,fontsize=9,handlelength=2.7,labelspacing=.5)
    fields=[]
    for j,path in enumerate([root/'burst_D0.180000000_N512.npz',root/'LPC_burst_N1024.npz',None]):
        if path is not None:field,meta=orbit_field(s,path)
        else:
            path=PERIODIC_OUT/'crossing_runs/D0.188700000/trajectory.npz';z=np.load(path)
            idx=len(z['time_ms'])-500;field=z['field_E_hz'][idx-25:idx+25].mean(0)
            meta=dict(source=str(path),window_ms=[int(z['time_ms'][idx-25]),int(z['time_ms'][idx+24])],D=.1887)
        bx=fig.add_subplot(gs[j,1]);im=spatial(bx,field,s)
        bx.text(-.22,1.09,chr(66+j),transform=bx.transAxes,fontsize=17,fontweight='bold')
        bx.text(.5,1.04,f'{j+1}    D = {meta["D"]:.6f}',transform=bx.transAxes,ha='center',fontsize=11)
        fields.append(meta)
    fig.subplots_adjust(right=.90);cax=fig.add_axes([.925,.18,.014,.62]);fig.colorbar(im,cax=cax,label='E rate (Hz / neuron)')
    metadata=dict(model='Frozen 935-group, 1 mm spatial rate DDE',J_EE_core=1,Z='held spatial path per D',M='dynamic',
        periodic_bifurcation=read(PERIODIC_OUT/'LPC_burst_N1024.json'),spatial_panels=fields,
        stability_checks=verified,branch_stability_style='Family stability is sampled; max/min squares are drawn only at classified Floquet samples. Unsampled narrow stability windows are not excluded.',
        equilibrium_style='Gray branches have not all been classified for the synchronized dynamic closure; dark high-rate line uses representative argument-principle checks.',
        high_rate_hopf=hopf,high_rate_hopf_normal_form=nf,
        temporal_mesh_refinement='512 and 1024 Fourier points; this is not spatial-grid refinement',
        SNN_equivalence='NOT_VALIDATED',human_visual_acceptance='PENDING')
    save(fig,'fig_zm_periodic_bifurcation_spatial',metadata)
    # The spatial tangent is separate from an activity snapshot.
    z=np.load(PERIODIC_OUT/'fold_spatial_mode.npz');field=z['field_energy'];field=field/field.sum()
    fig,ax=plt.subplots(figsize=(5.4,4.6));im=ax.imshow(field.reshape(20,20),origin='lower',extent=[0,20,0,20],cmap='viridis')
    from matplotlib.patches import Circle
    for label,xy in zip('AB',s.geo['centers_mm']):ax.add_patch(Circle(xy,1.5,fill=False,ec='#26cbd0',lw=1));ax.text(xy[0],xy[1]+1.7,label,ha='center',color='#1c9297')
    ax.set(xlabel='x (mm)',ylabel='y (mm)',xticks=[0,10,20],yticks=[0,10,20]);fig.colorbar(im,ax=ax,label='Fraction of E tangent energy')
    save(fig,'fig_zm_LPC_spatial_tangent',read(PERIODIC_OUT/'fold_spatial_mode.json'))


if __name__=='__main__':main()
