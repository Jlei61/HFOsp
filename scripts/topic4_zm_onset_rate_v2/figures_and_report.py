"""Read-only scientific aggregation and figures for the synchronized stage."""
from audit_and_spectrum import states
from model_zm import *
from run_conditions import summary
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from scipy.ndimage import uniform_filter1d

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':12,
 'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})


def save(fig,name,metadata):
    dest=DEST/'figures';dest.mkdir(exist_ok=True)
    for ext in ['png','pdf','svg']:fig.savefig(dest/f'{name}.{ext}',dpi=190,bbox_inches='tight')
    plt.close(fig);write(dest/f'{name}_metadata.json',metadata)


def spatial(ax,field,s):
    im=ax.imshow(field.reshape(s.grid,s.grid),origin='lower',extent=[0,20,0,20],vmin=0,vmax=500,cmap='magma')
    for name,xy in zip(['A','B'],s.geo['centers_mm']):
        ax.add_patch(Circle(xy,1.5,fill=False,ec='#26cbd0',lw=1.2));ax.text(xy[0],xy[1]+1.7,name,color='#199297',ha='center',fontsize=10)
    ax.set(xticks=[0,10,20],yticks=[0,10,20],xlabel='x (mm)',ylabel='y (mm)')
    return im


def entry(a):
    rate=a['global_E_hz'];ids=np.flatnonzero(np.convolve((rate>=200).astype(int),np.ones(200,dtype=int),mode='valid')==200)
    return int(ids[0]) if len(ids) else None


def trajectory(s):
    file=DEST/'runs/main_dynamic/trajectory.npz';a=np.load(file);k=entry(a);r=a['global_E_hz'];t=a['time_ms'];reg=s.geo['group_region']
    irest=800+int(np.argmin(r[800:1400]));ibur=3500+int(np.argmax(uniform_filter1d(r[3500:4500],50)))
    selected=[irest,ibur,k-300,k+25]
    fig=plt.figure(figsize=(12,6.8));gs=fig.add_gridspec(2,4,height_ratios=[1,1.25],hspace=.5,wspace=.4)
    ax=fig.add_subplot(gs[0,:]);bx=ax.twinx();colors=['#b45078','#2e98aa','#8c78a8']
    for j in range(3):
        mask=s.E&(reg==j);z=np.average(a['Z'][:,mask],axis=1,weights=s.sizes[mask]);m=np.average(a['M_current'][:,mask],axis=1,weights=s.sizes[mask])
        ax.plot(t/1000,z,color=colors[j],lw=1.3,label=['Core A','Core B','Surround'][j]);bx.plot(t/1000,m,'--',color=colors[j],lw=1)
    ax.set(xlabel='Time (s)',ylabel='Inhibitory resource Z',xlim=(0,12.5),ylim=(0,1.03));bx.set_ylabel(r'$\eta_M M$ (mV)');bx.set_ylim(0,.25)
    regional_legend=ax.legend(loc='lower left',frameon=False,ncol=3);ax.add_artist(regional_legend)
    ax.text(-.06,1.04,'A',transform=ax.transAxes,fontsize=19,fontweight='bold')
    ax.legend(handles=[Line2D([],[],color='k',label='Z'),Line2D([],[],color='k',ls='--',label=r'$\eta_M M$')],loc='upper right',frameon=False)
    snapshots=[]
    for j,idx in enumerate(selected):
        start=max(0,idx-25);stop=idx+25;field=a['field_E_hz'][start:stop].mean(0)
        ax.axvline(t[idx]/1000,color=['#6a727a','#298bc5','#d88925','#ca405d'][j],ls=':',lw=1)
        ax.text(t[idx]/1000,.98,str(j+1),ha='center',va='top')
        cx=fig.add_subplot(gs[1,j]);im=spatial(cx,field,s)
        cx.set_title(f'{j+1}    {t[idx]/1000:.3f} s\nD = {a["D"][idx]:.3f}',fontsize=11)
        snapshots.append(dict(index=idx,time_ms=int(t[idx]),D=float(a['D'][idx]),window_ms=[start+1,stop]))
    fig.subplots_adjust(right=.92);cb=fig.colorbar(im,cax=fig.add_axes([.94,.12,.014,.31]));cb.set_label('E rate (Hz / neuron)')
    save(fig,'fig_zm_rate_dynamic_spatial',dict(source=str(file),snapshots=snapshots,Z='dynamic',M='dynamic',
        figure_is='Same rate DDE slow-variable trajectory and 50-ms spatial activity; not SNN',human_visual_acceptance='PENDING'))
    return a,k


def condition_plot(s):
    specs=read(DEST/'critical_spectra/result.json')['rows'];fig=plt.figure(figsize=(13.5,7.7))
    gs=fig.add_gridspec(2,3,width_ratios=[2.2,1,1],wspace=.4,hspace=.45);ax=fig.add_subplot(gs[:,0])
    branches=['D_arclength_lower','D_arclength_upper_focused','D_arclength_upper_onset_range',
              'D_arclength_upper_onset_range_v2','D_arclength_upper_onset_range_v3','D_arclength_upper_onset_range_v4',
              'D_gap_lower_guarded','D_gap_middle_down','D_gap_middle_up','D_gap_middle_to_low']
    for name in branches:
        rows=read(OLD/'g20'/name/'result.json')['rows'];rows=[q for q in rows if q.get('converged',True)]
        ax.plot([q['D'] for q in rows],[q['global_E_hz'] for q in rows],'o',ms=1.5,mfc='white',mec='#555555',mew=.4)
    coarse=read(OLD/'g20/conditional_dynamic_M/result.json')['rows']
    rows=[q for q in coarse if q['converged'] and q['direction']=='decreasing' and q['D']>.4]
    ax.plot([q['D'] for q in rows],[q['global_E_hz'] for q in rows],'o',ms=2,mfc='white',mec='#555555',mew=.5)
    for q in specs:
        if q['label'].startswith('SN'):
            ax.plot(q['D'],q['global_E_hz'],'*',ms=9,color='#bb3b42')
            if q['label'] in ['SN1','SN2','SN7']:
                ax.annotate(q['label'],(q['D'],q['global_E_hz']),xytext=(12,12 if q['label']!='SN2' else 23),textcoords='offset points',fontsize=10,
                            arrowprops=dict(arrowstyle='-',lw=.6,color='#bb3b42'))
        if q['models']['rate_dde']['stability']=='UNSTABLE':ax.plot(q['D'],q['global_E_hz'],'x',ms=5,color='k')
    ins=ax.inset_axes([.52,.54,.43,.23])
    upper=read(OLD/'g20/D_arclength_upper_focused/result.json')['rows']
    ins.plot([q['D'] for q in upper],[q['global_E_hz'] for q in upper],'o',ms=1.5,mfc='white',mec='#555555',mew=.4)
    for j,q in enumerate([q for q in specs if q['label'] in ['SN3','SN4','SN5','SN6']]):
        ins.plot(q['D'],q['global_E_hz'],'*',color='#bb3b42',ms=7)
        ins.annotate(q['label'],(q['D'],q['global_E_hz']),xytext=[(-10,-15),(-10,12),(12,-12),(0,12)][j],textcoords='offset points',fontsize=8,
                     arrowprops=dict(arrowstyle='-',lw=.5))
    ins.set(xlim=(.365,.4),ylim=(410,420),xticks=[.37,.39],yticks=[410,415,420]);ins.tick_params(labelsize=8)
    conditions=[]
    for p in sorted((DEST/'runs').glob('main_D*/result.json')):
        q=read(p);conditions.append(q);d=q['dynamics'][0]
        ax.plot(q['D_initial'],d['mean_hz'],'D',color='#cb8b2c',ms=5)
    ax.set(xlim=(0,1),ylim=(.1,550),yscale='log',xlabel=r'$D=1-\langle Z_E\rangle$',ylabel='Global E rate (Hz / neuron)')
    ax.set_yticks([.1,1,10,100,500]);ax.set_yticklabels(['0.1','1','10','100','500'])
    ax.text(-.13,1.02,'A',transform=ax.transAxes,fontsize=19,fontweight='bold')
    handles=[Line2D([],[],marker='o',mfc='white',mec='#555555',ls='none',ms=4,label='Equilibrium: stability pending'),
             Line2D([],[],marker='x',color='k',ls='none',ms=5,label='Verified unstable point'),
             Line2D([],[],marker='*',color='#bb3b42',ls='none',ms=9,label='Equilibrium fold (SN)'),
             Line2D([],[],marker='D',color='#cb8b2c',ls='none',ms=5,label='Finite-time mean (2–8 s)')]
    ax.legend(handles=handles,frameon=False,loc='lower right',fontsize=9)
    selected=[]
    for j,D in enumerate([.18,.20,.22,.26]):
        file=DEST/f'runs/main_D{D:.3f}/trajectory.npz';a=np.load(file)
        # Last second: center on the largest 50-ms global activity window.
        g=uniform_filter1d(a['global_E_hz'],50);idx=len(g)-1000+int(np.argmax(g[-1000:-25]));start=idx-25;stop=idx+25
        bx=fig.add_subplot(gs[j//2,1+j%2]);im=spatial(bx,a['field_E_hz'][start:stop].mean(0),s)
        bx.set_title(f'{j+1}    D = {D:.2f}',fontsize=12)
        bx.text(-.25,1.08,chr(66+j),transform=bx.transAxes,fontsize=18,fontweight='bold')
        q=next(q for q in conditions if abs(q['D_initial']-D)<1e-9)
        ax.annotate(str(j+1),(D,q['dynamics'][0]['mean_hz']),xytext=(12,-16 if j%2 else 12),textcoords='offset points',color='#9d671b',fontsize=10)
        selected.append(dict(D=D,source=str(file),window_ms=[start+1,stop]))
    fig.subplots_adjust(right=.92);cb=fig.colorbar(im,cax=fig.add_axes([.947,.2,.012,.55]));cb.set_label('E rate (Hz / neuron)')
    save(fig,'fig_zm_rate_conditional_states',dict(scope='Conditional equilibrium branches plus finite-time rate-DDE observations; not completed periodic continuation',
        equilibrium_sources=str(OLD),dynamic_sources=str(DEST/'runs'),J_EE_core=1,Z='fixed per condition',M='dynamic',snapshots=selected,
        stability='Only marked samples have recalculated rate-DDE roots; negative selected roots do not establish stability',
        periodic_branches='NOT_COMPUTED',human_visual_acceptance='PENDING'))


def response_plot():
    rows=read(DEST/'local_response/result.json')['rows'];half=read(DEST/'local_response_half/result.json')['rows']
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.5),layout='constrained');colors={'SN1':'#2d8f9e','SN7':'#ce8526','upper_D0228':'#a15087'}
    for pop,marker in [('E','o'),('I','s')]:
        for state in colors:
            q=next(q for q in rows if q['population']==pop and q['state']==state and q['channel']=='mean' and q['frequency_hz']==0)
            axes[0].scatter(q['measured_rate_hz'],q['predicted_rate_hz'],color=colors[state],marker=marker,s=50)
            q=next(q for q in rows if q['population']==pop and q['state']==state and q['channel']=='variance_E' and q['frequency_hz']==0)
            axes[1].errorbar(q['measured'][0],q['predicted_rate_dde'][0],xerr=1.96*q['complex_sem'],fmt=marker,color=colors[state],ms=6)
    for q in half:
        if q['channel']=='variance_E' and q['frequency_hz']==0:
            axes[1].errorbar(q['measured'][0],q['predicted_rate_dde'][0],xerr=1.96*q['complex_sem'],fmt='o' if q['population']=='E' else 's',
                color=colors['SN7'],mfc='white',ms=8)
    axes[0].plot([0,300],[0,300],'--',color='k',lw=.8)
    axes[0].set(xlabel='Local LIF assay rate (Hz)',ylabel='Rate-model prediction (Hz)',xlim=(0,300),ylim=(0,300))
    axes[1].plot([-.1,.75],[-.1,.75],'--',color='k',lw=.8);axes[1].axhline(0,color='k',lw=.5);axes[1].axvline(0,color='k',lw=.5)
    axes[1].set(xlabel=r'Measured $\partial r/\partial v_E$ (Hz / mV$^2$)',ylabel=r'Predicted $\partial r/\partial v_E$ (Hz / mV$^2$)',xlim=(-.1,.75),ylim=(-.1,.75))
    axes[0].legend(handles=[Line2D([],[],marker='o',ls='none',color=c,label='D = 0.228' if k=='upper_D0228' else k) for k,c in colors.items()],frameon=False,loc='upper left')
    axes[1].legend(handles=[Line2D([],[],marker='o',ls='none',color='k',label='E'),Line2D([],[],marker='s',ls='none',color='k',label='I'),
        Line2D([],[],marker='o',mfc='white',ls='none',color='k',label='Half step and amplitude')],frameon=False,loc='lower right',fontsize=9)
    for j,ax in enumerate(axes):ax.text(-.14,1.03,chr(65+j),transform=ax.transAxes,fontsize=18,fontweight='bold')
    save(fig,'fig_zm_rate_local_response_check',dict(source=str(DEST/'local_response/result.json'),refinement=str(DEST/'local_response_half/result.json'),
        scope='Uncoupled colored-LIF local response; error bars 1.96 Monte Carlo SEM, no whole-network equivalence assertion'))


def spatial_control(s):
    folders=['matched_path_D0.200','actual_entry_field_D0.200'];fig,axes=plt.subplots(2,2,figsize=(8,7.4))
    cells=s.geo['group_cell'][s.E];n=s.sizes[s.E];den=np.bincount(cells,weights=n,minlength=s.grid**2)
    metadata=[]
    for j,name in enumerate(folders):
        f=DEST/'runs'/name;a=np.load(f/'trajectory.npz');z=a['Z'][0,s.E]
        field=np.bincount(cells,weights=n*z,minlength=s.grid**2)/den
        im0=axes[0,j].imshow(field.reshape(s.grid,s.grid),origin='lower',extent=[0,20,0,20],vmin=0,vmax=1,cmap='viridis')
        axes[0,j].set(xlabel='x (mm)',ylabel='y (mm)',xticks=[0,10,20],yticks=[0,10,20])
        im1=spatial(axes[1,j],a['field_E_hz'][-1000:].mean(0),s)
        metadata.append(dict(source=str(f/'trajectory.npz'),mean_D=float(a['D'][0]),
            mean_E_hz_2_8s=read(f/'result.json')['dynamics'][0]['mean_hz'],activity_window_ms=[7001,8000]))
    for i,ax in enumerate(axes.ravel()):ax.text(-.18,1.03,chr(65+i),transform=ax.transAxes,fontsize=18,fontweight='bold')
    fig.subplots_adjust(left=.09,right=.85,bottom=.08,top=.96,hspace=.38,wspace=.35)
    cb=fig.colorbar(im0,cax=fig.add_axes([.88,.58,.02,.31]));cb.set_label('Inhibitory resource Z')
    cb=fig.colorbar(im1,cax=fig.add_axes([.88,.10,.02,.31]));cb.set_label('E rate (Hz / neuron)')
    save(fig,'fig_zm_equal_D_spatial_control',dict(columns=['Prescribed native 9.420s power path','Spatial Z field from autonomous rate trajectory at entry'],
        panels='Top Z; bottom mean spatial activity during 7–8s',rows=metadata,all_other_initial_states='Zero',
        stochastic_drive='None; same private input moments',M='dynamic',Z='fixed',scope='Within-model finite-time spatial Z effect, not native validation'))
    return metadata


def aggregate(s,a,k):
    half=np.load(DEST/'runs/half_step_dynamic/trajectory.npz');kh=entry(half);D=float(a['D'][k]);s.set_D(D)
    rms=float(np.sqrt(np.average((a['Z'][k,s.E]-s.Z[s.E])**2,weights=s.sizes[s.E])))
    metrics=dict(entry_definition='Global E rate >=200 Hz continuously for 200 ms; operational threshold, not bifurcation classification',
        full_dynamic=dict(entry_time_ms=int(a['time_ms'][k]),entry_D=D,final_D=float(a['D'][-1]),
            spatial_Z_RMS_from_prescribed_path_at_same_D=rms,entry_mean_M_current=float(np.average(a['M_current'][k,s.E],weights=s.sizes[s.E])),
            entry_instantaneous_equilibrium_M_current=float(.0005*a['global_E_hz'][k])),
        step_halving=dict(entry_time_ms=int(half['time_ms'][kh]),entry_D=float(half['D'][kh]),
            final_D_difference=float(half['D'][-1]-a['D'][-1]),
            global_waveform_relative_RMS=float(np.linalg.norm(a['global_E_hz']-half['global_E_hz'])/np.linalg.norm(a['global_E_hz']))),
        full_trajectory_early=summary(np.c_[a['global_E_hz'],a['regional_rates_hz']][500:3000]),
        conditions=[dict(folder=p.parent.name,**read(p)) for p in sorted((DEST/'runs').glob('*/result.json'))],
        scientific_acceptance=dict(numerical_equations='PASS',same_nonlinear_rate_for_trajectory_and_spectrum=True,
            local_depleted_workpoint_response='FAIL: variance gain sign and magnitude mismatch persists after step/amplitude reduction',
            SNN_propagation_equivalence='NOT_VALIDATED',onset_bifurcation_type='NOT_ESTABLISHED',periodic_Floquet='NOT_COMPUTED'))
    write(DEST/'analysis_summary.json',metrics)
    return metrics


if __name__=='__main__':
    s=ZMSpatialRate();a,k=trajectory(s);condition_plot(s);response_plot();spatial_control(s);q=aggregate(s,a,k)
    print(q['full_dynamic']);print(q['step_halving'])
