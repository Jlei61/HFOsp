"""Classify the PD child and show its actual field/readout modulation."""
from plot_rate_periodic_composite import *


def main():
    q=read(PERIODIC_OUT/'PD_double_low_validation.json');assert q['status']=='VALIDATED_PD'
    low=read(PERIODIC_OUT/'PDchild_branch_N3072.json');hi=read(PERIODIC_OUT/'PDchild_branch_N4096.json')[-1]
    followup=PERIODIC_OUT/'stability_coverage/PDchild_a120.00000_N4096_positive_followup.json'
    assert followup.exists(), 'Resolve child orbit positivity before restoring its criticality figure'
    repaired=read(followup)
    assert repaired['status']=='UNSTABLE' and repaired['resolution']['minimum_rate_Hz']>=-1e-9
    assert repaired['resolution'].get('filter_state_check',{}).get('positive',False), 'Child constituent filters still unresolved'
    assert q.get('full_acceptance',False), 'Physical parent and critical mode must pass before classifying its child'
    # The repaired fixed-J orbit replaces only this endpoint; the departure
    # geometry and independent mesh comparison retain their original sources.
    hi=dict(hi,orbit=repaired['analyzed_orbit'],minimum_group_rate_hz=repaired['resolution']['minimum_rate_Hz'])
    choices=list((PERIODIC_OUT/'floquet').glob('PDchild_a120.00000_N4096_dt*.json'))
    f=Path(repaired['checks'][-1]['source']);floq=read(f)
    vals=np.array([complex(*v) for v in floq['multipliers']]);mu=vals[np.argmax(abs(vals))]
    assert mu.real>1.01 and abs(mu.imag)<1e-6 and max(floq['residuals'])<1e-6
    near=[v for v in low if 5<=v['amplitude_hz']<=40]
    assert all(v['J_shift']>0 for v in near)
    assert abs(hi['J_shift']-low[-1]['J_shift'])<1e-7
    s=RateField();w=s.geo['group_size']*s.E;w=w/w.sum();rr=[]
    for a in low:
        if a['amplitude_hz']==hi['amplitude_hz']:a=hi
        z=np.load(a['orbit']);r=z['r'];half=len(r)//2
        from audit_rate_filter_states import filter_state_minima
        assert filter_state_minima(s,r,float(z['T']))['positive'], 'Departure branch needs fine physical profiles'
        D=float(np.sqrt(np.mean(((r[half:]-r[:half])*1000)**2,axis=0)@w))
        rr.append(dict(amplitude_hz=a['amplitude_hz'],J_EE_core=a['J_EE_core'],J_minus_same_mesh_PD=a['J_shift'],
                       rms_half_period_difference_E_hz=D,T_ms=float(z['T']),orbit=a['orbit']))
    z=np.load(hi['orbit']);r=z['r'];T=float(z['T']);half=len(r)//2;diff=(r[half:]-r[:half])*1000
    region=np.array([s.regional_rates(v/1000) for v in diff]);t=np.arange(half)*T/len(r)
    from scipy.optimize import minimize_scalar
    regional=np.array([s.regional_rates(v) for v in r]);fullt=np.arange(len(r))*T/len(r);peaks={}
    for k in [0,1]:
        ids=find_peaks(regional[:,k],height=30,distance=len(r)//8)[0]
        spline=CubicSpline(np.r_[fullt,T],np.r_[regional[:,k],regional[0,k]],bc_type='periodic')
        ts=np.array([minimize_scalar(lambda x:-float(spline(x)),bounds=(max(0,fullt[i]-2*T/len(r)),min(T,fullt[i]+2*T/len(r))),method='bounded').x for i in ids])
        vv=spline(ts);peaks['AB'[k]]=dict(times_ms=ts,heights_hz=vv,
            interburst_intervals_ms=np.diff(np.r_[ts,ts[0]+T]))
        if len(ts)==4:
            peaks['AB'[k]].update(paired_peak_time_shift_ms=ts[2:]-ts[:2]-T/2,paired_peak_height_change_hz=vv[2:]-vv[:2])
    q.update(criticality='SUBCRITICAL_PD',child_branch_side='Parent-stable side: J > J_PD',
             child_unstable_multiplier=mu,child_floquet_source=str(f),child_mesh_J_shift_difference=abs(hi['J_shift']-low[-1]['J_shift']))
    write(PERIODIC_OUT/'PD_double_low_validation.json',q)
    proof=dict(status='SUBCRITICAL_PD',critical_J=q['J_EE_core'],critical_T_ms=q['T_ms'],
               full_physical_child_checks=True,
               parent_stable_side='J > J_PD locally',child_side='J > J_PD',child_mu=mu,
               child_at_checked_orbit='unstable',child_mesh_J_shift_difference=q['child_mesh_J_shift_difference'],
               small_amplitude_J_shift_over_squared_amplitude=[v['J_shift_over_amplitude_squared'] for v in near],
               rows=rr,maximum_core_A_B_surround_half_period_difference_hz=np.max(abs(region),axis=0),
               core_burst_peaks=peaks,
               minimum_group_rate_hz=hi['minimum_group_rate_hz'],
               numerical_note='Endpoint temporal refinement passes oversampled positivity and paired-step independent instability checks; rates are not clipped.',
               inference_limit='A doubled unstable periodic orbit is not evidence of a stable irregular attractor or a switch between propagation templates.')
    write(PERIODIC_OUT/'PD_child_classification.json',proof)
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axs=plt.subplots(2,2,figsize=(13,9.5),layout='constrained');ax=axs[0,0]
    x=np.array([a['J_minus_same_mesh_PD'] for a in rr])*1e6;y=[a['rms_half_period_difference_E_hz'] for a in rr]
    ax.plot(x,y,'o-',color=FAMILY['PDchild'],ms=5,lw=1.3,label='Computed 2T child')
    ax.plot([-4,0],[0,0],'k--',lw=1.5);ax.plot([0,x.max()*1.1],[0,0],'k-',lw=1.5,label='Parent: locally stable side')
    ax.plot(0,0,'v',mfc='white',mec='black',ms=8);ax.annotate('PD1',(0,0),xytext=(-29,10),textcoords='offset points')
    ax.plot(x[-1],y[-1],'x',color='black',ms=9,mew=1.6);ax.annotate(f'Unstable: μ={mu.real:.3f}',(x[-1],y[-1]),xytext=(-145,-18),textcoords='offset points')
    ax.set(xlabel=r'$(J_{\mathrm{EE,core}}-J_{\mathrm{PD1}})\times10^6$',ylabel='RMS difference between halves (Hz / E cell)',title='a   Subcritical period doubling')
    ax.legend(frameon=False,fontsize=9,loc='upper left');style(ax)
    ax=axs[0,1]
    for k in [0,1]:ax.plot(t,region[:,k],color=COL[k],lw=1.1,label=f'Core {"AB"[k]}')
    ax.axhline(0,color='black',lw=.6);ax.legend(frameon=False)
    ax.set(xlim=(0,T/2),xlabel='Time within the first half (ms)',ylabel='Second − first half rate (Hz / cell)',title=f'b   Actual 2T orbit: full period {T:.1f} ms');style(ax)
    cell=s.geo['group_cell'];sz=s.geo['group_size'];e=s.E
    power=np.mean(diff*diff,axis=0);num=np.bincount(cell[e],weights=sz[e]*power[e],minlength=400);den=np.bincount(cell[e],weights=sz[e],minlength=400)
    field=np.sqrt(num/np.maximum(den,1));ax=axs[1,0]
    im=ax.imshow(field.reshape(20,20),origin='lower',extent=(0,20,0,20),cmap='magma',vmin=0)
    for k,center in enumerate(s.geo['centers_mm']):
        ax.add_patch(plt.Circle(center,1.5,fill=False,color='white',lw=1));ax.text(*center,'AB'[k],color='white',ha='center',va='center')
    geo=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz');xy=geo['contact_xy']
    ax.scatter(xy[:,0],xy[:,1],s=18,facecolors='none',edgecolors='cyan',linewidths=.7)
    ax.set(xlabel='x (mm)',ylabel='y (mm)',title='c   Spatial modulation between halves')
    fig.colorbar(im,ax=ax,label='RMS rate difference (Hz / E cell)',shrink=.8)
    contact=diff@s.geo['contact_rate_weights'];order=contact_indices(geo['contact_names'].tolist());ax=axs[1,1]
    limit=float(np.max(abs(contact)));im=ax.imshow(contact[:,order].T,aspect='auto',origin='upper',extent=(0,T/2,14.5,-.5),cmap='RdBu_r',vmin=-limit,vmax=limit)
    ax.set(yticks=np.arange(15),yticklabels=CONTACT_ORDER,xlabel='Time within the first half (ms)',title='d   Corresponding contact readout change')
    ax.tick_params(axis='y',labelsize=9);ax.axhline(3.5,color='black',lw=.7)
    for tick,n in zip(ax.get_yticklabels(),CONTACT_ORDER):tick.set_color(SHAFT_COLORS[n[:3]])
    fig.colorbar(im,ax=ax,label='Second − first half rate (Hz / cell)',shrink=.8)
    save(fig,'period_doubled_branch_and_spatial_change')
    with (F/'README.md').open('a') as f:
        f.write('\n### period_doubled_branch_and_spatial_change\n展示 PD1 出发的两倍周期子分支及其实际空间场、触点读出在两个半周期之间的差异。子分支伸向母轨道稳定的一侧，所检验子轨道的 Floquet 乘子大于 +1，支持亚临界倍周期；这不是稳定吸引子的活动示例。**关注点**：空间和读出图均来自同一不稳定 rate 周期解，逐周期调制不等价于 TA/TB 模板切换或稳定 irregular 状态。\n')
    print('PD CHILD CLASSIFICATION',proof,flush=True)


if __name__=='__main__':main()
