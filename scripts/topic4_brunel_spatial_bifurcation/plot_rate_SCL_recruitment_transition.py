"""Same-branch physical waveforms and a frozen-observer SCL7 detection boundary."""
from plot_rate_sameJ_burst_pair import *
from check_rate_live_Bleading_observer import physical_pass
from compare_rate_torus_periodic_targets import distances


def main():
    folder=DATA/'Bleading_extension/live_CPU_observer'
    saved=[read(folder/f'point_{i:04d}.json') for i in [31,32]]
    assert all(physical_pass(q['check']) for q in saved)
    model=RateField();contract=read(OLD/'observer_firing.json');names=contract['contact_names']
    channel=names.index('SCL7');profiles=[];envelopes=[];regions=[]
    for q in saved:
        z=np.load(q['check']['path']);r=z['r'];T=float(z['T']);profiles.append((r,T))
        rates=np.array([model.regional_rates(v) for v in r])[:,:3]
        shift=int(np.argmax(rates[:,1])-round(60/T*len(r)))
        regions.append(np.roll(rates,-shift,axis=0))
        contacts=r@model.geo['contact_rate_weights']*1000
        spline=CubicSpline(np.arange(len(r)+1)*T/len(r),np.r_[contacts,contacts[:1]],bc_type='periodic')
        first=max(4,int(np.ceil(contract['burnin_ms']/T))+2)
        window=[first*T,(first+8)*T];stop=int((first+12)*T)//2*2
        traces=[]
        for offset,stored in zip([0.,.5,1.,1.5],q['readout']['records']):
            times=(np.arange(stop*4)+.5)/4+offset
            sampled=spline(times%T).reshape(stop,4,15).mean(1)
            env=smooth2(sampled.reshape(-1,2,15).sum(1)/1000)
            observed=observer.observe(env.T,2.,contract)
            frames=np.arange(len(env))*2+1+offset
            interior=(frames>=window[0])&(frames<window[1])
            threshold=np.asarray(observed['threshold'])
            ratio=env[interior,channel]/threshold[channel]
            assert abs(ratio.max()-stored['contact_peak_to_threshold'][channel])<1e-10
            # Common reference per orbit: the offset-zero filtered SCL7 peak.
            if offset==0:peak_phase=float(frames[interior][np.argmax(ratio)]%T)
            relative=(frames[interior]%T-peak_phase+T/2)%T-T/2
            visible=abs(relative)<=20
            traces.append(dict(bin_origin_ms=offset,relative_ms=relative[visible],
                               threshold_ratio=ratio[visible]))
        envelopes.append(traces)
    weights=model.geo['group_size']/model.geo['group_size'].sum()
    x,y=[r*1000 for r,T in profiles]
    distance,phase=distances(x[:,None,:],y,weights)
    scale=np.sqrt(np.mean(np.sum((x-x.mean(0))**2*weights,axis=1)))
    differences=[];common_order=[];participation=[]
    for i in range(4):
        a,b=[q['readout']['records'][i]['metrics'] for q in saved]
        assert a['N']==b['N']==8
        differences.append(compare(a,b,names))
        aa,bb=[np.asarray(m['within_shaft_order_probability'],float) for m in [a,b]]
        common=np.isfinite(aa)&np.isfinite(bb)
        common_order.append(dict(ordered_pairs=int(common.sum()),
            maximum_absolute_change=float(abs(aa[common]-bb[common]).max())))
        participation.append([a['participation'][channel],b['participation'][channel]])
    assert all(q['maximum_absolute_change']==0 for q in common_order)
    # Counterfactual readout only: keep the qualified events and omit the
    # newly recruited contact when normalizing rank. Never change the
    # original detector or its primary metrics.
    observed_right=observe_cycle(profiles[1][0],profiles[1][1],model,contract)
    controls=[]
    for record,reference,stored in zip(observed_right['records'],saved[0]['readout']['records'],saved[1]['readout']['records']):
        assert all(v==0 for v in compare(stored['metrics'],record['metrics'],names).values())
        centroids=np.array(record['contact_centroids_ms'])[record['qualified_event_indices']].copy()
        centroids[:,channel]=np.nan
        held=describe(centroids,names)
        controls.append(dict(bin_origin_ms=record['bin_origin_ms'],qualified_events=len(centroids),
            SCL7_excluded_rank_control=compare(reference['metrics'],held,names),
            original_difference=compare(reference['metrics'],record['metrics'],names)))
    assert all(all(v==0 for v in q['SCL7_excluded_rank_control'].values()) for q in controls)
    control_source=DATA/'SCL7_fixed_participant_rank_control.json'
    write(control_source,dict(status='CONTROL_COMPLETE',rows=controls,source_indices=[31,32],
        observer_source=str(OLD/'observer_firing.json'),
        intervention='Derived readout control only: omit the newly recruited SCL7 centroid from the right-hand rank calculation while holding its original qualified-event set fixed. Original observer outputs are unchanged.',
        scope='Separates rank renormalization due to the added participant from reordering of the previously participating contacts. This is not a change to the model or an alternative primary detector.'))
    output=dict(status='FROZEN_OBSERVER_RECRUITMENT_BRACKET_CHECKED',
        sources=[str(folder/f'point_{i:04d}.json') for i in [31,32]],
        J_bracket=[q['check']['J'] for q in saved],T_ms=[t for r,t in profiles],
        common_phase_aligned_935_population_RMS_difference_Hz=float(distance[0]),
        relative_waveform_difference=float(distance[0]/scale),
        common_phase_shift_cycles=float(phase[0]),SCL7_group_participation_by_bin_origin=participation,
        original_three_metric_differences=differences,common_order_checks=common_order,
        fixed_participant_rank_control=str(control_source),
        observer_source=str(OLD/'observer_firing.json'),
        minimum_detection_ms=contract['minimum_detection_ms'],
        statistical_unit='One physical periodic solution at each J; eight repeats and four bin origins are deterministic sampling checks, not independent events or confidence intervals.',
        scope='SCL7 begins meeting the original amplitude-and-duration detector within this bracket. This is an observation boundary, not an independently identified dynamical bifurcation. Stability of these return-arm profiles remains unclassified.')
    write(DATA/'SCL7_recruitment_transition.json',output)
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(15.2,4.8),gridspec_kw={'width_ratios':[1.15,1.1,1]})
    fig.subplots_adjust(left=.065,right=.985,bottom=.29,top=.84,wspace=.36)
    for i,((r,T),regional) in enumerate(zip(profiles,regions)):
        for k,color in enumerate([*COL,'#777777']):
            axes[0].plot(np.arange(len(r))*T/len(r),regional[:,k],
                color=color,ls='-' if i==0 else '--',lw=1.5 if i==0 else 1.15)
    axes[0].set(xlim=(0,profiles[0][1]),xlabel='Time (ms)',ylabel='Rate (Hz / E cell)',
                title='A  Adjacent B-leading cycles')
    colors=['#256e9e','#cb5b2b']
    for i,traces in enumerate(envelopes):
        for trace in traces:
            axes[1].scatter(trace['relative_ms'],trace['threshold_ratio'],s=8,
                color=colors[i],alpha=.38,linewidths=0)
    axes[1].axhline(1,color='black',ls='--',lw=1)
    axes[1].text(.5,.98,'Detection: 2 consecutive 2-ms bins',transform=axes[1].transAxes,
                 ha='center',va='top',fontsize=9)
    axes[1].set(xlim=(-15,15),ylim=(.70,1.055),xlabel='Time from SCL7 peak (ms)',
        ylabel='Filtered envelope / contact threshold',title='B  SCL7 threshold neighborhood')
    keys=['mean_rank_difference','within_shaft_order_difference','participation_difference']
    for j,key in enumerate(keys):
        vals=np.array([d[key] for d in differences])
        axes[2].plot([j,j],[vals.min(),vals.max()],color='#555555',lw=1.3)
        axes[2].scatter(j+np.linspace(-.08,.08,4),vals,s=25,color='#345870',zorder=3)
    axes[2].set(xticks=[0,1,2],xticklabels=['Mean\nrank','Within-shaft\norder','Contact\nparticipation'],
        ylabel='Difference between the two solutions',ylim=(-.006,.08),
        title='C  Three frozen contact metrics')
    axes[2].axhline(0,color='#777777',lw=.7)
    axes[2].plot(0,0,marker='^',mfc='white',mec='#b04433',ms=7,ls='',
                 label='Rank: original participants',zorder=5)
    axes[2].legend(frameon=False,fontsize=9,loc='upper left')
    for ax in axes:style(ax)
    fig.legend(handles=[Line2D([0],[0],color=c,label=n) for c,n in zip([*COL,'#777777'],['Core A','Core B','Surround'])],
        loc='lower left',bbox_to_anchor=(.055,.04),ncol=3,frameon=False,fontsize=10)
    handles=[Line2D([0],[0],color=colors[i],marker='o',lw=0,
        label=rf'$J_{{\mathrm{{EE,core}}}}={q["check"]["J"]:.6f}$'+(' (solid)' if i==0 else ' (dashed)'))
        for i,q in enumerate(saved)]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.59,.04),ncol=2,frameon=False,fontsize=10)
    fig.text(.73,.025,'Four bin origins; no trial-based uncertainty',fontsize=9,ha='center')
    fig.suptitle('SCL7 recruitment changes without a change in the observed common-contact order',fontsize=13,y=.98)
    save_new(fig,'SCL7_recruitment_transition')
    path=OUTPUT/'figures/README.md';body=path.read_text();name='SCL7_recruitment_transition.png'
    body=re.sub(r'^### '+re.escape(name)+r'\s*\n.*?(?=^### |\Z)','',body,flags=re.M|re.S).rstrip()
    body+='\n\n### '+name+'\n比较 B 主导返回段两个相邻物理周期解的核内波形、SCL7 阈值附近包络及原有三项触点评价的差异。SCL7 只有达到冻结观察器的幅度及连续两个 2 ms bin 条件才计为参与，原先共同参与触点的杆内顺序不变；保持事件集合、仅在派生 rank 对照中排除新加入的 SCL7 后，三项差异均归零。**关注点**：原始指标不被对照替换；四个 bin 起点是确定性离散化检查，不是独立重复或置信区间；这里定位的是读出边界，尚未确证新的动力学分岔。\n'
    path.write_text(body)
    print('SCL7 TRANSITION',output,flush=True)


if __name__=='__main__':main()
