"""Validated primary burst folds: actual cycles and unchanged contact observer."""
from plot_rate_sameJ_burst_pair import *
from validate_rate_mean_fold import validation_matches_latest_root


def two_event_readouts(observation, peak_pairs, period, names):
    """Describe each model propagation direction without clinical TA/TB labels."""
    centers=np.array([np.mean([v[0] for v in pair]) for pair in peak_pairs])
    labels=['_to_'.join('AB'[v[1]] for v in pair) for pair in peak_pairs]
    assert set(labels)=={'A_to_B','B_to_A'}
    separation=abs((centers[1]-centers[0]+period/2)%period-period/2)
    scl=np.array([n.startswith('SCL') for n in names]);rows=[]
    for record in observation['records']:
        ids=np.asarray(record['qualified_event_indices'],int)
        centroids=np.asarray(record['contact_centroids_ms'],float)
        phases=np.nanmean(centroids[ids],axis=1)%period
        distance=abs((phases[:,None]-centers[None,:]+period/2)%period-period/2)
        assignment=np.argmin(distance,axis=1)
        maximum=float(distance[np.arange(len(ids)),assignment].max())
        assert maximum<separation/4, 'Event-to-core-peak pairing is ambiguous'
        for k,label in enumerate(labels):
            selected=ids[assignment==k]
            rows.append(dict(bin_origin_ms=record['bin_origin_ms'],model_direction=label,
                qualified_events=len(selected),
                SCL_qualified_events=int(np.isfinite(centroids[selected][:,scl]).any(1).sum()),
                metrics=describe(centroids[selected],names),
                maximum_event_center_to_peak_pair_distance_ms=maximum))
    return dict(rows=rows,core_peak_pairs=peak_pairs,
        method='Match each qualified event mean contact time modulo the network period to the nearest within-event A/B peak-pair center; no clinical template labels.',
        scope='Conditional readout within one exact periodic solution; eight repetitions are not independent events for statistical inference.')


def direction_metric_figure(case,names):
    rows=[q for q in case['within_cycle_direction_readouts']['rows'] if q['bin_origin_ms']==0]
    order=contact_indices(names);colors=['#2166ac','#ad42a4']
    fig,axes=plt.subplots(1,4,figsize=(17,6.2),gridspec_kw={'width_ratios':[1,1.12,1.12,1]})
    fig.subplots_adjust(left=.06,right=.975,bottom=.25,top=.78,wspace=.40)
    for column,key in [(0,'mean_rank'),(3,'participation')]:
        ax=axes[column]
        for q,color in zip(rows,colors):
            ax.plot(np.asarray(q['metrics'][key],float)[order],np.arange(15),'o-',color=color,ms=5,lw=1)
        ax.set(xlim=(-.04,1.04),ylim=(14.5,-.5),yticks=np.arange(15),yticklabels=CONTACT_ORDER,
            xlabel='Early → late' if column==0 else 'Participating fraction',
            title='Mean normalized rank' if column==0 else 'Contact participation')
        ax.axhline(3.5,color='black',lw=.6);style(ax)
        for tick,name in zip(ax.get_yticklabels(),CONTACT_ORDER):tick.set_color(SHAFT_COLORS[name[:3]])
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#d9d9d9')
    for ax,q in zip(axes[1:3],rows):
        matrix=np.asarray(q['metrics']['within_shaft_order_probability'],float)[np.ix_(order,order)]
        im=ax.imshow(np.ma.masked_invalid(matrix),cmap=cmap,vmin=0,vmax=1)
        ax.set(title=q['model_direction'].replace('_to_',' → ')+': within-shaft order',
            xticks=np.arange(15),xticklabels=CONTACT_ORDER,yticks=np.arange(15),yticklabels=CONTACT_ORDER)
        ax.tick_params(axis='x',labelrotation=90,labelsize=7);ax.tick_params(axis='y',labelsize=7)
        ax.axhline(3.5,color='black',lw=.6);ax.axvline(3.5,color='black',lw=.6)
    fig.colorbar(im,cax=fig.add_axes([.41,.13,.29,.018]),orientation='horizontal',
        label='Fraction: row contact precedes column contact')
    from matplotlib.patches import Patch
    handles=[Line2D([0],[0],color=c,marker='o',label=q['model_direction'].replace('_to_',' → ')+
        f': {q["qualified_events"]} eligible repeats') for q,c in zip(rows,colors)]
    handles.append(Patch(facecolor='#d9d9d9',label='Pair not estimable / different shafts'))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.005),ncol=3,frameon=False,fontsize=9)
    fig.suptitle('LPC20: both propagation directions within the same periodic solution\n'
        'Eight repeated cycles; conditional observables, not independent trials',fontsize=13,y=.96)
    save_new(fig,'LPC20_direction_contact_metrics')


def main():
    labels=['LPC_double_low','LPC_Bleading_low','LPC_burst_low']
    s=RateField();contract_source=OLD/'observer_firing.json';contract=read(contract_source)
    geo=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz')
    names=contract['contact_names'];assert geo['contact_names'].tolist()==names
    order=contact_indices(names);cases=[]
    for name in labels:
        assert validation_matches_latest_root(name)
        validation=read(PERIODIC_OUT/(name+'_validation.json'))
        check=validation['continuous_defect'];assert check['filter_state_check']['positive']
        assert check['maximum_group_defect_Hz']<.001
        source=Path(validation['mesh_checks'][-1]['orbit']);z=np.load(source)
        r=z['r'];N=len(r);T=float(z['T']);J=float(z['J']);assert r.shape[1]==935
        assert Path(check['orbit']).resolve()==source.resolve()
        rg=np.array([s.regional_rates(v) for v in r])
        shift=int(np.argmin(rg[:,:2].sum(1)))
        if name!='LPC_double_low':
            peaks=np.argmax(rg[:,:2],axis=0)
            lag=(peaks[1]-peaks[0]+N/2)%N-N/2;leader=1 if lag<0 else 0
            shift=int(peaks[leader]-round(60/T*N))%N
        r=np.roll(r,-shift,axis=0);rg=np.roll(rg,-shift,axis=0)
        if name=='LPC_double_low':
            peak_records=[]
            for k in [0,1]:
                peaks=find_peaks(np.tile(rg[:,k],3),height=10,prominence=5,
                                 distance=max(1,int(40/T*N)))[0]
                peaks=peaks[(peaks>=N)&(peaks<2*N)]-N
                assert len(peaks)==2, 'Retain both events of the full alternating cycle'
                peak_records.extend((int(i),k) for i in peaks)
            peak_records.sort();ids=[v[0] for v in peak_records]
            pairs=[[(float(i*T/N),k) for i,k in peak_records[start:start+2]] for start in [0,2]]
        else:
            peaks=np.argmax(rg[:,:2],axis=0)
            times=[peaks[leader]*T/N-20,peaks[leader]*T/N,
                   peaks[1-leader]*T/N,peaks[1-leader]*T/N+40]
            assert min(times)>=0 and max(times)<T
            ids=[int(round(t/T*N))%N for t in times]
        observation=observe_cycle(r,T,s,contract)
        directional=(two_event_readouts(observation,pairs,T,names)
                     if name=='LPC_double_low' else None)
        cases.append(dict(label=CRITICAL_LABELS[name],internal_label=name,orbit=str(source),
            validation_source=str(PERIODIC_OUT/(name+'_validation.json')),J_EE_core=J,T_ms=T,
            r=r,regional=rg,phase_shift_samples=shift,snapshot_indices=ids,
            snapshot_times_ms=[float(i*T/N) for i in ids],observer=observation,
            within_cycle_direction_readouts=directional))
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(17,10))
    grid=fig.add_gridspec(3,3,width_ratios=[1.15,1.85,1.3],left=.058,right=.98,
                         top=.89,bottom=.13,wspace=.31,hspace=.58)
    cells=s.geo['group_cell'];sz=s.geo['group_size']
    counts=np.bincount(cells[s.E],weights=sz[s.E],minlength=400)
    limit=max(q['regional'].max() for q in cases)*1.07
    for row,q in enumerate(cases):
        r=q['r'];rg=q['regional'];T=q['T_ms'];N=len(r);t=np.arange(N)*T/N
        ax=fig.add_subplot(grid[row,0])
        for k,color in enumerate([*COL,'#555555']):ax.plot(t,rg[:,k],color=color,lw=1.25)
        ax.set(xlim=(0,T),ylim=(0,limit),xlabel='Time (ms)',ylabel='Rate (Hz / E cell)',
            title=f'{q["label"]} | '+rf'$J_{{\mathrm{{EE,core}}}}={q["J_EE_core"]:.6f}$'+f'\nT = {T:.1f} ms')
        style(ax);sub=grid[row,1].subgridspec(1,4,wspace=.16)
        for col,index in enumerate(q['snapshot_indices']):
            field=np.bincount(cells[s.E],weights=sz[s.E]*r[index,s.E]*1000,minlength=400)/np.maximum(counts,1)
            ax=fig.add_subplot(sub[col]);imf=ax.imshow(np.ma.masked_less_equal(field.reshape(20,20),0),
                origin='lower',extent=(0,20,0,20),cmap='inferno',norm=LogNorm(.03,500))
            for center in s.geo['centers_mm']:ax.add_patch(plt.Circle(center,1.5,fill=False,color='white',lw=.8))
            ax.scatter(geo['contact_xy'][:,0],geo['contact_xy'][:,1],s=10,facecolors='none',edgecolors='#4dcbd2',linewidths=.6)
            ax.set(title=f'{t[index]:.0f} ms',xticks=[0,10,20],yticks=[0,10,20]);ax.tick_params(labelsize=7)
            if col==0:ax.set_ylabel('y (mm)',fontsize=8)
            else:ax.set_yticklabels([])
            if row==2:ax.set_xlabel('x (mm)',fontsize=8)
        contact=r@s.geo['contact_rate_weights']*1000
        ax=fig.add_subplot(grid[row,2]);imc=ax.imshow(contact[:,order].T,origin='upper',aspect='auto',
            extent=(0,T,14.5,-.5),cmap='magma',norm=LogNorm(.03,200))
        ax.set(yticks=np.arange(15),yticklabels=CONTACT_ORDER,xlabel='Time (ms)')
        ax.tick_params(axis='y',labelsize=7);ax.axhline(3.5,color='white',lw=.6)
        for tick,n in zip(ax.get_yticklabels(),CONTACT_ORDER):tick.set_color(SHAFT_COLORS[n[:3]])
    for x,title in [(.058,'A  Core / surround activity'),(.375,'B  Same-orbit spatial propagation'),
                    (.795,'C  SEEG-site rate readout')]:fig.text(x,.943,title,weight='bold',fontsize=12)
    fig.legend(handles=[Line2D([0],[0],color=c,label=n) for c,n in
        zip([*COL,'#555555'],['Core A','Core B','Surround'])],loc='lower left',
        bbox_to_anchor=(.055,.032),frameon=False,ncol=3,fontsize=9)
    fig.colorbar(imf,cax=fig.add_axes([.40,.057,.26,.012]),orientation='horizontal',label='E rate (Hz / cell; log scale)')
    fig.colorbar(imc,cax=fig.add_axes([.78,.057,.19,.012]),orientation='horizontal',label='Contact rate (Hz / cell; log scale)')
    name='primary_burst_fold_activity';save_new(fig,name)
    direction_metric_figure(cases[0],names)
    metadata=dict(rows=[{k:v for k,v in q.items() if k not in ['r','regional','snapshot_indices']} for q in cases],
        contact_names=names,observer_source=str(contract_source),
        model='Unchanged 400-cell / 935-population spatial rate DDE',
        scope='Exact critical periodic solutions. Local fold validation does not establish attraction or a connection between these three families. Contact-weighted rate is not voltage; event participation uses the unchanged observer. Eight repeated cycles and four bin origins are not independent trials.')
    write(DATA/(name+'.json'),metadata)
    path=OUTPUT/'figures/README.md';body=path.read_text()
    body=re.sub(r'^### '+name+r'\.png\s*\n.*?(?=^### |\Z)','',body,flags=re.M|re.S).rstrip()
    body+='\n\n### '+name+'.png\n展示 LPC20、LPC7、LPC5 三个已通过物理波形和独立临界模态检查的周期折点，逐行对应同一解的双核及核外波形、四帧二维场和触点发放率。LPC20 保留一个完整周期内的两次传播，另外两行显示各自的先行核到后行核过程；原冻结观察器下的三个触点指标保存在同名数值文件中。**关注点**：这是临界周期解，不把它们当作稳定吸引子或已连接分支；模态方向另见 primary_burst_fold_shape_modes。\n'
    metric_name='LPC20_direction_contact_metrics'
    body=re.sub(r'^### '+metric_name+r'\.png\s*\n.*?(?=^### |\Z)','',body,flags=re.M|re.S).rstrip()
    body+='\n\n### '+metric_name+'.png\n对 LPC20 同一个周期解内部的 A→B、B→A 两类事件分别计算平均归一化 rank、杆内相对顺序和触点参与比例，保留共同参与不足时的缺失。两类事件按模型两核峰时序配对，不使用患者 TA/TB 标签；图示原观察器一个 bin 起点，其余三个起点也保存在数值结果中。**关注点**：这是同一个临界周期解的条件读出，不是独立试验或两种稳定吸引子；重复周期数不提供独立统计样本。\n'
    path.write_text(body)
    for q in cases:print(q['label'],[(v['qualified_events'],v['sustained_SCL_contact_names']) for v in q['observer']['records']],flush=True)


if __name__=='__main__':main()
