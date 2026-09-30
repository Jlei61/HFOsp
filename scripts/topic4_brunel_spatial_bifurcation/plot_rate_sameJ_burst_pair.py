"""Exact same-J branch waveforms, spatial fields and frozen contact statistics."""
from plot_rate_branch_completion import *
from expanded_readouts import observer,smooth2,describe,compare,OLD
from scipy.interpolate import CubicSpline
from collections import Counter


def observe_cycle(r,T,s,contract):
    names=contract['contact_names'];N=len(r)
    contacts=r@s.geo['contact_rate_weights']*1000
    spline=CubicSpline(np.arange(N+1)*T/N,np.r_[contacts,contacts[:1]],bc_type='periodic')
    first=max(4,int(np.ceil(contract['burnin_ms']/T))+2);window=[first*T,(first+8)*T]
    stop=int((first+12)*T)//2*2;records=[]
    for offset in [0.,.5,1.,1.5]:
        times=(np.arange(stop*4)+.5)/4+offset
        rates=spline(times%T).reshape(stop,4,15).mean(1)
        env=smooth2(rates.reshape(-1,2,15).sum(1)/1000)
        ob=observer.observe(env.T,2.,contract)
        centroids=np.asarray(ob['centroid_ms'],float).reshape(-1,15)+offset
        anchors=np.array([np.mean(e['window_ms'])+offset for e in ob['events']])
        ids=np.flatnonzero((anchors>=window[0])&(anchors<window[1]))
        primary=np.array([i for i in ids if ob['events'][i]['primary_eligible']],int)
        scl=np.array([n.startswith('SCL') for n in names])
        frame_times=np.arange(len(env))*2+1+offset
        interior=(frame_times>=window[0])&(frame_times<window[1])
        threshold=np.asarray(ob['threshold'])
        detected=env.T>threshold[:,None]
        minimum=max(1,int(np.ceil(contract['minimum_detection_ms']/2.)))
        for ci in range(len(names)):
            for start,end in observer.runs(detected[ci]):
                if end-start<minimum:detected[ci,start:end]=False
        recruited=detected[:,interior].any(1)
        records.append(dict(bin_origin_ms=offset,detected_events=len(ids),qualified_events=len(primary),
            SCL_qualified_events=int(np.isfinite(centroids[primary][:,scl]).any(1).sum()),
            sustained_contact_names=[n for n,v in zip(names,recruited) if v],
            sustained_SCL_contact_names=[n for n,v in zip(names,recruited&scl) if v],
            contact_peak_to_threshold=env[interior].max(0)/threshold,
            exclusions=dict(Counter(reason for i in ids for reason in ob['events'][i]['primary_exclusion_reasons'])),
            metrics=describe(centroids[primary],names),qualified_event_indices=primary,
            contact_centroids_ms=centroids,observation=ob))
    return dict(interior_window_ms=window,interior_cycles=8,records=records)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pair-source',type=Path,default=DATA/'Bleading_extension/sameJ_pair.json')
    parser.add_argument('--output-prefix',default='sameJ_burst')
    args=parser.parse_args();assert Path(args.output_prefix).name==args.output_prefix
    source=args.pair_source;pair=read(source)
    if source.name!='sameJ_pair.json':assert args.output_prefix!='sameJ_burst'
    assert pair['status']=='TWO_PHYSICAL_SAME_J_PERIODIC_SOLUTIONS'
    assert pair['corrected_profile_check']['filter_state_check']['positive']
    assert pair['source_profile_check']['filter_state_check']['positive']
    assert abs(pair['rows'][0]['J_EE_core']-pair['rows'][1]['J_EE_core'])<1e-12
    s=RateField();geo=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz')
    contract_source=OLD/'observer_firing.json';contract=read(contract_source)
    names=contract['contact_names'];assert geo['contact_names'].tolist()==names
    order=contact_indices(names);cases=[]
    for row in pair['rows']:
        z=np.load(row['orbit']);r=resample(z['r'],4096,axis=0);N=len(r);T=float(z['T'])
        rg=np.array([s.regional_rates(v) for v in r]);peaks=np.argmax(rg[:,:2],axis=0)
        lag=(peaks[1]-peaks[0]+N/2)%N-N/2;leader=1 if lag<0 else 0
        shift=int(peaks[leader]-round(60/T*N))%N
        r=np.roll(r,-shift,axis=0);rg=np.roll(rg,-shift,axis=0)
        peaks=np.argmax(rg[:,:2],axis=0);times=peaks*T/N
        snapshots=[times[leader]-20,times[leader],times[1-leader],times[1-leader]+40]
        assert min(snapshots)>=0 and max(snapshots)<T
        observation=observe_cycle(r,T,s,contract)
        cases.append(dict(**row,r=r,regional=rg,time=np.arange(N)*T/N,N=N,
            phase_shift_samples=shift,leading_core='AB'[leader],snapshots_ms=snapshots,
            contacts=r@s.geo['contact_rate_weights']*1000,observer=observation))
    contrasts=[compare(cases[0]['observer']['records'][i]['metrics'],cases[1]['observer']['records'][i]['metrics'],names) for i in range(4)]
    metadata=dict(source=str(source),J_EE_core=pair['rows'][0]['J_EE_core'],observer_source=str(contract_source),
        contact_names=names,rows=[{k:v for k,v in q.items() if k not in ['r','regional','time','N','contacts']} for q in cases],
        shaft_balanced_contact_differences=contrasts,
        statistical_unit='One exact periodic solution on each branch at identical J. Eight repeated cycles and four bin origins are discretization checks, not independent trials.',
        scope='Distinct physical periodic solutions and their rate readouts, not established stable attractors or a demonstrated switching bifurcation. Missing jointly participating SCL contacts makes the corresponding balanced rank/order contrast undefined.')
    write(source.with_name(source.stem+'_readouts.json'),metadata)
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(17,6.9));grid=fig.add_gridspec(2,3,width_ratios=[1.1,1.8,1.2],
        left=.06,right=.98,bottom=.18,top=.82,wspace=.30,hspace=.50)
    cells=s.geo['group_cell'];sz=s.geo['group_size'];counts=np.bincount(cells[s.E],weights=sz[s.E],minlength=400)
    rate_limit=max(105.,max(q['regional'].max() for q in cases)*1.07)
    for i,q in enumerate(cases):
        ax=fig.add_subplot(grid[i,0]);T=q['T_ms'];t=q['time']
        for k,color in enumerate([*COL,'#555555']):ax.plot(t,q['regional'][:,k],color=color,lw=1.35)
        recruited=q['observer']['records'][0]['sustained_SCL_contact_names']
        scl_label=', '.join(recruited)+' above threshold' if recruited else 'No SCL above threshold'
        ax.set(xlim=(0,T),ylim=(0,rate_limit),xlabel='Time (ms)',ylabel='Rate (Hz / E cell)',
            title=f'{q["leading_core"]}-leading | T = {T:.1f} ms\n{scl_label}')
        style(ax)
        sub=grid[i,1].subgridspec(1,4,wspace=.17)
        for j,ms in enumerate(q['snapshots_ms']):
            ix=int(round(ms/T*q['N']))%q['N']
            field=np.bincount(cells[s.E],weights=sz[s.E]*q['r'][ix,s.E]*1000,minlength=400)/np.maximum(counts,1)
            ax=fig.add_subplot(sub[j]);imf=ax.imshow(np.ma.masked_less_equal(field.reshape(20,20),0),
                origin='lower',extent=(0,20,0,20),cmap='inferno',norm=LogNorm(.1,400))
            for center in s.geo['centers_mm']:ax.add_patch(plt.Circle(center,1.5,fill=False,color='white',lw=.9))
            ax.scatter(geo['contact_xy'][:,0],geo['contact_xy'][:,1],s=9,facecolors='none',edgecolors='#4dcbd2',linewidths=.6)
            ax.set(title=f'{ms:.0f} ms',xticks=[0,10,20],yticks=[0,10,20]);ax.tick_params(labelsize=7)
            if i==1:ax.set_xlabel('x (mm)',fontsize=8)
            if j==0:ax.set_ylabel('y (mm)',fontsize=8)
            else:ax.set_yticklabels([])
        ax=fig.add_subplot(grid[i,2]);imc=ax.imshow(q['contacts'][:,order].T,origin='upper',aspect='auto',
            extent=(0,T,14.5,-.5),cmap='magma',norm=LogNorm(.1,300))
        ax.set(yticks=np.arange(15),yticklabels=CONTACT_ORDER,xlabel='Time (ms)');ax.tick_params(axis='y',labelsize=7)
        ax.axhline(3.5,color='white',lw=.6)
        for tick,name in zip(ax.get_yticklabels(),CONTACT_ORDER):tick.set_color(SHAFT_COLORS[name[:3]])
        if i==0:ax.set_title('Contact-weighted rate')
    fig.suptitle(rf'$J_{{\mathrm{{EE,core}}}}={pair["rows"][0]["J_EE_core"]:.7f}$'
                 ' | Two periodic solutions; stability unclassified',fontsize=13,y=.97)
    fig.text(.37,.88,'Same-orbit spatial E activity',fontsize=12)
    fig.legend(handles=[Line2D([0],[0],color=c,label=n) for c,n in zip([*COL,'#555555'],['Core A','Core B','Surround'])],
        loc='lower left',bbox_to_anchor=(.05,.04),ncol=3,frameon=False,fontsize=9)
    fig.colorbar(imf,cax=fig.add_axes([.38,.095,.28,.017]),orientation='horizontal',label='E rate (Hz / cell; log scale)')
    fig.colorbar(imc,cax=fig.add_axes([.76,.095,.21,.017]),orientation='horizontal',label='Contact rate (Hz / cell; log scale)')
    save_new(fig,args.output_prefix+'_spatial_comparison')
    # Three original contact observables: vectors and pair order, not invented
    # scalar patient-fit scores. Use one bin origin here; retain all four above.
    fig,axes=plt.subplots(1,4,figsize=(17,6.2),gridspec_kw={'width_ratios':[1,1.12,1.12,1]})
    fig.subplots_adjust(left=.06,right=.975,bottom=.25,top=.78,wspace=.40)
    metrics=[q['observer']['records'][0]['metrics'] for q in cases]
    colors=['#795548','#d17c00']
    for column,key in [(0,'mean_rank'),(3,'participation')]:
        ax=axes[column]
        for m,color,q in zip(metrics,colors,cases):
            values=np.asarray(m[key],float)[order]
            ax.plot(values,np.arange(15),'o-',ms=5,lw=1,color=color,
                    label=f'{q["family"]} (n={m["N"]})')
        ax.set(xlim=(-.04,1.04),ylim=(14.5,-.5),yticks=np.arange(15),yticklabels=CONTACT_ORDER,
            xlabel='Early → late' if column==0 else 'Participating fraction',
            title='Mean normalized rank' if column==0 else 'Contact participation')
        ax.axhline(3.5,color='#777777',lw=.7);style(ax)
        for tick,name in zip(ax.get_yticklabels(),CONTACT_ORDER):tick.set_color(SHAFT_COLORS[name[:3]])
    # Display each solution's order where defined. An all-missing difference
    # matrix would hide the available A-leading ICL ordering and its denominator.
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#d9d9d9')
    for ax,m,q in zip(axes[1:3],metrics,cases):
        matrix=np.asarray(m['within_shaft_order_probability'],float)[np.ix_(order,order)]
        im=ax.imshow(np.ma.masked_invalid(matrix),cmap=cmap,vmin=0,vmax=1)
        ax.set(title=f'{q["leading_core"]}-leading: within-shaft order',
            xticks=np.arange(15),xticklabels=CONTACT_ORDER,yticks=np.arange(15),yticklabels=CONTACT_ORDER)
        ax.tick_params(axis='x',labelrotation=90,labelsize=7);ax.tick_params(axis='y',labelsize=7)
        ax.axhline(3.5,color='black',lw=.6);ax.axvline(3.5,color='black',lw=.6)
        if not m['N']:
            ax.text(.5,.5,'No eligible group events\nOrder undefined',transform=ax.transAxes,
                ha='center',va='center',fontsize=10,bbox=dict(facecolor='white',edgecolor='none',pad=7))
    fig.colorbar(im,cax=fig.add_axes([.41,.13,.29,.018]),orientation='horizontal',
        label='Fraction: row contact precedes column contact')
    from matplotlib.patches import Patch
    handles=[Line2D([0],[0],color=c,marker='o',label=f'{q["leading_core"]}-leading: {m["N"]} eligible events')
             for c,q,m in zip(colors,cases,metrics)]
    handles.append(Patch(facecolor='#d9d9d9',label='Pair not estimable / different shafts'))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.005),ncol=3,frameon=False,fontsize=9)
    fig.suptitle('Frozen contact observer on two same-J periodic solutions\n'
                 'One solution per branch; eight repeated cycles, not independent trials',fontsize=13,y=.96)
    save_new(fig,args.output_prefix+'_contact_metrics')
    file=OUTPUT/'figures/README.md';body=file.read_text()
    entries={args.output_prefix+'_spatial_comparison':'在完全相同的 J 下并列两条已通过物理波形检查的周期解，显示两核及 surround 时序、400格二维 E 场和固定触点率读出。时间原点只为便于观看而平移，四帧均来自各自行内的同一条轨道。**关注点**：不同周期解不等于已证明两个稳定吸引子共存，也不是已经定位传播切换分岔。',
             args.output_prefix+'_contact_metrics':'对相同 J 的两条周期解应用原冻结观察器，展示平均归一化 rank、各分支的杆内顺序矩阵，以及各触点参与比例。图示一个 bin 起点，四个起点的结果均存于配套数值中；重复周期不作为独立试验。**关注点**：有单触点检出仍可能没有合格群事件；缺乏合格事件或共同参与触点的指标保留不可估计，不能填零。'}
    for name,description in entries.items():
        body=re.sub(r'^### '+name+r'\.png\s*\n.*?(?=^### |\Z)','',body,flags=re.M|re.S).rstrip()
        body+='\n\n### '+name+'.png\n'+description+'\n'
    file.write_text(body)
    print('SAME-J READOUTS',[(q['family'],[(r['qualified_events'],r['SCL_qualified_events']) for r in q['observer']['records']]) for q in cases],contrasts,flush=True)


if __name__=='__main__':main()
