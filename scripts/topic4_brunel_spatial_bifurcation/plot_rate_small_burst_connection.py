"""Expose same-J return cycles, Floquet evidence and finite basin probes."""
from plot_rate_focused_composite import *
from run_rate_sameJ_basin_bridge import DEST as DATA
from rate_floquet_poincare import values
from matplotlib.colors import LogNorm

OUTPUT=ROOT/'results/topic4_sef_hfo/interictal_rate_small_burst_connection_20260919'


def save_new(fig,name):
    folder=OUTPUT/'figures';folder.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf','svg']:fig.savefig(folder/f'{name}.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)


def cases(s):
    manifest=read(PERIODIC_OUT/'composite_case_resolution.json')
    intermediate=read(DATA/'sameJ_intermediate_cycles.json')
    assert intermediate['status']=='CORRECTIONS_FINISHED'
    assert all(q['status']=='RESOLUTION_CHECKED' for q in intermediate['rows'])
    paths=[next(q['orbit'] for q in manifest['rows'] if q['case']=='b'),
        *[next(q['orbit'] for q in intermediate['rows'] if q['label']==label) for label in ['H1_middle','H1_return']],
        next(q['orbit'] for q in manifest['rows'] if q['case']=='c')]
    out=[]
    for label,title,path in zip(['b','u1','u2','c'],['Small oscillation','H1 first return','H1 later return','Alternating bursts'],paths):
        path=Path(path);z=np.load(path);r=z['r'];T=float(z['T']);J=float(z['J']);assert J==.942
        rg=np.array([s.regional_rates(x) for x in r]);shift=np.argmin(rg[:,:2].sum(1))
        check=PERIODIC_OUT/'poincare_floquet'/f'{path.stem}_step_check.json'
        assert check.exists(),check
        status=read(check)
        candidates=[(f,read(f)) for f in check.parent.glob(path.stem+'_dt*.json')]
        fine_path,fine=min(candidates,key=lambda pair:abs(pair[1]['dt_ms']-status['dt_ms'][-1]))
        assert abs(fine['dt_ms']-status['dt_ms'][-1])<1e-12
        status['fine_spectrum_file']=str(fine_path)
        mu=values(fine)
        margins=np.maximum(2e-5,np.maximum(6*np.array(status['matched_multiplier_changes']),
                                          4*fine['phase_tangent_relative_defect']))
        reliable=np.array(fine['residuals'])/np.maximum(1,abs(mu))<1e-6
        outside=reliable&(abs(mu)>1+margins);inside=reliable&(abs(mu)<1-margins)
        status['per_mode_safety_margins']=margins
        status['numerically_resolved_unstable_dimension']=(int(outside.sum()) if
            status['filter_coverage'] and np.all(outside|inside) and max(fine['phase_overlap'])<1e-6 else None)
        status['confirmed_unstable_dimension_lower_bound']=int(outside.sum())
        out.append(dict(label=label,title=title,orbit=str(path),r=np.roll(r,-shift,axis=0),
            regional=np.roll(rg,-shift,axis=0),T=T,J=J,stability=status,floquet=fine))
    return out


def draw_branches(ax,k,rows,fs):
    # The H1 route is shown through the return segment bracketing the new u2
    # correction; it includes the excursion to lower J that the focus plot hid.
    a=fs['A'];target=next(i for i,q in enumerate(a) if Path(q['path']).stem=='arcAreturnStrong_0048_N512')
    cycle_line(ax,a[:target+1],k,FAMILY['A'])
    cycle_line(ax,hopf_departures(fs)['B'],k,FAMILY['B'],width=1.1)
    cycle_line(ax,fs['double'],k,FAMILY['double'])
    equilibrium(ax,k)
    for q in rows:
        y=q['regional'].mean(0)[k];status=q['stability']['status']
        ax.plot(.942,y,'x' if status=='UNSTABLE' else 'o',
            mfc='white' if status=='UNRESOLVED' else None,
            color=FAMILY['double'] if q['label']=='c' else FAMILY['A'],ms=6,zorder=7)
        offset={'b':(-35,-17),'u1':(-44,2),'u2':(-32,10),'c':(-20,12)}[q['label']]
        if k==1:offset={'b':(-36,-17),'u1':(29,5),'u2':(-32,10),'c':(-20,12)}[q['label']]
        ax.annotate(q['label'],(.942,y),xytext=offset,textcoords='offset points',weight='bold',
            arrowprops=dict(arrowstyle='-',lw=.7),bbox=dict(fc='white',ec='none',pad=.5))
    h=read(RATE_OUT/'hopfs.json')['rows'][0]
    ax.plot(h['J_EE_core'],h['rates_hz'][k],'o',color='black',ms=4)
    ax.annotate('H1',(h['J_EE_core'],h['rates_hz'][k]),xytext=(-62,12) if k else (-48,12),
        textcoords='offset points',fontsize=8,arrowprops=dict(arrowstyle='-',lw=.6),
        bbox=dict(fc='white',ec='none',pad=.3))
    # Two independently checked folds delimit the large excursion in J.
    cycle_critical(ax,k,['LPC_A1','LPC_A_low_extension'],annotate=False)
    for name,offset in [('LPC_A1',(4,27)),('LPC_A_low_extension',(14,14))]:
        root=next(q for q in critical() if q['label']==name)
        meta=read(Path(root['orbit']).with_suffix('.json'))
        ax.annotate(CRITICAL_LABELS[name],(root['J_EE_core'],meta['mean_rates_hz'][k]),
            xytext=offset,textcoords='offset points',fontsize=8,
            arrowprops=dict(arrowstyle='-',lw=.6),bbox=dict(fc='white',ec='none',pad=.3))
    ax.set(xlim=(.695,.983),ylim=(.53,25),yscale='log',xlabel=r'$J_{\mathrm{EE,core}}$',
        ylabel=rf'$\langle r_{{{"AB"[k]}}}\rangle$ (Hz / E cell)',title=f'Core {"AB"[k]}: H1 return route')
    ax.yaxis.set_major_locator(FixedLocator([1,2,5,10,20]));ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'))
    style(ax)


def composite(s,rows):
    fs=families()
    repaired=read(DATA/'displayed_H1_path_refinement.json');assert repaired['status']=='COMPLETE'
    routecheck=read(DATA/'displayed_H1_path_continuous_check.json');assert routecheck['status']=='PASS'
    mapping={q['source']:q['orbit'] for q in repaired['rows']}
    fs['A']=[read(Path(mapping[q['path']]).with_suffix('.json')) if q['path'] in mapping else q for q in fs['A']]
    fig=plt.figure(figsize=(21,10.8));grid=fig.add_gridspec(4,4,
        width_ratios=[1.35,1.1,1.75,1.25],left=.05,right=.985,bottom=.13,top=.88,wspace=.4,hspace=.7)
    left=grid[:,0].subgridspec(2,1,hspace=.37)
    for k in [0,1]:draw_branches(fig.add_subplot(left[k]),k,rows,fs)
    geo=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz')
    xy=geo['contact_xy'];order=contact_indices(geo['contact_names'].tolist())
    field_cmap=plt.get_cmap('inferno').copy();field_cmap.set_bad('black');field_cmap.set_under('black')
    contact_cmap=plt.get_cmap('magma').copy();contact_cmap.set_bad('black');contact_cmap.set_under('black')
    cell=s.geo['group_cell'];size=s.geo['group_size'];ct=np.bincount(cell[s.E],weights=size[s.E],minlength=400)
    for i,q in enumerate(rows):
        rg=q['regional'];r=q['r'];N=len(r);T=q['T'];t=np.arange(N)*T/N
        ax=fig.add_subplot(grid[i,1])
        for k,color in enumerate([*COL,'#555555']):ax.plot(t,rg[:,k],color=color,lw=1.1)
        ax.set(xlabel='Time (ms)',ylabel='Hz / E cell',xlim=(0,T),ylim=(0,max(1.,rg.max()*1.1)))
        verdict={'NUMERICALLY_STABLE':'stable','UNSTABLE':'unstable'}.get(q['stability']['status'],'unresolved')
        ax.set_title(f'{q["label"]}  {q["title"]}\n{verdict}; T={T:.1f} ms',loc='left',fontsize=10);style(ax)
        if q['label']=='c':
            ids=[]
            for k in [0,1]:
                peaks=find_peaks(np.tile(rg[:,k],3),height=20,distance=N//4)[0]
                ids.extend((peaks[(peaks>=N)&(peaks<2*N)]-N).tolist())
            ids=sorted(ids);assert len(ids)==4
        else:ids=(np.arange(4)*N//4).tolist()
        sub=grid[i,2].subgridspec(1,4,wspace=.15)
        for j,index in enumerate(ids):
            fld=np.bincount(cell[s.E],weights=size[s.E]*r[index,s.E]*1000,minlength=400)/np.maximum(ct,1)
            ax=fig.add_subplot(sub[j]);imf=ax.imshow(fld.reshape(20,20),origin='lower',extent=(0,20,0,20),
                cmap=field_cmap,norm=LogNorm(.03,500))
            for center in s.geo['centers_mm']:ax.add_patch(plt.Circle(center,1.5,fill=False,color='white',lw=.8))
            ax.scatter(xy[:,0],xy[:,1],s=6,facecolors='none',edgecolors='cyan',linewidths=.5)
            ax.set(xticks=[0,20],yticks=[0,20],title=f'{t[index]:.0f} ms');ax.tick_params(labelsize=8)
            if j:ax.tick_params(labelleft=False)
            else:ax.set_ylabel('y (mm)')
            if i==3:ax.set_xlabel('x (mm)')
        ax=fig.add_subplot(grid[i,3]);contact=r@s.geo['contact_rate_weights']*1000
        imc=ax.imshow(contact[:,order].T,origin='upper',aspect='auto',extent=(0,T,14.5,-.5),
            cmap=contact_cmap,norm=LogNorm(.03,200))
        ax.set(yticks=np.arange(15),yticklabels=CONTACT_ORDER,xlabel='Time (ms)');ax.tick_params(axis='y',labelsize=7)
        ax.axhline(3.5,color='white',lw=.6)
        for tick,n in zip(ax.get_yticklabels(),CONTACT_ORDER):tick.set_color(SHAFT_COLORS[n[:3]])
    fig.suptitle(r'Same parameter $J_{\mathrm{EE,core}}=0.942$: periodic solutions and their stability',y=.985,fontsize=14)
    for x,title in [(.31,'Periodic waveforms'),(.53,'Same-orbit spatial field'),(.815,'SEEG-site rate readout')]:fig.text(x,.945,title,weight='bold',fontsize=11)
    fig.legend(handles=[Line2D([0],[0],color=FAMILY['A'],ls=LINESTYLE,label='H1 traced branch'),
        Line2D([0],[0],color=FAMILY['B'],ls=LINESTYLE,label='H2 local branch'),
        Line2D([0],[0],color=FAMILY['double'],ls=LINESTYLE,label='Alternating-burst branch'),
        Line2D([0],[0],color='black',ls='-',label='Stable equilibrium'),
        Line2D([0],[0],color='black',ls=LINESTYLE,label='Dotted: stability unclassified'),
        Line2D([0],[0],color='black',marker='s',ls='',label='Cycle fold: orbit + mode checked'),
        Line2D([0],[0],color='black',marker='o',ls='',label='Stable cycle: checked point'),
        Line2D([0],[0],color='black',marker='x',ls='',label='Unstable cycle: checked point')],
        loc='lower left',bbox_to_anchor=(.035,.016),ncol=2,frameon=False,fontsize=8.5)
    fig.legend(handles=[Line2D([0],[0],color=c,label=n) for c,n in zip([*COL,'#555555'],['Core A','Core B','Surround'])],
        loc='lower left',bbox_to_anchor=(.30,.035),frameon=False,fontsize=9,ncol=3)
    fig.colorbar(imf,cax=fig.add_axes([.55,.062,.18,.012]),orientation='horizontal',label='E rate (Hz / cell; log scale)')
    fig.colorbar(imc,cax=fig.add_axes([.822,.062,.14,.012]),orientation='horizontal',label='Contact rate (Hz / cell; log scale)')
    save_new(fig,'sameJ_return_cycles_spatial_composite')


def diagnostics(rows):
    fig,axs=plt.subplots(2,3,figsize=(15,8.2));fig.subplots_adjust(wspace=.32,hspace=.42,bottom=.09,top=.91)
    ax=axs[0,0];names=[q['label'] for q in rows];radii=[max(abs(values(q['floquet']))) for q in rows]
    ax.bar(names,radii,color=[{'NUMERICALLY_STABLE':'#1b9e77','UNSTABLE':'#d95f02'}.get(q['stability']['status'],'#999999') for q in rows])
    ax.axhline(1,color='black',ls='--',lw=1);ax.set(yscale='log',ylabel=r'Largest returned nontrivial $|\mu|$',title='A  Same-J Floquet stability')
    for x,y in zip(names,radii):ax.annotate(f'{y:.4g}',(x,y),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9)
    for q in rows:
        v=values(q['floquet']);axs[0,1].plot(v.real,v.imag,'o',ms=4,label=q['label'])
    angle=np.linspace(0,2*np.pi,300);axs[0,1].plot(np.cos(angle),np.sin(angle),'--',color='black',lw=.8)
    axs[0,1].set(xlabel=r'Re $\mu$',ylabel=r'Im $\mu$',xlim=(-1.25,1.25),ylim=(-1.25,1.25),
        title='B  Multipliers near the unit circle')
    axs[0,1].set_aspect('equal',adjustable='box')
    axs[0,1].legend(frameon=False,ncol=4,loc='upper center',bbox_to_anchor=(.5,-.19),fontsize=8)
    old=read(PERIODIC_OUT/'unstable_cycle_departure_diagnostics.json');windows=old['windows']
    for k,color in enumerate(COL):
        tt=np.array([np.mean(q['window_ms'])/1000 for q in windows]);lo=[q['core'][k]['minimum_Hz'] for q in windows];hi=[q['core'][k]['maximum_Hz'] for q in windows]
        axs[0,2].fill_between(tt,lo,hi,color=color,alpha=.2)
        axs[0,2].plot(tt,[q['core'][k]['mean_Hz'] for q in windows],color=color,label=f'Core {"AB"[k]}')
    axs[0,2].set(xlabel='Time (s)',ylabel='Hz / E cell',title='C  Weak-cycle departure at J=0.946');axs[0,2].legend(frameon=False)
    for ax,fraction in zip(axs[1],[.01,.25,.5]):
        folder=DATA/'runs'/f'fraction{fraction:g}_dt0.1';result=read(folder/'result.json');z=np.load(folder/'trajectory.npz')
        for k,color in enumerate(COL):ax.plot(z['time_ms']/1000,z['regional_rates_hz'][:,k],color=color,lw=.65)
        ax.set(xlabel='Time (s)',ylabel='Hz / E cell',title=f'Initial mixture {fraction:g} → {result["finite_window_match"]}')
    for ax in axs.ravel():style(ax)
    save_new(fig,'sameJ_floquet_and_initial_history_outcomes')


def departures():
    fig,axs=plt.subplots(2,2,figsize=(13,7.3),sharex=True,sharey='col')
    fig.subplots_adjust(left=.075,right=.985,bottom=.09,top=.87,hspace=.43,wspace=.22)
    records=[]
    for i,amplitude in enumerate([.005,.0025]):
        for j,sign in enumerate([-1,1]):
            path=DATA/'departures'/f'sign{sign:+d}_dt0.05_a{amplitude:g}'/'result.json'
            assert path.exists(),'Complete both amplitudes and sides before drawing'
            q=read(path);z=np.load(path.parent/'trajectory.npz');ax=axs[i,j]
            records.append(dict(source=str(path),result=q))
            for k,color in enumerate([*COL,'#555555']):
                ax.plot(z['time_ms']/1000,z['regional_rates_hz'][:,k],color=color,lw=.85,label=['Core A','Core B','Surround'][k])
            ax.set(ylabel='Hz / E cell',title=f'Amplitude {amplitude:g} Hz; direction {sign:+d} → {q["finite_window_match"]}')
            style(ax)
    axs[0,0].legend(frameon=False,ncol=3)
    for ax in axs[-1]:ax.set_xlabel('Time (s)')
    fig.suptitle('u1: both sides of one unstable Floquet direction at J=0.942',fontsize=13)
    save_new(fig,'sameJ_unstable_return_departures')
    outcome=all(q['result']['finite_window_match']==('b' if q['result']['contract']['sign']==-1 else 'c') for q in records)
    return dict(status='REPRODUCED_TWO_SIDED_DEPARTURE' if outcome else 'OUTCOME_REVIEW_REQUIRED',trials=records,
        scope='Two amplitudes of transverse Floquet perturbation lead to the two different stable cycles. This supports a local basin-boundary role for u1; its complete stable manifold and a global parameter-space branch junction are not computed.')


def main():
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    s=RateField();rows=cases(s);composite(s,rows);diagnostics(rows);boundary=departures()
    write(OUTPUT/'connection_evidence.json',dict(
        cases=[{k:v for k,v in q.items() if k not in ['r','regional']}|dict(mean_rates_hz=q['regional'].mean(0)) for q in rows],
        known_connection='b, u1 and u2 belong to the traced H1 continuation family.',
        displayed_H1_route_filter_audit=str(DATA/'displayed_H1_path_filter_check.json'),
        displayed_H1_route_refinement=str(DATA/'displayed_H1_path_refinement.json'),
        displayed_H1_route_continuous_check=str(DATA/'displayed_H1_path_continuous_check.json'),
        unresolved_connection='A global parameter-space branch junction between the H1 family and c is not established; the complete basin-boundary stable manifold is not computed.',
        two_sided_departure_evidence=boundary,
        boundary='Initial-history interpolation is not continuation in J. u1/u2 are periodic solutions, with their own checked stability; no stable intermediate state is inferred from intermediate mean rate.',
        initial_history_trials=[dict(source=str(f),result=read(f)) for f in sorted((DATA/'runs').glob('*/result.json'))],
        unstable_departure_trials=[dict(source=str(f),result=read(f)) for f in sorted((DATA/'departures').glob('*/result.json'))],
        native_equivalence='Not assessed here; frozen rate model only',human_visual_acceptance='PENDING'))
    (OUTPUT/'figures/README.md').write_text(
        '### sameJ_return_cycles_spatial_composite.png\n在同一 J=0.942 下对照小振荡、两条 H1 返回周期解和交替 burst；每行波形、二维场和十五触点读出来自同一周期解。左图恢复通向两条返回解的 H1 路径，实点或叉号只表示精确案例的稳定性，点线不推断整段稳定性。**关注点**：返回解不因均值居中就成为稳定中间态，也尚未证明其为 burst 吸引域边界。\n\n'
        '### sameJ_floquet_and_initial_history_outcomes.png\n上排给出同参数周期解的 Floquet 乘子，以及已有 J=0.946 弱周期扰动后的约 180 秒结果；下排展示固定 J=0.942、改变完整初态和延迟历史的自主运行。乘子柱高是多项式谱筛选返回的最大模，稳定性另经单位圆外覆盖判据和步长复核，不能把柱高当成完整谱的谱半径。**关注点**：初态混合系数不是连接强度，有限时长的去向不替代全局分支连接证明。\n\n'
        '### sameJ_unstable_return_departures.png\n从 u1 周期解沿 Poincare 返回映射的主要实不稳定特征向量正负扰动，完整保留局部状态和延迟历史，再自主积分。符号仅表示特征向量方向，终态以全部群体对周期模板的匹配判定。**关注点**：两个方向的去向用于检验这条返回解是否可能分隔小振荡与 burst，不能仅凭均值居中就认定它是分界。\n')


if __name__=='__main__':main()
