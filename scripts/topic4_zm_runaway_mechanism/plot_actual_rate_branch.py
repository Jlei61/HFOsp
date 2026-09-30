"""Evidence-aware conditional bifurcation diagram and matched spatial readouts.

Only saved periodic BVP solutions supply cycle means/extrema. Finite-time
irregular trajectories supply spatial panels, never purported periodic branches.
Unclassified cycles use dotted means and plus signs, reserving open squares
for demonstrated instability. The actual onset critical type is not guessed.
"""
from native_path import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


def period_rows():
    folder=OUT/'periodic';rows=[];s=model()
    for name in ['rate_seed_N1024','rate_second_N1024']:
        q=read(folder/f'{name}.json');q['orbit']=q['path'];rows.append(q)
    # Stop before the known alias-sensitive tiny turns in this coarse branch.
    for q in read(folder/'rate_up1024/continuation.json')['rows']:
        if q['D']>.14488:continue
        rows.append(dict(q,N=1024,path=q['orbit'],mean_hz=q['global_mean_hz'],
                         min_hz=q['global_min_hz'],max_hz=q['global_max_hz']))
    for label in ['rate_period_path_G2049_M8192','rate_period_path_G2049_linear',
                  'rate_physical_fine_G4097_M16384','rate_small_period_G4097',
                  'rate_compact_period_G4097','rate_refined_period_G4097',
                  'rate_compact_period_G8193','rate_compact_period_G8193_restart80',
                  'rate_refined_T2671_G8193','rate_refined_T26735_G8193',
                  'rate_upper_G8193_M65536','rate_upper_G16385_M65536',
                  'rate_upper_G32769_M262144_cached6',
                  'rate_turn_mesh_G16385_M65536','rate_turn_center_G16385_M65536',
                  'rate_turn_sides_G16385_M65536']:
        f=folder/label/'result.json'
        if not f.exists():continue
        for q in read(f)['rows']:
            if q['residual']<2e-8:rows.append(dict(q,orbit=q['path']))
    accepted=list((folder/'rate_turn_G4097').glob('eval_[0-9][0-9][0-9].npz'))
    pair=folder/'rate_same_D_pair_G8193'
    if (pair/'result.json').exists() and read(pair/'result.json')['status']=='TWO_DISTINCT_CONVERGED_CYCLES_AT_IDENTICAL_Z':
        accepted.append(pair/'upper_same_D.npz')
    for path in accepted:
        z=np.load(path)
        assert float(z['residual'])<2e-8
        g=z['r'][:,s.E]@s.mean_weights*1000
        rows.append(dict(N=len(z['r']),T_ms=float(z['T']),D=float(z['D']),
                         residual=float(z['residual']),mean_hz=float(g.mean()),
                         min_hz=float(g.min()),max_hz=float(g.max()),orbit=str(path)))
    # Refinements revisit identical periods. Connect the physical branch in T
    # order and retain the finest accepted mesh at a duplicated period.
    unique={}
    for q in rows:
        key=round(q['T_ms'],6)
        priority=lambda row:(row['N'],row.get('nonlinear_samples',row['N']))
        if key not in unique or priority(q)>priority(unique[key]):unique[key]=q
    rows=sorted(unique.values(),key=lambda q:q['T_ms'])
    for q in rows:
        q['plot_stability']='UNRESOLVED'
        evidence=[]
        for f in (OUT/'floquet').glob('*.json'):
            if f.name.endswith(('.progress.json','.phase.json')):continue
            data=read(f)
            if not isinstance(data,dict) or 'orbit' not in data:continue
            if Path(data['orbit']).resolve()!=Path(q['orbit']).resolve():continue
            if data.get('phase_valid') and max(data.get('eigen_residuals',[1]))<1e-5:
                evidence.append(dict(path=str(f),**data))
        q['floquet_evidence']=[e['path'] for e in evidence]
        if len({round(e['dt_ms'],5) for e in evidence})>=2:
            types={e['sampled_stability'] for e in evidence}
            radii=[e['max_transverse_modulus'] for e in evidence]
            if len(types)==1 and max(radii)-min(radii)<.005:q['plot_stability']=types.pop()
        joint=OUT/'floquet/rate_near_upper_joint_stability_acceptance.json'
        if joint.exists():
            accepted=read(joint)
            if accepted['status']=='UNSTABLE_WITH_JOINT_ORBIT_AND_TIME_REFINEMENT' and Path(accepted['orbit']).resolve()==Path(q['orbit']).resolve():
                q['plot_stability']='UNSTABLE'
                q['floquet_evidence'].append(str(joint))
    return rows


def main():
    s=model();attach_rate_entry_path(s);periodic=period_rows()
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(13.0,7.2));gs=fig.add_gridspec(2,4,width_ratios=[3.5,1,1,1],
        left=.075,right=.90,bottom=.23,top=.92,wspace=.37,hspace=.30)
    ax=fig.add_subplot(gs[:,0]);ax.set_xlim(.14293,.14530);ax.set_ylim(0,260)
    ax.set_yscale('symlog',linthresh=1,linscale=.6)
    ax.set_yticks([0,1,10,100,250]);ax.set_yticklabels(['0','1','10','100','250'])
    ax.set_xlabel(r'$D=1-\langle Z_E\rangle$');ax.set_ylabel('Global E rate (Hz / neuron)')
    ax.text(-.14,1.04,'A',transform=ax.transAxes,fontweight='bold',fontsize=15)
    inset=ax.inset_axes([.25,.27,.72,.23]);inset.set_xlim(.14497165,.1449731);inset.set_ylim(28.03,28.12)
    inset.ticklabel_format(axis='x',style='plain',useOffset=False)
    inset.set_xticks([.1449720,.1449725,.1449730]);inset.set_xticklabels(['0.1449720','0.1449725','0.1449730'],fontsize=8)
    inset.tick_params(axis='y',labelsize=9)
    stability={(q['branch'],q['index']):q['status'] for q in read(OUT/'rate_equilibrium_branch_stability.json')['rows']}
    eq=[]
    for label in ['rate_up','rate_down']:
        rows=read(OUT/f'equilibria/{label}/result.json')['rows']
        assert all(stability[(label,q['index'])]=='UNSTABLE' for q in rows)
        ax.plot([q['D'] for q in rows],[q['global_E_hz'] for q in rows],'--',color='#252525',lw=1.5)
        eq.extend(rows)
    folds=read(OUT/'equilibria/rate_down_fold_audit/summary.json')['rows']
    for q in folds:
        if not ax.get_xlim()[0]<=q['D']<=ax.get_xlim()[1]:continue
        assert q['static_type']=='SN_static_conditions_met' and q['characteristic_zero']['temporal_simple_zero']
        ax.plot(q['D'],q['global_E_hz'],'D',mfc='white',mec='black',ms=6)
        ax.annotate(r'SN$_{eq}$',(q['D'],q['global_E_hz']),xytext=(-35,-23),textcoords='offset points',fontsize=10)
    for target in [ax,inset]:
        for q in periodic:
            kind=q['plot_stability'];D=q['D'];colour='#bd6d12'
            target.plot(D,q['mean_hz'],'.',color=colour,ms=4)
            if kind in ['STABLE','UNSTABLE']:
                if target is ax:target.plot([D,D],[q['min_hz'],q['max_hz']],ls='none',marker='s',ms=4,
                    mec='#16876f',mfc='#16876f' if kind=='STABLE' else 'white')
            elif target is ax:target.plot([D,D],[q['min_hz'],q['max_hz']],ls='none',marker='+',ms=4,color='#16876f')
        for first,second in zip(periodic[:-1],periodic[1:]):
            same=first['plot_stability']==second['plot_stability']
            status=first['plot_stability'] if same else 'UNRESOLVED'
            style='-' if status=='STABLE' else '--' if status=='UNSTABLE' else ':'
            target.plot([first['D'],second['D']],[first['mean_hz'],second['mean_hz']],style,color='#bd6d12',lw=1.3)
    # Operational long-time bracket, not a bifurcation star or named critical point.
    # The directly simulated loss bracket is wider than the local branch inset;
    # retain it in metadata rather than filling the whole inset with a band.
    specs=[(['endpoint_D0.1429804_dt0.05_rate'],1),
           (['rate_tail_D14497_D0.1449700_dt0.05'],2),
           (['rate_critical_finer_restart_D0.1449750_dt0.05',
             'rate_tail_D144975_D0.1449750_dt0.05'],3)]
    spatial=[];image_axes=[]
    for col,(names,number) in enumerate(specs,1):
        folder=OUT/'runs'/names[0];z=np.load(folder/'trajectory.npz');parts=[z['field_E_hz']]
        for name in names[1:]:
            extra=np.load(OUT/'runs'/name/'trajectory.npz')
            assert np.array_equal(extra['cell_counts'],z['cell_counts'])
            parts.append(extra['field_E_hz'])
        field=np.concatenate(parts)
        g=field@(z['cell_counts']/z['cell_counts'].sum());smoothed=np.convolve(g,np.ones(50)/50,mode='same')
        selection='Largest 50-ms mean in the final 4-s window'
        if number==3:
            events=next(q['events'] for q in read(OUT/'near_cycle_event_sequence.json')['rows'] if q['label']==names[0])
            event=next(e for e in events if e['duration_ms']>=200)
            peak=int(event['peak_ms'])
            selection='Peak of the first complete event lasting at least 200 ms; focuses on initial loss of short self-limited events'
        else:peak=len(g)-4000+int(np.argmax(smoothed[-4000:-300]))
        centers=[peak,peak+100]
        D=read(folder/'result.json')['D_initial'];windows=[]
        for row,t in enumerate(centers):
            a=fig.add_subplot(gs[row,col]);image_axes.append(a)
            win=[t-25,t+25];rate=field[win[0]:win[1]].mean(0)
            im=a.imshow(rate.reshape(20,20),origin='lower',extent=[0,20,0,20],cmap='magma',vmin=0,vmax=500)
            a.set_xticks([0,10,20]);a.set_yticks([0,10,20]);a.tick_params(labelsize=9)
            if row==0:a.text(.5,1.12,f'{number}   $D={D:.6f}$',transform=a.transAxes,ha='center',fontsize=10)
            if row==0 and col==1:a.text(-.25,1.4,'B',transform=a.transAxes,fontweight='bold',fontsize=15)
            if row==1:a.set_xlabel('x (mm)')
            if col==1:a.set_ylabel(('Peak' if row==0 else '+100 ms')+'\ny (mm)')
            for center,label in zip(s.geo['centers_mm'],'AB'):
                a.add_patch(Circle(center,1.5,fill=False,ec='#20c4cf',lw=1.0))
                a.text(center[0],center[1]+2.,label,color='#20c4cf',fontsize=8,ha='center')
            windows.append(dict(window_ms=win,global_E_hz=float(rate@(z['cell_counts']/z['cell_counts'].sum()))))
        spatial.append(dict(number=number,sources=[str(OUT/'runs'/name/'trajectory.npz') for name in names],
                            D=D,selection=selection,windows=windows))
    cax=fig.add_axes([.919,.31,.012,.46]);fig.colorbar(im,cax=cax,label='E rate (Hz)',ticks=[0,250,500])
    handles=[Line2D([],[],color='#252525',ls='--',label='Unstable equilibrium'),
             Line2D([],[],color='#bd6d12',ls=':',label='Cycle mean: stability unresolved'),
             Line2D([],[],color='#16876f',marker='+',ls='none',label='Cycle extrema: stability unresolved'),
             Line2D([],[],color='black',marker='D',mfc='white',ls='none',label='Equilibrium saddle-node')]
    if any(q['plot_stability']=='STABLE' for q in periodic):
        handles.append(Line2D([],[],color='#16876f',marker='s',ls='none',label='Cycle extrema: stable'))
    if any(q['plot_stability']=='UNSTABLE' for q in periodic):
        handles.append(Line2D([],[],color='#16876f',marker='s',mfc='white',ls='none',label='Cycle extrema: unstable'))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.51,.04),ncol=2,frameon=False,fontsize=10)
    dest=OUT/'figures';dest.mkdir(exist_ok=True);name='fig_actual_rate_conditional_branch_current'
    for ext in ['png','pdf','svg']:fig.savefig(dest/f'{name}.{ext}',dpi=190)
    plt.close(fig)
    write(dest/f'{name}.json',dict(status='CURRENT_PARTIAL_BIFURCATION_EVIDENCE',periodic=periodic,
        spatial_panels=spatial,equilibrium_rows=eq,Z='held spatial field',M='dynamic',
        unresolved_bracket=[.144970,.144975],bifurcation_type='NOT_ESTABLISHED',
        bracket_meaning='20-s finite-history loss of the regular short cycle at dt=.05ms; not a certified critical point',
        cycle_resolution='Coarse prefix excludes old spurious turns; near-critical branch refined by dealiased Galerkin',
        human_visual_acceptance='PENDING'))
    readme=dest/'README.md';text=readme.read_text() if readme.exists() else '';heading=f'### {name}.png / .pdf / .svg'
    description=('左侧是同一实际rate空间Z路径的已求得平衡支与周期解，横轴D为全E细胞数加权耗减，纵轴为放电率而非爆发重复频率。'
        '黑色虚线及菱形只表示已证实不稳定的平衡支和它的SN；周期稳定性尚未完成的段用点线均值、加号极值，不冒充稳定或不稳定方块。'
        '右侧三列为基线、临界近旁仍自限、首次长事件附近的实际50ms空间窗，均固定Z、动态M；第三列取第一段持续至少200ms的完整事件之峰值，下排相对各自上排延后100ms。'
        '**关注点**：内嵌图放大周期均值的回折；只有通过时间步复核的周期点使用实心或空心方块，高率平衡SN不能直接当作onset分岔。这条直线参数切面与逐10ms实际Z轨迹的边界应分别解释。\n')
    if heading not in text:readme.write_text(text+'\n'+heading+'\n'+description)
    else:
        before,after=text.split(heading,1);tail=after.find('\n### ')
        readme.write_text(before+heading+'\n'+description+(after[tail:] if tail>=0 else ''))
    log('ACTUAL RATE FIGURE',dest/f'{name}.png')


if __name__=='__main__':main()
