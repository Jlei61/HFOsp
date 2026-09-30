"""Actual physical doubled children and their spatial/contact changes, in Hz."""
from plot_rate_branch_completion import *


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--profiles-only',action='store_true',
        help='Show checked physical child waveforms while withholding unresolved criticality')
    parser.add_argument('--branch',choices=['pd4','pd1','pd3'],default='pd4')
    args=parser.parse_args()
    pd={'pd1':'PD1','pd3':'PD3','pd4':'PD4'}[args.branch]
    label={'pd1':'PD_double_low','pd3':'PD_A_return','pd4':'PD_H2_after_LPC13'}[args.branch]
    source=PERIODIC_OUT/f'{label}_child_validation.json'
    if args.branch=='pd1':
        folder=DATA/'physical_children/PD_double_low'
        branch_source=PERIODIC_OUT/'PD1_physical_20260920_branch_N16384.json'
        rows=[]
        for row in read(branch_source):
            physical=read(folder/(Path(row['orbit']).stem+'_physical.json'))
            assert Path(physical['orbit']).resolve()==Path(row['orbit']).resolve()
            rows.append(dict(**row,physical_check=physical))
        assert len(rows)==3 and {v['amplitude_hz'] for v in rows}=={10.,20.,40.}
        source=folder/'physical_snapshot.json'
        q=dict(rows=rows,child_orbit=rows[-1]['orbit'],criticality='PENDING',child_stability='UNCLASSIFIED',
            branch_source=str(branch_source),scope='Frozen physical profiles only; the separate live child-spectrum worker remains authoritative for later stability checks.')
        stability_source=folder/'result.json'
        if stability_source.exists():
            from complete_rate_positive_stability import paired_modes
            stability=read(stability_source)
            if stability.get('full_physical_child_checks') and stability.get('attempts'):
                last=stability['attempts'][-1]
                assert all(Path(v['orbit']).resolve()==Path(q['child_orbit']).resolve() for v in last['spectra'])
                verdict=paired_modes(*last['spectra'])
                if verdict['status']=='UNSTABLE':
                    q['child_stability']='UNSTABLE'
                    q['child_stability_source']=str(stability_source)
                    q['child_stability_check']=verdict
        if not args.profiles_only:
            from validate_rate_PD1_physical_child import assess
            checked=assess();canonical=read(PERIODIC_OUT/f'{label}_child_validation.json')
            assert checked['status']=='SUBCRITICAL_PD' and canonical['canonical_criticality_promoted']
            assert Path(checked['child_orbit']).resolve()==Path(q['child_orbit']).resolve()
            assert Path(canonical['child_orbit']).resolve()==Path(q['child_orbit']).resolve()
            q['criticality']='SUBCRITICAL_PD'
            q['classification_source']=str(PERIODIC_OUT/f'{label}_child_validation.json')
        write(source,q)
    elif args.branch=='pd3':
        assert not args.profiles_only
        classification_source=PERIODIC_OUT/'PD_return_child_classification.json'
        classification=read(classification_source)
        assert classification['status']=='SUPERCRITICAL_PD'
        assert classification['child_stability']=='UNSTABLE'
        assert classification['full_physical_child_checks']
        modes=read(classification['physical_mode_source'])
        assert modes['full_physical_child_checks'] and modes['child_stability']=='UNSTABLE'
        assert modes['departure_coefficient_relative_spread']<.01
        assert modes['dominant_relative_step_change']<.01
        rows=[dict(**row,physical_check=row['physical']) for row in modes['physical_geometry']]
        assert len(rows)==4
        assert Path(rows[-1]['orbit']).resolve()==Path(classification['child_orbit']).resolve()
        assert all(Path(v['orbit']).resolve()==Path(rows[-1]['orbit']).resolve()
                   and abs(values(v)[0])>1000 for v in modes['dominant_spectra'])
        source=DATA/'PD3_child_followup/figure_snapshot.json'
        q=dict(rows=rows,child_orbit=classification['child_orbit'],
            criticality='SUPERCRITICAL_PD',child_stability='UNSTABLE',
            classification_source=str(classification_source),
            child_stability_source=classification['physical_mode_source'],
            departure_fit_orbits=[row['orbit'] for row in rows[:3]],
            scope='The first three small children show local departure; the fourth physical child displays the finite half-period spatial/contact difference. This unstable child is not an autonomous attractor.')
        write(source,q)
    elif args.profiles_only:
        source=DATA/'H2_local_PD/physical_children.json';rows=read(source)['rows']
        q=dict(rows=rows,child_orbit=rows[-1]['orbit'],criticality='PENDING',child_stability='UNCLASSIFIED')
        instability=DATA/'H2_local_PD/child_inherited_instability.json'
        if instability.exists():
            v=read(instability)
            assert v['status']=='PHYSICAL_CHILD_UNSTABLE'
            assert Path(v['orbit']).resolve()==Path(q['child_orbit']).resolve()
            q['child_stability']='UNSTABLE'
    else:
        q=read(source);assert q['status']=='LOCALLY_CHECKED_PD_CHILD' and q['full_physical_child_checks']
        identity_file=Path(q['radial_mode_identity_source'])
        identity=read(identity_file)
        assert identity['status']=='RADIAL_MODE_IDENTITY_PASS'
        assert Path(identity['orbit']).resolve()==Path(q['child_orbit']).resolve()
    parent=read(PERIODIC_OUT/f'{label}_validation.json');assert parent['full_acceptance']
    s=RateField();z=np.load(q['child_orbit']);r=z['r'];N=len(r);half=N//2;T=float(z['T'])
    assert r.min()>0
    assert all(row['physical_check']['filter_state_check']['positive'] and
        row['physical_check']['maximum_group_defect_Hz']<1e-6 for row in q['rows'])
    if args.branch=='pd1':write(source,q)
    regional=np.array([s.regional_rates(v) for v in r])
    shift=int(np.argmin(regional[:half,:2].sum(axis=1)))
    r=np.roll(r,-shift,axis=0);regional=np.roll(regional,-shift,axis=0)
    t=np.arange(N)*T/N;difference=(r[half:]-r[:half])*1000
    geo=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz')
    order=contact_indices(geo['contact_names'].tolist());xy=geo['contact_xy']
    contact=r@s.geo['contact_rate_weights']*1000;contact_delta=contact[half:]-contact[:half]
    cell=s.geo['group_cell'];mass=s.geo['group_size'];e=s.E
    count=np.bincount(cell[e],weights=mass[e],minlength=400)
    projection=sparse.coo_matrix((mass[e]/np.maximum(count[cell[e]],1),
        (np.flatnonzero(e),cell[e])),shape=(s.P,400)).tocsr()
    field_difference=np.asarray(difference@projection)
    field_rms=np.sqrt(np.mean(field_difference**2,axis=0))
    weight=mass*e;weight/=weight.sum();departure=[]
    fit_orbits=set(q.get('departure_fit_orbits',[row['orbit'] for row in q['rows']]))
    for row in q['rows']:
        if row['orbit'] not in fit_orbits:continue
        assert row['physical_check']['filter_state_check']['positive']
        zz=np.load(row['orbit']);rr=zz['r'];h=len(rr)//2
        odd=(rr[h:]-rr[:h])*500
        departure.append(dict(J_shift=row['J_shift'],odd_E_RMS_Hz=float(np.sqrt(np.mean(odd*odd,axis=0)@weight)),orbit=row['orbit']))
    departure.sort(key=lambda row:row['odd_E_RMS_Hz'])
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,3,figsize=(14.5,8.6))
    fig.subplots_adjust(left=.065,right=.98,bottom=.10,top=.88,wspace=.47,hspace=.55)
    ax=axes[0,0]
    xx=np.array([row['J_shift'] for row in departure])
    yy=np.array([row['odd_E_RMS_Hz'] for row in departure])
    coefficient=float(np.mean(xx/yy**2))
    grid=np.linspace(0,xx[-1],200)
    exponent=9 if args.branch=='pd3' else 6
    ax.plot(grid*10**exponent,np.sqrt(grid/coefficient),'--',color='#168469',lw=1.,label='Local square-root scaling')
    ax.plot(xx*10**exponent,yy,'o',color='#168469',ms=4,label='Checked child')
    ax.plot(0,0,'v',color='black',ms=5)
    ax.legend(frameon=False,fontsize=8,loc='lower right')
    ax.set(xlabel=rf'$10^{exponent}(J_{{\mathrm{{EE,core}}}}-J_{{\mathrm{{{pd}}}}})$',
        ylabel='Odd-component RMS (Hz / E cell)',
        title='A   Three small children near PD3' if args.branch=='pd3' else 'A   Nonzero doubled branch')
    for k,color in enumerate([*COL,'#555555']):
        axes[0,1].plot(t,regional[:,k],color=color,lw=1.1,label=['Core A','Core B','Surround'][k])
        axes[0,2].plot(t[:half],regional[half:,k]-regional[:half,k],color=color,lw=1.1)
    axes[0,1].axvline(T/2,color='black',ls='--',lw=.7)
    axes[0,1].set(xlim=(0,T),xlabel='Time (ms)',ylabel='Rate (Hz / E cell)',title='B   Actual full child period')
    axes[0,1].legend(frameon=False,fontsize=8,loc='upper left')
    axes[0,2].axhline(0,color='black',lw=.6)
    axes[0,2].set(xlim=(0,T/2),xlabel='Time within each half (ms)',
        ylabel='Second − first half (Hz / E cell)',title='C   Alternation in regional rate')
    for ax in axes[0]:style(ax)
    ax=axes[1,0];im=ax.imshow(field_rms.reshape(20,20),origin='lower',extent=(0,20,0,20),cmap='magma',vmin=0)
    for k,center in enumerate(s.geo['centers_mm']):
        ax.add_patch(plt.Circle(center,1.5,fill=False,color='white',lw=.9))
        ax.text(*center,'AB'[k],color='white',ha='center',va='center',fontsize=8)
    ax.scatter(xy[:,0],xy[:,1],s=11,facecolors='none',edgecolors='cyan',linewidths=.6)
    ax.set(xlabel='x (mm)',ylabel='y (mm)',title='D   Actual spatial-field difference')
    fig.colorbar(im,ax=ax,pad=.04,shrink=.85,label='RMS difference (Hz / E cell)')
    for ax,data,end,title,delta in [(axes[1,1],contact,T,'E   SEEG-site rate readout',False),
                                   (axes[1,2],contact_delta,T/2,'F   Consecutive-half difference',True)]:
        lim=float(abs(data).max());opts=dict(cmap='RdBu_r',vmin=-lim,vmax=lim) if delta else dict(cmap='magma',vmin=0,vmax=lim)
        im=ax.imshow(data[:,order].T,aspect='auto',origin='upper',extent=(0,end,14.5,-.5),**opts)
        ax.axhline(3.5,color='black',lw=.7)
        if not delta:ax.axvline(T/2,color='white',ls='--',lw=.7)
        ax.set(yticks=np.arange(15),yticklabels=CONTACT_ORDER,xlabel='Time (ms)',title=title)
        ax.tick_params(axis='y',labelsize=7,length=2)
        for tick,name in zip(ax.get_yticklabels(),CONTACT_ORDER):tick.set_color(SHAFT_COLORS[name[:3]])
        fig.colorbar(im,ax=ax,pad=.04,shrink=.85,label='Rate difference (Hz / cell)' if delta else 'Rate (Hz / cell)')
    classification=('Criticality pending' if args.profiles_only else
        ('Subcritical' if q['criticality']=='SUBCRITICAL_PD' else 'Supercritical')+' in the flip direction')
    stability='Unstable child' if q['child_stability']=='UNSTABLE' else 'Child stability pending'
    figure_parameter_prefix='Displayed child: ' if args.branch=='pd3' else ''
    fig.suptitle(f'{pd}: nonzero doubled branch | {classification} | {stability}\n'+figure_parameter_prefix+
        rf'$J_{{\mathrm{{EE,core}}}}={float(z["J"]):.9f}$'+f' | Full period {T:.2f} ms',fontsize=13,y=.975)
    name=({'pd1':'PD1_physical_child','pd3':'PD3_physical_child'}[args.branch]
          if args.branch in ['pd1','pd3'] else
          ('H2_PD4_physical_child' if args.profiles_only else 'H2_PD4_nonlinear_child'))
    save_new(fig,name)
    write(DATA/(name+'.json'),dict(source=str(source),orbit=q['child_orbit'],criticality=q['criticality'],
        child_stability=q['child_stability'],departure=departure,phase_shift_samples=shift,
        child_stability_source=q.get('child_stability_source'),
        local_scaling_J_shift_per_odd_E_RMS_squared=coefficient,
        scaling_line_semantics='Square-root fit to the three small checked children; not additional continuation samples or a criticality verdict. The finite profile shown in B-F may lie farther along that same child branch.',
        maximum_regional_half_difference_Hz=np.max(abs(regional[half:]-regional[:half]),axis=0),
        maximum_contact_half_difference_Hz=np.max(abs(contact_delta),axis=0),
        contact_names=geo['contact_names'].tolist(),scope='Finite nonlinear child differences in physical Hz, not an arbitrarily normalized mode. Criticality and stability follow their separately stated evidence status; physical profile validity does not certify an attractor. Contact-weighted rate is not SEEG voltage.'))
    path=OUTPUT/'figures/README.md';body=path.read_text()
    body=re.sub(r'^### '+name+r'\.png\s*\n.*?(?=^### |\Z)','',body,flags=re.M|re.S).rstrip()
    note=('局部超／亚临界分类仍待特征模态验证；这张图只确认真实两倍周期轨道及其空间读出。'
          if args.profiles_only else '子轨道已验证为不稳定低率周期解；局部超／亚临界分类不等于稳定 burst 出现，也不等于传播模板切换。')
    if args.branch=='pd1' and q['child_stability']=='UNSTABLE':
        note=('已确认父轨道两侧稳定性、子分支向稳定侧的二次偏离，以及完整状态／历史中的增长方向对应，支持局部亚临界倍周期。子轨道不稳定，不能当作自发吸引子或新传播模板。'
              if q['criticality']=='SUBCRITICAL_PD' else
              '该子轨道的两步长完整延迟谱已确认增长模态，父分支两侧的稳定性另有独立证据；子分支模态对应及局部超／亚临界分类仍单独验收。')
    if args.branch=='pd3':
        note='A 展示三个小振幅子解的局部平方根生长，B–F 展示同一分支上较大振幅子解的实际空间与触点变化。倍周期方向局部超临界，但子解仍继承其他增长模态；不能解释为稳定 burst、irregular 状态或新传播模板的出现。'
    body+='\n\n### '+name+'.png\n展示 '+pd+' 的真实两倍周期子分支、完整周期波形、两半周期的实际放电率差、二维场差和固定触点率读出。所有差值均为该非线性周期解的实际 Hz，没有任意模态幅度归一化。**关注点**：'+note+'\n'
    path.write_text(body)


if __name__=='__main__':main()
