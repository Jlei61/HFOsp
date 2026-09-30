"""Freeze the newly checked burst stability and H2 continuation for display."""
import plot_rate_focused_composite as base
from plot_rate_focused_composite import *
from rate_floquet_poincare import values
from matplotlib.colors import LogNorm

OUTPUT=ROOT/'results/topic4_sef_hfo/interictal_rate_branch_completion_20260920'
DATA=Path('/data/hfosp/topic4_sef_hfo/interictal_rate_branch_completion_20260920')


def save_new(fig,name):
    folder=OUTPUT/'figures';folder.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf','svg']:fig.savefig(folder/f'{name}.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)


def checked_sites():
    return [read(f) for f in sorted((DATA/'sites').glob('[0-9][0-9][0-9].json'))
            if read(f).get('status') in ['NUMERICALLY_STABLE','UNSTABLE']]


def evidence(rows):
    result=physical_stability_samples()
    from check_rate_PD1_parent_display import verified_parent_witnesses
    base.EXACT_STABILITY_SAMPLES=verified_parent_witnesses()
    from check_rate_PD3_parent_display import verified_parent_witnesses as verified_PD3
    pd3_samples=verified_PD3()
    base.EXACT_STABILITY_SAMPLES+=pd3_samples
    for sample in pd3_samples:
        for key in ['original_orbit','orbit']:
            result[str(Path(sample[key]).resolve())]=dict(
                status=sample['status'],source=sample['source'],orbit=sample['orbit'])
    for q in rows:
        assert q['resolution']['filter_state_check']['positive']
        assert q['resolution']['status']=='RESOLUTION_CHECKED'
        assert max(q['attempts'][-1]['requested_steps_ms'])<=.1
        result[str(Path(q['original_orbit']).resolve())]=dict(status=q['status'],
            source=str(DATA/'sites'/f'{q["index"]:03d}.json'),orbit=q['analyzed_orbit'])
    from check_rate_Bleading_return_witness import verified_return_witness
    witness=verified_return_witness()
    if witness is not None:
        q=witness['evidence']
        result[str(Path(q['orbit']).resolve())]=dict(status=q['status'],
            source=witness['source'],orbit=q['orbit'])
    return result


def fold_detail(s,sites,fs,confirmed=True):
    by={q['index']:q for q in sites}
    fig,axs=plt.subplots(2,3,figsize=(14.6,8.4))
    fig.subplots_adjust(left=.075,right=.98,top=.92,bottom=.12,hspace=.45,wspace=.34)
    labels=[('LPC_Bleading_low','Bleading',[91,90],'B-leading burst fold'),
            ('LPC_burst_low','single',[68,69],'A-leading burst fold')]
    records=[];row_passes=[]
    for row,(label,family,indices,title) in enumerate(labels):
        v=read(PERIODIC_OUT/(label+'_validation.json'))
        root=max([read(f) for f in PERIODIC_OUT.glob(label+'_N*.json')],key=lambda q:q['N'])
        checked=v['mesh_checks'][-1]
        row_confirmed=(v['status']=='VALIDATED_CYCLE_FOLD' and checked['N']==root['N'] and
            Path(checked['orbit']).resolve()==Path(root['orbit']).resolve() and
            v['continuous_defect'].get('filter_state_check',{}).get('positive',False))
        row_passes.append(row_confirmed)
        if confirmed:assert row_confirmed,label
        meta=read(Path(root['orbit']).with_suffix('.json'))
        ax=axs[row,0];center=root['J_EE_core'];rr=fs[family]
        delta=4e-6 if family=='Bleading' else .0015
        k=0 if family=='Bleading' else 1;col=COL[k]
        cycle_line(ax,rr,k,col,width=1.3)
        ax.plot(center,meta['mean_rates_hz'][k],'s',ms=5,mec=col,
                mfc=col if row_confirmed else 'white')
        subset=[by[i] for i in indices if i in by]
        for q in subset:
            r=read(Path(q['analyzed_orbit']).with_suffix('.json'))
            ax.plot(q['J_EE_core'],r['mean_rates_hz'][k],
                'o' if q['status']=='NUMERICALLY_STABLE' else 'x',ms=6,color=col)
        ax.set(xlim=(center-delta*.15,center+delta),
            xlabel=r'$J_{\mathrm{EE,core}}$',ylabel=f'Core {"AB"[k]} period mean (Hz / E cell)',title=title)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.6f' if family=='Bleading' else '%.4f'))
        if family=='Bleading':ax.set_ylim(meta['mean_rates_hz'][k]-.22,meta['mean_rates_hz'][k]+.22)
        else:ax.set_ylim(14.5,20.5)
        ax.annotate(CRITICAL_LABELS[label],(center,meta['mean_rates_hz'][k]),
                    xytext=(10,15),textcoords='offset points',arrowprops=dict(arrowstyle='-',lw=.7))
        ax=axs[row,1]
        fold_modes=[]
        for j,q in enumerate(subset):
            mu=values(q['classification'])
            nearest=mu[np.argmin(abs(mu-1))];fold_modes.append(nearest)
            ax.plot(nearest.real,nearest.imag,'o' if q['status']=='NUMERICALLY_STABLE' else 'x',
                color=['#168469','#cf593c'][j],ms=7,label='Stable sample' if q['status']=='NUMERICALLY_STABLE' else 'Unstable sample')
        theta=np.linspace(0,2*np.pi,500)
        ax.plot(np.cos(theta),np.sin(theta),'--',color='black',lw=.8)
        if row_confirmed:ax.plot(1,0,'s',color='black',ms=5,label='Cycle fold')
        span=max(.05,max(abs(v-1) for v in fold_modes)*1.3)
        ax.set(xlim=(1-span,1+span),ylim=(-span,span),aspect='equal',xlabel=r'Re $\mu$',ylabel=r'Im $\mu$',
               title=r'Returned multiplier nearest $+1$')
        if row==0:ax.legend(frameon=False,fontsize=8,loc='upper left')
        # The independently reconstructed fold tangent has a generalized
        # +1 relation. Remove its phase component using all local/history
        # states before showing its output at the reference phase.
        if row_confirmed:
            modefile=PERIODIC_OUT/f'{label}_monodromy_check_N{root["N"]}_dt0.0125_analytic.npz'
            z=np.load(modefile)
            full=np.r_[z['local'].ravel(),z['history'].ravel()]
            phase=np.r_[z['phase_local'].ravel(),z['phase_history'].ravel()]
            full-=phase*(phase@full)/(phase@phase)
            local=full[:9*s.P].reshape(9,s.P)
            mode_title='Fold mode at reference phase';mode_J=center
        else:
            unstable=next(q for q in subset if q['status']=='UNSTABLE')
            source=Path(unstable['attempts'][-1]['sources'][-1]);q=read(source)
            mu=values(q);index=int(np.argmax(abs(mu)));modefile=source.with_suffix('.npz')
            z=np.load(modefile);vector=z['local_vectors'][:,index]
            assert abs(mu[index].imag)<1e-8 and np.linalg.norm(vector.imag)<1e-6*np.linalg.norm(vector.real)
            local=vector.real.reshape(9,s.P)
            mode_title='Nearby unstable cycle: rate mode';mode_J=unstable['J_EE_core']
        dr=s.alpha*local[0]+(1-s.alpha)*local[1]
        dr/=np.max(abs(dr[s.E]))
        if dr[np.flatnonzero(s.E)[np.argmax(abs(dr[s.E]))]]<0:dr=-dr
        cell=s.geo['group_cell'];sz=s.geo['group_size']
        counts=np.bincount(cell[s.E],weights=sz[s.E],minlength=400)
        field=np.bincount(cell[s.E],weights=sz[s.E]*dr[s.E],minlength=400)/np.maximum(1,counts)
        ax=axs[row,2];im=ax.imshow(field.reshape(20,20),origin='lower',extent=(0,20,0,20),
            cmap='RdBu_r',vmin=-1,vmax=1)
        for c in s.geo['centers_mm']:ax.add_patch(plt.Circle(c,1.5,fill=False,color='black',lw=.9))
        ax.set(xlabel='x (mm)',ylabel='y (mm)',title=mode_title)
        energy=np.array([np.sum(sz[s.E&(s.geo['group_region']==k)]*
             dr[s.E&(s.geo['group_region']==k)]**2) for k in range(3)])
        records.append(dict(label=label,validation=str(PERIODIC_OUT/(label+'_validation.json')),
            J_EE_core=center,critical_status=v['status'],mode_J_EE_core=mode_J,
            mode_kind='generalized critical fold tangent' if row_confirmed else 'unstable eigenvector of the specified nearby periodic orbit',
            mode_source=str(modefile),mode_energy_A_B_surround_at_reference_phase=energy/energy.sum(),
            scope='Mode energy is phase-dependent; it does not establish a causal core-to-core interaction.'))
    for ax in axs[:,:2].ravel():style(ax)
    fig.legend(handles=[Line2D([0],[0],color=c,label=f'Core {k}') for c,k in zip(COL,'AB')],
        loc='lower left',bbox_to_anchor=(.06,.015),ncol=2,frameon=False)
    fig.colorbar(im,cax=fig.add_axes([.79,.04,.15,.012]),orientation='horizontal',label='Relative rate perturbation')
    if not all(row_passes):
        fig.legend(handles=[Line2D([0],[0],marker='s',mfc='white',mec='black',ls='',
                   label='Fold location: fine-profile check pending')],loc='lower left',
                   bbox_to_anchor=(.30,.015),frameon=False,fontsize=9)
    save_new(fig,'burst_fold_stability_and_spatial_modes' if confirmed else 'burst_fold_neighborhood_evidence')
    return records


def h2_detail(fs):
    check=read(DATA/'H2_extension_check.json');assert check['status']=='SAMPLED_PASS'
    rr=[read(Path(p).with_suffix('.json')) for p in check['included_orbits']]
    prefix=[read(Path(p).with_suffix('.json')) for p in check['predecessor_sources']]
    new_paths={q['path'] for q in rr};older=[q for q in fs['B'] if q['path'] not in new_paths]
    located=[]
    spectrum_path=DATA/'primary_folds/LPC_B2_root_spectrum_assessment.json'
    if spectrum_path.exists():
        spectrum=read(spectrum_path)
        validation=read(PERIODIC_OUT/'LPC_B2_validation.json')
        root=validation['mesh_checks'][-1]
        assert validation['status']=='VALIDATED_CYCLE_FOLD'
        assert spectrum['status']=='FOLD_WITH_VERIFIED_UNSTABLE_MODES'
        assert spectrum['physical_profile_check']['positive']
        assert Path(root['orbit']).resolve()==Path(spectrum['orbit']).resolve()
        assert abs(root['J_EE_core']-spectrum['J_EE_core'])<1e-12
        located.append(('LPC_B2',root,read(Path(root['orbit']).with_suffix('.json')),True))
    for label in ['LPC_B_stage3_turn1','LPC_B_stage3_turn2',
                  'LPC_B_stage4_turn1','LPC_B_stage4_turn2']:
        path=PERIODIC_OUT/(label+'_validation.json')
        if not path.exists():continue
        v=read(path)
        versions=[read(f) for f in PERIODIC_OUT.glob(label+'_N*.json')]
        if len(versions)<2:continue
        root=max(versions,key=lambda q:q['N'])
        checked=v['mesh_checks'][-1]
        if checked['N']!=root['N'] or Path(checked['orbit']).resolve()!=Path(root['orbit']).resolve():continue
        if abs(v['J_EE_core']-root['J_EE_core'])>1e-10:continue
        if not v['continuous_defect'].get('filter_state_check',{}).get('positive',False):continue
        located.append((label,root,read(Path(root['orbit']).with_suffix('.json')),
                        v.get('status')=='VALIDATED_CYCLE_FOLD'))
    fig,axs=plt.subplots(1,2,figsize=(12,5.2));fig.subplots_adjust(bottom=.23,wspace=.28)
    for k,ax in enumerate(axs):
        cycle_line(ax,older,k,FAMILY['B'],width=1.1,alpha=.65)
        cycle_line(ax,prefix+rr,k,'#d2691e',width=2)
        for label,root,meta,passed in located:
            point=(root['J_EE_core'],meta['mean_rates_hz'][k])
            ax.plot(*point,'s',ms=5,mec='#222222',mfc='#222222' if passed else 'white',zorder=7)
            text=CRITICAL_LABELS[label]+(' (unstable)' if label=='LPC_B2' else '')
            ax.annotate(text,point,xytext=(9,13),textcoords='offset points',
                fontsize=8,arrowprops=dict(arrowstyle='-',lw=.6),
                bbox=dict(facecolor='white',edgecolor='none',pad=.2))
        base.cycle_critical(ax,k,['PD_H2_after_LPC13'])
        ax.plot(rr[-1]['J_EE_core'],rr[-1]['mean_rates_hz'][k],'|',color='#d2691e',ms=11,mew=2)
        ax.set(xlabel=r'$J_{\mathrm{EE,core}}$',ylabel='Period mean (Hz / E cell)',
            title=f'H2 continued family: Core {"AB"[k]}',yscale='log')
        ax.yaxis.set_major_locator(FixedLocator([.6,.8,1,1.2,1.4,1.6,1.8,2]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'))
        ax.yaxis.set_minor_locator(FixedLocator([]))
        style(ax)
    handles=[Line2D([0],[0],color=FAMILY['B'],ls=LINESTYLE,label='Previous H2 continuation'),
        Line2D([0],[0],color='#d2691e',ls=LINESTYLE,lw=2,label='New checked extension'),
        Line2D([0],[0],color='#d2691e',marker='|',ls='',ms=10,label='Computational endpoint')]
    if any(v[3] for v in located):
        handles.append(Line2D([0],[0],color='#222222',marker='s',ls='',ms=5,label='Cycle fold: checked'))
    if any(not v[3] for v in located):
        handles.append(Line2D([0],[0],color='#222222',mfc='white',marker='s',ls='',ms=5,label='Fold: checks pending'))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.02),
        ncol=2 if located else 3,frameon=False,fontsize=9)
    save_new(fig,'H2_extended_branch_coverage')
    write(OUTPUT/'H2_displayed_critical_points.json',dict(verified_labels=[v[0] for v in located if v[3]],
        pending_labels=[v[0] for v in located if not v[3]],
        validation_sources=[str(PERIODIC_OUT/(v[0]+'_validation.json')) for v in located],
        interval_completeness=False))
    return check


def h1_burst_projection(selected,evidence,wide=False):
    fig,axes=plt.subplots(1,3,figsize=(14.4,5.1))
    fig.subplots_adjust(left=.055,right=.985,bottom=.25,top=.91,wspace=.29)
    xlim=(.936,max(.962,max(q['J_EE_core'] for q in selected['A'])+.003)) if wide else (.936,.962)
    top=26
    if wide:
        top=max(top,1.15*max(max(q['mean_rates_hz'][:2]) for rows in selected.values()
                            for q in rows if xlim[0]<=q['J_EE_core']<=xlim[1]))
    for k,ax in enumerate(axes[:2]):
        equilibrium(ax,k)
        for name,rr in selected.items():cycle_line(ax,rr,k,FAMILY[name])
        draw_samples(ax,selected,k,evidence)
        offsets=({'LPC_Bleading_low':(-31,18),'LPC_burst_low':(16,13)}
                 if wide and k==1 else None)
        base.cycle_critical(ax,k,['LPC_A_return_recruitment','LPC_Bleading_low','LPC_burst_low'],
                            annotation_offsets=offsets)
        ax.set(xlim=xlim,ylim=(.5,top),yscale='log',
            xlabel=r'$J_{\mathrm{EE,core}}$',ylabel='Period mean (Hz / E cell)',title=f'Core {"AB"[k]}')
        ax.yaxis.set_major_locator(FixedLocator([v for v in [.5,1,2,5,10,20,50,100,200] if v<=top]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'))
        ax.yaxis.set_minor_locator(FixedLocator([]));style(ax)
    ax=axes[2]
    for name,rr in selected.items():
        bounds=[0,*continuation_breaks(rr),len(rr)]
        for start,stop in zip(bounds[:-1],bounds[1:]):
            part=rr[start:stop]
            ax.plot([q['J_EE_core'] for q in part],[q['T_ms'] for q in part],
                color=FAMILY[name],lw=1.5,ls=LINESTYLE)
    ax.set(xlim=xlim,ylim=(100,950),yscale='log',
        xlabel=r'$J_{\mathrm{EE,core}}$',ylabel='Full-network period (ms)',title='Period along each family')
    ax.yaxis.set_major_locator(FixedLocator([100,150,300,500,900]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'))
    ax.yaxis.set_minor_locator(FixedLocator([]));style(ax)
    handles=[Line2D([0],[0],color=FAMILY[name],ls=LINESTYLE,label=base.NAMES[name]) for name in selected]
    handles += [Line2D([0],[0],marker='o',color='#222222',ls='',label='Stable sample'),
                Line2D([0],[0],marker='x',color='#222222',ls='',label='Unstable sample'),
                Line2D([0],[0],marker='s',color='#222222',ls='',label='Checked cycle fold'),
                Line2D([0],[0],marker='s',mfc='white',mec='#222222',ls='',label='Fold checks pending')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.01),ncol=3,frameon=False,fontsize=9)
    save_new(fig,'H1_extension_and_burst_comparison' if wide else 'H1_return_and_burst_projection')


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',action='store_true');a=p.parse_args()
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    rows=checked_sites();fs=families();ev=evidence(rows);s=RateField()
    assert ({85,91,90,68,69} if a.snapshot else {85,91,90,68,69,99,100}).issubset({q['index'] for q in rows})
    assert read(DATA/'H2_extension_check.json')['status']=='SAMPLED_PASS'
    if not a.snapshot:
        for label in ['LPC_Bleading_low','LPC_burst_low']:
            assert read(PERIODIC_OUT/(label+'_validation.json'))['status']=='VALIDATED_CYCLE_FOLD'
    ccheck=read(PERIODIC_OUT/'poincare_floquet/refined_J0.942000000_N1536_accuracy_N3072_accuracy_N6144_step_check.json')
    if not a.snapshot:
        assert ccheck['status'] in ['NUMERICALLY_STABLE','UNSTABLE'] and max(ccheck['dt_ms'])<=.1
    # Reuse the accepted layout and exact all-rate case producers.
    base.DEST=OUTPUT;base.FIG=OUTPUT/'figures';base.SHOW_PD=True
    base.CONTINUATION_LABEL='Continuation estimate: checks pending'
    cases=load_cases(s)
    selected=primary(fs)
    previous=Path('/data/hfosp/topic4_sef_hfo/interictal_rate_small_burst_connection_20260919')
    routecheck=read(previous/'displayed_H1_path_continuous_check.json');assert routecheck['status']=='PASS'
    correction=read(previous/'displayed_H1_path_refinement.json');assert correction['status']=='COMPLETE'
    mapping={q['source']:q['orbit'] for q in correction['rows']}
    extension_file=DATA/'H1_display_extension.json'
    display_extension=read(extension_file) if extension_file.exists() else None
    extended=bool(display_extension and display_extension.get('status')=='PASS')
    last=84 if extended else 48
    if extended:
        assert {q['index'] for q in display_extension['rows']}==set(range(49,85))
        assert all(q['status']=='PASS' and q['branch_match_pass'] and
                   q['check']['filter_state_check']['positive'] for q in display_extension['rows'])
        mapping.update({q['source']:q['orbit'] for q in display_extension['rows']})
        base.EXTRA_RETURN_FOLDS=['LPC_A_return_recruitment']
    pd3_extension_file=DATA/'H1_to_PD3_display_extension.json'
    pd3_prefix=[]
    if extended and pd3_extension_file.exists():
        extension=read(pd3_extension_file)
        for row in extension.get('rows',[]):
            if row['index']!=last+1 or row['status']!='PASS':break
            assert row['branch_match_pass'] and row['check']['filter_state_check']['positive']
            assert row['check']['maximum_group_defect_Hz']<.001
            pd3_prefix.append(row);mapping[row['source']]=row['orbit'];last=row['index']
    stop=next(i for i,q in enumerate(fs['A']) if Path(q['path']).stem==f'arcAreturnStrong_{last:04d}_N512')
    selected['A']=[read(Path(mapping.get(q['path'],q['path'])).with_suffix('.json')) for q in fs['A'][:stop+1]]
    if extended:
        root=read(PERIODIC_OUT/'LPC_A_return_recruitment_validation.json')
        assert root['status']=='VALIDATED_CYCLE_FOLD' and root['continuous_defect']['filter_state_check']['positive']
        # The exact root belongs between original arclength samples 76 and
        # 77. Retain this order even though J itself turns at the fold.
        where=next(i for i,q in enumerate(fs['A'][:stop+1]) if Path(q['path']).stem=='arcAreturnStrong_0076_N512')
        selected['A'].insert(where+1,read(Path(root['mesh_checks'][-1]['orbit']).with_suffix('.json')))
    inserted_H1=['LPC_A_return_recruitment'] if extended else []
    if last>=139:
        pd3=read(PERIODIC_OUT/'PD_A_return_validation.json')
        assert pd3['full_acceptance'] and pd3['criticality']=='SUPERCRITICAL_PD'
        origin=str(PERIODIC_OUT/'orbits/arcAreturnStrong_0138_N512.npz')
        where=next(i for i,q in enumerate(selected['A']) if Path(q['path']).resolve()==Path(mapping[origin]).resolve())
        point=read(Path(pd3['accepted_parent_orbit']).with_suffix('.json'))
        assert selected['A'][where+1]['J_EE_core']<point['J_EE_core']<selected['A'][where]['J_EE_core']
        selected['A'].insert(where+1,point)
        base.EXTRA_RETURN_FOLDS.append('PD_A_return');inserted_H1.append('PD_A_return')
    h2_return_file=DATA/'H2_display_return.json'
    h2_return=read(h2_return_file) if h2_return_file.exists() else None
    if h2_return and h2_return.get('status')=='PASS':
        assert len(h2_return['rows'])==113
        assert all(q['status']=='PASS' and q['branch_match_pass'] and
                   q['check']['filter_state_check']['positive'] for q in h2_return['rows'])
        b_rows=[read(Path(q['orbit']).with_suffix('.json')) for q in h2_return['rows']]
        assert Path(h2_return['rows'][110]['source']).stem=='arcB_0101_N64'
        local=read(DATA/'H2_fold_neighborhood/dense_geometry_checks.json')
        assert local['status']=='PASS'
        extra=[read(Path(q['orbit']).with_suffix('.json')) for q in local['rows']]
        fold=read(PERIODIC_OUT/'LPC_B2_validation.json')
        assert fold['status']=='VALIDATED_CYCLE_FOLD'
        extra.append(read(Path(fold['mesh_checks'][-1]['orbit']).with_suffix('.json')))
        pd_file=PERIODIC_OUT/'PD_H2_after_LPC13_validation.json'
        if pd_file.exists() and read(pd_file).get('full_acceptance',False):
            extra.append(read(Path(read(pd_file)['accepted_parent_orbit']).with_suffix('.json')))
        # The mean is monotone only on this local continuation segment;
        # retain the original arclength order everywhere else, never J-sort.
        tail={q['path']:q for q in b_rows[111:]+extra}
        tail=sorted(tail.values(),key=lambda q:q['mean_rates_hz'][1])
        assert b_rows[110]['mean_rates_hz'][1]<tail[0]['mean_rates_hz'][1]
        selected['B']=b_rows[:111]+tail
        base.H2_RETURN_ROWS=selected['B'];base.NAMES['B']='H2 periodic family'
        mapping.update({q['source']:q['orbit'] for q in h2_return['rows']})
    target_corrections=[]
    target_file=DATA/'Bleading_extension/nearest_target_refinement.json'
    if target_file.exists():
        for row in read(target_file).get('rows',[]):
            if row.get('status')!='MATCHED_TARGET_RECHECKED':continue
            assert row['same_branch_refinement_pass']
            assert row['resolution']['filter_state_check']['positive']
            assert row['resolution']['maximum_group_defect_Hz']<.001
            source=row['original_pair']['second_orbit'];actual=row['corrected_target']
            family=row['family'];mapping[source]=actual
            for index,old in enumerate(selected[family]):
                if Path(old['path']).resolve()==Path(source).resolve():
                    fine=read(Path(actual).with_suffix('.json'))
                    assert abs(fine['J_EE_core']-old['J_EE_core'])<1e-12
                    selected[family][index]=fine
            target_corrections.append(dict(source=source,corrected_orbit=actual,family=family))
    for file in sorted((DATA/'Aleading_profile_gaps').glob('corrected_site_*.json')):
        row=read(file)
        if row.get('status')!='SAME_J_PHYSICAL_PROFILE_CHECKED':continue
        check=row['resolution']
        assert check['filter_state_check']['positive'] and check['maximum_group_defect_Hz']<.001
        assert row['relative_waveform_change']<.02 and row['relative_period_change']<.001
        source,actual=row['original_orbit'],row['orbit'];mapping[source]=actual
        for index,old in enumerate(selected['single']):
            if Path(old['path']).resolve()==Path(source).resolve():
                fine=read(Path(actual).with_suffix('.json'))
                assert abs(fine['J_EE_core']-old['J_EE_core'])<1e-12
                selected['single'][index]=fine
        target_corrections.append(dict(source=source,corrected_orbit=actual,family='single',evidence=str(file)))
    for source,actual in mapping.items():
        original=str(Path(source).resolve())
        if original in ev:ev[str(Path(actual).resolve())]=ev[original]
    base.NAMES['A']='H1 periodic family'
    base.composite(s,selected,ev,cases,log_readout=True)
    base.standalone(selected,ev,cases)
    base.local_onset(fs,ev)
    if extended:
        h1_burst_projection(selected,ev)
        if pd3_prefix:h1_burst_projection(selected,ev,wide=True)
    folds=fold_detail(s,rows,fs,confirmed=not a.snapshot);extension=h2_detail(fs)
    write(OUTPUT/'figure_metadata.json',dict(producer=str(Path(__file__).resolve()),
        source_model='Frozen 400-cell / 935-population rate DDE',new_sites=rows,
        snapshot=bool(a.snapshot),sameJ_alternating_burst_stability=ccheck['status'],
        displayed_H1_route_check=str(previous/'displayed_H1_path_continuous_check.json'),
        displayed_H1_points=len(selected['A']),
        displayed_H1_continuation_samples=stop+1,
        inserted_H1_critical_points=inserted_H1,
        exact_additional_stability_samples=base.EXACT_STABILITY_SAMPLES,
        displayed_H1_last_return_index=last,
        H1_to_PD3_displayed_prefix_points=len(pd3_prefix),
        H1_to_PD3_display_source=str(pd3_extension_file) if pd3_prefix else None,
        H1_frozen_observer_segment=(str(DATA/'H1_extension_observer_summary.json')
            if (DATA/'H1_extension_observer_summary.json').exists() else None),
        H1_frozen_observer_later_segment=(str(DATA/'H1_later_observer_summary.json')
            if (DATA/'H1_later_observer_summary.json').exists() else None),
        same_J_spatial_pair_readouts=(str(DATA/'Bleading_extension/sameJ_pair_readouts.json')
            if (DATA/'Bleading_extension/sameJ_pair_readouts.json').exists() else None),
        same_J_small_and_burst_local_stability=(str(DATA/'sameJ_small_burst_stability_readout.json')
            if (DATA/'sameJ_small_burst_stability_readout.json').exists() else None),
        SCL_checked_branch_readouts=(str(DATA/'SCL_branch_scan/summary.json')
            if (DATA/'SCL_branch_scan/summary.json').exists() else None),
        SCL_identical_orbit_stability_readouts=(str(DATA/'SCL_branch_scan/stability_conditioned_samples.json')
            if (DATA/'SCL_branch_scan/stability_conditioned_samples.json').exists() else None),
        Bleading_return_stability_witness=(str(DATA/'Bleading_extension/return_witness24/result.json')
            if (DATA/'Bleading_extension/return_witness24/result.json').exists() else None),
        finite_period_ratio_connection_screens=next((str(DATA/name) for name in
            ['period_multiple_1_to_8_connection_evidence.json',
             'period_multiple_connection_evidence_1_to_8.json'] if (DATA/name).exists()),None),
        TR2_full_state_direction_diagnostic=next((str(DATA/name) for name in
            ['TR2_saddle_directions/exactJ_diagnostic/checks.json','TR2_saddle_directions/checks.json']
            if (DATA/name).exists()),None),
        PD1_parent_and_child_orientation=(str(DATA/'PD1_parent_witnesses/local_child_orientation.json')
            if (DATA/'PD1_parent_witnesses/local_child_orientation.json').exists() else None),
        PD4_current_child_spectrum_evidence=(str(DATA/'H2_local_PD/current_child_spectrum_evidence.json')
            if (DATA/'H2_local_PD/current_child_spectrum_evidence.json').exists() else None),
        PD3_current_parent_spectrum_evidence=(str(DATA/'PD3_parent_spectra/current_evidence.json')
            if (DATA/'PD3_parent_spectra/current_evidence.json').exists() else None),
        Bleading_new_segment_integer_encounters=[str(path) for path in sorted(
            (DATA/'Bleading_extension').glob('prefix*_integer_encounter_bounds.json'))
            if read(path).get('status')=='NEW_PREFIX_FINITE_WINDOW_PAIRS_BOUNDED'],
        PD4_sampled_spectral_route=(str(DATA/'H2_local_PD/spectral_route_evidence.json')
            if (DATA/'H2_local_PD/spectral_route_evidence.json').exists() else None),
        Bleading_checked_extension=(read(PERIODIC_OUT/'arcBleadingConnection_20260920_accuracy.json')
            if (PERIODIC_OUT/'arcBleadingConnection_20260920_accuracy.json').exists() else None),
        additional_same_J_profile_corrections=target_corrections,
        displayed_H1_extension_check=str(extension_file) if extended else None,
        fold_evidence=folds,H2_extension=extension,
        root_spectrum_assessments=[str(path) for label in ['LPC_Bleading_low','LPC_burst_low','LPC_B2']
            if (path:=DATA/'primary_folds'/f'{label}_root_spectrum_assessment.json').exists()],
        period_doubling_checks={name:read(PERIODIC_OUT/(name+'_validation.json'))
            for name in ['PD_double_low','PD_double_upper','PD_A_return','PD_H2_after_LPC13']
            if (PERIODIC_OUT/(name+'_validation.json')).exists()},
        H2_displayed_return_source=str(h2_return_file) if base.H2_RETURN_ROWS is not None else None,
        H2_displayed_return_points=len(selected['B']),
        delay_step_correction=str(DATA/'delay_step_audit.json'),
        line_semantics='Dotted curves are numerical continuation estimates. Several burst profiles still require temporal refinement and intervals lack exhaustive stability classification. Exact filled circles/crosses denote physically checked stable/unstable samples; neither missing physical checks nor missing interval spectra are inferred from the connecting line. The right-hand representative cases have separate physical checks.',
        pending_Aleading_temporal_profiles=(str(DATA/'Aleading_profile_gaps/summary.json')
            if (DATA/'Aleading_profile_gaps/summary.json').exists() else None),
        current_paired_stability_intervals=(str(DATA/'current_interval_evidence.json')
            if (DATA/'current_interval_evidence.json').exists() else None),
        current_interval_root_associations=(str(DATA/'current_interval_root_associations.json')
            if (DATA/'current_interval_root_associations.json').exists() else None),
        current_interval_association_coverage=(str(DATA/'current_interval_association_coverage.json')
            if (DATA/'current_interval_association_coverage.json').exists() else None),
        interval_root_temporal_verification=(str(DATA/'interval_root_temporal_verification.json')
            if (DATA/'interval_root_temporal_verification.json').exists() else None),
        located_fold_evidence_inventory=(str(DATA/'current_fold_inventory.json')
            if (DATA/'current_fold_inventory.json').exists() else None),
        primary_burst_fold_shape_modes=(str(PERIODIC_OUT/'primary_burst_fold_shape_modes.json')
            if (PERIODIC_OUT/'primary_burst_fold_shape_modes.json').exists() else None),
        LPC6_branch_side_correction=(str(DATA/'LPC6_branch_side_audit.json')
            if (DATA/'LPC6_branch_side_audit.json').exists() else None),
        readout='Same rate solutions at fixed SEEG contact locations; not electrical voltage',
        global_branch_completeness=False,human_visual_acceptance='PENDING'))
    entries={
      'spatial_rate_focused_composite':f'沿用 a–e 的全 rate 排版，显示已逐点检查的 {len(selected["A"])} 个 H1 返回分支点，并更新通过正值波形及有效延迟步长复核的稳定性采样点。右侧仍由同一空间 rate 方程生成波形、二维场和固定触点读出。**关注点**：虚点线不表示已经分类的稳定区间，触点图是率读出。',
      'primary_bifurcation_core_A_B':'单独放大合图的两核周期均值分岔图，包含 H1 向低参数折返后再次返回的已检查路径，点位和稳定性证据与合图相同。保留真实折返，不连接未经确认的不同分支。**关注点**：均值曲线相交不是解在完整空间中连接。',
      'burst_fold_stability_and_spatial_modes':'分别展示 B-leading 与 A-leading burst 的低参数折返、两侧非平凡 Floquet 乘子和去除相位分量后的临界空间模态。折点经过细化周期解、全延迟变分方程及时间步长收敛复核。**关注点**：模态图只对应参考相位，不能直接视为整周期的空间因果贡献。',
      'H2_extended_branch_coverage':'展示 H2 家族的延续路径、新增的已检查延续段，以及完成双网格定位的周期折返点。LPC13 标为已不稳定周期解上的折叠，其非平凡谱见 H2_unstable_fold_spectrum；空心方块表示独立临界模态检查尚未完成。**关注点**：折点位置、完整局部验证和分支稳定性区间分别判断；端点短线仅代表计算停止位置。'}
    entries['onset_bifurcation_detail']='放大低参数区间，把新增的稳定和不稳定周期采样点放回原分支。空心折点仍保留待完成的物理波形检查标记。**关注点**：采样点之间仍不能推断没有遗漏的分岔。'
    if base.H2_RETURN_ROWS is not None:
        entries['spatial_rate_focused_composite']=f'沿用 a–e 的全 rate 排版，显示 {len(selected["A"])} 个 H1 返回分支点和 {len(selected["B"])} 个 H2 分支点，H2 已延伸到 LPC13 之后。新增 H2 路径经逐点同参数波形、滤波状态及离网格方程检查，折点附近加密点按同一分支的 Core B 均值顺序插入。**关注点**：右侧仍是同一空间 rate 方程的波形、二维场和触点率读出；点线不表示整段稳定性已经分类。'
        entries['onset_bifurcation_detail']='放大低参数区间，H2 沿已检查的返回路径跨过 LPC13，不再只截到第一个折返。近邻临界点的位置与稳定性分别按各自数值验证标记。**关注点**：靠得很近的临界点须结合 H2 局部图阅读；样本间仍不声明分岔完备。'
    if extended:
        entries['H1_return_and_burst_projection']=f'放大 H1 返回路径与主要 burst 分支所处的同一参数窗，同时比较两核周期均值和全网络周期。返回路径已逐点检查到第 {last} 点，新增解均经过同参数波形细化、滤波状态正值性及离网格方程检查；LPC27 按延续顺序插入原第 76 与 77 点之间。**关注点**：均值投影接近不表示周期轨道已经连接；虚点线仍未获得整个区间的稳定性分类。'
        if pd3_prefix:
            entries['H1_extension_and_burst_comparison']=f'把比较窗口扩到 H1 已核验返回路径的第 {last} 点，同时显示 Core A、Core B 的周期均值以及完整网络周期。保留各条延拓曲线自身的顺序，展示均值曲线靠近或相交时，另一核心和周期是否仍然分离。**关注点**：这是现有分支在相同坐标中的比较；不能凭二维曲线交叠认定全空间解已连接，也不能把未分类段视为稳定吸引子。'
    if a.snapshot:
        entries.pop('burst_fold_stability_and_spatial_modes')
        entries['burst_fold_neighborhood_evidence']='左列显示两个 burst 折点的最新坐标及已检查的两侧周期解，中列显示返回乘子。已完成全部局部检查的行显示实心折点及其参考相位临界模态，待检查的行保留空心折点，右侧显示附近不稳定周期解的特征向量；两种空间图分别明确标注。**关注点**：邻近不稳定模态不能替代折点的临界模态，采样点之间的完整稳定性区间仍未确认。'
    entries['spatial_rate_focused_composite']=entries['spatial_rate_focused_composite'].replace(
        '**关注点**：', '部分 burst 分支仍使用待时间网格细化的延续估计，不能当作整段已核验的物理解。**关注点**：')
    if base.EXACT_STABILITY_SAMPLES:
        for name in ['spatial_rate_focused_composite','primary_bifurcation_core_A_B','onset_bifurcation_detail']:
            entries[name]=entries[name].replace('**关注点**：',
                'PD1 与 PD3 两侧的稳定／不稳定样本按各自精确参数标记，不改变延拓曲线顺序。**关注点**：')
    path=OUTPUT/'figures/README.md';txt=path.read_text() if path.exists() else ''
    for name,body in entries.items():
        pattern=r'^### '+re.escape(name)+r'(?:\.png)?\s*\n.*?(?=^### |\Z)'
        txt=re.sub(pattern,'',txt,flags=re.M|re.S).rstrip()
        txt+='\n\n### '+name+'.png\n'+body+'\n'
    path.write_text(txt.lstrip())


if __name__=='__main__':main()
