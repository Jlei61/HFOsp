"""Show why J below a cycle fold does not identify its incoming branch."""
from plot_rate_branch_completion import *
from validate_rate_mean_fold import validation_matches_latest_root


def main():
    label='LPC_burst_high';assert validation_matches_latest_root(label)
    validation=read(PERIODIC_OUT/(label+'_validation.json'))
    root=validation['mesh_checks'][-1]
    assert validation['continuous_defect']['filter_state_check']['positive']
    fs=families()['single'];plan=read(DATA/'frozen_survey_plan.json')['rows']
    locations={i:next(k for k,q in enumerate(fs) if Path(q['path']).resolve()==Path(plan[i]['orbit']).resolve())
               for i in [41,40]}
    association=read(DATA/'current_interval_root_associations.json')
    interval=next(q for q in association['rows'] if q['family']=='single' and
                  q['ends'][-1]['site_index']==40 and
                  q['ends'][0]['status']=='NUMERICALLY_STABLE')
    match=next(q for q in interval['nearby_roots'] if q['internal_label']==label)
    assert match['within_sampled_polyline_neighborhood']
    coordinate=match['nearest_segment']['left_index']+match['nearest_segment']['fraction']
    temporal_source=DATA/'interval_root_temporal_verification.json'
    if temporal_source.exists():
        refined=next(q for q in read(temporal_source)['rows'] if q['internal_label']==label)
        assert refined['association_retained']
        assert Path(refined['orbits'][0]).resolve()==Path(root['orbit']).resolve()
        coordinate=refined['selected_left_continuation_index']+refined['fine']['fraction']
    assert locations[41]<coordinate<locations[40]
    current_source=DATA/'current_interval_evidence.json'
    current={q['site_index']:q for q in read(current_source)['sites']}
    records=[]
    for index in [41,40]:
        source=Path(plan[index]['orbit']);fine=DATA/'Aleading_profile_gaps'/f'corrected_site_{index:03d}.json'
        site=DATA/'sites'/f'{index:03d}.json';physical=False;status='PENDING';evidence=None
        if site.exists():
            q=read(site)
            if q.get('resolution',{}).get('filter_state_check',{}).get('positive',False):
                source=Path(q['analyzed_orbit']);physical=True;status=q['status'];evidence=str(site)
        if fine.exists() and not physical:
            q=read(fine);assert q['status']=='SAME_J_PHYSICAL_PROFILE_CHECKED'
            source=Path(q['orbit']);physical=True;evidence=str(fine)
        z=np.load(source);meta=read(source.with_suffix('.json'))
        paired=current.get(index)
        dimension=None
        if paired is not None:
            assert Path(paired['analyzed_orbit']).resolve()==source.resolve()
            assert paired['status']==status
            dimension=paired['numerical_unstable_dimension']
        records.append(dict(site_index=index,continuation_index=locations[index],orbit=str(source),
            J_EE_core=float(z['J']),T_ms=float(z['T']),mean_rates_hz=meta['mean_rates_hz'],
            physical_profile_checked=physical,stability=status,evidence=evidence,
            numerical_unstable_dimension=dimension,paired_evidence_source=str(current_source),
            branch_side='incoming' if locations[index]<coordinate else 'returning'))
    assert records[1]['stability']=='UNSTABLE'
    assert all(q['J_EE_core']<root['J_EE_core'] for q in records)
    before,after=records;assert before['T_ms']>root['T_ms']>after['T_ms']
    meta=read(Path(root['orbit']).with_suffix('.json'))
    path=[dict(q) for q in fs[locations[41]:locations[40]+2]]
    for q in records:
        path[q['continuation_index']-locations[41]]=dict(path[q['continuation_index']-locations[41]],
            J_EE_core=q['J_EE_core'],T_ms=q['T_ms'],mean_rates_hz=q['mean_rates_hz'])
    path.insert(int(coordinate)-locations[41]+1,meta)
    incoming_checked=before['stability'] in ['NUMERICALLY_STABLE','UNSTABLE']
    interval_scope=('Both displayed samples have paired stability classifications.' if incoming_checked
                    else 'The incoming sample still needs its paired spectrum.')
    evidence=dict(status='BRANCH_SIDE_CORRECTION_CHECKED',root_source=str(PERIODIC_OUT/(label+'_validation.json')),
        root_J_EE_core=root['J_EE_core'],root_T_ms=root['T_ms'],
        full_waveform_association_source=str(DATA/'current_interval_root_associations.json'),
        temporal_association_verification_source=str(temporal_source) if temporal_source.exists() else None,
        root_continuation_coordinate=coordinate,records=records,
        correction='Site 40 is on the returning branch AFTER LPC6 in continuation order. Its J lies below the maximum J at the fold; this does not make it a pre-fold point. Site 40 does not establish an additional instability before LPC6.',
        existing_file_alias='Historical LPC6_left_site40 names mean smaller J only, not the incoming branch.',
        scope='Known LPC6 lies within the current sampled stable-to-unstable bracket. '+interval_scope+' This does not exclude additional crossings before or after it.',
        global_branch_completeness=False)
    write(DATA/'LPC6_branch_side_audit.json',evidence)
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(11.4,5.1))
    fig.subplots_adjust(left=.085,right=.985,bottom=.25,top=.89,wspace=.30)
    for k,ax in enumerate(axes):
        yy=[q['mean_rates_hz'][0] if k==0 else q['T_ms'] for q in path]
        xx=[q['J_EE_core'] for q in path]
        ax.plot(xx,yy,color='#cf7f00',ls=(0,(1.3,2.1)),lw=1.6)
        for left,right in [(0,1),(len(path)-3,len(path)-2)]:
            ax.annotate('',xy=(xx[right],yy[right]),xytext=(xx[left],yy[left]),
                arrowprops=dict(arrowstyle='->',color='#cf7f00',lw=1.2))
        for q in records:
            val=q['mean_rates_hz'][0] if k==0 else q['T_ms']
            incoming=q['branch_side']=='incoming'
            stable=q['stability']=='NUMERICALLY_STABLE'
            unstable=q['stability']=='UNSTABLE'
            color='#2475ae' if incoming else '#ba3333'
            ax.plot(q['J_EE_core'],val,'x' if unstable else 'o',
                mfc=color if stable else 'white',color=color,ms=7,mew=1.5)
            text=('Incoming branch' if incoming else 'Returning branch')+'\n'+(
                'stable sample' if stable else
                ('1 unstable direction' if q['numerical_unstable_dimension']==1 else 'unstable')
                if unstable else 'stability pending')
            offset=((8,62) if k==0 else (7,-38)) if incoming else ((-35,17) if k==0 else (-40,-35))
            ax.annotate(text,(q['J_EE_core'],val),xytext=offset,
                textcoords='offset points',ha='left' if incoming else 'right',fontsize=10,
                arrowprops=dict(arrowstyle='-',color='black',lw=.65))
        val=meta['mean_rates_hz'][0] if k==0 else root['T_ms']
        ax.plot(root['J_EE_core'],val,'s',color='black',ms=6)
        ax.annotate('LPC6',(root['J_EE_core'],val),xytext=(-8,12),textcoords='offset points',ha='right')
        ax.set(xlim=(1.6250,1.62665),xlabel=r'$J_{\mathrm{EE,core}}$',
            ylabel='Period mean (Hz / E cell)' if k==0 else 'Period (ms)',
            title='Core A' if k==0 else 'Full-network periodic orbit')
        if k==0:ax.set_ylim(74.2,80.0)
        ax.xaxis.set_major_locator(FixedLocator([1.6250,1.6255,1.6260,1.6265]))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'));style(ax)
    handles=[Line2D([0],[0],color='#cf7f00',ls=(0,(1.3,2.1)),label='Continuation estimate; arrows show continuation order'),
        Line2D([0],[0],color='black',marker='s',ls='',label='Validated cycle fold')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.03),ncol=1,frameon=False,fontsize=10)
    save_new(fig,'LPC6_continuation_order')
    file=OUTPUT/'figures/README.md';body=file.read_text()
    name='LPC6_continuation_order.png'
    body=re.sub(r'^### '+re.escape(name)+r'\s*\n.*?(?=^### |\Z)','',body,flags=re.M|re.S).rstrip()
    incoming_text=('进入支样本已通过配对步长稳定性检查，标记反映该样本的分类' if incoming_checked
                   else '进入支样本的稳定性仍待计算')
    body+='\n\n### '+name+'\n展示 LPC6 附近按实际延续顺序排列的两条周期支，用 Core A 均值和全网络周期区分进入支与返回支。两个样本的 J 均低于折点，但编号 41 在折返前、编号 40 在折返后；后者已确认不稳定，不能据此推断折点之前另有一次失稳。**关注点**：点线是延续估计，'+incoming_text+'；已知折点在稳定性变化区间内不等于排除额外分岔。\n'
    file.write_text(body)
    print('LPC6_SIDES',[(q['site_index'],q['branch_side'],q['T_ms'],q['stability']) for q in records],flush=True)


if __name__=='__main__':main()
