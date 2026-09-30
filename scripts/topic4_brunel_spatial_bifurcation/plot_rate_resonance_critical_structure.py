"""Separate cycle folds from complex unit-circle crossings in the H1 return."""
from plot_rate_branch_completion import *
from audit_rate_filter_states import filter_state_minima


def main():
    s=RateField();route=Path('/data/hfosp/topic4_sef_hfo/interictal_rate_small_burst_connection_20260919')
    assert read(route/'displayed_H1_path_continuous_check.json')['status']=='PASS'
    correction=read(route/'displayed_H1_path_refinement.json');assert correction['status']=='COMPLETE'
    mapping={q['source']:q['orbit'] for q in correction['rows']}
    rows=[q for q in families()['A'] if Path(q['path']).stem.startswith('resonanceA_')]
    rows=[read(Path(mapping.get(q['path'],q['path'])).with_suffix('.json')) for q in rows]
    criticals=[]
    for name,label in [('LPC_resonance_upper','LPC1'),('LPC_resonance_lower','LPC2'),
                       ('TR_A_B','TR1'),('TR_A_return','TR2')]:
        source=PERIODIC_OUT/(name+'_validation.json');v=read(source)
        fold=label.startswith('LPC');root=v['mesh_checks'][-1] if fold else v['critical_point']
        z=np.load(root['orbit']);physical=filter_state_minima(s,z['r'],float(z['T']))
        assert physical['positive']
        if fold:assert v['status']=='VALIDATED_CYCLE_FOLD'
        else:
            assert root['eigen_residual']<1e-9
            assert abs(complex(*root['multiplier'])-1)>1e-3
            assert len(v['independent_full_state_mode_checks'])>=2
        meta=read(Path(root['orbit']).with_suffix('.json'))
        point=dict(name=name,label=label,source=str(source),validation=v,root=root,
            mean_B=meta['mean_rates_hz'][1],physical=physical,fold=fold)
        noncritical_source=DATA/'TR1_root_spectrum/result.json'
        if label=='TR1' and noncritical_source.exists():
            spectrum=read(noncritical_source)
            assert Path(spectrum['orbit']).resolve()==Path(root['orbit']).resolve()
            if spectrum['status']=='NONCRITICAL_SPECTRUM_RESOLVED':
                assert spectrum['physical']['filter_state_check']['positive']
                assert spectrum['paired_spectrum']['filter_coverage']
                point['noncritical_spectrum_source']=str(noncritical_source)
                point['noncritical_unstable_dimension']=spectrum['noncritical_unstable_dimension']
                point['noncritical_spectrum_scope']=spectrum['scope']
        criticals.append(point)
    samples=[]
    for index in [22,24,25,26,27,28]:
        path=DATA/'sites'/f'{index:03d}.json'
        if not path.exists():continue
        q=read(path)
        if q.get('classification',{}).get('numerical_unstable_dimension') is None:continue
        assert q['resolution']['filter_state_check']['positive']
        samples.append(dict(index=index,source=str(path),result=q,
            meta=read(Path(q['analyzed_orbit']).with_suffix('.json'))))
    center=.9458
    plotted_samples=[q for q in samples if -66<=(q['meta']['J_EE_core']-center)*1e6<=64
                     and .7431<=q['meta']['mean_rates_hz'][1]<=.7477]
    sample_styles={0:('o','#168469','Stable cycle sample'),
                   1:('x','#cf593c','Cycle: 1 unstable direction'),
                   2:('+','#2f6ca3','Cycle: 2 unstable directions')}
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(14.4,5.0))
    fig.subplots_adjust(left=.065,right=.98,bottom=.23,top=.85,wspace=.42)
    ax=axes[0]
    ax.plot([(q['J_EE_core']-center)*1e6 for q in rows],
        [q['mean_rates_hz'][1] for q in rows],color=FAMILY['A'],ls=LINESTYLE,lw=1.4)
    offsets={'LPC1':(-42,14),'LPC2':(-12,20),'TR1':(-57,-24),'TR2':(42,14)}
    colors={'TR1':'#ae4c1c','TR2':'#7054a1'}
    for q in criticals:
        xy=((q['root']['J_EE_core']-center)*1e6,q['mean_B'])
        color='black' if q['fold'] else colors[q['label']]
        ax.plot(*xy,'s' if q['fold'] else 'D',ms=5,color=color,zorder=6)
        ax.annotate(q['label'],xy,xytext=offsets[q['label']],textcoords='offset points',
            color=color,arrowprops=dict(arrowstyle='-',lw=.7,color=color))
    for sample in plotted_samples:
        q=sample['result'];meta=sample['meta']
        marker,color,_=sample_styles[q['classification']['numerical_unstable_dimension']]
        ax.plot((meta['J_EE_core']-center)*1e6,meta['mean_rates_hz'][1],
            marker,ms=5,color=color,zorder=5)
    ax.set(xlim=(-66,64),ylim=(.7431,.7477),xlabel=r'$10^6(J_{\mathrm{EE,core}}-0.94580)$',
        ylabel='Core B period mean (Hz / E cell)',title='A  Distinct crossings on one family')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax=axes[1];theta=np.linspace(-.012,.012,800)
    ax.plot((np.cos(theta)-1)*1e3,np.sin(theta)*1e3,'--',color='black',lw=.8)
    ax.plot(0,0,'s',ms=5,color='black')
    ax.annotate('Fold: +1',(0,0),xytext=(12,0),textcoords='offset points',fontsize=9)
    for q in criticals:
        if q['fold']:continue
        mu=complex(*q['root']['multiplier'])
        for sign in [-1,1]:ax.plot((mu.real-1)*1e3,sign*abs(mu.imag)*1e3,'D',ms=5,color=colors[q['label']])
        offset=(33,13) if q['label']=='TR1' else (15,1)
        ax.annotate(q['label'],((mu.real-1)*1e3,mu.imag*1e3),xytext=offset,
                    textcoords='offset points',color=colors[q['label']],
                    arrowprops=dict(arrowstyle='-',lw=.6,color=colors[q['label']]))
    for sample in plotted_samples:
        q=sample['result']['classification'];mu=values(q)
        marker,color,_=sample_styles[q['numerical_unstable_dimension']]
        for val in mu[abs(mu-1)<.015]:
            ax.plot((val.real-1)*1e3,val.imag*1e3,marker,ms=5,color=color)
    ax.set(xlim=(-5,5),ylim=(-9,9),xlabel=r'$10^3(\mathrm{Re}\,\mu-1)$',
        ylabel=r'$10^3\mathrm{Im}\,\mu$',title='B  Nontrivial Floquet multipliers')
    ax=axes[2]
    for q in criticals:
        if q['fold']:continue
        checks=sorted(q['validation']['independent_full_state_mode_checks'],key=lambda x:-x['dt_ms'])
        dt=np.array([r['dt_ms'] for r in checks]);err=np.array([r['full_state_mode_relative_defect'] for r in checks])
        ax.loglog(dt,err,'o-',color=colors[q['label']],label=q['label'])
    ax.set(xlabel='Integration step (ms)',ylabel=r'$\|Mv-\mu v\|/\|v\|$',
        title='C  Independent delay propagation')
    ax.set_xticks([.05,.1]);ax.set_xticklabels(['0.05','0.1']);ax.minorticks_off()
    ax.legend(frameon=False)
    for ax in axes:style(ax)
    fig.suptitle('Two cycle folds and two torus crossings',fontsize=14,y=.97)
    handles=[Line2D([0],[0],color=FAMILY['A'],ls=LINESTYLE,label='Continuation geometry'),
        Line2D([0],[0],marker='s',color='black',ls='',label='Checked cycle fold'),
        Line2D([0],[0],marker='D',color=colors['TR2'],ls='',label='Checked torus crossing')]
    for dimension,(marker,color,label) in sample_styles.items():
        if any(q['result']['classification']['numerical_unstable_dimension']==dimension for q in plotted_samples):
            handles.append(Line2D([0],[0],marker=marker,color=color,ls='',label=label))
    fig.legend(handles=handles,
        loc='lower center',bbox_to_anchor=(.5,.025),ncol=3,frameon=False,fontsize=9)
    name='H1_resonance_fold_torus_structure';save_new(fig,name)
    sequence=sorted(samples,key=lambda q:next(v['index'] for v in q['result']['memberships'] if v['family']=='A'))
    write(DATA/(name+'.json'),dict(critical_points=criticals,paired_spectrum_samples=samples,
        plotted_sample_indices=[q['index'] for q in plotted_samples],
        sampled_sequence_in_continuation_order=[dict(index=q['index'],J_EE_core=q['meta']['J_EE_core'],
            numerical_unstable_dimension=q['result']['classification']['numerical_unstable_dimension'],
            source=q['source']) for q in sequence],
        scope='Known, separately checked local bifurcations on the same H1 periodic family. Filled samples require positive physical profiles and numerical spectrum coverage; no line-wide stability or stable finite torus is asserted.',
        interval_completeness=False))
    path=OUTPUT/'figures/README.md';text=path.read_text()
    text=re.sub(r'^### '+name+r'\.png\s*\n.*?(?=^### |\Z)','',text,flags=re.M|re.S).rstrip()
    text+='\n\n### '+name+'.png\n放大 H1 在 H2 附近折返的局部结构，区分 LPC1／LPC2 周期折叠与 TR1／TR2 环面分岔；右侧展示对应非平凡 Floquet 乘子及独立延迟传播检查。稳定性符号仅用于已通过物理波形和谱覆盖检查的采样点，点线只表示延续路径。**关注点**：图中不是所有临界点都属于 fold 或 PD；环面分岔本身也不能证明稳定 irregular burst 或传播模板切换。\n'
    path.write_text(text)


if __name__=='__main__':main()
