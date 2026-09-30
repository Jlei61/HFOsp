"""Separate individual contact recruitment from eligible network events."""
from plot_rate_branch_completion import *
from matplotlib.colors import TwoSlopeNorm


def main():
    sources=[DATA/'H1_extension_observer_summary.json',DATA/'H1_later_observer_summary.json']
    summaries=[read(p) for p in sources]
    assert all(q['status']=='FROZEN_OBSERVER_SEGMENT_COMPLETE' for q in summaries)
    rows=[read(r['source']) for q in summaries for r in q['rows']]
    assert [r['index'] for r in rows]==list(range(85,112))
    names=summaries[0]['contact_names'];order=contact_indices(names)
    assert all(q['contact_names']==names for q in summaries)
    assert all(r['qualified_events']==0 and r['detected_events']==0
               for q in rows for r in q['records'])
    J=np.array([q['J_EE_core'] for q in rows]);assert np.all(np.diff(J)>0)
    regional=np.array([np.max([r['regional_peak_observer_filtered_Hz'] for r in q['records']],axis=0) for q in rows])
    contact=np.array([np.max([r['smoothed_contact_peak_Hz'] for r in q['records']],axis=0) for q in rows])
    ratio=np.array([np.max([r['contact_peak_to_threshold'] for r in q['records']],axis=0) for q in rows])
    counts=np.array([max(r['maximum_unique_contacts_in_group_window'] for r in q['records']) for q in rows])
    required=rows[0]['records'][0]['required_unique_contacts']
    contract=read(summaries[0]['observer']);threshold=np.asarray(contract['threshold'])*500
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(15,5.5))
    grid=fig.add_gridspec(1,3,width_ratios=[1.05,1.25,.9],left=.055,right=.97,bottom=.20,top=.85,wspace=.42)
    ax=fig.add_subplot(grid[0])
    for k,color in enumerate(COL):ax.plot(J,regional[:,k],color=color,lw=1.6,label=f'Core {"AB"[k]}')
    ax.plot(J,contact.max(1),color='#333333',lw=1.6,label='Largest contact rate')
    ax.axhspan(threshold.min(),threshold.max(),color='#e8bf83',alpha=.55,label='Contact thresholds')
    ax.set(title='A   Field and contact peaks',ylabel='Filtered peak (Hz / cell)',xlabel=r'$J_{\mathrm{EE,core}}$')
    ax.legend(frameon=False,fontsize=8,loc='upper left');style(ax)
    ax=fig.add_subplot(grid[1])
    edges=np.r_[J[0]-(J[1]-J[0])/2,(J[:-1]+J[1:])/2,J[-1]+(J[-1]-J[-2])/2]
    im=ax.pcolormesh(edges,np.arange(16)-.5,ratio[:,order].T,cmap='RdBu_r',
        norm=TwoSlopeNorm(vmin=0,vcenter=1,vmax=max(1.5,float(ratio.max()))),shading='flat')
    ax.set(ylim=(14.5,-.5),yticks=np.arange(15),yticklabels=CONTACT_ORDER,
        xlabel=r'$J_{\mathrm{EE,core}}$',title='B   Individual contact recruitment')
    ax.axhline(3.5,color='black',lw=.6);ax.tick_params(axis='y',labelsize=8,length=2)
    for tick,name in zip(ax.get_yticklabels(),CONTACT_ORDER):tick.set_color(SHAFT_COLORS[name[:3]])
    fig.colorbar(im,ax=ax,orientation='horizontal',pad=.18,fraction=.05,
        label='Peak / threshold; red exceeds 1')
    ax=fig.add_subplot(grid[2])
    ax.step(J,counts,where='mid',color='#4f6192',lw=1.6,label='Participating contacts')
    ax.axhline(required,color='black',ls='--',lw=1,label=f'Group requirement ({required})')
    ax.set(ylim=(-.2,required+.8),yticks=[0,2,4,6,8],xlabel=r'$J_{\mathrm{EE,core}}$',
        ylabel='Contacts in expanded detection window',title='C   Group-event eligibility')
    ax.legend(frameon=False,fontsize=8,loc='center left',bbox_to_anchor=(0,.78));style(ax)
    fig.suptitle('H1 continuation: individual contact thresholds do not define a dynamical bifurcation',fontsize=12,y=.96)
    name='H1_contact_recruitment_extension';save_new(fig,name)
    write(DATA/(name+'.json'),dict(sources=list(map(str,sources)),indices=[r['index'] for r in rows],
        J_range=[float(J.min()),float(J.max())],maximum_peak_threshold_ratio=float(ratio.max()),
        maximum_simultaneous_participating_contacts=int(counts.max()),required_unique_contacts=required,
        group_events_at_all_points_and_bin_origins=0,
        contact_threshold_first_sample=next((dict(index=q['index'],J_EE_core=q['J_EE_core']) for q,v in zip(rows,ratio) if v.max()>1),None),
        bin_origin_summary='Maximum over 0, 0.5, 1, 1.5 ms origins, not a confidence interval.',
        scope='Detection thresholds on existing periodic solutions; not a located dynamical bifurcation. No qualified events, so event propagation metrics remain undefined.'))
    path=OUTPUT/'figures/README.md';body=path.read_text()
    body=re.sub(r'^### '+name+r'\.png\s*\n.*?(?=^### |\Z)','',body,flags=re.M|re.S).rstrip()
    body+='\n\n### '+name+'.png\n对 H1 第85–111个连续检查通过的点，使用原冻结触点观察器，展示相同滤波下的场峰值、逐触点检测阈值和群体事件所需参与数。后段个别触点可以超过阈值，但四个 bin 起点均未形成合格群体事件，三项事件传播指标仍不可估计。**关注点**：触点越过读出阈值是观测层的变化，不能标成动力学分岔。\n'
    path.write_text(body)


if __name__=='__main__':main()
