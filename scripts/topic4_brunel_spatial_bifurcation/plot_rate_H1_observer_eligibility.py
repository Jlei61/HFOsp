"""Explain why the asymmetric H1 cycles yield no frozen-observer events."""
from plot_rate_branch_completion import *


def main():
    source=DATA/'H1_extension_observer_summary.json';summary=read(source)
    assert summary['status']=='FROZEN_OBSERVER_SEGMENT_COMPLETE'
    rows=[read(q['source']) for q in summary['rows']]
    assert all(r['qualified_events']==0 and r['detected_events']==0
               for q in rows for r in q['records'])
    names=summary['contact_names'];order=contact_indices(names)
    contract=read(summary['observer']);threshold=np.asarray(contract['threshold'])*500
    J=np.array([q['J_EE_core'] for q in rows])
    regional=np.array([np.max([v['regional_peak_observer_filtered_Hz'] for v in q['records']],axis=0) for q in rows])
    contact=np.array([np.max([v['smoothed_contact_peak_Hz'] for v in q['records']],axis=0) for q in rows])
    ratio=np.array([np.max([v['contact_peak_to_threshold'] for v in q['records']],axis=0) for q in rows])
    assert ratio.max()<1
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axs=plt.subplots(1,2,figsize=(12.3,5.5),gridspec_kw={'width_ratios':[1,1.25]})
    fig.subplots_adjust(left=.075,right=.91,bottom=.17,top=.84,wspace=.35)
    ax=axs[0]
    for k,color in enumerate(COL):
        ax.plot(J,regional[:,k],color=color,lw=1.7,label=f'Core {"AB"[k]}')
    ax.plot(J,contact.max(1),color='#333333',lw=1.7,label='Largest contact rate')
    ax.axhspan(threshold.min(),threshold.max(),color='#f0c987',alpha=.55,
               label='Frozen contact thresholds')
    ax.set(xlabel=r'$J_{\mathrm{EE,core}}$',ylabel='Filtered peak (Hz / cell)',
           title='A   Identical temporal filtering')
    ax.legend(frameon=False,fontsize=8,loc='upper left');style(ax)
    ax=axs[1]
    # Midpoint edges retain the actual, slightly nonuniform continuation J's.
    edges=np.r_[J[0]-(J[1]-J[0])/2,(J[:-1]+J[1:])/2,J[-1]+(J[-1]-J[-2])/2]
    im=ax.pcolormesh(edges,np.arange(16)-.5,ratio[:,order].T,
                     cmap='magma',vmin=0,vmax=1,shading='flat')
    ax.set(ylim=(14.5,-.5),yticks=np.arange(15),yticklabels=CONTACT_ORDER,
           xlabel=r'$J_{\mathrm{EE,core}}$',title='B   Contact peak / frozen threshold')
    ax.axhline(3.5,color='white',lw=.8);ax.tick_params(axis='y',labelsize=8,length=2)
    for tick,name in zip(ax.get_yticklabels(),CONTACT_ORDER):tick.set_color(SHAFT_COLORS[name[:3]])
    fig.colorbar(im,cax=fig.add_axes([.93,.18,.015,.64]),label='Ratio; detection requires > 1')
    fig.suptitle('H1 segment: field oscillation without detected contact-group events',fontsize=13,y=.96)
    name='H1_contact_observer_eligibility';save_new(fig,name)
    write(DATA/(name+'.json'),dict(source=str(source),maximum_contact_threshold_ratio=float(ratio.max()),
        smoothed_contact_threshold_range_Hz=[float(threshold.min()),float(threshold.max())],
        maximum_filtered_contact_rate_Hz=float(contact.max()),
        same_temporal_filter_for_regions_and_contacts=True,
        bin_origins_ms=[0,.5,1,1.5],bin_origin_summary='Maximum over the four origins; not a confidence interval',
        observer_metrics=dict(mean_normalized_rank=None,within_shaft_order=None,participation=None),
        scope='Every tested point and bin origin remains below individual contact thresholds. Thus there are no group events and the three event metrics are undefined. This is a frozen-observer result, not absence of spatial activity, instability or a dynamical bifurcation.'))
    path=OUTPUT/'figures/README.md';body=path.read_text()
    body=re.sub(r'^### '+name+r'\.png\s*\n.*?(?=^### |\Z)','',body,flags=re.M|re.S).rstrip()
    body+='\n\n### '+name+'.png\n对 H1 第85–100点使用原冻结观察器，比较经相同时间滤波的两核峰值和触点峰值；右侧保留固定15触点，显示各触点相对自身检测阈值的峰值。四个bin起点均无超过阈值的触点，因此没有群体事件，三项事件传播指标不可估计。**关注点**：场中存在周期传播不等于触点观察器检出事件；这不是零传播、零误差或新的动力学分岔。\n'
    path.write_text(body)


if __name__=='__main__':main()
