"""Frozen observer on repeated exact rate cycles; keep exclusions visible."""
from plot_rate_periodic_composite import *
from expanded_readouts import observer,smooth2,describe,OLD
from collections import Counter
from scipy.ndimage import maximum_filter1d


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='periodic_contact_metrics')
    options=p.parse_args()
    s=RateField();contract=read(OLD/'observer_firing.json');names=contract['contact_names'];order=contact_indices(names)
    rows=[]
    for letter,name,J,title in CASES:
        r,T=loadcase(s,name,J);N=len(r);stop=int(16*T)//2*2
        spline=CubicSpline(np.arange(N+1)*T/N,np.concatenate([r,r[:1]]),bc_type='periodic')
        # Integrate four quadrature samples per ms, then sum to the original 2-ms count observable.
        t=(np.arange(stop*4)+.5)/4
        rates=(spline(t%T)@s.geo['contact_rate_weights']*1000).reshape(stop,4,15).mean(1)
        env=smooth2(rates.reshape(-1,2,15).sum(1)/1000)
        ob=observer.observe(env.T,2.,contract);mu=np.asarray(ob['centroid_ms'],float).reshape(-1,15)
        detected=(env.T>np.asarray(ob['threshold'])[:,None]);minimum=int(np.ceil(contract['minimum_detection_ms']/2))
        for ci in range(15):
            for aa,bb in observer.runs(detected[ci]):
                if bb-aa<minimum:detected[ci,aa:bb]=False
        pad=int(round(contract['extension_ms']/2));simultaneous=maximum_filter1d(detected.astype(np.uint8),2*pad+1,axis=1,mode='constant').sum(0)
        # Equal coverage of all orbit phases; full observer windows are already
        # checked against the longer repeated record. Do not drop one phenotype
        # merely because its window straddles an artificial cycle boundary.
        ids=np.array([i for i,e in enumerate(ob['events']) if 4*T<=np.mean(e['window_ms'])<12*T],int)
        primary=np.array([i for i in ids if ob['events'][i]['primary_eligible']],int)
        why=Counter(x for i in ids for x in ob['events'][i]['primary_exclusion_reasons'])
        stats=describe(mu[primary],names)
        scl=np.array([n.startswith('SCL') for n in names]);scln=int(np.isfinite(mu[primary][:,scl]).any(1).sum())
        row=dict(case=letter,orbit=name,J_EE_core=J,T_ms=T if name else None,qualified_events=len(primary),all_detected_events=len(ids),SCL_events=scln,
          exclusion_counts=dict(why),maximum_unique_contacts_in_observer_window=int(simultaneous.max()),required_unique_contacts=ob['required_unique_contacts'],boundary_or_low_window_support=ob['boundary_or_low_window_support'],observation=ob,metrics=stats,all_detected_metrics=describe(mu[ids],names),centroids_ms=mu[primary],
          window_ms=[4*T,12*T],selection='Event anchors in eight complete interior orbit periods; original observer windows/eligibility unchanged',unit='Eight interior repetitions of one deterministic orbit; not independent events or realizations')
        rows.append(row);print(letter,'detected',len(ids),'qualified',len(primary),'SCL',scln,'excluded',dict(why),flush=True)
    source=PERIODIC_OUT/'periodic_contact_observations.json'
    if source.exists():write(PERIODIC_OUT/'attempt_history'/f'contact_observations_before_refined_profiles_{time.time_ns()}.json',read(source))
    write(source,dict(rows=rows,contact_names=names,observer=str(OLD/'observer_firing.json'),
      case_resolution_source=str(PERIODIC_OUT/'composite_case_resolution.json'),
      readout='Contact-weighted firing rate converted to weighted counts per 2 ms, frozen observer unchanged',
      no_events='No qualifying events means rank/participation statistics undefined, not zero spatial activity.'))
    fig,axs=plt.subplots(5,3,figsize=(13,13),gridspec_kw={'width_ratios':[1,1,1.25]},layout='constrained')
    for i,q in enumerate(rows):
        rank=np.asarray(q['metrics']['mean_rank'],float)[order];part=np.asarray(q['metrics']['participation'],float)[order]
        o=np.asarray(q['metrics']['within_shaft_order_probability'],float)[np.ix_(order,order)]
        for j,(values,label) in enumerate([(rank,'Mean normalized rank'),(part,'Participation probability')]):
            ax=axs[i,j]
            for shaft,indices in [('SCL',np.arange(4)),('ICL',np.arange(4,15))]:
                ax.plot(values[indices],indices,'o-',lw=1,ms=4,color=SHAFT_COLORS[shaft])
            ax.set(ylim=(14.5,-.5),xlim=(-.05,1.05),yticks=np.arange(15),yticklabels=CONTACT_ORDER,xlabel=label);ax.tick_params(labelsize=8);ax.axhline(3.5,color='#bbbbbb',lw=.5);style(ax)
            for tick,n in zip(ax.get_yticklabels(),CONTACT_ORDER):tick.set_color(SHAFT_COLORS[n[:3]])
            if not q['qualified_events']:ax.text(.5,.5,'Not estimable\nNo qualified events',ha='center',va='center',transform=ax.transAxes)
        ax=axs[i,2];im=ax.imshow(o,vmin=0,vmax=1,cmap='coolwarm');ax.set(xticks=np.arange(15),xticklabels=CONTACT_ORDER,yticks=[0,3,4,14],yticklabels=[CONTACT_ORDER[k] for k in [0,3,4,14]],xlabel='P(column later than row)');ax.tick_params(labelsize=8)
        ax.tick_params(axis='x',rotation=90,labelsize=7)
        for tick,n in zip(ax.get_xticklabels(),CONTACT_ORDER):tick.set_color(SHAFT_COLORS[n[:3]])
        if not q['qualified_events']:ax.text(.5,.5,'Not estimable',ha='center',va='center',transform=ax.transAxes)
        axs[i,0].set_title(f"{q['case']}  J={q['J_EE_core']:g}: {q['qualified_events']}/{q['all_detected_events']} qualified",loc='left')
    fig.colorbar(im,ax=axs[:,2],shrink=.5,label='Within-shaft order probability')
    save(fig,options.output)
    update_readme({options.output:'对通过物理状态检查的 a–e 解应用原冻结触点观察器，给出平均标准化 rank、杆内先后概率和参与概率。统计覆盖八个完整内部周期，按事件锚点选择，不裁掉跨越人为周期边界的事件；重复轨道不能视为独立事件样本。**关注点**：零合格事件不等于零传播；接近 250 ms 的周期会受到冻结观察器的窗口重叠判据和 2 ms 量化影响。'})

if __name__=='__main__':main()
