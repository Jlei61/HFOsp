"""Frozen event/readout metrics on every checked H1 point in a fixed segment.

Each statistical unit is one exact periodic solution, repeated only to remove
record boundaries.  Four bin origins measure readout discretization, not
event randomness or independent network realizations.
"""
from analyze_rate_PD1_contact_alternation import *
from scipy.ndimage import maximum_filter1d


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--first-index',type=int,default=85)
    parser.add_argument('--last-index',type=int,default=100)
    parser.add_argument('--output-tag',default='')
    args=parser.parse_args()
    indices=list(range(args.first_index,args.last_index+1));assert indices
    source=DATA/'H1_to_PD3_display_extension.json'
    progress=read(source)
    selected=[q for q in progress['rows'] if q['index'] in indices]
    assert [q['index'] for q in selected]==indices
    assert all(q['status']=='PASS' and q['branch_match_pass'] and
               q['check']['filter_state_check']['positive'] for q in selected)
    contract_source=OLD/'observer_firing.json';contract=read(contract_source)
    names=contract['contact_names'];s=RateField()
    geo=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz')
    assert geo['contact_names'].tolist()==names
    regional_weights=np.zeros((s.P,3))
    for k in range(3):
        mask=s.E&(s.geo['group_region']==k)
        regional_weights[mask,k]=s.geo['group_size'][mask]/s.geo['group_size'][mask].sum()
    summaries=[];folder=DATA/(args.output_tag+'_observer' if args.output_tag else 'H1_observer');folder.mkdir(exist_ok=True)
    for point in selected:
        path=Path(point['orbit']);z=np.load(path);T=float(z['T']);r=z['r'];N=len(r)
        contact=r@s.geo['contact_rate_weights']*1000
        spline=CubicSpline(np.arange(N+1)*T/N,np.r_[contact,contact[:1]],bc_type='periodic')
        regional=r@regional_weights*1000
        regional_spline=CubicSpline(np.arange(N+1)*T/N,np.r_[regional,regional[:1]],bc_type='periodic')
        first_cycle=max(4,int(np.ceil(contract['burnin_ms']/T))+2)
        analysis_window=[first_cycle*T,(first_cycle+8)*T]
        stop=int((first_cycle+12)*T)//2*2;records=[]
        for offset in [0.,.5,1.,1.5]:
            t=(np.arange(stop*4)+.5)/4+offset
            rates=spline(t%T).reshape(stop,4,15).mean(1)
            env=smooth2(rates.reshape(-1,2,15).sum(1)/1000)
            rr=regional_spline(t%T).reshape(stop,4,3).mean(1)
            regional_env=smooth2(rr.reshape(-1,2,3).sum(1)/1000)
            ob=observer.observe(env.T,2.,contract)
            centroids=np.asarray(ob['centroid_ms'],float).reshape(-1,15)+offset
            anchors=np.array([np.mean(event['window_ms'])+offset for event in ob['events']])
            ids=np.flatnonzero((anchors>=analysis_window[0])&(anchors<analysis_window[1]))
            primary=np.array([i for i in ids if ob['events'][i]['primary_eligible']],int)
            why=Counter(reason for i in ids for reason in ob['events'][i]['primary_exclusion_reasons'])
            scl=np.array([name.startswith('SCL') for name in names])
            bar=np.asarray(ob['threshold'])
            detected=env.T>bar[:,None]
            minimum=max(1,int(np.ceil(contract['minimum_detection_ms']/2.)))
            for ci in range(15):
                for aa,bb in observer.runs(detected[ci]):
                    if bb-aa<minimum:detected[ci,aa:bb]=False
            pad=int(round(contract['extension_ms']/2.))
            expanded=maximum_filter1d(detected.astype(np.uint8),2*pad+1,axis=1,mode='constant')>0
            frame_times=np.arange(len(env))*2+1+offset
            interior=(frame_times>=analysis_window[0])&(frame_times<analysis_window[1])
            peak=env[interior].max(0)
            records.append(dict(bin_origin_ms=offset,detected_events=len(ids),qualified_events=len(primary),
                SCL_qualified_events=int(np.isfinite(centroids[primary][:,scl]).any(1).sum()),
                exclusion_counts=dict(why),required_unique_contacts=ob['required_unique_contacts'],
                maximum_unique_contacts_in_group_window=int(expanded[:,interior].sum(0).max()),
                sustained_suprathreshold_contacts=[names[i] for i in np.flatnonzero(detected[:,interior].any(1))],
                contact_peak_to_threshold=peak/bar,smoothed_contact_peak_Hz=peak*500,
                regional_peak_observer_filtered_Hz=regional_env[interior].max(0)*500,
                event_window_widths_ms=sorted(set(float(np.diff(ob['events'][i]['window_ms'])[0]) for i in ids)),
                observer_anchor_intervals_ms=np.diff(anchors[ids]),
                metrics=describe(centroids[primary],names),
                all_detected_metrics=describe(centroids[ids],names),
                qualified_event_indices=primary,interior_event_indices=ids,
                contact_centroids_ms=centroids,observation=ob))
        result=dict(index=point['index'],orbit=str(path),J_EE_core=float(z['J']),T_ms=T,
            analysis_window_ms=analysis_window,total_repeat_cycles=first_cycle+12,
            physical_check_source=str(source),records=records)
        output=folder/f'point_{point["index"]:03d}.json';write(output,result)
        summary=dict(index=point['index'],J_EE_core=float(z['J']),T_ms=T,source=str(output),
            detected_counts=[q['detected_events'] for q in records],
            qualified_counts=[q['qualified_events'] for q in records],
            SCL_counts=[q['SCL_qualified_events'] for q in records],
            exclusion_counts=[q['exclusion_counts'] for q in records],
            window_widths_ms=[q['event_window_widths_ms'] for q in records],
            maximum_unique_contacts=[q['maximum_unique_contacts_in_group_window'] for q in records],
            required_unique_contacts=records[0]['required_unique_contacts'],
            metrics_estimable=[q['qualified_events']>0 for q in records])
        summaries.append(summary)
        print('H1 FROZEN OBSERVER',summary,flush=True)
    summary_name=args.output_tag+'_observer_summary.json' if args.output_tag else 'H1_extension_observer_summary.json'
    write(DATA/summary_name,dict(status='FROZEN_OBSERVER_SEGMENT_COMPLETE',
        source=str(source),observer=str(contract_source),indices=indices,
        contact_names=names,rows=summaries,
        statistical_unit='One exact deterministic orbit per parameter point. Eight interior repeats and four bin origins are not independent events or realizations.',
        definitions='Frozen primary eligibility; normalized mean rank, within-shaft joint-participation order and contact participation. No qualified events means these metrics are undefined, not zero.',
        scope='Readout eligibility and descriptive propagation statistics on an existing H1 periodic segment. Does not establish stability, irregular dynamics, native-SNN equivalence, a new bifurcation or connection to another periodic family.'))


if __name__=='__main__':main()
