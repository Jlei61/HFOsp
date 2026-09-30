"""Observed access to low Core-A activity, not a crisis certificate.

Use already audited complete histories. Thresholds distinguish disappearance
of a low-activity visit from arbitrary placement of the5Hz reporting cutoff.
Time samples and repeated events in one history are not independent trials.
"""
from common import OUT,np,read,write
from scipy.ndimage import uniform_filter1d


def main():
    base=OUT/'core_a_bifurcation_type_20260924'
    sources=[
        ('returning',OUT/'core_a_transition_continuation_20260924/mid1_lower_30s_regional.npz',
         OUT/'core_a_transition_continuation_20260924/mid1_lower_30s_history.json'),
        ('late_return',base/'late_return_counterexample/joined70s_regional.npz',base/'late_return_counterexample/result.json'),
        ('no_return_observed',base/'censoring_controls/above_SN/joined70s_regional.npz',base/'censoring_controls/above_SN/joined70s_audit.json')]
    rows=[]
    for label,path,auditpath in sources:
        audit=read(auditpath);assert audit['status']=='AUDIT_PASS'
        data=np.load(path);rate=data['regional_rate_hz'][:,1]
        sm=uniform_filter1d(rate,10,mode='nearest')
        assert np.max(abs(sm-data['Core_A_smoothed_hz']))<1e-8
        stop=len(rate);window=sm[2000:]
        coordinate=audit['D_A'] if 'D_A' in audit else audit['coordinates']['D_A']
        entry=dict(label=label,source=str(path),audit=str(auditpath),D_A=coordinate,
            analyzed_window_ms=[2000,stop],minimum10ms_hz=float(window.min()),
            time_sample_quantiles_hz=dict(zip(['q0001','q001','q01','q05','q50'],np.quantile(window,[.0001,.001,.01,.05,.5]).tolist())),thresholds=[])
        for threshold in [1.,5.,10.,50.]:
            edge=np.diff(np.r_[False,sm<threshold,False].astype(int))
            all_intervals=[(int(a),int(b)) for a,b in zip(np.flatnonzero(edge==1),np.flatnonzero(edge==-1)) if b-a>=20]
            quiet=[(a,b) for a,b in all_intervals if a>=2000 and b<stop]
            entry['thresholds'].append(dict(threshold_hz=threshold,min_duration_ms=20,
                time_fraction_below=float(np.mean(window<threshold)),complete_quiet_visits_after2s=len(quiet),
                qualified_intervals_ms=quiet))
        rows.append(entry)
    result=dict(status='AUDITED_TRAJECTORY_SUPPORT_READOUT',rows=rows,
        observable='Core-A E rate, cell-count weighted Hz/neuron,10ms smoothing; previously audited whole trajectories at dt.05ms, remove only first2s for displayed occupancy. Quiet visits require20ms. Each threshold is a diagnostic sensitivity, not a new state classifier.',
        statistical_unit='One deterministic complete history per local Z field. Time samples/events are dependent; no independent-replicate inference, lifetime-law fit or infinite-time no-return claim.',
        interpretation='The visited low-activity region becomes rare and is not seen in the stronger-field70s record. This motivates distinguishing an invariant-set change from finite sampling or a smooth shift of rate minima. These observations alone do not establish an interior crisis, a boundary crisis, a periodic bifurcation, or an asymptotic critical parameter.',
        phase_and_mesh_limit='All rows are original coarse trajectories; separate complete fine trajectories also return at the weaker fields, but the specific late-return time was not locally mesh-converged.',
        model_promoted=False)
    write(base/'low_activity_access.json',result)
    for row in rows:print(row['label'],row['D_A'],row['minimum10ms_hz'],[(q['threshold_hz'],q['time_fraction_below'],q['complete_quiet_visits_after2s']) for q in row['thresholds']],flush=True)


if __name__=='__main__':main()
