"""Collect audited local state evidence; never label finite traces as branches."""
from common import OUT,np,model,read,write
from scipy.ndimage import uniform_filter1d
import csv

OLD=OUT/'core_a_resource_bifurcation_20260923'
DEST=OUT/'core_a_transition_continuation_20260924'


def main():
    s=model(40);A=s.E&(s.geo['group_region']==0);w=s.sizes[A]/s.sizes[A].sum()
    paths=[OLD/'reference',OLD/'coreA_depleted']
    paths += sorted(p.parent for p in DEST.glob('*/local_state_audit.json'))
    rows=[]
    for folder in paths:
        a=read(folder/'local_state_audit.json');assert a['status']=='AUDIT_PASS'
        independent=read(folder/'independent_audit.json');job=read(folder/'jobs.json')
        z=np.load(folder/f'block{job["completed_blocks"][-1]:02d}.npz')
        rate=z['group_rate_hz'][:,A].astype(float)@w
        sm=uniform_filter1d(rate,10,mode='nearest');local=a['rows'][1]
        assert abs(rate.mean()-local['mean_rate_hz'])<1e-9
        assert np.max(abs(np.array([sm.min(),sm.max()])-local['range_10ms_hz']))<1e-8
        if local['range_10ms_hz'][0]>50:category='SUSTAINED_HIGH_OBSERVED'
        elif local['complete_local_episodes']>=2:category='SELF_LIMITED_OBSERVED'
        else:category='UNRESOLVED'
        active=sm>=5;edges=np.diff(np.r_[False,active,False].astype(int))
        duration=np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)
        rows.append(dict(label=folder.name,source=str(folder),**a['coordinates'],
            cumulative_window_start_ms=float(a['window_ms'][0]+job['condition'].get('prior_same_field_elapsed_ms',0)),
            cumulative_window_end_ms=float(a['window_ms'][1]+job['condition'].get('prior_same_field_elapsed_ms',0)),
            finite_window_category=category,Core_A_mean_Hz=local['mean_rate_hz'],
            Core_A_min_10ms_Hz=float(sm.min()),Core_A_max_10ms_Hz=float(sm.max()),
            Core_A_quiet_fraction=local['quiet_fraction'],
            Core_A_complete_local_episodes=local['complete_local_episodes'],
            Core_A_longest_active_ms=int(duration.max(initial=0)),
            Core_A_final_active_right_censored=bool(active[-1]),
            Core_A_M_mean_mV=float((z['M_current'][:,A]@w).mean()),
            Core_B_quiet_fraction=a['rows'][2]['quiet_fraction'],
            persistent_spatial_fraction=independent['windows'][-1]['persistent_spatial_fraction'],
            global_high_entry_ms=independent['original_high_onset_elapsed_ms'],
            full_field_best_recurrence_MSE=independent['recurrence']['best_local_minima'][0]['relative_MSE']))
    rows.sort(key=lambda a:(a['D_A'],a['label']))
    with (DEST/'audited_local_states.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader();writer.writerows(rows)
    write(DEST/'audited_local_states.json',dict(rows=rows,
        classification='Final5s: minimum10ms CoreA rate>50Hz => sustained high observed; otherwise>=2 complete5Hz activity episodes bounded by20ms quiet => self-limited observed; else unresolved. These are observational summaries, not attractor or bifurcation labels.',
        original_readout='Local episodes use the previously registered regional rule; global whole-field event/onset criteria remain separate.',
        history_warning='Rows at identicalD_A may be successive windows from the same trajectory, not independent replicates. Sustained_high_observed labels only the listed finite window; later observed termination overrides any permanent-state interpretation.',
        bifurcation_type='NOT_ESTABLISHED',model_promoted=False))
    for r in rows:print(r['label'],r['Z_A'],r['finite_window_category'],r['Core_A_quiet_fraction'],r['Core_A_complete_local_episodes'],flush=True)


if __name__=='__main__':main()
