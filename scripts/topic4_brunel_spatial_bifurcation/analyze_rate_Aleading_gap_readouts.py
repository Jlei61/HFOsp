"""Frozen contact readout at the three corrected A-leading survey sites.

Keep contact recruitment separate from the fixed 250-ms group-window filter;
a periodic solution faster than that window can lose all eligible groups.
"""
from plot_rate_sameJ_burst_pair import observe_cycle, RateField, read, write, DATA, OLD
import numpy as np


def main():
    source=DATA/'Aleading_profile_gaps/corrected_profiles.json'
    profiles=read(source)
    assert profiles['status']=='THREE_SAME_J_PROFILES_CHECKED'
    contract=read(OLD/'observer_firing.json');model=RateField()
    names=contract['contact_names'];scl=np.array([n.startswith('SCL') for n in names])
    rows=[]
    for item in profiles['rows']:
        check=item['resolution']
        assert check['status']=='RESOLUTION_CHECKED'
        assert check['filter_state_check']['positive'] and check['maximum_group_defect_Hz']<.001
        z=np.load(item['orbit']);observed=observe_cycle(z['r'],float(z['T']),model,contract)
        records=[]
        for q in observed['records']:
            records.append({key:q[key] for key in ['bin_origin_ms','detected_events',
                'qualified_events','SCL_qualified_events','sustained_contact_names',
                'sustained_SCL_contact_names','contact_peak_to_threshold','exclusions','metrics']})
        rows.append(dict(index=item['index'],orbit=item['orbit'],J_EE_core=float(z['J']),T_ms=float(z['T']),
            group_window_ms=contract['window_ms'],period_minus_group_window_ms=float(z['T'])-contract['window_ms'],
            records=records,interior_cycles=observed['interior_cycles']))
    result=dict(status='THREE_CORRECTED_PROFILE_READOUTS_CHECKED',source=str(source),
        observer_source=str(OLD/'observer_firing.json'),contact_names=names,rows=rows,
        scope='Eight repeated cycles of each exact periodic solution, with four bin origins. '
              'No independent realizations or stable-attractor claim. Group-window exclusions '
              'are properties of the frozen observer, not dynamical bifurcations.')
    write(DATA/'Aleading_profile_gaps/readout_diagnostics.json',result)
    for row in rows:
        r=row['records'];ratios=np.array([q['contact_peak_to_threshold'] for q in r])[:,scl]
        print('A_LEADING_READOUT',row['index'],row['J_EE_core'],row['T_ms'],
              'SCL', [q['sustained_SCL_contact_names'] for q in r],
              'detected',[q['detected_events'] for q in r],
              'eligible',[q['qualified_events'] for q in r],
              'exclusions',[q['exclusions'] for q in r],
              'maximum_SCL_peak_to_threshold',float(ratios.max()),flush=True)


if __name__=='__main__':main()
