"""Inspect cached A-leading profiles before requesting more GPU corrections.

This is a filter-state screen, not complete physical-orbit acceptance.
Positive candidates still require an independent full-equation check.
"""
from plot_rate_branch_completion import *
from complete_rate_positive_stability import best_cached
from audit_rate_survey_filter_states import fingerprint
import os,time


def main():
    folder=DATA/'Aleading_profile_gaps';folder.mkdir(exist_ok=True)
    existing=read(DATA/'SCL_branch_scan/manifest.json')
    covered={q['index'] for q in existing['points'] if q['family']=='single'}
    rows=primary(families())['single'];s=RateField();results=[]
    for index,row in enumerate(rows):
        if index in covered:continue
        source=Path(row['path']);actual=best_cached(dict(orbit=str(source),
            J_EE_core=row['J_EE_core'],T_ms=row['T_ms']))
        output=folder/'points'/f'{index:03d}.json'
        cached=read(output) if output.exists() else None
        if cached and cached['profile_fingerprint']==fingerprint(actual):result=cached
        else:
            write(folder/'worker.json',dict(status='CPU_FILTER_SCREEN',pid=os.getpid(),
                index=index,completed=len(results),expected=len(rows)-len(covered)))
            z=np.load(actual);check=filter_state_minima(s,z['r'],float(z['T']))
            result=dict(index=index,original_orbit=str(source),orbit=str(actual),
                profile_fingerprint=fingerprint(actual),J_EE_core=float(z['J']),T_ms=float(z['T']),
                N=len(z['r']),filter_state_check=check,stored_collocation_error_Hz=float(z['residual']),
                status='FILTER_NONNEGATIVE_CANDIDATE' if check['positive'] else 'TEMPORAL_CORRECTION_REQUIRED')
            write(output,result)
        results.append(result)
        print('ALEADING GAP',index,result['J_EE_core'],result['N'],result['status'],flush=True)
    output=dict(status='CACHED_PROFILE_SCREEN_COMPLETE',timestamp=time.time(),rows=results,
        previously_covered_indices=sorted(covered),primary_family_size=len(rows),
        scope='No new solutions or branch labels. Nonnegative filter states are only a necessary physical check; candidates require full nine-state/physical-delay residual verification. Negative undershoot requires temporal correction and does not establish a false model or bifurcation.')
    write(folder/'summary.json',output)
    write(folder/'worker.json',dict(status='COMPLETE',pid=os.getpid(),completed=len(results),
        nonnegative_candidates=sum(q['status']=='FILTER_NONNEGATIVE_CANDIDATE' for q in results)))


if __name__=='__main__':main()
