"""Conditional saddle-node bottleneck prediction, without fitting escape times.

This is a consistency test, not a substitute for a periodic fold/Floquet test.
The normal form x'=a(D-Dc)+b*x*x predicts an infinite-exit time from the old
stable branch. Our finite long-event readout should generally occur earlier.
No activity duration, onset time, or trajectory is used to fit a*b.
"""
from canonical_readouts import OUT, R
from pathlib import Path
import json
import math
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def long_events(field, counts, duration_ms):
    g = field @ (counts / counts.sum())
    g = g[:len(g)//10*10].reshape(-1, 10).mean(1)
    separators, events = R.find_events(g)
    items = [dict(start_ms=e['start_bin']*10,
                  end_ms=e['end_bin']*10, duration_ms=e['duration_ms'],
                  right_censored=False)
             for e in events if e['qualifies'] and e['duration_ms'] >= duration_ms]
    if separators:
        start = int(separators[-1][1])
        if (len(g)-start)*10 >= duration_ms and g[start:].max() >= R.EVENT_PEAK_HZ:
            items.append(dict(start_ms=start*10, end_ms=None,
                              observed_duration_ms=(len(g)-start)*10,
                              right_censored=True))
    return items


def main():
    folder = OUT/'floquet'
    spectra = []
    for step in ['0.003125', '0.0015625']:
        path = folder/f'rate_refined_T2671_G8193_point0000_endpoint_dt{step}_quotient_chainphase_streamed.json'
        if path.exists():
            q = read(path)
            if q['phase_valid'] and max(q['eigen_residuals']) < 1e-5:
                spectra.append(dict(q, source=str(path)))
    assert spectra
    q = min(spectra, key=lambda x:x['dt_ms'])
    mu = q['max_transverse_modulus']
    assert 0 < mu < 1
    lam = math.log(mu) / (q['T_ms']/1000)
    turn = OUT/'periodic/rate_turn_G4097/evaluations.json'
    Dc = max(x['D'] for x in read(turn))
    Dc_status = 'largest observed D on the refined candidate turn, not a certified fold'
    final_turn = turn.with_name('result.json')
    if final_turn.exists() and read(final_turn).get('status') == 'NUMERICAL_PERIODIC_TURN_REFINED':
        Dc = read(final_turn)['D']
        Dc_status = 'refined Galerkin turn; bifurcation certification remains separate'
    D0 = .144970
    ab = lam*lam/(4*(Dc-q['D']))
    source = OUT/'runs/rate_critical_near_restart_D0.1449700_dt0.05/trajectory.npz'
    specs = [
        ('rate_critical_finer_restart_D0.1449750_dt0.05', 'rate_tail_D144975_D0.1449750_dt0.05'),
        ('rate_critical_finer_restart_D0.1449775_dt0.05', None),
        ('rate_matched_bottleneck_check_D0.1449800_dt0.05', None),
        ('rate_matched_bottleneck_check_D0.1449900_dt0.05', None),
    ]
    rows = []
    for name, tail in specs:
        path = OUT/'runs'/name
        if not (path/'result.json').exists():
            rows.append(dict(label=name, status='RUNNING_OR_MISSING')); continue
        contract = read(path/'contract.json')
        assert Path(contract['initial']).resolve() == source.resolve()
        assert contract['M'] == 'dynamic' and contract['Z'] == 'held'
        z = np.load(path/'trajectory.npz');field=z['field_E_hz'];counts=z['cell_counts']
        if tail:
            zz=np.load(OUT/'runs'/tail/'trajectory.npz')
            assert np.array_equal(counts,zz['cell_counts'])
            field=np.concatenate([field,zz['field_E_hz']])
        D=contract['D_initial'];delta=D-Dc
        assert delta>0 and Dc>D0
        predicted=(math.pi/2+math.atan(math.sqrt((Dc-D0)/delta)))/math.sqrt(ab*delta)
        observed={}
        for threshold in [200,500,1000]:
            events=long_events(field,counts,threshold)
            observed[str(threshold)]=events[0] if events else None
        rows.append(dict(label=name,status='COMPLETE',D=D,duration_ms=len(field),
                         predicted_infinite_exit_s=predicted,first_long_events_by_threshold_ms=observed))
    result=dict(status='CONDITIONAL_NORMAL_FORM_CONSISTENCY_ONLY',
                Dc=Dc,Dc_status=Dc_status,initial_D=D0,shared_complete_initial_history=str(source),
                transverse_multiplier=mu,lambda_per_s=lam,a_times_b=ab,
                spectrum_source=q['source'],spectrum_refinement_pending=len(spectra)<2,
                formula='[pi/2 + atan(sqrt((Dc-D0)/(D-Dc)))] / sqrt(a*b*(D-Dc))',
                coefficient_rule='a*b=lambda(Db)^2/[4*(Dc-Db)]; no escape data fitted',
                assumptions=['Local quadratic cycle-fold normal form applies',
                             'The initial complete state is close to the old stable cycle',
                             'A scalar center coordinate captures the slow bottleneck'],
                limit='Finite long-event onset differs from the normal form infinite-exit time; neither agreement nor disagreement alone establishes bifurcation type.',
                rows=rows)
    upper=folder/'rate_upper_G16385_M65536_point0000_endpoint_dt0.003125_quotient_chainphase_rk4_cubic_streamed_fastgrid.json'
    if upper.exists():
        u=read(upper)
        if u['phase_valid'] and max(u['eigen_residuals'])<1e-5:
            exponent=math.log(u['max_transverse_modulus'])/(u['T_ms']/1000)
            result['opposite_branch_check']=dict(source=str(upper),D=u['D'],
                multiplier=u['max_transverse_modulus'],lambda_per_s=exponent,
                ratio_to_stable_decay_magnitude=exponent/abs(lam),
                limit='Upper time-step refinement pending; not used to refit the escape prediction.')
    (OUT/'bottleneck_consistency.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
