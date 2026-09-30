"""Independent demodulation and original-gate accounting; no new simulations."""
from pathlib import Path
import json
import numpy as np

OUT=Path(__file__).resolve().parents[2]/'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918'


def main():
    read=lambda p:json.loads(p.read_text())
    c=read(OUT/'conditional_density_linear_response_contract.json')
    dest=OUT/'conditional_density_linear_response';rows=[];max_difference=0.
    for case in c['cases']:
        path=dest/f'case{case["id"]:03d}.json'
        if not path.exists():continue
        result=read(path);z=np.load(path.with_suffix('.npz'))
        assert result['case_id']==case['id'] and result['workpoint']==case['workpoint']
        dt=case['dt_ms'];f=case['frequency_hz'];amp=case['absolute_amplitude'];T=case['duration_ms']
        rates=z['rate_hz'];n=rates.shape[1]
        assert rates.shape==(2,round(T/dt)) and np.isfinite(rates).all() and rates.min()>=0
        assert float(z['amplitude'])==amp and float(z['burn_ms'])==case['burn_ms']
        delta=rates[0]-rates[1]
        if f:
            angles=2*np.pi*f/1000*(dt*np.arange(1,n+1)+case['burn_ms'])
            gain=complex(np.dot(delta,np.sin(angles)),np.dot(delta,np.cos(angles)))*dt/(T*amp)
        else:
            gain=complex((rates[0].sum()-rates[1].sum())*dt/(2*T*amp))
        dc_scale=abs(case['reference']['dc_measured'])
        error=abs(gain-complex(*case['reference']['measured']))/max(dc_scale,1e-12)
        difference=abs(gain-complex(*result['predicted']));max_difference=max(max_difference,difference)
        assert difference<1e-8
        if dc_scale == 0:
            # The original reference explicitly excludes these zero-DC
            # rows. Dividing roundoff by a numeric floor is not a meaningful
            # response error. Keep the raw gain audit and report undefined.
            assert not case['reference']['counted']
        else:
            assert np.isclose(error,result['normalized_error'],rtol=1e-12,atol=1e-8)
        passed=error<=case['reference']['tol'] if case['reference']['counted'] else None
        assert passed==result['passed']
        rows.append(dict(id=case['id'],kind=case['kind'],channel=result['channel'],frequency_hz=f,
            counted=case['reference']['counted'],normalized_error=float(error) if dc_scale else None,
            normalization_status='DEFINED' if dc_scale else 'ZERO_REFERENCE_DC_UNDEFINED',passed=passed,
            sign_ok=result['sign_ok'],numerical_pass=result['numerical_pass']))
    ac=[r for r in rows if r['frequency_hz']>0 and r['counted']]
    dc=[r for r in rows if r['frequency_hz']==0 and r['counted']]
    ac_failed=sum(not r['passed'] for r in ac);dc_failed=sum(not r['passed'] for r in dc)
    complete=len(rows)==len(c['cases'])
    q=dict(status='DENSITY_LINEAR_VALIDATION_COMPLETE' if complete else 'DENSITY_LINEAR_VALIDATION_PENDING',
        completed_cases=len(rows),expected_cases=len(c['cases']),AC_counted=len(ac),AC_failed=ac_failed,
        DC_counted=len(dc),DC_failed=dc_failed,DC_per_row_tolerance=.15,
        AC_original_gate_pass=bool(len(ac)==146 and ac_failed<=14) if complete else None,
        low_frequency_sign_failures=sum(r['sign_ok'] is False for r in rows),
        numerical_failures=sum(not r['numerical_pass'] for r in rows),
        independent_demodulation_max_difference=max_difference,rows=rows,
        model_promoted=False,scope=c['scope'],
        limitation='No formal response acceptance until complete; DC failures and numerical/sign failures remain separate from the original10percentACfailure allowance. No autonomous network or bifurcation follows from this test.')
    temp=dest/'validation_summary.json.tmp';temp.write_text(json.dumps(q,indent=2)+'\n');temp.replace(dest/'validation_summary.json')
    print(json.dumps({k:v for k,v in q.items() if k!='rows'}))


if __name__=='__main__':main()
