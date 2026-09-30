"""Independent flux/count, demodulation and saved covariance audit."""
from pathlib import Path
import json
import numpy as np

OUT = Path(__file__).resolve().parents[2] / 'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918'


def main():
    read = lambda p: json.loads(p.read_text())
    dest = OUT / 'joint_voltage_current_density'; contract = read(dest / 'contract.json')
    rows = []; max_readback = 0.
    for case in contract['cases']:
        prefix = dest / f'case{case["id"]:02d}'
        if not prefix.with_suffix('.json').exists(): continue
        result = read(prefix.with_suffix('.json')); data = np.load(prefix.with_suffix('.npz'))
        assert result['id'] == case['id'] and result['kind'] == case['kind']
        dt = float(data['dt_ms'])
        if case['kind'] == 'waveform':
            rate = data['step_rate_hz']; T = float(data['T_ms']); n = len(rate); bins = len(data['predicted_hz'])
            index = np.minimum(((((np.arange(n)+1)*dt/T) % 1)*bins).astype(int), bins-1)
            exposure = np.bincount(index, minlength=bins)*dt
            own = np.bincount(index, weights=rate*dt, minlength=bins)/exposure
            delta = float(abs(own-data['predicted_hz']).max()); max_readback = max(max_readback, delta)
            assert delta < 1e-8
            reference = np.load(OUT / 'factorial_waveform/response.npz')
            counts = reference['counts'][case['index']]
            target = (counts/(exposure[None, :]/1000)).mean(0)
            assert np.max(abs(target-data['measured_hz'])) < 1e-8
            error = float(np.linalg.norm(own-target)/max(np.linalg.norm(target), np.sqrt(bins)))
            target_mean = float(counts.sum(1).mean()/(n*dt)*1000)
            bias = float(abs(rate.mean()-target_mean)/max(target_mean, 1.))
            assert abs(error-result['waveform_L2']) < 1e-10 and abs(bias-result['relative_mean_error']) < 1e-10
            passed = error <= .15 and bias <= .1
            moment_blocks = [data['final_free'], data['final_refractory']]
            assert abs(sum(block[:, 0, 0].sum() for block in moment_blocks)-1) < 1e-8
            detail = dict(waveform_L2=error, relative_mean_error=bias)
        else:
            original = case['original']; rate = data['rate_hz']; f = float(data['frequency_hz'])
            assert rate.shape == (2, round(original['duration_ms']/dt))
            difference = rate[0]-rate[1]; amplitude = float(data['amplitude'])
            phase = 2*np.pi*f/1000*((np.arange(len(difference))+1)*dt+float(data['burn_ms']))
            gain = (complex(np.dot(difference, np.sin(phase)), np.dot(difference, np.cos(phase)))*dt/original['duration_ms']/amplitude) if f else complex(difference.sum()*dt/original['duration_ms']/(2*amplitude))
            delta = abs(gain-complex(*result['predicted'])); max_readback = max(max_readback, delta)
            assert delta < 1e-9
            ref = original['reference']; error = float(abs(gain-complex(*ref['measured']))/max(abs(ref['dc_measured']), 1e-12))
            assert abs(error-result['normalized_error']) < 1e-8
            passed = error <= ref['tol'] if ref['counted'] else None
            moment_blocks = list(data['final_free'])
            detail = dict(original_case_id=original['id'], channel=original['channel'],
                          frequency_hz=f, normalized_error=error, tolerance=ref['tol'], counted=ref['counted'])
        assert np.isfinite(rate).all() and rate.min() >= -1e-10
        minimum_mass = 1.; min_eigenvalue = 0.
        for block in moment_blocks:
            minimum_mass = min(minimum_mass, float(block[:, 0, 0].min()))
            for raw in block:
                if raw[0, 0] > 1e-12:
                    mean = raw[0, 1:]/raw[0, 0]
                    covariance = raw[1:, 1:]/raw[0, 0]-np.outer(mean, mean)
                    relative = float(np.linalg.eigvalsh((covariance+covariance.T)/2).min()/max(np.trace(covariance), 1.))
                    min_eigenvalue = min(min_eigenvalue, relative)
        assert minimum_mass > -1e-12 and min_eigenvalue > -1e-8
        rows.append(dict(id=case['id'], kind=case['kind'], passed=passed, numerical_pass=result['numerical_pass'],
                         minimum_mass=minimum_mass, minimum_relative_covariance_eigenvalue=min_eigenvalue, **detail))
    complete = len(rows) == len(contract['cases'])
    passed = complete and all(r['passed'] is not False and r['numerical_pass'] for r in rows)
    result = dict(status=('JOINT_LOCAL_PILOT_PASS' if passed else 'JOINT_LOCAL_PILOT_FAIL') if complete else 'JOINT_LOCAL_PILOT_PENDING',
                  completed=len(rows), expected=16, waveform_count=sum(r['kind']=='waveform' for r in rows),
                  waveform_failed=sum(r['kind']=='waveform' and not r['passed'] for r in rows),
                  linear_count=sum(r['kind']=='linear' for r in rows),
                  linear_failed=sum(r['kind']=='linear' and r['passed'] is False for r in rows),
                  numerical_failed=sum(not r['numerical_pass'] for r in rows),
                  independent_readback_max_difference=max_readback, rows=rows, model_promoted=False,
                  scope=contract['limitations'], selection=contract['selection'])
    (dest / 'independent_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
