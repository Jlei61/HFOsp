"""Independently read raw predicted rates and compare both refinement steps."""
from pathlib import Path
import json
import numpy as np

OUT = Path(__file__).resolve().parents[2] / 'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918'


def main():
    read = lambda p: json.loads(p.read_text())
    dest = OUT / 'conditional_density_grid_diagnostic'
    contract = read(dest / 'contract.json')
    rows = []; max_difference = 0.; completed = 0
    for case in contract['cases']:
        values = {}; details = []
        for grid in [256, 512, 1024]:
            prefix = (OUT / 'conditional_density_linear_response' / f'case{case["id"]:03d}') if grid == 256 else dest / f'case{case["id"]:03d}_grid{grid}'
            if not prefix.with_suffix('.json').exists():
                continue
            result = read(prefix.with_suffix('.json'))
            data = np.load(prefix.with_suffix('.npz'))
            rate = data['rate_hz']; dt = float(data['dt_ms']); f = float(data['frequency_hz'])
            assert result['case_id'] == case['id'] and result['workpoint'] == case['workpoint']
            assert dt == case['dt_ms'] and f == case['frequency_hz']
            assert rate.shape == (2, round(case['duration_ms'] / dt))
            assert np.isfinite(rate).all() and rate.min() >= -1e-10
            difference = rate[0] - rate[1]; amplitude = float(data['amplitude'])
            assert amplitude == case['absolute_amplitude']
            if f:
                phase = 2 * np.pi * f / 1000 * ((np.arange(rate.shape[1]) + 1) * dt + float(data['burn_ms']))
                gain = complex(np.dot(difference, np.sin(phase)), np.dot(difference, np.cos(phase))) * dt / case['duration_ms'] / amplitude
            else:
                gain = complex(difference.sum() * dt / case['duration_ms'] / (2 * amplitude))
            delta = abs(gain - complex(*result['predicted'])); max_difference = max(max_difference, delta)
            assert delta < 1e-9
            error = abs(gain - complex(*case['reference']['measured'])) / max(abs(case['reference']['dc_measured']), 1e-12)
            assert abs(error - result['normalized_error']) < 1e-8
            values[grid] = gain
            details.append(dict(grid=grid, gain=[gain.real, gain.imag], error=float(error),
                                passed=bool(error <= case['reference']['tol']), numerical_pass=result['numerical_pass']))
            completed += grid != 256
        changes = {}
        for low, high in [(256, 512), (512, 1024)]:
            if low in values and high in values:
                changes[f'{low}_to_{high}'] = float(abs(values[high]-values[low]) / max(abs(case['reference']['dc_measured']), 1e-12))
        rows.append(dict(case_id=case['id'], population=case['workpoint']['pop'],
                         channel=case['channel'], frequency_hz=case['frequency_hz'],
                         counted=case['reference']['counted'], tolerance=case['reference']['tol'],
                         levels=details, normalized_changes=changes,
                         converged_at_registered_tolerance=(changes['512_to_1024'] <= .02) if '512_to_1024' in changes else None))
    final = [level for row in rows for level in row['levels'] if level['grid'] == 1024 and row['counted']]
    result = dict(status='GRID_DIAGNOSTIC_COMPLETE' if completed == 24 else 'GRID_DIAGNOSTIC_PENDING',
                  completed=completed, expected=24, max_independent_demodulation_difference=max_difference,
                  finest_counted=len(final), finest_failed=sum(not x['passed'] for x in final),
                  converged_rows=sum(r['converged_at_registered_tolerance'] is True for r in rows),
                  rows=rows, model_promoted=False,
                  scope=contract['scope'], selection=contract['selection'])
    (dest / 'independent_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
