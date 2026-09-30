"""Keep same-clock acceptance separate from descriptive entry coordinates."""
from common import *


def main():
    s = model(); dest = OUT / 'shared_variance_network_sensitivity'
    folder = dest / 'units_history_delay_covariance'
    review = read(dest / 'independent_comparison.json'); assert review['status'] == 'READOUT_AUDIT_PASS'
    row = review['rows'][-1]; assert row['label'] == 'units_history_delay_covariance'
    result = read(folder / 'result.json'); data = np.load(folder / 'trajectory.npz')
    win = row['complete_event_windows']['1000-9420']; duration = win['median_duration_ms']
    contract = read(BASE / 'a4_contract.json')
    native_D = read(BASE / 'native_reference/checkpoint_projections.json')['9870']['D']
    checks = dict(self_limited_events=bool(win['n'] > 0 and 50 <= duration <= 200 and result['quiet_fraction'] >= .15),
                  two_core_participation=bool(win['both_cores']/win['n'] >= .5),
                  surround_recruitment=bool(.3 <= win['median_area'] <= 1.),
                  propagation=bool(win['forward'] > 0 and win['reverse'] > 0 and 5 <= win['median_extent_mm'] <= 20),
                  entry=bool(result['high_onset_ms'] is not None and 7000 <= result['high_onset_ms'] <= 13000),
                  D_track=bool(abs(row['D_at_native_checkpoints']['9870'] - native_D) <= .05))
    same_clock = dict(status='PASS' if all(checks.values()) else 'PARTIAL', checks=checks,
                      n_passed=sum(checks.values()), n_total=len(checks), contract=str(BASE/'a4_contract.json'),
                      definitions=contract['acceptance'], complete_event_window=win,
                      quiet_fraction_whole_record=result['quiet_fraction'],
                      quiet_fraction_1_to_9p42_s=row['quiet_fraction_by_window']['1000-9420'],
                      D_native_9870=native_D, D_rate_9870=row['D_at_native_checkpoints']['9870'])
    assert same_clock['quiet_fraction_whole_record'] >= .15 and same_clock['quiet_fraction_1_to_9p42_s'] >= .15
    p = OUT.parent/'fig5_current_network_z_state_v1/replay/runs/eta0.0005_s9108401/checkpoints/t9870ms.npz'
    native = np.load(p)
    zn = np.ones(s.P); zn[s.E] = s.project(native['slow__z'][:32000])[s.E]
    mn = np.zeros(s.P); mn[s.E] = .0005*s.project(native['slow__m'][:32000])[s.E]
    assert abs(1-zn[s.E]@s.mean_weights-native_D) < 1e-12
    def describe(label, t, z, m):
        regions = []
        for k, name in enumerate(['Core A','Core B','Surround']):
            mask = s.E & (s.geo['group_region'] == k)
            regions.append(dict(region=name,Z=float(np.average(z[mask],weights=s.sizes[mask])),
                                M_current_mV=float(np.average(m[mask],weights=s.sizes[mask]))))
        return dict(label=label,time_ms=float(t),global_Z=float(z[s.E]@s.mean_weights),
                    D=float(1-z[s.E]@s.mean_weights),regions=regions,
                    group_Z_RMS_from_native_9870=float(np.sqrt(((z[s.E]-zn[s.E])**2)@s.mean_weights)))
    coords = [describe('native_near_entry',9870,zn,mn)]
    entry = result['high_onset_ms']; j = int(np.searchsorted(data['state_time_ms'],entry))
    assert 0 < j < len(data['state_time_ms'])
    for i in [j-1,j]:
        coords.append(describe('rate_entry_bracket',data['state_time_ms'][i],
                               data['Z'][i].astype(float),data['M_current'][i].astype(float)))
    assert coords[1]['time_ms'] <= entry <= coords[2]['time_ms']
    q = dict(status='DESCRIPTIVE_AND_ORIGINAL_CRITERIA_AUDIT_COMPLETE', same_clock_acceptance=same_clock,
             native_reference_checkpoint=str(p), rate_entry_ms=entry, resource_samples=coords,
             measurement='Exact native9.870s checkpoint versus two recorded rate samples bracketing its own operational entry. No resource interpolation.',
             scope='This event-aligned post-hoc description does not replace or relax the registered same-clock D criterion. One realization; resource proximity does not imply identical vector field, basin or bifurcation.',
             replacement_promoted=False,bifurcation_type='NOT_ESTABLISHED')
    write(dest/'entry_resource_coordinates.json',q)
    log('VARIANCE ENTRY COORDINATES',same_clock['n_passed'],'of',same_clock['n_total'],coords)


if __name__ == '__main__': main()
