#!/usr/bin/env python3
"""Connect the observed spatial-field effect to the original G/Z equations."""
import math
from campaign import ROOT, NATIVE, read, write, sha


def main():
    pairs = []
    for root, name in [(NATIVE, 'exit_z0.21_k9_high'),
                       (ROOT / 'exit_return_probes', 'exit_z0.21_k9_fields16p7_high')]:
        r = next(x for x in read(root / 'extended_analysis_summary.json')['rows'] if x['name'] == name)
        assert r['full_horizon'] and r['duration_s'] == 30.
        fb, job = r['tail_feedback'], r['job']
        threshold = job['threshold'] / (18. - job['global_reversal_mV'])
        # G has a nonnegative target. Between20ms left-endpoint samples it
        # cannot decay faster than the original exponential tauG update.
        continuous_lower = fb['G_raw']['minimum'] * math.exp(-.02 / job['global_tau_s'])
        all_blocked = continuous_lower >= threshold
        if all_blocked:
            for z, drift, eligible in zip(r['held_Z_mean_allE_A_B_other'],
                r['counterfactual_drift_mean_allE_A_B_other'], r['tail_Z_recovery_eligible_fraction_allE_A_B_other']):
                assert abs(drift[0] + z / job['tau_Z_s']) < 1e-11
                assert abs(eligible) < 1e-11
        pairs.append(dict(name=name, allE_A_B_rates_Hz=r['tail_mean_Hz'][:3],
            G_raw=fb['G_raw'], between_sample_G_lower_bound=continuous_lower,
            recovery_block_threshold=threshold, all_E_recovery_blocked_through_tail=all_blocked,
            natural_Z_drift_allE_A_B=[x[0] for x in r['counterfactual_drift_mean_allE_A_B_other'][:3]],
            recovery_eligibility_allE_A_B=r['tail_Z_recovery_eligible_fraction_allE_A_B_other'][:3]))
    assert not pairs[0]['all_E_recovery_blocked_through_tail'] and pairs[1]['all_E_recovery_blocked_through_tail']
    write(ROOT / 'exit_field_feedback_audit/analysis.json', dict(status='COMPLETE_EXISTING_NATIVE_RECORDS',
        rows=pairs, producer_sha256=sha(__file__),
        interpretation='PairedZ/Kspatialfieldreplacement raises persistentactivity and thereby maintainsG above the all-cell recovery-block limit. This exactly accountsfor negativeZdrift in the actualfieldconditionalbranch. It doesnot identify which of the jointly changedZ/Kfields isresponsible, orprove an autonomous exit or bifurcation.',
        bounds='NonnegativeGtarget and exactexpdecay boundGraw below overeach20ms interval. Native rawII>=0 then forces allE Ztargets tozero. Naturaldrift isread whileZ/Kremainclamped.',
        formal_bifurcation_allowed=False))
    print(pairs, flush=True)


if __name__ == '__main__':
    main()
