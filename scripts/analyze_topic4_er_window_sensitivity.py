#!/usr/bin/env python3
"""Bounded diagnostic: onset-window and contact-reference effects on ER."""
from pathlib import Path
import json, sys, argparse
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_topic4_recurrence_readout as obs


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--readout',choices=['applied','legacy'],default='applied');args=parser.parse_args()
    legacy = args.readout=='legacy';stem = 'er_legacy_sensitivity' if legacy else 'er_sensitivity'
    out = obs.OUT
    protocol = {
        'question': 'Does the common ER mismatch depend on the one-second onset window or model contact referencing?',
        'windows_relative_onset_s': [[-1, 0], [0, 1], [1, 2]],
        'model_reference_variants': ['native_proxy', 'contact_CAR'],
        'fixed': 'Same trajectory, fixed E1146 SZ13, 1-150Hz, baseline -60 to -30s for every window.',
        'use': 'Diagnostic only; do not replace the main panel with the most favorable correlation.',
        'readout_source': 'lfp_raw' if legacy else 'lfp_effective',
    }
    obs.dump(out/f'{stem}_protocol.json', protocol)
    with np.load(obs.BASE/'weak_fast_z_refill_v1/runs/weak_fast_z_refill.npz') as f:
        model = f['lfp_raw' if legacy else 'lfp_effective']; names = f['contact_names'].copy()
    inv = obs.clinical._inventory_for_subject('epilepsiae', '1146')
    eeg = obs.clinical._eeg_offset_from_inventory('epilepsiae', inv[12])
    pre, post = obs.clinical._extract_bounds(eeg)
    patient = obs.clinical.extract_seizure_window('epilepsiae/1146', 12, pre_sec=pre, post_sec=post,
                                                reference=obs.clinical.ICTAL_REFERENCE['epilepsiae'])
    aliases = [obs.clinical.bipolar_alias_label(n) for n in patient.ch_names]
    px = patient.signal[[aliases.index(str(n)) for n in names]].T

    def er(x, fs, t0, onset, lo, hi):
        t = np.arange(len(x))/fs+t0-onset
        target = x[(t>=lo-1e-9)&(t<hi-1e-9)]
        baseline = x[(t>=-60-1e-9)&(t<-30-1e-9)]
        assert len(target)==round(fs) and len(baseline)==round(30*fs)
        return obs.power(target, fs)[0].mean(0)/obs.power(baseline, fs)[0].mean(0)

    rows = []
    for ref in protocol['model_reference_variants']:
        mx = model if ref=='native_proxy' else model-model.mean(1, keepdims=True)
        for lo, hi in protocol['windows_relative_onset_s']:
            m = er(mx, 2000, 0, 73.48, lo, hi)
            p = er(px, patient.fs, -patient.pre_sec, 0, lo, hi)
            rows.append({'model_reference': ref, 'window_s': [lo,hi], 'rho': float(spearmanr(m,p).statistic),
                         'model_ER_above1': int((m>1).sum()), 'patient_ER_above1': int((p>1).sum()),
                         'model_ER': m.tolist(), 'patient_ER': p.tolist()})
    obs.dump(out/f'{stem}.json', {'protocol': protocol, 'contacts': names.tolist(), 'clinical_reference_pool_n':len(patient.ch_names), 'clinical_reference_pool_names':list(patient.ch_names),'results': rows})
    print(json.dumps([{k:v for k,v in r.items() if k not in ('model_ER','patient_ER')} for r in rows], indent=2))


if __name__ == '__main__':
    main()
