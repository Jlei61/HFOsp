"""Export exact spikes, common-window observables and portable D/E source data."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def episodes(fraction, threshold, bin_ms=10., gap_ms=100.):
    active = np.flatnonzero(fraction >= threshold)
    merged = []
    for j in active:
        if merged and j*bin_ms-merged[-1][1] <= gap_ms:
            merged[-1][1] = float((j+1)*bin_ms)
        else:
            merged.append([float(j*bin_ms), float((j+1)*bin_ms)])
    return [dict(start_ms=a, end_ms=b, duration_ms=b-a,
                 peak_active_fraction=float(fraction[int(a/bin_ms):int(b/bin_ms)].max()),
                 short_burst=b-a<=500.) for a, b in merged]


def analyze(run, dest, d0, refresh_derived=False):
    design = json.loads((run/'design.json').read_text())
    pairing = json.loads((run/'paired_validation.json').read_text())
    if not all(pairing[k] for k in ['same_external_stream', 'same_final_rng', 'deterministic_input_exact', 'complete_duration']):
        raise RuntimeError('Unpaired or incomplete data')
    if refresh_derived:
        if json.loads((dest/'observables.json').read_text())['schema']!='fig4.native_de.observables.v1':raise ValueError('Not this task delivery')
    dest.mkdir(parents=True, exist_ok=refresh_derived)
    shutil.copyfile(run/'substrate.npz', dest/'substrate.npz')
    renamed={'design.json':'threshold_pair_design.json','params.json':'frozen_base_params.json','applied_physics.json':'E_applied_physics.json',
             'paired_validation.json':'threshold_pair_validation.json','source_manifest.json':'threshold_pair_source_manifest.json'}
    for name in ['design.json', 'params.json', 'applied_physics.json', 'paired_validation.json', 'candidate.json', 'contact_observer.json', 'source_manifest.json']:
        shutil.copyfile(run/name, dest/renamed.get(name,name))
    resolved_params=json.loads((run/'params.json').read_text())
    previous_T=resolved_params['T'];resolved_params['T']=design['duration_ms']
    write(dest/'params.json',resolved_params)
    write(dest/'parameter_resolution.json',dict(T_ms_before_runner_override=previous_T,T_ms_executed=design['duration_ms'],
        explanation='The runner records inherited base params before assigning params.T=duration_ms; params.json resolves the actually executed value and frozen_base_params.json retains the original record. Execution duration is independently confirmed by both native arrays and execution.json.'))
    for name in ['design.json','applied_physics.json','source_manifest.json']:
        shutil.copyfile(d0/name,dest/('D0_frozen_execution_design.json' if name=='design.json' else 'D0_'+name))
    d0_design=json.loads((d0/'design.json').read_text())
    correction='The frozen D0 execution design phrase same tonic external mean applies only outside the reference ROI. E ROI has max(0.95*nu + OU,0) Poisson input; D0 ROI restores deterministic nu. Both local mean and fluctuations change. No trajectory was altered or rerun.'
    corrected_retained='identical neuron positions, all connection partners and delays, global EE kernel angle/spreads, synaptic kinetics, initial voltage, other E/I weights, and outside-ROI tonic input; within-ROI mean and fluctuations intentionally change'
    corrected_d0=dict(d0_design,retained=corrected_retained,metadata_correction=correction)
    corrected_d0['removed']=dict(d0_design['removed'],local_input_mean='E: max(0.95*nu+OU,0) rate driving Poisson arrivals; D0: nu deterministic expected arrivals; mean is not matched')
    write(dest/'D0_design.json',corrected_d0)
    write(dest/'design.json',dict(schema='fig4.native_de.primary_delivery.v1',duration_ms=design['duration_ms'],
        burnin_ms=design['burnin_ms'],topology_seed=design['topology_seed'],dynamics_seed=design['dynamics_seed'],
        primary_D='D_native.npz = genuine D0 zero-core background',primary_E='E_native.npz = historical one-core canary',
        diagnostic='diagnostic_threshold_D_native.npz = threshold-only ablation with exact stochastic-input pairing to E; never label this no-core',
        removed_from_D=corrected_d0['removed'],retained=corrected_retained,metadata_correction=correction,
        pairing_boundary=d0_design['pairing_boundary'],source_candidate=design['source_candidate'],
        source_candidate_sha256=design['source_candidate_sha256'],selection=design['selection'],
        source_runs=dict(D=str(d0/'D0'),E=str(run/'E'),diagnostic_threshold_D=str(run/'D'))))
    di=json.loads((d0/'applied_physics.json').read_text())['identity']
    ei=json.loads((run/'applied_physics.json').read_text())['identity']
    write(dest/'primary_validation.json',dict(same_positions=di['positions_E_sha256']==ei['positions_E_sha256'],
        same_AMPA_partners_and_delays=di['ampa_topology_sha256']==ei['ampa_topology_sha256'],
        same_GABA_graph_and_weights=di['gaba_values_sha256']==ei['gaba_values_sha256'],
        AMPA_weights_differ_as_intended=di['ampa_values_sha256']!=ei['ampa_values_sha256'],
        stochastic_input_matching='INTENTIONALLY_NOT_MATCHED: D0 removes all core input; exact matching pertains only to threshold diagnostic',
        fixed_local_ROI_and_LFP_site=True,matched_time_and_gain=True))
    with np.load(run/'substrate.npz') as z:
        ne, ni = len(z['positions_E']), len(z['positions_I'])
        patch = z['input_patch_E'] >= 0
        ipatch = z['input_patch_I'] >= 0
    begin, end = design['burnin_ms'], design['duration_ms']
    duration_s = (end-begin)/1000.
    summary = dict(schema='fig4.native_de.observables.v1', native_run=str(run),
                   statistical_unit='One 10 s realization per condition on one shared topology; threshold diagnostic is paired on exact stochastic input; D0 vs E intentionally changes all core-specific physics.', analysis_and_display_window_ms=[begin, end],
                   primary_D_identity='D0 true zero-core; all local threshold, input and weight changes removed',
                   diagnostic_D_identity='Threshold-ablation only, NOT no-core; identical input stream to E',
                   primary_D_run=str(d0),
                   input_mean_boundary=correction,
                   n_E=ne, n_I=ni, n_patch_E=int(patch.sum()), n_patch_I=int(ipatch.sum()),
                   all_neurons_shown_in_raster=False, raster='All cells of the same fixed circular ROI, E then I, ascending original ID within type; no active-cell selection; E red and I blue.',
                   lfp_definition='LFPRecorder spatially weighted mean of abs(I_E)+abs(I_I) over E near the fixed original core centre; unfiltered native current proxy, not measured extracellular voltage.',
                   current_units='Voltage-equivalent model current (mV in LIF equation), not nA or extracellular mV.',
                   EI_balance_definition='Separate mean excitatory/inhibitory magnitudes, ratio I_I/I_E, signed residual and mean per-cell abs(I_E-I_I)/(mean I_E + mean I_I); read-only 1 ms samples. No balance inference from sparsity.',
                   event_definition=design['operational_burst'], arms={})
    all_events = []
    sources={'D':d0/'D0','E':run/'E','diagnostic_threshold_D':run/'D'}
    for arm, source in sources.items():
        path = source/'trajectory.npz'
        shutil.copyfile(path, dest/f'{arm}_native.npz')
        execution=json.loads((source/'execution.json').read_text())
        if execution['actual_duration_ms']!=end or execution['deterministic_max_error']!=0:raise RuntimeError('Incomplete or incorrect deterministic input')
        shutil.copyfile(source/'execution.json', dest/f'{arm}_execution.json')
        shutil.copyfile(source/'contact_events.json', dest/f'{arm}_contact_events.json')
        with np.load(path) as z:
            dt = float(z['dt_ms']); step=z['spike_steps']; ids=z['spike_ids']
            times = step.astype(float)*dt
            keep=(times>=begin)&(times<end)
            counts=np.bincount(ids[keep], minlength=ne+ni)
            if not np.isclose(counts[:ne].sum()/ne/duration_s, z['rate_E'][int(begin/dt):int(end/dt)].mean(), rtol=1e-10):
                raise RuntimeError('Exact spike counts disagree with native rate_E')
            if not np.isclose(counts[ne:].sum()/ni/duration_s, z['rate_I'][int(begin/dt):int(end/dt)].mean(), rtol=1e-10):
                raise RuntimeError('Exact spike counts disagree with native rate_I')
            nb=round(end/10.)
            e=ids<ne
            encoded=(step[e]//round(10/dt)).astype(np.int64)*ne+ids[e]
            unique=np.unique(encoded)
            frac=np.bincount(unique//ne,minlength=nb)/ne
            patchunique=unique[patch[unique%ne]]
            patchfrac=np.bincount(patchunique//ne,minlength=nb)/patch.sum()
            native_events={}
            for name, value, threshold in [('global_E',frac,.01), ('input_patch_E',patchfrac,.10)]:
                ev=episodes(value,threshold)
                eligible=[x for x in ev if x['start_ms']>=begin and x['end_ms']<end]
                native_events[name]=dict(threshold_distinct_fraction=threshold,
                    count=len(eligible), short_burst_count=sum(x['short_burst'] for x in eligible),
                    prolonged_count=sum(not x['short_burst'] for x in eligible), events=eligible,
                    boundary_excluded_count=len(ev)-len(eligible),
                    right_censored_events=[x for x in ev if x['start_ms']>=begin and x['end_ms']>=end],
                    boundary_rule='Events touching trajectory end are right-censored and excluded from complete burst counts; retain their observed interval as lower bound.')
                all_events += [dict(arm=arm, detector=name, **x) for x in eligible]
            columns=z['trace_columns'].tolist(); trace=z['trace']; sample=(trace[:,0]>=begin)&(trace[:,0]<end)
            balance={}
            for group in ['allE','patchE','outsideE','allI']:
                vals={suffix:float(trace[sample,columns.index(group+'_'+suffix)].mean()) for suffix in ['I_E','I_I','abs_net_I','V']}
                vals.update(inhibitory_to_excitatory_ratio=vals['I_I']/vals['I_E'],
                    mean_signed_net_current=vals['I_E']-vals['I_I'],
                    cancellation_residual=vals['abs_net_I']/(vals['I_E']+vals['I_I']))
                balance[group]=vals
            recurrent={}
            pt=z['pathway_time_ms']; pm=(pt>=begin)&(pt<end)
            for name in ['recurrent_E_to_E_mean','recurrent_E_to_I_mean','GABA_to_E_mean']:
                recurrent[name]=float(z['pathway_'+name][pm].mean())
            contact=json.loads((source/'contact_events.json').read_text())
            contact_eligible=[x for x in contact['events'] if x['window_ms'][0]>=begin and x['window_ms'][1]<=end]
            native_lfp=z['lfp_proxy'][int(begin/dt):int(end/dt),0]
            sm=frac[int(begin/10):int(end/10)]
            s=dict(E_spikes=int(counts[:ne].sum()), I_spikes=int(counts[ne:].sum()),
                E_rate_hz=float(counts[:ne].sum()/ne/duration_s), I_rate_hz=float(counts[ne:].sum()/ni/duration_s),
                patch_E_rate_hz=float(counts[:ne][patch].sum()/patch.sum()/duration_s),
                patch_I_rate_hz=float(counts[ne:][ipatch].sum()/ipatch.sum()/duration_s),
                patch_E_spikes=int(counts[:ne][patch].sum()),patch_I_spikes=int(counts[ne:][ipatch].sum()),
                outside_E_rate_hz=float(counts[:ne][~patch].sum()/(~patch).sum()/duration_s),
                fraction_E_ever_active=float(np.mean(counts[:ne]>0)),
                mean_E_active_fraction_10ms=float(sm.mean()), peak_E_active_fraction_10ms=float(sm.max()),
                no_spikes=bool(counts.sum()==0), native_population_events=native_events,
                frozen_contact_events_detected=len(contact_eligible),
                frozen_contact_events_primary=sum(x['primary_eligible'] for x in contact_eligible),
                current_balance=balance, pathway_currents=recurrent,
                LFP_proxy_min=float(native_lfp.min()), LFP_proxy_max=float(native_lfp.max()),
                input_ROI_afferent_mean_per_ms=float(trace[sample,columns.index('patch_afferent_rate_per_ms')].mean()),
                input_ROI_afferent_sd_per_ms=float(trace[sample,columns.index('patch_afferent_rate_per_ms')].std()),
                input_ROI_zero_rate_fraction=float(np.mean(trace[sample,columns.index('patch_afferent_rate_per_ms')]==0)),
                input_rate_sampling='1 ms read-only samples of actual nonnegative applied rate; not raw Poisson counts',
                full_trajectory_sha256=sha(path))
            s['low_rate_by_stated_thresholds']=s['E_rate_hz']<1 and s['mean_E_active_fraction_10ms']<.01
            s['sparse_by_stated_definition']=0<s['E_rate_hz']<1 and 0<s['mean_E_active_fraction_10ms']<.01
            summary['arms'][arm]=s
    D,E=summary['arms']['D'],summary['arms']['E']
    summary['expectations']=dict(D_low_rate_sparse=D['sparse_by_stated_definition'],
        D_low_rate_including_silence=D['low_rate_by_stated_thresholds'],D_completely_silent=D['no_spikes'],
        D_no_global_population_events=D['native_population_events']['global_E']['count']==0,
        D_no_patch_population_events=D['native_population_events']['input_patch_E']['count']==0,
        E_short_global_bursts=E['native_population_events']['global_E']['short_burst_count']>0,
        E_short_patch_bursts=E['native_population_events']['input_patch_E']['short_burst_count']>0,
        E_run_under_stochastic_input_without_kick=True,
        noise_necessity_tested=False, EI_balance_claim='NOT_ESTABLISHED; report currents directly',
        author_visual_acceptance='PENDING_AUTHOR_VISUAL_REVIEW')
    write(dest/'observables.json',summary)
    with (dest/'events.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['arm','detector','start_ms','end_ms','duration_ms','peak_active_fraction','short_burst'])
        w.writeheader();w.writerows(all_events)
    with (dest/'firing_rates.csv').open('w') as f:
        keys=['E_spikes','I_spikes','E_rate_hz','I_rate_hz','patch_E_rate_hz','patch_I_rate_hz','patch_E_spikes','patch_I_spikes','outside_E_rate_hz','mean_E_active_fraction_10ms','peak_E_active_fraction_10ms']
        w=csv.DictWriter(f,fieldnames=['arm','duration_s']+keys);w.writeheader()
        for arm in sources:w.writerow(dict(arm=arm,duration_s=duration_s,**{k:summary['arms'][arm][k] for k in keys}))
    write(dest/'export_manifest.json',{p.name:sha(p) for p in sorted(dest.iterdir()) if p.is_file() and p.name!='export_manifest.json'})
    print(json.dumps(summary['expectations'],indent=2))
    print((dest/'observables.json').resolve())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('destination',type=Path);p.add_argument('--d0',type=Path,required=True)
    p.add_argument('--refresh-derived',action='store_true')
    a=p.parse_args();analyze(a.run,a.destination,a.d0,a.refresh_derived)
