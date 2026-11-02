"""Export the new homogeneous-Poisson D; retain all frozen E numerical data."""
from pathlib import Path
import argparse, json, shutil, csv
import numpy as np
from export_native_de import sha, write, episodes


def main(run,dest):
    design=json.loads((run/'design.json').read_text())
    summary=json.loads((dest/'observables.json').read_text())
    begin,end=summary['analysis_and_display_window_ms'];duration_s=(end-begin)/1000.
    with np.load(dest/'substrate.npz') as z:
        ne,ni=len(z['positions_E']),len(z['positions_I']);patch=z['input_patch_E']>=0;ipatch=z['input_patch_I']>=0
    arm='D';source=run/'D0';path=source/'trajectory.npz';all_events=[]
    assert json.loads((source/'execution.json').read_text())['actual_duration_ms']==end
    shutil.copyfile(path,dest/'D_native.npz')
    for name in ['execution.json','contact_events.json']:shutil.copyfile(source/name,dest/('D_'+name))
    for name in ['design.json','applied_physics.json','source_manifest.json']:shutil.copyfile(run/name,dest/('D_poisson_'+name))
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
    summary.update(primary_D_identity='Zero core; all background thresholds and global weights; independent Poisson afferents to every E/I neuron at the previous silent-D mean; no local input patch or OU drive',
        primary_D_run=str(run.resolve()),statistical_unit='One 10 s native trajectory per condition on the same topology; E unchanged; not independent seed replication.',
        input_mean_boundary='D: independent Poisson at nu throughout; E: unchanged local max(0.95*nu+OU,0) Poisson patch and deterministic exterior. D/E are different working regimes, not a one-factor causal test.')
    summary['expectations'].update(D_low_rate_sparse=D['sparse_by_stated_definition'],D_low_rate_including_silence=D['low_rate_by_stated_thresholds'],D_completely_silent=D['no_spikes'],
        D_no_global_population_events=D['native_population_events']['global_E']['count']==0,
        D_no_patch_population_events=D['native_population_events']['input_patch_E']['count']==0,
        EI_balance_claim='Measured currents reported; sparse background alone is not evidence of balanced E/I currents.')
    write(dest/'observables.json',summary)
    delivery=json.loads((dest/'design.json').read_text())
    delivery.update(primary_D='D_native.npz = zero-core native network with homogeneous independent Poisson background',primary_E='E_native.npz unchanged from first candidate',
        contrast='Homogeneous noisy background vs localized one-core bursting regime; not an exactly input-paired core ablation.',
        D_background_rate_matched_to_previous_silent_D=True)
    delivery['source_runs']['D']=str(source.resolve())
    delivery['removed_from_D']['stochastic_input']='Local OU/Poisson patch removed; globally uniform independent Poisson background now active.'
    delivery['metadata_correction']=summary['input_mean_boundary']
    write(dest/'design.json',delivery)
    primary=json.loads((dest/'primary_validation.json').read_text())
    primary.update(stochastic_input_matching='Intentionally different regimes: homogeneous Poisson D vs original localized OU/Poisson E',matched_time_and_gain=False,same_time_window=True,LFP_vertical_limits_independent=True)
    write(dest/'primary_validation.json',primary)
    with (dest/'firing_rates.csv').open('w') as f:
        keys=['E_spikes','I_spikes','E_rate_hz','I_rate_hz','patch_E_rate_hz','patch_I_rate_hz','patch_E_spikes','patch_I_spikes','outside_E_rate_hz','mean_E_active_fraction_10ms','peak_E_active_fraction_10ms']
        w=csv.DictWriter(f,fieldnames=['arm','duration_s']+keys);w.writeheader()
        for arm in ['D','E']:w.writerow(dict(arm=arm,duration_s=duration_s,**{k:summary['arms'][arm][k] for k in keys}))
    oldevents=list(csv.DictReader((dest/'events.csv').open()));oldevents=[r for r in oldevents if r['arm']!='D']
    with (dest/'events.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['arm','detector','start_ms','end_ms','duration_ms','peak_active_fraction','short_burst']);w.writeheader();w.writerows(oldevents+all_events)
    write(dest/'export_manifest.json',{p.name:sha(p) for p in sorted(dest.iterdir()) if p.is_file() and p.name!='export_manifest.json'})
    print(json.dumps({k:D[k] for k in ['E_rate_hz','I_rate_hz','patch_E_rate_hz','patch_I_rate_hz','peak_E_active_fraction_10ms','current_balance','native_population_events']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('destination',type=Path);a=p.parse_args();main(a.run,a.destination)
