"""Validate native data conservation and parent-sized renderer correspondence."""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from render_native_de import render_de


def validate(output):
    source=Path(__file__).parent;data=source/'native_de_data'
    info=json.loads((data/'observables.json').read_text())
    fig=plt.figure(figsize=(320/25.4,245/25.4))
    rects={'D':(10/320,130/245,147/320,66/245),'E':(163/320,130/245,147/320,66/245)}
    result=render_de(fig,rects['D'],rects['E'])
    fig.canvas.draw()
    ids=np.asarray(result['raster_neuron_ids']);begin,end=result['window_ms']
    full_end=info['analysis_and_display_window_ms'][1]
    checks={}
    checks['resolved_duration_matches_execution']=json.loads((data/'params.json').read_text())['T']==json.loads((data/'design.json').read_text())['duration_ms']==10000.
    manifest=json.loads((data/'export_manifest.json').read_text())
    checks['export_hashes']=all(hashlib.sha256((data/name).read_bytes()).hexdigest()==expected for name,expected in manifest.items())
    p=json.loads((data/'threshold_pair_validation.json').read_text())
    checks['diagnostic_exact_input_pairing']=all(p[k] for k in ['same_external_stream','same_final_rng','deterministic_input_exact','complete_duration'])
    execution=json.loads((data/'E_execution.json').read_text())
    checks['historical_native_prefix']=all(execution['historical_prefix_parity'].values())
    events=info['arms']['E']['native_population_events']
    checks['complete_events_not_terminal_censored']=all(e['end_ms']<full_end for group in events.values() for e in group['events'])
    checks['terminal_global_event_retained_as_censored']=len(events['global_E']['right_censored_events'])==1
    checks['26_local_short_bursts']=events['input_patch_E']['short_burst_count']==26 and events['input_patch_E']['prolonged_count']==0
    checks['local_burst_durations_20_to_40ms']=all(20<=e['duration_ms']<=40 for e in events['input_patch_E']['events'])
    for arm in ['D','E']:
        ax=result['axes'][arm]
        with np.load(data/f'{arm}_native.npz') as z:
            dt=float(z['dt_ms']);t=z['spike_steps']*dt
            mask=(t>=begin)&(t<end)&np.isin(z['spike_ids'],ids)
            plotted=sum(len(c.get_offsets()) for c in ax['raster'].collections)
            checks[arm+'_all_ROI_spikes_plotted']=int(mask.sum())==plotted
            exact=z['lfp_proxy'][round(begin/dt):round(end/dt),0]
            checks[arm+'_LFP_exact_native_samples']=bool(np.array_equal(exact,ax['lfp'].lines[0].get_ydata()))
            counts=np.bincount(z['spike_ids'][(t>=begin)&(t<end)],minlength=info['n_E']+info['n_I'])
            checks[arm+'_E_spike_count']=bool(np.isclose(counts[:info['n_E']].sum()/info['n_E']/((end-begin)/1000),z['rate_E'][round(begin/dt):round(end/dt)].mean(),rtol=1e-10))
        b=ax['field'].get_position();sizes=np.asarray([b.width*320,b.height*245])
        checks[arm+'_45mm_square']=bool(np.allclose(sizes,[44.982,44.982],atol=.05))
    checks['identical_raster_y']=result['axes']['D']['raster'].get_ylim()==result['axes']['E']['raster'].get_ylim()
    checks['independent_LFP_limits_explicit']=result['lfp_axes_independently_scaled'] and result['axes']['D']['lfp'].get_ylim()!=result['axes']['E']['lfp'].get_ylim()
    checks['LFP_labels_requested']=all(result['axes'][arm]['lfp'].get_ylabel()=='LFP (a.u.)' for arm in ['D','E'])
    checks['matched_time_window']=all(result['axes'][a][kind].get_xlim()==(begin/1000,end/1000) for a in ['D','E'] for kind in ['lfp','raster'])
    checks['E_red_I_blue']=result['colors']==dict(E='#c84040',I='#386c9d')
    checks['shared_threshold_scale_covers_actual_field']=result['threshold_lowering_limits_mV']==[0,4] and all(0<=lo<=hi<=4 for lo,hi in result['measured_lowering_range_mV'].values())
    checks['D_sparse_nonzero']=0<info['arms']['D']['E_rate_hz']<1 and 0<info['arms']['D']['I_rate_hz']<1
    checks['D_no_population_bursts']=all(g['count']==0 and len(g['right_censored_events'])==0 for g in info['arms']['D']['native_population_events'].values())
    checks['D_no_core_circle']=len(result['axes']['D']['field'].patches)==0
    checks['D_white_field']=result['axes']['D']['field'].get_facecolor()==(1.,1.,1.,1.)
    with np.load(data/'D_native.npz') as z: checks['D_uniform_background_threshold']=bool(np.all(z['vtheta']==18.))
    parent=source.parent.parent/'snn_layout_20261024/source/native_de_data/E_native.npz'
    checks['E_trajectory_byte_unchanged']=hashlib.sha256(parent.read_bytes()).digest()==hashlib.sha256((data/'E_native.npz').read_bytes()).digest()
    previous=source.parent.parent/'snn_layout_refined_20261024/source/native_de_data'
    checks['D_trajectory_byte_unchanged']=hashlib.sha256((previous/'D_native.npz').read_bytes()).digest()==hashlib.sha256((data/'D_native.npz').read_bytes()).digest()
    checks['matched_1400ms_display']=result['window_ms']==[1500.,2900.]
    displayed=[e for e in events['input_patch_E']['events'] if e['start_ms']>=begin and e['end_ms']<=end]
    checks['three_complete_local_bursts_displayed']=len(displayed)==3
    with np.load(data/'E_native.npz') as z, np.load(data/'substrate.npz') as sub:
        core=sub['input_patch_E']>=0
        overlay=next(c for c in result['axes']['E']['field'].collections if c.get_gid()=='native_core_threshold_overlay')
        checks['core_overlay_exact_thresholds']=bool(np.array_equal(overlay.get_array(),(18-z['vtheta'][:info['n_E']])[core]))
        checks['core_overlay_exact_positions']=bool(np.array_equal(overlay.get_offsets(),sub['positions_E'][core]))
    output.write_text(json.dumps(dict(checks=checks,all_pass=all(checks.values()),
        displayed_E_bursts=displayed,counts=result['neuron_counts'],window_ms=result['window_ms'],lfp_ylim=result['lfp_ylim'],
        acceptance='PENDING_AUTHOR_VISUAL_REVIEW'),indent=2)+'\n')
    plt.close(fig)
    print(output.read_text())
    if not all(checks.values()):raise RuntimeError('Native panel validation failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    validate(p.parse_args().output)
