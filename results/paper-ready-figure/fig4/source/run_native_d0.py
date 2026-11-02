"""True zero-core native background; paired-network companion to fixed E.

Threshold lowering, stochastic input patch, and local weight multipliers are
removed together. The global anisotropic topology and delay map are preserved.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
import time
from run_native_de import REPO, WT, HISTORY, digest, write


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();ref=a.reference_run.resolve()
    if not out.is_relative_to(WT):raise ValueError('Unauthorized output')
    out.mkdir(parents=True,exist_ok=False)
    os.environ['TOPIC4_CORE_COUNT_OUT']=str(out);os.environ['MPLCONFIGDIR']=str(out/'mpl')
    sys.dont_write_bytecode=True
    sys.path[:0]=[str(REPO/'scripts/topic4_core_count_optimization'),str(WT),str(WT/'src/snn_engine')]
    import numpy as np
    import physics
    from params import compute_nu_theta
    from lfp import LFPRecorder
    from src.topic4_streaming_spike_readout import simulate_streaming
    from src.topic4_observation_repaired import observe
    candidate=json.loads((ref/'candidate.json').read_text())
    applied_ref=json.loads((ref/'applied_physics.json').read_text())
    cache=Path(applied_ref['graph']['global_cache'])
    (out/'global_graph_cache').mkdir();shutil.copyfile(cache,out/'global_graph_cache'/cache.name)
    design=json.loads((ref/'design.json').read_text())
    design.update(schema='fig4.native_d0.v1', reference_run=str(ref),
        contrast='D0 vs E removes all core-specific threshold, stochastic-afferent and local-weight physics together.',
        removed=dict(threshold_lowering='all E thresholds restored to background 18 mV',
            stochastic_input='all cells receive exact expected tonic arrivals; local OU and Poisson support empty; global OU loading zero',
            EE_same_core_scale=[candidate['parameters']['EE_same_core_scale'],1.],
            EE_core_to_out_scale=[candidate['parameters']['EE_core_to_out_scale'],1.]),
        retained='identical neuron positions, all connection partners and delays, global EE kernel angle/spreads, synaptic kinetics, tonic external mean, initial voltage and other E/I weights',
        pairing_boundary='Same seed and network topology. External streams intentionally differ because the entire stochastic patch is removed; not a pure threshold intervention.',
        zero_core_count=0, source_candidate_role='Global topology provenance only; no core masks/thresholds/weight blocks active in D0')
    write(out/'design.json',design)
    started=time.time();print('BUILDING D0',flush=True)
    sub,groups,loading,deterministic,applied,(ec,ic)=physics.build_unit(candidate,2511,847401)
    with (out/'global_graph_cache'/cache.name).open('rb') as f:ampa,kernel_audit=pickle.load(f)
    assert applied['graph']['stage_audits']['local_output']['enabled'] is False
    assert applied['graph']['stage_audits']['degree']['exact_noop']
    assert all(candidate['parameters'][name]==1 for name in ['EI_same_core_scale','IE_same_core_scale','II_same_core_scale','EE_out_to_out_scale'])
    # Restore immutable global-graph weights, rather than divide scaled floats.
    gaba=sub.net['gaba_by_delay'];length=max(len(ampa),len(gaba))
    ampa=physics.v2._pad(ampa,length,ampa[0].shape);gaba=physics.v2._pad(gaba,length,gaba[0].shape)
    sub.net['ampa_by_delay']=ampa;sub.net['gaba_by_delay']=gaba;sub.net['max_delay_steps']=length-1
    ne,ni=sub.n_e,sub.n_i;nt=ne+ni;dt=sub.params.dt
    sub.vtheta=np.full(nt,float(sub.engine['v_base']));sub.delta_vtheta=np.zeros(ne)
    sub.h_e=np.zeros(ne);sub.h_i=np.zeros(ni);sub.params.T=design['duration_ms']
    identity=dict(ampa_topology_sha256=physics.base._hash_sparse_bins(ampa,include_data=False),
        ampa_values_sha256=physics.base._hash_sparse_bins(ampa),gaba_values_sha256=physics.base._hash_sparse_bins(gaba),
        positions_E_sha256=physics.base.array_sha256(np.asarray(sub.positions_e,np.float32)))
    for key in ['ampa_topology_sha256','gaba_values_sha256','positions_E_sha256']:
        assert identity[key]==applied_ref['identity'][key],key
    write(out/'applied_physics.json',dict(core_count=0,n_lowered=0,n_raised=0,n_stochastic=0,n_deterministic=nt,
        core_specific_weight_multipliers={'EE_same_core_scale':1.,'EE_core_to_out_scale':1.},
        identity=identity,retained_global_kernel=kernel_audit,reference_identity=applied_ref['identity']))
    patch=np.flatnonzero(ec>=0)
    sites=np.asarray([candidate['centers_mm'][0],np.asarray(candidate['centers_mm'][0])+[2.,0.]])
    signal=float(sub.params.nu_ext_ratio*compute_nu_theta(sub.params)[0])
    class Recorder:
        def __init__(self):
            self.steps=[];self.ids=[];self.trace=[];self.max_error=0.;self.stream=hashlib.sha256()
            self.groups={'allE':np.arange(ne),'patchE':patch,'outsideE':np.flatnonzero(ec<0),'allI':np.arange(ne,nt)}
        def observe(self,t,tm,xi,nu_now,ext,delta,V,IE,II,spk):
            self.max_error=max(self.max_error,float(np.max(abs(ext-signal*dt))))
            self.stream.update(np.asarray([xi,nu_now],'<f8').tobytes())
            ids=np.flatnonzero(spk).astype(np.uint32)
            if len(ids):self.steps.append(np.full(len(ids),t,np.uint32));self.ids.append(ids)
            if t%round(1/dt)==0:
                row=[tm,0.,signal]
                for idx in self.groups.values():
                    e,i=IE[idx],II[idx];row += [float(e.mean()),float(i.mean()),float(abs(e-i).mean()),float(V[idx].mean())]
                self.trace.append(row)
            if (t+1)%round(1000/dt)==0:print('D0 simulated_ms',tm+dt,'wall_s',round(time.time()-started,1),flush=True)
    recorder=Recorder();sub.net['rng']=np.random.default_rng(847401)
    result=simulate_streaming(physics.base.simulate_kick,sub.params,sub.net,
        positions=sub.positions_e,montage=sub.montage,KICK_BOOST=0.,t_kick=1e9,
        V_th_per_neuron=sub.vtheta,slow=None,early_stop_runaway=True,**physics.base.RUNAWAY,
        global_ou_loading=np.zeros(nt),deterministic_external_mask=np.ones(nt,bool),
        step_observer=recorder,external_e_rate_drive=None,
        lfp_recorder=LFPRecorder(sub.params,sub.net['pos'],sub.net['labels'],sites=sites),dump_pathway_trace=True)
    spikes=result.pop('E_spk_bool');movie=spikes.native();env,envdt,_=spikes.envelope()
    observed=observe(env,float(envdt),json.loads((ref/'contact_observer.json').read_text()))
    cols=['time_ms','patch_ou_per_ms','patch_afferent_rate_per_ms']
    for group in recorder.groups:cols += [group+'_'+x for x in ['I_E','I_I','abs_net_I','V']]
    steps=np.concatenate(recorder.steps) if recorder.steps else np.array([],np.uint32)
    ids=np.concatenate(recorder.ids) if recorder.ids else np.array([],np.uint32)
    d=out/'D0';d.mkdir()
    np.savez_compressed(d/'trajectory.npz',spike_steps=steps,spike_ids=ids,dt_ms=dt,duration_ms=len(result['rate_E'])*dt,
        rate_E=result['rate_E'],rate_I=result['rate_I'],lfp_proxy=result['lfp_trace'],trace=np.asarray(recorder.trace),trace_columns=np.asarray(cols),
        contact_envelope=env,contact_envelope_dt_ms=envdt,contact_names=np.asarray(sub.contact_names),contact_xy_mm=sub.contact_xy,
        sheet_activity_counts=movie['activity_counts'],sheet_frame_ms=movie['frame_ms'],vtheta=sub.vtheta,
        **{'pathway_'+k:v for k,v in result['pathway_trace'].items()})
    write(d/'contact_events.json',observed)
    write(d/'execution.json',dict(status='COMPLETE',actual_duration_ms=len(result['rate_E'])*dt,
        runaway_early_stop_ms=result.get('runaway_early_stop_ms'),n_spikes=len(ids),deterministic_max_error=recorder.max_error,
        wall_seconds=time.time()-started,trajectory_sha256=digest(d/'trajectory.npz')))
    assert len(result['rate_E'])*dt==design['duration_ms'] and recorder.max_error==0
    shutil.copyfile(Path(__file__),out/'run_native_d0.snapshot.py')
    write(out/'source_manifest.json',{str(Path(__file__).resolve()):digest(__file__),
        str(Path(__file__).with_name('run_native_de.py')):digest(Path(__file__).with_name('run_native_de.py')),
        str(WT/'src/snn_engine/kick_probe_core_input_v2.py'):digest(WT/'src/snn_engine/kick_probe_core_input_v2.py')})
    print('COMPLETE D0',flush=True)


if __name__=='__main__':main()
