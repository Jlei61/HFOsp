#!/usr/bin/env python3
"""Verify cached delay bins and record the fixed clocks in the XY substrate."""
from pathlib import Path
import pickle
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_topic4_joint_xy_replicated_search as run
from src.snn_engine.params import Params


def delay_summary(counts, dt):
    counts = np.asarray(counts, dtype=np.int64)
    indices = np.arange(len(counts))
    quantiles = np.searchsorted(np.cumsum(counts),
                               np.array([.5,.9,.99])*counts.sum())*dt
    return {'edge_count': int(counts.sum()), 'delay_p50_p90_p99_ms': quantiles,
            'mean_delay_ms': float(np.dot(indices*dt,counts)/counts.sum()),
            'max_delay_ms': float(indices[counts>0][-1]*dt)}


def main():
    stage_path = ROOT/'results/topic4_sef_hfo/data_driven_core_field/config/stage_config.json'
    transition_path = ROOT/'config/topic4_rev22_dci_transition_execution.json'
    transition = run.read(transition_path)
    assert run.sha(stage_path) == transition['inputs']['stage_config']['sha256']
    engine = run.read(stage_path)['engine']
    params = Params(dt=engine['dt'],L=engine['L'],density=engine['density'],g=engine['g'])
    clocks = {k:getattr(params,k) for k in [
        'dt','delay_dt','tau0','v_axon','tau_m_E','tau_m_I','tau_ref_E','tau_ref_I',
        'tau_r_AMPA','tau_d_AMPA','tau_r_GABA','tau_d_GABA','tau_n']}
    assert params.dt == params.delay_dt
    rows = []
    sources = [stage_path, transition_path, Path(__file__)]
    for seed in range(2511,2519):
        record_path = run.v1.OLD/'network_records'/f'{seed}.json'
        record = run.read(record_path)
        cache = Path(record['path'])
        assert run.sha(cache) == record['sha256']
        with cache.open('rb') as f:
            payload = pickle.load(f)
        cfg = payload['config']
        for name in ['dt','delay_dt','tau0','v_axon','tau_m_E','tau_m_I','tau_r_AMPA','tau_r_GABA']:
            assert cfg[name] == clocks[name]
        net = payload['net']; ne = payload['NE']; pos = np.asarray(net['pos'])
        hist = {k:[] for k in ['E_to_E','E_to_I','I_to_E','I_to_I']}
        rng = np.random.default_rng(2026090600+seed)
        checked = 0; max_quantization_error = 0.
        for key, prefix, offset in [('ampa_by_delay','E',0),('gaba_by_delay','I',ne)]:
            for step, stored_matrix in enumerate(net[key]):
                # Cache bins may be CSC; inspect rows in a local CSR view.
                matrix = stored_matrix.tocsr(copy=False)
                first = int(matrix.indptr[ne]); total = matrix.nnz
                hist[prefix+'_to_E'].append(first)
                hist[prefix+'_to_I'].append(total-first)
                if not total:
                    continue
                edge = rng.choice(total,min(32,total),replace=False)
                dst = np.searchsorted(matrix.indptr,edge,side='right')-1
                src = matrix.indices[edge]+offset
                delay = params.tau0+np.linalg.norm(pos[dst]-pos[src],axis=1)/params.v_axon
                expected = np.maximum(1,np.round(delay/params.delay_dt).astype(int))
                np.testing.assert_array_equal(expected,np.full(len(edge),step))
                max_quantization_error = max(max_quantization_error,float(np.max(abs(delay-step*params.dt))))
                checked += len(edge)
        rows.append({'seed':seed,'network_sha256':record['sha256'],
                     'pathways':{k:delay_summary(v,params.dt) for k,v in hist.items()},
                     'sampled_edges_matching_distance_delay_recipe':checked,
                     'sampled_max_quantization_error_ms':max_quantization_error})
        sources.append(record_path)
        print(seed, rows[-1]['pathways']['E_to_E'], flush=True)
        del payload, net, pos
    source_code = ['src/snn_engine/params.py','src/snn_engine/kick_probe.py',
                   'src/snn_engine/connectivity_rot.py','src/topic4_zm_ictal_transition.py',
                   'src/sef_hfo_snn_adapter.py','scripts/run_topic4_rev12_node_worker.py']
    sources += [ROOT/p for p in source_code]
    locks = {**run.read(run.OUT/'objective_contract.json')['source_hashes'],
             **run.read(run.OUT/'analysis_input_lock.json')['hashes']}
    run.v1.runtime.verify_amendment(locks)
    out = run.OUT/'native_time_parameter_audit';out.mkdir(exist_ok=True)
    run.write(out/'summary.json',{'status':'NATIVE_TIME_PARAMETERS_AND_DELAY_BINS_VERIFIED',
        'effective_parameter_values':clocks,'spatial_ou':transition['spatial_ou'],
        'firing_density_observer':{'bin_ms':2.,'temporal_gaussian_sigma_ms':5.,'spatial_kernel_width_mm':.25},
        'networks':rows,'live_search_changed':False,'heldout_opened':False,
        'claim_boundary':'Delay quantiles count all stored edges; formula parity checks a reproducible sample per delay bin. '
                         'These are fixed model clocks, not fitted patient physiology or a causal diagnosis of slow propagation.',
        'source_hashes':{str(p):run.sha(p) for p in sources}})


if __name__ == '__main__':
    main()
