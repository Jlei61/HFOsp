#!/usr/bin/env python3
"""Sixteen prespecified graph controls at shared spatial Z/K fields and histories."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import copy
import json
import shutil
import time
from pathlib import Path
import numpy as np
from scipy import sparse
import run_topic4_loop_axis_native as axis
from run_topic4_recovery_window import assert_same_state
from topic4_loop_locality_cpu import wrap_simulator

native=axis.native
PRIMARY=native.OUT
OUT=axis.ROOT/'conditional_runs'
CONTRACT=axis.ROOT/'conditional_control_contract.json'


def prepare(condition):
    dest=OUT/condition;path=dest/'protocol.json'
    if (dest/'queue.json').exists():return native.base.read(path)
    dest.mkdir(parents=True,exist_ok=True)
    contract=native.base.read(CONTRACT)
    audit=native.base.read(axis.graph_folder(condition)/'audit.json')
    assert condition=='reference' or audit['status']=='STATIC_CONTROL_PASS'
    p=copy.deepcopy(native.base.read(PRIMARY/'protocol.json'))
    p['identity'].update(ampa_values_sha256=audit['ampa_values_sha256'],
                         ampa_topology_sha256=audit['ampa_topology_sha256'])
    p.update(stage='SPATIAL_AXIS_CONDITIONAL_DIAGNOSTIC',axis_condition=condition,
        created_epoch=time.time(),deadline_epoch=time.time()+7*86400,
        axis_conditional_sha256=native.base.sha(__file__),contract_file=str(CONTRACT),
        contract_sha256=native.base.sha(CONTRACT),source_identity=audit['source_identity'],
        graph_audit=str(axis.graph_folder(condition)/'audit.json'),
        graph_audit_sha256=native.base.sha(axis.graph_folder(condition)/'audit.json'),
        maximum_grid_runs=8 if condition!='reference' else 2,maximum_all_new_scientific_runs=16,
        max_workers=4,initial_jobs=[],branch_jobs=[],
        question=contract['question'],conditional_control_contract=contract,
        source_network_switch='Only future EE deliveries use the replacement graph. The original populated delay rings, currents, voltages, refractory, M, G and RNG are carried; no claim of an equilibrium of the new graph at branch start.',
        diagnostic_only=True,counts_as_autonomous_loop=False,
        backend='Native ordered locality CPU; unchanged physical update,8threads; validated against original serial/CUDA trajectories.')
    native.write(path,p)
    shutil.copy2(PRIMARY/'geometry.npz',dest/'geometry.npz')
    previous_out,previous_prepare=native.OUT,native.prepare
    native.OUT=dest;native.prepare=lambda:p
    names=[];checks=[]
    try:
        points=contract['points'] if condition!='reference' else [dict(Z=.75,K=2.)]
        for point in points:
            for history,state in [('high','entry1_checkpoint.pkl'),('interictal','t50s.pkl')]:
                z,k=point['Z'],point['K']
                name=f'z{z:g}_k{k:g}_{history}' if condition!='reference' else f'qa_{history}'
                duration=30. if condition!='reference' else .2
                job=native.make_job(name,state,duration,True,z,k,common_input=True)
                cp=dest/'runs'/name/'checkpoint.pkl'
                saved=native.read_pickle(cp)
                assert saved['identity']==p['source_identity']
                # Preserve the complete physical state made by the already audited
                # branch constructor. Only declared graph/job metadata change here.
                engine_before=copy.deepcopy(saved['engine'])
                job.update(axis_condition=condition,stage='axis_conditional',
                    declared_graph_switch_at_s=job['branch_start_s'],
                    prior_graph_identity=p['source_identity'],new_graph_identity=p['identity'],
                    source_history=history,counts_as_autonomous_loop=False)
                saved['job']=job;saved['identity']=copy.deepcopy(p['identity'])
                native.base.save_pickle(cp,saved)
                native.write(dest/'jobs'/f'{name}.json',job)
                restored=native.read_pickle(cp)
                assert_same_state(engine_before,restored['engine'])
                assert restored['job']==job
                checks.append(dict(name=name,status='PASS',whole_engine_unchanged_by_graph_metadata=True,
                    source_checkpoint=job['source_checkpoint'],source_identity=p['source_identity'],
                    new_identity=p['identity']))
                names.append(name)
    finally:
        native.OUT,native.prepare=previous_out,previous_prepare
    native.write(dest/'prepared_state_qa.json',dict(status='PASS',checks=checks,scientific_runs=len(names) if condition!='reference' else 0))
    native.write(dest/'queue.json',dict(names=names,bounded=True,total=len(names),diagnostic_only=True))
    return p


def worker(condition,name):
    p=prepare(condition);dest=OUT/condition
    assert native.base.sha(__file__)==p['axis_conditional_sha256']
    assert native.base.sha(CONTRACT)==p['contract_sha256']
    assert native.base.sha(p['graph_audit'])==p['graph_audit_sha256']
    setup0=native.carrier.base.old.setup
    def setup(seed):
        s,tr,frozen,identity=setup0(seed)
        assert identity==p['source_identity']
        bins=[sparse.load_npz(path) for path in sorted((axis.graph_folder(condition)/'ampa_by_delay').glob('*.npz'))]
        assert len(bins)==len(s.net['ampa_by_delay'])
        assert axis.sparse_digest(bins)==p['identity']['ampa_values_sha256']
        assert axis.sparse_digest(bins,topology=True)==p['identity']['ampa_topology_sha256']
        for old,new in zip(s.net['ampa_by_delay'],bins):
            a,b=old.tocsr()[s.n_e:],new.tocsr()[s.n_e:]
            assert np.array_equal(a.indptr,b.indptr) and np.array_equal(a.indices,b.indices)
            assert np.array_equal(a.data,b.data)
        s.net=dict(s.net);s.net['ampa_by_delay']=bins
        removed=axis._invalidate_ampa_caches(s.net)
        native.write(dest/'runs'/name/'graph_loader_qa.json',dict(status='PASS',all_nonEE_unchanged=True,
            identity=p['identity'],invalidated_caches=removed,cold_start=False,declared_state_transfer=True))
        return s,tr,frozen,copy.deepcopy(p['identity'])
    native.carrier.base.old.setup=setup
    native.OUT=dest;native.prepare=lambda:p
    fixed0=native.fixed.worker
    def cpu_worker(target):
        native.carrier.wrap_simulator=wrap_simulator
        return fixed0(target)
    native.fixed.worker=cpu_worker
    native.worker(name)


def reference_qa():
    rows=[]
    for name,reference in [('qa_high','clamp_mechanism_qa'),('qa_interictal','clamp_input_qa')]:
        dest=OUT/'reference/runs'/name
        target=PRIMARY/'runs'/reference
        paths=list((dest/'chunks').glob('*.npz'));old=list((target/'chunks').glob('*.npz'))
        assert len(paths)==len(old)==1
        with np.load(paths[0]) as a,np.load(old[0]) as b:
            checks={key:bool(np.array_equal(a[key],b[key])) for key in
                    ['spikes_1ms','regions_1ms','field_5ms','raster','Z','M','currents','inputs']}
        assert all(checks.values()),checks
        a=native.read_pickle(dest/'checkpoint.pkl')['engine'];b=native.read_pickle(target/'checkpoint.pkl')['engine']
        assert_same_state(a,b)
        rows.append(dict(name=name,checks=checks,full_engine_bitwise=True))
    native.write(OUT/'reference/route_qa.json',dict(status='PASS',checks=rows,
        producer_sha256=native.base.sha(__file__),mechanism='Same cached reference graph plus same clamped initial states and new dispatch route reproduces both original full engine states and observations.'))
    print(json.dumps(rows),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','worker','reference-qa'])
    parser.add_argument('condition',nargs='?',choices=['reference','rotated','isotropic']);parser.add_argument('name',nargs='?')
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.condition)
    elif args.command=='worker':worker(args.condition,args.name)
    else:reference_qa()
