#!/usr/bin/env python3
"""Paired 8-second p030 regression, changing only the sampled self-edge exclusion."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import pickle
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT,ROOT/'src/snn_engine'):sys.path.insert(0,str(path))
from src.topic4_zm_ictal_transition import build_substrate,load_round_config,make_external_drive
from kick_probe import simulate_kick


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['legacy','corrected'],required=True)
    args=parser.parse_args();start=time.time()
    art=Path('/home/honglab/leijiaxin/HFOsp')
    out=ROOT/'results/topic4_sef_hfo/substrate_autapse_correction'
    audit=json.loads((out/'graph_rebuild_audit.json').read_text())
    record=audit['old_cache'] if args.arm=='legacy' else audit['new_cache']
    path=Path(record.get('frozen_cache_path',record.get('path')))
    expected=record.get('cache_sha256',record.get('sha256'))
    if sha(path)!=expected:raise RuntimeError('graph hash changed')
    with open(path,'rb') as f:cached=pickle.load(f)
    manifest_path=art/'results/topic4_sef_hfo/data_driven_dual_core_interictal_identifiability/response_fit/final_execution_candidate_manifest.json'
    candidate=next(r for r in json.loads(manifest_path.read_text())['candidates'] if r['candidate_id']=='dci_p030')
    m=candidate['mechanisms']
    config_path=ROOT/'config/topic4_rev22_dci_transition_execution.json'
    config=load_round_config(config_path)
    def selected_cache(*unused_args,**unused_kwargs):
        return cached['net'],cached['NE'],cached['NI'],True,{'frozen_cache_path':str(path),'cache_sha256':expected,'explicit_correction_arm':args.arm}
    # Replace only the loader, within this process; the production transformations,
    # parameters, Node field, contact geometry and OU implementation are unchanged.
    with patch('scripts.run_topic4_rev9_node_kick_canary._load_network',selected_cache):
        sub=build_substrate(config,'joint_04_control',2511,cache_dir=str(path.parent),
             ee_dose=m['g_EE'],etoi_dose=m['g_EtoI'],node_candidate_override=candidate['node_field'],
             ee_ellipse_angle_deg=m['ellipse_angle_deg'],ee_ellipse_aspect_ratio=m['ellipse_aspect_ratio'],
             ee_ellipse_reference_angle_deg=m['ellipse_reference_angle_deg'],
             ee_ellipse_reference_aspect_ratio=m['ellipse_reference_aspect_ratio'],
             artifact_root=art,topology_seed=2511,dynamics_seed=2511)
    del cached
    sub.params.T=8000.;sub.net['rng']=np.random.default_rng(2511)
    print(f'{args.arm}: start p030, 8 seconds, Z/M off, identical initial drive seed.',flush=True)
    result=simulate_kick(sub.params,sub.net,KICK_BOOST=0.,t_kick=1e9,
            V_th_per_neuron=sub.vtheta,slow=None,early_stop_runaway=False,
            external_e_rate_drive=make_external_drive(sub,config['spatial_ou'],2511))
    re=np.asarray(result['rate_E'],np.float32);ri=np.asarray(result['rate_I'],np.float32)
    if not np.isfinite(re).all() or not np.isfinite(ri).all():raise RuntimeError('nonfinite rate')
    dt=float(sub.params.dt);n=int(round(20/dt))
    r20=re[:len(re)//n*n].reshape(-1,n).mean(1)
    dest=out/'dynamic_check';dest.mkdir(exist_ok=True)
    arrays=dest/f'{args.arm}_p030_topology2511.npz'
    np.savez_compressed(arrays,rate_E_hz=re,rate_I_hz=ri,rate_E_20ms_hz=r20,dt_ms=dt)
    summary={'status':'DYNAMIC_CHECK_COMPLETE','arm':args.arm,'candidate_id':'dci_p030',
             'duration_ms':len(re)*dt,'topology_seed':2511,'dynamics_seed':2511,'Z_M':'off',
             'mean_E_rate_hz':float(re.mean()),'last_second_E_rate_hz':float(re[-int(round(1000/dt)):].mean()),
             'peak_20ms_E_rate_hz':float(r20.max()),'finite_rates':True,
             'arrays':{'path':str(arrays),'sha256':sha(arrays)},'graph':{'path':str(path),'sha256':expected},
             'source_hashes':{str(p):sha(p) for p in [config_path,manifest_path,Path(__file__),ROOT/'src/snn_engine/kick_probe.py']},
             'elapsed_seconds':time.time()-start,
             'claim_boundary':'Single-topology short dynamic regression. Does not establish patient fit, final geometry, Z/M transition or seizure reproduction.'}
    (dest/f'{args.arm}_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
