#!/usr/bin/env python3
"""Native SNN counterpart of the autonomous corrected-rate high-state assay."""
from validate_topic4_fixed_rate_base import ROOT, setup, write, simulate_kick, spatial_cell_index
import argparse
import time
import resource
import numpy as np

OUT=ROOT/'results/topic4_sef_hfo/corrected_rate_high_activity_screen_v1'


def run(q,tau,duration,seed):
    start=time.time();s,tr,frozen,identity=setup(seed)
    # Parameter controls act after the frozen graph identity has been verified.
    for matrix in s.net['gaba_by_delay']:matrix.data*=q
    # The engine may otherwise reuse stale source-indexed weight caches.
    s.net.pop('ampa_flat',None);s.net.pop('gaba_flat',None)
    s.params.tau_d_GABA=tau;s.params.sigma_n=0.;s.params.T=duration
    s.net['rng']=np.random.default_rng(seed)
    result=simulate_kick(s.params,s.net,KICK_BOOST=0.,V_th_per_neuron=s.vtheta,
        slow=None,external_e_rate_drive=None,early_stop_runaway=False)
    dt=s.params.dt;spikes=result['E_spk_bool'];stride=int(round(1/dt));nf=len(spikes)//stride
    cells=spatial_cell_index(s.positions_e,n_grid=20,sheet_l_mm=20.);count=np.bincount(cells,minlength=400)
    field=np.empty((nf,400),np.float32)
    for k in range(nf):field[k]=np.bincount(cells,weights=spikes[k*stride:(k+1)*stride].sum(0),minlength=400)/count*1000
    dest=OUT/'snn';dest.mkdir(exist_ok=True);name=f'q{q:g}_gaba{tau:g}_seed{seed}_{duration:g}ms'
    np.savez_compressed(dest/f'{name}.npz',field_e_hz=field,rate_e_hz=result['rate_E'],rate_i_hz=result['rate_I'],dt_ms=dt,frame_ms=1.)
    write(dest/f'{name}.json',{'status':'COMPLETE','q':q,'tau_gaba_ms':tau,'duration_ms':duration,'seed':seed,
        'frozen_substrate_identity':identity,'q_scope':'all GABA jumps including I-to-E and I-to-I; cached flattened weights invalidated',
        'input':'constant nominal afferent rate; both global and spatial OU off; neuron-level Poisson spikes retained',
        'initialization':'native SNN cold start; distinct from rate high-history initialization, so low-rate accessibility is tested separately',
        'Z_M':'off','mean_e_hz_last_half':float(result['rate_E'][len(spikes)//2:].mean()),
        'seconds':time.time()-start,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2})
    print(name,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--q',type=float,default=.5);p.add_argument('--tau',type=float,default=9.);p.add_argument('--duration',type=float,default=3000.);p.add_argument('--seed',type=int,default=9108301);a=p.parse_args();run(a.q,a.tau,a.duration,a.seed)
