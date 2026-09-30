#!/usr/bin/env python3
"""Read-only raw-current observer replay of one frozen density realization.

No density physics source is edited. The full retained physical state and RNG
must equal the prior run, and every stored group observation must be unchanged.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import sys
import time
import numpy as np
from campaign import ROOT, REPO, read, write, sha
from density_spatial import DensityNetwork
sys.path.insert(0, str(REPO))

OUT = ROOT / 'density_contact_replay'
SOURCE = ROOT / 'density_spatial_onset/R2048_num927612'
STATE_KEYS = ['state', 'ref', 'history', 'rng', 'clock', 'global_state', 'accumulator']
GROUP_KEYS = ['group_rate_Hz', 'group_Z', 'group_M', 'group_K', 'group_IE',
              'group_applied_II', 'group_V', 'group_abs_current']
OBSERVER = r'''
extern "C" __global__ void observe(const double* state, const int* clock,
 double* output, int P, int R){
 int tick=clock[0]; if(tick%5!=0)return;
 int g=blockIdx.x,lane=threadIdx.x; double sum=0.;
 for(int k=lane;k<R;k+=128){const double* x=state+((long long)g*R+k)*8;
  sum+=fabs(x[2])+fabs(x[4]);}
 __shared__ double buf[128];buf[lane]=sum;__syncthreads();
 for(int k=64;k>0;k/=2){if(lane<k)buf[lane]+=buf[lane+k];__syncthreads();}
 if(lane==0)output[((tick/5-1)%20)*P+g]=buf[0]/R;
}
'''


class ObservedDensity(DensityNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.raw_output = self.cp.zeros((20, self.P))
        self.observe = self.cp.RawKernel(OBSERVER, 'observe', options=('--fmad=false',))

    def step(self):
        super().step()
        self.observe((self.P,), (128,), (self.state, self.clock, self.raw_output,
                                       np.int32(self.P), np.int32(self.R)))


def contact_weights(engine):
    """Original Eq9–11 kernel, integrated within the existing density groups."""
    from src.snn_engine.lfp import LFPRecorder
    from types import SimpleNamespace
    original = np.load(REPO / 'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz')
    pos = engine.geo['original_positions']
    assert np.array_equal(pos[:32000], original['positions_e'])
    recorder = LFPRecorder(SimpleNamespace(**engine.p), pos,
                           np.r_[np.zeros(32000), np.ones(8000)], sites=original['contact_xy'])
    matrix = np.zeros((engine.P, len(recorder.sites)))
    cg = engine.geo['cell_group'][:32000]
    for j, (ids, w) in enumerate(zip(recorder._idx, recorder._w)):
        np.add.at(matrix[:, j], cg[ids], w)
    assert np.allclose(matrix.sum(0), 1.) and not matrix[~engine.E].any()
    # Verify aggregation algebra independently on arbitrary group-constant currents.
    rng = np.random.default_rng(92527)
    ie = rng.normal(20., 7., engine.P); ii = rng.normal(10., 5., engine.P)
    expected = recorder.sample(ie[engine.geo['cell_group']], ii[engine.geo['cell_group']])
    error = float(abs((abs(ie) + abs(ii)) @ matrix - expected).max())
    assert error < 1e-12
    return matrix, error


def main(device):
    assert read(SOURCE / 'result.json')['status'] == 'COMPLETE'
    assert not (OUT / 'contract.json').exists()
    OUT.mkdir(exist_ok=True)
    write(OUT / 'contract.json', dict(
        status='REGISTERED_BEFORE_REPLAY', created_epoch=time.time(),
        question='Do native and density contact envelopes/ordering agree when the same raw-current observable and kernel are used?',
        design='Exactly one full12.5s replay of the existing2048/927612 trajectory; read-only observer every0.5ms; no new physical or numerical seed and no parameter change.',
        observable='Per particle abs(rawIE)+abs(rawII), averaged inside each existing group; original native Eq9–11 contact weights summed by group. Group spatial exchangeability is still an approximation; no Z gating is inserted.',
        gate='All original stored group observations and complete final physical state/RNG must equal the source. Kernel identity and independent group-constant current algebra checked. No contact correspondence claimed from observer implementation alone.',
        source=str(SOURCE), source_sha256={n:sha(SOURCE/n) for n in ['trajectory.npz','final_state.npz']},
        producer_sha256=sha(__file__), engine_sha256=sha(__import__('density_spatial').__file__),
        device=device, formal_bifurcation_allowed=False))
    start=time.time()
    e=ObservedDensity(replicas=2048,seed=927612,device=device,duration_ms=12500,gain=0.)
    weights, weight_error=contact_weights(e)
    e.graph()
    with np.load(SOURCE/'trajectory.npz') as z:
        expected=np.stack([z[k] for k in GROUP_KEYS],axis=1)
    contacts=[]
    for tick in range(0,12500,10):
        physical=e.chunk()
        assert np.array_equal(physical.astype('f4'),expected[tick:tick+10]), tick
        contacts.append(e.raw_output.get() @ weights)
        if (tick+10)%250==0:
            write(OUT/'progress.json',dict(status='RUNNING',pid=os.getpid(),time_ms=tick+10,elapsed_s=time.time()-start))
    equal={}
    with np.load(SOURCE/'final_state.npz') as z:
        for k in STATE_KEYS:
            equal[k]=bool(np.array_equal(getattr(e,k).get(),z[k]));assert equal[k], k
    np.savez_compressed(OUT/'contacts.npz',time_ms=np.arange(1,25001)*.5,
                        lfp_raw=np.concatenate(contacts),group_lfp_weights=weights)
    result=dict(status='COMPLETE',elapsed_s=time.time()-start,
                all_stored_group_observations_bitwise_equal=True,final_state_bitwise_equal=equal,
                independent_kernel_algebra_error=weight_error,
                contact_correspondence_certified=False,formal_bifurcation_allowed=False)
    write(OUT/'result.json',result);write(OUT/'progress.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0)
    main(p.parse_args().device)
