"""Bounded sanitizer target for the actual near-entry spatial state.

One nominal and one passive-tangent step, with all full-state buffers retained.
This is a numerical diagnosis, not a dynamics or bifurcation result.
"""
from common import OUT,np,log
from onset_state_continuation import build
from fine_rate_frozen_Z_fields import capture,restore,arrays
from onset_tangent_cuda import Tangent
from onset_variational_return import Coordinates
import argparse


def main(device):
    p=OUT/'core_a_bifurcation_type_20260924/reference_stability_gap/actual_native_entry_midpoint/late_recurrence/three_burst_verified_replay/node01.npz'
    e=build(device);base=dict(np.load(p));restore(e,base)
    e.step();e.cp.cuda.get_current_stream().synchronize();expected=capture(e)
    restore(e,base);t=Tangent(e);t.graph(ms=e.dt);c=Coordinates(base,e.s)
    nominal=[(a.data.ptr,a.data.ptr+a.nbytes) for a in arrays(e).values()]
    for a in [t.syn,t.local,t.history,t.physical,t.arr,t.rate]:
        assert all(a.data.ptr+a.nbytes<=lo or a.data.ptr>=hi for lo,hi in nominal)
    restore(e,base);v=np.random.default_rng(92501).normal(size=c.size);v/=np.linalg.norm(v);c.set_tangent(t,v)
    assert all(np.array_equal(value,capture(e)[key]) for key,value in base.items())
    t.chunk();actual=capture(e)
    log('SANITIZER TARGET NOMINAL DIFFERENCES',{key:float(np.max(abs(value.astype(float)-expected[key].astype(float)))) for key,value in actual.items()})
    assert all(np.array_equal(value,expected[key]) for key,value in actual.items())
    log('SANITIZER TARGET ONE STEP PASS')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);main(p.parse_args().device)
