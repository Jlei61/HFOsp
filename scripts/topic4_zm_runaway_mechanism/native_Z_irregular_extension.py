"""Longer held-native-Z continuation to distinguish escape from continued events.

This is a deterministic rate-model control along the native checkpoint Z path,
not a native SNN simulation. Survival is finite-time censoring, never a proof
of a chaotic attractor. Both the source fast state and delay history are kept.
"""
from native_path import *
from endpoint_runs import EndpointIntegrator
import runner,argparse


def main(a):
    s=model();attach_native_path(s);D=.2196;s.set_D(D)
    source=OUT/'runs/endpoint_D0.2196000_dt0.05/trajectory.npz'
    z=np.load(source)
    assert np.max(abs(s.Z-z['final_state'][11]))<1e-12
    source_contract=read(source.parent/'contract.json')
    assert source_contract['dt_ms']==.05 and source_contract['Z']=='held' and source_contract['M']=='dynamic'
    assert source_contract['rate_history_scheme']=='instantaneous rate at the labelled endpoint'
    state,history=checkpoint_initial(source)
    assert history.shape==(round(s.delays[-1]/.05)+1,s.P)
    initial=OUT/'initial_states/native_Z_D2196_source12s_dt005.npz'
    np.savez_compressed(initial,state=state,history=history,tick=0,dt_ms=.05,source=str(source))
    runner.Integrator=EndpointIntegrator
    label='native_Z_D2196_long_from12s_dt005'
    contract=dict(status='RUNNING',source=str(source),initial_source_elapsed_ms=12000,
        extension_ms=60000,D=D,global_Z=1-D,Z='held',M='dynamic',
        model='frozen spatial rate equations, native-checkpoint spatial Z path',
        question='Does the previously irregular self-terminating/intermediate activity escape during a further60s at fixed Z?',
        interpretation_rule='An escape is finite-time evidence; survival is censored and does not establish an asymptotic attractor or bifurcation type.')
    write(OUT/'native_Z_irregular_extension.json',contract)
    result=runner.run_condition(s,s.Z.copy(),label,60000,a.device,initial,
                               dt=.05,dynamic_z=False,dynamic_m=True)
    contract.update(status='EXECUTION_COMPLETE_READOUT_PENDING',result=result)
    write(OUT/'native_Z_irregular_extension.json',contract)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);main(p.parse_args())
