"""Time-step check for the spatial-refinement state discrepancy."""
from native_spatial_refinement import DEST, INITIAL, restriction_and_parent
from common import *
import runner
from endpoint_runs import EndpointIntegrator
import argparse


def main(device):
    contract=read(OUT/'native_spatial_time_refinement_contract.json')
    s20=model(20);s40=model(40);Q,parent=restriction_and_parent(s20,s40)
    original=np.load(INITIAL);small=np.load(INITIAL.with_name('seed_N1024_dt0.025.npz'))
    assert Path(str(original['source'])).resolve()==Path(str(small['source'])).resolve()
    assert float(small['dt_ms'])==.025 and int(small['tick'])==0
    i=(-np.arange(len(original['history'])))%len(original['history'])
    j=(-2*np.arange(len(original['history'])))%len(small['history'])
    assert np.array_equal(original['history'][i],small['history'][j])
    assert np.max(abs(original['state']-small['state']))<1e-11
    lifted=np.load(DEST/'initial_g40.npz');reference=np.load(OUT/contract['reference']/'trajectory.npz')
    state=lifted['state'];history=small['history'][:,parent]
    assert np.array_equal(state,original['state'][:,parent])
    initial=DEST/'initial_g40_dt0.025.npz'
    np.savez_compressed(initial,state=state,history=history,tick=0,dt_ms=.025,
        source=str(INITIAL),history_source=str(INITIAL.with_name('seed_N1024_dt0.025.npz')))
    write(DEST/'time_refinement_preparation.json',dict(status='PASS',
        state_bitwise_equal=True,shared_time_history_bitwise_equal=True,
        original_finer_reconstruction_state_roundoff=float(np.max(abs(original['state']-small['state']))),
        action='Use original coarse-step state exactly, and exact finer history samples.'))
    runner.Integrator=EndpointIntegrator
    q=runner.run_condition(s40,reference['Z_source'],
        'native_g40_liftedZ_D0.2190000_dt0.025',12000,device,initial,dt=.025)
    write(DEST/'time_refinement_run.json',q)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0)
    main(p.parse_args().device)
