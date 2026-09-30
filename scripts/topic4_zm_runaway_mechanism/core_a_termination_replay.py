"""Exact, passive replay of the observed16s event's final5s.

No parameter intervention: expose the fast input/adaptation balance and save
full states immediately around termination for subsequent invariant analysis.
"""
from common import OUT,np,read,write,log
from onset_state_continuation import build
from fine_rate_frozen_Z_fields import capture,restore
import argparse,os,time

DEST=OUT/'core_a_bifurcation_type_20260924/termination_replay'
LOCAL=OUT/'core_a_transition_continuation_20260924'


def main(device):
    DEST.mkdir(exist_ok=True);assert not (DEST/'jobs.json').exists()
    source=LOCAL/'mid1_from_lower_ext/final_state.npz'
    expected=np.load(LOCAL/'mid1_from_lower_ext_long/block00.npz')['group_rate_hz']
    write(DEST/'contract.json',dict(question='Which exact full-state flow accompanies termination of the observed long Core A event?',
        source=str(source),physical_elapsed_window_ms=[15000,20000],
        controls='Unchanged same-field equations, full Z held, all M dynamic. Require every replayed1ms group rate to match the original float32 record bitwise.',
        interpretation='Passive mechanism diagnosis and source states for return-map analysis. Input correlations do not certify a bifurcation.',model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid(),elapsed_ms=0);write(DEST/'jobs.json',jobs)
    e=build(device);restore(e,np.load(source));Z=e.syn[5].get();currents=[];start=time.time()
    for k in range(500):
        r=e.chunk().astype('f4');assert np.array_equal(r[:,0],expected[k*10:(k+1)*10])
        currents.append(np.vstack([e.syn.get()[[1,3,4]],e.local.physical.get()[0]]))
        tm=15000+(k+1)*10
        if tm in [18700,18800,18900,18950,19000,19050,19100,19150,19200]:
            np.savez_compressed(DEST/f'state{tm}.npz',**capture(e))
        if (k+1)%100==0:
            jobs['elapsed_ms']=tm;write(DEST/'jobs.json',jobs);log('CORE A TERMINATION REPLAY',tm,time.time()-start)
    assert np.array_equal(e.syn[5].get(),Z)
    np.savez_compressed(DEST/'record.npz',time_ms=np.arange(15010,20001,10),Z=Z,
        currents_ampa_rawgaba_M_mu=np.array(currents),group_rate_hz=expected)
    write(DEST/'result.json',dict(status='EXACT_REPLAY_PASS',all5000ms_all3479_group_rates_bitwise=True,
        all_Z_held=True,all_M_dynamic=True,source=str(source),model_promoted=False))
    jobs['status']='COMPLETE';write(DEST/'jobs.json',jobs)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0);main(p.parse_args().device)
