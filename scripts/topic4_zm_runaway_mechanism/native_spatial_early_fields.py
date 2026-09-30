"""Two earlier actual native fields on the fixed0.5mm rate network."""
from native_spatial_refinement import DEST
from common import *
import runner
from endpoint_runs import EndpointIntegrator
import argparse


def main(device):
    c=read(OUT/'native_spatial_early_fields_contract.json')
    assert read(DEST/'preparation.json')['status']=='PREPARATION_PASS'
    fields=np.load(DEST/'native_fields_g40.npz');s=model(40)
    runner.Integrator=EndpointIntegrator;rows=[]
    for t,D in zip(c['times_ms'],c['D']):
        j=np.flatnonzero(fields['times_ms']==t).item();Z=fields['fields'][j]
        assert fields['D'][j]==D and abs(1-Z[s.E]@s.mean_weights-D)<2e-14
        dt=c['dt_ms'];initial=DEST/('initial_g40.npz' if dt==.05 else f'initial_g40_dt{dt}.npz')
        name=f'native_g40_nativeZ_t{t}_dt{dt}'
        rows.append(runner.run_condition(s,Z,name,12000,device,initial,dt=dt))
        write(DEST/'early_fields_batch.json',dict(status='RUNNING',rows=rows))
    write(DEST/'early_fields_batch.json',dict(status='COMPLETE',rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0)
    main(p.parse_args().device)
