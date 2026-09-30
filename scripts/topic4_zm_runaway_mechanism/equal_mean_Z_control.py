"""Equal global depletion, different spatial Z: a matched finite-time control."""
from native_path import *
from endpoint_runs import EndpointIntegrator
import runner
import subprocess
import json
import argparse


def main(a):
    s=model();runner.Integrator=EndpointIntegrator
    source=OUT/'runs/actual_fine_Z_t7770_matched_history_dt005'
    c=read(source/'contract.json');native=np.load(source/'trajectory.npz')
    actual=native['Z_source'];D=float(1-actual[s.E]@s.mean_weights)
    attach_rate_entry_path(s);s.set_D(D);affine=s.Z.copy()
    assert abs((affine-actual)[s.E]@s.mean_weights)<1e-13
    assert c['Z']=='held' and c['M']=='dynamic' and c['duration_ms']==8000 and c['dt_ms']==.05
    label='equal_global_D_actual7770_affine_matched_history_dt005'
    runner.run_condition(s,affine,label,8000,a.device,c['initial'],dt=.05)
    rows=[]
    for name,folder in [('actual7770',source),('affine_same_global_D',OUT/'runs'/label)]:
        cc=read(folder/'contract.json')
        assert cc['initial']==c['initial'] and cc['duration_ms']==c['duration_ms'] and cc['dt_ms']==c['dt_ms']
        q=json.loads(subprocess.check_output([sys.executable,str(HERE/'canonical_case_readout.py'),str(folder)],text=True))
        rows.append(dict(condition=name,source=str(folder/'trajectory.npz'),readout=q))
    w=s.sizes*s.E;w/=w.sum()
    write(OUT/'equal_mean_Z_control.json',dict(status='COMPLETE',shared_initial=c['initial'],D=D,global_Z=1-D,
        spatial_Z_weighted_RMS_difference=float(np.sqrt(w@((affine-actual)**2))),
        region_Z_actual=(np.array(s.regional_rates(actual))/1000).tolist(),
        region_Z_affine=(np.array(s.regional_rates(affine))/1000).tolist(),rows=rows,
        Z='held in both conditions',M='dynamic in both conditions',
        question='Is the global mean depletion sufficient to predict the held-field finite-time state at this matched history?',
        limit='One matched deterministic history and finite 8-s window; this control does not classify a bifurcation or test all fields at this D.'))
    log('EQUAL MEAN Z CONTROL',rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);main(p.parse_args())
