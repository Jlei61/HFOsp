"""Matched held-Z spatial readout; finite trajectories never become cycle branches."""
from common import *
from native_post_transition_geometry import describe
import subprocess,json


def main():
    batch=read(OUT/'endpoint_runs_native_postcritical.json')
    assert batch['status']=='COMPLETE'
    names=['endpoint_D0.2190000_dt0.05','endpoint_D0.2196000_dt0.05',
           'endpoint_D0.2220000_dt0.05','endpoint_D0.2288448_dt0.05',
           'endpoint_D0.2400000_dt0.05','endpoint_D0.2500000_dt0.05',
           'endpoint_D0.2563444_dt0.05']
    rows=[];initial=None
    for name in names:
        folder=OUT/'runs'/name;contract=read(folder/'contract.json')
        current=Path(contract['initial']).resolve()
        if initial is None:initial=current
        assert current==initial
        assert contract['dt_ms']==.05 and contract['duration_ms']==12000
        assert contract['Z']=='held' and contract['M']=='dynamic'
        assert contract['rate_history_scheme']=='instantaneous rate at the labelled endpoint'
        z=np.load(folder/'trajectory.npz');field=z['field_E_hz'];counts=z['cell_counts']
        # Native readouts and the frozen rate engine each have a common.py;
        # retain their established separate import contexts.
        q=json.loads(subprocess.check_output([sys.executable,str(HERE/'canonical_case_readout.py'),
                                              str(folder)],text=True))
        spatial=describe(folder/'trajectory.npz')
        cell=field.reshape(-1,10,field.shape[1]).mean(1);weight=counts/counts.sum()
        rate=cell@weight;occupation=(cell>=50)@weight
        mask=(rate>=200)&(occupation>=.75)
        edges=np.diff(np.r_[0,mask.astype(int),0])
        broad=next((int(a*10) for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))
                    if b-a>=20),None)
        rows.append(dict(D=contract['D_initial'],global_Z=1-contract['D_initial'],
                         canonical=q,spatial=spatial,
                         first_broad75_and200Hz200ms_start_ms=broad))
        log(name,q['category'],q['tail']['mean_rate_hz'],broad)
    out=dict(status='COMPLETE',initial=str(initial),rows=rows,
             model='Frozen v3 spatial rate; native checkpoint Z path; M dynamic',
             statistical_unit='One deterministic complete-history continuation per held spatial Z field',
             scope='Matched 12 s observations. Neither a continued periodic branch nor a bifurcation certificate; no asymptotic claim.')
    write(OUT/'native_postcritical_endpoint_audit.json',out)


if __name__=='__main__':main()
