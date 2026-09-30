"""Equilibria reached by fixed native Z fields and their delayed stability."""
from native_path import *
from root_count_v3 import count, refine_root
from scipy.sparse.linalg import eigs
import argparse


def seed_equilibria(stability=False):
    s=model();path=attach_native_path(s);out=OUT/'equilibria/native_seeds';out.mkdir(parents=True,exist_ok=True)
    write(out/'path.json',path)
    rows=[]
    for t,Z in zip(path['times_ms'],path['fields']):
        src=BASE/f'stage_b/runs/c_relaxed_native_{t}.npz'
        z=np.load(src);s.set_Z(Z);rf=s.output(z['final_state'])
        candidates=[('instantaneous',rf),('tail_average',z['group_rate_hz'][-2000:].mean(0)/1000)]
        for folder in ['upper_cont','lower_cont','upper']:
            info=read(BASE/f'equilibria/{folder}/result.json')
            ordered=sorted(info['rows'],key=lambda x:abs(x['D']-s.D))[:1]
            for q in ordered:
                point=np.load(BASE/f'equilibria/{folder}/point{q["index"]:04d}.npz')
                candidates.append((folder,point['r']))
        attempts=[]
        for seed_name,seed in candidates:
            r,ok,tr=s.solve(seed)
            attempts.append(dict(seed=seed_name,converged=ok,trace=tr))
            if ok:break
        row=dict(time_source_ms=t,D=s.D,converged=ok,trace=tr,global_E_hz=s.global_rate(r),
                 regional_hz=s.regional_rates(r),final_time_rate_hz=s.global_rate(rf),
                 tail_group_temporal_relative_std=float(np.linalg.norm(z['group_rate_hz'][-2000:].std(0))/np.linalg.norm(z['group_rate_hz'][-2000:].mean(0))))
        row['attempts']=attempts
        if ok:
            y=s.equilibrium_state(r);f,rout=s.rhs(y,[A@r for A in s.matrices()],dynamic_z=False)
            row.update(full_rhs_max=float(abs(f).max()),rate_consistency_hz=float(abs(rout-r).max()*1000))
            assert abs(f).max()<1e-8 and abs(rout-r).max()<1e-9
            ev=eigs(s.jacobian(r),k=6,sigma=0,return_eigenvectors=False)
            row['static_eigenvalues']=[[v.real,v.imag] for v in ev]
            np.savez_compressed(out/f't{t}.npz',r=r,Z=Z,D=s.D,state=y)
            if stability:
                row['stability']=count(s,r,N=128)
        rows.append(row);log('SEED',row);write(out/'result.json',rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stability',action='store_true');a=p.parse_args()
    seed_equilibria(a.stability)
