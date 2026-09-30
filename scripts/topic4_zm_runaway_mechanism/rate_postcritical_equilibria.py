"""Test whether broad post-transition activity is an equilibrium of the same path."""
from native_path import *


def main():
    s=model();attach_rate_entry_path(s);dest=OUT/'equilibria/rate_postcritical';dest.mkdir(parents=True,exist_ok=True);rows=[]
    for D in [.16,.20,.30]:
        source=OUT/'runs'/f'rate_postcritical_broad_D{D:.7f}_dt0.05'/'trajectory.npz'
        z=np.load(source);s.set_D(D);rates=z['group_rate_hz'][-2000:]/1000
        candidates=[('tail_mean',rates.mean(0)),('final_instantaneous',s.output(z['final_state']))];attempts=[]
        for name,seed in candidates:
            r,ok,trace=s.solve(seed)
            attempts.append(dict(seed=name,converged=bool(ok),trace=trace))
            if ok:break
        row=dict(D=D,source=str(source),attempts=attempts,converged=bool(ok),
                 tail_field_relative_std=float(np.linalg.norm(rates.std(0))/np.linalg.norm(rates.mean(0))))
        if ok:
            state=s.equilibrium_state(r);rhs,rr=s.rhs(state,[A@r for A in s.matrices()],dynamic_z=False)
            row.update(global_E_hz=s.global_rate(r),regional_hz=s.regional_rates(r),
                full_rhs_max=float(abs(rhs).max()),rate_consistency_hz=float(abs(rr-r).max()*1000),stability='NOT_ESTABLISHED')
            assert row['full_rhs_max']<1e-8 and row['rate_consistency_hz']<1e-6
            file=dest/f'rate_D{D:.5f}.npz';np.savez_compressed(file,r=r,Z=s.Z,D=D,state=state);row['path']=str(file)
        rows.append(row);write(dest/'result.json',dict(status='RUNNING',rows=rows));log('POSTCRITICAL EQUILIBRIUM',row)
    write(dest/'result.json',dict(status='COMPLETE',rows=rows,
        path_scope='Physical extrapolation of the local rate 7.7--7.8 s spatial Z direction; not the later actual autonomous Z trajectory'))


if __name__=='__main__':main()
