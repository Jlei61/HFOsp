"""Compare complete variational actions before/after gain-memory blocking."""
import rate_floquet as rf
from rate_floquet import *
import gc


def legacy_gains(orbit, full_moments, block_size=2048):
    cp=orbit.cp;n=full_moments.shape[1];P=orbit.s.P
    mom=cp.asarray(full_moments)
    def phi(values):
        out=cp.empty((n,P))
        orbit.phik(((n*P+127)//128,),(128,),
            (cp.ascontiguousarray(values),orbit.pars,out,np.int32(n)))
        return out
    gains=[]
    for k in range(3):
        step=1e-5*cp.maximum(cp.abs(mom[k]),1.)
        hi=mom.copy();lo=mom.copy();hi[k]+=step;lo[k]-=step
        gains.append((phi(hi)-phi(lo))/(2*step))
    return cp.stack(gains,axis=1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0)
    a=p.parse_args();s=RateField()
    path=PERIODIC_OUT/'orbits/LPC_A_stage4_turn3_N2048.npz'
    rows=[];reference=None;gain_reference=None;new=rf.temporal_gain_blocks
    for name,fun in [('blocked',new),('legacy',legacy_gains)]:
        rf.temporal_gain_blocks=fun;m=rf.Monodromy(s,path,.05,a.device)
        phase=m.phase_vector();random=np.random.default_rng(8129).normal(size=m.dim)
        outputs=np.array([m.matvec(phase),m.matvec(random)])
        row=dict(method=name,phase_defect=float(np.linalg.norm(outputs[0]-phase)/np.linalg.norm(phase)))
        if reference is None:
            reference=outputs;gain_reference=m.gains.get()
        else:
            row.update(maximum_absolute_difference=float(np.max(abs(outputs-reference))),
                       relative_action_difference=float(np.linalg.norm(outputs-reference)/np.linalg.norm(reference)),
                       maximum_gain_difference=float(np.max(abs(m.gains.get()-gain_reference))))
            # A finite-difference derivative amplifies floating-point
            # evaluation changes. Demand agreement far below the 1e-4
            # independent-mode tolerance rather than bitwise equality.
            assert row['relative_action_difference']<1e-9,row
        rows.append(row)
        cp=m.cp;del m;gc.collect();cp.fft.config.get_plan_cache().clear();cp.get_default_memory_pool().free_all_blocks()
    rf.temporal_gain_blocks=new
    prior=read(PERIODIC_OUT/'floquet/LPC_A_stage4_turn3_N2048_dt0.05.json')
    prior_error=abs(rows[0]['phase_defect']-prior['phase_tangent_relative_defect'])
    assert prior_error<1e-9,prior_error
    out=dict(status='PASS',orbit=str(path),rows=rows,stored_phase_defect_difference=prior_error,
        relative_action_tolerance=1e-9,stored_phase_defect_tolerance=1e-9,
        scope='Same complete delayed variational flow, two independent vectors, legacy full-array versus bounded-block gains. No model or temporal-mode change.')
    write(PERIODIC_OUT/'monodromy_gain_blocks_regression.json',out);print(out,flush=True)


if __name__=='__main__':main()
