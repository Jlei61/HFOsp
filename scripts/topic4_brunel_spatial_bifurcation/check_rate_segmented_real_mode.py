"""Match a real Floquet mode over every section, retaining full delay states."""
from verify_rate_real_mode import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True)
    p.add_argument('--N',type=int,required=True);p.add_argument('--device',type=int,default=0)
    p.add_argument('--dt',type=float,nargs='+',default=[.1,.05,.025]);p.add_argument('--segments',type=int,default=16)
    p.add_argument('--stream-harmonics',action='store_true')
    a=p.parse_args();q=read(PERIODIC_OUT/f'{a.label}_N{a.N}.json')
    assert q['status']=='EIGENPAIR_CONVERGED_CHECKS_PENDING'
    assert read(PERIODIC_OUT/'segmented_variational_flow_check.json')['status']=='PASS'
    z=np.load(PERIODIC_OUT/f'{a.label}_mode_N{a.N}.npz');growth=float(z['lam']);s=RateField()
    f=RealFloquet(s,q['orbit'],a.N,a.device,a.stream_harmonics);cp=f.cp
    data=harmonic_mode(f,z['u'],growth);phase=harmonic_mode(f,f.phase.get(),0.)
    del f;gc.collect();cp.get_default_memory_pool().free_all_blocks();rows=[]
    for dt in a.dt:
        m=Monodromy(s,q['orbit'],dt,a.device,capture=False,stream_harmonics=a.stream_harmonics)
        v0=at_time(data,m.D,m.dt);vT=at_time(data,m.D,m.dt,m.T);mu=np.exp(growth*m.T)
        boundary=float(np.linalg.norm(vT-mu*v0)/np.linalg.norm(v0))
        ph0=at_time(phase,m.D,m.dt)
        reconstruction=float(np.linalg.norm(ph0-m.phase_vector())/np.linalg.norm(ph0))
        assert boundary<1e-10 and reconstruction<1e-10
        edges=np.linspace(0,m.n,a.segments+1,dtype=int);matches=[]
        for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
            lo,hi=int(lo),int(hi);x=at_time(data,m.D,m.dt,lo*m.dt);y=at_time(data,m.D,m.dt,hi*m.dt)
            result=m.interval_matvec(x,lo,hi)
            phx=at_time(phase,m.D,m.dt,lo*m.dt);phy=at_time(phase,m.D,m.dt,hi*m.dt)
            phresult=m.interval_matvec(phx,lo,hi)
            v=dict(segment=i,start_ms=lo*m.dt,end_ms=hi*m.dt,
                mode_relative_defect=float(np.linalg.norm(result-y)/(np.linalg.norm(x)+np.linalg.norm(y))),
                phase_relative_defect=float(np.linalg.norm(phresult-phy)/(np.linalg.norm(phx)+np.linalg.norm(phy))))
            if i in [0,a.segments//2]:
                bad=np.roll(x.reshape(-1,s.P),1,axis=1).ravel();result=m.interval_matvec(bad,lo,hi)
                v['permuted_group_negative_control']=float(np.linalg.norm(result-y)/(np.linalg.norm(x)+np.linalg.norm(y)))
            matches.append(v)
        row=dict(dt_ms=m.dt,segments=a.segments,matches=matches,
            maximum_mode_relative_defect=max(v['mode_relative_defect'] for v in matches),
            maximum_phase_relative_defect=max(v['phase_relative_defect'] for v in matches),
            minimum_negative_control_defect=min(v['permuted_group_negative_control'] for v in matches if 'permuted_group_negative_control' in v),
            boundary_relative_error=boundary,phase_reconstruction_error=reconstruction)
        rows.append(row);print('SEGMENTED REAL MODE',{k:v for k,v in row.items() if k!='matches'},flush=True)
        del m;gc.collect();cp.get_default_memory_pool().free_all_blocks()
        write(PERIODIC_OUT/f'{a.label}_segmented_checks_N{a.N}.json',dict(status='COMPLETE' if len(rows)==len(a.dt) else 'RUNNING',
            label=a.label,orbit=q['orbit'],N=a.N,growth_per_ms=growth,multiplier=mu,checks=rows,
            scope='Each section integrates the independently reconstructed exponential-periodic mode in all nine states and the complete history; exact Floquet boundary checked separately. No complete spectrum is claimed.'))


if __name__=='__main__':main()
