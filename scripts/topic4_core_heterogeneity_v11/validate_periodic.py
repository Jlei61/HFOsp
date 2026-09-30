"""Verify coexisting periodic states and their transverse stability."""
from from_scan import *
from integrate import simulate,orbit_state,describe
sys.path.append(str(ROOT/'scripts/topic4_core_network_bifurcation_v7'))
import analytic_poincare as poincare


def run(h,g,direction):
    path=refine(h,g,direction,1024);z=np.load(path);r=z['r'];T=float(z['T']);s=System(h)
    q,t=load_seed(path,2048);normal=np.zeros_like(q);normal[-1]=1
    q,tt,err,*_=Chart(s,q,normal,2048,.01).solve(q,0)
    r2=q[:-2].reshape(2048,6)*.01;T2=float(np.exp(q[-2]))
    p2=path.with_name(path.name.replace('N1024','N2048'))
    np.savez_compressed(p2,r=r2,T=T2,g=g,h=h,N=2048,tangent=tt,residual=err)
    refined=dict(period_difference_ms=T2-T,rate_difference_hz=float(abs(r2-resample(r,2048,axis=0)).max()*1000),residual=err)
    poincare.System=lambda:System(h)
    spectra=[poincare.compute(p2,dtmax=dt,method='rk4',nev=4) for dt in (.05,.025)]
    # Start from the same exact periodic history; changing dt requires a freshly
    # sampled history, rather than reusing indices from the coarser ring buffer.
    simulations=[]
    for dt in (.1,.05):
        rate,state=simulate(s,g,duration_ms=max(3000,6*T2),dt=dt,state=orbit_state(s,r2,T2,dt))
        d=describe(rate);d['dt_ms']=dt;simulations.append(d)
        np.savez_compressed(p2.with_name(p2.stem+f'_dt{dt:g}_trajectory.npz'),r=rate,sample_dt=.5,h=h,g=g)
    result=dict(h=h,g=g,direction=direction,source=str(p2.relative_to(ROOT)),grid_refinement=refined,
                poincare=spectra,time_step=simulations,
                stable_periodic_attractor=bool(all(a['max_transverse']<1 and a['orbit_tangent_defect']<.02 for a in spectra)))
    write(OUT/'validated_attractors'/f'h{h:.5f}_{direction}_g{g:.5f}.json',result)
    print('VALIDATED ATTRACTOR',h,g,direction,result['stable_periodic_attractor'],flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--h',type=float,default=0.);p.add_argument('--g',type=float,required=True);p.add_argument('--direction',default='up');a=p.parse_args()
    run(a.h,a.g,a.direction)
