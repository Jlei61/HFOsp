"""Numerical checks tied to the parameter and bifurcation claims."""
from common import *
from spectral import leading
from eigen import root_at
from branches import fold
import argparse


def equilibria():
    rows=[];base=read(OUT/'equilibrium_fold.json')
    for h in [0,.25,.5,.75,.95,.963,.964,.975,1.]:
        row=min(base,key=lambda q:abs(q['h']-h));h=row['h'];s=System(h)
        rf=np.array(row['r_hz'])/1000;g=row['g']
        e40=leading(s,rf,g,40);e64=leading(s,rf,g,64)
        exact=root_at(s,rf,g,e64[np.argmin(abs(e64))]);other=np.delete(e64,np.argmin(abs(e64)))
        before=[];r=np.array([.0001,.0002,0,0,0,0])
        for gg in [.5,.9,g-.02,g-.002,g-.00002]:
            # Track from the low state to avoid convergence to an unstable root.
            for gi in np.linspace(.5,gg,31):
                r,err,ok=s.solve(gi,r)
                if not ok:raise RuntimeError(('low branch refinement',h,gi))
            ee=leading(s,r,gg,40);ee64=leading(s,r,gg,64)
            # Stable uncoupled synaptic poles are roots of the full generator;
            # eliminating filters makes the 6x6 characteristic singular there.
            pole_distance=min(abs(ee[0]+1000/s.rise).min(),abs(ee[0]+1000/s.decay).min())
            rr=root_at(s,r,gg,ee[0]) if pole_distance>1e-5 else None
            before.append(dict(g=gg,leading_real_per_s=float(ee[0].real),leading_frequency_hz=float(abs(ee[0].imag)/(2*np.pi)),characteristic_residual=rr['residual'] if rr else None,
                               filter_pole=bool(pole_distance<=1e-5),generator_N40_N64_difference=float(abs(ee[0]-ee64[0]))))
        s64=System(h,groups=64);ff=fold(s64,np.r_[rf,g,row['v']])
        result=dict(h=h,g=g,zero_N40=[e40[np.argmin(abs(e40))].real,e40[np.argmin(abs(e40))].imag],
                    zero_N64=[e64[np.argmin(abs(e64))].real,e64[np.argmin(abs(e64))].imag],
                    exact_zero_residual=exact['residual'] if exact else None,
                    leading_other_N64_per_s=float(other.real.max()),stable_side=before,
                    quadrature64_g=ff['g'],quadrature_g_difference=ff['g']-g)
        rows.append(result);write('equilibrium_validation.json',rows);print('VALIDATE EQ',h,result['quadrature_g_difference'],flush=True)
    assert all(all(p['leading_real_per_s']<0 and (p['characteristic_residual'] is not None or p['filter_pole']) and p['generator_N40_N64_difference']<1e-5 for p in r['stable_side']) for r in rows)
    assert all(r['leading_other_N64_per_s']<0 and r['exact_zero_residual']<1e-8 and abs(r['quadrature_g_difference'])<1e-7 for r in rows)


def response():
    rows=[];mu=np.linspace(0,45,901)
    for h in (0,.25,.5,.75,1.):
        s=System(h);r=[]
        for value in mu:
            m=s.ext_mu.copy();m[0]=value
            r.append(s.phi(m,s.ext_var,np.zeros(6))[0]*1000)
        r=np.array(r);rows.append(dict(h=h,mu_mV=mu.tolist(),rate_hz=r.tolist(),gain_hz_per_mV=np.gradient(r,mu).tolist(),
            sigma_A_mV=s.original_std_A*h,mean_A_mV=s.mean_A,external_variance_mV2=float(s.ext_var[0])))
    write('population_response.json',rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['equilibria','response']);a=p.parse_args()
    globals()[a.kind]()
