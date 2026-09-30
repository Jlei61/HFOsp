"""Local response support on the native-path candidate, not the affine path.

Weight operating points by cells, rate, and phase-removed family deformation.
The last measure is descriptive; a BVP tangent is not a Floquet critical mode.
"""
from native_path import *
from large_quadrature_periodic import LargeQuadratureGalerkin
from scipy.fft import irfft
from scipy.signal import resample
import argparse,gc


def main(a):
    source=OUT/'periodic/native_turn_center_fine_tangent_refined/point_with_tangent.npz'
    z=np.load(source);s=model();attach_native_path(s);s.set_D(float(z['D']))
    assert np.array_equal(s.Z,z['Z']) and float(z['residual'])<2e-8
    contract=read(OUT/'native_cycle_response_domain_contract.json')
    assert contract['samples']==[65536,131072]
    N=len(z['r']);r=z['r'];v=z['tangent'][:r.size].reshape(r.shape)
    weights=s.sizes*s.E;weights=weights/weights.sum()
    phase=np.fft.irfft(2j*np.pi*np.arange(N//2+1)[:,None]*
                      np.fft.rfft(r*1000,axis=0),n=N,axis=0)
    projection=np.sum(v*phase*weights)/np.sum(phase*phase*weights)
    deformation=v-projection*phase
    o=LargeQuadratureGalerkin(s,N,65536,a.device);o.cache_mean_operators=False
    cp=o.cp;cp.fft.config.get_plan_cache().set_size(0)
    h=o.harmonics(o.linear_inputs,cp.asarray(r),float(z['T']),s.Z).get()
    o.cache_key=None;o.cache=None;cp.get_default_memory_pool().free_all_blocks()
    rows=[]
    for M in contract['samples']:
        inp=np.empty((8,M,s.P))
        for k in range(8):inp[k]=irfft(np.ascontiguousarray(h[k].T),n=M,axis=1,workers=2).T*(M/N)
        for k in (0,3):inp[k]+=s.private_mu
        for k in (1,4,6):inp[k]+=s.private_ve
        deform=resample(deformation,M,axis=0)
        totals=np.zeros(3);outside=np.zeros(3);by_component={k:np.zeros(3) for k in ['mu_low','mu_high','sigma_E_high','sigma_I_high']}
        for lo in range(0,M,512):
            hi=min(lo+512,M);b=cp.asarray(inp[:,lo:hi]);n=(hi-lo)*s.P
            args=[cp.ascontiguousarray(q.ravel()) for q in b]
            ph=cp.empty((25,n));o.phik(((n+127)//128,),(128,),
                (*args,o.pars,o.consts,o.SE,o.SI,o.WE,o.WI,ph,np.int32(n)))
            rate=ph[0].reshape(hi-lo,s.P).get()
            mu,ve,vi=inp[:3,lo:hi,s.E];sc=(s.theta[s.E]-11)[None,:]
            table=s.resp.tables['E'];u=np.arcsinh((mu-11)/sc)
            masks=dict(mu_low=u<table.u[0],mu_high=u>table.u[-1],
                sigma_E_high=np.sqrt(np.maximum(ve,0))/sc>table.sEmax,
                sigma_I_high=np.sqrt(np.maximum(vi,0))/sc>table.sImax)
            allmask=np.logical_or.reduce(list(masks.values()))
            masses=[np.broadcast_to(weights[s.E],mu.shape),rate[:,s.E]*weights[s.E],
                    deform[lo:hi,s.E]**2*weights[s.E]]
            for j,mass in enumerate(masses):
                totals[j]+=mass.sum();outside[j]+=mass[allmask].sum()
                for key,mask in masks.items():by_component[key][j]+=mass[mask].sum()
            del b,args,ph
        assert np.all(totals>0)
        names=['E_cell_time_fraction','E_rate_mass_fraction','E_family_deformation_energy_fraction']
        q=dict(M=M,response_outside_tabulated_domain=dict(zip(names,(outside/totals).tolist())),
            components={key:dict(zip(names,(value/totals).tolist())) for key,value in by_component.items()})
        rows.append(q);log('NATIVE CYCLE RESPONSE DOMAIN',q)
        del inp,deform;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    difference={k:abs(rows[0]['response_outside_tabulated_domain'][k]-rows[1]['response_outside_tabulated_domain'][k])
                for k in rows[0]['response_outside_tabulated_domain']}
    q=dict(status='DESCRIPTIVE_COMPLETE',source=str(source),D=float(z['D']),T_ms=float(z['T']),N=N,
        rows=rows,absolute_sampling_differences=difference,
        phase_projection_removed=float(projection),
        observable='E-count weighted operating-domain support; family deformation is squared rate derivative with one common phase projected out',
        limitation='Domain coverage does not validate response accuracy. Family deformation is not a Floquet mode or proof of critical causality. Frozen equations unchanged.')
    write(OUT/'native_cycle_response_domain.json',q)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);main(p.parse_args())
