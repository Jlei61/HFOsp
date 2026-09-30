"""Independently check the conditional low equilibrium at physical D=0."""
from native_path import *
from equilibrium_spectrum import cache_characteristic
from root_count_v3 import refine_root, count


def main():
    source=OUT/'equilibria/native_low_seed/endpoint.npz'
    z=np.load(source);s=model();attach_native_path(s);s.set_D(float(z['D']));r=z['r']
    residual=float(abs(s.residual(r)).max()*1000)
    assert residual<2e-8 and np.max(abs(s.Z-z['Z']))<1e-13
    checks=cache_characteristic(s,r)
    dest=OUT/'equilibria/native_low_endpoint_stability';dest.mkdir(exist_ok=True)
    contract=dict(source=str(source),D=s.D,Z='held',M='dynamic',
        question='Is the low equilibrium already unstable at D=0, before later equilibrium turns?',
        methods='Independent characteristic root and refined right-half-plane contour count; do not inherit full Z/M stability')
    write(dest/'contract.json',contract)
    ans=refine_root(s,r,.01+.03j,tol=1e-11)
    roots=[]
    if ans is not None:
        lam,v,err=ans
        energy=s.E*s.sizes*abs(v)**2;energy/=energy.sum()
        roots.append(dict(lambda_per_ms=[lam.real,lam.imag],residual=err,
            frequency_hz=abs(lam.imag)*1000/(2*np.pi),
            mode_energy_A_B_surround=[float(energy[s.geo['group_region']==j].sum()) for j in range(3)]))
        np.savez_compressed(dest/'mode.npz',r=r,Z=s.Z,D=s.D,v=v,lambda_per_ms=lam)
    write(dest/'result.json',dict(status='COUNT_RUNNING',roots=roots,contract=contract))
    counted=count(s,r,N=256)
    unstable=any(q['lambda_per_ms'][0]>1e-7 for q in roots) or (counted['unstable_roots'] or 0)>0
    q=dict(status='COMPLETE',D=s.D,global_E_hz=s.global_rate(r),
        equilibrium_residual_hz=residual,roots=roots,contour=counted,checks=checks,
        classification='UNSTABLE' if unstable else 'STABLE_BY_CONTOUR' if counted['unstable_roots']==0 else 'NOT_ESTABLISHED',
        contract=contract,
        scope='One conditional low equilibrium at physical D=0; does not prove existence or stability of a bursting attractor, or locate the prior instability.')
    write(dest/'result.json',q);log('NATIVE LOW ENDPOINT STABILITY',q)


if __name__=='__main__':main()
