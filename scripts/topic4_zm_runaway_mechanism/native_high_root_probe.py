"""Independent characteristic roots near the high-branch first turn.

A converged positive root proves instability, but sampled roots alone cannot
prove stability or count all unstable directions.
"""
from native_path import *
from equilibrium_spectrum import cache_characteristic
from root_count_v3 import refine_root


def main():
    src=OUT/'equilibria/native_high_descent';out=OUT/'equilibria/native_high_descent_audit'
    rows=[]
    for index in [118,119,96]:
        s=model();p=src/f'point{index:04d}.npz';z=np.load(p);r=z['r'];s.set_Z(z['Z'])
        assert abs(s.residual(r)).max()*1000<2e-8
        checks=cache_characteristic(s,r);roots=[]
        for guess in [.002+.02j,.002+.04j,.002+.08j,.002+.15j]:
            try:ans=refine_root(s,r,guess)
            except Exception as e:log('HIGH ROOT FAILURE',index,guess,repr(e));continue
            if ans is None:continue
            lam,v,error=ans
            if any(abs(lam-complex(*q['lambda_per_ms']))<1e-7 for q in roots):continue
            energy=s.E*s.sizes*abs(v)**2;energy/=energy.sum()
            root=dict(lambda_per_ms=[float(lam.real),float(lam.imag)],residual=error,
                frequency_hz=abs(float(lam.imag))*1000/(2*np.pi),
                mode_energy_A_B_surround=[float(energy[s.geo['group_region']==i].sum()) for i in range(3)])
            np.savez_compressed(out/f'point{index:04d}_root{len(roots)}.npz',r=r,Z=s.Z,D=s.D,v=v,lambda_per_ms=lam)
            roots.append(root);log('HIGH ROOT',index,root)
        q=dict(index=index,source=str(p),D=s.D,global_E_hz=s.global_rate(r),roots=roots,
            status='UNSTABLE_BY_POSITIVE_ROOT' if any(r['lambda_per_ms'][0]>1e-7 for r in roots) else 'SAMPLED_ROOTS_ONLY',
            checks=checks)
        rows.append(q);write(out/'root_probe.json',dict(status='RUNNING',rows=rows))
    write(out/'root_probe.json',dict(status='COMPLETE',rows=rows,
        scope='FrozenZ, dynamicM. Root search on an explicitly extended nativeZ path; not the observed onset. A positive complex root is not by itself a located Hopf crossing.'))


if __name__=='__main__':main()
