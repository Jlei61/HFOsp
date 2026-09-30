"""Verify unstable temporal eigenpairs at newly continued equilibria.

Positive roots certify instability only. Failure to find one never certifies
stability. Cached results refer to immutable continuation point files.
"""
from temporal_modes import *


def main(a):
    s=ZMRate(a.grid)
    parent=DEST/f'g{a.grid}'/a.branch
    dest=parent/'temporal_gap_modes';dest.mkdir(exist_ok=True)
    info=read(parent/'result.json')
    indices=list(range(0,len(info['rows']),a.stride))
    if indices[-1]!=len(info['rows'])-1:indices.append(len(info['rows'])-1)
    previous=read(dest/'result.json')['rows'] if (dest/'result.json').exists() else []
    cached={row['point']:row for row in previous}
    last=None;rows=[]
    for k in indices:
        name=f'point{k:04d}.npz'
        if name in cached:
            row=cached[name];rows.append(row)
            if (dest/name).exists():
                with np.load(dest/name) as z:last=(complex(z['lam']),z['vector'])
            continue
        with np.load(parent/name) as z:r=z['r'];D=float(z['D'])
        L=Linearization(s,r,D)
        delta=L.matrix(0)+s.jacobian(r,D)
        assert not delta.nnz or abs(delta.data).max()<1e-10
        starts=([last] if last is not None else [])+[(.01072+.02846j,None),(.015+.13j,None),(.01+.19j,None),(.01+0j,None)]
        attempts=[];accepted=None
        for lam,v in starts:
            try:
                found=refine(L,lam,v)
            except (ValueError,RuntimeError,FloatingPointError) as exc:
                attempts.append(dict(seed=[lam.real,lam.imag],error=str(exc)));continue
            if found is None:
                attempts.append(dict(seed=[lam.real,lam.imag],status='UNRESOLVED'));continue
            ll,vv,res,it=found
            attempts.append(dict(seed=[lam.real,lam.imag],root=ll,residual=res))
            if ll.real>1e-7 and res<1e-9:
                accepted=found;break
        row=dict(point=name,D=D,global_E_hz=s.global_rate(r),attempts=attempts,
            equilibrium_residual_hz=float(abs(s.residual(r,D)).max()*1000),
            equilibrium_stability='UNRESOLVED')
        if accepted is not None:
            ll,vv,res,it=accepted;last=(ll,vv)
            energy=s.sizes*abs(vv)**2;energy/=energy.sum()
            row.update(equilibrium_stability='UNSTABLE',lambda_per_ms=ll,
                frequency_hz=abs(ll.imag)*1000/(2*np.pi),characteristic_residual=res,
                E_mode_energy_A_B_surround=[float(energy[s.E&(s.geo['group_region']==i)].sum()) for i in range(3)],
                inhibitory_energy=float(energy[~s.E].sum()))
            np.savez_compressed(dest/name,lam=ll,vector=vv,D=D)
        rows.append(row)
        write(dest/'result.json',dict(status='RUNNING',rows=rows,spectrum_complete=False,
            scope='Original dynamic-M characteristic; selected roots only'))
        print(k,D,row['global_E_hz'],row['equilibrium_stability'],flush=True)
    write(dest/'result.json',dict(status='COMPLETE',rows=rows,spectrum_complete=False,
        scope='Original dynamic-M characteristic; selected roots only'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--branch',required=True)
    p.add_argument('--grid',type=int,default=20);p.add_argument('--stride',type=int,default=10)
    main(p.parse_args())
