"""Compare factored and dense projections; the nonlinear DDE is unchanged."""
from rate_invariant_torus import *


def main():
    s=RateField();o=Torus(s,32,8,0);cp=o.cp
    z=np.load(PERIODIC_OUT/'tori/arcTR2_0011_N64x32.npz');r=resample(resample(z['r'],32,axis=0),8,axis=1)
    q=cp.asarray(resample(z['q'],32,axis=0));dr=cp.fft.ifft(1j*o.kt[:,None,None]*cp.fft.fft(cp.asarray(r),axis=0),axis=0).real
    modes=[dr.mean(1),q.real,q.imag];sub=SeparableTorusSubspace(o,modes,3)
    raw=[];angle=2*cp.pi*cp.arange(8)/8
    for l in range(4):
        waves=[cp.cos(l*angle)]+([cp.sin(l*angle)] if l else [])
        for m in modes:
            for wave in waves:raw.append(cp.r_[(m[:,None,:]*wave[None,:,None]).ravel(),cp.zeros(3)])
    for j in range(3):
        v=cp.zeros(sub.size);v[-3+j]=1;raw.append(v)
    dense=cp.stack(raw,axis=1);Q,R=cp.linalg.qr(dense);rng=np.random.default_rng(890731)
    v=cp.asarray(rng.normal(size=sub.size));c=cp.asarray(rng.normal(size=sub.dimension))
    projection_error=float(cp.linalg.norm(sub.lift(sub.project(v))-Q@(Q.T@v))/cp.linalg.norm(Q@(Q.T@v)))
    orthogonality_error=float(cp.linalg.norm(sub.project(sub.lift(c))-c)/cp.linalg.norm(c))
    assert projection_error<1e-10 and orthogonality_error<1e-10
    write(PERIODIC_OUT/'torus_separable_subspace_check.json',dict(status='PASS',projection_relative_error=projection_error,
        orthogonality_relative_error=orthogonality_error,subspace_dimension=sub.dimension,scope='Identical preconditioner subspace to dense lifted vectors; no equation, spatial or harmonic truncation.'))
    print('SEPARABLE TORUS SUBSPACE PASS',projection_error,orthogonality_error,flush=True)


if __name__=='__main__':main()
