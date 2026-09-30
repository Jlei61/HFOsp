"""Full-state Floquet candidates from a closed multiple-shooting root.

The full segment variational products are composed; numerical segmentation
does not reduce the physical network or freeze M. Neutral/flow failures stop
the spectral calculation and cannot be interpreted as bifurcations.
"""
from common import np,read,write,log
from onset_state_continuation import build
from onset_poincare_corrector import SectionReturn
from onset_segment_flow import SegmentDerivative,fixed_times
from fine_rate_frozen_Z_fields import capture,restore
from pathlib import Path
import argparse,os,time,math


def root_parts(e,parent,cache_segments=None):
    result=read(parent/'result.json')
    assert result['status'] in ['NUMERICAL_MULTIPLE_SHOOTING_ROOT','NUMERICAL_SEGMENTED_SINGLE_SHOOTING_ROOT']
    row=result['iterations'][-1];T=row['period_ms']
    folder=parent/f"iteration{row['iteration']:02d}"
    paths=sorted(folder.glob('node[0-9][0-9].npz'));K=len(paths)
    assert K>=2 and [p.name for p in paths]==[f'node{j:02d}.npz' for j in range(K)]
    h=np.array(row.get('durations_ms') or [T/K]*K)
    assert len(h)==K and abs(h.sum()-T)<1e-8
    parts=[SectionReturn(dict(np.load(p)),e,float(t)) for p,t in zip(paths,h)]
    power=sum(((A.c.raw(A.base)**2)*(A.c.weight**2)).sum(1) for A in parts)/K
    scale=np.maximum(np.sqrt(power)[:,None],parts[0].c.floor)
    for A in parts:
        A.c.scale=scale.copy();A.xref=A.c.pack(A.base)
        assert np.array_equal(A.base['syn'][5],parts[0].base['syn'][5])
        assert np.all(A.base['parameters'][19]==0) and np.all(A.base['parameters'][20]==1)
    if cache_segments is not None and cache_segments!=K:
        assert cache_segments>=2
        A=parts[0];step=round(T/cache_segments/e.dt)*e.dt
        durations=[step]*(cache_segments-1)+[T-step*(cache_segments-1)]
        assert min(durations)>2*e.dt
        prefix=fixed_times(A,A.xref,np.cumsum(durations)[:-1])
        states=[A.base]+[A.state(y) for y,_ in prefix]
        parts=[]
        for state,duration in zip(states,durations):
            B=SectionReturn(state,e,duration);B.c.scale=scale.copy();B.xref=B.c.pack(state)
            assert B.admissible(B.xref)
            parts.append(B)
    return parts,T,folder


def main(a):
    parent=Path(a.parent).resolve();out=parent/'multiple_section_spectrum'
    out.mkdir(exist_ok=True);assert not (out/'jobs.json').exists()
    contract=read(parent/'contract.json');dt=contract['dt_ms']
    write(out/'contract.json',dict(source=str(parent),dt_ms=dt,maximum_Krylov=a.krylov,
        cache_segments=a.cache_segments,
        cache_partition='Optional finer storage partition uses actual uninterrupted integer-step states and keeps the original root cycle-RMS coordinates. Only peak GPU cache memory changes; total full-state monodromy and all M/delay variables are retained.',
        equation='Same complete spatial delayed rate model with every Z held and every E M dynamic. Compose full segment variational maps in one common invertible cycle-RMS coordinate system.',
        prerequisites='Numerically closed multiple-shooting root; independent unsegmented original-flow whole-period closure combined<1e-7 and eachblock<1e-6. Cached/original derivatives checked at every root node. Full-period neutral direction residual<1e-3 required before Arnoldi.',
        method='Second/fourth order phase velocities from original forward steps; full fixed-time monodromy is product of all segment derivatives. Return derivative projects endpoint along the autonomous phase velocity onto the original local transverse plane. Two-pass full-state Arnoldi and independent full-state Ritz residuals.',
        limits='A finite Krylov spectrum gives verified numerical eigenpairs, not a completeness proof. Independent mesh, shifted orbit, critical crossing and nondegeneracy remain required for a physical bifurcation label.',model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid());write(out/'jobs.json',jobs);started=time.time()
    try:
        e=build(a.device,dt);parts,T,folder=root_parts(e,parent,a.cache_segments)
        audit=read(folder/'independent_whole_flow/result.json')
        closure=next(z for z in audit['exact_period_returns'] if z['divisor']==1)
        passed=closure['combined_relative_rms']<1e-7 and max(z['relative_rms'] for z in closure['blocks'].values())<1e-6
        assert passed,('Unsegmented periodic closure failed',closure)
        A=parts[0];restore(e,A.base);diff=[]
        for _ in range(4):
            e.step();e.cp.cuda.get_current_stream().synchronize();diff.append(A.c.pack(capture(e))-A.xref)
        velocities={n:sum(((-1)**(j+1)*math.comb(n,j)/j)*diff[j-1] for j in range(1,n+1))/dt for n in [2,4]}
        derivatives=[None]*len(parts);shared=None;checks=[]
        for j in np.argsort([-A.period for A in parts]):
            A=parts[j]
            J=SegmentDerivative(A,A.xref,A.period,shared=shared);shared=J.shared
            rng=np.random.default_rng(925070+j);v=A.xref*rng.normal(size=A.xref.size);v/=np.linalg.norm(v)
            actual=J(v);old=SegmentDerivative(A,A.xref,A.period,cached=False);expected=old(v)
            error=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
            checks.append(dict(segment=j,relative_error=error,status='PASS' if error<1e-10 else 'FAIL'))
            write(out/'cache_checks.json',checks);assert error<1e-10,checks[-1]
            derivatives[j]=J;del old;log('MULTIPLE SPECTRUM CACHE',j,error)
        def monodromy(v):
            for J in derivatives:v=J(v)
            return v
        if a.cache_segments is not None:
            A=parts[0];B=SectionReturn(A.base,e,T);B.c.scale=A.c.scale.copy();B.xref=B.c.pack(B.base)
            rng=np.random.default_rng(925425);v=B.xref*rng.normal(size=B.xref.size);v/=np.linalg.norm(v)
            actual=monodromy(v);whole=SegmentDerivative(B,B.xref,T,cached=False);expected=whole(v)
            error=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
            write(out/'partition_whole_derivative_check.json',dict(status='PASS' if error<1e-10 else 'FAIL',
                relative_error=error,comparison='Composition of the new storage partition versus one uninterrupted original full-period variational flow'))
            assert error<1e-10,('Storage partition changes full derivative',error)
            del whole
        neutral=[]
        for n,v in velocities.items():
            product=monodromy(v);err=float(np.linalg.norm(product-v)/np.linalg.norm(v))
            neutral.append(dict(order=n,relative_error=err,pass_gate=bool(err<1e-3)))
            write(out/'neutral_checks.json',neutral);log('MULTIPLE SPECTRUM NEUTRAL',neutral[-1])
        if not neutral[-1]['pass_gate']:
            write(out/'result.json',dict(status='NEUTRAL_MODE_GATE_FAILED_NO_PHYSICAL_FLOQUET',neutral=neutral,
                closure=closure,model_promoted=False,bifurcation_type='NOT_ESTABLISHED'))
            jobs.update(status='COMPLETE',scientific_status='NEUTRAL_MODE_GATE_FAILED');write(out/'jobs.json',jobs);return
        flow=velocities[4];normal=flow/np.linalg.norm(flow);den=float(normal@flow)
        def project(v):return v-flow*float(normal@v)/den
        def section(v):return project(monodromy(v))
        rng=np.random.default_rng(925080);v=rng.normal(size=normal.size)
        v.reshape(-1,e.s.P)[4,~e.s.E]=0;v=project(v);v/=np.linalg.norm(v)
        V=[v];H=np.zeros((a.krylov+1,a.krylov));rows=[]
        for k in range(a.krylov):
            q=section(V[k])
            for _ in range(2):
                for j in range(k+1):
                    h=float(V[j]@q);H[j,k]+=h;q-=h*V[j]
            H[k+1,k]=np.linalg.norm(q)
            values,vectors=np.linalg.eig(H[:k+1,:k+1]);res=abs(H[k+1,k]*vectors[-1])
            order=np.argsort(-abs(values))[:8]
            row=dict(dimension=k+1,candidates=[dict(real=float(values[j].real),imag=float(values[j].imag),
                modulus=float(abs(values[j])),estimated_residual=float(res[j])) for j in order])
            rows.append(row);write(out/'progress.json',rows);log('MULTIPLE SPECTRUM ARNOLDI',row)
            if H[k+1,k]<1e-13:break
            if k+1<a.krylov:V.append(q/H[k+1,k])
        np.savez_compressed(out/'hessenberg.npz',H=H[:k+2,:k+1],values=values,estimated_residuals=res)
        ordering=list(np.argsort(-abs(values)))+list(np.argsort(abs(values-1)))+list(np.argsort(abs(values+1)))
        selected=[];verified=[]
        for j in ordering:
            if j in selected or values[j].imag< -1e-8 or res[j]>1e-5*max(1,abs(values[j])):continue
            selected.append(int(j))
            if len(selected)==4:break
        for index,j in enumerate(selected):
            z=sum(c*v for c,v in zip(vectors[:,j],V));z/=np.linalg.norm(z)
            product=section(z.real)+1j*(section(z.imag) if np.linalg.norm(z.imag)>1e-14 else np.zeros_like(z.real))
            error=float(np.linalg.norm(product-values[j]*z)/max(1,abs(values[j])))
            entry=dict(real=float(values[j].real),imag=float(values[j].imag),relative_residual=error,
                status='VERIFIED_NUMERICAL_EIGENPAIR' if error<1e-6 else 'FAILED_FULL_RESIDUAL')
            verified.append(entry);write(out/'verified.json',verified)
            np.savez_compressed(out/f'mode{index:02d}.npz',vector=z,product=product,multiplier=values[j],
                coordinate_scale=parts[0].c.scale,coordinate_weight=parts[0].c.weight)
        write(out/'result.json',dict(status='BOUNDED_NUMERICAL_SPECTRUM_COMPLETE',closure=closure,neutral=neutral,
            verified=verified,seconds=time.time()-started,completeness='NOT_ESTABLISHED',
            physical_Floquet='PENDING_SHIFT_AND_MESH',bifurcation_type='NOT_ESTABLISHED',model_promoted=False))
        jobs.update(status='COMPLETE');write(out/'jobs.json',jobs)
    except BaseException as exc:
        jobs.update(status='FAILED',error=repr(exc));write(out/'jobs.json',jobs);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('parent');p.add_argument('--device',type=int,default=1)
    p.add_argument('--krylov',type=int,default=24);p.add_argument('--cache-segments',type=int)
    main(p.parse_args())
