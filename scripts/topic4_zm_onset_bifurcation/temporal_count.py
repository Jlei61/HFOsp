"""Numerically resolved full-determinant Nyquist count for fixed equilibria.

Counts use the unchanged analytic shifted-bound response; they do not apply
to an empirical correction depending on imaginary frequency alone. The
finite frequency tail is reported explicitly, not treated as an interval
arithmetic proof of all arbitrarily high-frequency roots.
"""
from temporal_modes import *
from scipy.sparse.linalg import splu
import argparse,time

def parity(p):
    seen=np.zeros(len(p),bool);cycles=0
    for i in range(len(p)):
        if seen[i]:continue
        cycles+=1;j=i
        while not seen[j]:seen[j]=True;j=p[j]
    return (len(p)-cycles)%2

def main(a):
    s=ZMRate(a.grid);dest=DEST/f'g{a.grid}'/'temporal_counts'/a.label;dest.mkdir(parents=True,exist_ok=True)
    d=np.load(a.state);r=d['r'];D=float(d['D']);L=Linearization(s,r,D);cache={};started=time.time()
    def evaluate(f):
        f=float(f)
        if f in cache:return cache[f]
        M=L.matrix(2j*np.pi*f/1000).tocsc()
        if not np.isfinite(M.data).all():raise RuntimeError(('nonfinite characteristic',f))
        lu=splu(M);diag=lu.U.diagonal()
        phase=np.angle(np.exp(1j*(np.angle(diag).sum()+np.pi*(parity(lu.perm_r)+parity(lu.perm_c)))))
        K=sparse.eye(s.P)-M;bound=float(abs(K).sum(1).max())
        cache[f]=(float(phase),bound)
        if len(cache)%20==0:print('evaluated',len(cache),f,'Hz',round(time.time()-started,1),'s',flush=True)
        return cache[f]
    freq=np.unique(np.r_[0.,np.linspace(.1,20,41),np.linspace(20,100,41),np.linspace(100,500,41),[650,800,1000,1500,2000]])
    counts=[];jump=100.
    previous=dest/'result.json'
    if a.resume and previous.exists():
        old=read(previous)
        assert abs(old['D']-D)<1e-12 and old['source_state']==str(Path(a.state).resolve())
        freq=np.array(old['frequency_hz']);counts=old['counts_on_refinement'][:]
        cache.update({float(f):(float(np.angle(np.exp(1j*q))),None) for f,q in zip(freq,old['unwrapped_phase'])})
        for row in old['tail']:cache[float(row['frequency_hz'])]=(row['phase'],row['row_norm_bound'])
        saved=dest/f"result_before_resume_{len(counts)}.json"
        if not saved.exists():saved.write_text(previous.read_text())
    for cycle in range(a.refinements):
        for f in freq:evaluate(f)
        phases=np.unwrap([evaluate(f)[0] for f in freq]);jump=float(abs(np.diff(phases)).max())
        count=-int(round((phases[-1]-phases[0])/np.pi));counts.append(count)
        write(dest/'progress.json',dict(status='RUNNING',D=D,cycle=cycle,frequency_points=len(freq),count=count,
            maximum_phase_increment=jump,wall_s=time.time()-started))
        print('count',D,cycle,count,'max phase step',jump,flush=True)
        if len(counts)>=3 and counts[-1]==counts[-2] and jump<.4:break
        # At least one complete midpoint refinement prevents trusting a missed
        # winding from aligned coarse endpoints. Then refine large increments.
        mask=np.ones(len(freq)-1,bool) if cycle==0 and not a.resume else abs(np.diff(phases))>.2
        freq=np.sort(np.r_[freq,(freq[:-1]+freq[1:])[mask]/2])
    tail=[dict(frequency_hz=f,row_norm_bound=evaluate(f)[1],phase=evaluate(f)[0]) for f in [500.,1000.,1500.,2000.]]
    accepted=len(counts)>=3 and counts[-1]==counts[-2] and jump<.4 and tail[-1]['row_norm_bound']<.2
    result=dict(status='COMPLETE',source_state=str(Path(a.state).resolve()),D=D,global_E_hz=s.global_rate(r),
        frequency_hz=freq,unwrapped_phase=phases,counts_on_refinement=counts,unstable_root_count=count,
        maximum_phase_increment=jump,tail=tail,frequency_resolution_pass=accepted,
        stability=('UNSTABLE' if count>0 else 'NUMERICALLY_STABLE_IN_RESOLVED_SPECTRUM') if accepted else 'UNRESOLVED',
        scope='Fixed-Z dynamic-M equilibrium; finite-frequency numerical spectral test; no nonlinear onset claim',
        wall_s=time.time()-started)
    write(dest/'result.json',result);print({k:v for k,v in result.items() if k not in ('frequency_hz','unwrapped_phase')},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--state',required=True);p.add_argument('--label',required=True)
    p.add_argument('--grid',type=int,default=20);p.add_argument('--refinements',type=int,default=5)
    p.add_argument('--resume',action='store_true');main(p.parse_args())
