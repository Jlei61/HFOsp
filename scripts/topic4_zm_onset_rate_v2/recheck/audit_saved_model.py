"""Independent CPU reconstruction of saved periodic/equilibrium evidence."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model_zm import *
import hashlib

OUT=DEST/'recheck_20260917'
P=DEST/'periodic_completion'


def periodic_rhs(s,path):
    z=np.load(path);r=z['r'];T=float(z['T']);D=float(z['D']);s.set_D(D)
    N=len(r);rf=np.fft.rfft(r,axis=0);K=len(rf)
    state=np.zeros((10,K,s.P),complex);arr=np.zeros((4,K,s.P),complex)
    for k,v in enumerate(rf):
        lam=2j*np.pi*k/T;a,b,qa,qb=s.matrices(1.,lam)
        delayed=np.array([o@v for o in [a,b,qa,qb]]);arr[:,k]=delayed
        drive=v/s.filter_response(lam)
        state[0,k]=drive/(1+lam*s.tf);state[1,k]=drive/(1+lam*s.ts)
        state[2,k]=s.tm*s.area[0]*delayed[0]/(1+lam*s.rise[0]);state[3,k]=state[2,k]/(1+lam*s.decay[0])
        state[4,k]=s.tm*s.area[1]*delayed[1]/(1+lam*s.rise[1]);state[5,k]=state[4,k]/(1+lam*s.decay[1])
        state[6,k]=s.tm*s.area[0]**2*delayed[2]/(1+lam*s.tau[0]/2)
        state[7,k]=s.tm*s.area[1]**2*delayed[3]/(1+lam*s.tau[1]/2)
        state[8,k]=.5*s.E*v/(1+lam*1000)
    state[9,0]=s.Z*N
    y=np.fft.irfft(state,n=N,axis=1);dy=np.fft.irfft(state*(2j*np.pi*np.arange(K)/T)[None,:,None],n=N,axis=1)
    aa=np.fft.irfft(arr,n=N,axis=1);rhs=np.stack([s.rhs(y[:,j],aa[:,j],False) for j in range(N)],axis=1)
    err=rhs-dy;global_rate=r[:,s.E]@s.mean_weights*1000
    return dict(path=str(path),N=N,D=D,T_ms=T,
        relative_full_RHS_defect=float(np.linalg.norm(err)/np.linalg.norm(dy)),
        rate_state_RHS_max_error_hz_per_ms=float(abs(err[:2]).max()*1000),
        max_error_by_state=abs(err).max(axis=(1,2)),
        min_group_rate_hz=float(r.min()*1000),global_mean_hz=float(global_rate.mean()),
        D_error=float(abs(1-s.Z[s.E]@s.mean_weights-D)),
        method='CPU harmonic reconstruction of all ten states and delayed arrivals, then direct nonlinear RHS at every phase')


def main():
    OUT.mkdir(exist_ok=True);s=ZMSpatialRate();identity=read(DEST/'model_identity.json')
    files=identity.get('files',identity.get('snapshots',[]))
    # Use the stored paths rather than any concurrently evolving shared source.
    if not files:
        files=next(v for v in identity.values() if isinstance(v,list) and v and isinstance(v[0],dict) and 'sha256' in v[0])
    hashes=[dict(path=q['snapshot'],matches=hashlib.sha256(Path(q['snapshot']).read_bytes()).hexdigest()==q['sha256']) for q in files]
    native=read(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/reference_identity.json')['reference_identity']
    rows=[]
    for name in ['LPC_burst_N512','highHopf_A16_N64']:
        row=periodic_rhs(s,P/'orbits'/f'{name}.npz');rows.append(row);print('PERIODIC',clean(row),flush=True)
    h=np.load(P/'hopf_high.npz');D=float(h['D']);r=h['r'];v=h['q'];lam=1j*float(h['w'])
    dy=s.eigenstate(r,D,lam,v);y=s.state(r);arr=np.array([a@r for a in s.matrices(1.)]);da=np.array([a@v for a in s.matrices(1.,lam)])
    eps=1e-5/abs(dy).max();parts=[]
    for part in [np.real,np.imag]:
        parts.append((s.rhs(y+eps*part(dy),arr+eps*part(da))-s.rhs(y-eps*part(dy),arr-eps*part(da)))/(2*eps))
    hopf=dict(D=D,characteristic_residual=float(np.linalg.norm(s.characteristic(r,D,lam)@v)/np.linalg.norm(v)),
        full_RHS_eigenmode_relative_error=float(np.linalg.norm(parts[0]+1j*parts[1]-lam*dy)/np.linalg.norm(lam*dy)))
    ds=[]
    for D in [0.,.188524,.188644,.2,.362741,1.]:
        s.set_D(D);ds.append(dict(D=D,weighted_mean_error=float(abs(1-s.Z[s.E]@s.mean_weights-D)),
            E_Z_bounds=[float(s.Z[s.E].min()),float(s.Z[s.E].max())],I_Z_exactly_one=bool(np.all(s.Z[~s.E]==1))))
    out=dict(snapshot_hashes=hashes,graph_identity_matches_reference=(native==s.prep['graph_identity']),
        state_model='Same 935-group spatial rate DDE; conditional Z fixed, M dynamic',
        periodic_RHS_checks=rows,Hopf_RHS_check=hopf,D_path_checks=ds,
        status='PASS_INTERNAL_EQUATIONS_ONLY',native_SNN_equivalence='NOT_VALIDATED')
    assert all(q['matches'] for q in hashes) and native==s.prep['graph_identity']
    assert max(q['relative_full_RHS_defect'] for q in rows)<1e-6
    assert hopf['full_RHS_eigenmode_relative_error']<1e-4
    write(OUT/'independent_equation_audit.json',out)


if __name__=='__main__':main()
