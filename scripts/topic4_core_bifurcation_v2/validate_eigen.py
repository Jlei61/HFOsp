"""Fine fold approach, exact characteristic residuals, and spectral refinement."""
from model import System,OUT
from branches import fold
from spectral import leading,generator
from eigen import root_at
import numpy as np,json
from scipy.linalg import eig,svd

def winding(s,r,g,m=512):
    # Counter-clockwise box in the right half-plane; all filter poles are left.
    corners=[1e-8-2j,1.-2j,1.+2j,1e-8+2j,1e-8-2j]
    zz=np.concatenate([np.linspace(a,b,m,endpoint=False) for a,b in zip(corners[:-1],corners[1:])]);pre=s.blocks(r,g)
    vals=np.array([np.linalg.det(s.characteristic(z,r,g,pre)) for z in zz]);vals=np.r_[vals,vals[:1]]
    d=np.angle(vals[1:]/vals[:-1]);return dict(count=int(round(d.sum()/2/np.pi)),phase_increment_max=float(abs(d).max()),points=len(zz),min_abs_det=float(abs(vals).min()))

def main():
    s=System();f=json.loads((OUT/'fold.json').read_text());rf=np.array(f['r_hz'])/1000;gc=f['g'];v=np.array(f['v']);fine=[]
    for delta in np.geomspace(1e-8,.12,55):
        for direction in [-1,1]:
            guess=rf+direction*np.sqrt(2*f['transversality']*delta/f['quadratic'])*v
            r,err,ok=s.solve(gc-delta,guess);assert ok
            lamguess=direction*np.sqrt(2*f['transversality']*f['quadratic']*delta)/f['dynamic_normalization_ms']*1000
            ev=root_at(s,r,gc-delta,complex(lamguess,0));assert ev is not None
            fine.append(dict(g=gc-delta,delta=float(delta),direction=direction,r_hz=(r*1000).tolist(),lambda_per_s=[ev['lam'].real,ev['lam'].imag],eigen_residual=ev['residual'],equilibrium_residual=err))
    (OUT/'fine_fold_branch.json').write_text(json.dumps(fine,indent=2)+'\n')
    selected=[];allmodes={}
    for g,direction in [(1.,-1),(1.12,-1),(gc-1e-6,-1),(gc-1e-6,1)]:
        guess=rf+direction*np.sqrt(2*f['transversality']*(gc-g)/f['quadratic'])*v
        r,err,ok=s.solve(g,guess);assert ok
        specs={str(N):leading(s,r,g,N) for N in [24,40,64]}
        crit=root_at(s,r,g,complex(specs['64'][0]))
        # At small g the leading full-system root is an uncoupled filter pole.
        active=root_at(s,r,g,complex(direction*30,0))
        uu,ww,vv=s.blocks(r,g)
        bound=float(np.max((abs(uu[:,None]*ww)+abs(vv)).sum(1)/s.tr)*1000)
        row=dict(g=g,direction=direction,r_hz=(r*1000).tolist(),spectra={n:[[x.real,x.imag] for x in ev[:24]] for n,ev in specs.items()},winding=[winding(s,r,g,m) for m in [512,1024]],unstable_root_modulus_bound_per_s=bound)
        selected.append(row)
        if active:
            allmodes[f'g{g:.8f}_d{direction}_v']=active['v'];allmodes[f'g{g:.8f}_d{direction}_w']=active['w']
        print('CHECK',g,direction,specs['64'][:3],row['winding'],flush=True)
    atfold={str(N):[[x.real,x.imag] for x in leading(s,rf,gc,N)[:24]] for N in [24,40,64]}
    M=s.characteristic(0,rf,gc);w=np.array(f['w'])
    report=dict(selected=selected,fold_spectra=atfold,right_residual=float(np.linalg.norm(M@v)),left_residual=float(np.linalg.norm(w@M)),
        fine_max_eigen_residual=max(x['eigen_residual'] for x in fine),fine_max_equilibrium_residual=max(x['equilibrium_residual'] for x in fine))
    # The other turning point is already unstable; compute it independently.
    arc=json.loads((OUT/'equilibrium_arc.json').read_text());part=[x for x in arc if x['direction']==1 and x['step']<70];lo=min(part,key=lambda x:x['g']);r=np.array(lo['r_hz'])/1000
    ev,vv=eig(s.jac(r,lo['g']));vv=vv[:,np.argmin(abs(ev))].real
    upper=fold(s,np.r_[r,lo['g'],vv]);upper['dynamic_spectrum_per_s']=[[z.real,z.imag] for z in leading(s,np.array(upper['r_hz'])/1000,upper['g'])[:12]]
    (OUT/'upper_fold.json').write_text(json.dumps(upper,indent=2)+'\n')
    report['upper_fold']=upper
    (OUT/'eigen_validation.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(OUT/'critical_modes.npz',v=v,w=w,**allmodes)
    G=generator(s,rf,gc,64)*1000;vals,L,R=eig(G,left=True,right=True);j=np.argmin(abs(vals))
    vr=R[:,j].real;vr/=vr[0];wl=L[:,j].real;wl/=wl@vr
    rr=float(np.linalg.norm(G@vr-vals[j]*vr)/np.linalg.norm(vr));lr=float(np.linalg.norm(wl@G-vals[j]*wl)/np.linalg.norm(wl))
    assert rr<1e-6 and lr<1e-6
    np.savez_compressed(OUT/'full_generator_critical_mode_N64.npz',right=vr,left=wl,lambda_per_s=vals[j],theta_ms=(np.cos(np.arange(65)*np.pi/64)-1)*s.delay[-1]/2)
    full=dict(dimension=len(vr),right_relative_residual_per_s=rr,left_relative_residual_per_s=lr,normalization=float(wl@vr),lambda_per_s=[vals[j].real,vals[j].imag],state_order='r[6], first synaptic filter h[6], second-filter c histories[65,6]; history starts at current time',note='Figure shows rate-space characteristic vectors; this file also contains full collocated generator eigenvectors.')
    (OUT/'full_generator_eigenvector_validation.json').write_text(json.dumps(full,indent=2)+'\n')
    print('EIGEN VALIDATED',report['right_residual'],report['left_residual'],flush=True)

if __name__=='__main__':main()
