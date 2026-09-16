"""Nonlinear eigenproblem of the population delay system, including modal vectors."""
from model import System,OUT
import json,numpy as np
from scipy.linalg import eig,svd
from scipy.optimize import root,brentq

def root_at(s,r,g,guess):
    pre=s.blocks(r,g)
    def fn(x):
        lam=(x[0]+1j*x[1])/1000
        M=s.characteristic(lam,r,g,pre)
        val=np.linalg.det(M)/np.prod(1+lam*s.tr)
        return [val.real,val.imag]
    sol=root(fn,[guess.real,guess.imag],tol=1e-10)
    lam=complex(*sol.x)
    if not np.isfinite(lam) or abs(lam.real)>2000 or abs(lam.imag)>20000:return None
    M=s.characteristic(lam/1000,r,g,pre);u,ss,vh=svd(M)
    if ss[-1]>1e-8:return None
    v=vh[-1].conj();w=u[:,-1]
    v*=np.exp(-1j*np.angle(v[np.argmax(abs(v))]))
    return dict(lam=lam,v=v,w=w,residual=float(np.linalg.norm(M@v)))

def spectrum(s,r,g,previous=()):
    guesses=list(previous)+[complex(a,2*np.pi*f) for a in [-10.,10.] for f in [0.,2.,5.,10.,20.,40.,80.,150.,300.]]
    found=[]
    for guess in guesses:
        rr=root_at(s,r,g,guess)
        if rr is None:continue
        if rr['lam'].imag<0:
            rr['lam']=rr['lam'].conjugate();rr['v']=rr['v'].conjugate();rr['w']=rr['w'].conjugate()
        if not any(abs(x['lam']-rr['lam'])<1e-4 for x in found):found.append(rr)
    return sorted(found,key=lambda x:x['lam'].real,reverse=True)

def main():
    s=System();branch=json.loads((OUT/'initial_branch.json').read_text());rows=[];modes={};previous=[]
    for row in branch[::2]:
        g=row['g'];r=np.array(row['r_hz'])/1000
        found=spectrum(s,r,g,previous);previous=[x['lam'] for x in found[:4]]
        if not found:continue
        ev=found[0]['lam'];key=f'g{g:.5f}'
        modes[key+'_v']=found[0]['v'];modes[key+'_w']=found[0]['w']
        rows.append(dict(g=g,r_hz=row['r_hz'],leading_real_per_s=ev.real,frequency_hz=abs(ev.imag)/2/np.pi,
            roots=[[x['lam'].real,x['lam'].imag,x['residual']] for x in found]))
        print(g,ev,flush=True)
    (OUT/'initial_spectrum.json').write_text(json.dumps(rows,indent=2)+'\n');np.savez_compressed(OUT/'initial_modes.npz',**modes)

if __name__=='__main__':main()
