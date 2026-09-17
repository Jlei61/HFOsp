"""Extended fold solve and pseudo-arclength equilibrium continuation."""
from model import System,OUT
import json,numpy as np
from scipy.optimize import root
from scipy.linalg import eig,svd

def fold(s,initial=None):
    if initial is None:
        r,_,ok=s.solve(1.12,np.array([.00035,.00028,0,0,0,0]));assert ok
        ev,vv=eig(s.jac(r,1.12));v=vv[:,np.argmin(abs(ev))].real
        initial=np.r_[r,1.13,v]
    def fn(y):
        r,g,v=y[:6],y[6],y[7:]
        return np.r_[s.F(r,g)*1000,s.jac(r,g)@v,v@v-1]
    sol=root(fn,initial,tol=1e-10);err=abs(fn(sol.x)).max();assert err<1e-7,(err,sol.message)
    r,g=sol.x[:6],sol.x[6];ev,L,R=eig(s.jac(r,g),left=True,right=True);j=np.argmin(abs(ev))
    v=R[:,j].real;v/=np.linalg.norm(v);v*=np.sign(v[np.argmax(abs(v))]);w=L[:,j].real;w/=w@v
    h=1e-7;fg=(s.F(r,g+1e-5)-s.F(r,g-1e-5))/2e-5
    frr=(s.F(r+h*v,g)-2*s.F(r,g)+s.F(r-h*v,g))/h**2
    Mp=(s.characteristic(1e-6,r,g)-s.characteristic(-1e-6,r,g))/2e-6
    return dict(g=float(g),r_hz=(r*1000).tolist(),v=v.tolist(),w=w.tolist(),
        residual=float(err),static_eigenvalues=[[x.real,x.imag] for x in ev],
        transversality=float(w@fg),quadratic=float(w@frr),dynamic_normalization_ms=float(w@Mp@v))

def arc(s,f,steps=350):
    scale=.0002;r=np.array(f['r_hz'])/1000;g=f['g'];v=np.array(f['v']);rows=[]
    for direction in [-1,1]:
        y=np.r_[np.arcsinh(r/scale),g];t=np.r_[v/np.sqrt(r*r+scale*scale),0.]*direction;t/=np.linalg.norm(t)
        ds=.035
        for k in range(steps):
            pred=y+ds*t
            def fn(x):return np.r_[s.F(scale*np.sinh(x[:6]),x[6])/scale,(x-pred)@t]
            def jac(x):
                rr=scale*np.sinh(x[:6]);gg=x[6];fg=(s.F(rr,gg+1e-5)-s.F(rr,gg-1e-5))/2e-5/scale
                return np.vstack([np.c_[s.jac(rr,gg)*np.cosh(x[:6])[None,:],fg],t])
            sol=root(fn,pred,jac=jac,tol=1e-10)
            if abs(fn(sol.x)).max()>1e-6:
                ds*=.5
                if ds<1e-5:break
                continue
            y=sol.x;rr=scale*np.sinh(y[:6]);gg=y[6]
            if gg<.05 or gg>1.8 or rr.min()<-1e-9:break
            _,_,vh=svd(jac(y)[:-1]);tt=vh[-1];tt*=np.sign(tt@t);t=tt
            rows.append(dict(direction=direction,step=k,g=float(gg),r_hz=(rr*1000).tolist(),residual=float(abs(s.F(rr,gg)).max())))
            ds=min(.07,ds*1.1)
    return rows

def main():
    s=System();f=fold(s);print('FOLD',json.dumps(f),flush=True)
    (OUT/'fold.json').write_text(json.dumps(f,indent=2)+'\n')
    rows=arc(s,f);(OUT/'equilibrium_arc.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('ARC',len(rows),[(d,min(x['g'] for x in rows if x['direction']==d),max(x['r_hz'][0] for x in rows if x['direction']==d)) for d in [-1,1]],flush=True)
    # Independently seek elevated equilibria; never count failed roots as branches.
    found=[];rng=np.random.default_rng(7711)
    starts=[np.array([a,b,.001,.02,.02,.001]) for a in [.005,.02,.08,.2] for b in [.0002,.02,.1]]
    for initial in starts:
        r,err,ok=s.solve(1.,initial)
        if ok and not any(np.linalg.norm(r-x['r'])<1e-6 for x in found):
            found.append(dict(r=r.tolist(),err=err));print('ROOT',r*1000,err,flush=True)
    (OUT/'elevated_equilibria.json').write_text(json.dumps(found,indent=2)+'\n')

if __name__=='__main__':main()
