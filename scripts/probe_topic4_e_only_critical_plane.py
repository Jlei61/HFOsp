#!/usr/bin/env python3
"""Actual nonlinear full-delay trajectories near the critical complex eigenspace."""
from analyze_topic4_e_only_bifurcation import Family,OUT,read,write
from topic4_e_only_z_tangent import EOnlyTangent
from scipy.sparse.linalg import eigs
import numpy as np
import time


def run(side):
    start=time.time();hopf=read(OUT/'reduced_bifurcation/hopf_native_path_tau20.6116_dt0.1.json')
    q=hopf['q']+side*.0002;f=Family('native_path');s=f.s;m=s.m;n=s.n;dt=s.dt;z=f.field(q)
    eq,err,ok=f.solve(q,np.full(200,.00005));assert ok
    tangent=EOnlyTangent(s,eq,z);shift=np.exp((.00003+2j*np.pi*hopf['frequency_hz']/1000)*dt)
    vals,vecs=eigs(tangent.inverse(shift),k=2,which='LM',tol=1e-11,ncv=20,v0=np.random.default_rng(91).normal(size=tangent.size).astype(complex))
    multipliers=shift-1/vals;lam=np.log(multipliers)/dt*1000
    j=np.argmin(abs(lam.imag-2*np.pi*hopf['frequency_hz']));v=vecs[:,j]
    B,_=np.linalg.qr(np.c_[v.real,v.imag]);Amap=B.T@np.column_stack([tangent.action(B[:,k]) for k in range(2)])
    A=(Amap-np.eye(2))/dt*1000
    history=np.broadcast_to(eq.reshape(2,1,n),(2,tangent.D,n)).copy()
    gating=tangent.B*tangent.drive(history)/(1-tangent.ar);current=gating.copy()
    xeq=np.r_[eq,gating.ravel(),current.ravel(),history.ravel()]
    nu=m.nu_ext_per_ms;ext=np.r_[np.full(n,m.tau_mem_e_ms*s.ga*m.j_ext_e_mv*nu),np.full(n,m.tau_mem_i_ms*s.ga*m.j_ext_i_mv*nu)]
    def step(x):
        r,g,c,h=tangent.unpack(x);ng=tangent.ar*g+tangent.B*tangent.drive(h);nc=tangent.ad*c+(1-tangent.ad)*ng
        _,ex,inh=s.moments(r,z);phi=s.phi(tangent.signed(nc)+ext,ex,inh)
        nr=(1-tangent.alpha)*r+tangent.alpha*phi
        nh=np.empty_like(h);nh[:,0]=r.reshape(2,1,n)[:,0];nh[:,1:]=h[:,:-1]
        return np.r_[nr,ng.ravel(),nc.ravel(),nh.ravel()]
    eqerr=float(np.max(abs(step(xeq)-xeq)));assert eqerr<1e-8,eqerr
    # Fixed amplitude small enough to stay near the equilibrium even on the growing side.
    scale=float(np.min(eq/np.maximum(abs(B[:200,0]),1e-20))*.0005)
    x=xeq+scale*B[:,0];records=[];perp=[];duration=6000.;steps=round(duration/dt)
    for k in range(steps):
        x=step(x)
        if (k+1)%10==0:
            delta=x-xeq;ab=B.T@delta
            records.append(np.r_[(k+1)*dt,ab/scale,np.average(x[:n],weights=m.count_e)*1000,np.average(x[n:2*n],weights=m.count_i)*1000])
            if (k+1)%1000==0:perp.append(float(np.linalg.norm(delta-B@ab)/max(np.linalg.norm(delta),1e-30)))
        if k%10000==0:write(OUT/f'critical_plane_{side:+d}_progress.json',{'status':'RUNNING','time_ms':k*dt})
    records=np.array(records);radii=np.linalg.norm(records[:,1:3],axis=1)
    row={'q':q,'side':side,'lambda_real_per_s':float(lam[j].real),'frequency_hz':float(lam[j].imag/2/np.pi),
         'full_tangent_residual':float(np.linalg.norm(tangent.action(v)-multipliers[j]*v)),
         'equilibrium_update_error':eqerr,'maximum_orthogonal_fraction':float(max(perp)),
         'linear_expected_envelope_ratio':float(np.exp(lam[j].real*duration/1000)),
         'observed_radius_ratio':float(radii[-1]/radii[0]),'seconds':time.time()-start,
         'scope':'Nonlinear 72600-state delayed rate trajectory initialized in the critical eigenspace. Displayed nullclines/vector field use its local projected linearization; not an exact global 2D closure.'}
    folder=OUT/'reduced_bifurcation';np.savez_compressed(folder/f'critical_plane_{side:+d}.npz',A_per_s=A,records=records,scale=scale,B=B,equilibrium=xeq)
    write(folder/f'critical_plane_{side:+d}.json',row);print(row,flush=True);return row


def main():
    rows=[];write(OUT/'critical_plane_status.json',{'status':'RUNNING','rows':rows})
    for side in [1,-1]:
        rows.append(run(side));write(OUT/'critical_plane_status.json',{'status':'RUNNING','rows':rows})
    write(OUT/'critical_plane_status.json',{'status':'COMPLETE','rows':rows})


if __name__=='__main__':
    try:main()
    except Exception as exc:
        write(OUT/'critical_plane_status.json',{'status':'FAILED','error':repr(exc)});raise
