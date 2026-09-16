#!/usr/bin/env python3
"""Resolve winding near a narrow crossing and check transfer quadrature."""
from analyze_topic4_prescribed_z_phase import *
from scipy.linalg import eigvals
from scipy.optimize import root


def count_local(s,r,z,frequency):
    pre=s.blocks(r,z);center=2*np.pi*frequency/1000*s.dt;checks=[]
    for length,local in ((1600,801),(3200,1601)):
        theta=np.unique(np.r_[0,np.geomspace(1e-7,np.pi,length),np.linspace(max(1e-9,center-1e-4),center+1e-4,local)])
        angles=[]
        for th in theta:
            mu=np.exp(1j*th);sign,_=np.linalg.slogdet(s.characteristic(mu,r,z,pre=pre));ref=(mu-1)*s.tr/s.dt+1
            angles.append(np.angle(sign*np.exp(-1j*np.angle(ref).sum())))
        phase=np.unwrap(angles);count=-(phase[-1]-phase[0])/np.pi
        checks.append({'global_grid':length,'local_grid':local,'count':float(count),'maximum_phase_step':float(np.max(abs(np.diff(phase))))})
    ok=abs(checks[-1]['count']-checks[0]['count'])<1e-6 and abs(checks[-1]['count']-round(checks[-1]['count']))<1e-6 and checks[-1]['maximum_phase_step']<np.pi/2
    return {'status':'PASS' if ok else 'UNRESOLVED','unstable_roots':round(checks[-1]['count']) if ok else None,'checks':checks,
        'method':'Argument principle, global log-angle grid plus explicit narrow critical-frequency mesh; both grids doubled. Numerical, not interval certification.'}


def main():
    result=read(OUT/'first_instability.json');a=np.load(OUT/'first_instability.npz');_,_,fields=source();tm=result['time_parameter_ms'];s=MixedTimescaleSystem(quadrature=33)
    counts=[]
    for off in (-1.,1.):
        z=field_at(fields,tm+off);r,err,ok=s.solve(z,a['r']);assert ok
        counts.append({'time_ms':tm+off,**count_local(s,r,z,result['frequency_hz'])})
    quad=MixedTimescaleSystem(quadrature=65);r,err,ok=quad.solve(a['z'],a['r']);assert ok;pre=quad.blocks(r,a['z'])
    def equation(x):
        ev=eigvals(quad.characteristic(np.exp(complex(*x)/1000*quad.dt),r,a['z'],pre=pre));v=ev[np.argmin(abs(ev))];return [v.real,v.imag]
    ans=root(equation,a['lambda_per_s'],tol=1e-9);err=float(max(abs(np.array(equation(ans.x)))));assert err<1e-7
    validation={'status':'PASS' if all(c['status']=='PASS' for c in counts) and abs(ans.x[0])<.01 else 'UNRESOLVED',
        'counts':counts,'quadrature65_at_same_field':{'equilibrium_residual':float(abs(quad.F(r,a['z'])).max()),'root_residual':err,
            'real_per_s':float(ans.x[0]),'frequency_hz':float(abs(ans.x[1])/2/np.pi),'maximum_equilibrium_rate_change_hz':float(abs(r-a['r']).max()*1000)}}
    write(OUT/'crossing_validation.json',validation)
    result['initial_coarse_counts']=result['counts'];result['counts']=counts;result['numerical_validation']=validation['status']
    result['classification']='Single conjugate pair crosses the unit circle (Hopf-type linear instability of the frozen delayed rate map).' if [x['unstable_roots'] for x in counts]==[0,2] else 'Oscillatory crossing; complete stability count unresolved.'
    result['nonlinear_limit']='First Lyapunov coefficient, periodic branch stability and attribution to the later native runaway remain NOT_ESTABLISHED.'
    write(OUT/'first_instability.json',result)


if __name__=='__main__':main()
