"""Solve an already-audited Galerkin seed with balanced phase/period coordinates."""
from exact_periodic import *
from native_cycles import save
import argparse


def main(a):
    s=model();(attach_rate_entry_path if a.family=='rate' else attach_native_path)(s)
    z=np.load(a.orbit);D=float(z['D']) if a.D is None else a.D;T=float(z['T'])
    checks=read(OUT/'exact_parameter_columns_rate_near_G2049_M8192.json')
    assert all(min(q['relative_error'] for q in checks if q['column']==k)<1e-6 for k in ['logT','D'])
    o=ExactGalerkin(s,len(z['r']),a.M,a.device);o.cache_mean_operators=False;cp=o.cp
    r=cp.asarray(z['r']);dr=cp.fft.irfft(2j*np.pi*cp.arange(o.K)[:,None]*cp.fft.rfft(r,axis=0),n=o.N,axis=0)
    old_phase_norm=float(RS/cp.linalg.norm(dr));log('PHASE ROW NORM',old_phase_norm,'balanced',1.)
    sol=o.solve(z['r'],T,D,maxiter=18,tol=2e-8,restart=a.restart)
    q=save(s,sol,OUT/'periodic',a.label)
    q.update(phase_row_norm_before=old_phase_norm,phase_row_norm_after=1.,
             period_right_scaling=.001,parameter_columns='analytic, independently checked',
             nonlinear_samples=a.M,method='dealiased Fourier-Galerkin; balanced coordinates')
    write(OUT/'periodic'/f'{a.label}.json',q)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('orbit');p.add_argument('--D',type=float);p.add_argument('--label',required=True)
    p.add_argument('--family',choices=['rate','native'],default='rate');p.add_argument('--M',type=int,default=8192)
    p.add_argument('--device',type=int,default=1);p.add_argument('--restart',type=int,default=40);main(p.parse_args())
