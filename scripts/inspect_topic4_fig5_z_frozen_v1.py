#!/usr/bin/env python3
"""Check crossing modes and tabulated-transfer derivative sensitivity."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import json
import numpy as np
from scipy.linalg import eig
from topic4_fig5_z_frozen_v1 import Characteristic,OUT,transfer_gains
from topic4_fig5_z_branch_dynamics import siegert_table


def main():
    c=Characteristic();m=c.m;n=m.n;xs,gs=siegert_table.table();rows=[]
    for name in ['core_b_crossing','oscillatory_crossing']:
        a=np.load(OUT/f'{name}_state.npz');s=float(a['s']);r=a['r_hz'];om=float(a['omega_per_s']);c.at(r,s)
        vals,vec=eig(c.matrix(1j*om),check_finite=False);k=int(np.argmin(abs(vals)));v=vec[:,k];energy=abs(v[:n])**2
        fractions=[float(energy@m.region_w[f'175_{i}']/(energy@m.count_e)) for i in range(3)]
        row=dict(name=name,s=s,core_a_b_surround_mode_energy=fractions,fd_characteristic_residual=float(abs(vals[k])))
        np.savez_compressed(OUT/f'{name}_mode.npz',mode=v,r_hz=r,s=s,omega_per_s=om)
        q=c.eq.last
        c.u,c.v,c.w=transfer_gains(q['mu'],np.repeat(q['ex'],m.K),q['inh'],m.theta_u,m.te,m.tref_e,m.v_reset,m.ra+m.ta,m.w2cv_e,m.gh_x,m.gh_w,xs,gs)
        re,ri=r[:n]/1000,r[n:]/1000
        mui=m.ti*(m.gaA*(m.w_ie@re+m.ji*m.nu_sig)-m.gaG*(m.w_ii@ri))
        ei=m.ti*(m.v_ie@re+m.ji*m.ji*m.nu_sig);ii=m.ti*(m.v_ii@ri)
        c.ui,c.vi,c.wi=transfer_gains(mui,ei,ii,np.full(n,m.theta_i),m.ti,m.tref_i,m.v_reset,m.ra+m.ta,m.w2cv_i,m.gh_x,m.gh_w,xs,gs)
        lam,error=c.refine(1j*om)
        row.update(analytic_table_lambda_per_s=[lam.real,lam.imag],analytic_characteristic_residual=error,
                   interpretation='Finite-difference crossing coordinate; analytic table derivative sensitivity at the same state.')
        rows.append(row);print(row,flush=True)
        (OUT/'crossing_mode_and_derivative_qa.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
