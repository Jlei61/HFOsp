"""Scientific summary and final QA for the bounded physical-D exploration."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,csv
from pathlib import Path
import numpy as np
from scipy.linalg import svdvals,eig
from topic4_fig5_D_physical_model import Equilibrium,Characteristic,OUT,ROOT
from topic4_fig5_z_characteristic_root_newton import root

def write(name,x):(OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def read(p):return json.loads(p.read_text())

def summarize():
    critical=[]
    for q in [1.,1.25]:
        for core in 'ab':
            h=read(OUT/f'q{q:g}_H{core}.json');rows=read(OUT/f'q{q:g}_cycles_{core}_N32/summary.json')
            # Extrapolate the amplitude-squared branch coefficient at zero.
            small=[r for r in rows if r['control_amplitude_hz']<=.2]
            x=np.array([r['control_amplitude_hz']**2 for r in small]);y=np.array([r['branch_D_shift_per_amplitude_squared'] for r in small]);intercept=float(np.polyfit(x,y,1)[1]);beta=h['transversality_per_s_per_D']
            cubic=-beta*intercept
            row=dict(q_ie=q,core=core,D=h['D'],mean_E_hz=h['mean_e_hz'],frequency_hz=h['frequency_hz'],
              type='SUPERCRITICAL_NEIMARK_SACKER' if cubic<0 else 'SUBCRITICAL_NEIMARK_SACKER',
              continuous_time_interpretation='Hopf-type oscillatory instability',amplitude_squared_branch_coefficient=intercept,
              radial_cubic_per_s_per_Hz_squared=cubic,small_amplitude_coefficients=y.tolist(),critical_transversality=beta,
              source=str(OUT/f'q{q:g}_H{core}.json'))
            critical.append(row)
    for name in ['old_F0','old_TP','old_TP2']:
        a=read(OUT/f'{name}.json');critical.append(dict(name=name,q_ie=1.,D=a['D'],mean_E_hz=a['mean_e_hz'],type=a['type'],stability=a['full_stability']))
    a=read(OUT/'old_Q3.json');critical.append(dict(name='old_Q3',q_ie=1.,D=a['D'],mean_E_hz=a['global_E_hz'],type=a['type'],stability='UNSTABLE'))
    write('critical_point_summary.json',critical)
    with (OUT/'critical_point_summary.csv').open('w') as f:
        keys=['q_ie','name','core','D','mean_E_hz','frequency_hz','type','stability'];w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(critical)
    c=Characteristic(1.25);states=[]
    for core in 'ab':
        z=np.load(OUT/f'q1.25_H{core}.npz');c.at(z['r_hz'],float(z['D']));j=c.eq.evaluate(z['r_hz'],float(z['D']),True)[1]
        states.append(dict(core=core,characteristic_zero_vs_steady_jacobian_max_error=float(abs(c.matrix(0)+j).max())))
    write('characteristic_steady_qa.json',states)
    linear=[]
    for q in [1.,1.25]:
        cq=Characteristic(q)
        for core in 'ab':
            a=np.load(OUT/f'q{q:g}_H{core}.npz');om=float(a['omega_per_s']);cq.at(a['r_hz'],float(a['D']));C=cq.matrix(1j*om);sv=svdvals(C)
            es,ll,rr=eig(C,left=True,right=True);k=np.argmin(abs(es));v=rr[:,k];w=ll[:,k];w/=np.vdot(w,v).conjugate();h=.001
            derivative=np.vdot(w,(cq.matrix(1j*om+h)-cq.matrix(1j*om-h))@v/(2*h));mu=np.exp(1j*om*cq.m.dt/1000)
            linear.append(dict(q_ie=q,core=core,two_smallest_singular_values=sv[-2:].tolist(),root_derivative_modulus=float(abs(derivative)),strong_resonance_distances=[float(abs(mu**k-1)) for k in range(1,5)]))
    write('hopf_linear_qa.json',linear)
    # The contour is undefined on the critical pair itself. Replace those
    # boundary counts by a positive-root witness from the other core.
    p=OUT/'q1.25_branch_stability.json';rows=read(p)
    aa=np.load(OUT/'q1.25_Ha.npz');bb=np.load(OUT/'q1.25_Hb.npz');c.at(bb['r_hz'],float(bb['D']));ll,v,er,h=root(c,1+36j,aa['mode'])
    assert ll.real>0 and er<1e-7
    for row in rows:
        if row['branch'] in ['low','middle'] and row['index']==0:
            row.update(stability='unstable',method='positive complex root of the already unstable core A',lambda_per_s=[ll.real,ll.imag],residual=er);row.pop('counts',None)
    for file in sorted(OUT.glob('stability_subset_*.json')):rows+=read(file)
    by={(r['branch'],r['index']):r for r in rows};rows=sorted(by.values(),key=lambda r:(r['branch'],r['index']));write('q1.25_branch_stability.json',rows)
    modes=[];m=c.m
    for name,key,label in [('q1_Hb','mode','H1 / core B'),('q1_Ha','mode','H2 / core A'),('old_TP','right_mode','SN1'),('old_TP2','right_mode','SN2')]:
        a=np.load(OUT/(name+'.npz'));power=abs(a[key][:400])**2*m.count_e;power/=power.sum();order=np.argsort(power)[::-1];idx=int(order[0]);n95=int(np.searchsorted(np.cumsum(power[order]),.95)+1)
        modes.append(dict(name=name,label=label,D=float(a['D']),cell_mode_energy_fraction=power.tolist(),maximum_cell_fraction=float(power[idx]),peak_cell_xy_mm=[idx%20+.5,idx//20+.5],cells_containing_95percent_energy=n95,effective_cells=float(1/np.sum(power**2)),total_cells=400))
    write('critical_spatial_modes.json',modes)
    hom=OUT/'fold_working_point_homotopy.json'
    if hom.exists():
        rows=read(hom);write('fold_working_point_homotopy_status.json',dict(status='NUMERICAL_STOP_UNCLASSIFIED',last_successful_q=max(r['q_ie'] for r in rows),target_q=1.25,target_reached=False,not_a_cusp_or_global_bifurcation_certificate=True,note='The old TP identity was not successfully tracked to q=1.25. New-workpoint folds are not claimed to be that same branch.'))
    print('SUMMARIZED',len(critical),'critical entries',len(by),'stability samples',flush=True)

if __name__=='__main__':summarize()
