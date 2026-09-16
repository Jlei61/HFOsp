"""Validate claims about branch identity, critical spectra and numerical resolution."""
from common import *
from periodic import Orbit
from connections import compare,main as connection_checks
import numpy as np
from scipy.signal import resample

def main():
    connection_checks();connections=read(OUT/'connection_checks.json')
    a=OUT/'periodic/verified_mixed_high_continued_N1024/g1.34964621_N2048.npz';b=OUT/'periodic/verified_tonic_low_continued_N1024/g1.34964621_N2048.npz'
    if a.exists() and b.exists():connections.append(dict(name='mixed_tonic_N2048',**compare(a,b)));write('connection_checks.json',connections)
    for c in connections:
        assert abs(c['J_difference'])<1e-9 and abs(c['period_difference_ms'])<1e-6
        assert c['max_waveform_difference_hz']<1e-4,c
    crit=[]
    for name,folder in [('burst_end_fold','folds'),('mixed_lower_flip','flips'),('mixed_flip','flips'),('tonic_lower_flip','flips')]:
        a=read(OUT/folder/f'{name}_N2048.json');b=read(OUT/folder/f'{name}_N4096.json');row=dict(name=name,J=a['g'],J_resolution_difference=abs(a['g']-b['g']),period_resolution_difference_ms=abs(a['T_ms']-b['T_ms']))
        assert row['J_resolution_difference']<1e-9
        if folder=='folds':
            assert a['fixed_parameter_null_residual']<1e-8 and abs(a['left_Fg'])>1e-6
            assert a['second_derivatives'][0]['left_Fvv']*a['second_derivatives'][1]['left_Fvv']>0
        else:assert a['anti_null_residual']<1e-8 and a['left_null_residual']<1e-8
        crit.append(row)
    s=System();checks=[];paths=[]
    for c in connections:paths.extend([Path(c['source_a']),Path(c['source_b'])])
    for name in ['mixed_high_continued_N1024','tonic_low_continued_N1024']:
        rows=read(OUT/'arcs'/name/'progress.json');paths.extend(Path(c['source']) for c in rows[::5])
    for path in dict.fromkeys(paths):
        z=np.load(path);N=len(z['r']);r=resample(z['r'],2*N,axis=0);F=Orbit(s,float(z['g']),2*N).evaluate(np.r_[(r/.01).ravel(),np.log(float(z['T']))],r,np.zeros_like(r));defect=float(abs(F[:-1]).max()*10)
        checks.append(dict(source=str(path),N=N,offgrid_defect_hz=defect));print('OFFGRID',path.name,defect,flush=True)
        assert defect<5e-5,(path,defect)
    children=[]
    for name in ['mixed_lower_period2','mixed_period2','tonic_lower_period2']:
        fl=read(OUT/'poincare'/name/'amp0.06_N4096/rk4_orthogonal_dt0.05.json');row=read(OUT/'periodic'/name/'amp0.06_N4096.json');isstable=name!='mixed_period2'
        assert (fl['max_transverse']<1)==isstable and row['half_period_difference']>.001
        assert fl['orbit_tangent_defect']<1e-6
        children.append(dict(name=name,J=row['g'],period_ms=row['T_ms'],stable=isstable,max_transverse=fl['max_transverse'],half_period_difference=row['half_period_difference']))
    all_children=[]
    for name in ['mixed_lower_period2','mixed_period2','tonic_lower_period2']:
        for path in (OUT/'periodic'/name).glob('amp*.json'):
            a=read(path);fl=read(OUT/'poincare'/name/path.stem/'rk4_orthogonal_dt0.05.json')
            assert (fl['max_transverse']<1)==(name!='mixed_period2')
            all_children.append(dict(source=a['source'],amplitude=a['amplitude'],max_transverse=fl['max_transverse'],orbit_tangent_defect=fl['orbit_tangent_defect']))
    assert len(all_children)==10
    switching=[]
    for name in ['after_burst_fold','after_mixed_flip']:
        a=read(OUT/'transitions'/f'{name}_dt0.05.json');b=read(OUT/'transitions'/f'{name}_dt0.025.json');dd=float(max(abs(np.array(a['mean_hz'])-b['mean_hz'])));dt=abs(a['recurrence']['period_ms']-b['recurrence']['period_ms']);assert dd<.01 and dt<.001
        switching.append(dict(name=name,max_mean_resolution_difference_hz=dd,period_resolution_difference_ms=dt))
    row=dict(status='PASS',connection_checks=connections,critical_resolution=crit,offgrid_checks=checks,period_doubled_children=children,all_period_doubled_points=all_children,step_refinement=switching,
        limits=['Critical parameter precision describes this frozen deterministic closure, not biological parameter accuracy.','Poincare spectra on strongly expanding unstable arcs can be ill-conditioned; no fine subdominant Floquet claims are inferred there.','The full global inventory of secondary bifurcations and remote attractors is not exhausted.'])
    write('numerical_validation.json',row);print('VALIDATION_PASS',flush=True)

if __name__=='__main__':main()
