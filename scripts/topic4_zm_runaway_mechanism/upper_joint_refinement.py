"""Evaluate the recorded joint orbit/time refinement contract without hiding failures."""
from pathlib import Path
import json
import numpy as np
from scipy.signal import resample

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918'


def read(p):return json.loads(Path(p).read_text())


def main():
    contract=read(OUT/'upper_joint_refinement_contract.json');gates=contract['criteria']
    coarse=OUT/contract['coarse_source'];fine=OUT/contract['intended_fine_source']
    meta0=OUT/contract['coarse_spectrum']
    meta1=OUT/'floquet'/(fine.parent.name+'_'+fine.stem+
        '_endpoint_dt0.0018_quotient_chainphase_rk4_cubic_streamed_fastgrid.json')
    q=dict(status='WAITING_FOR_FINE_ORBIT_AND_SPECTRUM',sources=[str(meta0),str(meta1)],
        orbit=str(fine),contract=str(OUT/'upper_joint_refinement_contract.json'),
        scope='Upper periodic solution only; joint orbit/quadrature/time refinement, not a same-orbit time-step test',
        bifurcation_type='NOT_CERTIFIED_BY_THIS_TEST_ALONE')
    failed=meta1.with_name(meta1.stem+'.phase.json')
    if failed.exists() and not read(failed)['phase_valid']:
        q.update(status='FINE_PHASE_CHECK_FAILED',phase_check=read(failed))
    if fine.exists() and meta1.exists():
        z0=np.load(coarse);z1=np.load(fine);a=read(meta0);b=read(meta1)
        assert Path(a['orbit']).resolve()==coarse.resolve() and Path(b['orbit']).resolve()==fine.resolve()
        rr=resample(z0['r'],len(z1['r']),axis=0)
        values=dict(period_difference_ms=abs(float(z0['T'])-float(z1['T'])),
            D_difference=abs(float(z0['D'])-float(z1['D'])),
            Z_difference=float(np.max(abs(z0['Z']-z1['Z']))),
            rate_relative_L2_difference=float(np.linalg.norm(rr-z1['r'])/np.linalg.norm(z1['r'])),
            BVP_residuals=[float(z0['residual']),float(z1['residual'])],
            phase_defects=[v['phase_defect'] for v in [a,b]],
            phase_projection_errors=[abs(v['phase_projection']-1) for v in [a,b]],
            eigen_residuals=[max(v['eigen_residuals']) for v in [a,b]],
            leading_multipliers=[v['multipliers'][0] for v in [a,b]],
            actual_dt_ms=[v['dt_ms'] for v in [a,b]],N=[len(z0['r']),len(z1['r'])])
        mu=[complex(*v) for v in values['leading_multipliers']]
        values['multiplier_difference']=abs(mu[0]-mu[1])
        checks=dict(
            BVP=max(values['BVP_residuals'])<gates['both_BVP_residual_max_hz'],
            period=values['period_difference_ms']<gates['max_period_difference_ms'],
            parameter=values['D_difference']<gates['max_D_difference'],
            spatial_Z=values['Z_difference']<gates['max_Z_difference'],
            waveform=values['rate_relative_L2_difference']<gates['max_rate_relative_L2_difference'],
            phase=all(v['phase_valid'] for v in [a,b]) and max(values['phase_defects'])<gates['both_phase_defect_max'],
            neutral_projection=max(values['phase_projection_errors'])<gates['both_phase_projection_error_max'],
            eigen_residual=max(values['eigen_residuals'])<gates['both_eigen_residual_max'],
            multiplier_agreement=values['multiplier_difference']<gates['max_multiplier_difference'],
            real_positive_instability=all(v.real>gates['both_leading_real_multiplier_min'] and abs(v.imag)<1e-5 for v in mu),
            finer_discretizations=len(z1['r'])>len(z0['r']) and b['dt_ms']<a['dt_ms'])
        q.update(status=contract['acceptance_label'] if all(checks.values()) else 'JOINT_REFINEMENT_NOT_ACCEPTED',
            checks=checks,values=values,D=float(z1['D']),T_ms=float(z1['T']),
            retained_failed_same_orbit_test='N16385 at finer dt=.0018 failed phase invariance; see previous phase/progress files')
    (OUT/'floquet/rate_near_upper_joint_stability_acceptance.json').write_text(json.dumps(q,indent=2)+'\n')
    print(json.dumps(q,indent=2))


if __name__=='__main__':main()
