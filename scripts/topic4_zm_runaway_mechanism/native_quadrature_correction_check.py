"""Apply the pre-run comparison criteria to the doubled-quadrature root."""
from native_joint_orbit_check import OUT,phase_distance,sanity,np,json


def main():
    sanity()
    contract_path=OUT/'native_T2644_quadrature_correction_contract.json'
    contract=json.loads(contract_path.read_text());gates=contract['gates']
    source=OUT/contract['source'];target=OUT/contract['target']
    a=np.load(source);b=np.load(target)
    assert len(a['r'])==len(b['r'])==contract['N']
    values=dict(residuals_hz=[float(a['residual']),float(b['residual'])],
        D_difference=abs(float(a['D'])-float(b['D'])),
        max_Z_difference=float(np.max(abs(a['Z']-b['Z']))),
        period_difference_ms=abs(float(a['T'])-float(b['T'])),
        **phase_distance(a['r'],b['r']))
    checks=dict(residual=max(values['residuals_hz'])<gates['root_maxF_hz'],
        period=values['period_difference_ms']<1e-8 and abs(float(b['T'])-contract['period_ms'])<1e-8,
        D=values['D_difference']<gates['max_D_difference'],
        Z=values['max_Z_difference']<gates['max_Z_difference'],
        waveform=values['phase_aligned_relative_L2']<gates['max_phase_aligned_rate_relative_L2_difference'])
    q=dict(status='ORBIT_QUADRATURE_CHECK_PASS' if all(checks.values()) else 'ORBIT_QUADRATURE_CHECK_NOT_MET',
        contract=str(contract_path),sources=[str(source),str(target)],values=values,checks=checks,
        scope='Orbit comparison only. Does not accept phase invariance, Floquet spectrum, a fold, or native-SNN correspondence.')
    (OUT/'periodic/native_T2644_quadrature_correction_check.json').write_text(json.dumps(q,indent=2)+'\n')
    print(json.dumps(q,indent=2),flush=True)


if __name__=='__main__':main()
