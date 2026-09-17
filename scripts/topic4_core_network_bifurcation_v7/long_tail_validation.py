"""Reject a gain-biased apparent flip; retain a conservative stability bound."""
from common import *
gain=read(OUT/'analytic_gain_validation.json');phase=read(OUT/'long_tail_phase_origin.json')
accepted=[]
for folder,dt in [('long_period_tail',.025),('long_period_tail',.0125),('long_period_tail_phase',.025)]:
    r=read(OUT/f'analytic_poincare/{folder}/T2200_N16384/rk4_orthogonal_dt{dt:g}.json')
    assert r['gain_derivative']=='analytic frozen transfer quadrature'
    assert r['max_transverse']<.02 and r['orbit_tangent_defect']<.002 and max(r['residuals'])<2e-5
    accepted.append(r)
rejected=[]
for folder,dt in [('long_period_tail',.025),('long_period_tail',.0125),('long_period_tail_phase',.025)]:
    rejected.append(read(OUT/f'poincare/{folder}/T2200_N16384/rk4_orthogonal_dt{dt:g}.json'))
assert gain['analytic_chain_rule_error']<1e-12 and phase['mean_difference_hz']<1e-10
write('long_tail_stability_validation.json',dict(status='STABLE_BOUND',scope='Stable 2200ms finite-period sample. Exact small multiplier not precision accepted; no infinite-time homoclinic BVP.',
    accepted_analytic_returns=accepted,rejected_finite_difference_gain_returns=rejected,
    gain_validation=gain,phase_origin=phase,
    interpretation='Analytic differentiation of the same transfer removes the apparent multiplier below -1. Independent starting phases and step sizes have transverse modulus far below one. Finite-difference gain bias is strongly amplified near the saddle; reducing only integration dt did not cure it.'))
print('LONG_TAIL_STABLE_BOUND',[r['max_transverse'] for r in accepted],flush=True)
