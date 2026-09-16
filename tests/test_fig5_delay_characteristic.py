from test_fig5_local_bifurcation import setup
import numpy as np
from scipy import linalg
from src.topic4_fig5_local_bifurcation import corrected_delay_matrix
from src.topic4_fig5_delay_characteristic import DelayCharacteristic
from src.topic4_fig5_delay_characteristic import positive_majorant_certificate


def test_characteristic_matches_native_map_eigenvalues():
    f=setup('tau_gaba');x=np.array([.08,.12]);c=DelayCharacteristic(f,x)
    vals=linalg.eigvals(corrected_delay_matrix(f,x,1).toarray())
    tested=0
    poles=[1-.1/f.base.tau_ampa_ms,1-.1/f.base.tau_gaba_ms,1-.1/500]
    for value in vals:
        if abs(value)<.5 or min(abs(value-p) for p in poles)<1e-7:continue
        exponent=np.log(complex(value))/.1
        singular=linalg.svdvals(c.matrix(exponent,1,.1))[-1]
        assert singular<1e-7
        tested+=1
    assert tested>=2


def test_majorant_certificate_against_full_spectrum():
    from dataclasses import replace
    f=setup('tau_gaba');certified=0;f.eta=.0001
    f.base=replace(f.base,**{p+k:getattr(f.base,p+k)*(.01 if p=='w_' else .0001)
                            for p in ('w_','v_') for k in ('ee','ei','ie','ii')})
    f.operators=replace(f.operators,**{'w_'+k+'_history':getattr(f.operators,'w_'+k+'_history')*.01
                                      for k in ('ee','ei','ie','ii')})
    for rate in [.00001,.001,.01,.08,.3]:
        x=np.array([rate,rate]);r=positive_majorant_certificate(f,x,1.)
        if r['certified']:
            certified+=1
            spectral=max(abs(linalg.eigvals(corrected_delay_matrix(f,x,1).toarray())))
            assert spectral<r['multiplier_bound']
    assert certified>0
