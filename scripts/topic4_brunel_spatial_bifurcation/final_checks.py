"""Numerical consistency and compact evidence inventory for the spatial analysis."""
from common import *
from model import SpatialBrunel
from response import characteristic, white
import mpmath as mp


def main():
    s = SpatialBrunel(response='calibrated_full')
    r, ok, _ = s.solve(.94)
    assert ok
    J = .94
    static = characteristic(s, r, J, 0.) + s.jacobian(r, J)
    dc_error = float(abs(static).max())
    lam = .003 + .037j
    m = characteristic(s, r, J, lam)
    conjugacy = float(abs(characteristic(s, r, J, lam.conjugate()) - m.conjugate()).max())
    h = 1e-6
    dr = (characteristic(s, r, J, lam+h)-characteristic(s, r, J, lam-h))/(2*h)
    di = (characteristic(s, r, J, lam+1j*h)-characteristic(s, r, J, lam-1j*h))/(2j*h)
    cr_error = float(sparse.linalg.norm(dr-di)/sparse.linalg.norm(dr))
    assert dc_error < 1e-12 and conjugacy < 1e-12 and cr_error < 1e-6
    mp.mp.dps = 45
    details = s.phi(*s.moments(r,J),details=True)
    group = 293
    errors = []
    for f in (1.,5.,20.,80.,200.):
        lo, hi, sig = [float(x[group]) for x in details[1:4]]
        tm = float(s.tm[group]); rate = float(details[0][group])
        z = mp.mpc(0,2*np.pi*f*tm/1000)
        def u(y):
            return mp.hyperu(z/2,mp.mpf('.5'),y*y) if y <= 0 else (
                mp.sqrt(mp.pi)/mp.gamma((1+z)/2)*mp.hyp1f1(z/2,mp.mpf('.5'),y*y)
                + 2*mp.sqrt(mp.pi)*y/mp.gamma(z/2)*mp.hyp1f1((1+z)/2,mp.mpf('1.5'),y*y))
        denominator = u(hi)-u(lo)
        ref = np.array([complex(rate/sig*(mp.diff(u,hi)-mp.diff(u,lo))/denominator/(1+z)),
                        complex(rate/sig**2*(mp.diff(u,hi,2)-mp.diff(u,lo,2))/denominator/(2+z))])
        got = white(2j*np.pi*f/1000,np.array([lo]),np.array([hi]),np.array([sig]),np.array([tm]),np.array([rate]))[:,0]
        error = float(np.linalg.norm(got-ref)/np.linalg.norm(ref))
        assert error < 1e-9
        errors.append(dict(frequency_hz=f,relative_error=error))
    write(OUT/'calibrated_numerical_checks.json',dict(response='calibrated_full',DC_operator_error=dc_error,
        conjugacy_error=conjugacy,complex_analytic_derivative_relative_error=cr_error,hypergeometric_checks=errors))

    critical=[]
    for g in (20,40):
        for core in ('A','B'):
            path=OUT/f'g{g}/hopf_{core}_calibrated_full/result.json'
            if not path.exists():
                raise FileNotFoundError(path)
            q=read(path)
            critical.append(dict(grid=g,space_cells=g*g,rate_groups=q['rate_groups'],kind='oscillatory crossing',core=core,
                J_EE_core=q['J_EE_core'],frequency_hz=q['frequency_hz'],equilibrium_residual=q['equilibrium_residual'],
                characteristic_residual=q['characteristic_residual'],regional_energy=q['regional_energy'],inhibitory_energy=q['inhibitory_energy']))
        q=read(OUT/f'g{g}/fold/result.json')
        critical.append(dict(grid=g,space_cells=g*g,kind='stationary fold',J_EE_core=q['J_EE_core'],source=str(OUT/f'g{g}/fold/result.json')))
    def compact_fits(name):
        return [{k:v for k,v in q.items() if k!='rows'} for q in read(OUT/f'local_response_fit/{name}')['rows']]
    write(OUT/'analysis_summary.json',dict(
        status='SPATIAL_LINEAR_ONSET_ANALYSIS_COMPLETE_NONLINEAR_REPLACEMENT_NOT_VALIDATED',
        reference=dict(title='A Spatially Structured Spiking Network Model of Beta Traveling Waves and Their Attenuation in Motor Cortex',
            doi='10.64898/2026.03.18.712701',local_file=str(ROOT/'docs/paper/brunel2026/paper.md'),
            method='Spatial stationary self-consistency and mean/variance susceptibility, Eqs 13-30 and 36; local causal response calibration replaces extrapolation of authors Table 2'),
        actual_network=dict(topology=6101,neurons=40000,spatial_extent_mm=[20,20],threshold_bin_mv=.5,
            parameter='Within-core EE multiplier applied to both A and B; mean weights multiply J, squared weights multiply J squared',
            Z='fixed interictal baseline 1',M='original dynamic feedback',
            forcing='Original private Poisson mean and variance; shared and spatial OU fluctuations held at zero'),
        critical_points=critical,
        nyquist=read(OUT/'g20/nyquist_calibrated_full/result.json'),
        local_mean_response=compact_fits('result.json'),local_variance_response=compact_fits('variance_result.json'),
        native_check_source=str(OUT/'native_private_only/summary.json'),
        limits=[
            'The stationary transfer is a colored-noise diffusion approximation; local response spectral calibration preserves its DC gains. Static gain discrepancies, especially inhibitory variance gain, remain.',
            'Oscillatory crossings are numerically established in this approximation. Supercritical/subcritical Hopf classification and nonlinear periodic branches have not been calculated.',
            'Spatial mode amplitudes are infinitesimal eigenvectors, not finite-amplitude event propagation snapshots or SEEG rank validation.',
            'Native checks are single-seed 5-second runs with 0.5-second burn-in and new frozen-Z, zero-OU conditions. They are not full original-condition interictal distribution recovery.',
            'Native events occur below and above the crossings; no equivalence of Hopf frequency and burst repetition frequency is asserted.',
            'A second localized mode crossing on an unstable branch is not a codimension-two double-Hopf point.',
            'Nyquist count is numerically refined over the sampled frequency contour, with a small loop norm at the endpoint; no analytic bound for every unsampled tail frequency is claimed.'],
        human_visual_acceptance=False))
    print('Saved calibrated checks and complete evidence inventory',flush=True)


if __name__=='__main__':
    main()
