"""Manufactured-solution check of a positivity-preserving renewal integrator.

These scalar tests validate a numerical formula, not network dynamics.
"""
from common import OUT, np, write


def solve(h, tau, mean, amplitude, omega, duration, midpoint):
    n = round(tau/h); assert abs(n*h-tau) < 1e-12
    def average(a, b):
        return mean + amplitude*(np.cos(omega*a)-np.cos(omega*b))/(omega*(b-a))
    def occupancy(t):
        return mean*tau + amplitude*(np.cos(omega*(t-tau))-np.cos(omega*t))/omega
    def rho(t):
        return (mean+amplitude*np.sin(omega*t))/(1-occupancy(t))
    hist = average(np.arange(-n, 0)*h, np.arange(-n+1, 1)*h)
    initial = hist.copy(); values = []; exact = []; states = []
    for k in range(round(duration/h)):
        t = k*h; slot = k%n; old = h*hist.sum(); release = hist[slot]
        hazard = rho(t + (h/2 if midpoint else h))
        if midpoint:
            p = -np.expm1(-hazard*h)
            r = (1-old)*p/h + release*(1-p/(hazard*h))
        else:
            r = hazard*(1-old+h*release)/(1+hazard*h)
        hist[slot] = r; values.append(r); exact.append(average(t, t+h)); states.append(h*hist.sum())
    values, exact, states = map(np.asarray, [values, exact, states])
    assert values.min() >= 0 and states.min() >= 0 and states.max() <= 1+1e-13
    return dict(dt_ms=h, bin_flux_rms_error_per_ms=float(np.sqrt(np.mean((values-exact)**2))),
                bin_flux_max_error_per_ms=float(abs(values-exact).max()),
                final_occupancy_error=float(abs(states[-1]-occupancy(duration))),
                flux_min=float(values.min()), occupancy_range=[float(states.min()), float(states.max())],
                initial_integral_error=float(abs(initial.sum()*h-occupancy(0))))


def main():
    cases = [dict(name='smooth_E', tau=2., mean=.25, amplitude=.08, omega=.7),
             dict(name='smooth_I', tau=1., mean=.45, amplitude=.15, omega=1.3),
             dict(name='near_saturation_E', tau=2., mean=.499, amplitude=.0001, omega=.7)]
    rows = []
    for case in cases:
        pars = {k:v for k,v in case.items() if k != 'name'}
        for method in ['old_endpoint', 'exponential_midpoint']:
            meshes = [.05, .025, .0125, .00625]
            results = [solve(h, duration=20., midpoint=method=='exponential_midpoint', **pars) for h in meshes]
            ratios = [a['bin_flux_rms_error_per_ms']/b['bin_flux_rms_error_per_ms'] for a,b in zip(results[:-1],results[1:])]
            rows.append(dict(case=case['name'], method=method, parameters=pars, results=results, error_ratios=ratios))
            print(case['name'], method, ratios, flush=True)
    equilibrium = []
    for hazard in [.01, 1., 100., 1e5]:
        for tau in [1., 2.]:
            r = hazard/(1+tau*hazard)
            for h in [.1, .05, .0125]:
                p = -np.expm1(-hazard*h)
                value = (1-tau*r)*p/h + r*(1-p/(hazard*h))
                error = abs(value-r); assert error < 1e-13
                equilibrium.append(dict(hazard=hazard, tau=tau, dt=h, error=error))
    out = OUT/'core_a_bifurcation_type_20260924/numerical_checks/exponential_midpoint'
    write(out/'scalar_renewal_check.json', dict(status='MANUFACTURED_SCALAR_CHECK_COMPLETE',
          rows=rows, equilibria=equilibrium,
          scope='Exact scalar manufactured hazards and analytic bin-integrated rates; validates only this numerical renewal formula, not full spatial midpoint integration or bifurcation. Stiff cases must be interpreted separately.', model_promoted=False))


if __name__ == '__main__': main()
