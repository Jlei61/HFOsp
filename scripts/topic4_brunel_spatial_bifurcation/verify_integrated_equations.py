"""Bounded CPU checks using only the spatial package in this checkout.

Re-evaluates saved equilibria and both Hopf tangent states. Does not launch
continuation, native SNN runs, or certify all saved periodic branches.
"""
from pathlib import Path
import json
import numpy as np
from rate_field import RateField, ROOT, OUT, RATE_OUT


def main():
    system = RateField()
    assert system.folder.is_relative_to(ROOT)
    assert system.P == 935 and len(np.unique(system.geo['group_cell'])) == 400
    assert int(system.geo['group_size'].sum()) == 40000
    with np.load(OUT/'critical_revision/branch.npz') as branch:
        residuals = []
        for i in np.linspace(0, len(branch['J'])-1, 25, dtype=int):
            coupling = float(branch['J'][i])
            rates = branch['rates'][i]
            state = system.equilibrium_state(rates, coupling)
            assert state.shape == (9, 935)
            arrivals = np.array([a @ rates for a in system.matrices(coupling)])
            residuals.append(float(np.abs(system.rhs(state, arrivals)).max()))
    assert max(residuals) < 1e-7
    tangent_checks = []
    for core in ('A', 'B'):
        with np.load(RATE_OUT/f'hopf_{core}.npz') as z:
            rates, coupling, lam, vector = z['rates'], float(z['J']), 1j*float(z['omega']), z['vector']
        state = system.equilibrium_state(rates, coupling)
        tangent = system.eigenstate(rates, coupling, lam, vector)
        arrivals = np.array([a @ rates for a in system.matrices(coupling)])
        delta = np.array([a @ vector for a in system.matrices(coupling, lam)])
        step = 1e-5 / np.max(np.abs(tangent))
        finite = []
        for component in (np.real, np.imag):
            plus = system.rhs(state+step*component(tangent), arrivals+step*component(delta))
            minus = system.rhs(state-step*component(tangent), arrivals-step*component(delta))
            finite.append((plus-minus)/(2*step))
        error = float(np.linalg.norm(finite[0]+1j*finite[1]-lam*tangent)/np.linalg.norm(lam*tangent))
        assert error < 1e-4
        tangent_checks.append(dict(core=core, J_EE_core=coupling, relative_error=error))
    result = dict(status='PASS', model_id='SPATIAL_RATE_INTERICTAL', checkout=str(ROOT),
                  spatial_cells=400, populations=935, local_states=8415,
                  equilibrium_checks=25, maximum_rhs_residual=max(residuals), tangent=tangent_checks,
                  external_artifacts_used=False,
                  scope='Saved equilibria and two Hopf tangent states only; no new periodic or native-equivalence certification.')
    destination = ROOT/'docs/archive/workspace/integration_2026-09-30/spatial_equation_validation.json'
    destination.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
