"""Identify the child radial eigenfunction by physical branch tangents.

Remove the common phase tangent only for this comparison. The variational
equation and independent full-state propagation checks retain that mode.
"""
from rate_periodic import *


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--N', type=int, default=2048)
    p.add_argument('--label', default='PD4_physical_child_radial_20260920')
    args = p.parse_args()
    folder = Path('/data/hfosp/topic4_sef_hfo/interictal_rate_branch_completion_20260920/H2_local_PD')
    label = args.label
    source = PERIODIC_OUT/f'{label}_N{args.N}.json'
    result = read(source)
    assert result['status'] == 'EIGENPAIR_CONVERGED_CHECKS_PENDING'
    rows = read(folder/'physical_children.json')['rows']
    target = Path(result['orbit'])
    assert any(Path(row['orbit']).resolve() == target.resolve() for row in rows)
    s = RateField()
    weight = s.geo['group_size']/s.geo['group_size'].sum()
    n = args.N
    r = resample(np.load(target)['r'], n, axis=0)
    phase = np.fft.irfft(np.fft.rfft(r, axis=0)*
                        (2j*np.pi*np.arange(n//2+1))[:, None], n=n, axis=0)

    def inner(x, y):
        return float(np.mean(np.sum(x*y*weight, axis=1)))

    def quotient(x):
        return x-phase*(inner(phase, x)/inner(phase, phase))

    def cosine(x, y):
        x, y = quotient(x), quotient(y)
        return abs(inner(x, y))/np.sqrt(inner(x, x)*inner(y, y))

    u = np.load(PERIODIC_OUT/f'{label}_mode_N{n}.npz')['u']
    parent = read(PERIODIC_OUT/'PD_H2_after_LPC13_validation.json')
    seed = np.load(parent['accepted_mode'])['u'].real
    seed = resample(np.r_[seed, -seed], n, axis=0)
    comparisons = []
    for earlier in rows:
        if Path(earlier['orbit']).resolve() == target.resolve():
            continue
        lower = resample(np.load(earlier['orbit'])['r'], n, axis=0)
        # All children use the same parent phase condition. Dividing by the
        # amplitude difference would cancel in this normalized comparison.
        tangent = r-lower
        comparisons.append(dict(earlier_orbit=earlier['orbit'],
            phase_quotient_mode_tangent_cosine=cosine(u, tangent),
            parent_mode_tangent_cosine=cosine(seed, tangent)))
    identity = cosine(u, seed)
    qu = quotient(u)
    odd = (qu-np.roll(qu, n//2, axis=0))/2
    odd_fraction = inner(odd, odd)/inner(qu, qu)
    passed = identity > .95 and all(
        q['phase_quotient_mode_tangent_cosine'] > .95 for q in comparisons)
    output = dict(status='RADIAL_MODE_IDENTITY_PASS' if passed else 'REVIEW_REQUIRED',
        source=str(source),orbit=str(target),N=n,multiplier=result['multiplier'],
        parent_antiperiodic_mode_cosine=identity,
        odd_half_period_energy_fraction=odd_fraction,
        tangent_comparisons=comparisons,
        scope='Neuron-weighted waveform comparison after quotienting the common phase. Confirms local eigenmode identity only; not a replacement for variational residual, temporal refinement, or full delay-history propagation.')
    write(folder/f'{label}_identity_N{n}.json', output)
    print(output, flush=True)
    assert passed, 'Do not classify PD4 using an unidentified real eigenmode'


if __name__ == '__main__':
    main()
