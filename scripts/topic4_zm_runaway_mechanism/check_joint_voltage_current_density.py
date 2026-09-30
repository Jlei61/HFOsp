"""Independent affine/moment and deterministic timing checks, no fitting."""
from common import OUT, np, write
from joint_voltage_current_density import joint_advance, truncate, reset_voltage, joint_partition, voltage_edges, simulate_joint
from lif_mc import condition
from scipy.integrate import quad


def main():
    rng = np.random.default_rng(20920)
    transition_error = 0.; partition_error = 0.; quadrature_error = 0.
    for pop in ['E', 'I']:
        pars = condition(0., 18., 1., 1., pop, dt=.05)
        for index in range(6):
            mean = rng.normal(size=5); mean[0] += 17.
            root = rng.normal(size=(5, 5)); covariance = root@root.T+.2*np.eye(5)
            raw = np.empty((6, 6)); raw[0, 0] = .3
            raw[0, 1:] = raw[1:, 0] = .3*mean
            raw[1:, 1:] = .3*(covariance+np.outer(mean, mean))
            # Homogeneous affine matrix and innovation covariance, built
            # independently from native parameter entries.
            A = np.zeros((6, 6)); A[0, 0] = A[1, 1] = 1.
            A[2, 2] = pars[6]; A[3, 2] = pars[7]; A[3, 3] = pars[8]
            A[4, 4] = pars[17]; A[5, 4] = pars[9]; A[5, 5] = pars[10]
            B = np.zeros((6, 4)); B[2, 0] = pars[11]; B[3, :2] = pars[12:14]
            B[4, 2] = pars[14]; B[5, 2:] = pars[15:17]
            ve, vi, mu = 1.3, .7, 40.
            variances = np.array([ve, ve, vi, vi])
            before = A@raw@A.T+.3*(B*variances)@B.T
            for active in [False, True]:
                membrane = np.eye(6); membrane[1] = 0.
                if active:
                    membrane[1, [0, 1, 3, 5]] = [(1-pars[18])*mu, pars[18], 1-pars[18], -(1-pars[18])]
                else:
                    membrane[1, 0] = pars[21]
                expected = membrane@before@membrane.T
                actual = joint_advance(raw, pars, mu, ve, vi, active)
                transition_error = max(transition_error, float(abs(expected-actual).max()))
            sigma = np.sqrt(covariance[0, 0]); direction = covariance[:, 0]/sigma
            low, high = -1.7, .8
            actual = truncate(raw, mean, direction, low, high)
            residual = np.zeros((6, 6)); residual[1:, 1:] = covariance-np.outer(direction, direction)
            for i in range(6):
                for j in range(6):
                    def integrand(x):
                        value = np.r_[1., mean+direction*x]
                        return .3*(value[i]*value[j]+residual[i, j])*np.exp(-x*x/2)/np.sqrt(2*np.pi)
                    expected = quad(integrand, low, high, epsabs=1e-11, epsrel=1e-11)[0]
                    quadrature_error = max(quadrature_error, abs(expected-actual[i, j]))
            edges = voltage_edges(18., 11., 32); free = np.zeros((len(edges)-1, 6, 6)); spikes = np.zeros((6, 6))
            joint_partition(raw, edges, 11., free, spikes)
            total = free.sum(0)+spikes; keep = [0, 2, 3, 4, 5]
            partition_error = max(partition_error, float(abs(total[np.ix_(keep, keep)]-raw[np.ix_(keep, keep)]).max()))
            assert free[:, 0, 0].min() > -1e-12 and spikes[0, 0] >= 0
    assert transition_error < 1e-10 and quadrature_error < 1e-10 and partition_error < 1e-10
    deterministic = []
    for pop in ['E', 'I']:
        dt = .05; pars = condition(0., 18., 0., 0., pop, dt=dt)
        wave = np.zeros((3, 2)); wave[0] = 40.
        voltage = 11.; refractory = 0; exact = np.zeros(2000)
        for tick in range(len(exact)):
            refractory = max(refractory-1, 0)
            if refractory == 0:
                voltage = pars[18]*voltage+(1-pars[18])*40.
                if voltage >= 18.:
                    exact[tick] = 1000/dt; voltage = 11.; refractory = int(pars[19])
            else:
                voltage = 11.
        for nodes in [32, 128]:
            answer = simulate_joint(pars, wave, voltage_edges(18., 11., nodes), dt, 100., 0, 2000, 100)
            assert np.array_equal(answer[3], exact), (pop, nodes, np.max(abs(answer[3]-exact)))
            assert np.max(abs(answer[6][:4])) < 1e-10
            deterministic.append(dict(population=pop, grid=nodes, stepwise_spikes_exact=True))
    result = dict(status='JOINT_MOMENT_IMPLEMENTATION_PASS', affine_transition_max_error=transition_error,
                  independent_quadrature_max_error=quadrature_error, current_partition_max_error=partition_error,
                  deterministic=deterministic, model_promoted=False,
                  scope='Algebra and noiseless discrete-LIF invariance only; no noisy response, compact rate model, network or bifurcation acceptance.')
    write(OUT / 'joint_voltage_current_density_implementation_check.json', result)
    print(result)


if __name__ == '__main__':
    main()
