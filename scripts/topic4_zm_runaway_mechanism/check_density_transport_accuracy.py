"""Zero-noise transport diagnostic, independent of Gaussian current closure.

Identical noise-free cells must stay identical. Positive voltage remapping
can broaden their distribution even though it preserves its mean. This
test measures that numerical broadening, not a physical population variance.
"""
from common import OUT, np, read, write
from conditional_current_density import simulate, voltage_grid, deposit_point
from lif_mc import condition
from datetime import datetime
import time


def main():
    destination = OUT / 'conditional_density_transport_accuracy'
    destination.mkdir(exist_ok=True)
    contract_path = destination / 'contract.json'
    assert not contract_path.exists()
    write(contract_path, dict(created_local=datetime.now().astimezone().isoformat(),
          question='Does voltage remapping broaden identical zero-noise cells, separately from conditional-current Gaussian closure?',
          populations=['E', 'I'], mean_input_mv=40., threshold_mv=18., dt_ms=.05,
          duration_ms=100., burn_ms=0., grids=[128, 256, 512, 1024],
          selection='Two population types under identical deterministic input; no tuning against original weak-response errors.',
          readout='One-step voltage variance created by remapping, full finite-time rate discrepancy, first-spike distribution mean/spread, mass accounting.',
          scope='Numerical diagnosis. First-spike spread is not physiological variability and does not by itself establish which term dominates the noisy gain error.'))
    rows = []
    for pop in ['E', 'I']:
        dt = .05; steps = 2000
        pars = condition(0., 18., 0., 0., pop, dt=dt)
        wave = np.zeros((3, 2)); wave[0] = 40.
        voltage = pars[21]; refractory = 0; spikes = np.zeros(steps)
        for j in range(steps):
            refractory = max(refractory-1, 0)
            if refractory == 0:
                voltage = pars[18]*voltage+(1-pars[18])*40.
                if voltage >= pars[1]:
                    spikes[j] = 1.; voltage = pars[21]; refractory = int(pars[19])
            else:
                voltage = pars[21]
        first = np.flatnonzero(spikes)[0]
        # Use the midpoint between the first two exact spikes, before any
        # second exact spike; retain mass explicitly if the tails differ.
        second = np.flatnonzero(spikes)[1]
        window_end = (first+second)//2
        for nodes in [128, 256, 512, 1024]:
            grid = voltage_grid(18., 11., nodes)
            initial = np.zeros((5, 5)); initial[0, 0] = 1.
            ideal_voltage = pars[18]*11.+(1-pars[18])*40.
            mapped = np.zeros((len(grid), 5, 5))
            deposit_point(initial, ideal_voltage, grid, mapped)
            p = mapped[:, 0, 0]
            variance_added = float(np.dot(p, (grid-ideal_voltage)**2))
            assert abs(np.dot(p, grid)-ideal_voltage) < 1e-11
            started = time.monotonic()
            answer = simulate(pars, wave, grid, dt, 100., 0, steps, 100)
            probability = answer[3]*dt/1000.
            times = (np.arange(window_end+1)+1)*dt
            mass = probability[:window_end+1].sum()
            mean = float(np.dot(times, probability[:window_end+1])/mass)
            spread = float(np.sqrt(np.dot((times-mean)**2, probability[:window_end+1])/mass))
            row = dict(population=pop, grid=nodes, one_step_added_voltage_variance_mv2=variance_added,
                       exact_first_spike_ms=float((first+1)*dt), first_spike_mass=float(mass),
                       predicted_first_spike_mean_ms=mean, first_spike_sd_ms=spread,
                       exact_spike_count=int(spikes.sum()), predicted_spike_count=float(probability.sum()),
                       largest_step_spike_probability=float(probability.max()),
                       conservation=answer[6].tolist(), elapsed_seconds=time.monotonic()-started)
            rows.append(row)
            np.savez_compressed(destination/f'{pop}_grid{nodes}.npz',
                                predicted_spike_probability=probability, exact_spike=spikes, dt_ms=dt)
            print(row, flush=True)
    result = dict(status='ZERO_NOISE_TRANSPORT_DIAGNOSIS_COMPLETE', rows=rows,
                  physical_noise_variance=0., model_promoted=False,
                  interpretation='Positive remapping creates voltage dispersion even for identical noiseless cells. Refinement trend is reported; this test alone does not attribute every noisy gain error to this dispersion.')
    write(destination / 'result.json', result)


if __name__ == '__main__':
    main()
