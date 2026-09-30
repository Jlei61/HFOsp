#!/usr/bin/env python3
"""One paired fine-group input repair, with the frozen density kernel."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import time
import numpy as np
from campaign import ROOT, REPO, read, write, sha
import density_spatial as engine

OUT = ROOT / 'density_fine_forcing'
BASE = ROOT / 'density_spatial_grouping'
FORCING = REPO / 'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918/native_fine_external_drive'
FIELDS = ['group_rate_Hz', 'group_Z', 'group_M', 'group_K', 'group_IE',
          'group_applied_II', 'group_V', 'group_abs_current']


def main(device):
    assert read(BASE / 'comparison.json')['status'] == 'COMPLETE'
    assert read(BASE / 'comparison.json')['dynamics']['original_A4_checks']['D_track'] is False
    assert read(BASE / 'result.json')['engine_sha256_unchanged']
    reconstruction = read(FORCING / 'result.json')
    assert reconstruction['status'] == 'FORCING_RECONSTRUCTION_COMPLETE'
    assert len(reconstruction['checkpoints']) == 7
    assert all(x['spatial_state_bitwise'] and x['rng_bitwise'] for x in reconstruction['checkpoints'])
    assert sha(engine.__file__) == read(BASE / 'contract.json')['engine_sha256']
    assert not (OUT / 'contract.json').exists()
    OUT.mkdir(exist_ok=True)
    (OUT / 'operators').symlink_to(BASE / 'operators', target_is_directory=True)
    write(OUT / 'contract.json', dict(
        created_epoch=time.time(),
        question='Does recovering original within-1mm spatial external drive remove the recruitment/resource deficit at fixed fine groups?',
        evidence='Fine3479x2048 candidate stillhasarea.5705 andD9870.205705 vsnative.676-.746 and.25634; Ddifference.05064 fails original.05 gate. Furthergridrefinementstopped.',
        bounded_design='One12.5s g40x2048 run, numericalseed927611, originalG/Koff and1msinputclock. Replaceonly externaldrive with actualmembermeans oftheoriginalSpatialOUreconstruction. No newnative seed, biologicalparameterfit, furthergrid, or automaticcontinuation.',
        paired_baseline=str(BASE), forcing_source=str(FORCING),
        implementation_gate='Existingparentforcingbitwise equal; originalgroupidentity; Iforcingunchanged; restoreparentand reproduceall8saved100msprefixobservablesbitwise beforefineforcingrun.',
        decision_rule='Use unchangedA4,completeevents,area/Zand frozencontacts. Improvement alone isnot fullG/Kcorrespondence or stabilityvalidation. If substantialdeficits remain, reject forcingaveraging as a sufficient explanation; no onset/Zretuning.',
        limits='Originalstoredglobalratefloat32 and1ms holding remain. Finegroupmean is not percellforcing; recurrentGaussian independentarrivals remain an approximation. NumericalRNG isnot physicalnoise.',
        engine_sha256=sha(engine.__file__), producer_sha256=sha(__file__),
        original_forcing_producer_sha256=sha(REPO / 'scripts/topic4_zm_runaway_mechanism/native_fine_external_drive.py'),
        device=device, formal_bifurcation_allowed=False, human_review='PENDING'))
    start = time.time()
    prior = engine.OPERATORS
    try:
        engine.OPERATORS = OUT / 'operators'
        e = engine.DensityNetwork(replicas=2048, seed=927611, device=device, duration_ms=12500, gain=0.)
    finally:
        engine.OPERATORS = prior
    with np.load(FORCING / 'drive.npz') as z:
        assert np.array_equal(z['time_ms'], np.arange(12500))
        assert np.array_equal(e.drive_cpu, z['drive_g40_parent'])
        fine = z['drive_g40']
        parent = z['parent_g20']
    coarse = dict(np.load(engine.OPERATORS / 'geometry.npz'))
    assert np.array_equal(parent[e.geo['cell_group']], coarse['cell_group'])
    assert np.array_equal(e.geo['original_positions'], coarse['original_positions'])
    assert np.array_equal(fine[:, ~e.E], e.drive_cpu[:, ~e.E])
    assert np.isfinite(fine).all() and fine.min() >= 0
    e.graph()
    prefix = np.concatenate([e.chunk() for _ in range(10)])
    checks = {}
    with np.load(BASE / 'trajectory.npz') as z:
        for j, key in enumerate(FIELDS):
            checks[key] = bool(np.array_equal(prefix[:, j].astype('f4'), z[key][:100]))
    assert all(checks.values()), checks
    write(OUT / 'implementation_check.json', dict(status='PASS', parent_input_bitwise=True,
        group_membership_and_original_positions=True, inhibitory_forcing_unchanged=True,
        baseline_prefix_ms=100, baseline_prefix_bitwise=checks,
        scope='Only input replacement implementation; independent scientific correspondence remains unaccepted.'))
    e.reset()
    e.drive_cpu[:] = fine
    e.drive[:] = e.cp.asarray(fine)
    assert np.array_equal(e.drive.get(), fine)
    del fine, prefix, coarse
    output = []
    for tick in range(0, 12500, 10):
        x = e.chunk()
        assert np.isfinite(x).all() and x[:, 1].min() >= 0 and x[:, 1].max() <= 1
        output.append(x)
        if (tick + 10) % 250 == 0:
            write(OUT / 'progress.json', dict(status='RUNNING', pid=os.getpid(), time_ms=tick + 10,
                                             elapsed_s=time.time() - start))
    data = np.concatenate(output)
    cell = e.geo['group_cell']
    field = np.zeros((len(data), 400)); count = np.zeros(400)
    for g in np.flatnonzero(e.E):
        field[:, cell[g]] += data[:, 0, g] * e.sizes[g]
        count[cell[g]] += e.sizes[g]
    field /= np.maximum(count, 1)
    arrays = dict(time_ms=np.arange(12500) + 1., field_E_Hz=field.astype('f4'),
                  cell_counts=count, group_sizes=e.sizes, population_E=e.E)
    for j, key in enumerate(FIELDS):
        arrays[key] = data[:, j].astype('f4')
    np.savez_compressed(OUT / 'trajectory.npz', **arrays)
    np.savez_compressed(OUT / 'final_state.npz', state=e.state.get(), ref=e.ref.get(),
        history=e.history.get(), rng=e.rng.get(), clock=e.clock.get(), global_state=e.global_state.get(),
        accumulator=e.accumulator.get(), particle_count=2048, seed=927611)
    result = dict(status='COMPLETE', duration_ms=12500, elapsed_s=time.time() - start,
        groups=e.P, particles_per_group=e.R,
        engine_sha256_unchanged=sha(engine.__file__) == read(OUT / 'contract.json')['engine_sha256'],
        native_correspondence_certified=False, formal_bifurcation_allowed=False)
    write(OUT / 'result.json', result); write(OUT / 'progress.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--device', type=int, default=1)
    main(p.parse_args().device)
