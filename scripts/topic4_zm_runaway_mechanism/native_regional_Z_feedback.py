"""Two regional Z-update interventions on the unchanged native 9 s history.

The accepted operator geometry supplies cell membership only. No projected
state, rate approximation, graph, threshold or noise stream enters this run.
"""
from native_same_history_feedback import native, ROOT
import argparse
import time

np = native.np
OUT = ROOT / 'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918'
DEST = OUT / 'native_regional_Z_feedback'
GEOMETRY = ROOT / 'results/topic4_sef_hfo/interictal_brunel_spatial_bifurcation_20260917/operators/g20/geometry.npz'


class RegionalSlow:
    def __init__(self, slow, held):
        self.slow = slow
        self.held = np.asarray(held, bool)
        assert self.held.shape == slow.z.shape
        self.original = slow.step
        # Native checkpoint restoration happens inside the simulator, after
        # this observer is attached. Capture on the first actual step.
        self.initial = None
        self.steps = 0
        slow.step = self.step

    def step(self, spk, labels, dt):
        if self.initial is None:
            self.initial = self.slow.z[self.held].copy()
        self.original(spk, labels, dt)
        self.slow.z[self.held] = self.initial
        assert np.array_equal(self.slow.z[self.held], self.initial)
        self.steps += 1


def application_check():
    old = native.core.old
    cfg = old.MZSlowVarsConfig(use_z=True, use_m=True, tau_z=5000.,
                              I_th_EI=old.THRESHOLD, tau_adp=1000., eta_m=.0005)
    rng = np.random.default_rng(919910)
    live = old.ReleaseZ(8, 18., cfg, NE=6)
    partial = old.ReleaseZ(8, 18., cfg, NE=6)
    z = rng.uniform(.5, .9, 8); z[6:] = 1.
    m = rng.uniform(0., 30., 8); m[6:] = 0.
    held = np.array([True, False, True, False, False, True, False, False])
    patch = RegionalSlow(partial, held)
    # Also checks attach-before-checkpoint-restore ordering used by the engine.
    for s in (live, partial):
        s.z[:] = z; s.m[:] = m
    for _ in range(3000):
        ie, ii = rng.uniform(0., 300., (2, 8)); spikes = rng.random(8) < .1
        live.apply_currents(ie, ii)
        current = partial.apply_currents(ie, ii)
        assert np.array_equal(current, ie - partial.z * ii - cfg.eta_m * partial.m)
        live.step(spikes, None, .1); partial.step(spikes, None, .1)
        assert np.array_equal(partial.z[~held], live.z[~held])
        assert np.array_equal(partial.m, live.m)
    assert patch.steps == 3000 and not np.array_equal(partial.z[~held], z[~held])
    q = dict(status='PASS',steps=patch.steps, held_Z_exact_every_step=True,
             unheld_Z_and_all_M_match_prescribed_input_control=True,
             held_Z_still_multiplies_inhibitory_current=True)
    native.write(DEST / 'application_check.json', q)


def main(a):
    c = native.read(OUT / 'native_regional_Z_feedback_contract.json')
    assert c['arms'] == ['cores_dynamic', 'surround_dynamic']
    assert c['start_ms'] == 9000 and c['stop_ms'] == 12500
    DEST.mkdir(parents=True, exist_ok=True); (DEST / 'jobs').mkdir(exist_ok=True)
    application_check()
    reference = native.check_reference_sources()
    assert native.read(OUT / 'native_same_history_feedback/dynamic_replay_qa.json')['status'] == 'PASS'
    geo = dict(np.load(GEOMETRY))
    region = geo['group_region'][geo['cell_group'][:native.NE]]
    assert set(np.unique(region)) == {0, 1, 2}
    held = np.zeros(native.NE + native.NI, bool)
    held[:native.NE] = region == 2 if a.arm == 'cores_dynamic' else region < 2
    assert int(held.sum()) == (30460 if a.arm == 'cores_dynamic' else 1540)
    source = native.REPLAY_RUN / 'checkpoints/t9000ms.npz'
    state = native.replay_checkpoint(9000)
    assert int(state['step']) == 90000
    protocol = dict(identity=reference['identity'], source_hashes=reference['source_hashes'],
                    contract=str(OUT / 'native_regional_Z_feedback_contract.json'),
                    geometry=str(GEOMETRY), geometry_sha256=native.sha(GEOMETRY),
                    statistical_unit='One paired native history; regional arms are not independent replicates.')
    if (DEST / 'protocol.json').exists(): assert native.read(DEST / 'protocol.json') == protocol
    else: native.write(DEST / 'protocol.json', protocol)
    name = 'native_t9000_' + a.arm
    job = native.make_job(name, 9000, 9000, 'W1', duration_ms=3500)
    job.update(start_step=90000, anchor_ms=9000, horizon_s=12.5, freeze_z=False, freeze_m=False,
               regional_Z_update=a.arm,
               state_construction='Entire original checkpoint; only named regional Z updates suppressed.')
    path = DEST / 'jobs' / (name + '.json')
    if path.exists(): assert native.read(path) == job
    else: native.write(path, job)
    folder = DEST / 'runs' / name; folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / 'checkpoint.pkl'
    if not checkpoint.exists():
        tracker = native.core.fresh_tracker(); tracker['stop_s'] = 1e9
        native.save_pickle(checkpoint, dict(job=job, identity=protocol['identity'],
                           engine=state, tracker=tracker, restore_from=None))
        assert native.compare_states(native.load_pickle(checkpoint)['engine'], state) == []
        native.write(folder / 'continuation.json', dict(source=str(source), source_sha256=native.sha(source),
                     seeded_at=time.time(), initial_state_bitwise_identical=True,
                     clock_rebased=False, random_streams_replaced=False, M_dynamic=True,
                     initial_global_Z=float(state['slow']['z'][:native.NE].mean()),
                     region_cell_counts=[int((region == k).sum()) for k in range(3)],
                     held_Z_cells=int(held.sum()), regional_Z_update=a.arm))
    else: assert native.load_pickle(checkpoint)['job'] == job
    np.savez_compressed(folder / 'regional_masks.npz', held_Z=held, E_region=region)
    patches = []
    def attach(slow, freeze_z, freeze_m):
        assert not freeze_z and not freeze_m
        p = RegionalSlow(slow, held); patches.append(p)
        return p
    native.FrozenSlow = attach
    original = native.core.old.simulate_kick
    # The ordered CUDA wrapper obtains this unchanged helper from the wrapped
    # function's globals when the producer uses its constant external drive.
    globals()['compute_nu_theta'] = original.__globals__['compute_nu_theta']
    def checked_simulator(params, net, *args, **kwargs):
        assert np.array_equal(net['pos'], geo['original_positions'])
        d = np.linalg.norm(net['pos'][:native.NE, None] - geo['centers_mm'][None], axis=2)
        native_region = np.full(native.NE, 2)
        native_region[d[:, 0] < 1.75] = 0
        native_region[(d[:, 1] < 1.75) & (d[:, 1] < d[:, 0])] = 1
        assert np.array_equal(region, native_region)
        native.write(folder / 'geometry_check.json', dict(status='PASS', native_positions_bitwise=True,
                     geometry_sha256=native.sha(GEOMETRY), membership='Accepted original cell core/region membership; not threshold reclassification.'))
        return original(params, net, *args, **kwargs)
    native.core.old.simulate_kick = checked_simulator
    native.NATIVE = DEST; native.prepare = lambda: protocol; native.seed_folder = lambda job, device: folder
    result = native.worker(name, a.device)
    assert len(patches) == 1
    final = native.load_pickle(checkpoint)['engine']
    assert np.array_equal(final['slow']['z'][held], state['slow']['z'][held])
    assert not np.array_equal(final['slow']['m'], state['slow']['m'])
    native.write(folder / 'regional_application.json', dict(status='PASS', arm=a.arm,
                 steps_this_execution=patches[0].steps, held_Z_exact_every_step=True,
                 M_dynamic=True, held_Z_still_applied=True,
                 unheld_E_Z_changed=bool(np.any(final['slow']['z'][:native.NE][~held[:native.NE]] != state['slow']['z'][:native.NE][~held[:native.NE]]))))
    print(name, result['status'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--arm', choices=['cores_dynamic', 'surround_dynamic'], required=True)
    p.add_argument('--device', type=int, default=0); main(p.parse_args())
