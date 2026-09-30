"""Recover original spatial OU forcing at actual rate-group resolution.

No neural simulation or future native spikes. Keep the existing 1 ms clock
and stored global-rate precision; validate spatial OU states at checkpoints.
"""
from native_same_history_feedback import native, ROOT
from types import SimpleNamespace
from datetime import datetime
import os, time

np = native.np
OUT = ROOT / 'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918'
DEST = OUT / 'native_fine_external_drive'


def main():
    DEST.mkdir(exist_ok=True)
    assert not (DEST / 'contract.json').exists()
    native.check_reference_sources()
    frozen = native.read(ROOT / 'config/topic4_rate_model_dynamics_validation_v1.json')
    execution = native.read(next(x['path'] for x in frozen['inputs'] if x['path'].endswith('execution_config.json')))
    source = ROOT / '.worktrees/topic4-substrate-autapse-fix'
    trpath = source / execution['inputs']['transition_config']['path']
    tr = native.read(trpath)
    op = ROOT / 'results/topic4_sef_hfo/interictal_brunel_spatial_bifurcation_20260917/operators'
    geos = {g: dict(np.load(op/f'g{g}/geometry.npz')) for g in [20,40]}
    assert np.array_equal(geos[20]['original_positions'], geos[40]['original_positions'])
    positions = geos[20]['original_positions'][:native.NE]
    substrate = SimpleNamespace(positions_e=positions, engine=dict(L=20., dt=.1))
    drive = native.old.make_external_drive(substrate, tr['spatial_ou'], native.MAIN_SEED)
    native.write(DEST/'contract.json', dict(created_local=datetime.now().astimezone().isoformat(),
        question='How much original spatial external forcing was lost by lifting1mm cell means to0.5mm groups?',
        fixed='Same native SpatialOUDrive,positions,seed andconfiguration. Reconstruct spatial field from its own RNG; use stored global rate per1ms at its original float32precision. No future native firing, neuronstate transplantation, fitting or physicalparameter changes.',
        config_source=str(trpath), spatial_ou=tr['spatial_ou'], groups=[935,3479], interval_ms=[0,12500], sampling_ms=1,
        audit='Reconstructed20x20 spatialcell means agree with originalrecording within float32roundoff; spatialOU field andcachedvalues andRNG must match everyavailablecheckpoint.',
        contrasts='Original1mm spatialcell means lifted tofinegroups versus actual mean externalrate oforiginalmembers ofeach finegroup. Bothholdsampledforcingon1msintervals.',
        scope='Exogenous forcing recovery and read-only diagnosis. No newnetwork orscientificacceptance inthisscript.'))
    sizes = {g: geos[g]['group_size'] for g in geos}
    output = {g: np.empty((12500, len(sizes[g])), dtype=np.float64) for g in geos}
    coarse_cells = geos[20]['group_cell'][geos[20]['cell_group'][:native.NE]]
    coarse_count = np.bincount(coarse_cells, minlength=400)
    parent_min = np.full(len(sizes[40]), len(sizes[20]), int); parent_max = np.full(len(sizes[40]), -1, int)
    np.minimum.at(parent_min, geos[40]['cell_group'], geos[20]['cell_group'])
    np.maximum.at(parent_max, geos[40]['cell_group'], geos[20]['cell_group'])
    assert np.array_equal(parent_min, parent_max)
    parent_cell = geos[20]['group_cell'][parent_min]
    checkpoint_times = [8000,9000,9300,9420,9870,10370,12500]
    states = {tm: native.replay_checkpoint(tm)['external_drive'] for tm in checkpoint_times}
    checks=[]; maxerr=0.; rows=[]; ticks=[]; start=time.time()
    original_input = np.load(ROOT/'results/topic4_sef_hfo/fig5_zm_rate_v3_20260918/native_reference/seed9108401_external_drive.npz')
    glob = original_input['glob']; old_cells = original_input['drive_mean']
    assert len(glob) == 12500
    for tm in range(12500):
        if tm in states:
            state = states[tm]
            # The field changes only oninteger milliseconds; advance its clock
            # to the lastnative0.1msstep without drawing a newinnovation.
            drive.step(tm-.1)
            err = float(np.max(np.abs(drive._cached-state['cached'])))
            assert np.array_equal(drive._state, state['field_state']) and err == 0.
            assert drive._rng.bit_generator.state == state['rng_state']
            assert drive._next_step == state['next_step'] and drive._last_step == state['last_step']
            checks.append(dict(time_ms=tm, spatial_state_bitwise=True, rng_bitwise=True))
        delta = drive.step(float(tm))
        nu = np.full(40000, float(glob[tm])); nu[:native.NE] = np.maximum(nu[:native.NE]+delta, 0.)
        coarse = np.bincount(coarse_cells, weights=nu[:native.NE], minlength=400)/coarse_count
        error = float(np.max(np.abs(coarse-old_cells[tm])))
        # Bothstoredglobalandstoredcellmean arefloat32. Clippingis1-Lipschitz.
        bound = 4*np.finfo(np.float32).eps*max(float(nu.max()),1.)
        assert error < bound, (tm,error,bound)
        maxerr=max(maxerr,error)
        for g in geos:
            output[g][tm] = np.bincount(geos[g]['cell_group'], weights=nu, minlength=len(sizes[g]))/sizes[g]
        if (tm+1)%1000 == 0:
            print('FINE EXTERNAL',tm+1,'ms','seconds',round(time.time()-start,1),flush=True)
    drive.step(12499.9); state=states[12500]
    assert np.array_equal(drive._state,state['field_state']) and np.array_equal(drive._cached,state['cached'])
    assert drive._rng.bit_generator.state==state['rng_state'] and drive._next_step==state['next_step'] and drive._last_step==state['last_step']
    checks.append(dict(time_ms=12500,spatial_state_bitwise=True,rng_bitwise=True))
    fine=geos[40];E=fine['population']==0
    old=np.empty_like(output[40]);old[:,E]=old_cells[:,parent_cell[E]];old[:,~E]=glob[:,None]
    diff=output[40]-old
    for lo,hi in [(0,12500),(1000,9420),(8000,9420),(9420,10370)]:
        for reg,name in enumerate(['Core A','Core B','Surround']):
            mask=E&(fine['group_region']==reg);weights=sizes[40][mask]/sizes[40][mask].sum()
            x=diff[lo:hi,mask]
            rows.append(dict(window_ms=[lo,hi],region=name,cell_weighted_RMS_per_ms=float(np.sqrt((x*x)@weights).mean()),
                             pooled_RMS_per_ms=float(np.sqrt(np.mean((x*x)@weights))),max_abs_per_ms=float(np.abs(x).max())))
    np.savez_compressed(DEST/'drive.npz',time_ms=np.arange(12500),drive_g20=output[20],drive_g40=output[40],
                        drive_g40_parent=old,parent_g20=parent_min,global_rate_per_ms=glob)
    native.write(DEST/'result.json',dict(status='FORCING_RECONSTRUCTION_COMPLETE',pid=os.getpid(),seconds=time.time()-start,
        original_coarse_roundoff_max=maxerr,checkpoints=checks,rows=rows,neural_simulations=0,
        input_precision='Originalfloat32globalrate and1ms sampling retained; spatialOU exactat checkpoints. Not exactsub-msnativeforcing.',model_promoted=False))
    print('FINE EXTERNAL COMPLETE',rows,flush=True)


if __name__=='__main__':main()
