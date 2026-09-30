"""Bounded local validation of a deterministic population-density response."""
from conditional_current_density import simulate, voltage_grid
from common import OUT, read, write, np, log
from datetime import datetime
import argparse,time

DEST=OUT/'conditional_current_density'


def register():
    path=OUT/'conditional_current_density_contract.json';assert not path.exists()
    write(path,dict(created_local=datetime.now().astimezone().isoformat(),
        question='Can a deterministic population response retaining reset/refractory and conditional synaptic-current memory autonomously predict the strong waveforms missed by old rate filters?',
        model='Voltage density with mass plus4conditional current means and10second moments per voltage node; a Gaussian conditional-current closure. Gaussian transition truncated at threshold; linear positive remapping below threshold. Refractory cohorts preserve current moments and native0.1ms update order. Rate is computed threshold flux.',
        approximations=['Conditional Gaussian current law at each voltage node', 'Finite voltage grid and linear remapping', 'Input diffusion law already used by original local response assay'],
        preserved=['Native AMPA/GABA rise/decay matrices and exact innovation covariance', 'Native membrane exponential step, reset and refractory ordering', 'No resetting or whitening synaptic current on a spike', 'Original thresholds and already fixed local waveform inputs'],
        scope='Local candidate only, not a particle network or accepted spatial model. No measured future spikes supplied. If viable, it must be compressed to an analyzable response and pass original DC/linear/network gates before spatial use. Old static transfer is a validation target, not silently inherited.',
        statistical_unit='A deterministic predicted population response compared with original8192independent noise paths; all targets already seen, not a new blind validation.',
        source='factorial_waveform',full_indices=[0,3,6,9],allowed_grid_nodes=[128,256,512,1024],
        simulation=dict(dt_ms=.1,burn_cycles=5,record_cycles=20,phase_bins=128,voltage_minimum_mv=-500.),
        acceptance=dict(normalized_waveform_RMSE_max=.15,relative_cycle_mean_error_max=.1,
            grid_relative_waveform_difference_max=.02,mass_error_max=1e-8,
            global_current_moment_relative_error_max=1e-7,discarded_probability_max=1e-10,lower_edge_probability_sum_max=1e-8),
        budget='Four full waveforms on128and256nodes. At most two further grid levels per full waveform if refinement is needed to separate closure error from grid error. Only if all four full waveforms pass, extend to the8factorialcounterfactuals and12moderatein-domain targets. No coefficients fitted or network/bifurcation launched in this diagnostic.',
        continuation='A local pass is insufficient: assess practical state dimension and differentiability; independently verify static/linear response and new held-out waveforms before any full spatial rate replacement.'))


def run(a):
    c=read(OUT/'conditional_current_density_contract.json')
    assert read(OUT/'conditional_current_density_implementation_check.json')['status']=='CONDITIONAL_CURRENT_DENSITY_IMPLEMENTATION_PASS'
    assert a.grid in c['allowed_grid_nodes']
    dataset=a.dataset
    assert dataset in ['factorial_waveform','in_domain_waveform']
    if dataset!='factorial_waveform' or a.index not in c['full_indices']:
        assert read(DEST/'independent_audit.json')['full_waveform_gate_pass']
        assert a.grid==256 and 0<=a.index<12
    z=np.load(OUT/dataset/'prepared.npz');obs=np.load(OUT/dataset/'response.npz')
    info=read(OUT/dataset/'preparation.json')['rows'][a.index]
    original=read(OUT/dataset/'result.json')['rows'][a.index]
    sim=c['simulation'];p=z['pars'][a.index];T=float(z['T_ms'])
    grid=voltage_grid(p[1],p[21],a.grid,sim['voltage_minimum_mv'])
    burn=round(T*sim['burn_cycles']/sim['dt_ms']);steps=round(T*sim['record_cycles']/sim['dt_ms'])
    log('DENSITY ASSAY START',info,a.grid,len(grid),burn,steps)
    start=time.monotonic();answer=simulate(p,z['wave'][a.index],grid,sim['dt_ms'],T,burn,steps,sim['phase_bins'])
    elapsed=time.monotonic()-start
    prediction,voltage,ref,steps_rate,free,refractory,numeric=answer
    target=obs['measured_hz'][a.index];exposure=obs['occupancy_ms']
    error=float(np.linalg.norm(prediction-target)/max(np.linalg.norm(target),np.sqrt(len(target))))
    mean=float(np.average(prediction,weights=exposure));bias=abs(mean-original['MC_mean_hz'])/max(original['MC_mean_hz'],1.)
    gate=c['acceptance']
    numeric_pass=bool(numeric[0]<gate['mass_error_max'] and numeric[1]<gate['global_current_moment_relative_error_max'] and
                      numeric[2]<gate['discarded_probability_max'] and numeric[3]<gate['lower_edge_probability_sum_max'])
    q=dict(status='DENSITY_LOCAL_ASSAY_COMPLETE',dataset=dataset,source_index=a.index,source_info=info,requested_grid_nodes=a.grid,
        actual_grid_nodes=len(grid),elapsed_seconds=elapsed,dt_ms=sim['dt_ms'],T_ms=T,
        waveform_L2=error,relative_mean_error=bias,predicted_mean_hz=mean,reference_mean_hz=original['MC_mean_hz'],
        numerical=dict(zip(['max_mass_error','max_current_moment_error','discarded_probability','lower_edge_probability_sum','worst_relative_negative_covariance'],numeric.tolist())),
        numerical_pass=numeric_pass,waveform_pass=bool(error<=.15 and bias<=.1),grid_convergence='PENDING_PAIRED_LEVEL',
        model_promoted=False,scope=c['scope'])
    DEST.mkdir(exist_ok=True);prefix=DEST/f'{"moderate_" if dataset=="in_domain_waveform" else ""}index{a.index:02d}_grid{a.grid}'
    assert not prefix.with_suffix('.json').exists(), 'Preserve completed level'
    np.savez_compressed(prefix.with_suffix('.npz'),predicted_hz=prediction,measured_hz=target,mean_voltage=voltage,
        refractory_fraction=ref,step_rate_hz=steps_rate,final_free=free,final_refractory=refractory,
        grid_mv=grid,pars=p,T_ms=T,dt_ms=sim['dt_ms'],phase_centres=obs['phase_centres'])
    write(prefix.with_suffix('.json'),q);log('DENSITY ASSAY RESULT',q)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--register',action='store_true')
    parser.add_argument('--dataset',default='factorial_waveform')
    parser.add_argument('--grid',type=int,default=128);parser.add_argument('--index',type=int,default=6)
    a=parser.parse_args();register() if a.register else run(a)
