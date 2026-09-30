"""Exact ensemble membrane balance on two already measured local inputs.

Block reductions record the actual reset and refractory-clamp impulses at
every step without storing individual trajectories or altering their spikes.
"""
from native_cycle_waveform_response import CODE
from common import OUT, np, read, write, log
from datetime import datetime
import argparse

DEST = OUT / 'membrane_balance_diagnostic'


def register():
    path = OUT / 'membrane_balance_diagnostic_contract.json'
    assert not path.exists()
    sim = read(OUT / 'factorial_waveform_contract.json')
    write(path, dict(created_local=datetime.now().astimezone().isoformat(),
        question='Is a membrane-time-constant low-pass alone sufficient to reproduce recovery, or do reset/clamp impulses carry substantial missing voltage history?',
        source='factorial_waveform', indices=[6,7], source_group=39,
        inherited={k:sim[k] for k in ['replicates','phase_bins','dt_ms','burn_cycles','record_cycles','seed','device']},
        identity='V_next=a V_previous+(1-a)I-current_step_reset_impulse-current_step_refractory_clamp_impulse; exact native membrane arithmetic, a=exp(-dt/tau_m).',
        budget='Two recording-only repeats, exact count parity and ensemble conservation. No fitting, model promotion or network run.',
        acceptance=dict(spike_counts_bitwise=True, max_voltage_balance_error_mv=1e-8),
        statistical_unit='Independent noise path; paired conditions use identical path seeds, cycles combined within paths.',
        scope='Diagnostic identity using measured impulses and currents. The exact reconstruction is not a predictive rate closure and does not identify onset criticality.'))


def main():
    import cupy as cp
    c=read(OUT/'membrane_balance_diagnostic_contract.json');sim=c['inherited']
    data=np.load(OUT/c['source']/'prepared.npz');old=np.load(OUT/c['source']/'response.npz')
    assert read(OUT/c['source']/'independent_audit.json')['status']=='COUNT_LEVEL_AUDIT_PASS'
    indices=c['indices'];pars=np.ascontiguousarray(data['pars'][indices]);wave=np.ascontiguousarray(data['wave'][indices])
    P=len(indices);R=sim['replicates'];B=sim['phase_bins'];W=wave.shape[-1];T=float(data['T_ms']);dt=sim['dt_ms']
    steps=round(sim['record_cycles']*T/dt);burn=round(sim['burn_cycles']*T/dt);block=128
    assert R%block==0;nb=R//block
    code=CODE.replace('const double* wave,unsigned int* counts,','const double* wave,unsigned int* counts,double* records,')
    code=code.replace('double qa=0,ia=0,qg=0,ig=0,v=p[21];int ref=0;',
        'double qa=0,ia=0,qg=0,ig=0,v=p[21];int ref=0; __shared__ double buf[5][128];')
    old_line='double cur=mu+ia-ig;bool fired=false;ref=max(0,ref-1);'
    new_line='''double cur=mu+ia-ig;double oldv=v;
      double proposal=p[18]*oldv+(1-p[18])*cur;
      bool clamped=ref>1;bool fired=false;ref=max(0,ref-1);'''
    assert code.count(old_line)==1;code=code.replace(old_line,new_line)
    old_record='if(t>=0&&fired){int bin=min((int)(phase*B),B-1);counts[id*B+bin]++;}'
    new_record='''if(t>=0){
      int bin=min((int)(phase*B),B-1);if(fired)counts[id*B+bin]++;
      int lane=threadIdx.x;
      buf[0][lane]=v;buf[1][lane]=cur;
      buf[2][lane]=fired?proposal-p[21]:0.;
      buf[3][lane]=clamped?proposal-p[21]:0.;buf[4][lane]=oldv;
      __syncthreads();
      for(int offset=64;offset>0;offset>>=1){
       if(lane<offset)for(int q=0;q<5;q++)buf[q][lane]+=buf[q][lane+offset];
       __syncthreads();
      }
      if(lane==0){long long base=((long long)blockIdx.x*steps+t)*5;
       for(int q=0;q<5;q++)records[base+q]=buf[q][0];}
      __syncthreads();
    }'''
    assert code.count(old_record)==1;code=code.replace(old_record,new_record)
    cp.cuda.Device(sim['device']).use()
    counts=cp.zeros((P,R,B),dtype=cp.uint32);records=cp.empty((P,nb,steps,5),dtype=cp.float64)
    kernel=cp.RawKernel(code,'waveform',options=('--fmad=false',))
    log('MEMBRANE BALANCE RECORDING',P,R,steps,records.nbytes)
    kernel((P*nb,),(block,),(cp.asarray(pars),cp.asarray(wave),counts,records,
        np.int32(P),np.int32(R),np.int32(W),np.int32(B),np.int32(steps),np.int32(burn),float(dt),float(T),np.uint64(sim['seed'])))
    counts=counts.get();raw=records.get()
    assert np.array_equal(counts,old['counts'][indices])
    mean=raw.sum(axis=1)/R
    # Independent block summation and exact one-step balance including initial V.
    errors=[];outputs=[];rows=[]
    for j in range(P):
        a=pars[j,18];v,current,reset,clamp,previous=mean[j].T
        local=v-a*previous-(1-a)*current+reset+clamp
        assert np.max(abs(previous[1:]-v[:-1]))<1e-10
        base=np.empty(steps);reset_mem=np.empty(steps);clamp_mem=np.empty(steps)
        x=previous[0];qr=0.;qc=0.
        for k in range(steps):
            x=a*x+(1-a)*current[k];qr=a*qr-reset[k];qc=a*qc-clamp[k]
            base[k]=x;reset_mem[k]=qr;clamp_mem[k]=qc
        reconstructed=base+reset_mem+clamp_mem
        error=float(np.max(abs(reconstructed-v)))
        assert error<c['acceptance']['max_voltage_balance_error_mv']
        errors.append(error);outputs.append(np.array([base,reset_mem,clamp_mem,reconstructed]))
        # Discard one recorded cycle for component transient summaries only.
        select=np.arange(steps)*dt>=T
        peak=int(np.argmax(old['predicted_hz'][indices[j],1]-old['measured_hz'][indices[j]]))
        times=(np.arange(steps)+1)*dt
        phase=(times/T)%1;bins=np.minimum((phase*B).astype(int),B-1)
        at_peak=select&(bins==peak)
        row=dict(condition=['full','mean_only'][j], membrane_tau_ms=float(-dt/np.log(a)),
            max_step_balance_error_mv=float(np.max(abs(local))), max_full_reconstruction_error_mv=error,
            no_reset_or_clamp_RMSE_mv=float(np.sqrt(np.mean((base[select]-v[select])**2))),
            reset_history_RMS_mv=float(np.sqrt(np.mean(reset_mem[select]**2))),
            clamp_history_RMS_mv=float(np.sqrt(np.mean(clamp_mem[select]**2))),
            peak_error_phase_ms=float((peak+.5)/B*T),
            at_rate_overprediction=dict(actual_V_mv=float(v[at_peak].mean()),
                membrane_lowpass_only_mv=float(base[at_peak].mean()),
                reset_memory_mv=float(reset_mem[at_peak].mean()),
                refractory_clamp_memory_mv=float(clamp_mem[at_peak].mean())))
        rows.append(row)
    DEST.mkdir(exist_ok=True)
    np.savez_compressed(DEST/'balance.npz',counts=counts,block_sums=raw,ensemble_mean=mean,
        decomposition=outputs,dt_ms=dt,T_ms=T,pars=pars,selected_indices=indices)
    q=dict(status='EXACT_MEMBRANE_BALANCE_PASS',rows=rows,spike_counts_bitwise=True,
        max_voltage_error_mv=max(errors),scope=c['scope'],model_promoted=False,onset_type='NOT_ESTABLISHED')
    write(DEST/'result.json',q);log('MEMBRANE BALANCE',q)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--register',action='store_true');a=p.parse_args()
    register() if a.register else main()
