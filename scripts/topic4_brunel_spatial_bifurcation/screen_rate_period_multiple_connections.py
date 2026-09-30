"""Screen full-space cycles for encounters across integer period ratios.

This includes repetitions of a shorter orbit, not merely equal-period
matches. Repetition is a comparison device, never a newly found orbit or
proof of a period-multiplying bifurcation. No spatial permutation is allowed.
"""
from plot_rate_periodic_completion import *
from compare_rate_torus_periodic_targets import distances
from functools import lru_cache


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--multiples',type=int,nargs='+',default=[1,2,3,4,6,8])
    parser.add_argument('--phase-samples',type=int,default=4096)
    parser.add_argument('--output-name',default='period_multiple_connection_screen')
    args=parser.parse_args()
    assert Path(args.output_name).name==args.output_name
    assert args.phase_samples>=4096 and all(m>=1 for m in args.multiples)
    multiples=sorted(set(args.multiples))
    s=RateField();fs=families();N=args.phase_samples
    destination=Path('/data/hfosp/topic4_sef_hfo/interictal_rate_branch_completion_20260920')
    weights=s.geo['group_size']/s.geo['group_size'].sum()
    @lru_cache(maxsize=8)
    def profile(path,multiple):
        z=np.load(path);return resample(np.tile(z['r'],(multiple,1)),N,axis=0)*1000
    def amplitude(x):return float(np.sqrt(np.mean(np.sum((x-x.mean(0))**2*weights,axis=1))))
    rows=[]
    for first in ['A','B']:
        aa=fs[first]
        aj=np.array([v['J_EE_core'] for v in aa]);at=np.array([v['T_ms'] for v in aa])
        am=np.array([v['mean_rates_hz'] for v in aa]);ar=np.array([np.array(v['max_rates_hz'])-v['min_rates_hz'] for v in aa])
        for second in ['single','Bleading','double']:
            bb=fs[second]
            bj=np.array([v['J_EE_core'] for v in bb]);bt=np.array([v['T_ms'] for v in bb])
            bm=np.array([v['mean_rates_hz'] for v in bb]);br=np.array([np.array(v['max_rates_hz'])-v['min_rates_hz'] for v in bb])
            dj=abs(aj[:,None]-bj[None,:]);base=abs(np.log(at[:,None]/bt[None,:]))
            scale=np.maximum(np.maximum(np.linalg.norm(ar,axis=1)[:,None],np.linalg.norm(br,axis=1)[None,:]),.001)
            metadata=(np.linalg.norm(am[:,None,:]-bm[None,:,:],axis=2)+np.linalg.norm(ar[:,None,:]-br[None,:,:],axis=2))/scale
            for multiple in multiples:
                dt=abs(np.log(multiple*at[:,None]/bt[None,:]))
                pairs=np.argwhere((dj<=.01)&(dt<=np.log(1.10)))
                ranked=sorted((float(metadata[i,j]+10*dt[i,j]+10*dj[i,j]),int(i),int(j)) for i,j in pairs)
                selected=ranked[:20]
                # Preserve separated J neighborhoods as well as the best
                # metadata matches, recording the shortlist explicitly.
                if ranked:
                    lo,hi=min(aj[i] for _,i,j in ranked),max(aj[i] for _,i,j in ranked)
                    for left,right in zip(np.linspace(lo,hi+1e-12,6)[:-1],np.linspace(lo,hi+1e-12,6)[1:]):
                        selected += [v for v in ranked if left<=aj[v[1]]<right][:3]
                selected=list({(i,j):(score,i,j) for score,i,j in selected}.values());checks=[]
                for score,i,j in selected:
                    a,b=aa[i],bb[j];x,y=profile(a['path'],multiple),profile(b['path'],1)
                    d,phase=distances(x[:,None,:],y,weights);amp=max(amplitude(x),amplitude(y))
                    if multiple>1:
                        phase_factor=np.exp(-2j*np.pi*np.arange(N//2+1)/multiple)[:,None]
                        shifted=np.fft.irfft(np.fft.rfft(y,axis=0)*phase_factor,n=N,axis=0)
                        repeat=float(np.sqrt(np.mean(np.sum((y-shifted)**2*weights,axis=1))))
                    else:repeat=None
                    checks.append(dict(first_orbit=a['path'],second_orbit=b['path'],first_J=a['J_EE_core'],second_J=b['J_EE_core'],
                        first_T_ms=a['T_ms'],second_T_ms=b['T_ms'],first_period_repetitions=multiple,
                        metadata_score=score,common_phase_cycles=float(phase[0]),full_group_RMS_difference_Hz=float(d[0]),
                        normalized_difference=float(d[0]/amp),long_orbit_subperiod_mismatch_Hz=repeat))
                checks.sort(key=lambda q:q['normalized_difference'])
                row=dict(families=[first,second],period_multiple=multiple,window_pairs=len(pairs),compared_pairs=len(checks),
                    nearest=checks[:10],same_J_correction_candidates=[q for q in checks if q['normalized_difference']<.05])
                rows.append(row)
                write(destination/(args.output_name+'_progress.json'),dict(status='RUNNING',pid=os.getpid(),rows=rows))
                print(first,second,multiple,len(pairs),len(checks),checks[0]['normalized_difference'] if checks else None,flush=True)
    p=aa[0]['path'];control=profile(p,2);d,_=distances(np.roll(control,47,axis=0)[:,None,:],control,weights)
    assert d[0]<1e-7,d
    result=dict(status='SCREEN_COMPLETE',rows=rows,phase_samples=N,identity_phase_control_Hz=float(d[0]),
        parameter_window=.01,relative_period_window=1.10,multiples=multiples,
        observable='All 935 population waveforms, neuron-count weighting, retained means, one common temporal phase, and no exchange of core identities.',
        subperiod_diagnostic='Exact Fourier phase shift of T/m; no rounding to an integer sampling lag.',
        scope='Finite metadata shortlist over already traced branches, using repeated shorter cycles. Candidates require same-J correction and physical waveform checks. A negative screen does not exclude unsampled global connections, homoclinic limits, or period ratios outside this screen.')
    write(destination/(args.output_name+'.json'),result)
    write(destination/(args.output_name+'_progress.json'),dict(status='COMPLETE',pid=os.getpid(),source=str(destination/(args.output_name+'.json'))))


if __name__=='__main__':main()
