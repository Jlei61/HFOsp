"""GPU phase calculation + frozen native-ring diagnostics, historical and new runs.

Reported objects are rotational candidates, never confirmed spiral rotors.
Two GPU consumers partition immutable jobs; each job writes an atomic result.
"""
from pathlib import Path
import argparse,importlib.util,json,time,sys,hashlib
import numpy as np
from scipy.signal import butter,sosfiltfilt
from scipy.ndimage import gaussian_filter,gaussian_filter1d
ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/data/hfosp/topic4_sef_hfo/core_shape_output_response_20260911')
OLD=Path('/data/hfosp/topic4_sef_hfo/core_connectivity_search_20260910')
ORIGINAL=Path('/data/hfosp/topic4_sef_hfo/core_propagation_recovery_20260911/spiral_wave_detection/analyze.py')
spec=importlib.util.spec_from_file_location('original_rotation',ORIGINAL);orig=importlib.util.module_from_spec(spec);spec.loader.exec_module(orig)


def phase_gpu(movie,band,spatial,cp):
    # Keep the original CPU Gaussian and causal filter pair bit-for-bit; transfer
    # full contiguous time-by-space batches for Hilbert FFT and phase winding.
    from cupyx.scipy.signal import hilbert
    smooth=gaussian_filter(movie.astype(np.float32),(0,spatial,spatial),mode='nearest') if spatial else movie.astype(np.float32)
    filtered=sosfiltfilt(butter(3,band,btype='bandpass',fs=500,output='sos'),smooth,axis=0)
    analytic=hilbert(cp.asarray(np.ascontiguousarray(filtered)),axis=0)
    phase=cp.angle(analytic).astype(cp.float32);amp=cp.abs(analytic).astype(cp.float32)
    def wrap(x):return cp.angle(cp.exp(1j*x))
    p0=phase[:,:-1,:-1];p1=phase[:,:-1,1:];p2=phase[:,1:,1:];p3=phase[:,1:,:-1]
    q=cp.rint((wrap(p1-p0)+wrap(p2-p1)+wrap(p3-p2)+wrap(p0-p3))/(2*np.pi)).astype(cp.int8)
    return cp.asnumpy(phase),cp.asnumpy(amp),cp.asnumpy(q)


def analyze(path,gpu,cp):
    started=time.time();meta=json.loads(path.read_text());dest=OUT/'rotation'/hashlib.sha256(str(path).encode()).hexdigest()[:20]
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'result.json').exists():return
    with np.load(path.with_suffix('.npz')) as z:raw=z['sheet_activity_counts']
    end=min(float(meta['actual_duration_ms'])-500.,len(raw)*2-500.)
    if end<=1500:return
    first,last=750,int(end/2);orig.LO=1500.;orig.DT=2.
    phase,amp,q=phase_gpu(raw,(3,50),.5,cp)
    parity=OUT/'rotation'/f'gpu{gpu}_parity.json'
    if not parity.exists():
        cpu_ph,cpu_amp=orig.phase_maps(raw,(3,50),.5);cpu_q=orig.winding(cpu_ph)
        good=np.array_equal(cpu_q,q);phaseerr=float(np.max(abs(orig.wrap(phase-cpu_ph))))
        parity.write_text(json.dumps(dict(source=str(path),gpu=gpu,winding_identical=good,phase_max_error=phaseerr,amplitude_max_error=float(abs(amp-cpu_amp).max()))))
        if not good or phaseerr>1e-4:raise RuntimeError('GPU phase parity failed')
    tracks=orig.tracks_from_winding(q[first:last],amp[first:last]);long=[t for t in tracks if t['duration_ms']>=40]
    smooth=gaussian_filter1d(raw[first:last].astype(np.float32),1.5,axis=0)
    _,_,q_sensitivity=phase_gpu(raw,(3,30),.5,cp);_,_,q_unsmoothed=phase_gpu(raw,(3,50),0,cp)
    for tr in long:
        tr['rings']=[orig.ring_check(smooth,tr,r,False) for r in [2.,3.,4.]]
        best=max((v for v in tr['rings'] if v.get('available')),key=lambda v:v.get('largest_abs_turns',0),default=None)
        tr['best_fixed_ring']=best
        tr['best_fixed_turns']=0. if best is None else best['largest_abs_turns']
        arc=None if best is None else best['best_arc']
        tr['half_turn_candidate']=bool(arc and tr['amplitude_supported_fraction']>=.8 and abs(arc['net_turns'])>=.5 and arc['duration_ms']>=40 and arc['direction_consistency']>=.8)
        for label,other in [('3_30Hz',q_sensitivity),('no_spatial_smoothing',q_unsmoothed)]:
            match=[]
            for t,x,y,_ in tr['points']:
                yy,xx=np.where(other[first+int(t)]==tr['q']);match.append(bool(np.any((xx+1-x)**2+(yy+1-y)**2<=4)))
            tr['match_fraction_'+label]=float(np.mean(match))
    selected=[t for t in long if t['half_turn_candidate']];occupied=np.zeros(last-first,bool)
    for t in selected:occupied[np.array(t['points'])[:,0].astype(int)]=True
    events=[]
    for e in meta.get('events',[]):
        lo,hi=e['window_ms']
        if lo<1500 or hi>end:continue
        overlap=[t for t in selected if t['start_ms']<hi and t['end_ms']>=lo]
        events.append(dict(event=e['event_index'],mode=e['mode'],primary=e['primary_eligible'],rotation_candidates=len(overlap)))
    result=dict(status='COMPLETE',source=str(path),source_arrays_sha256=meta['arrays_sha256'],gpu=gpu,
        duration_analyzed_ms=end-1500,window_ms=[1500,end],candidate=meta.get('candidate_id',meta.get('job',{}).get('candidate')),
        job=meta.get('job',{}),phase_tracks_40ms=len(long),half_turn_candidates=len(selected),
        candidate_tracks_per_minute=len(selected)/((end-1500)/60000),candidate_time_fraction=float(occupied.mean()),
        maximum_fixed_ring_turns=max((t['best_fixed_turns'] for t in long),default=0),
        tracks=long,events=events,wall_seconds=time.time()-started,
        interpretation='Half-turn candidates are operational descriptors. Tracks may split; not independent spiral episodes or sustained rotors. Superposed nonrotating waves can pass screening.',
        contract='3-50Hz order3 Butterworth+Hilbert; >=40ms consecutive winding track; fixed radii2/3/4mm native3ms-smoothed ring; amplitude support>=0.8, angular concentration>=.25, mean activity>=1, net>=.5turn, duration>=40ms, direction consistency>=.8; diagnostic only',
        original_detector=str(ORIGINAL),original_detector_sha256=hashlib.sha256(ORIGINAL.read_bytes()).hexdigest())
    temp=dest/'result.tmp.json';temp.write_text(json.dumps(result,ensure_ascii=False));temp.replace(dest/'result.json')
    print(json.dumps({k:result[k] for k in ['candidate','gpu','phase_tracks_40ms','half_turn_candidates','maximum_fixed_ring_turns','wall_seconds']}),flush=True)


def main(gpu):
    import cupy as cp
    cp.cuda.Device(gpu).use();(OUT/'rotation').mkdir(parents=True,exist_ok=True)
    while True:
        paths=list(OLD.glob('*/units/*/*/workers/trajectory.json'))+list(OUT.glob('*/units/*/*/workers/trajectory.json'))
        paths.sort(key=lambda p:(not str(p).startswith(str(OUT)),str(p)))
        todo=[]
        for path in paths:
            key=hashlib.sha256(str(path).encode()).hexdigest()
            if int(key,16)%2!=gpu or (OUT/'rotation'/key[:20]/'result.json').exists():continue
            z=json.loads(path.read_text())
            if z.get('status')=='COMPLETE' and z.get('actual_duration_ms',0)>=20000:todo.append(path)
        if todo:
            analyze(todo[0],gpu,cp);cp.get_default_memory_pool().free_all_blocks();continue
        (OUT/'rotation'/f'gpu{gpu}_status.json').write_text(json.dumps(dict(status='WAITING_FOR_TRAJECTORIES',updated_unix=time.time())))
        status=OUT/'status.json'
        if status.exists() and json.loads(status.read_text()).get('status')=='ROUND_COMPLETE_PENDING_SCIENTIFIC_REVIEW':break
        time.sleep(15)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True,choices=[0,1]);a=p.parse_args();main(a.gpu)
