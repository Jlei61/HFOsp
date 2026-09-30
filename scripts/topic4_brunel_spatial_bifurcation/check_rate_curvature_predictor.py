"""Check a faster predictor against an already accepted full-space orbit."""
from rate_periodic_continue import *
from audit_rate_filter_states import filter_state_minima
import subprocess,os


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0);a=p.parse_args()
    folder=Path('/data/hfosp/topic4_sef_hfo/interictal_rate_branch_completion_20260920/predictor_check')
    folder.mkdir(exist_ok=True);worker=folder/'worker.json'
    def status(state,**kw):
        write(worker,dict(status=state,pid=os.getpid(),timestamp=time.time(),**kw))
        print(state,kw,flush=True)
    while True:
        free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),
            '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
        if free>4.5*1024:break
        status('WAITING_GPU_RESOURCE',free_mib=free);time.sleep(30)
    label='arcBleadingConnection_20260920'
    paths=[PERIODIC_OUT/'orbits'/f'{label}_{i:04d}_N4096.npz' for i in range(4)]
    accepted=read(PERIODIC_OUT/f'{label}_accuracy.json')
    assert all(str(f) in accepted['included_orbits'] for f in paths)
    assert accepted['status']=='SAMPLED_PASS'
    plan=read(PERIODIC_OUT/f'{label}_continuation.json')['rows']
    target=paths[-1];row=next(q for q in plan if Path(q['path']).resolve()==target.resolve())
    N=4096;s=RateField();o=Periodic(s,N,a.device)
    o.low_memory=True;o.host_krylov=True;o.stream_harmonics=True
    o.harmonic_chunk_size=64;o.derivative_chunk_size=64
    o.normalize_linear_rhs=True;o.linear_target_aware=True
    weight=np.r_[np.full(N*s.P,1/np.sqrt(N*s.P)),50.,1.]
    previous,x0,x1=[encode(np.load(f),N) for f in paths[:3]]
    guess,linear,tangent,screen=screened_curvature_guess(o,previous,x0,x1,row['ds'],weight)
    assert screen['method']=='CURVATURE_INITIAL_GUESS'
    assert abs(screen['arc_plane_displacement'])<1e-9
    reference=linear[:-2].reshape(N,s.P)/1000
    status('SOLVING_SAME_CONTINUATION_PLANES',screen=screen,target=str(target))
    r,T,J,err,history=o.solve(guess[:-2].reshape(N,s.P)/1000,np.exp(guess[-2]),guess[-1]/1000,
        arc=(linear,tangent,weight),phase_reference=reference,maxiter=12,tol=2e-11)
    actual=save_orbit(s,r,T,J,err,history,'curvature_predictor_Bleading_known_point3_N4096')
    expected=np.load(target)
    change=float(np.linalg.norm(r-expected['r'])/np.linalg.norm(expected['r']-expected['r'].mean(0)))
    physical=filter_state_minima(s,r,T)
    passed=(err<2e-11 and abs(J-float(expected['J']))<1e-10 and abs(T-float(expected['T']))<1e-8
            and change<1e-8 and physical['positive'] and len(history)<=len(expected['history']))
    result=dict(status='PASS' if passed else 'REVIEW_REQUIRED',target=str(target),corrected_orbit=str(actual),
        J_difference=J-float(expected['J']),T_difference_ms=T-float(expected['T']),
        relative_full_waveform_difference=change,initial_residual_screen=screen,
        newton_residual_hz=err,newton_iterations=len(history)-1,
        original_newton_iterations=len(expected['history'])-1,filter_state_check=physical,
        scope='Same full 935-population periodic equations, physical delays, temporal mesh, arc plane, phase plane and tolerance. Only the numerical initial guess changes. This does not modify or restart any live continuation batch.')
    write(PERIODIC_OUT/'curvature_predictor_same_orbit_check.json',result)
    status('FINISHED',result=result)
    assert passed


if __name__=='__main__':main()
