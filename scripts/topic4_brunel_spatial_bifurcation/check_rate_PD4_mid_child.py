"""Resolve one intermediate PD4 child where dominant-multiplier signs differ."""
from complete_rate_positive_stability import *
import subprocess


LABEL='PD4_child_midpoint_20260920'
FOLDER=DEST/'H2_local_PD/midpoint_spectrum'


def main():
    global LABEL, FOLDER
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--min-free-gib',type=float,default=8.)
    parser.add_argument('--amplitude',type=float,default=.03)
    parser.add_argument('--label',default=LABEL)
    parser.add_argument('--output-name',default='midpoint_spectrum')
    parser.add_argument('--after-pids',type=int,nargs='*',default=[])
    parser.add_argument('--trace-center-seed',action='store_true',
        help='Choose one amplitude from completed paired midpoint/upper spectra; interpolation is an initial-guess rule, not a crossing verdict')
    args=parser.parse_args();assert args.min_free_gib>=8
    assert Path(args.label).name==args.label and Path(args.output_name).name==args.output_name
    LABEL=args.label;FOLDER=DEST/'H2_local_PD'/args.output_name
    FOLDER.mkdir(parents=True,exist_ok=True)
    def status(stage,**kw):
        write(FOLDER/'worker.json',dict(status=stage,pid=os.getpid(),label=LABEL,**kw))
        print(stage,kw,flush=True)
    dependencies={pid:Path(f'/proc/{pid}/cmdline').read_bytes() for pid in args.after_pids
                  if Path(f'/proc/{pid}/cmdline').exists()}
    while dependencies:
        for pid,identity in list(dependencies.items()):
            file=Path(f'/proc/{pid}/cmdline')
            try:active=file.read_bytes()==identity
            except FileNotFoundError:active=False
            if not active:dependencies.pop(pid)
        if dependencies:
            status('WAITING_DEPENDENCIES',dependencies=list(dependencies));time.sleep(30)
    def gate(mesh=0,stage='NEXT_STAGE'):
        release(args.device)
        while True:
            free=float(subprocess.check_output(['nvidia-smi','-i',str(args.device),
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
            if free>=args.min_free_gib*1024:return
            status('WAITING_GPU_RESOURCE',free_mib=free,required_free_gib=args.min_free_gib,
                N=mesh,next_stage=stage);time.sleep(30)
    parent=read(PERIODIC_OUT/'PD_H2_after_LPC13_validation.json')
    assert parent['full_acceptance'] and parent['status']=='VALIDATED_PD'
    endpoints=read(DEST/'H2_local_PD/physical_children.json')['rows'][:2]
    assert [q['amplitude_hz'] for q in endpoints]==[.02,.04]
    assert all(q['physical_check']['filter_state_check']['positive'] for q in endpoints)
    amplitude=args.amplitude;seed_orbit=endpoints[0]['orbit']
    if args.trace_center_seed:
        assert args.output_name!='midpoint_spectrum' and LABEL!='PD4_child_midpoint_20260920'
        midpoint_folder=DEST/'H2_local_PD/midpoint_spectrum'
        midpoint=read(midpoint_folder/'result.json')
        physical=read(midpoint_folder/'physical_child.json')
        assert physical['physical_check']['filter_state_check']['positive']
        assert physical['physical_check']['maximum_group_defect_Hz']<1e-6
        sources=[midpoint['sources'],[str(PERIODIC_OUT/'poincare_floquet'/
            f'PD4_child_index1_full_k6_20260920_dt{dt}.json') for dt in ['0.05','0.025']]]
        invariants=[]
        for pair_sources,orbit,a in zip(sources,[midpoint['orbit'],endpoints[1]['orbit']],
                [physical['amplitude_hz'],endpoints[1]['amplitude_hz']]):
            pair=[read(file) for file in pair_sources];verdict=paired_modes(*pair)
            assert all(Path(q['orbit']).resolve()==Path(orbit).resolve() for q in pair)
            assert verdict['status']=='UNSTABLE' and verdict['reliable_outside_count']==4
            mu=values(pair[-1]);growing=np.flatnonzero(verdict['outside_unit_disk_mask'])
            real=growing[abs(mu[growing].imag)<1e-8]
            assert len(real)==2
            assert max(np.asarray(pair[-1]['phase_overlap'])[real])<1e-6
            invariants.append(dict(amplitude_Hz=a,trace=float(mu[real].real.sum()),
                determinant=float(mu[real].real.prod()),real_multipliers=mu[real].real,
                paired_classification=verdict,sources=pair_sources))
        left,right=invariants
        assert left['trace']>0>right['trace'] and left['determinant']>0 and right['determinant']>0
        fraction=left['trace']/(left['trace']-right['trace'])
        amplitude=float(np.sqrt((1-fraction)*left['amplitude_Hz']**2+
                                fraction*right['amplitude_Hz']**2))
        assert left['amplitude_Hz']<amplitude<right['amplitude_Hz']
        seed_orbit=midpoint['orbit']
        write(FOLDER/'amplitude_selection.json',dict(status='INITIAL_GUESS_SELECTED',
            requested_amplitude_Hz=amplitude,endpoint_spectral_invariants=invariants,
            rule='Interpolate the real-pair trace linearly in squared child amplitude and select its zero as one interior sample.',
            scope='Adaptive sample location only. A new physical BVP and paired full-history spectrum must test whether this pair becomes complex. No collision, unit-circle crossing, extra PD, or interval completeness is inferred from interpolation.'))
    assert .02<amplitude<.04
    branch=PERIODIC_OUT/f'{LABEL}_branch_N1024.json'
    if not branch.exists():
        gate(stage='CHILD_BVP')
        log=FOLDER/f'child_BVP_{time.time_ns()}.log'
        command=[sys.executable,'-u',str(Path(__file__).parent/'rate_period_doubled_branch.py'),
            str(PERIODIC_OUT/'PD_H2_after_LPC13_N512.json'),parent['accepted_mode'],
            '--N','512','--amplitudes',repr(amplitude),'--from-orbit',seed_orbit,
            '--device',str(args.device),'--label',LABEL,'--linear-normalize',
            '--stream-harmonics','--host-krylov','--tol','2e-11']
        with log.open('w') as output:
            child=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT)
            status('CHILD_BVP',child_pid=child.pid,log=str(log))
            code=child.wait()
        if code:raise RuntimeError(f'Child solve failed: {log}')
    rows=read(branch);assert len(rows)==1 and abs(rows[0]['amplitude_hz']-amplitude)<1e-14
    row=rows[0];source=Path(row['orbit'])
    assert endpoints[0]['J_EE_core']<row['J_EE_core']<endpoints[1]['J_EE_core']
    coefficients=np.array([q['J_shift_over_amplitude_squared'] for q in [*endpoints,row]])
    assert np.ptp(coefficients)/abs(np.mean(coefficients))<.1
    actual,check=prepare(source,args.device,max_N=4096,host_krylov=True,
        check_filter_states=True,harmonic_chunk_size=64,stream_harmonics=True,
        adaptive_memory=True,before_mesh=gate)
    if check['status']!='RESOLUTION_CHECKED' or check['maximum_group_defect_Hz']>=1e-6:
        status('PHYSICAL_CHILD_UNRESOLVED',resolution=check);return
    old,new=np.load(source),np.load(actual);model=RateField()
    weights=model.geo['group_size']/model.geo['group_size'].sum()
    mesh=max(len(old['r']),len(new['r']))
    before,after=[resample(z['r']*1000,mesh,axis=0) for z in [old,new]]
    distance,_=distances(before[:,None,:],after,weights)
    scale=np.sqrt(np.mean(np.sum((before-before.mean(0))**2*weights,axis=1)))
    assert float(distance[0]/scale)<.02 and abs(float(new['J']-old['J']))<1e-12
    assert abs(float(new['T']/old['T'])-1)<.001
    write(FOLDER/'physical_child.json',dict(**row,analyzed_orbit=str(actual),physical_check=check,
        relative_waveform_refinement_change=float(distance[0]/scale),
        coefficient_relative_spread=float(np.ptp(coefficients)/abs(np.mean(coefficients)))))
    pair=[];sources=[]
    for dt in [.05,.025]:
        gate(stage='FULL_HISTORY_SPECTRUM')
        target=PERIODIC_OUT/'poincare_floquet'/f'{LABEL}_k6_dt{dt:g}.json'
        status('FULL_HISTORY_SPECTRUM',dt_ms=dt,orbit=str(actual))
        q=read(target) if target.exists() else compute_return(actual,dt,6,args.device,16,
            stream_harmonics=True,output_label=LABEL+'_k6')
        assert Path(q['orbit']).resolve()==actual.resolve()
        pair.append(q);sources.append(str(target))
    verdict=paired_modes(*pair)
    result=dict(status=verdict['status'],source=str(FOLDER/'physical_child.json'),
        orbit=str(actual),sources=sources,classification=verdict,
        selected_amplitude_Hz=amplitude,
        amplitude_selection_source=str(FOLDER/'amplitude_selection.json') if args.trace_center_seed else None,
        scope='One intermediate physical doubled cycle between amplitudes .02 and .04. Tests the spectral sign-change interval; does not certify the entire interval or locate a new bifurcation.')
    write(FOLDER/'result.json',result);release(args.device)
    status('MIDPOINT_SPECTRUM_FINISHED',scientific_status=verdict['status'])


if __name__=='__main__':
    try:main()
    except Exception as exc:
        FOLDER.mkdir(parents=True,exist_ok=True);path=FOLDER/'worker.json'
        old=read(path) if path.exists() else {}
        write(path,{**old,'status':'COMPUTATION_FAILED','error':repr(exc)})
        raise
