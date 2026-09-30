"""Full-state/history finite-separation Lyapunov estimate on an autonomous run.

Renormalize an infinitesimal second trajectory after identical CUDA blocks.
No phase reset, history replacement, native-SNN input, or noise is used.
This is a finite-time diagnostic, not a bifurcation or chaos certificate.
"""
from run_rate_field_long import *
from scipy.interpolate import CubicSpline


def flow_direction(g):
    """Full physical flow at a graph-block boundary (delay ring tick zero)."""
    e=g.engine;cp=e.cp;e.arrivals(0)
    e.k['rhs'](((e.s.P+127)//128,),(128,),(e.y,e.arr,e.pars,e.f))
    fy=e.f.copy();h=e.history
    fh=(cp.roll(h,-1,axis=0)-cp.roll(h,1,axis=0))/(2*e.dt)
    fh[1]=(-3*h[1]+4*h[2]-h[3])/(2*e.dt)
    fh[0]=cp.asarray(e.s.alpha)*fy[0]+(1-cp.asarray(e.s.alpha))*fy[1]
    return fy,fh


def initial_from_saved(s,path,dt):
    z=np.load(path);y=z['final_state'];h=z['final_history'];old=float(z['dt_ms'])
    tick=int(z['final_history_tick_modulo_depth']);assert tick==0
    times=-np.arange(len(h)-1,-1,-1)*old
    chronological=h[(-np.arange(len(h)-1,-1,-1))%len(h)]
    depth=s.prep['max_delay_steps']*round(.1/dt)+1
    newtimes=-np.arange(depth-1,-1,-1)*dt
    hh=np.empty((depth,s.P));hh[(-np.arange(depth-1,-1,-1))%depth]=CubicSpline(times,chronological,axis=0)(newtimes)
    hh[0]=s.output(y)
    return y,hh


def main(a):
    s=RateField();check=RATE_OUT/'periodic_completion'/f'long_graph_check_dt{a.dt:g}.json'
    assert check.exists() and read(check)['status']=='PASS'
    if a.transverse:
        assert read(RATE_OUT/'periodic_completion/finite_lyapunov/flow_projection_checks.json')['status']=='PASS'
    y,h=initial_from_saved(s,a.trajectory,a.dt)
    one=GraphIntegrator(s,a.J,a.dt,a.device,y,h);two=GraphIntegrator(s,a.J,a.dt,a.device,y,h)
    cp=one.cp;weights=cp.asarray([1000.,1000.,1.,1.,1.,1.,.1,.1,1.])[:,None]
    e,f=one.engine,two.engine;dim=e.y.size+e.history.size
    rng=np.random.default_rng(76019);dy=cp.asarray(rng.normal(size=y.shape))/weights
    # A continuous initial rate history, with exact current endpoint consistency.
    alpha=cp.asarray(s.alpha);dh=cp.broadcast_to(alpha*dy[0]+(1-alpha)*dy[1],h.shape).copy()
    def norm(dy,dh):return float(cp.sqrt((cp.sum((dy*weights)**2)+cp.sum((dh*1000)**2))/dim))
    def set_perturbation(dy,dh):
        if a.transverse:
            fy,fh=flow_direction(one)
            coef=(cp.sum(dy*fy*weights**2)+cp.sum(dh*fh)*1e6)/(cp.sum((fy*weights)**2)+cp.sum(fh**2)*1e6)
            dy=dy-coef*fy;dh=dh-coef*fh
        size=norm(dy,dh);assert np.isfinite(size) and size>0
        f.y[:]=e.y+dy*(a.epsilon/size);f.history[:]=e.history+dh*(a.epsilon/size)
        cp.cuda.get_current_stream().synchronize()
        return size
    set_perturbation(dy,dh);rows=[];start=time.time();blocks=int(np.ceil(a.duration/one.samples))
    dest=RATE_OUT/'periodic_completion/finite_lyapunov';dest.mkdir(exist_ok=True)
    tag=f'J{a.J:.7f}_dt{a.dt:g}_eps{a.epsilon:g}'
    if a.transverse:tag+='_transverse'
    if (dest/(tag+'.json')).exists():raise RuntimeError('Completed diagnostic exists')
    source=Path(a.trajectory).parent/'contract.json';assert read(source)['J_EE_core']==a.J
    contract=dict(J_EE_core=a.J,dt_ms=a.dt,epsilon=a.epsilon,source=str(a.trajectory),
        initial_history_interpolation='Cubic on saved physical-time history; endpoint fixed to current rate',
        norm='RMS over all nine local states and every delayed rate. Local scaling [1000,1000,1,1,1,1,.1,.1,1], delayed rate scaling 1000.',
        block_ms=one.samples,block_steps=one.steps,requested_duration_ms=a.duration,
        scope='Finite-time largest Lyapunov estimate. Compare perturbation amplitudes, integration steps and periodic control before an attractor claim.')
    contract['phase_projection']='Weighted full-flow orthogonal projection after every block; centered history derivative and exact current RHS' if a.transverse else 'None'
    write(dest/(tag+'_contract.json'),contract)
    for k in range(blocks):
        reference=one.block();two.block()
        dy=f.y-e.y;dh=f.history-e.history;size=set_perturbation(dy,dh)
        rows.append(dict(time_ms=one.elapsed_ms,log_growth=np.log(size/a.epsilon),
            final_reference_rates_hz=s.regional_rates(s.output(e.y.get()))))
        if k%5==0 or k+1==blocks:
            write(dest/(tag+'_status.json'),dict(status='RUNNING',pid=os.getpid(),blocks_complete=k+1,blocks_total=blocks,
                elapsed_seconds=time.time()-start,rows=rows))
            print('FINITE LYAPUNOV',tag,k+1,'/',blocks,'lambda per s',sum(q['log_growth'] for q in rows)/one.elapsed_ms*1000,flush=True)
    estimates=[]
    for discard in [0,2000,5000,10000]:
        rr=[q for q in rows if q['time_ms']-one.samples>=discard]
        if rr:estimates.append(dict(discard_ms=discard,duration_ms=len(rr)*one.samples,
            exponent_per_second=sum(q['log_growth'] for q in rr)/(len(rr)*one.samples)*1000))
    result=dict(status='COMPLETE',contract=contract,rows=rows,estimates=estimates,
        elapsed_seconds=time.time()-start,attractor_class='NOT_ESTABLISHED')
    checkpoint=dest/tag;checkpoint.mkdir(exist_ok=True)
    np.savez_compressed(checkpoint/'state.npz',final_state=e.y.get(),final_history=e.history.get(),
        final_history_tick_modulo_depth=0,dt_ms=a.dt,continued_duration_ms=one.elapsed_ms)
    write(checkpoint/'contract.json',dict(J_EE_core=a.J,source=str(a.trajectory),continued_duration_ms=one.elapsed_ms,
        purpose='Reference trajectory endpoint, reusable without repeating this interval.'))
    result['reference_checkpoint']=str(checkpoint/'state.npz')
    write(dest/(tag+'.json'),result);write(dest/(tag+'_status.json'),dict(status='COMPLETE',source=str(dest/(tag+'.json'))))
    print('LYAPUNOV ESTIMATES',estimates,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('trajectory');p.add_argument('--J',type=float,default=.946)
    p.add_argument('--dt',type=float,default=.1);p.add_argument('--epsilon',type=float,default=1e-6)
    p.add_argument('--duration',type=int,default=40000);p.add_argument('--device',type=int,default=1)
    p.add_argument('--transverse',action='store_true',help='Remove the neutral flow direction from full state and delay history before renormalization')
    main(p.parse_args())
