"""Full delayed-state, phase-conditioned shooting correction near onset.

The return map integrates the unchanged conditional drift. Anderson mixing
solves its fixed-point residual; it does not reduce the physical state or
certify stability. Fractional returns use canonical-history interpolation,
so mesh refinement remains mandatory before any bifurcation label.
"""
from common import np, read, write, log
from onset_state_continuation import DEST, build, regrid_state
from onset_variational_return import Coordinates
from onset_period_return import dynamical_state, errors
from fine_rate_frozen_Z_fields import capture, restore
from datetime import datetime
import argparse, os, time


def regional_rate_section(A,region,orientation=None):
    """Choose an actual E-rate phase plane without changing the flow."""
    e=A.e;mask=e.s.E&(e.s.geo['group_region']==region)
    w=e.s.sizes*mask;w=w/w.sum()
    normal=np.zeros_like(A.xref)
    normal.reshape(-1,e.s.P)[47]=w*A.c.scale[47]/A.c.weight[0]
    seed_direction=float(normal@A.normal)
    if orientation is None:
        assert abs(seed_direction)>1e-14,'Regional phase plane is not transverse at the seed'
        orientation=np.sign(seed_direction)
    assert orientation in [-1,1]
    normal*=orientation;A.normal=normal/np.linalg.norm(normal)


def regional_M_section(A,region,orientation=None):
    """Use a regional mean-M crossing to mark phase; M stays dynamic."""
    e=A.e;mask=e.s.E&(e.s.geo['group_region']==region)
    w=e.s.sizes*mask;w=w/w.sum()
    normal=np.zeros_like(A.xref)
    normal.reshape(-1,e.s.P)[4]=w*A.c.scale[4]/A.c.weight[0]
    seed_direction=float(normal@A.normal)
    if orientation is None:
        assert abs(seed_direction)>1e-14,'M section is not transverse at the seed'
        orientation=np.sign(seed_direction)
    assert orientation in [-1,1]
    normal*=orientation;A.normal=normal/np.linalg.norm(normal)


class SectionReturn:
    def __init__(self, state, engine, period, halfwidth=3.):
        self.e=engine; self.base={k:v.copy() for k,v in state.items()}
        self.c=c=Coordinates(state,engine.s); self.xref=c.pack(state)
        self.period=period; self.halfwidth=halfwidth; self.calls=0
        restore(engine,state)
        assert not engine.noise and not engine.transport.drive_on
        assert np.all(state['parameters'][19]==0) and np.all(state['parameters'][20]==1)
        points=[self.xref]
        for _ in range(2):
            engine.step();engine.cp.cuda.get_current_stream().synchronize()
            points.append(c.pack(capture(engine)))
        tangent=(-3*points[0]+4*points[1]-points[2])/(2*engine.dt)
        self.speed=float(np.linalg.norm(tangent));self.normal=tangent/self.speed
        restore(engine,state)
        with engine.stream:
            engine.stream.begin_capture()
            for _ in range(round(1/engine.dt)):engine.step()
            self.one_ms=engine.stream.end_capture()

    def state(self,x):
        c=self.c;a=x.reshape(-1,c.P)*c.scale/c.weight
        state={k:v.copy() for k,v in self.base.items()}
        state['syn'][:5]=a[:5];state['local'][:]=a[5:47]
        state['history'][(c.tick-np.arange(c.depth))%c.depth]=a[47:]
        return state

    def admissible(self,x):
        if not np.isfinite(x).all():return False
        a=x.reshape(-1,self.c.P)*self.c.scale/self.c.weight
        if a[:5].min() < -1e-12 or a[5:11].min() < -1e-12:return False
        if a[47:].min() < -1e-12 or np.max(abs(a[4,~self.e.s.E]))>1e-12:return False
        for mask,ref in [(self.e.s.E,2.),(~self.e.s.E,1.)]:
            # All windows of the complete lag history, not only the latest.
            h=a[47:,mask];n=round(ref/self.e.dt)
            sums=np.vstack([np.zeros((1,h.shape[1])),np.cumsum(h,axis=0)])
            if ((sums[n:]-sums[:-n])*self.e.dt).max()>1+1e-7:return False
        return True

    def __call__(self,x):
        e=self.e;c=self.c
        assert self.admissible(x), 'Corrector must not clamp an inadmissible state'
        restore(e,self.state(x));initial_tick=c.tick
        lo=max(1,int(np.floor(self.period-self.halfwidth)))
        hi=int(np.ceil(self.period+self.halfwidth))
        whole=lo//10
        for _ in range(whole):e.chunk()
        for _ in range(lo-10*whole):
            self.one_ms.launch(e.stream);e.stream.synchronize()
        left=capture(e);u=c.pack(left);g=float(self.normal@(u-self.xref))
        if g>=0:raise RuntimeError('Section crossing precedes the declared search window')
        crossing=None
        for tm in range(lo+1,hi+1):
            self.one_ms.launch(e.stream);e.stream.synchronize()
            right=capture(e);v=c.pack(right);h=float(self.normal@(v-self.xref))
            if g<=0<h:
                crossing=(left,tm-1);break
            left=right;u=v;g=h
        if crossing is None:raise RuntimeError('No positively oriented section return in declared window')
        left,tm=crossing;restore(e,left);u=c.pack(left);g=float(self.normal@(u-self.xref))
        for k in range(1,round(1/e.dt)+1):
            e.step();e.cp.cuda.get_current_stream().synchronize()
            right=capture(e);v=c.pack(right);h=float(self.normal@(v-self.xref))
            if g<=0<h:
                alpha=-g/(h-g);out=(1-alpha)*u+alpha*v
                period=tm+(k-1+alpha)*e.dt
                assert int(right['clock'][0])-initial_tick==round((tm+k*e.dt)/e.dt)
                assert np.array_equal(right['syn'][5],self.base['syn'][5])
                self.calls+=1
                self.last_time_slope=(v-u)/e.dt
                return out,dict(period_ms=float(period),fraction=float(alpha),
                    section_residual=float(self.normal@(out-self.xref)),
                    section_speed=float((h-g)/e.dt),initial_phase=float(self.normal@(x-self.xref)))
            u=v;g=h
        raise RuntimeError('Fine section bracket failed')


def correct(label,device,dt=None,iterations=10,source_name='final_state.npz',halfwidth=3.,name=None,field=None):
    source=DEST/label;jobs=read(source/'jobs.json');source_dt=jobs['condition'].get('dt_ms',.05)
    assert jobs['status']=='COMPLETE'
    dt=source_dt if dt is None else dt
    period=read(source/'period_return/result.json')['interpolated_period_ms']
    name=name or 'shooting_corrector_dt'+str(dt).replace('.','p')
    out=source/name;out.mkdir(exist_ok=True);assert not (out/'jobs.json').exists()
    write(out/'contract.json',dict(created_local=datetime.now().astimezone().isoformat(),
        source=str(source/source_name),source_dt_ms=source_dt,dt_ms=dt,seed_period_ms=period,
        equations='Unchanged full g40 conditional drift, full spatial Z held, M dynamic, constant original input mean, no future count innovations.',
        method='Solve P(x)-x=0 on a fixed hyperplane through the seed, normal to its full-state flow. Integrate the actual delay/refractory/synaptic/local/M states. Positive crossing searched within declared period window, refined to actual step then linearly interpolated in canonical lag coordinates. Anderson memory3, otherwise physical Picard iterate. No state clamp.',
        return_search_halfwidth_ms=halfwidth,
        target_field=field or jobs['condition']['field'],
        field_change='Only full spatial Z changes when an explicit target field is supplied. Every initial fast, M and delay state comes from the declared source.',
        residual_gate='Combined six-block relative RMS <1e-7 and every block <1e-6, section residual <1e-9; at least two evaluated returns. At most declared iterations. This is a numerical shooting-root check, not continuous-time, stability, fundamental-period or bifurcation certification.',
        iterations=iterations,model_promoted=False))
    status=dict(status='RUNNING',pid=os.getpid(),iteration=0);write(out/'jobs.json',status)
    start=time.time()
    try:
        e=build(device,dt=dt);raw={k:v for k,v in np.load(source/source_name).items()}
        base=regrid_state(raw,e,source_dt)
        if field:base['syn'][5]=np.load(DEST/'fields.npz')[field]
        A=SectionReturn(base,e,period,halfwidth)
        x=A.xref.copy();roundtrip=A.c.pack(A.state(x))
        assert np.linalg.norm(roundtrip-x)/max(np.linalg.norm(x),1)<1e-14
        assert A.admissible(x)
        rows=[];xs=[];ys=[];fs=[];weights=e.s.sizes/e.s.sizes.sum();success=False
        for it in range(iterations):
            y,phase=A(x);f=y-x
            err=errors(dynamical_state(A.state(x)),dynamical_state(A.state(y)),weights)
            row=dict(iteration=it+1,**phase,**err,weighted_coordinate_relative=float(np.linalg.norm(f)/max(np.linalg.norm(x),1)))
            rows.append(row);write(out/'iterations.json',rows)
            log('ONSET SHOOTING',label,dt,it+1,phase['period_ms'],err['combined_relative_rms'])
            np.savez_compressed(out/'latest_state.npz',**A.state(x))
            status.update(iteration=it+1,last_residual=err['combined_relative_rms']);write(out/'jobs.json',status)
            if it>=1 and err['combined_relative_rms']<1e-7 and max(v['relative_rms'] for v in err['blocks'].values())<1e-6 and abs(phase['section_residual'])<1e-9:
                success=True;break
            xs.append(x);ys.append(y);fs.append(f)
            xs=xs[-3:];ys=ys[-3:];fs=fs[-3:]
            proposal=y;method='Picard';coefficients=[1.]
            if len(fs)>1:
                gram=np.array([[float(a@b) for b in fs] for a in fs]);n=len(fs)
                gram+=np.eye(n)*max(float(np.trace(gram)),1e-30)*1e-12
                kkt=np.zeros((n+1,n+1));kkt[:n,:n]=gram;kkt[n,:n]=1;kkt[:n,n]=1
                rhs=np.zeros(n+1);rhs[n]=1
                coeff=np.linalg.solve(kkt,rhs)[:n]
                mixed=sum(c*v for c,v in zip(coeff,ys))
                if np.sum(abs(coeff))<=10 and np.linalg.norm(mixed-x)<=.01*max(np.linalg.norm(x),1) and A.admissible(mixed):
                    proposal=mixed;method='Anderson3';coefficients=coeff.tolist()
            row['next_method']=method;row['mixing_coefficients']=coefficients
            write(out/'iterations.json',rows);x=proposal
        result=dict(status='NUMERICAL_SHOOTING_ROOT' if success else 'CORRECTOR_BUDGET_EXHAUSTED_NOT_A_BIFURCATION',
            iterations=rows,dt_ms=dt,period_ms=rows[-1]['period_ms'],elapsed_seconds=time.time()-start,
            D=float(1-base['syn'][5,e.s.E]@e.s.mean_weights),field=field or jobs['condition']['field'],
            scope='Interpolated discrete-step section-map root only. Same-branch mesh refinement, independent residual, fundamental-period and Floquet certification still required.',model_promoted=False)
        write(out/'result.json',result);status['status']='COMPLETE';write(out/'jobs.json',status)
    except BaseException as exc:
        status.update(status='FAILED',error=repr(exc));write(out/'jobs.json',status);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--device',type=int,default=0)
    p.add_argument('--dt',type=float);p.add_argument('--iterations',type=int,default=10)
    p.add_argument('--source-name',default='final_state.npz');p.add_argument('--halfwidth',type=float,default=3.)
    p.add_argument('--name');p.add_argument('--field')
    a=p.parse_args();assert 2<=a.iterations<=12
    correct(a.label,a.device,a.dt,a.iterations,a.source_name,a.halfwidth,a.name,a.field)
