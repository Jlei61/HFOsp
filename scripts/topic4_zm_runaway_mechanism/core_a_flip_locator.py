"""Bounded refinement of an existing numerical -1 crossing bracket.

No physical model, spectrum gate, or onset claim is promoted by this driver.
Every new point uses the original full spatial shooting and derivative tools.
"""
from common import OUT,np,read,write,log,model
from core_a_parameter_path_audit import NativeTimeFamily
from pathlib import Path
import argparse,os,subprocess,sys,time

ROOT=OUT/'core_a_bifurcation_type_20260924/reference_stability_gap/three_burst_entry_continuation'
HERE=Path(__file__).resolve().parent


def root_state(parent):
    r=read(parent/'result.json');assert r['status']=='NUMERICAL_SEGMENTED_SINGLE_SHOOTING_ROOT'
    row=r['iterations'][-1]
    return parent/f"iteration{row['iteration']:02d}"/'node00.npz',row['period_ms']


def negative_mode(parent):
    folder=parent/'section_spectrum_segmented';r=read(folder/'result.json')
    pairs=[(i,v) for i,v in enumerate(r['verified']) if v['status']=='VERIFIED_NUMERICAL_EIGENPAIR' and v['imag']==0 and v['real']<0]
    assert len(pairs)==1,('Need unique verified negative candidate',pairs)
    index,pair=pairs[0]
    return pair,folder/f'mode{index:02d}.npz'


def main(a):
    dest=ROOT/'flip_locator';dest.mkdir(exist_ok=True);assert not(dest/'jobs.json').exists()
    write(dest/'contract.json',dict(question='Locate the already observed numerical negative-mode crossing of -1 on the same three-burst branch.',
        initial_parameter_interval_ms=[9573.043350229,9583.9750181025],
        precomputed_next_point=str(ROOT/'native9576/whole_cycle'),
        maximum_new_parameter_points=a.points,negative_multiplier_target_tolerance=1e-4,
        method='Bracket-preserving safeguarded secant updates of the native spatial-field time parameter. At each point correct the full delayed spatial cycle and evaluate the original numerical Poincare spectrum with independent eigenvector residuals. Neighboring physical right-mode overlaps are reported to check branch identity.',
        equation='Unchanged3479-group0.5mm physical-delay conditional rate, only withinCoreA nativeZ varied, outsideA native9s, everyZheld and allE Mdynamic.',
        scope='Numerical crossing refinement only. Prior phase/mesh failures are retained. No complete spectrum, normal-form nondegeneracy, physical bifurcation type, global onset or native-SNN attribution is certified.',model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid());write(dest/'jobs.json',jobs);started=time.time()
    def run(args,logpath):
        jobs.update(current_command=args,current_log=str(logpath));write(dest/'jobs.json',jobs)
        with logpath.open('w') as output:
            proc=subprocess.Popen([sys.executable,'-u',*args],stdout=output,stderr=subprocess.STDOUT)
            jobs.update(child_pid=proc.pid);write(dest/'jobs.json',jobs)
            rc=proc.wait()
        assert rc==0,('Child failed',rc,str(logpath))
        jobs.pop('child_pid',None);write(dest/'jobs.json',jobs)
    try:
        s=model(40);family=NativeTimeFamily(s);rows=[]
        def inspect(theta,parent):
            source,T=root_state(parent);pair,path=negative_mode(parent)
            field,_=family.field_at_time(theta)
            state=np.load(source);assert np.array_equal(state['syn'][5],field)
            assert np.all(state['parameters'][19]==0) and np.all(state['parameters'][20]==1)
            za=float(np.average(field[family.A],weights=s.sizes[family.A]))
            row=dict(theta_ms=theta,D_A=1-za,Z_A=za,parent=str(parent),period_ms=T,
                multiplier=pair['real'],eigenpair_relative_residual=pair['relative_residual'],mode_path=str(path),
                physical_Floquet=read(parent/'section_spectrum_segmented/result.json')['physical_Floquet'])
            if rows:
                nearest=min(rows,key=lambda r:abs(r['theta_ms']-theta));old=np.load(nearest['mode_path']);new=np.load(path)
                u=old['vector'].real;v=(new['vector'].real.reshape(-1,s.P)*new['coordinate_scale']/new['coordinate_weight']*old['coordinate_weight']/old['coordinate_scale']).ravel()
                row['nearest_mode_absolute_overlap']=float(abs(u@v)/(np.linalg.norm(u)*np.linalg.norm(v)))
                row['nearest_mode_parameter_ms']=nearest['theta_ms']
                assert row['nearest_mode_absolute_overlap']>.8,('Numerical mode identity unresolved',row)
            rows.append(row);write(dest/'points.json',rows);log('NUMERICAL FLIP BRACKET POINT',row)
            return row
        inspect(9573.043350229,ROOT/'native9573/whole_cycle')
        inspect(9583.9750181025,ROOT/'native9584/whole_cycle_finish')
        current=ROOT/'native9576/whole_cycle';root_state(current)
        if not(current/'section_spectrum_segmented/result.json').exists():
            run([str(HERE/'core_a_section_spectrum.py'),str(current),'--device',str(a.device),'--krylov','16','--cache-segments','6','--measure-neutral'],dest/'native9576_spectrum.log')
        inspect(9576.328713509794,current)
        status='POINT_BUDGET_COMPLETE_PHYSICAL_TYPE_PENDING'
        for k in range(a.points):
            best=min(rows,key=lambda r:abs(r['multiplier']+1))
            if abs(best['multiplier']+1)<1e-4:
                status='NUMERICAL_MINUS_ONE_LOCATED_PHYSICAL_TYPE_PENDING';break
            low=max([r for r in rows if r['multiplier']>-1],key=lambda r:r['theta_ms'])
            high=min([r for r in rows if r['multiplier']<-1],key=lambda r:r['theta_ms'])
            lo,hi=low['theta_ms'],high['theta_ms'];assert lo<hi
            theta=lo+(-1-low['multiplier'])*(hi-lo)/(high['multiplier']-low['multiplier'])
            theta=float(np.clip(theta,lo+.02*(hi-lo),hi-.02*(hi-lo)))
            near=min(rows,key=lambda r:abs(r['theta_ms']-theta));source,T=root_state(Path(near['parent']))
            predictor=T+(theta-near['theta_ms'])*(high['period_ms']-low['period_ms'])/(hi-lo)
            parent=dest/f'refine{k:02d}'
            run([str(HERE/'core_a_whole_cycle_newton.py'),str(source),'--destination',str(parent),'--period',str(predictor),
                '--target-native-time',str(theta),'--device',str(a.device),'--dt','.05','--source-dt','.05','--segments','6',
                '--iterations','5','--krylov','24','--radius','.001','--max-trial-evaluations','6','--cache-check','ends'],dest/f'refine{k:02d}_root.log')
            root_state(parent)
            run([str(HERE/'core_a_section_spectrum.py'),str(parent),'--device',str(a.device),'--krylov','16','--cache-segments','6','--measure-neutral'],dest/f'refine{k:02d}_spectrum.log')
            inspect(theta,parent)
        best=min(rows,key=lambda r:abs(r['multiplier']+1))
        if abs(best['multiplier']+1)<1e-4:status='NUMERICAL_MINUS_ONE_LOCATED_PHYSICAL_TYPE_PENDING'
        write(dest/'result.json',dict(status=status,best=best,points=rows,seconds=time.time()-started,
            target_entry_type='NOT_ESTABLISHED',nonlinear_criticality='NOT_ESTABLISHED',physical_phase_mesh='NOT_QUALIFIED',model_promoted=False))
        jobs.update(status='COMPLETE');write(dest/'jobs.json',jobs)
    except BaseException as exc:
        jobs.update(status='FAILED',error=repr(exc));write(dest/'jobs.json',jobs);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);p.add_argument('--points',type=int,default=3)
    a=p.parse_args();assert 1<=a.points<=4;main(a)
