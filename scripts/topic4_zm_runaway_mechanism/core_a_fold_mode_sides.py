"""Check the zero-mode sign and pre-existing instabilities around the fold."""
from common import OUT,np,read,write,log
from core_a_equilibrium_branch import Family
from physical_delay_conditional_drift import PhysicalDelayConditionalDrift
from current_rate_logit_root import value_and_jacobian
from core_a_static_spectrum import refine
from scipy.sparse.linalg import spsolve

DEST=OUT/'core_a_bifurcation_type_20260924/fold_certificate'


def main():
    cert=read(DEST/'result.json');assert cert['status']=='GENERIC_EQUILIBRIUM_SADDLE_NODE_CERTIFIED'
    data=np.load(DEST/'critical_point.npz');s=PhysicalDelayConditionalDrift();family=Family(s)
    q0=data['q'];D0=float(data['D_A']);v=data['v_logit'];a=cert['transversality'][-1]['parameter_coefficient'];b=cert['quadratic'][-1]['quadratic_coefficient']
    mode=np.load(DEST.parent/'numerical_root_homotopy/stationary_flow_spectrum/rate_mode0.npz');rows=[];delta=1e-5
    for side in [-1,0,1]:
        D=D0+(delta if side else 0.);family.set(D);q=q0+side*np.sqrt(-a/b*delta)*v
        for it in range(15):
            r,F,op,J=value_and_jacobian(s,q)
            if abs(op['rate']-r).max()<1e-11:break
            direction=spsolve(J,-F);norm=np.linalg.norm(F)
            for alpha in 2.**-np.arange(18):
                trial=q+alpha*direction
                if np.linalg.norm(value_and_jacobian(s,trial,False)[1])<norm:q=trial;break
            else:raise RuntimeError('Side root correction stalled')
        r,F,op=value_and_jacobian(s,q,False);error=float(abs(s.residual(r)).max());assert error<1e-11,error
        results=[]
        for kind,guess,vector in [('zero',side*.0001,data['v_rate']),('growing',complex(mode['lambda_per_ms']),mode['v'])]:
            for dt in [.05,.025,None]:
                ll,vv,tr=refine(s,r,op,guess,dt,vector.copy());assert ll is not None,(side,kind,dt,tr[-1])
                err=float(np.linalg.norm(s.characteristic(r,ll,dt=dt)@vv));assert err<1e-8
                en=s.sizes*abs(vv)**2;en/=en.sum()
                results.append(dict(kind=kind,dt_ms=dt,lambda_per_ms=[float(ll.real),float(ll.imag)],growth_per_s=float(ll.real*1000),frequency_hz=float(abs(ll.imag)*1000/(2*np.pi)),residual=err,
                    mode_energy_E_A_B_surround_I=[float(en[s.E&(s.geo['group_region']==j)].sum()) for j in range(3)]+[float(en[~s.E].sum())]))
        np.savez_compressed(DEST/f'side{side:+d}.npz',q=q,r=r,Z=s.Z,D_A=D)
        row=dict(side=side,D_A=D,regional_rates_hz=s.regional_rates(r),original_rate_residual_per_ms=error,modes=results);rows.append(row);write(DEST/'mode_sides_progress.json',rows);log('CORE A FOLD SIDE MODES',row)
    zero={row['side']:next(m['growth_per_s'] for m in row['modes'] if m['kind']=='zero' and m['dt_ms']==.05) for row in rows}
    assert zero[-1]*zero[1]<0
    assert all(m['growth_per_s']>0 for row in rows for m in row['modes'] if m['kind']=='growing')
    write(DEST/'mode_sides.json',dict(status='SN_WITH_PREEXISTING_OSCILLATORY_INSTABILITY_VERIFIED',rows=rows,
        meaning='The simple real eigenvalue changes sign between the two equilibrium sheets, while an independently identified positive complex pair persists on both. This is a saddle-node of already unstable equilibria, not destruction of a stable resting state.',
        actual_attractor_transition='NOT_ESTABLISHED',model_promoted=False))


if __name__=='__main__':main()
