"""Isolate dynamic M feedback at the SAME equilibrium and SAME M field.

Holding M at its equilibrium field removes its dynamic feedback without
changing its static inhibitory current or the equilibrium's operating point.
This control differs from replacing M by a reference-time mean or zero.
"""
from temporal_modes import *

def main():
    s=ZMRate();f=ZMRate(mode='frozen_M');rows=[]
    for label in ['snapshot1_D0p228','snapshot2_D0p256','snapshot3_D0p275']:
        p=DEST/'g20/contour_modes'/label;d=np.load(p/'modes.npz');roots=d['roots'];k=int(np.argmax(roots.real))
        r=d['r'];D=float(d['D']);v=d['vectors'][k];lam=roots[k]
        star=1000*f.E*r;mean=float(star[f.E]@f.mean_weights)
        f.m_current=.0005*mean;f.m_shape=star/mean
        e_dynamic=float(abs(s.residual(r,D)).max());e_frozen=float(abs(f.residual(r,D)).max())
        assert max(e_dynamic,e_frozen)<1e-8
        dyn=refine(Linearization(s,r,D),lam,v);held=refine(Linearization(f,r,D),lam,v)
        assert dyn is not None and held is not None
        row=dict(label=label,D=D,global_E_hz=s.global_rate(r),M_mean=mean,eta_M_mean_mv=f.m_current,
            dynamic_M_lambda_per_ms=dyn[0],fixed_same_M_lambda_per_ms=held[0],
            eigenvalue_change_per_ms=held[0]-dyn[0],
            relative_growth_change=float((held[0].real-dyn[0].real)/dyn[0].real),
            equilibrium_residual_dynamic=e_dynamic,equilibrium_residual_frozen=e_frozen,
            eigen_residual_dynamic=dyn[2],eigen_residual_frozen=held[2],
            scope='Only the selected growing mode at this equilibrium; does not test slow-path or nonlinear M effects')
        rows.append(row);print(row,flush=True)
    write(DEST/'m_dynamic_feedback_same_state_control.json',dict(status='COMPLETE',rows=rows,
        intervention='Freeze M to the SAME spatial equilibrium field; all other values unchanged',
        nonlinear_onset_effect='NOT_ESTABLISHED'))

if __name__=='__main__':main()
