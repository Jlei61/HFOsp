"""Reproducible entry point for validated critical-curve continuation."""
import periodic_boundaries as p
import argparse


if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('label',choices=list(p.seeds()));a.add_argument('--N',type=int,default=1024)
    a.add_argument('--step',type=float,default=.05);a.add_argument('--min-step',type=float,default=.00002)
    a.add_argument('--initial');a.add_argument('--start-h',type=float,default=1.)
    a.add_argument('--direction',type=float,default=-1.);a.add_argument('--suffix',default='_rerun')
    a.add_argument('--fold-method',choices=['strict','joint'],default='strict')
    x=a.parse_args();pd=x.label.startswith('PD')
    if pd:
        from pd_joint import correct
        p.correct_pd_bordered=correct
        # The fixed-g particular derivative is ill-conditioned near PD2;
        # the joint corrector from the preceding orbit is much more accurate.
        p.orbit_derivative=lambda h,z,N:(p.np.zeros_like(z),p.np.zeros_like(z))
    elif x.fold_method=='joint':
        from fold_joint import correct
        p.correct_fold=correct
        p.fold_derivative=lambda h,z,t,N:(p.np.zeros_like(z),p.np.zeros_like(z))
    p.run(x.label,x.N,x.step,x.initial,x.start_h,x.direction,x.suffix,pd,x.min_step)
