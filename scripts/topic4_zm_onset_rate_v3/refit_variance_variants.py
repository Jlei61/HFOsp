"""Compare variance-channel structures on the assay (fit band 2-40 Hz), with the mean channel fixed as
alpha + (1-alpha)/(1+iw tau_s) (poles from closure.json):
  V0 (current): R_c = R_c0 + Rmu0 eta (iw t_c)/(1+iw t_c)
  V1: R_c = R_c0 H_mu(w) + Rmu0 eta (iw t_c)/(1+iw t_c)           (static part through the same mean-channel filter)
  V2: R_c = R_c0 [a_c + (1-a_c)/(1+iw tau_s)] + Rmu0 eta (iw t_c)/(1+iw t_c)   (own mixing weight a_c)
Per-point linear LS for (eta) or (a_c, eta); t_c refitted globally per variant. Reports rms_rel medians.
"""
from fit_response_closure_B import *
from scipy.optimize import least_squares
def fit_variant(pop,variant):
    pts=load(pop);cl=read(DEST/'response_closure/closure.json');ts=cl['poles'][pop]['tau_s'];keys=sorted(pts)
    use=[k for k in keys if abs(channel_data(pts[k][0],0)[4])/channel_data(pts[k][0],0)[5]>=10]
    # alpha per point from the mean channel with the fixed pole
    alpha={}
    for k in use:
        f,w,R,sem,R0,s0=channel_data(pts[k][0],0);sel=np.isin(f,FIT_F);Hs=1/(1+w*ts);X=(1-Hs)[sel];y=(R/R0-Hs)[sel];wt=1/np.maximum(sem[sel]/abs(R0),1e-3)
        A=np.r_[X.real*wt,X.imag*wt];b=np.r_[y.real*wt,y.imag*wt];alpha[k]=float(np.clip(A@b/(A@A),0,1))
    def point(k,ch,tau):
        f,w,R,sem,Rc0,s0=channel_data(pts[k][ch],ch);sel=np.isin(f,FIT_F);wt=1/np.maximum(sem[sel],1e-4);Rmu0=channel_data(pts[k][0],0)[4]
        hp=(w*tau/(1+w*tau))[sel];Hs=(1/(1+w*ts))[sel];a=alpha[k]
        if variant=='V0':base=Rc0*np.ones(sel.sum());cols=[Rmu0*hp]
        elif variant=='V1':base=Rc0*(a+(1-a)*Hs);cols=[Rmu0*hp]
        else:base=Rc0*Hs;cols=[Rc0*(1-Hs),Rmu0*hp]
        X=np.array(cols).T;y=R[sel]-base
        A=np.r_[X.real*wt[:,None],X.imag*wt[:,None]];b=np.r_[y.real*wt,y.imag*wt];coef,*_=np.linalg.lstsq(A,b,rcond=None)
        model=base+X@coef;res=(model-R[sel])*wt;scale=max(abs(R[sel]).max(),1e-9)
        return res,float(np.sqrt(np.mean(abs(model-R[sel])**2))/scale),coef,abs(R[sel]).max()/np.median(sem[sel])
    out={}
    for ch in (1,2):
        pk=[k for k in use if ch in pts[k]]
        def total(logtau):return np.concatenate([np.r_[point(k,ch,np.exp(logtau[0]))[0].real,point(k,ch,np.exp(logtau[0]))[0].imag] for k in pk])
        best=min((least_squares(total,[np.log(t0)],bounds=([np.log(.5)],[np.log(60.)])) for t0 in [3.,8.,20.]),key=lambda r:r.cost);tau=float(np.exp(best.x[0]))
        rms=[point(k,ch,tau)[1] for k in pk if point(k,ch,tau)[3]>=10];out[ch]=dict(tau=tau,n=len(rms),median=float(np.median(rms)),p90=float(np.percentile(rms,90)))
    return out
for pop in 'EI':
    for v in ['V0','V1','V2']:
        r=fit_variant(pop,v);print(pop,v,{ch:{k:(round(x,3) if isinstance(x,float) else x) for k,x in d.items()} for ch,d in r.items()})
