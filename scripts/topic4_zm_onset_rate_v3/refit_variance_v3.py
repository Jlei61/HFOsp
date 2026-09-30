"""Variant V3 for the variance channels: R_c = R_c0 [a_c + (1-a_c)/(1+iw tau_vc)] + Rmu0 eta (iw tau_c)/(1+iw tau_c),
a_c in [-1,1] (a_c=-1, short tau_vc = first-order Pade of a pure delay), own poles (tau_vc, tau_c) per channel."""
from fit_response_closure_B import *
def fit_v3(pop,ch,ts):
    pts=load(pop);keys=sorted(pts);use=[k for k in keys if abs(channel_data(pts[k][0],0)[4])/channel_data(pts[k][0],0)[5]>=10 and ch in pts[k]]
    def point(k,tv,tc):
        f,w,R,sem,Rc0,s0=channel_data(pts[k][ch],ch);sel=np.isin(f,FIT_F);Rmu0=channel_data(pts[k][0],0)[4];wt=1/np.maximum(sem[sel],1e-4)
        Hv=(1/(1+w*tv))[sel];hp=(w*tc/(1+w*tc))[sel];base=Rc0*Hv;X=np.array([Rc0*(1-Hv),Rmu0*hp]).T;y=R[sel]-base
        A=np.r_[X.real*wt[:,None],X.imag*wt[:,None]];b=np.r_[y.real*wt,y.imag*wt];coef,*_=np.linalg.lstsq(A,b,rcond=None);ac=float(np.clip(coef[0],-1,1))
        Xe=Rmu0*hp;y2=R[sel]-Rc0*(ac+(1-ac)*Hv);Ae=np.r_[Xe.real*wt,Xe.imag*wt];be=np.r_[y2.real*wt,y2.imag*wt];eta=float(Ae@be/(Ae@Ae))
        model=Rc0*(ac+(1-ac)*Hv)+Rmu0*eta*hp;scale=max(abs(R[sel]).max(),1e-9)
        return (model-R[sel])*wt,float(np.sqrt(np.mean(abs(model-R[sel])**2))/scale),ac,eta,abs(R[sel]).max()/np.median(sem[sel])
    def tot(lt):return np.concatenate([np.r_[point(k,np.exp(lt[0]),np.exp(lt[1]))[0].real,point(k,np.exp(lt[0]),np.exp(lt[1]))[0].imag] for k in use])
    best=min((least_squares(tot,np.log(t0),bounds=(np.log([.3,.5]),np.log([60.,60.]))) for t0 in [[2.,8.],[5.,20.],[1.,3.],[10.,5.]]),key=lambda r:r.cost);tv,tc=np.exp(best.x)
    res=[point(k,tv,tc) for k in use];rms=[r[1] for r in res if r[4]>=10];acs=[r[2] for r in res]
    return dict(tau_v=float(tv),tau_c=float(tc),n=len(rms),median=float(np.median(rms)),p90=float(np.percentile(rms,90)),a_c_quantiles=[float(q) for q in np.percentile(acs,[10,50,90])])
cl=read(DEST/'response_closure/closure.json')
for pop in 'EI':
    for ch in (1,2):print(pop,'ch',ch,fit_v3(pop,ch,cl['poles'][pop]['tau_s']))
