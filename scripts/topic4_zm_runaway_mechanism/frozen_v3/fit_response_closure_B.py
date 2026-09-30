"""Fit the response closure (structure V2) to the coarse-grid assay, two stages per population.

Local linear response at a workpoint (per population; w = 2 pi f):
  mean channel:       R_mu(w) = R_mu(0) [ a + (1-a)/(1+iw tau_s) ]
  variance channel c: R_c(w)  = R_c(0) [ a_c + (1-a_c)/(1+iw tau_vc) ] + R_mu(0) eta_c (iw tau_c)/(1+iw tau_c),  c in {E,I}
with a_c in [-1,1] (a_c=-1 with a short tau_vc is the first-order Pade form of a pure delay, which the
low-rate data show) and a fast transient eta_c acting as a mean shift. The assay measures variance
responses through the synaptic variance filter 1/(1+iw tau_syn/2), which the model also contains, so
it is divided out. Stage 1: tau_s (global) and a (per point) from the mean channel. Stage 2 per channel:
(tau_vc, tau_c) global and (a_c, eta_c) per point. Fit band 2-40 Hz; 80 Hz reported only.
Outputs response_closure/closure.json (+ .npz tables of a, a_E, a_I, eta_E, eta_I) and per-point rows.
"""
from common_v3 import *
from scipy.optimize import least_squares
import argparse
PARAMS=read(OPERATORS/'g20/prepared.json')['params'];TAU_SYN=[None,PARAMS['tau_r_AMPA']+PARAMS['tau_d_AMPA'],PARAMS['tau_r_GABA']+PARAMS['tau_d_GABA']]
FIT_F=[2.,5.,10.,20.,40.];SMOOTH=0.6

def load(pop):
    rows=read(DEST/'dynamic_assay/rows.json')['rows'];pts={}
    for r in rows:
        if r['pop']!=pop:continue
        pts.setdefault((r['x'],r['sigma_E'],r['sigma_I']),{}).setdefault(r['channel'],{})[r['frequency_hz']]=(complex(*r['response']),r['sem'],r['rate_hz'])
    return pts
def channel_data(chan,ch):
    f=np.array(sorted(chan));R=np.array([chan[q][0] for q in f]);sem=np.array([chan[q][1] for q in f]);R0=R[f==0][0].real;s0=sem[f==0][0];w=2j*np.pi*f/1000
    if ch>0:R=R*(1+w*TAU_SYN[ch]/2);sem=sem*abs(1+w*TAU_SYN[ch]/2)
    return f,w,R,sem,R0,s0
def mean_fit(pt,ts):
    f,w,R,sem,R0,s0=channel_data(pt[0],0);sel=np.isin(f,FIT_F);Hs=1/(1+w*ts);X=(1-Hs)[sel];y=(R/R0-Hs)[sel];wt=1/np.maximum(sem[sel]/abs(R0),1e-3)
    A=np.r_[X.real*wt,X.imag*wt];b=np.r_[y.real*wt,y.imag*wt];a=float(np.clip(A@b/(A@A),0,1));model=Hs+a*(1-Hs)
    return a,((model-R/R0)[sel])*wt,dict(dc=R0,snr=abs(R0)/s0,rms_norm=float(np.sqrt(np.mean(abs((model-R/R0)[sel])**2))),max_norm=float(abs((model-R/R0)[sel]).max()))
def var_fit(pt,ch,tv,tau):
    f,w,R,sem,Rc0,s0=channel_data(pt[ch],ch);sel=np.isin(f,FIT_F);Rmu0=channel_data(pt[0],0)[4];wt=1/np.maximum(sem[sel],1e-4)
    Hv=(1/(1+w*tv))[sel];hp=(w*tau/(1+w*tau))[sel];base=Rc0*Hv;X=np.array([Rc0*(1-Hv),Rmu0*hp]).T;y=R[sel]-base
    A=np.r_[X.real*wt[:,None],X.imag*wt[:,None]];b=np.r_[y.real*wt,y.imag*wt];coef,*_=np.linalg.lstsq(A,b,rcond=None);ac=float(np.clip(coef[0],-1,1))
    Xe=Rmu0*hp;y2=R[sel]-Rc0*(ac+(1-ac)*Hv);Ae=np.r_[Xe.real*wt,Xe.imag*wt];be=np.r_[y2.real*wt,y2.imag*wt];eta=float(Ae@be/(Ae@Ae))
    model=Rc0*(ac+(1-ac)*Hv)+Rmu0*eta*hp;scale=max(abs(R[sel]).max(),1e-9)
    return ac,eta,(model-R[sel])*wt,dict(dc=Rc0,snr=abs(Rc0)/s0,rms_rel=float(np.sqrt(np.mean(abs(model-R[sel])**2))/scale),max_rel=float(abs(model-R[sel]).max()/scale),snr_fast=float(abs(R[sel]).max()/np.median(sem[sel])))
def fit_population(pop,min_snr=10):
    pts=load(pop);keys=sorted(pts);use=[k for k in keys if abs(channel_data(pts[k][0],0)[4])/channel_data(pts[k][0],0)[5]>=min_snr]
    def tot1(lt):return np.concatenate([np.r_[mean_fit(pts[k],np.exp(lt[0]))[1].real,mean_fit(pts[k],np.exp(lt[0]))[1].imag] for k in use])
    r1=min((least_squares(tot1,[np.log(t0)],bounds=([np.log(2.)],[np.log(60.)])) for t0 in [6.,12.,25.]),key=lambda r:r.cost);ts=float(np.exp(r1.x[0]))
    alpha={k:mean_fit(pts[k],ts)[0] for k in use};poles=dict(tau_s=ts)
    for ch,name in [(1,'E'),(2,'I')]:
        pk=[k for k in use if ch in pts[k]]
        def tot2(lt):return np.concatenate([np.r_[var_fit(pts[k],ch,np.exp(lt[0]),np.exp(lt[1]))[2].real,var_fit(pts[k],ch,np.exp(lt[0]),np.exp(lt[1]))[2].imag] for k in pk])
        r2=min((least_squares(tot2,np.log(t0),bounds=(np.log([.3,.5]),np.log([60.,60.]))) for t0 in [[2.,8.],[5.,20.],[1.,3.],[10.,5.]]),key=lambda r:r.cost)
        poles[f'tau_v{name}']=float(np.exp(r2.x[0]));poles[f'tau_c{name}']=float(np.exp(r2.x[1]))
    rows=[]
    for k in keys:
        rate=list(pts[k][0].values())[0][2];row=dict(x=k[0],sigma_E=k[1],sigma_I=k[2],rate_hz=rate,used=k in use,diag={})
        if k in use:
            a,_,d=mean_fit(pts[k],ts);row['alpha']=a;row['diag']['mean']=d
            for ch,key,akey,name in [(1,'eta_E','a_E','E'),(2,'eta_I','a_I','I')]:
                if ch in pts[k]:
                    ac,eta,_,d=var_fit(pts[k],ch,poles[f'tau_v{name}'],poles[f'tau_c{name}']);row[akey]=ac;row[key]=eta;row['diag'][key]=d
                else:row[akey]=1.;row[key]=0.
        else:row.update(alpha=np.nan,a_E=np.nan,a_I=np.nan,eta_E=np.nan,eta_I=np.nan)
        rows.append(row)
    return poles,rows
def build_tables(rows):
    xs=sorted(set(r['x'] for r in rows));sEs=sorted(set(r['sigma_E'] for r in rows));sIs=sorted(set(r['sigma_I'] for r in rows));V=np.full((5,len(xs),len(sEs),len(sIs)),np.nan)
    for r in rows:
        i,j,k=xs.index(r['x']),sEs.index(r['sigma_E']),sIs.index(r['sigma_I']);V[:,i,j,k]=[r['alpha'],r['a_E'],r['a_I'],r['eta_E'],r['eta_I']]
    for p in range(5):
        for i in range(len(xs)):
            for j in range(len(sEs)):
                col=V[p,i,j];bad=np.isnan(col)
                if bad.all():continue
                col[bad]=np.interp(np.flatnonzero(bad),np.flatnonzero(~bad),col[~bad])
        for j in range(len(sEs)):
            for k in range(len(sIs)):
                col=V[p,:,j,k];bad=np.isnan(col)
                if bad.any() and not bad.all():col[bad]=np.interp(np.flatnonzero(bad),np.flatnonzero(~bad),col[~bad])
        # still-NaN (whole x-column missing): fill with the population median of the parameter
        V[p][np.isnan(V[p])]=np.nanmedian(V[p])
    # smoothing: the per-point weights are noisy and a_c switches between its bounds; a small Gaussian
    # kernel over grid indices bounds the table gradients (the linearised tangent dynamics is otherwise
    # stiff through the (state - filtered state) * d(weight)/d(moment) terms)
    from scipy.ndimage import gaussian_filter
    for p in range(5):V[p]=gaussian_filter(V[p],sigma=SMOOTH,mode='nearest')
    return np.array(xs),np.array(sEs),np.array(sIs),V
def main(a):
    out=DEST/'response_closure';out.mkdir(exist_ok=True);closure=dict(kind='tables',structure='V3: mean a+(1-a)/(1+iw tau_s); variance c: a_c+(1-a_c)/(1+iw tau_vc) (a_c in [-1,1]) plus fast transient eta_c iw tau_c/(1+iw tau_c) as mean shift; workpoint tables (alpha,a_E,a_I,eta_E,eta_I)',fit_band_hz=FIT_F,poles={},summary={});npz={}
    for pop in 'EI':
        poles,rows=fit_population(pop);closure['poles'][pop]=poles;used=[r for r in rows if r['used']]
        mr=[r['diag']['mean']['rms_norm'] for r in used];eE=[r['diag']['eta_E']['rms_rel'] for r in used if 'eta_E' in r['diag'] and r['diag']['eta_E']['snr_fast']>=10];eI=[r['diag']['eta_I']['rms_rel'] for r in used if 'eta_I' in r['diag'] and r['diag']['eta_I']['snr_fast']>=10]
        closure['summary'][pop]=dict(points=len(rows),used=len(used),mean_channel_rms_norm=dict(median=float(np.median(mr)),p90=float(np.percentile(mr,90)),max=float(max(mr))),
            varE_channel_rms_rel=dict(n=len(eE),median=float(np.median(eE)),p90=float(np.percentile(eE,90))),varI_channel_rms_rel=dict(n=len(eI),median=float(np.median(eI)),p90=float(np.percentile(eI,90))),
            ranges={k:[float(np.nanmin([r[k] for r in used])),float(np.nanmax([r[k] for r in used]))] for k in ['alpha','a_E','a_I','eta_E','eta_I']})
        xs,sEs,sIs,V=build_tables(rows);npz.update({f'x_{pop}':xs,f'sE_{pop}':sEs,f'sI_{pop}':sIs,f'values_{pop}':V});write(out/f'points_{pop}.json',dict(poles=poles,rows=rows))
        log(pop,'poles',{k:round(v,3) for k,v in poles.items()},json.dumps(closure['summary'][pop]))
        for r in rows:
            if r['sigma_E']==.7 and r['sigma_I']==.7 and r['used']:print(f"   x={r['x']:6.2f} rate={r['rate_hz']:7.1f} a={r['alpha']:.2f} aE={r['a_E']:.2f} aI={r['a_I']:.2f} etaE={r['eta_E']:+.3f} etaI={r['eta_I']:+.3f} rms mean={r['diag']['mean']['rms_norm']:.3f} vE={r['diag'].get('eta_E',{}).get('rms_rel',np.nan):.3f} vI={r['diag'].get('eta_I',{}).get('rms_rel',np.nan):.3f}")
    closure['source_assay']=str(DEST/'dynamic_assay/rows.json');closure['table_smoothing_sigma_gridpoints']=SMOOTH;closure['note']='alpha clipped to [0,1], a_E,a_I to [-1,1]; eta unclipped; 80 Hz not fitted; mean-channel high-frequency lead of mean-driven neurons is outside the closure'
    np.savez_compressed(out/'closure.npz',**npz);write(out/'closure.json',closure);log('WROTE closure')
if __name__=='__main__':main(argparse.ArgumentParser().parse_args())
