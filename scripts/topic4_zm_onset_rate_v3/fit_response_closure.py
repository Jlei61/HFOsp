"""Analyse the coarse-grid dynamic assay: per-workpoint normalised response shapes per channel,
fit candidate finite-dimensional filter structures, and quantify workpoint dependence.

Shapes: H_c(w) = R_c(w) / R_c(0), with the model's own synaptic variance filter
1/(1 + i w tau_syn/2) divided out for the variance channels (the assay modulates input noise
intensity, which the model represents through the va/vg states).
Candidates: (S1) beta + (1-beta)/(1+i w tau); (S2) alpha/(1+i w tf) + (1-alpha)/(1+i w ts).
"""
from common_v3 import *
from scipy.optimize import least_squares
import argparse
PARAMS=read(OPERATORS/'g20/prepared.json')['params'];TAU_SYN=[None,PARAMS['tau_r_AMPA']+PARAMS['tau_d_AMPA'],PARAMS['tau_r_GABA']+PARAMS['tau_d_GABA']]

def load():
    rows=read(DEST/'dynamic_assay/rows.json')['rows'];pts={}
    for r in rows:
        key=(r['pop'],r['x'],r['sigma_E'],r['sigma_I']);pts.setdefault(key,{}).setdefault(r['channel'],{})[r['frequency_hz']]=(complex(*r['response']),r['sem'],r['rate_hz'])
    return pts

def shapes(chan,ch):
    f=np.array(sorted(chan));R=np.array([chan[q][0] for q in f]);sem=np.array([chan[q][1] for q in f]);R0=R[f==0][0].real;s0=sem[f==0][0]
    w=2j*np.pi*f/1000;H=R/R0
    if ch>0:H=H*(1+w*TAU_SYN[ch]/2)
    return f,H,sem/abs(R0),R0,s0

def fit(f,H,w_err,structure):
    w=2j*np.pi*f/1000;sel=f>0
    def model(p):
        if structure=='S1':return p[0]+(1-p[0])/(1+w*p[1])
        return p[0]/(1+w*p[1])+(1-p[0])/(1+w*p[2])
    def res(p):
        d=(model(p)-H)[sel]/np.maximum(w_err[sel],1e-3);return np.r_[d.real,d.imag]
    best=None
    starts=[[.3,3.],[.7,1.],[.1,10.],[.9,.5]] if structure=='S1' else [[.5,1.,10.],[.3,.5,20.],[.8,2.,30.],[.2,3.,8.]]
    bounds=([0,.05],[1,200.]) if structure=='S1' else ([0,.05,.05],[1,200.,400.])
    for x0 in starts:
        try:r=least_squares(res,x0,bounds=bounds)
        except Exception:continue
        if best is None or r.cost<best.cost:best=r
    p=best.x
    if structure=='S2' and p[1]>p[2]:p=np.array([1-p[0],p[2],p[1]])
    m=model(p);err=np.sqrt(np.mean(abs(m-H)[sel]**2));return p,float(err),float(abs(m-H)[sel].max())

def main(a):
    pts=load();out=[]
    for (pop,x,sE,sI),chans in sorted(pts.items()):
        row=dict(pop=pop,x=x,sigma_E=sE,sigma_I=sI,rate_hz=list(chans[0].values())[0][2],channels={})
        for ch in sorted(chans):
            f,H,we,R0,s0=shapes(chans[ch],ch);snr=abs(R0)/max(s0,1e-12)
            item=dict(dc=R0,dc_sem=s0,snr=snr,H=[[float(h.real),float(h.imag)] for h in H],frequencies=f.tolist(),norm_sem=we.tolist())
            if snr>=10:
                for st in ['S1','S2']:
                    p,err,mx=fit(f,H,we,st);item[st]=dict(params=p.tolist(),rms=err,max=mx)
            row['channels'][ch]=item
        # channel agreement: max |H_mean - H_var| at f<=40 where both have snr>=10
        Hm=row['channels'][0];agree={}
        for ch in (1,2):
            if ch in row['channels'] and row['channels'][ch]['snr']>=10 and Hm['snr']>=10:
                a_=np.array(Hm['H']);b_=np.array(row['channels'][ch]['H']);f=np.array(Hm['frequencies']);sel=(f>0)&(f<=40)
                agree[ch]=float(np.max(np.hypot(*(a_-b_)[sel].T)))
        row['channel_max_difference']=agree;out.append(row)
    write(DEST/'dynamic_assay/shapes_and_fits.json',dict(rows=out))
    # summary tables
    for pop in 'EI':
        rows=[r for r in out if r['pop']==pop]
        print('==== population',pop,len(rows),'points')
        for st in ['S1','S2']:
            for ch in (0,1,2):
                e=[r['channels'][ch][st]['rms'] for r in rows if ch in r['channels'] and st in r['channels'][ch]]
                if e:print(f'  {st} ch{ch}: n={len(e)} rms median {np.median(e):.3f} p90 {np.percentile(e,90):.3f} max {max(e):.3f}')
        d=[v for r in rows for v in r['channel_max_difference'].values()]
        if d:print(f'  channel shape max difference (mean vs variance, f<=40): median {np.median(d):.3f} p90 {np.percentile(d,90):.3f}')
        # parameter dependence on x for the mean channel, S1
        print('  mean channel S1 params (beta, tau) vs x at sigma_E=0.7, sigma_I=0.7:')
        for r in rows:
            if r['sigma_E']==.7 and r['sigma_I']==.7 and 'S1' in r['channels'][0]:
                p=r['channels'][0]['S1']['params'];q=r['channels'][0].get('S2',{}).get('params',[np.nan]*3)
                print(f"    x={r['x']:6.2f} rate={r['rate_hz']:7.2f} S1 beta={p[0]:.2f} tau={p[1]:6.2f} rms={r['channels'][0]['S1']['rms']:.3f} | S2 a={q[0]:.2f} tf={q[1]:.2f} ts={q[2]:.2f} rms={r['channels'][0].get('S2',{}).get('rms',np.nan):.3f}")
if __name__=='__main__':main(argparse.ArgumentParser().parse_args())
