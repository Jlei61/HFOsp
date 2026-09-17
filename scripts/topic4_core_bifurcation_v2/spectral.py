"""Chebyshev generator spectrum, refined with the exact delay characteristic.

Only six filtered-rate histories require collocation; present rates and first
synaptic filters add 12 variables. This retains all stable filter modes.
"""
from model import System,OUT
from eigen import root_at
import json,numpy as np
from scipy.linalg import eig,eigvals

def generator(s,r,g,N=40):
    x=np.cos(np.arange(N+1)*np.pi/N);theta=(x-1)*s.delay[-1]/2
    bw=(-1.)**np.arange(N+1);bw[[0,-1]]*=.5
    dx=theta[:,None]-theta[None,:];np.fill_diagonal(dx,1.)
    D=(bw[None,:]/bw[:,None])/dx;np.fill_diagonal(D,0.);np.fill_diagonal(D,-D.sum(1))
    interp=np.empty((len(s.delay),N+1))
    for k,d in enumerate(s.delay):
        dist=-d-theta;j=np.argmin(abs(dist))
        if abs(dist[j])<1e-10:interp[k]=0.;interp[k,j]=1.
        else:
            a=bw/dist;interp[k]=a/a.sum()
    W,_=s.weights(g);u,_,var=s.blocks(r,g)
    G=np.zeros((12+6*(N+1),)*2)
    G[:6,:6]=(var-np.eye(6))/s.tr[:,None]
    G[6:12,:6]=np.diag(1/s.rise);G[6:12,6:12]=-np.diag(1/s.rise)
    G[12:18,6:12]=np.diag(1/s.decay);G[12:18,12:18]=-np.diag(1/s.decay)
    C=np.einsum('dn,dij->nij',interp,W)*((u*s.tm/s.tr)[None,:,None])*(s.area*s.sign)[None,None,:]
    for k in range(N+1):G[:6,12+6*k:18+6*k]=C[k]
    G[18:,12:]=np.kron(D[1:],np.eye(6))
    return G

def leading(s,r,g,N=40):
    vals=eigvals(generator(s,r,g,N))*1000
    return vals[np.argsort(vals.real)[::-1]]

def main():
    s=System();arc=json.loads((OUT/'equilibrium_arc.json').read_text());rows=[]
    # Stop before the second core folds, so the branch follows A onset with B low.
    sel=[x for x in arc if x['direction']==-1 or (x['direction']==1 and x['step']<=137)]
    for i,row in enumerate(sel):
        r=np.array(row['r_hz'])/1000;g=row['g'];ev=leading(s,r,g)
        row.update(leading_real_per_s=float(ev[0].real),leading_freq_hz=float(abs(ev[0].imag)/2/np.pi),unstable_count=int((ev.real>1e-6).sum()),
            spectrum_per_s=[[float(v.real),float(v.imag)] for v in ev[:24]])
        rows.append(row)
        if i%15==0:print(i,g,r[0]*1000,ev[:4],flush=True)
    (OUT/'equilibrium_spectrum.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
