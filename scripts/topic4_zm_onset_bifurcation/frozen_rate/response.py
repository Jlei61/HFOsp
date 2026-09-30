"""Frequency response from the paper's Eqs.36--47, without an assumed rate tau.

The hypergeometric U is evaluated by its convergent Taylor recurrence
U''=2 y U'+2 z U. Ratios cancel its arbitrary normalization. Static limits
are matched to derivatives of the SAME colored stationary transfer, including
the dependence of effective correlation time on AMPA/GABA variances.

Table-2 corrections were fitted by the authors for their operating points;
their extension to heterogeneous core thresholds is tested as an approximation,
and is never silently interpreted as an exact native-SNN susceptibility.
"""
from common import *
from scipy.special import loggamma,erfcx
from numba import njit

@njit(cache=True)
def ratios(lo,hi,z,c0,c1):
    nmax=180;out=np.empty((2,len(lo)),np.complex128)
    for p in range(len(lo)):
        coeff=np.zeros(nmax+1,np.complex128);coeff[0]=c0[p];coeff[1]=c1[p]
        for n in range(nmax-1):coeff[n+2]=2*(n+z[p])*coeff[n]/((n+2)*(n+1))
        # Evaluate U-c0 and its derivatives, avoiding subtraction of the DC constant.
        vals=np.zeros((2,3),np.complex128)
        for side in range(2):
            y=lo[p] if side==0 else hi[p];a=0j;b=0j;c=0j
            for n in range(nmax,0,-1):
                a=a*y+coeff[n];b=b*y+n*coeff[n]
                if n>=2:c=c*y+n*(n-1)*coeff[n]
            vals[side,0]=a*y;vals[side,1]=b;vals[side,2]=c
        den=vals[1,0]-vals[0,0]
        out[0,p]=(vals[1,1]-vals[0,1])/den/(1+z[p])
        out[1,p]=(vals[1,2]-vals[0,2])/den/(2+z[p])
    return out

def white(lam,lo,hi,sigma,tm,rate):
    z=lam*tm
    c0=np.exp(-loggamma((1+z)/2));c1=2*np.exp(-loggamma(z/2))
    rr=ratios(lo,hi,z,c0,c1)
    return np.array([rate/sigma*rr[0],rate/sigma**2*rr[1]])

def normalized_white(lam,details,tm):
    rate,lo,hi,sigma,teff=details
    if abs(lam)<1e-10:return np.ones((2,len(rate)),complex)
    # Divide by the actual lambda->0 formula, evaluated independently, so
    # refractory/DC conventions cannot silently change equilibrium gains.
    dc=white(1e-8+0j,lo,hi,sigma,tm,rate).real
    return white(complex(lam),lo,hi,sigma,tm,rate)/dc

def paper_correction(freq_response,omega,details,pop):
    rate,lo,hi,sigma,teff=details;answer=freq_response.copy();x=abs(omega)*teff
    if abs(omega)<1e-12:return answer
    bad=np.zeros((2,len(rate)),bool)
    for kind in (0,1):
        for p in (0,1):
            take=pop==p;xx=x[take];r=rate[take]*1000;ss=sigma[take]
            magnitude=abs(freq_response[kind,take])*1000
            angle=np.angle(freq_response[kind,take])
            if kind==0:
                a,b,c=([1.807,2.1096,-.2989] if p==0 else [2.0571,4.2311,-.5295])
                high=1.3238*r/ss*np.sqrt(teff[take]/(20. if p==0 else 10.))
                corrected=magnitude+xx**a/(b+xx**a)*(c/xx+high)
                a,b=([1.0970,1.9558] if p==0 else [1.1220,1.5426])
                phase=angle*(1-xx**a/(b+xx**a))
            else:
                a,b,c,d=([2.3585,1.4783,-.5842,.4065] if p==0 else [2.6043,3.9899,-.9964,.3834])
                corrected=magnitude-xx**a/(b+xx**a)*(c/xx+d)
                a,b,c=([-1.8830,.7653,.3683] if p==0 else [-2.7212,.7712,.2490])
                # The paper's phase is a positive lag; numpy angle is negative.
                phase=b*(angle+np.sign(omega)*c*xx**a/(1+xx**a))
            bad[kind,take]=corrected<0
            answer[kind,take]=corrected/1000*np.exp(1j*phase)
    return answer,bad

def susceptibility(model,r,J,lam,method='shifted_white'):
    moments=model.moments(r,J);details=model.phi(*moments,details=True)
    gains=model.gains(moments)
    norm=normalized_white(lam,details,model.tm)
    if method=='paper' and abs(lam.imag)>1e-10:
        iw=1j*lam.imag;at_iw=normalized_white(iw,details,model.tm)
        # Original correction is defined for total variance. Its spectral
        # shape is shared by the two derivatives; their individual DC gains
        # still account for unequal synaptic correlation times.
        f=details[0];sigma=details[3]
        raw=white(iw,*details[1:4],model.tm,f)
        corrected,bad=paper_correction(raw,lam.imag,details,model.geo['population'])
        if bad.any():raise ValueError(f'Paper Table-2 response extrapolates to negative amplitudes in {bad.sum()} groups')
        norm*=corrected/raw
    return gains[0]*norm[0],gains[1]*norm[1],gains[2]*norm[1]

def characteristic(model,r,J,lam,method='shifted_white'):
    a,b,qa,qb=model.matrices(J,lam);gm,ge,gi=susceptibility(model,r,J,lam,method)
    ha=1/((1+lam*model.rise[0])*(1+lam*model.decay[0]))
    hg=1/((1+lam*model.rise[1])*(1+lam*model.decay[1]))
    hqa=2/(2+lam*model.tau[0]);hqg=2/(2+lam*model.tau[1])
    K=sparse.diags(gm*model.tm)@(model.area[0]*ha*a-sparse.diags(model.Z*model.area[1]*hg)@b)
    K+=sparse.diags(ge*model.tm*model.area[0]**2*hqa)@qa+sparse.diags(gi*model.tm*(model.Z*model.area[1])**2*hqg)@qb
    return sparse.diags(1+.5*model.E*gm/(1+lam*1000))-K
