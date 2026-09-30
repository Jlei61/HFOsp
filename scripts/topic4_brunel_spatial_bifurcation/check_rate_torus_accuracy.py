"""Off-grid defect of the same DDE on a denser two-angle torus mesh.

Split both Nyquist harmonics before recovering continuous input moments.
Apply the nonlinear transfer function on the denser mesh, then all rate
filters at every generated frequency. No additional spatial aggregation.
"""
from rate_periodic import *


def check(path,factor=2,device=0,batch=128):
    s=RateField();z=np.load(path);r=z['r'];nt,np_,P=r.shape;T=float(z['T']);nu=float(z['nu']);J=float(z['J'])
    coeff=np.fft.fftshift(np.fft.fft2(r,axes=(0,1)),axes=(0,1))/(nt*np_)
    expanded=np.empty((nt+1,np_+1,P),complex);expanded[:nt,:np_]=coeff
    expanded[-1,:np_]=coeff[0];expanded[:nt,-1]=coeff[:,0];expanded[-1,-1]=coeff[0,0]
    expanded[[0,-1]]*=.5;expanded[:,[0,-1]]*=.5
    kt=np.arange(-nt//2,nt//2+1);kp=np.arange(-np_//2,np_//2+1)
    lam=1j*(kt[:,None]*2*np.pi/T+kp[None,:]*nu).ravel()[:,None]
    K=len(lam);batch=min(batch,K);o=Periodic(s,2*(batch-1),device);cp=o.cp
    rf=cp.asarray(expanded.reshape(K,P));ll=cp.asarray(lam)
    arr=[]
    for k,(d,mask,index,ptr) in enumerate(o.raw):
        scale=cp.where(mask,J**(1 if k==0 else 2),1.) if k in [0,2] else 1.
        output=cp.empty_like(rf);edges=d.shape[0]
        for start in range(0,K,batch):
            stop=min(start+batch,K);n=stop-start
            phase=cp.exp(-cp.asarray(s.delays)[:,None]*ll[start:stop,0])
            data=(d@phase).T.copy();data*=scale
            op=o.cs.csr_matrix((data.ravel(),index[:n*edges],ptr[:n*P+1]),shape=(n*P,n*P))
            output[start:stop]=(op@rf[start:stop].ravel()).reshape(n,P)
            del op,data,phase
        arr.append(output)
        cp.get_default_memory_pool().free_all_blocks()
    a,b,qa,qb=arr;tm,ref,th,alpha,tf,ts,E=o.gpars
    ha=1/((1+ll*s.rise[0])*(1+ll*s.decay[0]));hg=1/((1+ll*s.rise[1])*(1+ll*s.decay[1]))
    va=1/(1+ll*s.tau[0]/2);vg=1/(1+ll*s.tau[1]/2);m=1/(1+1000*ll)
    cf=[tm*(s.area[0]*ha*a-s.area[1]*hg*b)-.5*E*m*rf,
        tm*s.area[0]**2*va*qa,tm*s.area[1]**2*vg*qb]
    shape=(nt*factor,np_*factor);ii=cp.asarray(kt%shape[0]);jj=cp.asarray(kp%shape[1]);size=np.prod(shape)
    def interpolate(c):
        grid=cp.zeros((*shape,P),complex)
        grid[ii[:,None],jj[None,:]]=c.reshape(nt+1,np_+1,P)*size
        return cp.fft.ifft2(grid,axes=(0,1)).real
    rr=interpolate(rf);mom=cp.stack([interpolate(c).reshape(size,P) for c in cf])+o.private[:,None,:]
    o.N=size;phi=o.phi(mom).reshape(*shape,P)
    freq=1j*(cp.fft.fftfreq(shape[0])[:,None]*shape[0]*2*np.pi/T+
        cp.fft.fftfreq(shape[1])[None,:]*shape[1]*nu)[:,:,None]
    H=alpha/(1+freq*tf)+(1-alpha)/(1+freq*ts)
    expected=cp.fft.ifft2(cp.fft.fft2(phi,axes=(0,1))*H,axes=(0,1)).real
    defect=(rr-expected).reshape(size,P)*1000;regional=[]
    for region in range(3):
        mask=s.E&(s.geo['group_region']==region);w=cp.asarray(s.geo['group_size'][mask]);w/=w.sum()
        regional.append(float(cp.max(cp.abs(defect[:,mask]@w))))
    out=dict(source=str(path),J_EE_core=J,T_ms=T,modulation_period_ms=2*np.pi/nu,
        original_mesh=[nt,np_],check_mesh=list(shape),maximum_group_defect_Hz=float(cp.max(cp.abs(defect))),
        regional_defect_Hz=regional,minimum_rate_Hz=float(cp.min(rr)*1000),
        harmonic_batch=batch,
        method='Independent two-angle Fourier interpolation with both Nyquist splits, exact physical frequencies/delays, denser nonlinear transfer evaluation.')
    folder=PERIODIC_OUT/'torus_accuracy';folder.mkdir(exist_ok=True)
    write(folder/f'{Path(path).stem}_factor{factor}_batch{batch}.json',out);print('TORUS CONTINUOUS DEFECT',out,flush=True)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('path');p.add_argument('--factor',type=int,default=2);p.add_argument('--device',type=int,default=0)
    a=p.parse_args();check(a.path,a.factor,a.device)
