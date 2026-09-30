"""Double-precision GPU linear algebra; CPU equations retain acceptance authority."""
import numpy as np
import torch
from topic4_fig5_z_cycle_preconditioner import MeanGainInverse


class GpuLinear:
    def __init__(self,o,jr,device=0):
        torch.set_num_threads(1);self.device=torch.device(f'cuda:{device}');self.o=o;self.m=o.m;self.U=o.U;self.N=o.N
        self.H={k:self.tensor(v) for k,v in o.cached[1].items()}
        self.le,self.li,self.lm=[self.tensor(x) for x in o.cached[2:5]]
        self.w=self.tensor(o.m.w_u);self.z=self.tensor(o.z);self.z2=self.tensor(o.z2)
        self.ge,self.gi=[[self.tensor(x) for x in row] for row in jr.transfer_gains]

    def tensor(self,x):return torch.as_tensor(np.ascontiguousarray(x),device=self.device)
    def filt(self,x,L):return torch.fft.irfft(torch.fft.rfft(x,dim=0)*L[:,None],n=self.N,dim=0)
    def conv(self,key,x):return torch.einsum('kij,kj->ki',self.H[key],x)
    def repeat(self,x):return torch.repeat_interleave(x,self.m.K,dim=1)

    def derivative_tensor(self,x):
        m=self.m;u=x[:,:self.U];i=x[:,self.U:];e=(u*self.w).reshape(self.N,m.n,m.K).sum(2)
        fe=torch.fft.rfft(e,dim=0);fi=torch.fft.rfft(i,dim=0)
        def timeconv(k,r):return torch.fft.irfft(self.conv(k,r),n=self.N,dim=0)
        mu=m.te*(self.repeat(timeconv('ee',fe))-self.z*self.repeat(timeconv('ei',fi)))-m.eta_M*self.filt(u,self.lm)
        ex=m.te*self.repeat(timeconv('vee',fe));inh=m.te*self.z2*self.repeat(timeconv('vei',fi))
        mui=m.ti*(timeconv('ie',fe)-timeconv('ii',fi));ei=m.ti*timeconv('vie',fe);ii=m.ti*timeconv('vii',fi)
        de=self.ge[0]*mu+self.ge[1]*ex+self.ge[2]*inh;di=self.gi[0]*mui+self.gi[1]*ei+self.gi[2]*ii
        return x-torch.cat([self.filt(de,self.le),self.filt(di,self.li)],dim=1)

    def __call__(self,x):return self.derivative_tensor(self.tensor(x)).cpu().numpy()

    def make_inverse(self,jr):
        inv=MeanGainInverse(self.o,*jr.transfer_gains)
        self.den=self.tensor(inv.den);self.gu=self.tensor(inv.gu);self.gv=self.tensor(inv.gv);self.gw=self.tensor(inv.gw)
        # Column-major matrices avoid a full batched LU copy in every solve.
        self.lu=self.tensor(np.array([a.T for a,p in inv.lus])).transpose(-1,-2)
        self.piv=torch.as_tensor(np.array([p+1 for a,p in inv.lus]),device=self.device,dtype=torch.int32)

    def inverse_tensor(self,x):
        m=self.m;q=torch.fft.rfft(x,dim=0);u=q[:,:self.U]/self.den
        macro=torch.cat([(u*self.w).reshape(len(self.le),m.n,m.K).sum(2),q[:,self.U:]],dim=1)
        sol=macro.clone();nf=len(self.lu)
        sol[:nf]=torch.linalg.lu_solve(self.lu,self.piv,macro[:nf,:,None]).squeeze(2);e,i=sol[:,:m.n],sol[:,m.n:]
        mean=self.repeat(self.conv('ee',e))-self.z*self.repeat(self.conv('ei',i))
        va=self.repeat(self.conv('vee',e));vg=self.z2*self.repeat(self.conv('vei',i))
        du=u+self.le[:,None]*m.te*(self.gu*mean+self.gv*va+self.gw*vg)/self.den
        return torch.fft.irfft(torch.cat([du,i],dim=1),n=self.N,dim=0)

    def bordered(self,jr,ft,fs,pp,ap,at,ass):
        self.make_inverse(jr);vt=self.inverse_tensor(self.tensor(ft));vs=self.inverse_tensor(self.tensor(fs));pp=self.tensor(pp);ap=self.tensor(ap)
        B=torch.stack([torch.stack([(vt*pp).sum(),(vs*pp).sum()]),torch.stack([(vt*ap).sum()-at,(vs*ap).sum()-ass])])
        inverse_border=self.tensor(np.linalg.inv(B.cpu().numpy()))
        def apply_tensor(x):
            r=self.inverse_tensor(x[:-2].reshape(self.N,-1))
            par=inverse_border@torch.stack([(r*pp).sum()-x[-2],(r*ap).sum()-x[-1]]);r=r-vt*par[0]-vs*par[1]
            return torch.cat([r.flatten(),par])
        def apply(rhs):return apply_tensor(self.tensor(rhs)).cpu().numpy()
        apply.tensor=apply_tensor
        return apply
