"""Float64 GPU evaluation of the frozen-D map, checked against its CPU updater.

The delay products are unchanged CSR matrices. No time step or physics changes.
"""
import numpy as np
import torch
from scipy import sparse
import siegert_table


class GPUMap:
    def __init__(self, eq, states, device=0, compile_step=True):
        self.m = m = eq.m
        self.device = f'cuda:{device}'
        self.B = len(states)
        assert all(s['pending'] is None and not s['freeze_m'] for s in states)
        self.rate_step_e = m.dt/float(m.tau_rate[0])
        self.rate_step_i = m.dt/float(m.tau_rate[-1])
        def tensor(x):
            return torch.as_tensor(np.asarray(x), dtype=torch.float64, device=self.device)
        self.tensor = tensor
        self.w = tensor(m.w_u)
        self.theta = tensor(m.theta_u)
        self.xx = tensor(m.gh_x)[None, :, None]
        self.ww = tensor(m.gh_w)[None, :, None]
        xs, gs = siegert_table.table()
        self.xs, self.gs = tensor(xs), tensor(gs)
        self.xlo, self.xdx = float(xs[0]), float(xs[1]-xs[0])
        self.a = tensor([m.arA,m.arG,m.arA,m.arG])[None,:,None]
        self.b = tensor([m.adA,m.adG,m.adA,m.adG])[None,:,None]
        self.amp = tensor([m.BAE,m.BGE,m.BAI,m.BGI])[None,:,None]
        self.extmean = tensor([m.je*m.nu_sig,0.,m.ji*m.nu_sig,0.])[None,:,None]
        self.extvar = tensor([m.je*m.je*m.nu_sig,0.,m.ji*m.ji*m.nu_sig,0.])[None,:,None]
        self.coeff = torch.stack([self.a*self.a,self.b*self.b,self.a*self.b],dim=2)
        self.norm = self.a*self.a/(1-self.a*self.a)+self.b*self.b/(1-self.b*self.b)-2*self.a*self.b/(1-self.a*self.b)
        self.r = tensor([s['r_u'] for s in states]);self.ri = tensor([s['r_i'] for s in states])
        self.M = tensor([s['m_u'] for s in states])
        self.z = tensor([s['z_u'] for s in states]);self.z2 = tensor([s['z2_u'] for s in states])
        self.g = tensor([[s[k] for k in ('gAE','gGE','gAI','gGI')] for s in states])
        self.c = tensor([[s[k] for k in ('cAE','cGE','cAI','cGI')] for s in states])
        self.y = tensor([[s['y'][k] for k in ('ee','ei','ie','ii')] for s in states])
        self.hE = tensor([s['hE'] for s in states]);self.hI = tensor([s['hI'] for s in states])
        self.original = states
        self.steps = 0
        zero = sparse.csr_matrix(m.ops['ee'].shape)
        rows = []
        for key in ('ee','ei','ie','ii'):
            mat = sparse.vstack([m.ops[key],m.vops[key]],format='csr')
            empty = sparse.vstack([zero,zero],format='csr')
            rows.append(sparse.hstack([mat,empty] if key in ('ee','ie') else [empty,mat],format='csr'))
        mat = sparse.vstack(rows,format='csr')
        self.matrix = torch.sparse_csr_tensor(torch.as_tensor(mat.indptr,device=self.device),
                                             torch.as_tensor(mat.indices,device=self.device),
                                             tensor(mat.data),size=mat.shape)
        self.update = torch.compile(self._update, fullgraph=True) if compile_step else self._update

    def primitive(self,x):
        i = torch.clamp(((x-self.xlo)/self.xdx).to(torch.int64),0,len(self.gs)-2)
        mid = self.gs[i]+(x-self.xs[i])/(self.xs[i+1]-self.xs[i])*(self.gs[i+1]-self.gs[i])
        low = self.gs[0]-(torch.log(torch.abs(x))-np.log(abs(self.xlo))-.25*(1/x**2-1/self.xlo**2))/np.sqrt(np.pi)
        high = self.gs[-1]+torch.exp(x*x)/x*(1+1/(2*x*x)+3/(4*x**4))-np.exp(26.**2)/26.*(1+1/(2*26.**2)+3/(4*26.**4))
        return torch.where(x<self.xlo,low,torch.where(x>26.,high,mid))

    def phi(self,mu,ex,inh,theta,tm,ref,w2):
        sigma = torch.sqrt(torch.clamp(ex,min=1e-12))
        sigg = torch.sqrt(torch.clamp(inh*w2,min=0.))
        shift = mu-1.0325*torch.sqrt(torch.clamp(ex*(self.m.ra+self.m.ta)/tm,min=1e-16))
        mean = shift[:,None,:]+np.sqrt(2.)*sigg[:,None,:]*self.xx
        integral = self.primitive((theta-mean)/sigma[:,None,:])-self.primitive((self.m.v_reset-mean)/sigma[:,None,:])
        den = ref+tm*np.sqrt(np.pi)*integral
        rates = torch.where(torch.isfinite(den)&(den>0),torch.clamp(1/den,max=1/ref),0.)
        return (self.ww*rates).sum(dim=1)

    def _update(self,r,ri,M,g,c,y,z,z2,products):
        m=self.m
        means = products[:,0].permute(2,0,1)
        seconds = products[:,1].permute(2,0,1)
        gn = self.a*g+self.amp*(means+self.extmean)
        cn = gn+(c-gn)*self.b
        yn = self.coeff*(y+(seconds+self.extvar)[:,:,None,:])
        v = (yn[:,:,0]+yn[:,:,1]-2*yn[:,:,2])/self.norm
        ex = m.te*v[:,0].repeat_interleave(m.K,dim=1)
        inh = m.te*z2*v[:,1].repeat_interleave(m.K,dim=1)
        mu = cn[:,0].repeat_interleave(m.K,dim=1)-z*cn[:,1].repeat_interleave(m.K,dim=1)-m.eta_M*M
        pe = self.phi(mu,ex,inh,self.theta,m.te,m.tref_e,m.w2cv_e)
        pi = self.phi(cn[:,2]-cn[:,3],m.ti*v[:,2],m.ti*v[:,3],m.theta_i,m.ti,m.tref_i,m.w2cv_i)
        rn = r+self.rate_step_e*(pe-r)
        rin = ri+self.rate_step_i*(pi-ri)
        Mn = M-m.dt/m.tau_M*M+m.dt*rn
        return rn,rin,Mn,gn,cn,yn

    def step(self):
        re = (self.r*self.w).reshape(self.B,self.m.n,self.m.K).sum(dim=2)
        oldri = self.ri
        histories = torch.cat([self.hE.reshape(self.B,-1),self.hI.reshape(self.B,-1)],dim=1).T
        products = torch.sparse.mm(self.matrix,histories).reshape(4,2,self.m.n,self.B)
        self.r,self.ri,self.M,self.g,self.c,self.y = self.update(self.r,self.ri,self.M,self.g,self.c,self.y,self.z,self.z2,products)
        self.hE = torch.cat([re[:,None,:],self.hE[:,:-1]],dim=1)
        self.hI = torch.cat([oldri[:,None,:],self.hI[:,:-1]],dim=1)
        self.steps += 1

    def state_dicts(self):
        arrays = dict(r_u=self.r,r_i=self.ri,m_u=self.M,z_u=self.z,z2_u=self.z2,hE=self.hE,hI=self.hI)
        for i,k in enumerate(('gAE','gGE','gAI','gGI')):arrays[k]=self.g[:,i]
        for i,k in enumerate(('cAE','cGE','cAI','cGI')):arrays[k]=self.c[:,i]
        arrays = {k:v.detach().cpu().numpy() for k,v in arrays.items()}
        ys = self.y.detach().cpu().numpy()
        result=[]
        for i,old in enumerate(self.original):
            state={k:v[i].copy() for k,v in arrays.items()}
            state.update(pending=None,freeze_m=False,step_count=old['step_count']+self.steps)
            state['y']={k:v.copy() for k,v in old['y'].items()}
            for j,key in enumerate(('ee','ei','ie','ii')):state['y'][key]=ys[i,j].copy()
            result.append(state)
        return result
