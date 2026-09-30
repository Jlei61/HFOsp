"""Candidate population-rate closure with one endogenous fast reset trace.

q is a causal exponentially decaying count per cell, not measured voltage.
It is driven only by the model's own firing during prediction.  This is a
local candidate, not a new accepted spatial model or a bifurcation result.
"""
from refractory_rate_response import *
from conditioned_refractory_rate import load_models as load_parent, DEST as PARENT
from nonlinear_rate_response import physical_from_features
from pathlib import Path
from datetime import datetime
import argparse,hashlib

DEST=OUT/'reset_memory_rate'


def reset_kernel(pop,frequency,dt=DT0):
    a=np.exp(-dt/PARAMS['tau_m_'+pop]);z=np.exp(-2j*np.pi*np.asarray(frequency)*dt/1000)
    return a*dt*z/(1-a*z)


@njit
def reset_trace(fired,dt,tau):
    """State before each flux: current and future counts cannot enter."""
    q=0.;a=np.exp(-dt/tau);out=np.empty(len(fired))
    for k in range(len(fired)):
        out[k]=q;q=a*(q+fired[k])
    return out


class ResetReadout(nn.Module):
    def __init__(self,pop):
        super().__init__();self.pop=pop;self.ref=PARAMS['tau_ref_'+pop]
        self.tau=PARAMS['tau_m_'+pop];self.maximum=1000/self.ref
        z=np.load(DEST/f'conditioning/{pop}.npz')
        self.register_buffer('center',torch.tensor(z['center'],dtype=torch.float32))
        self.register_buffer('transform',torch.tensor(z['transform'],dtype=torch.float32))
        self.network=nn.Sequential(nn.Linear(40,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))

    def correction(self,f):return self.network((f-self.center)@self.transform.T).squeeze(-1)

    def logits(self,f,b):return b+self.correction(f)

    def rate_given_q(self,f,b):
        """Stationary renewal flux for SUPPLIED q; not a solved equilibrium."""
        return self.maximum*torch.sigmoid(self.logits(f,b)+np.log(self.ref/DT0))

    def linear_at_equilibrium(self,f,b,bg,ig,channel,bank,cov,K,Kq,rate_hz,create_graph=False):
        """Exact discrete susceptibility when supplied rate satisfies residual.

        During fitting, the residual at training calibration rates is penalized
        separately. Final validation must solve the endogenous q equilibrium.
        """
        f=f.detach().requires_grad_(True);corr=self.correction(f)
        grad=torch.autograd.grad(corr.sum(),f,create_graph=create_graph)[0]
        p=torch.sigmoid(b+corr);i=torch.arange(len(f),device=f.device)
        direct=bg[i,channel]+grad[i,channel]*ig[i,channel]
        hist=grad[:,3:39].reshape(-1,3,12)[i,channel]
        G=(direct+(hist*bank).sum(1)*ig[i,channel])*cov[i,channel]
        q=torch.expm1(3*f[:,39]);ell_q=grad[:,39]/(3*(1+q))
        den=(1-p)+p*K/DT0-rate_hz/1000*(1-p)*ell_q*Kq
        return rate_hz*(1-p)*G/den


def register():
    DEST.mkdir(exist_ok=True);assert not (DEST/'contract.json').exists()
    write(DEST/'contract.json',dict(created_local=datetime.now().astimezone().isoformat(),status='REGISTERED_BEFORE_FIT',
        question='Can an endogenous membrane-timescale reset-count trace repair local transient response missing from input-only history plus absolute refractoriness?',
        evidence='Locked readout differs from independent local Gaussian LIF before native entry even with actual group mean current; most corresponding features are inside original calibration support.',
        interpretation='q is a phenomenological reset-memory coordinate, not exact conditional membrane voltage. Success is empirical and must be checked autonomously.',
        equation='q[k]=sum(j>=1,exp(-j*dt/tau_m)*fired[k-j]); q_next=exp(-dt/tau_m)*(q+fired); fired=available*sigmoid(ell+log(dt/0.1)); available from own past flux.',
        tau_ms=dict(E=20.,I=10.),features='Original39features plus log1p(q)/3. Original invertible39conditioning plus training-only scalar q centering/scaling. Same64/64tanh readout.',
        initialization='Exact embedding of locked conditioned39feature network with zero coefficient for new q input. No validation checkpoint selection.',
        training='Exact original480 training profiles and original static/linear calibration only. q label computed from past local-LIF counts; native trajectory, onset, spatial and validation labels excluded.',
        equilibrium='Static residual enforced at original static targets. Linear calibration at baseline measured mean rate also enforces stationary residual. Final susceptibility is only evaluated at solved own equilibrium, never assumes target q is self-consistent.',
        training_budget=dict(steps_per_pop=12000,threads=2,seed=920130,learning_rate=[[0,.001],[4000,.0003],[8000,.0001]],loss='10*conditional-flux KL + static asinh residual + normalized complex linear gain error + linear-workpoint asinh equilibrium residual'),
        validation='Original24 strong waveforms and original broad validation sets at dt0.1/0.05; unchanged waveform/mean/step gates. Matched282 finite-amplitude gain assay required if waveform gates pass; independent fresh64 required before any spatial promotion.',
        stop='One fixed fit, no schedule extension or architecture enlargement after scoring. Failure retained; do not launch network or bifurcation on local failure.',
        boundaries='This is a rate-equation candidate with one extra fast state; no individual particles in model; graph,Z/M,weights,delays and physical neuron parameters unchanged.'))


def prepare():
    assert (DEST/'contract.json').exists();folder=DEST/'training_arrays';folder.mkdir(exist_ok=True)
    cf=DEST/'conditioning';cf.mkdir(exist_ok=True)
    old=OUT/'nonlinear_rate_response';refr=OUT/'refractory_rate_response'
    from common import BASE
    linear=read(BASE/'dynamic_assay/rows.json')['rows']
    for pop in 'EI':
        path=folder/f'{pop}.npz';assert not path.exists()
        parent=np.load(refr/f'training_arrays/{pop}.npz');data={k:parent[k] for k in parent.files};qq=[];ids=[]
        for src,targets in [(old,OUT/'local_refractory_flux/training_data'),(refr,refr/'local_data')]:
            for row in read(src/'profiles.json')['rows']:
                if row['split']!='train' or row['pop']!=pop:continue
                z=np.load(targets/f'profile{row["id"]:03d}.npz');n=row['record_steps'];burn=row['burn_steps']
                choose=np.minimum(((np.arange(2048)+.5)*n/2048).astype(int),n-1)
                fired=z['spike_counts'].astype(float)/row['replicates'];q=reset_trace(fired,.1,PARAMS['tau_m_'+pop])
                assert np.max(abs(fired[burn:][choose]-data['flux_fired'][len(qq)*2048:(len(qq)+1)*2048]))<1e-7
                qq.append(q[burn:][choose]);ids.append(f'{src.name}/{row["id"]}')
        q=np.concatenate(qq);fq=np.log1p(q)/3
        data['flux_features']=np.column_stack([data['flux_features'],fq]).astype('f4')
        K0=float(reset_kernel(pop,[0])[0].real)
        sr=data['static_target'];data['static_features']=np.column_stack([data['static_features'],np.log1p(K0*sr/1000)/3]).astype('f4')
        rows=[r for r in linear if r['pop']==pop];dc={(r['x'],r['sigma_E'],r['sigma_I'],r['channel']):(r['response'][0],r['sem']) for r in rows if r['frequency_hz']==0}
        eligible=[r for r in rows if abs(dc[(r['x'],r['sigma_E'],r['sigma_I'],r['channel'])][0])/max(dc[(r['x'],r['sigma_E'],r['sigma_I'],r['channel'])][1],1e-15)>=10]
        lf=data['linear_features'];assert len(eligible)==len(lf)
        p=np.array([[11+7*r['x'],49*r['sigma_E']**2,49*r['sigma_I']**2] for r in eligible])
        assert np.max(abs(normalized_input(p)/SCALE-lf[:,:3]))<1e-6
        # Use the calibration's measured DC baseline, not an arbitrary AC mean.
        rate0={(r['x'],r['sigma_E'],r['sigma_I'],r['channel']):r['rate_hz'] for r in rows if r['frequency_hz']==0}
        lr=np.array([rate0[(r['x'],r['sigma_E'],r['sigma_I'],r['channel'])] for r in eligible])
        data['linear_rate_hz']=lr.astype('f4');data['linear_Kq']=reset_kernel(pop,[r['frequency_hz'] for r in eligible]).astype('c8')
        data['linear_features']=np.column_stack([lf,np.log1p(K0*lr/1000)/3]).astype('f4')
        z=np.load(PARENT/f'conditioning/{pop}.npz');center=np.r_[z['center'],fq.mean()];A=np.zeros((40,40));A[:39,:39]=z['transform'];A[39,39]=1/max(fq.std(),1e-4)
        np.savez_compressed(cf/f'{pop}.npz',center=center,transform=A)
        np.savez_compressed(path,**data)
        write(folder/f'{pop}_provenance.json',dict(status='TRAINING_ONLY',profiles=ids,rows=len(q),past_only_trace=True,validation_targets_used=False,native_firing_used=False,q_min=float(q.min()),q_max=float(q.max())))
        log('RESET MEMORY PREPARE',pop,len(q))


def embed(pop,parent):
    net=ResetReadout(pop).double();src=parent.network.layers
    with torch.no_grad():
        for i in [2,4]:net.network[i].load_state_dict(src[i].state_dict())
        net.network[0].weight[:,:39].copy_(src[0].weight);net.network[0].weight[:,39].zero_();net.network[0].bias.copy_(src[0].bias)
    return net


def train():
    import time,os
    c=read(DEST/'contract.json');assert read(DEST/'implementation_check.json')['status']=='PASS'
    torch.set_num_threads(c['training_budget']['threads']);folder=DEST/'fit';folder.mkdir(exist_ok=True);parents,_,_=load_parent()
    write(DEST/'jobs.json',dict(status='RUNNING',pid=os.getpid(),expected=2,completed=[]))
    for j,pop in enumerate('EI'):
        assert not (folder/f'{pop}_final.pt').exists();seed=c['training_budget']['seed']+j;torch.manual_seed(seed);rng=np.random.default_rng(seed)
        a=np.load(DEST/f'training_arrays/{pop}.npz');d={k:torch.as_tensor(a[k]) for k in a.files};net=embed(pop,parents[pop]).float();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-6)
        trace=[];start=time.time()
        for step in range(c['training_budget']['steps_per_pop']):
            if step in [4000,8000]:
                for g in opt.param_groups:g['lr']=.0003 if step==4000 else .0001
            wi=rng.integers(len(d['flux_fired']),size=512);si=rng.integers(len(d['static_target']),size=128);li=rng.integers(len(d['linear_target']),size=64)
            opt.zero_grad();ell=net.logits(d['flux_features'][wi],d['flux_logits'][wi]);lw=((d['flux_available'][wi]*torch.nn.functional.softplus(ell)-d['flux_fired'][wi]*ell-d['flux_entropy'][wi])/d['flux_norm'][wi]).mean()
            sr=net.rate_given_q(d['static_features'][si],d['static_logits'][si]);ls=(((torch.asinh(sr/.1)-torch.asinh(d['static_target'][si]/.1))/.1)**2).mean()
            eqr=net.rate_given_q(d['linear_features'][li],d['linear_logits'][li]);le=(((torch.asinh(eqr/.1)-torch.asinh(d['linear_rate_hz'][li]/.1))/.1)**2).mean()
            args=[d[k][li] for k in ['linear_features','linear_logits','linear_base_gradient','linear_input_gradient','linear_channel','linear_bank','linear_cov','linear_K','linear_Kq','linear_rate_hz']]
            gain=net.linear_at_equilibrium(*args,create_graph=True);ll=(abs((gain-d['linear_target'][li])/d['linear_norm'][li])**2).mean()
            loss=10*lw+ls+ll+le;assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),10);opt.step()
            if (step+1)%100==0:trace.append(dict(step=step+1,loss=float(loss.detach()),flux=float(lw.detach()),static=float(ls.detach()),linear=float(ll.detach()),linear_equilibrium=float(le.detach()),elapsed=time.time()-start))
            if (step+1)%1000==0:log('RESET MEMORY FIT',pop,trace[-1]);write(folder/f'{pop}_progress.json',dict(status='RUNNING',trace=trace))
        torch.save(dict(model=net.state_dict(),pop=pop,contract=c),folder/f'{pop}_final.pt');write(folder/f'{pop}_progress.json',dict(status='COMPLETE',trace=trace))
        jobs=read(DEST/'jobs.json');jobs['completed'].append(pop);write(DEST/'jobs.json',jobs)
    write(folder/'locked_weights.json',dict(status='FINAL_WEIGHTS_LOCKED_BEFORE_VALIDATION',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*_final.pt')},conditioning={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (DEST/'conditioning').glob('*.npz')},validation_scored=False))
    jobs['status']='COMPLETE';write(DEST/'jobs.json',jobs)


def load_models():
    locked=read(DEST/'fit/locked_weights.json');assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==locked['source_sha256'];nets={};bases={}
    for pop in 'EI':
        p=DEST/f'fit/{pop}_final.pt';assert hashlib.sha256(p.read_bytes()).hexdigest()==locked['files'][p.name]
        q=DEST/f'conditioning/{pop}.npz';assert hashlib.sha256(q.read_bytes()).hexdigest()==locked['conditioning'][q.name]
        n=ResetReadout(pop).double();n.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['model']);n.eval();nets[pop]=n;bases[pop]=BaseLogit(pop)
    return nets,bases,locked


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','prepare','train']);a=p.parse_args();{'register':register,'prepare':prepare,'train':train}[a.command]()
