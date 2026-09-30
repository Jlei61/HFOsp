"""Local rate candidate with input-weighted refractory/reset memory.

This is an approximate population mean-voltage coordinate, not an exact
voltage density. It retains the input x refractory-occupancy term absent
from the failed reset-count-only candidate. No native onset targets are fit.
"""
from refractory_rate_response import *
from common import BASE
from conditioned_refractory_rate import load_models as load_parent, DEST as PARENT
from nonlinear_rate_response import physical_from_features
from pathlib import Path
from datetime import datetime
import argparse, hashlib, time, os

DEST = OUT/'voltage_memory_rate'
VR = 11.


@njit
def voltage_trace(mu, fired, dt, tau, ref, theta=18.):
    nref = round(ref/dt); hist = np.zeros(nref); occupied = 0.; v = VR
    a = np.exp(-dt/tau); out = np.empty(len(mu))
    for k in range(len(mu)):
        slot = k % nref; occupied -= hist[slot]
        available = 1-occupied
        if available < -1e-9 or fired[k] > available+1e-9:
            raise ValueError('Invalid training population mass')
        vpre = a*v+(1-a)*(VR+(mu[k]-VR)*available)
        out[k] = vpre; v = vpre-(theta-VR)*fired[k]
        hist[slot] = fired[k]; occupied += fired[k]
    return out


def voltage_factors(pop, frequencies, mu, rate_hz, theta=18., dt=DT0):
    """Exact linear factors of the discrete voltage-memory recurrence.

    dVpre = U_mu*dmu + U_rate*dr, with dr in spikes/ms/cell.
    The rate readout must close this feedback, not treat Vpre as supplied.
    """
    freq = np.asarray(frequencies); mu = np.asarray(mu); rate = np.asarray(rate_hz)/1000
    tau = PARAMS['tau_m_'+pop]; ref = PARAMS['tau_ref_'+pop]
    a = np.exp(-dt/tau); z = np.exp(-2j*np.pi*freq*dt/1000)
    nref = round(ref/dt)
    past = dt*np.exp(-2j*np.pi*freq[:,None]*dt/1000*np.arange(1,nref)[None,:]).sum(1)
    u_mu = (1-a)*(1-(ref-dt)*rate)/(1-a*z)
    u_rate = -((1-a)*(mu-VR)*past+a*(theta-VR)*dt*z)/(1-a*z)
    return u_mu, u_rate


def stationary_voltage(pop, mu, rate_hz, theta=18., dt=DT0):
    a = np.exp(-dt/PARAMS['tau_m_'+pop]); ref = PARAMS['tau_ref_'+pop]
    return mu-((mu-VR)*(ref-dt)+a*(theta-VR)*dt/(1-a))*np.asarray(rate_hz)/1000


def voltage_feature(vpre, theta=18.):
    return np.arcsinh((vpre-VR)/(theta-VR))/3


class VoltageReadout(nn.Module):
    def __init__(self,pop):
        super().__init__(); self.pop=pop; self.ref=PARAMS['tau_ref_'+pop]
        self.tau=PARAMS['tau_m_'+pop]; self.maximum=1000/self.ref
        z=np.load(DEST/f'conditioning/{pop}.npz')
        self.register_buffer('center',torch.tensor(z['center'],dtype=torch.float32))
        self.register_buffer('transform',torch.tensor(z['transform'],dtype=torch.float32))
        self.network=nn.Sequential(nn.Linear(40,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))

    def correction(self,f): return self.network((f-self.center)@self.transform.T).squeeze(-1)
    def logits(self,f,b): return b+self.correction(f)
    def rate_given_voltage(self,f,b):
        return self.maximum*torch.sigmoid(self.logits(f,b)+np.log(self.ref/DT0))

    def linear_at_equilibrium(self,f,b,bg,ig,channel,bank,cov,K,Umu,Ur,rate_hz,create_graph=False):
        f=f.detach().requires_grad_(True); corr=self.correction(f)
        grad=torch.autograd.grad(corr.sum(),f,create_graph=create_graph)[0]
        p=torch.sigmoid(b+corr); index=torch.arange(len(f),device=f.device)
        direct=bg[index,channel]+grad[index,channel]*ig[index,channel]
        hist=grad[:,3:39].reshape(-1,3,12)[index,channel]
        G=(direct+(hist*bank).sum(1)*ig[index,channel])*cov[index,channel]
        ell_v=grad[:,39]/(21*torch.cosh(3*f[:,39]))  # calibration theta=18, reset=11
        G=G+ell_v*Umu*(channel==0)
        den=(1-p)+p*K/DT0-rate_hz/1000*(1-p)*ell_v*Ur
        return rate_hz*(1-p)*G/den


def register():
    DEST.mkdir(exist_ok=True); assert not (DEST/'contract.json').exists()
    assert read(OUT/'reset_memory_rate/validation/result.json')['status']=='LOCAL_CANDIDATE_FAIL'
    assert read(OUT/'refractory_current_closure/independent_audit.json')['status']=='REFRACTORY_CURRENT_INDEPENDENT_AUDIT_PASS'
    write(DEST/'contract.json',dict(created_local=datetime.now().astimezone().isoformat(),
        status='REGISTERED_BEFORE_PREPARATION_AND_FIT',
        question='Does input-weighted refractory clamping memory, missing from the failed count-only reset trace, repair strong-input recovery without changing graph or Z/M?',
        evidence='Exact local LIF membrane accounting found large accumulated refractory-clamp contributions. Count-only reset trace failed. This candidate specifically adds the previously omitted input times refractory-occupancy contribution.',
        equation='available=1-sum previous nref-1 fired; Vpre=a*Vpost+(1-a)*(Vr+(mu-Vr)*available); own fired=available*sigmoid(ell+log(dt/0.1)); Vpost=Vpre-(theta-Vr)*fired.',
        continuous_limit='tau_m*dV/dt=-V+mu-(mu-Vr)*P_ref-tau_m*(theta-Vr)*r; P_ref=integral over the absolute refractory interval of own r.',
        approximation='Factorizes refractory current with occupied fraction, and replaces reset charge by threshold-minus-reset times flux. Prior diagnostics quantify both missing conditional-current covariance and finite-step overshoot; coordinate is not claimed to be exact mean voltage.',
        physical=dict(tau_m_ms=dict(E=20.,I=10.),reset_mV=11.,ref_ms=dict(E=2.,I=1.)),
        features='Original39 input features plus asinh((Vpre-Vr)/(theta-Vr))/3. Original39 conditioning plus training-only scalar centering/scaling; same64/64tanh network.',
        initialization='Exact embedding of frozen conditioned39 network with zero voltage coefficient.',
        training='Same480 original local-LIF training profiles and original static/linear calibration. Voltage-memory features from strictly past observed training counts and current prescribed mean. No native firing, onset, spatial targets or validation targets in training.',
        budget=dict(steps_per_pop=12000,threads=2,seed=920150,schedule=[[0,.001],[4000,.0003],[8000,.0001]],
            loss='10*conditional-flux KL + static asinh residual + normalized complex gain residual + linear-workpoint equilibrium residual'),
        checks='Parent embedding, causal voltage recurrence, static formula and nonzero closed voltage-feedback susceptibility against independent direct integration before training.',
        validation='Original24 plus three reused64 sets, dt0.1/0.05, original wave/mean/step gates; all predictions autonomous in own flux and V. If passed, fresh validation and matched DC/AC required before spatial use.',
        stop='One fixed fit; no architecture/loss/schedule changes after validation. No new spatial run or bifurcation on local failure.',
        scope='Local rate-response repair only. Same continuous-rate objective; no membrane particles, physical parameter tuning or model promotion. Existing registered spatial pair proceeds unchanged.'))


def prepare():
    assert (DEST/'contract.json').exists(); folder=DEST/'training_arrays'; folder.mkdir(exist_ok=True)
    cf=DEST/'conditioning'; cf.mkdir(exist_ok=True)
    old=OUT/'nonlinear_rate_response'; refr=OUT/'refractory_rate_response'
    linear=read(BASE/'dynamic_assay/rows.json')['rows']
    for pop in 'EI':
        path=folder/f'{pop}.npz'; assert not path.exists()
        parent=np.load(OUT/f'reset_memory_rate/training_arrays/{pop}.npz')
        data={k:parent[k] for k in parent.files if k!='linear_Kq'}; vv=[]; ids=[]
        for src,targets in [(old,OUT/'local_refractory_flux/training_data'),(refr,refr/'local_data')]:
            waves=np.load(src/'prepared.npz')['wave']
            for row in read(src/'profiles.json')['rows']:
                if row['split']!='train' or row['pop']!=pop: continue
                z=np.load(targets/f'profile{row["id"]:03d}.npz'); n=row['record_steps']; burn=row['burn_steps']
                fired=z['spike_counts'].astype(float)/row['replicates']; assert len(fired)==burn+n
                wave=waves[row['id']]; T=row['period_ms']; W=wave.shape[1]
                pos=(((np.arange(len(fired))-burn+1)*.1/T)%1)*W
                lo=np.floor(pos).astype(int)%W; hi=(lo+1)%W; alpha=pos-np.floor(pos)
                mu=(1-alpha)*wave[0,lo]+alpha*wave[0,hi]
                choose=np.minimum(((np.arange(2048)+.5)*n/2048).astype(int),n-1)
                start=len(vv)*2048; fs=data['flux_features'][start:start+2048,:39]
                exact_mu=physical_from_features(fs)[:,0]
                assert np.max(abs(exact_mu-mu[burn:][choose]))<.002
                assert np.max(abs(fired[burn:][choose]-data['flux_fired'][start:start+2048]))<1e-7
                v=voltage_trace(mu,fired,.1,PARAMS['tau_m_'+pop],PARAMS['tau_ref_'+pop])
                vv.append(v[burn:][choose]); ids.append(f'{src.name}/{row["id"]}')
        v=np.concatenate(vv); vf=voltage_feature(v)
        data['flux_features'][:,39]=vf.astype('f4')
        smu=physical_from_features(data['static_features'][:,:39])[:,0]
        data['static_features'][:,39]=voltage_feature(stationary_voltage(pop,smu,data['static_target'])).astype('f4')
        rows=[r for r in linear if r['pop']==pop]
        dc={(r['x'],r['sigma_E'],r['sigma_I'],r['channel']):(r['response'][0],r['sem']) for r in rows if r['frequency_hz']==0}
        eligible=[r for r in rows if abs(dc[(r['x'],r['sigma_E'],r['sigma_I'],r['channel'])][0])/max(dc[(r['x'],r['sigma_E'],r['sigma_I'],r['channel'])][1],1e-15)>=10]
        assert len(eligible)==len(data['linear_features'])
        mu=np.array([11+7*r['x'] for r in eligible]); freq=np.array([r['frequency_hz'] for r in eligible]); rate=data['linear_rate_hz']
        assert abs(mu-physical_from_features(data['linear_features'][:,:39])[:,0]).max()<.002
        data['linear_features'][:,39]=voltage_feature(stationary_voltage(pop,mu,rate)).astype('f4')
        a,b=voltage_factors(pop,freq,mu,rate); data['linear_Umu']=a.astype('c8'); data['linear_Urate']=b.astype('c8')
        z=np.load(PARENT/f'conditioning/{pop}.npz'); center=np.r_[z['center'],vf.mean()]
        A=np.zeros((40,40)); A[:39,:39]=z['transform']; A[39,39]=1/max(vf.std(),1e-4)
        np.savez_compressed(cf/f'{pop}.npz',center=center,transform=A)
        np.savez_compressed(path,**data)
        write(folder/f'{pop}_provenance.json',dict(status='TRAINING_ONLY',profiles=ids,rows=len(v),
            causal=True,validation_targets_used=False,native_firing_used=False,Vpre_range_mV=[float(v.min()),float(v.max())]))
        log('VOLTAGE MEMORY PREPARE',pop,len(v))


def embed(pop,parent):
    net=VoltageReadout(pop).double(); src=parent.network.layers
    with torch.no_grad():
        for i in [2,4]: net.network[i].load_state_dict(src[i].state_dict())
        net.network[0].weight[:,:39].copy_(src[0].weight); net.network[0].weight[:,39].zero_()
        net.network[0].bias.copy_(src[0].bias)
    return net


def train():
    c=read(DEST/'contract.json'); assert read(DEST/'implementation_check.json')['status']=='PASS'
    torch.set_num_threads(c['budget']['threads']); folder=DEST/'fit'; folder.mkdir(exist_ok=True)
    parents,_,_=load_parent(); assert not (DEST/'jobs.json').exists()
    write(DEST/'jobs.json',dict(status='RUNNING',pid=os.getpid(),expected=2,completed=[]))
    for j,pop in enumerate('EI'):
        assert not (folder/f'{pop}_final.pt').exists(); seed=c['budget']['seed']+j
        torch.manual_seed(seed); rng=np.random.default_rng(seed)
        a=np.load(DEST/f'training_arrays/{pop}.npz'); d={k:torch.as_tensor(a[k]) for k in a.files}
        net=embed(pop,parents[pop]).float(); opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-6)
        trace=[]; start=time.time()
        for step in range(c['budget']['steps_per_pop']):
            if step in [4000,8000]:
                for g in opt.param_groups: g['lr']=.0003 if step==4000 else .0001
            wi=rng.integers(len(d['flux_fired']),size=512); si=rng.integers(len(d['static_target']),size=128); li=rng.integers(len(d['linear_target']),size=64)
            opt.zero_grad(); ell=net.logits(d['flux_features'][wi],d['flux_logits'][wi])
            lw=((d['flux_available'][wi]*torch.nn.functional.softplus(ell)-d['flux_fired'][wi]*ell-d['flux_entropy'][wi])/d['flux_norm'][wi]).mean()
            sr=net.rate_given_voltage(d['static_features'][si],d['static_logits'][si]); ls=(((torch.asinh(sr/.1)-torch.asinh(d['static_target'][si]/.1))/.1)**2).mean()
            eqr=net.rate_given_voltage(d['linear_features'][li],d['linear_logits'][li]); le=(((torch.asinh(eqr/.1)-torch.asinh(d['linear_rate_hz'][li]/.1))/.1)**2).mean()
            args=[d[k][li] for k in ['linear_features','linear_logits','linear_base_gradient','linear_input_gradient','linear_channel','linear_bank','linear_cov','linear_K','linear_Umu','linear_Urate','linear_rate_hz']]
            gain=net.linear_at_equilibrium(*args,create_graph=True); ll=(abs((gain-d['linear_target'][li])/d['linear_norm'][li])**2).mean()
            loss=10*lw+ls+ll+le; assert torch.isfinite(loss); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),10); opt.step()
            if (step+1)%100==0: trace.append(dict(step=step+1,loss=float(loss.detach()),flux=float(lw.detach()),static=float(ls.detach()),linear=float(ll.detach()),linear_equilibrium=float(le.detach()),elapsed=time.time()-start))
            if (step+1)%1000==0: log('VOLTAGE MEMORY FIT',pop,trace[-1]); write(folder/f'{pop}_progress.json',dict(status='RUNNING',trace=trace))
        torch.save(dict(model=net.state_dict(),pop=pop,contract=c),folder/f'{pop}_final.pt')
        write(folder/f'{pop}_progress.json',dict(status='COMPLETE',trace=trace))
        jobs=read(DEST/'jobs.json'); jobs['completed'].append(pop); write(DEST/'jobs.json',jobs)
    write(folder/'locked_weights.json',dict(status='FINAL_WEIGHTS_LOCKED_BEFORE_VALIDATION',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*_final.pt')},
        conditioning={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (DEST/'conditioning').glob('*.npz')},validation_scored=False))
    jobs['status']='COMPLETE'; write(DEST/'jobs.json',jobs)


def load_models():
    locked=read(DEST/'fit/locked_weights.json'); assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==locked['source_sha256']
    nets={}; bases={}; torch.set_num_threads(2)
    for pop in 'EI':
        p=DEST/f'fit/{pop}_final.pt'; assert hashlib.sha256(p.read_bytes()).hexdigest()==locked['files'][p.name]
        q=DEST/f'conditioning/{pop}.npz'; assert hashlib.sha256(q.read_bytes()).hexdigest()==locked['conditioning'][q.name]
        n=VoltageReadout(pop).double(); n.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['model']); n.eval()
        nets[pop]=n; bases[pop]=BaseLogit(pop)
    return nets,bases,locked


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['register','prepare','train']); a=p.parse_args()
    {'register':register,'prepare':prepare,'train':train}[a.command]()
