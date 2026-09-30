"""Finite continuous filter states with a smooth LIF-calibrated rate readout.

This is a candidate response approximation, not an accepted spatial model.
Its inputs are mean current and RAW AMPA/GABA diffusion variance drives.
The transfer table is only a baseline; fitted residuals alter its DC gains.
"""
from common import OUT, BASE, np, read, write, log
from transfer_spline import TransferSpline
from scipy.linalg import expm
from numba import njit
import torch
from torch import nn
from datetime import datetime
import hashlib

DEST = OUT / 'nonlinear_rate_response'
TAUS = np.array([1., 4., 16., 64.])
SCALE = np.array([3., 2., 2.])


def register_training():
    path = DEST/'training_contract.json'
    assert not path.exists()
    write(path, dict(created_local=datetime.now().astimezone().isoformat(),
        status='LOCKED_BEFORE_FIT_AND_VALIDATION_TARGET_ACCESS',
        seed=920045, device='cpu', threads=2, steps_per_population=12000,
        optimizer='AdamW; weight_decay=1e-6; gradient_norm_clip=10',
        learning_rate=[[0, .001], [4000, .0003], [8000, .0001]],
        architecture='Separate E/I,39 inputs,64 tanh,64 tanh,1 residual logit; last layer initially zero',
        history=dict(taus_ms=TAUS.tolist(), orders=[1,2,3], physical_states=36,
            features='[u,h-u],channel-major,then tau,then order; divide each channel by [3,2,2]',
            clocks='Initialize all history states to zero at the actual local reset/current initialization. Integrate all burn and recording steps. Never replace finite burn or startup by an FFT steady state.',
            integration='Exact exponential step for zero-order held normalized input at the native step endpoint; dt=0.1ms; timestep convergence assessed separately.'),
        baseline='Original smooth static transfer Phi (Hz), smoothly capped by r/(1+(r/Rmax)^16)^(1/16), probability regularization 1e-8. Output Rmax*sigmoid(baseline_logit+MLP). Rmax=500Hz E,1000Hz I. The cap is differentiable; no state clipping.',
        training_data='Original static table, original dynamic_assay calibration including all7frequencies,224 fresh train profiles only. No original validation workpoints or24waveforms, no64freshvalidationtargets, no SNN trajectory targets.',
        waveform_quadrature='16 stratified recording times per phase bin, actual finite burn and8record cycles; equal-weight stratified quadrature. Original targets use complete per-bin exposure. Full-step prediction convergence required before acceptance.',
        batch=dict(waveform_bins=96, static_points=128, linear_conditions=64),
        losses=dict(waveform='Mean squared error of bin-average prediction divided by max(profile RMS,1Hz), multiplied by100; each bin equally sampled. No fitting to individual noise paths.',
                    static='Squared asinh(rate/0.1Hz) error divided by0.1, all original static nodes eligible',
                    linear='Squared complex gain error normalized by max(tolerance*abs(measuredDC),2*SEM,1e-7), DC SNR>=10; tolerance0.10for0<f<=25Hz,0.15otherwise includingDC. Analytic derivative of SAME readout and history ODE.',
                    weights=[1.,1.,1.], schedule='All three losses at all12000steps. Final fixed-step weights only; no validation-selected checkpoint.'),
        runtime='Save progress each1000steps. Resume only same source/contract and optimizer/RNG state; no schedule extension after validation.',
        interface='Mean input already includes synaptic mean filtering,Z andM. Variance inputs are raw delayed variance intensities; do not add the old variance-response pole before this history model. Keep actual variance-moment state for Z target. Spatial integration and validation are later stages.',
        scope='A local fit is not native propagation recovery or a bifurcation result. Original acceptance gates remain unchanged.'))


def normalized_input(physical, theta=18.):
    physical = np.asarray(physical)
    sc = np.asarray(theta)-11.
    return np.stack([np.arcsinh((physical[...,0]-11)/sc),
                     np.log1p(physical[...,1]/sc**2), np.log1p(physical[...,2]/sc**2)], axis=-1)


def normalized_jacobian(physical, theta=18.):
    sc = np.asarray(theta)-11.; x=(physical[...,0]-11.)/sc
    return np.stack([1/(sc*np.sqrt(1+x*x)),1/(sc**2+physical[...,1]),1/(sc**2+physical[...,2])],axis=-1)/SCALE


@njit
def history_features(wave, period, dt, burn, steps, theta=18.):
    """All record-step features; input interpolation matches the native kernel."""
    h=np.zeros((3,4,3)); output=np.empty((steps,39)); W=wave.shape[1]
    scale=np.array([3.,2.,2.]); taus=np.array([1.,4.,16.,64.]); sc=theta-11.
    for step in range(-burn,steps):
        phase=((step+1)*dt/period)%1.
        pos=phase*W; lo=int(np.floor(pos))%W; hi=(lo+1)%W; a=pos-np.floor(pos)
        physical=(1-a)*wave[:,lo]+a*wave[:,hi]
        u=np.array([np.arcsinh((physical[0]-11)/sc),np.log1p(physical[1]/sc**2),np.log1p(physical[2]/sc**2)])/scale
        for c in range(3):
            for j in range(4):
                b=dt/taus[j]; e=np.exp(-b); h1,h2,h3=h[c,j]
                h[c,j,0]=e*h1+(1-e)*u[c]
                h[c,j,1]=e*(h2+b*h1)+(1-e*(1+b))*u[c]
                h[c,j,2]=e*(h3+b*h2+.5*b*b*h1)+(1-e*(1+b+.5*b*b))*u[c]
        if step>=0:
            output[step,:3]=u
            for c in range(3):
                for j in range(4):
                    for k in range(3):output[step,3+c*12+j*3+k]=h[c,j,k]-u[c]
    return output


def physical_from_features(features, theta=18.):
    u=features[...,:3]*SCALE;sc=np.asarray(theta)-11.
    return np.stack([11+sc*np.sinh(u[...,0]),sc**2*np.expm1(u[...,1]),sc**2*np.expm1(u[...,2])],axis=-1)


class Baseline:
    def __init__(self,pop):
        self.pop=pop;self.maximum=500. if pop=='E' else 1000.
        self.spline=TransferSpline(OUT/f'frozen_data/transfer_table/table_{pop}.npz')

    def evaluate(self,physical,theta=18.,derivatives=False):
        shape=physical.shape[:-1];p=physical.reshape(-1,3)
        t=np.broadcast_to(theta,shape).ravel()
        values=self.spline.evaluate(*p.T,t)
        r=values['rate']*1000.; R=self.maximum; eps=1e-8
        a=r/R; logden=np.logaddexp(0.,16*np.log(a))
        capped=a*np.exp(-logden/16)
        probability=eps+(1-2*eps)*capped
        logits=np.log(probability)-np.log1p(-probability)
        if not derivatives:return logits.reshape(shape)
        factor=(1-2*eps)/R*np.exp(-17*logden/16)/(probability*(1-probability))
        gradient=np.column_stack([values[key]*1000 for key in ['d_mu','d_ve','d_vi']])*factor[:,None]
        return logits.reshape(shape),gradient.reshape(shape+(3,))


class RateReadout(nn.Module):
    def __init__(self,pop):
        super().__init__(); self.pop=pop; self.maximum=500. if pop=='E' else 1000.
        self.network=nn.Sequential(nn.Linear(39,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)

    def forward(self,features,baseline_logits):
        return self.maximum*torch.sigmoid(baseline_logits+self.network(features).squeeze(-1))

    def linear_response(self,features,baseline_logits,baseline_gradient,input_gradient,frequency,channel,create_graph=False):
        features=features.detach().requires_grad_(True)
        correction=self.network(features).squeeze(-1)
        partial=torch.autograd.grad(correction.sum(),features,create_graph=create_graph)[0]
        rate=self.maximum*torch.sigmoid(baseline_logits+correction)
        derivative_factor=rate*(1-rate/self.maximum)
        index=torch.arange(len(features),device=features.device)
        dc=baseline_gradient[index,channel]+partial[index,channel]*input_gradient[index,channel]
        bank=(1+2j*torch.pi*frequency[:,None]*torch.as_tensor(np.repeat(TAUS,3),dtype=features.dtype,device=features.device)/1000)**(-torch.as_tensor(np.tile([1,2,3],4),device=features.device))-1
        p=partial[:,3:].reshape(-1,3,12)[index,channel]
        response=derivative_factor*(dc+(p*bank).sum(1)*input_gradient[index,channel])
        return response


def prepare_training():
    assert (DEST/'training_contract.json').exists()
    dest=DEST/'training_arrays';dest.mkdir(exist_ok=True)
    profiles=read(DEST/'profiles.json')['rows'];data=np.load(DEST/'prepared.npz')
    linear=read(BASE/'dynamic_assay/rows.json')['rows']
    for pop in 'EI':
        path=dest/f'{pop}.npz'; assert not path.exists()
        base=Baseline(pop);features=[];logits=[];targets=[];norms=[];ids=[]
        for row in profiles:
            if row['split']!='train' or row['pop']!=pop:continue
            k=row['id'];z=np.load(DEST/f'local_data/profile{k:03d}.npz')
            dt=float(z['dt_ms']);T=row['period_ms'];N=row['record_steps'];B=128;Q=16
            f=history_features(data['wave'][k],T,dt,row['burn_steps'],N)
            bins=np.minimum((((np.arange(N)+1)*dt/T)%1*B).astype(int),B-1)
            selected=[]
            for b in range(B):
                indices=np.flatnonzero(bins==b)
                selected.append(indices[np.minimum(((np.arange(Q)+.5)*len(indices)/Q).astype(int),len(indices)-1)])
            f=f[np.array(selected)]
            features.append(f);logits.append(base.evaluate(physical_from_features(f)))
            y=z['rate_hz'];targets.append(y);norms.append(np.full(B,max(np.sqrt(np.mean(y*y)),1.)));ids.extend([k]*B)
        table=np.load(OUT/f'frozen_data/transfer_table/table_{pop}.npz')
        x,se,si=np.meshgrid(table['x'],table['sigma_E'],table['sigma_I'],indexing='ij')
        p=np.column_stack([11+7*x.ravel(),49*se.ravel()**2,49*si.ravel()**2])
        static_features=np.zeros((len(p),39));static_features[:,:3]=normalized_input(p)/SCALE
        static_logits=base.evaluate(p);static_targets=table['rate_hz'].ravel()
        selected=[r for r in linear if r['pop']==pop];dc={}
        for r in selected:
            if r['frequency_hz']==0:dc[(r['x'],r['sigma_E'],r['sigma_I'],r['channel'])]=(r['response'][0],r['sem'])
        eligible=[]
        for r in selected:
            gain,sem=dc[(r['x'],r['sigma_E'],r['sigma_I'],r['channel'])]
            if abs(gain)/max(sem,1e-15)>=10:eligible.append(r)
        p=np.array([[11+7*r['x'],49*r['sigma_E']**2,49*r['sigma_I']**2] for r in eligible])
        lf=np.zeros((len(p),39));lf[:,:3]=normalized_input(p)/SCALE
        ll,lg=base.evaluate(p,derivatives=True)
        normalizers=[]
        for r in eligible:
            gain,_=dc[(r['x'],r['sigma_E'],r['sigma_I'],r['channel'])]
            tol=.1 if 0<r['frequency_hz']<=25 else .15
            normalizers.append(max(tol*abs(gain),2*r['sem'],1e-7))
        arrays=dict(wave_features=np.concatenate(features).astype('f4'),wave_logits=np.concatenate(logits).astype('f4'),
            wave_target=np.concatenate(targets).astype('f4'),wave_norm=np.concatenate(norms).astype('f4'),wave_profile=np.array(ids),
            static_features=static_features.astype('f4'),static_logits=static_logits.astype('f4'),static_target=static_targets.astype('f4'),
            linear_features=lf.astype('f4'),linear_logits=ll.astype('f4'),linear_base_gradient=lg.astype('f4'),linear_input_gradient=normalized_jacobian(p).astype('f4'),
            linear_channel=np.array([r['channel'] for r in eligible]),linear_frequency=np.array([r['frequency_hz'] for r in eligible],dtype='f4'),
            linear_target=np.array([complex(*r['response']) for r in eligible],dtype='c8'),linear_norm=np.array(normalizers,dtype='f4'))
        np.savez_compressed(path,**arrays)
        log('RATE TRAIN ARRAYS',pop,{k:v.shape for k,v in arrays.items()})
    write(dest/'provenance.json',dict(status='TRAINING_ONLY',train_profile_ids=[r['id'] for r in profiles if r['split']=='train'],
        validation_targets_opened=False,original_dynamic_calibration=str(BASE/'dynamic_assay/rows.json'),
        source_sha256=hashlib.sha256(open(__file__,'rb').read()).hexdigest()))


def train():
    import time
    contract=read(DEST/'training_contract.json');torch.set_num_threads(contract['threads'])
    dest=DEST/'fit';dest.mkdir(exist_ok=True)
    source_hash=hashlib.sha256(open(__file__,'rb').read()).hexdigest()
    for pop in 'EI':
        assert not (dest/f'{pop}_final.pt').exists()
        seed=contract['seed']+('EI'.index(pop));torch.manual_seed(seed)
        rng=np.random.default_rng(seed)
        data={k:torch.from_numpy(v) for k,v in np.load(DEST/f'training_arrays/{pop}.npz').items()}
        net=RateReadout(pop);optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-6)
        N=contract['steps_per_population'];t0=time.time();trace=[];start=0
        checkpoint=dest/f'{pop}_checkpoint.pt'
        if checkpoint.exists():
            saved=torch.load(checkpoint,weights_only=False,map_location='cpu')
            assert saved['source_hash']==source_hash
            net.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);start=saved['step']
            rng.bit_generator.state=saved['random_state'];torch.set_rng_state(saved['torch_rng'])
            trace=read(dest/f'{pop}_progress.json')['trace'];t0-=trace[-1]['elapsed']
        for step in range(start,N):
            lr=.001 if step<4000 else .0003 if step<8000 else .0001
            for group in optimizer.param_groups:group['lr']=lr
            wi=rng.integers(len(data['wave_target']),size=96);si=rng.integers(len(data['static_target']),size=128);li=rng.integers(len(data['linear_target']),size=64)
            optimizer.zero_grad(set_to_none=True)
            prediction=net(data['wave_features'][wi],data['wave_logits'][wi]).mean(1)
            lw=100*(((prediction-data['wave_target'][wi])/data['wave_norm'][wi])**2).mean()
            prediction=net(data['static_features'][si],data['static_logits'][si])
            ls=(((torch.asinh(prediction/.1)-torch.asinh(data['static_target'][si]/.1))/.1)**2).mean()
            predicted=net.linear_response(data['linear_features'][li],data['linear_logits'][li],data['linear_base_gradient'][li],data['linear_input_gradient'][li],data['linear_frequency'][li],data['linear_channel'][li],True)
            ll=(abs((predicted-data['linear_target'][li])/data['linear_norm'][li])**2).mean()
            loss=lw+ls+ll;assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),10.);optimizer.step()
            if step%100==0 or step==N-1:
                trace.append(dict(step=step+1,loss=float(loss.detach()),wave=float(lw.detach()),static=float(ls.detach()),linear=float(ll.detach()),elapsed=time.time()-t0))
            if (step+1)%1000==0:
                log('RATE FIT',pop,trace[-1]);write(dest/f'{pop}_progress.json',dict(status='RUNNING',step=step+1,expected=N,trace=trace))
                torch.save(dict(model=net.state_dict(),optimizer=optimizer.state_dict(),step=step+1,random_state=rng.bit_generator.state,torch_rng=torch.get_rng_state(),source_hash=source_hash),dest/f'{pop}_checkpoint.pt')
        torch.save(dict(model=net.state_dict(),pop=pop,source_hash=source_hash,steps=N,contract=contract),dest/f'{pop}_final.pt')
        write(dest/f'{pop}_progress.json',dict(status='COMPLETE',step=N,expected=N,trace=trace))
    write(dest/'locked_weights.json',dict(status='FINAL_WEIGHTS_LOCKED_BEFORE_VALIDATION',created_local=datetime.now().astimezone().isoformat(),
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.glob('*_final.pt')},source_sha256=source_hash,validation_scored=False))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','prepare','train']);a=p.parse_args()
    {'register':register_training,'prepare':prepare_training,'train':train}[a.command]()
