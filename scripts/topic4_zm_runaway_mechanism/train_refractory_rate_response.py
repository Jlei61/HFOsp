"""Fixed-budget local response fit. Validation targets are never read here."""
from refractory_rate_response import *
from nonlinear_rate_response import physical_from_features
from common import BASE
from scipy.special import xlogy
from datetime import datetime
import argparse,hashlib,time

def prepare():
    c=read(DEST/'contract.json');dest=DEST/'training_arrays';dest.mkdir(exist_ok=True)
    old=OUT/'nonlinear_rate_response';linear=read(BASE/'dynamic_assay/rows.json')['rows']
    sources=[(old,OUT/'local_refractory_flux/training_data'),(DEST,DEST/'local_data')]
    for pop in 'EI':
        path=dest/f'{pop}.npz';assert not path.exists();base=BaseLogit(pop)
        ff=[];bb=[];yy=[];aa=[];ee=[];nn=[];ids=[]
        for src,targets in sources:
            rows=read(src/'profiles.json')['rows'];inputs=np.load(src/'prepared.npz')
            for row in rows:
                if row['split']!='train' or row['pop']!=pop:continue
                k=row['id'];z=np.load(targets/f'profile{k:03d}.npz');R=row['replicates'];burn=row['burn_steps'];steps=row['record_steps']
                f=features(inputs['wave'][k],row['period_ms'],.1,burn,steps,pop)
                choose=np.minimum(((np.arange(2048)+.5)*steps/2048).astype(int),steps-1)
                f=f[choose];y=z['spike_counts'][burn:].astype(float)/R;a=z['available_counts'][burn:].astype(float)/R
                norm=max(y.mean(),.001);y=y[choose];a=a[choose]
                p=np.divide(y,a,out=np.zeros_like(y),where=a>0);entropy=-a*(xlogy(p,p)+xlogy(1-p,1-p))
                ff.append(f);bb.append(base.evaluate(physical_from_features(f)));yy.append(y);aa.append(a);ee.append(entropy);nn.append(np.full(2048,norm));ids.extend([f'{src.name}/{k}']*2048)
        # Reuse only the original calibration arrays, not old fitted predictions.
        calibration=np.load(old/f'training_arrays/{pop}.npz')
        sf=calibration['static_features'].astype(float);sp=physical_from_features(sf);sb=base.evaluate(sp)
        lf=calibration['linear_features'].astype(float);lp=physical_from_features(lf);lb,lg=base.evaluate(lp,derivatives=True)
        frequencies=calibration['linear_frequency'];bank,cov,K=transfer_factors(pop,frequencies)
        arrays=dict(flux_features=np.concatenate(ff).astype('f4'),flux_logits=np.concatenate(bb).astype('f4'),
            flux_fired=np.concatenate(yy).astype('f4'),flux_available=np.concatenate(aa).astype('f4'),
            flux_entropy=np.concatenate(ee).astype('f4'),flux_norm=np.concatenate(nn).astype('f4'),
            static_features=sf.astype('f4'),static_logits=sb.astype('f4'),static_target=calibration['static_target'],
            linear_features=lf.astype('f4'),linear_logits=lb.astype('f4'),linear_base_gradient=lg.astype('f4'),
            linear_input_gradient=normalized_jacobian(lp).astype('f4'),linear_channel=calibration['linear_channel'],
            linear_bank=bank.astype('c8'),linear_cov=cov.astype('c8'),linear_K=K.astype('c8'),
            linear_target=calibration['linear_target'],linear_norm=calibration['linear_norm'])
        np.savez_compressed(path,**arrays)
        write(dest/f'{pop}_provenance.json',dict(status='TRAINING_ONLY',profiles=sorted(set(ids)),validation_targets_opened=False,
            feature_rows=len(arrays['flux_features']),note='Observed refractory availability is a calibration label only; autonomous validation reconstructs availability from predicted flux.'))
        log('REFRACTORY TRAIN ARRAYS',pop,len(arrays['flux_features']))

def train():
    c=read(DEST/'contract.json');config=c['training'];torch.set_num_threads(config['threads'])
    dest=DEST/'fit';dest.mkdir(exist_ok=True)
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [__import__('pathlib').Path(__file__),__import__('pathlib').Path(__file__).with_name('refractory_rate_response.py')]}
    for pop in 'EI':
        assert not (dest/f'{pop}_final.pt').exists()
        seed=config['seed']+(pop=='I');torch.manual_seed(seed);rng=np.random.default_rng(seed)
        arrays=np.load(DEST/f'training_arrays/{pop}.npz');data={k:torch.as_tensor(arrays[k]) for k in arrays.files}
        net=RefractoryReadout(pop);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-6);trace=[];t0=time.time()
        for step in range(config['steps_per_pop']):
            if step in [4000,8000]:
                for group in opt.param_groups:group['lr']=.0003 if step==4000 else .0001
            wi=rng.integers(len(data['flux_fired']),size=512);si=rng.integers(len(data['static_target']),size=128);li=rng.integers(len(data['linear_target']),size=64)
            opt.zero_grad();ell=net.logits(data['flux_features'][wi],data['flux_logits'][wi])
            lw=((data['flux_available'][wi]*torch.nn.functional.softplus(ell)-data['flux_fired'][wi]*ell-data['flux_entropy'][wi])/data['flux_norm'][wi]).mean()
            sr=net.stationary(data['static_features'][si],data['static_logits'][si])
            ls=(((torch.asinh(sr/.1)-torch.asinh(data['static_target'][si]/.1))/.1)**2).mean()
            lr=net.linear_response(data['linear_features'][li],data['linear_logits'][li],data['linear_base_gradient'][li],data['linear_input_gradient'][li],data['linear_channel'][li],data['linear_bank'][li],data['linear_cov'][li],data['linear_K'][li],True)
            ll=(abs((lr-data['linear_target'][li])/data['linear_norm'][li])**2).mean()
            loss=10*lw+ls+ll;assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),10.);opt.step()
            if (step+1)%100==0:
                trace.append(dict(step=step+1,loss=float(loss.detach()),flux=float(lw.detach()),static=float(ls.detach()),linear=float(ll.detach()),elapsed=time.time()-t0))
            if (step+1)%1000==0:
                log('REFRACTORY RATE FIT',pop,trace[-1]);write(dest/f'{pop}_progress.json',dict(status='RUNNING',trace=trace))
        torch.save(dict(model=net.state_dict(),pop=pop,steps=config['steps_per_pop'],contract=c,source_hashes=source_hashes),dest/f'{pop}_final.pt')
        write(dest/f'{pop}_progress.json',dict(status='COMPLETE',trace=trace))
    write(dest/'locked_weights.json',dict(status='FINAL_WEIGHTS_LOCKED_BEFORE_VALIDATION',created_local=datetime.now().astimezone().isoformat(),source_hashes=source_hashes,
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.glob('*_final.pt')},validation_scored=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train']);a=p.parse_args();{'prepare':prepare,'train':train}[a.command]()
