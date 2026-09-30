"""Invertible input conditioning of the SAME refractory-rate function class.

No physical state, synapse, refractory equation, graph or Z/M law is added.
The parent failed candidate and its weights remain immutable.
"""
from refractory_rate_response import *
from refractory_rate_response import DEST as PARENT
from nonlinear_rate_response import physical_from_features
from datetime import datetime
from pathlib import Path
import argparse,hashlib,json

DEST=OUT/'conditioned_refractory_rate'

class ConditionedNetwork(nn.Module):
    def __init__(self,layers,pop):
        super().__init__();z=np.load(DEST/f'conditioning/{pop}.npz')
        self.register_buffer('center',torch.as_tensor(z['center'],dtype=torch.float32))
        self.register_buffer('transform',torch.as_tensor(z['transform'],dtype=torch.float32))
        self.layers=layers

    def forward(self,x):return self.layers((x-self.center)@self.transform.T)

class ConditionedReadout(RefractoryReadout):
    def __init__(self,pop):
        super().__init__(pop);self.network=ConditionedNetwork(self.network,pop)

def register():
    diag=read(PARENT/'error_decomposition/result.json');assert diag['status']=='DIAGNOSTIC_COMPLETE_NO_MODEL_CHANGE'
    DEST.mkdir(exist_ok=True);assert not (DEST/'contract.json').exists()
    old=read(PARENT/'contract.json');c=dict(old);c['created_local']=datetime.now().astimezone().isoformat()
    c['status']='LOCKED_BEFORE_CONDITIONED_FIT_AND_NEW_TARGETS'
    c['question']='Does fixed invertible feature conditioning address the documented approximately1e10covariance condition without adding physical memory or changing the rate function class?'
    c['parent']=str(PARENT)
    c['only_change']='Readout sees A*(features-training_mean), A=U diag(1/sqrt(max(eigenvalue,largest*1e-7))) U^T. All39directions retained. Same64/64tanh architecture, loss,schedule,seed,480trainingprofiles and originalcalibration; no onset/validationtarget fitting.'
    c['conditioning']=dict(training_only=True,eigenvalue_floor_relative=1e-7,drop_dimensions=False,proof='Map arbitrary parent network first layer through inverse transform; outputs and full linear-response derivatives must match before fit.')
    c['new_data']=dict(validation_generator_seed=920072,validation_noise_seed=920074,validation_profiles=64,replicates=8192,
        scope='Same preregistered broad stimulus generator. Allnewprofiles validationonly, no new training data. Parent64 and legacy64 are reused diagnostics with unchanged58/64gates; original24 requireallpass.')
    c['training']['source']='Exact parent480profile arrays, unchanged; whitening statistics computed only from these training features.'
    c['stop']='One fixed conditioning and original12000steps/pop. No more optimization, transform tuning or model change after scoring. Local pass still requires native autonomous dynamics and propagation before bifurcation.'
    write(DEST/'contract.json',c)
    folder=DEST/'conditioning';folder.mkdir()
    for pop in 'EI':
        z=np.load(PARENT/f'error_decomposition/{pop}_feature_covariance.npz');e=z['eigenvalues'];U=z['eigenvectors'];floor=e[-1]*1e-7
        A=(U*(1/np.sqrt(np.maximum(e,floor))))@U.T
        assert np.linalg.matrix_rank(A)==39
        np.savez_compressed(folder/f'{pop}.npz',center=z['center'],transform=A,covariance=z['covariance'],eigenvalues=e,floor=floor)
    # Reuse immutable training arrays, rather than rebuilding data after seeing validation.
    (DEST/'training_arrays').symlink_to((PARENT/'training_arrays').resolve(),target_is_directory=True)

def check():
    torch.set_num_threads(2);rows=[]
    for pop in 'EI':
        old=RefractoryReadout(pop).double();old.load_state_dict(torch.load(PARENT/f'fit/{pop}_final.pt',map_location='cpu',weights_only=False)['model'])
        new=ConditionedReadout(pop).double();new.network.layers.load_state_dict(old.network.state_dict())
        A=new.network.transform;center=new.network.center;inverse=torch.linalg.inv(A)
        with torch.no_grad():
            W=old.network[0].weight;new.network.layers[0].weight.copy_(W@inverse);new.network.layers[0].bias.copy_(old.network[0].bias+W@center)
        data=np.load(PARENT/f'training_arrays/{pop}.npz');f=torch.tensor(data['flux_features'][::499].astype(float));b=torch.tensor(data['flux_logits'][::499].astype(float))
        err=float((old.logits(f,b)-new.logits(f,b)).abs().max().detach());assert err<1e-9,(pop,err)
        choose=np.arange(0,len(data['linear_target']),73);args=[torch.tensor(data[k][choose].astype('complex128' if np.iscomplexobj(data[k]) else ('int64' if k=='linear_channel' else 'float64'))) for k in ['linear_features','linear_logits','linear_base_gradient','linear_input_gradient','linear_channel','linear_bank','linear_cov','linear_K']]
        lhs=old.linear_response(*args).detach().numpy();rhs=new.linear_response(*args).detach().numpy();diff=float(np.max(abs(lhs-rhs)));assert diff<1e-8,(pop,diff)
        z=np.load(DEST/f'conditioning/{pop}.npz');transformed=z['transform']@z['covariance']@z['transform'].T;e=np.linalg.eigvalsh(transformed)
        rows.append(dict(pop=pop,output_max_error=err,linear_gain_max_error=diff,new_covariance_condition=float(e[-1]/e[0])))
    write(DEST/'implementation_check.json',dict(status='PASS',rows=rows,same_function_class=True,physical_equations_unchanged=True))
    log('CONDITIONED FUNCTION CLASS CHECK PASS',rows)

def train():
    assert read(DEST/'implementation_check.json')['status']=='PASS'
    import train_refractory_rate_response as worker
    worker.DEST=DEST;worker.RefractoryReadout=ConditionedReadout;worker.train()
    path=DEST/'fit/locked_weights.json';locked=read(path)
    locked['conditioning_files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (DEST/'conditioning').glob('*.npz')}
    locked['conditioned_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();write(path,locked)

def load_models():
    locked=read(DEST/'fit/locked_weights.json');assert locked['status']=='FINAL_WEIGHTS_LOCKED_BEFORE_VALIDATION'
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==locked['conditioned_source_sha256']
    for name,h in locked['conditioning_files'].items():assert hashlib.sha256((DEST/'conditioning'/name).read_bytes()).hexdigest()==h
    torch.set_num_threads(2);nets={};bases={}
    for pop in 'EI':
        path=DEST/f'fit/{pop}_final.pt';assert hashlib.sha256(path.read_bytes()).hexdigest()==locked['files'][path.name]
        net=ConditionedReadout(pop).double();net.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);net.eval();nets[pop]=net;bases[pop]=BaseLogit(pop)
    return nets,bases,locked

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','check','train']);a=p.parse_args();{'register':register,'check':check,'train':train}[a.command]()
