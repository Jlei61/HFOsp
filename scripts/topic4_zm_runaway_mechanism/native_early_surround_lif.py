"""Conditional independent-Gaussian LIF reference, not a substitute network."""
from common import OUT,np,read,write,log
from native_input_local_lif import simulate,condition,static_run
from datetime import datetime
import argparse,os,time

SOURCE=OUT/'native_early_surround_inputs'
DEST=SOURCE/'local_lif_reference'


def register():
    assert read(SOURCE/'independent_response_audit.json')['status']=='PASS'
    DEST.mkdir(exist_ok=True);assert not (DEST/'contract.json').exists()
    write(DEST/'contract.json',dict(created_local=datetime.now().astimezone().isoformat(),
        question='Is the remaining early rate under-response already present in the independent Gaussian LIF input approximation, or introduced by the fixed conditioned39 rate surrogate?',
        reason='Native-input fixedreadout recoversabout88percentofselectedsurroundcounts, versus10percentinfreeclosure. Projectedversusactualinputmean makeslittledifference. Systematiclocalerrorcould matterinfeedback buthasnotbeencausallyproved.',
        groups=read(SOURCE/'contract.json')['selected_groups'],
        input='Actual native groupnetmean andZ from0-3s; correctedphysicalprivate rawAMPA/GABA varianceforcing. Independentexacttwo-poleGaussian noise, ZappliedafterGABAfilter.',
        thresholds='Samecurrentrate groupmean thresholds; allsurroundEandIare18mV. Two core controlsretainmeanthreshold; no assertionofpercellthresholdmixture.',
        min_replicates=8192,seed=920131,dt_ms=[.1,.05],history_ms=[0,500],primary_ms=[500,3000],bin_ms=50,
        initialization='Vreset11,zeroOUnoise currents,refractory0 atoriginal0s; inputsheldatnative0.1msforbothsteps.',
        sampling='RoundreplicatesuptooriginalN multiple. Replicatesindependentwithineachcondition; commonrandomnumbersbetweenconditions do notmakegrouporbincomparisonindependent.',
        numerical_gate='Perconditiontwo-stepbinnedmeanrelativeL2<=max(.05,3*MCsplitL2); constantinputcountspassoriginalLIFkernelbeforewaveforms.',
        interpretation='ComparebothrateandnativewiththisconditionalGaussianLIF. IfMCandrateagreebutbothunderpredictnative, inputdistributionclosureisimplicated. IfMCclosertonativeandrateunderpredictsMC, responseapproximationisimplicated. Neitheralone provesglobalfeedback mechanism.',
        budget='16geometryfixedgroups at2steps, onefixedseed,nofit,noadditionalnativeorwholenetworkrun,nobifurcationlaunch.'))


def run(device):
    c=read(DEST/'contract.json');assert not (DEST/'jobs.json').exists()
    jobs=dict(status='RUNNING',pid=os.getpid(),completed=[]);write(DEST/'jobs.json',jobs)
    # Same unmodified local reference kernel: independently verify bin totals.
    pars=np.array([condition(20.,18.,1.,1.,'E'),condition(16.,14.5,1.,1.,'I')]);R=256
    wave=np.ones((2,4,5000));wave[:,0]=np.array([20.,16.])[:,None]
    theta=np.broadcast_to(pars[:,1,None],(2,R)).copy()
    expected=static_run(pars,R,400.,100.,7918,device=device)[:,:,2]
    observed=simulate(pars,wave,theta,np.array([R,R]),.1,100.,8,50.,7918,device).sum(2)
    assert expected.sum()>0 and np.array_equal(expected,observed)
    write(DEST/'implementation_check.json',dict(status='PASS',constant_counts_bitwise=True,total_count=int(observed.sum())))
    g=np.load(SOURCE/'membership.npz');groups=g['selected_groups'];N=g['group_size'][groups].astype(int);G=len(groups)
    chunks=[np.load(p) for p in sorted((SOURCE/'inputs').glob('*.npz'))]
    m=np.concatenate([z['moments'] for z in chunks]);names=chunks[0]['moment_names'].tolist()
    raw=np.load(SOURCE/'projected_inputs.npz')['raw_private_variance_forcing']
    wave=np.concatenate([m[:,names.index('net'),None,:],raw,m[:,names.index('z'),None,:]],axis=1).transpose(2,1,0).copy()
    nrep=np.ceil(c['min_replicates']/N).astype(int)*N;R=int(nrep.max())
    theta=np.broadcast_to(g['threshold_mv'][groups,None],(G,R)).copy()
    for dt in c['dt_ms']:
        pars=np.array([condition(0.,g['threshold_mv'][group],1.,1.,'E' if g['population'][group]==0 else 'I',dt=dt) for group in groups])
        start=time.time();counts=simulate(pars,wave,theta,nrep,dt,500.,50,50.,c['seed'],device)
        np.savez_compressed(DEST/f'dt{dt:g}.npz',counts=counts,replicates=nrep,group_sizes=N,groups=groups,dt_ms=dt)
        jobs['completed'].append(dt);write(DEST/'jobs.json',jobs)
        log('EARLY CONDITIONAL LIF',dt,'seconds',round(time.time()-start,2))
    score()
    jobs['status']='COMPLETE';write(DEST/'jobs.json',jobs)


def score():
    c=read(DEST/'contract.json');fixed=np.load(SOURCE/'fixed_readout.npz')
    coarse=np.load(DEST/'dt0.1.npz');fine=np.load(DEST/'dt0.05.npz');rows=[];aggregates={}
    for j,g in enumerate(coarse['groups']):
        nr=int(coarse['replicates'][j]);N=int(coarse['group_sizes'][j]);assert nr%N==0
        x=coarse['counts'][j,:nr].astype(float);y=fine['counts'][j,:nr].astype(float)
        xc=x.mean(0)*N;yc=y.mean(0)*N
        split=(y[:nr//2].mean(0)-y[nr//2:].mean(0))*N
        denominator=max(np.linalg.norm(yc),1.);numerical=float(np.linalg.norm(xc-yc)/denominator);noise=float(np.linalg.norm(split)/denominator)
        native=fixed['native_counts'][:,j];rate=fixed['counts_native_mean_private_variance'][:,j]
        def difference(v,ref):return float(np.linalg.norm(v-ref)/max(np.linalg.norm(ref),1.))
        row=dict(c['groups'][j]);row.update(replicates=nr,windows=50,native_total_count=int(native.sum()),rate_total_count=float(rate.sum()),
            MC_native_step_total=float(xc.sum()),MC_fine_step_total=float(yc.sum()),
            rate_to_MC_native_step_L2=difference(rate,xc),rate_to_MC_fine_step_L2=difference(rate,yc),
            native_to_MC_native_step_L2=difference(native,xc),native_to_MC_fine_step_L2=difference(native,yc),
            numerical_L2=numerical,MCsplit_L2=noise,numerical_pass=bool(numerical<=max(.05,3*noise)))
        rows.append(row)
        role='surround_E' if j<9 else ('core_E' if j<11 else 'I')
        a=aggregates.setdefault(role,dict(native=np.zeros(50),rate=np.zeros(50),MC_native=np.zeros(50),MC_fine=np.zeros(50),neurons=0,groups=0))
        for key,value in [('native',native),('rate',rate),('MC_native',xc),('MC_fine',yc)]:a[key]+=value
        a['neurons']+=N;a['groups']+=1
    summary=[]
    for role,a in aggregates.items():
        total=float(a['native'].sum());r=dict(role=role,selected_neurons=a['neurons'],groups=a['groups'])
        for key in ['native','rate','MC_native','MC_fine']:r[key+'_total']=float(a[key].sum());r[key+'_to_native_count_ratio']=float(a[key].sum()/total)
        for key in ['MC_native','MC_fine']:r['rate_to_'+key+'_L2']=float(np.linalg.norm(a['rate']-a[key])/max(np.linalg.norm(a[key]),1.))
        summary.append(r)
    write(DEST/'result.json',dict(status='CONDITIONAL_LIF_REFERENCE_COMPLETE',rows=rows,selected_regional_summary=summary,
        numerical_all_pass=all(r['numerical_pass'] for r in rows),model_promoted=False,bifurcation_type='NOT_ESTABLISHED',
        scope='One suppliednativeinputhistory withconditionalGaussianreplicates. MC testsinputapproximationandsurrogate, notautonomouspropagation. No nativeconfidenceorcrossgroupindependenceclaimed.'))
    log('EARLY CONDITIONAL LIF RESULT',summary)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','run','score']);p.add_argument('--device',type=int,default=1);a=p.parse_args()
    {'register':register,'run':lambda:run(a.device),'score':score}[a.command]()
