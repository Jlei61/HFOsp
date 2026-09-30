#!/usr/bin/env python3
"""Source averaging and white-input diagnostics from the unchanged native run.

Synaptic finite-window balances retain the initial/final delayed queues and
filter states. The E external count is inferred, so its reconstructed mean is
not presented as an independent model fit.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time
import numpy as np
from scipy import sparse
from campaign import ROOT,read,write,sha
from observe_source_aggregation import OUT,NAME,INITIAL,STEPS
from coupled_density_exit import ADAPTED
from conditional_density_inputs import OPS
import run_topic4_loop_zk_conditional as native


def filter_square_sum(ar,ad):
    # Impulse h[n]=(1-ad)*sum(ar**j*ad**(n-j),j=0..n), n>=0.
    return (1-ad)**2/(ad-ar)**2*(ad*ad/(1-ad*ad)+ar*ar/(1-ar*ar)-2*ad*ar/(1-ad*ar))


def main(wait):
    dest=OUT/'input_analysis';dest.mkdir(exist_ok=True);assert not (dest/'result.json').exists()
    if not (dest/'contract.json').exists():
        write(dest/'contract.json',dict(status='REGISTERED_BEFORE_INPUT_RECONSTRUCTION',created_epoch=time.time(),
            question='Does replacing individual observed source rates by within-group means introduce appreciable input bias at the missed recruitment edge? Are observed current variances/covariance consistent with the existing independentwhite-source approximation?',
            method='Use original native weighted/delayed edges and percell emittedcounts. Delivered recurrentjump sum equals emittedweightedcounts plus initialqueuedmass minus finalqueuedmass. Reconstruct I mean with exactdiscrete filterboundary terms. The inhibitory balance is independent; excitatory externaljumps are inferred from observedEmean and checked against integerPoissoncounts. Compare group-vs-cell sourceinput and Poisson squared-edge variance, with unchanged targetheterogeneity.',
            limits='Observed2s conditional trajectory, four0.5sblocks. White-source variance excludes source temporal/cross-source correlations andslowdrift; it is a diagnostic assumption, not a fitted replacement. Cell/timeslice scatter is not independentseed uncertainty. Emean agreement after inferring externalcounts is algebraic, not validation.',
            gates='Numerical budget identity: relativeIerror<1e-10 andmaxabsoluteerror<1e-3mV-equiv. Recoveredexternalcount integerresidual<1e-4. No scientificacceptance tolerance or gain is changed.',
            producer_sha256=sha(__file__),formal_bifurcation_allowed=False))
    while not (OUT/'observer_audit.json').exists():
        write(dest/'progress.json',dict(status='WAITING_COMPLETE_NATIVE_OBSERVER',pid=os.getpid(),updated_epoch=time.time()))
        if not wait:return
        time.sleep(15)
    assert read(OUT/'observer_audit.json')['status']=='PASS'
    with np.load(OUT/'cell_statistics.npz') as z:
        counts=z['per_cell_spike_counts'].astype(float);mom=z['per_cell_mean_moments'];external=z['per_cell_mean_external_per_ms']
    start=native.read_pickle(INITIAL)['engine'];end=native.read_pickle(OUT/'runs'/NAME/'checkpoint.pkl')['engine']
    assert start['step']==420000 and end['step']==440000
    geo=dict(np.load(ADAPTED/'geometry.npz'));groups=geo['cell_group'];sizes=geo['group_size'];N=len(groups);P=len(sizes)
    assert N==40000 and counts.shape==(4,N)
    total_counts=counts.sum(0);grcount=np.bincount(groups,weights=total_counts,minlength=P)/sizes
    grouped_counts=grcount[groups]
    write(dest/'progress.json',dict(status='LOADING_ORIGINAL_GRAPH',pid=os.getpid(),updated_epoch=time.time()))
    sim,_,_,identity=native.base.old.setup(9108405);assert identity==read(OPS/'prepared.json')['graph_identity']
    assert np.array_equal(sim.net['pos'],geo['original_positions'])
    params=sim.params;tm=np.r_[np.full(32000,params.tau_m_E),np.full(8000,params.tau_m_I)]
    emitted=[];white_jump=[];projection_qa=[];filter_squares=[];ar_all=[];ad_all=[]
    for kind,source in [('ampa',slice(0,32000)),('gaba',slice(32000,40000))]:
        incoming=np.zeros((N,2));variance=np.zeros((N,2))
        vectors=np.c_[total_counts[source],grouped_counts[source]]
        for matrix in sim.net[kind+'_by_delay']:
            if not matrix.nnz:continue
            incoming+=matrix@vectors
            variance+=matrix.multiply(matrix)@vectors
        rise=getattr(params,'tau_r_'+kind.upper());decay=getattr(params,'tau_d_'+kind.upper())
        ar=np.exp(-.1/rise);ad=np.exp(-.1/decay);ar_all.append(ar);ad_all.append(ad)
        target_operator=sparse.load_npz(ROOT/'target_stationary_response_audit'/f'mean_{kind}_dc.npz')
        independently_grouped=(target_operator@grcount)*tm/rise
        err=float(abs(incoming[:,1]-independently_grouped).max())
        rel=float(np.linalg.norm(incoming[:,1]-independently_grouped)/max(np.linalg.norm(incoming[:,1]),1e-30))
        assert rel<1e-11
        projection_qa.append(dict(pathway=kind,max_jump_difference=err,relative_difference=rel))
        factor=filter_square_sum(ar,ad)
        impulse=np.zeros(20000);s=1.;current=0.
        for n in range(len(impulse)):
            current=ad*current+(1-ad)*s;impulse[n]=current;s*=ar
        assert abs(float(impulse@impulse)-factor)<1e-12
        filter_squares.append(factor);emitted.append(incoming);white_jump.append(variance/STEPS)
    emitted=np.array(emitted);white_jump=np.array(white_jump);ar_all=np.array(ar_all);ad_all=np.array(ad_all)
    observed=mom[:,:2].mean(0);means=mom.mean(0);boundary=[];delivered=[];required=[];prediction=[]
    for j,letter in enumerate(['E','I']):
        ar,ad=ar_all[j],ad_all[j]
        s0,sT=start['s_'+letter],end['s_'+letter];I0,IT=start['I_'+letter],end['I_'+letter]
        queue=start['ring_s'+letter].sum(0)-end['ring_s'+letter].sum(0)
        b=ar*(s0-sT)/(STEPS*(1-ar))+ad*(I0-IT)/(STEPS*(1-ad))
        delivery=emitted[j,:,0]+queue
        need=(observed[j]-b)*STEPS*(1-ar)
        boundary.append(b);delivered.append(delivery);required.append(need)
        prediction.append(delivery/(STEPS*(1-ar))+b)
    error=prediction[1]-observed[1]
    rel=float(np.linalg.norm(error)/max(np.linalg.norm(observed[1]),1e-30));maxerror=float(abs(error).max())
    assert rel<1e-10 and maxerror<1e-3,(rel,maxerror)
    ext_incr=tm/params.tau_r_AMPA*np.r_[np.full(32000,params.J_ext_E),np.full(8000,params.J_ext_I)]
    inferred_counts=(required[0]-delivered[0])/ext_incr
    integer_error=float(abs(inferred_counts-np.rint(inferred_counts)).max())
    assert inferred_counts.min()>-1e-4 and integer_error<1e-4,integer_error
    # Expected independent white source variance using observed source rates.
    # This second-order assumption is exactly the one being examined, not native truth.
    white_variance=white_jump*np.array(filter_squares)[:,None,None]
    white_variance[0]+=((np.rint(inferred_counts)/STEPS)*ext_incr**2*filter_squares[0])[:,None]
    observed_variance=np.stack([means[2]-means[0]**2,means[3]-means[1]**2])
    within_variance=np.stack([(mom[:,2]-mom[:,0]**2).mean(0),(mom[:,3]-mom[:,1]**2).mean(0)])
    covariance=means[4]-means[0]*means[1]
    source_bias=(emitted[:,:,1]-emitted[:,:,0])/(STEPS*(1-ar_all[:,None]))
    Z=start['slow']['z'];net_bias=source_bias[0]-Z*source_bias[1]
    observed_net_var=observed_variance[0]+Z*Z*observed_variance[1]-2*Z*covariance
    white_net_var=white_variance[0]+Z[:,None]**2*white_variance[1]
    assert observed_net_var.min()>-1e-6
    region=geo['group_region'][groups];cell=geo['group_cell'][groups];E=np.arange(N)<32000
    edge=[r['cell'] for r in read(OUT/'contract.json')['edge_cells']]
    masks=[('allE',E),('coreA',E&(region==0)),('coreB',E&(region==1)),('surroundE',E&(region==2)),('I',~E),('selected10edge_cells_E',E&np.isin(cell,edge))]
    rows=[]
    for label,m in masks:
        rows.append(dict(region=label,targets=int(m.sum()),
            mean_native_rate_Hz=float(total_counts[m].mean()/2),
            group_minus_cell_source_IE_II_mean_mV=source_bias[:,m].mean(1).tolist(),
            group_minus_cell_source_IE_II_RMS_mV=np.sqrt((source_bias[:,m]**2).mean(1)).tolist(),
            mean_net_source_bias_mV=float(net_bias[m].mean()),RMS_net_source_bias_mV=float(np.sqrt(np.mean(net_bias[m]**2))),
            observed_IE_II_variance_mean_mV2=observed_variance[:,m].mean(1).tolist(),
            within0p5s_IE_II_variance_mean_mV2=within_variance[:,m].mean(1).tolist(),
            independent_white_IE_II_variance_cell_group_mV2=white_variance[:,m].mean(1).tolist(),
            mean_IE_II_covariance_mV2=float(covariance[m].mean()),
            observed_net_variance_mean_mV2=float(observed_net_var[m].mean()),
            white_net_variance_cell_group_mean_mV2=white_net_var[m].mean(0).tolist()))
    np.savez_compressed(dest/'input_reconstruction.npz',native_rate_Hz=total_counts/2,grouped_source_rate_Hz=grouped_counts/2,
        observed_IE_II_mean=observed,source_group_minus_cell_mean_IE_II=source_bias,source_net_bias=net_bias,
        independent_white_variance_IE_II_cell_group=white_variance,observed_variance_IE_II=observed_variance,
        within0p5s_variance_IE_II=within_variance,observed_covariance_IE_II=covariance,
        inferred_actual_external_counts=inferred_counts,expected_external_counts=external.mean(0)*2000,
        inhibitory_exact_mean_reconstruction=prediction[1],filter_boundary=np.array(boundary))
    result=dict(status='COMPLETE_INDIVIDUAL_SOURCE_INPUT_DIAGNOSTIC',rows=rows,
        original_graph_identity=identity,projection_QA=projection_qa,
        inhibitory_finitewindow_mean_budget_relative_error=rel,inhibitory_mean_budget_max_error_mV=maxerror,
        recovered_E_external_count_max_integer_residual=integer_error,
        scope='Exact mean-input change from replacing measured individual source counts by groupmeans at fixed targetweights. White-source variance is an approximate counterfactual omitting correlations, refractory structure and timevariation. This does not isolate a causal change in firing or certify a closure repair.',
        formal_bifurcation_allowed=False,producer_sha256=sha(__file__))
    write(dest/'result.json',result);write(dest/'progress.json',dict(status='COMPLETE',updated_epoch=time.time()))
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');main(p.parse_args().wait)
