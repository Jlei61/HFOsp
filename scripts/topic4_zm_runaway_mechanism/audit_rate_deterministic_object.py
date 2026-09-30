"""Identify the two deterministic limits without running or fitting a network.

At fixed prescribed mean rates, quantify the recurrent variance operator
difference hidden by the current noise switch. These probes are NOT equilibria.
"""
from common import OUT, model, np, read, write, log
from shared_variance_network_sensitivity import split
from model_v3 import THRESHOLD_Z
from scipy import sparse
from scipy.special import ndtr
from pathlib import Path
import hashlib

DEST=OUT/'rate_deterministic_object_audit'


def main():
    DEST.mkdir(exist_ok=True);assert not (DEST/'result.json').exists()
    s=model(40);private,qa=split(s,.05)
    source=OUT/'conditioned_refractory_fine_forcing/recorded_drive_binomial_seed1/trajectory.npz'
    z=np.load(source);r=z['group_rate_hz'];drive=np.load(OUT/'native_fine_external_drive/drive.npz')['drive_g40']
    probes={'uniform_10Hz':(np.full(s.P,.01),drive.mean(0))}
    for lo,hi in [(8000,9420),(9420,9870),(11500,12500)]:
        select=(z['time_ms']>lo)&(z['time_ms']<=hi)
        probes[f'count_mean_{lo}_{hi}']=(r[select].astype(float).mean(0)/1000,drive[lo:hi].mean(0))
    rows=[];saved={};checks=[]
    for k,kind in enumerate(['ampa','gaba']):
        full=sparse.load_npz(s.folder/f'variance_{kind}.npz').tocsr();pv=private[kind]
        assert np.array_equal(full.indices,pv.indices) and np.array_equal(full.indptr,pv.indptr)
        assert np.all(full.data>=pv.data) and np.all(pv.data>=0)
        mean=sparse.load_npz(s.folder/f'mean_{kind}.npz').tocsr()
        target,source_group,a=s.raw[k];_,_,q=s.raw[k+2]
        qsum=np.asarray(q.sum(1)).ravel()
        for name,(rate,nu) in probes.items():
            history=np.tile(rate,len(s.delays))
            total=np.asarray(full@history);remaining=np.asarray(pv@history)
            independent=np.bincount(target,weights=qsum*rate[source_group],minlength=s.P)
            error=float(abs(independent-total).max());assert error<1e-10
            external=s.jext**2*nu if k==0 else np.zeros(s.P)
            removed=(total-remaining)/np.maximum(total+external,1e-30)
            fields={'full_variance_input':total,'private_variance_input':remaining,'external_variance_input':external,'removed_fraction_including_external':removed}
            if k==1:
                # Exact stationary current covariance of the same double-exponential kernel.
                mu=s.tm*s.area[k]*(mean@history)
                scale=s.tm**2*s.area[k]**2/(2*(s.rise[k]+s.decay[k]))
                target_full=ndtr((THRESHOLD_Z-mu)/np.sqrt(np.maximum(scale*total,1e-30)))
                target_private=ndtr((THRESHOLD_Z-mu)/np.sqrt(np.maximum(scale*remaining,1e-30)))
                fields.update(GABA_mean_mV=mu,Z_target_full=target_full,Z_target_private=target_private)
            for key,val in fields.items():saved[f'{kind}_{name}_{key}']=val
            for region,label in enumerate(['Core A E','Core B E','Surround E']):
                mask=s.E&(s.geo['group_region']==region);weight=s.sizes[mask]/s.sizes[mask].sum()
                row=dict(synapse=kind,probe=name,target=label,
                    mean_removed_recurrent_fraction=float(weight@((total-remaining)/np.maximum(total,1e-30))[mask]),
                    mean_removed_total_fraction=float(weight@removed[mask]),
                    minimum_removed_total_fraction=float(removed[mask].min()),maximum_removed_total_fraction=float(removed[mask].max()))
                if k==1:
                    row.update(mean_Z_target_full=float(weight@target_full[mask]),mean_Z_target_private=float(weight@target_private[mask]))
                rows.append(row)
            checks.append(dict(synapse=kind,probe=name,raw_operator_aggregation_error=error))
    np.savez_compressed(DEST/'fields.npz',**saved)
    files=['refractory_spatial_diagnostic.py','refractory_count_consistency.py','shared_variance_network_sensitivity.py']
    script=Path(__file__).resolve().parent
    result=dict(status='READ_ONLY_OBJECT_AUDIT_COMPLETE',rows=rows,checks=checks,source=str(source),
        source_sha256={f:hashlib.sha256((script/f).read_bytes()).hexdigest() for f in files},
        stationary_split_checks=qa,
        mean_field='noise=False constructs full recurrent diffusion Q and expected own rate. Finite count output absent.',
        conditional_drift='noise=True constructs private diffusion (1-f)Q and actual count history. Suppressing only count innovations while retaining these operators defines a different deterministic drift. It is not an exact ensemble mean.',
        interpretation='The two existing spatial arms differ in variance allocation as well as finite output. Their difference cannot isolate the causal role of count innovations. Fine versus coarse external input comparisons within each arm remain valid.',
        probe_scope='Fixed imposed mean rate vectors, not equilibria, trajectories, local-response validation or a Z-clock error attribution. Stationary Poisson split does not restore exact renewal or correlated covariance.',
        threshold_Z_mV=THRESHOLD_Z,model_modified=False,network_runs=0,bifurcation_type='NOT_ESTABLISHED')
    write(DEST/'result.json',result)
    log('DETERMINISTIC OBJECT',[(x['synapse'],x['target'],x['mean_removed_total_fraction']) for x in rows if x['probe']=='count_mean_8000_9420'])


if __name__=='__main__':main()
