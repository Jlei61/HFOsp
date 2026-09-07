"""Locked training-only kernel objective and sample-size-matched tolerances."""
import numpy as np
from src.topic4_joint_xy_kernel import event_kernel_features, kernel_map, mapped_distance
from src.topic4_xy_direction import onset_directions, direction_histogram, direction_distance, direction_summary
from src.topic4_xy_readout_audit import support_summary


def mean_noise_corrected_distance(mapped, reference_mean):
    """Remove within-sample diagonal bias for exploration only; not a p-value.

    Events can be dependent within a network, so this is a ranking correction,
    not an assertion of an unbiased biological population estimator.
    """
    if len(mapped) < 2: return None
    return mapped_distance(mapped, reference_mean)-float(np.var(mapped, axis=0, ddof=1, dtype=float).sum()/len(mapped))


class KernelObjective:
    def __init__(self, adapter, out, kernel_contract):
        self.adapter, self.out = adapter, out
        self.training, self.old_objective = adapter.base.training_contract()
        self.xy = np.asarray(adapter.read(adapter.OLD/'direction_objective_v2.json')['contact_xy_mm'])
        self.groups, self.pairs = self.training['groups'], self.training['pairs']
        self.patient = self.training['onsets_ms']
        self.contract = adapter.read(kernel_contract)
        if self.contract['patient_training_sha256'] != self.training['sha256']:
            raise RuntimeError('kernel training data mismatch')
        adapter.runtime.verify_amendment(self.contract['source_hashes'])
        self.maps = self.contract['maps']; self.time_scale = self.contract['time_scale_ms']
        self.mapped = self.map(self.patient)
        self.means = {k:v.mean(axis=0,dtype=float) for k,v in self.mapped.items()}
        self.patient_direction = direction_histogram(onset_directions(self.patient,self.xy))

    def map(self, t):
        features = event_kernel_features(t,self.xy,self.groups,self.time_scale)
        return {k:kernel_map(x,self.maps[k]) for k,x in features.items()}

    def metrics(self, table, *, reference=None, means=None, patient_direction=None):
        t=np.asarray(table); means=self.means if means is None else means
        mapped=self.map(t); distances={k:mapped_distance(x,means[k]) for k,x in mapped.items()}
        corrected=mean_noise_corrected_distance(mapped['joint'],means['joint'])
        if len(t):
            vector=self.old_objective.component_vector(t,self.training['reference'] if reference is None else reference,
                self.groups,self.pairs,self.training['embedding'],composite=False,components=('D_support','D_order','D_lag'))
        else:
            vector={k:{'value':None,'status':'NOT_ESTIMABLE_LOW_EVENTS'} for k in ('D_support','D_order','D_lag')}
        view=onset_directions(t,self.xy); dr=direction_summary(view); hist=direction_histogram(view)
        dr['histogram']=hist
        return {'n_events':len(t),'joint_distance':distances['joint'],
            'kernel_distances':distances,'joint_noise_corrected':corrected,
            'exploration_score':corrected if corrected is not None else 2.,
            **{k:vector[k]['value'] for k in ('D_support','D_order','D_lag')},
            'component_status':{k:vector[k]['status'] for k in ('D_support','D_order','D_lag')},
            'direction_distance':direction_distance(hist,self.patient_direction if patient_direction is None else patient_direction),
            'direction':dr,'support':support_summary(t,self.groups,self.pairs)}

    def calibrate(self,plan):
        path=self.out/'patient_calibration.json'
        if path.exists(): return self.adapter.read(path)
        rng=np.random.default_rng(plan['calibration']['seed']);blocks=self.training['block_ids'];unique=np.unique(blocks)
        samples={};keys=('joint_distance','D_support','D_order','D_lag','direction_distance')
        for n in plan['calibration']['sample_sizes']:
            draws=[]
            for _ in range(plan['calibration']['draws']):
                mask=np.isin(blocks,rng.choice(unique,len(unique)//2,replace=False))
                idx=rng.choice(np.flatnonzero(mask),n,replace=False);ref=self.patient[~mask]
                metric=self.metrics(self.patient[idx],reference=self.old_objective.patient_reference(
                    ref,self.groups,self.pairs,self.training['embedding']),
                    means={k:v[~mask].mean(axis=0,dtype=float) for k,v in self.mapped.items()},
                    patient_direction=direction_histogram(onset_directions(ref,self.xy)))
                draws.append({**{k:metric[k] for k in keys},'kernel_distances':metric['kernel_distances']})
            samples[str(n)]={'q95':{k:float(np.quantile([d[k] for d in draws if d[k] is not None],.95)) for k in keys},
                'kernel_q95':{k:float(np.quantile([d['kernel_distances'][k] for d in draws],.95)) for k in self.maps},'draws':draws}
            self.adapter.write(self.out/'calibration_progress.json',{'completed_sizes':list(samples)})
            print('Kernel calibration',n,flush=True)
        result={'version':plan['version'],'samples':samples,'patient_training_sha256':self.training['sha256'],
            'heldout_opened':False,'threshold_role':'Training-block development tolerances, not statistical equivalence.',
            'ranking_only':'Within-event diagonal correction is not an independence claim; confirmation uses fresh networks.'}
        self.adapter.write(path,result);return result
