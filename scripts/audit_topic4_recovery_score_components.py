"""Decompose the existing frozen score; no new objective or qualification.

The mode-feature coordinates still depend on mode frequency. They are not a
pure conditional-shape term and must not be interpreted as a variance split.
"""
import numpy as np
from scripts import analyze_topic4_propagation_recovery_night as s
from src.topic4_multievent_distribution_objective_v2_1 import off_diagonal_mean_distance


def main():
    obj=s.rt.load_objective(s.rt.read(s.an.run.PARENT))
    maskscale=s.rt.read(s.an.run.OUT/'analysis/mask_score_calibration.json')['a_mask']
    records=[]
    for phase in ['wave1','long','final_A','final_B']:
        path=s.night.OUT/f'analysis_{phase}/training_scores.json'
        if not path.exists():continue
        spec=s.rt.read(s.night.OUT/f'{phase}_units.json');actual={(c,int(t),int(seed)) for c,t,seed in spec['units']}
        for r in s.rt.read(path)['scores']:
            if (r['base_id'],r['topology_seed'],r['seed']) not in actual:continue
            z=r['distribution_score']
            if z.get('loss_off') is None:continue
            n=z['n_events'];counts=np.asarray(z['mode_counts']);labels=np.repeat(np.arange(obj.k),counts)
            x=np.eye(obj.k)[labels]/obj.proportions/np.sqrt(obj.k)
            freq=off_diagonal_mean_distance(x,np.ones(obj.k)/np.sqrt(obj.k))
            glob=.25*z['D_off']['global']/obj.normalizers['global']
            freq=.25*freq/obj.normalizers['balanced_modes']
            other=.25*z['D_off']['balanced_modes']/obj.normalizers['balanced_modes']-freq
            mask=.5*r['joint_participation_score']['D_mask_off']/maskscale
            total=glob+freq+other+mask
            assert abs(total-r['combined_search_loss'])<1e-10
            records.append(dict(phase=phase,candidate=r['base_id'],display_name=r['display_name'],topology=r['topology_seed'],seed=r['seed'],n=n,
                TB_count=int(counts[0]),TA_count=int(counts[1]),global_feature_contribution=glob,mode_frequency_coordinate_contribution=freq,
                other_mode_weighted_coordinates_contribution=other,joint_participation_contribution=mask,total=total))
    dest=s.night.OUT/'score_component_audit';dest.mkdir(exist_ok=True)
    s.an.writecsv(dest/'components.csv',records)
    s.rt.write(dest/'manifest.json',dict(producer=__file__,producer_sha256=s.rt.sha(__file__),records=len(records),
        proportions=obj.proportions.tolist(),normalizers=obj.normalizers,mask_positive_scale=maskscale,
        exact_identity='Lsearch = 0.25 global/a_global + 0.25 mode_constant/a_modes + 0.25 other_mode_coordinates/a_modes + 0.5 joint_mask/a_mask',
        meaning='Mode constant coordinates depend only on observed mode counts and frozen patient proportions. Other mode coordinates include both frequency and feature differences; this is not an independent pure morphology contribution.',
        not_variance_decomposition=True,not_new_score=True,negative_components_not_clipped=True,
        unit='Existing estimable full trajectory; no pooling events across noise realizations.'))
    print({'output':str(dest),'records':len(records)},flush=True)


if __name__=='__main__':main()
