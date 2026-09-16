#!/usr/bin/env python
"""measurement_manifest.json and split_manifest.json for the v0.3.11 root."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from src.topic5_group_event_state.v0311 import data as D
from src.topic5_group_event_state.v0311.packets import RATIO_BANDS,XLAG_PAIRS,MIN_DELAY_FOR_IQR

ROOT=Path('/data/hfosp_group_event_state_rich_event_identification_v0311')
SUBJECTS=('epilepsiae_1125','epilepsiae_1096','epilepsiae_253')
UNITS=dict(relative_delay_s='s',delay_rank_fraction='fraction of participating contacts',
           tied_lead_flag='indicator',band_log_energy='log integrated band energy (arbitrary units)',
           band_centroid_s='s relative to the event core',band_log_peak='log peak amplitude',
           cross_band_lag_s='s (signed, first band minus second)',
           wave_log_rms='log1p RMS of the detector-referenced core waveform',
           wave_log_peak='log1p peak absolute amplitude',wave_log_line_length='log1p mean absolute difference')

def main():
    meas={};split_manifest={}
    for s in SUBJECTS:
        p=torch.load(ROOT/'packets'/f'{s}.pt',weights_only=False)
        t=p['targets'];ev=p['event_time']
        finite=lambda a:float(np.isfinite(a).mean())
        blocks=p['blocks']
        rel=np.array([b['release']-b['start'] for b in blocks])
        comps={}
        for i,b in enumerate(RATIO_BANDS):
            comps[f'band_log_energy_ratio[{b}/ied_low]']=dict(
                estimable_fraction=finite(t['band_ratio'][:,i]),
                status='ESTIMABLE' if finite(t['band_ratio'][:,i])>0.05 else 'NOT_ESTIMABLE')
        for i,pair in enumerate(XLAG_PAIRS):
            comps[f'signed_cross_band_lag[{pair[0]}->{pair[1]}]']=dict(
                estimable_fraction=finite(t['signed_xlag'][:,i]),
                status='ESTIMABLE' if finite(t['signed_xlag'][:,i])>0.05 else 'NOT_ESTIMABLE')
        comps['participating_delay_iqr_s']=dict(estimable_fraction=finite(t['delay_iqr']),
            status='ESTIMABLE' if finite(t['delay_iqr'])>0.05 else 'NOT_ESTIMABLE',
            rule=f'requires at least {MIN_DELAY_FOR_IQR} participating contacts with a finite relative delay')
        meas[s]=dict(
            n_events=int(len(ev)),n_packets=int(len(p['packets']['start'])),n_blocks=len(blocks),
            contacts=p['selected_contacts'],coarse_communities=p['shafts'],
            coarse_community_rule=p.get('coarse_community_rule','physical shaft'),
            bands=p['bands'],band_edges_hz=p['band_edges_hz'],
            cross_band_pairs=[list(x) for x in p['cross_band_pairs']],
            contact_feature_names=p['contact_feature_names'],
            event_feature_names=p['event_feature_names'],units=UNITS,
            release_contract=p['release_contract'],
            release_delay_seconds=dict(median=float(np.median(rel)),max=float(rel.max()),min=float(rel.min())),
            observed_hours=float((p['observed_support'][:,1]-p['observed_support'][:,0]).sum()/3600),
            ambiguous_overlap_hours=float((p['ambiguous_intervals'][:,1]-p['ambiguous_intervals'][:,0]).sum()/3600)
                if len(p['ambiguous_intervals']) else 0.,
            ambiguous_rule='wall-clock intervals covered by two records are removed; a rate cannot pool them',
            background_context_missing_blocks=p.get('background_context_missing',[]),
            clock_restorations=int(sum(r['restored'] for r in p['clock_restorations'])),
            target_components=comps,
            source_cards=p['source_cards'][:3]+['... %d cards total'%len(p['source_cards'])])
        rows={}
        for name,fn in (('S-E',D.build_split),('S-ID',D.build_split_id)):
            sp=fn(p,s)
            px,pt,_=D.packet_tables(p,sp);sc=D.fit_scaling(p,sp,px,pt)
            rows[name]=dict(seed=sp['seed'],contract=sp['contract'],fit_end=sp['fit_end'],
                            se_cutoff=sp.get('se_cutoff'),gap_start=sp.get('se_gap_start'),
                            n_train_packets=int(sp['train_packet'].sum()),
                            n_inner=int(len(sp['inner_starts'])),n_forward=int(len(sp['forward_starts'])),
                            fit_hours=sc['fit_hours'],base_rate_per_hour=sc['base_rate_per_hour'],
                            n_seizures=len(sp['seizures']),
                            excluded_hours=float((sp['excluded_intervals'][:,1]-sp['excluded_intervals'][:,0]).sum()/3600)
                                if len(sp['excluded_intervals']) else 0.)
        split_manifest[s]=rows
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/'measurement_manifest.json').write_text(json.dumps(meas,indent=1,default=str))
    (ROOT/'split_manifest.json').write_text(json.dumps(split_manifest,indent=1,default=str))
    print(json.dumps({k:{'n_events':v['n_events'],'communities':v['coarse_communities'],
                         'not_estimable':[c for c,d in v['target_components'].items() if d['status']!='ESTIMABLE']}
                      for k,v in meas.items()},indent=1))
    print(json.dumps({k:{n:{kk:r[kk] for kk in ('n_inner','n_forward','fit_hours','base_rate_per_hour','n_seizures')}
                         for n,r in v.items()} for k,v in split_manifest.items()},indent=1,default=str))

if __name__=='__main__':main()
