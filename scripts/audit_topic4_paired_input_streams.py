"""Use already-recorded per-step input digests to qualify paired comparisons.

The digest includes xi, scalar nu_now, and the full float64 arrival vector on
every integration step. A shared seed alone is never marked as matched input.
"""
import argparse,json
from scripts import analyze_topic4_propagation_recovery_night as review
an=review.an;rt=review.rt


def main(phase):
    old,plan,spec,cases=review.stage_cases(phase);lookup={c['id']:c for c in cases};result=[]
    for c in cases:
      if 'parent_id' not in c:continue
      parent=lookup.get(f'{c["parent_id"]}@{c["topology"]}')
      if parent is None:
          path=an.run.OUT/'candidates'/f'{c["parent_id"]}.json'
          if not path.exists():continue
          p=rt.read(path);parent=dict(p,base_id=p['id'],topology=c['topology'],output_stage=p.get('stage','screen'))
      seeds=sorted({int(s) for cid,t,s in spec['units'] if cid==c['base_id'] and int(t)==c['topology']}) or old['seeds']
      for seed in seeds:
        paths=[an.run.result_path(x['output_stage'],x['base_id'],x['topology'],seed) for x in [parent,c]]
        row=dict(candidate=c['id'],parent_id=parent['id'],topology=c['topology'],seed=seed,paths=[str(p) for p in paths])
        if not all(an.run.complete(p) for p in paths):
            result.append(dict(row,status='PENDING_MISSING_PAIRED_OUTPUT'));continue
        a,b=[rt.read(p) for p in paths];sa=a['input_segments'];sb=b['input_segments'];n=min(len(sa),len(sb));equal=[]
        for aa,bb in zip(sa[:n],sb[:n]):
            equal.append(aa['n_steps']==bb['n_steps'] and aa['start_ms']==bb['start_ms'] and aa['stream_sha256']==bb['stream_sha256'])
        same_duration=a['actual_duration_ms']==b['actual_duration_ms'];matched=bool(n and all(equal))
        row.update(status='MATCHED_FULL_INPUT' if matched and same_duration else 'MATCHED_SHARED_PREFIX' if matched else 'SHARED_SEED_INPUT_REALIZATIONS_DIFFER',
            compared_segments=n,equal_segments=sum(equal),compared_duration_ms=min(a['actual_duration_ms'],b['actual_duration_ms']),same_duration=same_duration,
            interpretation='Digest includes each step of OU and all external arrivals. Different input under changed core membership, mean or between-core OU correlation is not an engineering failure; it limits pathwise paired interpretation.')
        result.append(row)
    out=review.night.OUT/('paired_input_streams_'+phase+'.json');rt.write(out,dict(results=result,producer=__file__,producer_sha256=rt.sha(__file__),
        no_new_simulation=True,initial_state='Workers use initial_voltage=None: all membrane voltages start at V_reset; dynamics_seed changes stochastic input, not randomly sampled initial voltages.'))
    counts={s:sum(r['status']==s for r in result) for s in sorted({r['status'] for r in result})};print(json.dumps(dict(output=str(out),counts=counts)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',default='wave1');args=parser.parse_args();main(args.phase)
