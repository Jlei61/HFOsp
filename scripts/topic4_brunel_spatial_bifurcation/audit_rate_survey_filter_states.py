"""Check the physical constituent filters of the exact surveyed profiles."""
from audit_rate_filter_states import *
from collections import Counter


def fingerprint(path):
    path=Path(path);st=path.stat()
    return dict(path=str(path.resolve()),size=st.st_size,mtime_ns=st.st_mtime_ns)


def main():
    s=RateField();folder=RATE_OUT/'periodic_completion/stability_coverage'
    destination=folder/'constituent_filter_audit.json'
    prior=read(destination).get('rows',[]) if destination.exists() else []
    cache={q['orbit']:q for q in prior};rows=[]
    for index,item in enumerate(read(folder/'plan.json')['rows']):
        source=folder/(Path(item['orbit']).stem+'.json')
        result=read(source) if source.exists() else dict(status='PENDING')
        actual=Path(result.get('analyzed_orbit',item['orbit']))
        identity=fingerprint(actual);old=cache.get(str(actual))
        if old and old.get('profile_fingerprint')==identity:check=old['filter_state_check']
        else:
            z=np.load(actual);check=filter_state_minima(s,z['r'],float(z['T']))
        status=result['status']
        if status in ['UNSTABLE','NUMERICALLY_STABLE'] and not check['positive']:
            status='PHYSICAL_PROFILE_REFINEMENT_REQUIRED'
        rows.append(dict(index=index,orbit=str(actual),result_source=str(source),
            profile_fingerprint=identity,filter_state_check=check,
            computed_multiplier_classification=result['status'],accepted_classification=status))
        if index%20==0:print('FILTER SURVEY',index,status,flush=True)
    out=dict(status='CHECKED',rows=rows,classification_counts=dict(Counter(q['accepted_classification'] for q in rows)),
        negative_filter_profiles=sum(not q['filter_state_check']['positive'] for q in rows),
        scope='Physical waveform check for the exact profile used in each persisted Floquet verdict. A negative constituent filter withholds acceptance of the exact periodic orbit; the numerical multiplier calculation remains recorded. This is neither a new instability nor a proof that the underlying equation is wrong.')
    write(destination,out);print({k:v for k,v in out.items() if k!='rows'},flush=True)


if __name__=='__main__':main()
