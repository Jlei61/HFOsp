"""Resolve final local links and verify completion metadata without modifying results."""
import sys,json,re,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.patient_state_v1.common import RUN,write_json
assert (RUN/'completion_audit.json').exists()
a=json.loads((RUN/'completion_audit.json').read_text());assert a['status']=='COMPLETE_SCIENTIFIC_PROTOTYPE_WITH_LIMITATIONS' and 8<=a['elapsed_hours']<=10
links=[]
for name in ['scientific_review.md','read_first.md','completion_audit.md','next_version_decision.md']:
    p=RUN/name;txt=p.read_text()
    for ref in re.findall(r'!?\[[^\]\n]*\]\(([^)\n]+)\)',txt):
        if ref.startswith(('http:','https:','#')):continue
        target=(p.parent/ref.split('#')[0]).resolve();assert target.exists(),(name,ref);links.append(dict(document=name,reference=ref,target=str(target)))
assert json.loads((RUN/'input_delivery_audit.json').read_text())['status']=='PASS'
assert json.loads((RUN/'figure_file_audit.json').read_text())['status']=='PASS'
assert json.loads((RUN/'joint_grid_refit_generation_v1_26/scientific_audit.json').read_text())['n_generated']==128
write_json(RUN/'final_delivery_link_audit.json',dict(status='PASS',created_unix=time.time(),n_resolved_local_links=len(links),links=links,scope='Final document targets and completed metadata; does not assert human figure acceptance or biological model adequacy'))
print(json.dumps(dict(status='PASS',n_links=len(links))))
