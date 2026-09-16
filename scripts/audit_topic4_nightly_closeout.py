from pathlib import Path
import json,subprocess,hashlib,math
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/topic4_sef_hfo/nightly_central_workpoint'
def main():
 units=[];missing=[]
 for batch in ['A','B']:
  score=json.loads((OUT/f'batches/{batch}/scores.json').read_text())
  for r in score['candidates']:
   if r['control']:continue
   for uid,u in r['units'].items():
    w=json.loads(Path(u['worker_path']).read_text());n=u['observation']['N']
    assert w['execution_status']=='COMPLETE'
    if n:
     f=OUT/f'batches/{batch}/figures/{r["candidate_id"]}_{uid}_all_events.pdf';info=subprocess.run(['pdfinfo',str(f)],capture_output=True,text=True,check=True).stdout
     pages=int(next(l.split(':')[1] for l in info.splitlines() if l.startswith('Pages:')));assert pages==math.ceil(n/8),(f,pages,n)
    units.append(dict(candidate=r['candidate_id'],unit=uid,N=n,physical_status=w['physical_status'],actual_ms=w['simulation']['actual_duration_ms'],all_primary_pages_verified=bool(n),worker=str(Path(u['worker_path']).resolve())))
 files=[]
 for folder in [OUT/'figures',OUT/'batches/A/figures',OUT/'batches/B/figures']:
  for f in sorted(folder.iterdir()):
   if f.suffix in ['.png','.gif']:
    with Image.open(f) as im:
     n=getattr(im,'n_frames',1)
     for i in range(n):im.seek(i);im.load()
     v=dict(frames=n,size=list(im.size))
   elif f.suffix=='.pdf':
    info=subprocess.run(['pdfinfo',str(f)],capture_output=True,text=True,check=True).stdout
    v=dict(pages=int(next(l.split(':')[1] for l in info.splitlines() if l.startswith('Pages:'))))
   else:continue
   files.append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),**v))
 assert len(units)==24 and sum(u['N'] for u in units)==812
 out=dict(new_units=units,new_unit_count=len(units),primary_events=sum(u['N'] for u in units),engineering_failures=0,physical_runaway=sum(u['physical_status']=='RUNAWAY' for u in units),files=files,human_visual_acceptance='PENDING',scope='Bounded A+B negative closeout; conditional response/new-noise stage not activated because no compatible workpoint.')
 (OUT/'completion_audit.json').write_text(json.dumps(out,indent=2));print({k:v for k,v in out.items() if k not in ['new_units','files']});print('validated files',len(files))
if __name__=='__main__':main()
