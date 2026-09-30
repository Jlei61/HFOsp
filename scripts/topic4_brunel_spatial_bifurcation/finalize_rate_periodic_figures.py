"""Rebuild the numerical figure package from current accepted orbit files."""
from plot_rate_periodic_completion import *
import subprocess,re


def main():
 producer=Path(__file__).parent
 scripts=['plot_rate_periodic_completion.py','plot_rate_critical_modes.py','plot_rate_periodic_composite.py',
 'plot_rate_cycle_interaction.py','plot_rate_burst_onset.py','plot_rate_cycle_cases.py','rate_periodic_readouts.py',
 'analyze_rate_periodic_gap.py','summarize_rate_periodic.py']
 if (PERIODIC_OUT/'PD_double_low_validation.json').exists():scripts.insert(-1,'plot_rate_period_doubling.py')
 if (PERIODIC_OUT/'floquet/PDchild_a120.00000_N4096_dt0.1.json').exists():scripts.insert(-1,'plot_rate_PD_child.py')
 for name in scripts:
  done=subprocess.run([sys.executable,str(producer/name)],capture_output=True,text=True)
  print(name,done.returncode,flush=True)
  if done.returncode:raise RuntimeError(done.stdout+'\n'+done.stderr)
 readme=F/'README.md';txt=readme.read_text();txt=re.sub(r'^### ([A-Za-z0-9_]+)$',r'### \1.png',txt,flags=re.M);readme.write_text(txt)
 fs=families();crit=critical();cases=read(PERIODIC_OUT/'composite_cases.json')['cases'];rejected=[]
 for q in cases:
  if q['orbit']:
   m=read(PERIODIC_OUT/f'orbits/{q["orbit"]}.json');assert m['status']=='CONVERGED';assert abs(q['J_EE_core']-m['J_EE_core'])<1e-10
 for q in crit:
  if q['label'].startswith('LPC'):assert abs(q.get('dJ_dcoordinate',q.get('dJ_dlogT')))<1e-7
 for f in (PERIODIC_OUT/'orbits').glob('gap*_N*.json'):
  q=read(f)
  if q['status']!='CONVERGED':rejected.append(dict(path=str(f),status=q['status'],residual_hz=q['residual_hz']))
 files=[]
 for f in sorted(F.glob('*.pdf')):
  info=subprocess.run(['pdfinfo',str(f)],capture_output=True,text=True,check=True).stdout
  assert re.search(r'Pages:\s+1\b',info),f
  assert f.with_suffix('.png').exists()
  files.append(dict(pdf=str(f),png=str(f.with_suffix('.png')),pdf_bytes=f.stat().st_size))
 write(PERIODIC_OUT/'delivery_validation.json',dict(periodic_families={k:len(v) for k,v in fs.items()},latest_refined_periodic_critical_count=len(crit),
   same_model_case_orbit_parameters_verified=True,all_listed_case_orbits_converged=True,all_latest_fold_derivatives_below_1e_minus_7=True,
   files=files,rejected_gap_candidates=rejected,
   scientific_status='Not an exhaustive Floquet survey. Native SNN dynamic equivalence remains unvalidated. J=.946 late modulation is unclassified.',
   visual_acceptance='Candidate for user visual review; no claim of human acceptance'))
 print('DELIVERY',len(files),'figures',len(crit),'periodic critical points',flush=True)

if __name__=='__main__':main()
