"""Bounded original-shared-input correspondence controls at two D values."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,os,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/topic4_sef_hfo/kinetic_population_bifurcation_20260916'


def main():
    folder=OUT/'population_pair_replication/native_shared_input_controls'
    folder.mkdir(parents=True,exist_ok=False)
    tests=[dict(D=.225,seed=s,duration=12000,microscopic=m) for s in (1901,1902) for m in (False,True)]
    tests += [dict(D=.25,seed=s,duration=4000,microscopic=True) for s in (1901,1902)]
    commands=[]
    for x in tests:
        c=[sys.executable,'-u',str(Path(__file__).with_name('particle_control.py')),
           '--D',str(x['D']),'--scale','1','--seed',str(x['seed']),'--duration',str(x['duration']),
           '--shared-noise','--device','1']
        if x['microscopic']:c.append('--microscopic')
        commands.append(c)
    contract=dict(question='Does restoring the original global and spatial OU input law preserve the interictal and sustained sides and spatial propagation?',
        tests=tests,commands=commands,
        interpretation='The autonomous density holds BOTH global and spatial shared OU fields at zero; earlier none-OU particle controls do the same. This test restores both original laws without changing D, M dynamics, thresholds or communication.',
        windows=dict(D225_primary_ms=[8000,12000],D225_secondary_ms=[[1000,4000],[4000,8000]],D25_ms=[1000,4000]),
        reference='Existing zero-shared-OU N1 controls; six private-input pairs at D=.225 and two at D=.25.',
        readouts=['self-limited events and quiet occupancy','global/core rates','complete-event and all-time spatial maps','core lead-lag changes'],
        statistical_unit='Two paired private/shared-input realizations per condition. This is a qualitative noise-law bridge and descriptive effect check, not an independent-pixel/event test or precise stochastic threshold estimate.',
        validation='Original SpatialOUDrive with fixed source parameters; independent global/spatial streams. The seed1901 individual12s run must reproduce the completed1s prefix, including shared-input records.',
        boundary='Six prespecified jobs only. No automatic model acceptance or parameter expansion; no original Fig5 realization replay claim.')
    (folder/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    (folder/'status.json').write_text(json.dumps(dict(status='RUNNING',pid=os.getpid(),workers=2,jobs=6),indent=2)+'\n')
    started=time.time()
    def one(pair):
        x,c=pair;label=f"D{x['D']:g}_seed{x['seed']}_{'individual' if x['microscopic'] else 'grouped'}"
        with (folder/f'{label}.log').open('w') as log:
            proc=subprocess.Popen(c,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            (folder/f'{label}_process.json').write_text(json.dumps(dict(pid=proc.pid,command=c))+'\n')
            code=proc.wait()
        suffix='_microscopic' if x['microscopic'] else ''
        source=OUT/'particle_controls/selected_g40'/f"D{x['D']:.6f}_Nscale1_seed{x['seed']}_{x['duration']}ms{suffix}_native_shared_OU"
        state=json.load(open(source/'status.json')) if (source/'status.json').exists() else {}
        row=dict(test=x,returncode=code,status=state.get('status','MISSING'),source=str(source.resolve()))
        (folder/f'{label}_result.json').write_text(json.dumps(row,indent=2)+'\n');print(row,flush=True)
        return row
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(one,zip(tests,commands)))
    passed=all(x['returncode']==0 and x['status']=='COMPLETE' for x in results)
    (folder/'status.json').write_text(json.dumps(dict(status='EXECUTION_COMPLETE' if passed else 'EXECUTION_HAS_FAILURE',results=results,wall_s=time.time()-started),indent=2)+'\n')


if __name__=='__main__':main()
