"""Bounded independent-seed extension of the actual-size paired assay."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json, os, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/topic4_sef_hfo/kinetic_population_bifurcation_20260916'


def main():
    folder = OUT/'population_pair_replication/target_N1_six_seed_extension'
    folder.mkdir(parents=True, exist_ok=False)
    tests = [dict(D=.225, scale=1, seed=seed, duration=12000, microscopic=micro)
             for seed in range(1903, 1907) for micro in (False, True)]
    contract = dict(question='At the actual N1 target size, does parent grouping change the distribution of interictal activity beyond variation among private-input realizations?',
        previous_seeds=[1901,1902], additional_tests=tests,
        fixed_conditions='Selected g40 communication, physical frozen Z path at D=.225, dynamic M, common OU zero; no fitted parameter or resource-strata selection.',
        primary_window_ms=[8000,12000], secondary_windows_ms=[[1000,4000],[4000,8000]],
        primary_observables=['mean global E rate','quiet fraction','complete self-limited event count'],
        spatial_observables=['50-ms event recruitment maps','A/B lead-lag variation'],
        statistical_unit='Six independent paired private-input realizations including the existing two; events and pixels are nested.',
        decision='Assess paired effects and uncertainty alongside individual-model between-input dispersion. No automatic acceptance from a nonsignificant test; qualitative self-limiting versus sustained behavior and spatial propagation remain required. N16 and deterministic-density limits are separate.',
        boundary='Exactly four additional pairs. No parameter sweep, automatic extra seeds, or bifurcation promotion.',
        commands=[])
    for x in tests:
        cmd=[sys.executable,'-u',str(Path(__file__).with_name('particle_control.py')),
             '--D',str(x['D']),'--scale','1','--seed',str(x['seed']),
             '--duration','12000','--device','1']
        if x['microscopic']:cmd.append('--microscopic')
        contract['commands'].append(cmd)
    (folder/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    start=time.time()
    (folder/'status.json').write_text(json.dumps(dict(status='RUNNING',pid=os.getpid(),jobs=8,workers=2),indent=2)+'\n')
    def one(item):
        x,cmd=item
        tag=f"seed{x['seed']}_{'individual' if x['microscopic'] else 'grouped'}"
        with (folder/f'{tag}.log').open('w') as f:
            p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
            (folder/f'{tag}_process.json').write_text(json.dumps(dict(pid=p.pid,command=cmd))+'\n')
            code=p.wait()
        suffix='_microscopic' if x['microscopic'] else ''
        source=OUT/'particle_controls/selected_g40'/f"D0.225000_Nscale1_seed{x['seed']}_12000ms{suffix}"
        state=json.load(open(source/'status.json')) if (source/'status.json').exists() else {}
        result=dict(test=x,returncode=code,status=state.get('status','MISSING'),source=str(source.resolve()))
        (folder/f'{tag}_result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(result,flush=True)
        return result
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(one,zip(tests,contract['commands'])))
    passed=all(x['returncode']==0 and x['status']=='COMPLETE' for x in results)
    (folder/'status.json').write_text(json.dumps(dict(status='EXECUTION_COMPLETE' if passed else 'EXECUTION_HAS_FAILURE',results=results,wall_s=time.time()-start),indent=2)+'\n')


if __name__=='__main__':main()
