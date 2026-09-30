#!/usr/bin/env python3
"""Archive campaign code/configuration without dispatching or altering workers."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    args = parser.parse_args()
    assert args.label and all(c.isalnum() or c in '_-' for c in args.label)
    dest = ROOT/'reproducibility'/args.label
    if dest.exists():
        raise FileExistsError('Preserve previous snapshots; choose a new label.')
    # These modules define functions/classes only at import; no worker entrypoint
    # or prepare/dispatch function is called here.
    modules = ['run_topic4_loop_zk_conditional', 'run_topic4_loop_axis_native',
               'run_topic4_loop_axis_conditional', 'run_topic4_loop_zk_locality_cpu',
               'run_topic4_loop_cuda_override', 'run_topic4_loop_axis_cuda_override']
    for name in modules:
        importlib.import_module(name)
    protocol = json.loads((ROOT/'protocol.json').read_text())
    expected = {Path(k): v for k, v in protocol['source_hashes'].items()}
    expected[REPO/'scripts/run_topic4_loop_zk_conditional.py'] = protocol['runner_sha256']
    for path, value in expected.items():
        assert digest(path) == value, ('Frozen source mismatch', str(path))
    files = set(expected)
    loaded = []
    engine_roots = set()
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if not path:
            continue
        path = Path(path).resolve()
        if path.is_relative_to(REPO) and path.suffix == '.py':
            files.add(path)
            loaded.append(dict(module=name, path=str(path)))
            if 'snn_engine' in path.parts:
                index = path.parts.index('snn_engine')
                engine_roots.add(Path(*path.parts[:index+1]))
    # Include lazy engine helpers alongside the actually imported engine files.
    for folder in engine_roots:
        files.update(folder.rglob('*.py'))
    files.add(Path(__file__).resolve())
    files.update((REPO/'scripts').glob('*topic4_loop*.py'))
    files.update((REPO/'scripts/paper_figures').glob('build_fig5_*loop*.py'))
    files.add(REPO/'scripts/paper_figures/build_fig5_conditional_zk.py')
    files.add(REPO/'scripts/paper_figures/build_fig5_conditional_spatial_fields.py')
    for name in ['execution_contract.md', 'protocol.json', 'queue.json']:
        files.add(ROOT/name)
    files.update((ROOT/'jobs').glob('*.json'))
    for condition in ['rotated', 'isotropic']:
        for category in ['native_runs', 'conditional_runs']:
            folder = ROOT/'axis_controls'/category/condition
            files.update(folder.glob('protocol.json'))
            files.update(folder.glob('queue.json'))
            files.update((folder/'jobs').glob('*.json'))
    files.update((ROOT/'axis_controls').glob('*contract*.json'))
    # QA gates preserve the execution-route identity used for existing workers.
    files.update((ROOT/'qa').rglob('gate.json'))
    files.update((ROOT/'qa').glob('*gate.json'))
    files.update((ROOT/'axis_controls/conditional_runs/reference').glob('*qa.json'))
    dest.mkdir(parents=True)
    records = []
    for source in sorted(files):
        source = source.resolve()
        if source.is_relative_to(ROOT):
            relative = Path('campaign')/source.relative_to(ROOT)
        elif source.is_relative_to(REPO):
            relative = Path('repo')/source.relative_to(REPO)
        else:
            raise ValueError(('Unexpected source root', source))
        target = dest/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        before = digest(source)
        shutil.copy2(source, target)
        assert digest(target) == before == digest(source), source
        records.append(dict(original=str(source), snapshot=str(relative), sha256=before,
                            bytes=target.stat().st_size,
                            checked_against_frozen_protocol=source in expected))
    packages = sorted([dict(name=d.metadata['Name'], version=d.version)
                       for d in importlib.metadata.distributions() if d.metadata.get('Name')],
                      key=lambda d: (d['name'].lower(), d['version']))
    conda = []
    for path in sorted((Path(sys.prefix)/'conda-meta').glob('*.json')):
        item = json.loads(path.read_text())
        conda.append({k: item.get(k) for k in ['name', 'version', 'build', 'subdir']})
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total',
                          '--format=csv,noheader'], capture_output=True, text=True)
    environment = dict(executable=sys.executable, python=sys.version, machine=platform.machine(),
                       operating_system=platform.system(), kernel_release=platform.release(),
                       installed_python_distributions=packages, conda_packages=conda,
                       gpu_query_exit_code=gpu.returncode, gpu_devices=gpu.stdout.strip(),
                       worker_thread_settings={k: os.environ[k] for k in
                           ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']},
                       note='Installed versions/builds, without credentials or package source URLs. No GPU context or simulation was started.')
    (dest/'environment.json').write_text(json.dumps(environment, indent=2)+'\n')
    manifest = dict(status='PASS_ARCHIVED_CODE_AND_CONFIGURATION',
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    expected_frozen_sources_verified=len(expected), files=records,
                    imported_project_modules=loaded, engine_source_roots=list(map(str, sorted(engine_roots))),
                    environment_sha256=digest(dest/'environment.json'),
                    limits='Code/configuration snapshot, not a standalone relocated data package. Large immutable source states, graphs and observations remain at their recorded campaign paths. Analysis producers may acquire later versions; take a final snapshot after final review. No simulation convergence or scientific validity is certified by this archive.')
    (dest/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (dest/'README.md').write_text('''# 本轮代码与配置快照

保存实际导入的项目模块、引擎源码、冻结方程链、执行包装器、分析和绘图脚本，以及固定科学任务配置和执行路径校验记录。manifest逐文件记录原路径、快照路径和SHA256；冻结方程链同时与运行协议中的哈希核对，复制后再次读回验证。

environment.json记录cuda_env中的Python版本、已安装包版本/构建号和GPU驱动信息，没有复制环境变量全表或访问凭证。此过程只导入定义和复制文件，没有创建GPU计算上下文、派发仿真或改变活动任务。

这是代码与配置快照，完整状态、结构图和原生观测仍保留在协议指定的数据路径；不能将本目录称作可迁移的独立容器或完整数据备份。快照只描述生成时的代码与配置；整体完成与候选图审阅状态见运行根目录的completion_audit.json（若已生成）。旧快照保留，不用新版本覆盖。
''')
    print(json.dumps(dict(snapshot=str(dest), files=len(records),
                          bytes=sum(r['bytes'] for r in records),
                          frozen_sources_verified=len(expected))), flush=True)


if __name__ == '__main__':
    main()
