#!/usr/bin/env python3
"""Publish the author-designated Figure 4A v10 and its checked full plate."""
from pathlib import Path
import hashlib
import json
import shutil

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/paper-ready-figure/fig4'
SOURCE = BASE / 'candidates/a_left_circuit_spacing_20261010'
ARCHIVE = ROOT / 'results/paper-ready-figure/archive/2026-10-10_pre_mechanism_fig4/fig4'
VERSION = 'compact_ai_v4_a_left_circuit_spacing_v10'
PRODUCER = 'scripts/paper_figures/build_fig4_a_mechanism_candidate.py'
TITLE = '局部E/I回路、二维空间基底、局部椭圆连接与SEEG采样读出'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    current = read(BASE / 'current_version.json')
    record = read(SOURCE / 'figure4_candidate_registry.json')
    assert current['layout_version'] == 'compact_ai_tighter_columns_larger_squares_v4', 'This promotion has already run or the current figure changed.'
    assert record['layout_version'] == VERSION
    assert sha(BASE / 'figures/fig4-complete-layout.png') == record['base_published_complete_sha256']
    for path, digest in current['outputs_sha256'].items():
        assert sha(ROOT / path) == digest, path
    for path, digest in record['outputs'].items():
        assert sha(SOURCE / path) == digest, path
    for key in ('panel_b', 'panel_c', 'panel_d', 'panel_e', 'panel_i'):
        assert current[key] == record[key], key
    untouched = {}
    for letter in 'bcdefghi':
        name = f'figures/fig4-panel{letter}.png'
        with Image.open(BASE / name) as a, Image.open(SOURCE / name) as b:
            assert a.size == b.size
            assert ImageChops.difference(a.convert('RGB'), b.convert('RGB')).getbbox() is None
        for ext in ('png', 'pdf', 'svg'):
            name = f'figures/fig4-panel{letter}.{ext}'
            untouched[name] = sha(BASE / name)
    assert not ARCHIVE.exists(), 'Preserve the existing archive; inspect before retrying.'
    shutil.copytree(BASE, ARCHIVE, ignore=shutil.ignore_patterns('candidates'))
    shutil.copy2(ROOT / 'docs/current_figure4.md', ARCHIVE / 'current_figure4_before_mechanism.md')
    for name in record['outputs']:
        if Path(name).name.startswith(('fig4-panela.', 'fig4-complete-layout')):
            shutil.copy2(SOURCE / name, BASE / name)
    shutil.copytree(SOURCE / 'source', BASE / 'source', dirs_exist_ok=True)
    for script in (Path(__file__), ROOT / PRODUCER):
        shutil.copy2(script, BASE / 'source' / script.name)
    accepted = dict(current.pop('pending_panel_a_revision'))
    accepted.update(status='AUTHOR_DESIGNATED_CURRENT', author_designated_on='2026-10-10', canonical_figure_replaced=True)
    record['panel_a'].update(status='AUTHOR_DESIGNATED_CURRENT', author_designated_on='2026-10-10')
    record['titles']['A'] = TITLE
    record['checks']['A_revision'] = record['panel_a']
    record['outputs'] = {name: sha(BASE / name) for name in record['outputs']}
    record.update(status='CURRENT_AUTHOR_DESIGNATED', human_visual_acceptance='PENDING_COMPOSITE_VISUAL_REVIEW',
                  updated_on='2026-10-10', canonical_figure_replaced=True,
                  published_from=str(SOURCE.relative_to(ROOT)), previous_version_archive=str(ARCHIVE.relative_to(ROOT)),
                  accepted_panel_a_revision=accepted)
    for name in ('figure4_panel_registry.json', 'figure4_candidate_registry.json'):
        write(BASE / name, record)
    qa = read(SOURCE / 'visual_qa.json')
    qa['human_visual_acceptance'] = record['human_visual_acceptance']
    qa['checks']['A_revision'] = record['panel_a']
    qa['checks'].pop('canonical_output_hashes_unchanged', None)
    qa['checks']['B_through_I_files_byte_identical_to_previous_formal'] = True
    write(BASE / 'visual_qa.json', qa)
    write(BASE / 'mechanism_validation.json', dict(record['panel_a'], unchanged_panels=record['unchanged_panels']))
    current.update({key: record[key] for key in ('status', 'human_visual_acceptance', 'updated_on',
        'producer', 'layout_version', 'published_from', 'previous_version_archive', 'panel_a')})
    current.update(author_designated_on='2026-10-10', accepted_panel_a_revision=accepted,
        outputs_sha256={str((BASE / name).relative_to(ROOT)): digest for name, digest in record['outputs'].items()})
    for name, digest in untouched.items():
        assert sha(BASE / name) == digest, name
    for name, digest in current['outputs_sha256'].items():
        assert sha(ROOT / name) == digest, name
    delivery = read(BASE / 'layout_delivery_validation.json')
    delivery.update(layout_version=VERSION, row_gaps_mm=qa['checks']['layout']['row_gaps_mm'],
        B_through_I_export_files_unchanged=untouched, human_visual_acceptance=record['human_visual_acceptance'],
        full_plate_matches_approved_A_candidate=True, archived_assets_verified=len(current['outputs_sha256']))
    write(BASE / 'layout_delivery_validation.json', delivery)
    manifest = read(BASE / 'release_manifest.json')
    manifest.update(layout_version=VERSION, updated_on='2026-10-10', producer=PRODUCER,
        previous_version_archive=str(ARCHIVE.relative_to(ROOT)),
        render_validation='Full v10 plate checked before promotion; B-I PNG/PDF/SVG retained byte-for-byte. See visual_qa.json and layout_delivery_validation.json.')
    write(BASE / 'release_manifest.json', manifest)
    temporary = BASE / 'current_version.json.tmp'
    write(temporary, current)
    temporary.replace(BASE / 'current_version.json')
    print('Published Figure 4A v10 and full plate; all 24 B-I export files unchanged.')


if __name__ == '__main__':
    main()
