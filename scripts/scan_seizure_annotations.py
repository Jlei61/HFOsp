"""Scan all Yuquan subjects for seizure annotations using PR1 functions."""
import sys, os, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(str(Path(__file__).resolve().parent.parent))

from src.preprocessing import (
    fast_read_edf_annotations,
    parse_seizure_annotation_events,
    read_edf_record_info,
)

DATA_ROOT = Path('/mnt/yuquan_data/yuquan_24h_edf')
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)
SEIZURE_DETECTION_DIR = RESULTS_DIR / 'seizure_detection'
SEIZURE_DETECTION_DIR.mkdir(exist_ok=True)

SEIZURE_LABELS = [
    'EEG SZ', 'SZ', 'SZ1', 'SZ2', 'SZ3', 'SZ4', 'SZ5', 'SZ6', 'SZ7',
    'SZ8', 'SZ9', 'SZ10',
    'EEG onset', 'seizure', 'Seizure', 'SEIZURE',
    'onset', 'Onset', 'ictal', 'Ictal',
    'sz onset', 'seizure onset', 'clinical seizure',
    'subclinical seizure', 'electrographic seizure',
]

def main():
    subjects = sorted([d.name for d in DATA_ROOT.iterdir() if d.is_dir()])
    print(f"=== Yuquan All-Subject Seizure Annotation Scan ===", flush=True)
    print(f"Subjects: {len(subjects)}", flush=True)

    global_summary = {}
    total_edfs = 0
    total_interval_edfs = 0
    total_orphan_only_edfs = 0
    total_intervals = 0
    total_orphan_onsets = 0
    total_duration_offsets = 0
    total_end_label_offsets = 0

    for subj in subjects:
        subj_dir = DATA_ROOT / subj
        edfs = sorted(subj_dir.glob('*.edf'))
        if not edfs:
            print(f"\n[{subj}] no EDFs found, skipping", flush=True)
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"[{subj}] {len(edfs)} EDFs", flush=True)

        subj_files = []
        subj_all_labels = set()
        subj_interval_count = 0
        subj_orphan_only_count = 0

        for edf_path in edfs:
            record = edf_path.stem
            t0 = time.time()
            try:
                anns = fast_read_edf_annotations(edf_path)
                descs = list(set(d for _, _, d in anns if d.strip()))
                subj_all_labels.update(descs)

                info = read_edf_record_info(edf_path)
                start_epoch = float(info["start_epoch"])
                parsed = parse_seizure_annotation_events(
                    edf_path, SEIZURE_LABELS, start_epoch
                )
                sz_intervals = list(parsed["intervals"])
                orphan_onsets = list(parsed["orphan_onsets"])
                raw_details = list(parsed["raw_interval_details"])
                duration_offsets = sum(
                    1 for x in raw_details if x.get("offset_source") == "duration"
                )
                end_label_offsets = sum(
                    1 for x in raw_details if x.get("offset_source") == "end_label"
                )
                elapsed = time.time() - t0

                if sz_intervals or orphan_onsets:
                    if sz_intervals:
                        subj_interval_count += 1
                        total_intervals += len(sz_intervals)
                    else:
                        subj_orphan_only_count += 1
                    total_orphan_onsets += len(orphan_onsets)
                    total_duration_offsets += duration_offsets
                    total_end_label_offsets += end_label_offsets
                    print(
                        f"  {record}: {len(anns)} anns, {len(sz_intervals)} intervals, "
                        f"{len(orphan_onsets)} orphan onsets, {elapsed:.1f}s  ***",
                        flush=True,
                    )
                    for i, (on, off) in enumerate(sz_intervals):
                        rel_on = on - start_epoch
                        rel_off = off - start_epoch
                        dur = off - on
                        print(
                            f"    SZ{i}: onset={rel_on:.1f}s offset={rel_off:.1f}s dur={dur:.1f}s",
                            flush=True,
                        )
                    for i, on in enumerate(orphan_onsets):
                        print(
                            f"    orphan{i}: onset={on - start_epoch:.1f}s",
                            flush=True,
                        )
                else:
                    print(f"  {record}: {len(anns)} anns, 0 seizure markers, {elapsed:.1f}s", flush=True)

                subj_files.append({
                    'record': record,
                    'elapsed_sec': round(elapsed, 4),
                    'n_annotations': len(anns),
                    'descriptions': sorted(descs),
                    'record_info': info,
                    'n_seizure_intervals': len(sz_intervals),
                    'n_orphan_onsets': len(orphan_onsets),
                    'n_duration_offsets': duration_offsets,
                    'n_end_label_offsets': end_label_offsets,
                    'seizure_intervals': [
                        {
                            'onset_epoch': o, 'offset_epoch': f,
                            'onset_rel_sec': round(o - start_epoch, 2),
                            'offset_rel_sec': round(f - start_epoch, 2),
                            'duration_sec': round(f - o, 2),
                        }
                        for o, f in sz_intervals
                    ],
                    'orphan_onsets': [
                        {
                            'onset_epoch': o,
                            'onset_rel_sec': round(o - start_epoch, 2),
                        }
                        for o in orphan_onsets
                    ],
                    'interval_details': raw_details,
                })
            except Exception as e:
                elapsed = time.time() - t0
                print(f"  {record}: ERROR {e} ({elapsed:.1f}s)", flush=True)
                subj_files.append({
                    'record': record,
                    'elapsed_sec': round(elapsed, 4),
                    'error': str(e),
                })

        total_edfs += len(edfs)
        total_interval_edfs += subj_interval_count
        total_orphan_only_edfs += subj_orphan_only_count

        subj_result = {
            'subject': subj,
            'n_edfs': len(edfs),
            'n_interval_edfs': subj_interval_count,
            'n_orphan_only_edfs': subj_orphan_only_count,
            'all_unique_labels': sorted(subj_all_labels),
            'seizure_labels_used': SEIZURE_LABELS,
            'files': subj_files,
        }
        global_summary[subj] = {
            'n_edfs': len(edfs),
            'n_interval_edfs': subj_interval_count,
            'n_orphan_only_edfs': subj_orphan_only_count,
            'n_sz_intervals': sum(f.get('n_seizure_intervals', 0) for f in subj_files),
            'n_orphan_onsets': sum(f.get('n_orphan_onsets', 0) for f in subj_files),
            'n_duration_offsets': sum(f.get('n_duration_offsets', 0) for f in subj_files),
            'n_end_label_offsets': sum(f.get('n_end_label_offsets', 0) for f in subj_files),
            'all_unique_labels': sorted(subj_all_labels),
        }

        out_json = SEIZURE_DETECTION_DIR / f'pr1_seizure_{subj}.json'
        with open(out_json, 'w') as fp:
            json.dump(subj_result, fp, indent=2, ensure_ascii=False)
        print(f"  -> saved {out_json}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"=== GLOBAL SUMMARY ===", flush=True)
    print(f"Total subjects: {len(subjects)}", flush=True)
    print(f"Total EDFs: {total_edfs}", flush=True)
    print(f"Total interval-bearing EDFs: {total_interval_edfs}", flush=True)
    print(f"Total orphan-only EDFs: {total_orphan_only_edfs}", flush=True)
    print(f"Total seizure intervals: {total_intervals}", flush=True)
    print(f"Total orphan onsets: {total_orphan_onsets}", flush=True)
    print(
        f"Offset quality: duration={total_duration_offsets}, end_label={total_end_label_offsets}",
        flush=True,
    )

    print(f"\nPer-subject seizure counts:", flush=True)
    for subj in subjects:
        info = global_summary.get(subj, {})
        n_sz = info.get('n_interval_edfs', 0)
        n_orphan_only = info.get('n_orphan_only_edfs', 0)
        n_int = info.get('n_sz_intervals', 0)
        flag = ' ***' if n_sz > 0 else ''
        print(
            f"  {subj:20s}: {info.get('n_edfs',0):3d} EDFs, "
            f"{n_sz:2d} interval-EDFs, {n_orphan_only:2d} orphan-only, "
            f"{n_int:3d} intervals{flag}",
            flush=True,
        )

    summary_json = SEIZURE_DETECTION_DIR / 'pr1_seizure_all_yuquan_summary.json'
    with open(summary_json, 'w') as fp:
        json.dump(global_summary, fp, indent=2, ensure_ascii=False)
    print(f"\n-> summary saved to {summary_json}", flush=True)
    print(f"=== DONE ===", flush=True)


if __name__ == '__main__':
    main()
