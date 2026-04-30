#!/usr/bin/env bash
# Cohort-level parallel driver for Epilepsiae new-pipeline pack + lagPat backfill.
#
# Plan: docs/archive/epilepsiae_lagpat/epilepsiae_lagpat_backfill_plan_2026-04-29.md
#       §3 Task B.5 Step 4.
#
# Each subject runs in its own Python process (one xargs slot); skip-existing
# is on by default, so re-running this after a SIGINT / SIGKILL resumes from
# wherever the previous run stopped without re-doing completed records.
#
# Memory: a 1024 Hz subject can hit ~9 GB RSS during the long loop. Verify
# `free -h` headroom (rule of thumb: N_JOBS x 10 GB <= 0.7 x MemAvailable)
# before raising N_JOBS above 5.
#
# Uses xargs -P (works without GNU parallel). Each worker tees its stdout to
# results/epilepsiae_lagpat_backfill/<subject>/_console.log so a multi-subject
# run does not interleave its line buffers in the foreground.
set -euo pipefail

cd "$(dirname "$0")/.."

N_JOBS=${N_JOBS:-5}
SUBJECTS="${SUBJECTS:-253 548 139 384 1077 1084 442 818 916 922 958 583 590 620 635 1073 1096 1125 1146 1150}"

mkdir -p results/epilepsiae_lagpat_backfill

worker() {
    local subj="$1"
    local out_dir="results/epilepsiae_lagpat_backfill/${subj}"
    mkdir -p "$out_dir"
    python scripts/run_epilepsiae_lagpat_backfill.py --subject "$subj" \
        >>"${out_dir}/_console.log" 2>&1
    echo "[backfill] subject=${subj} done"
}
export -f worker

# shellcheck disable=SC2086
printf '%s\n' $SUBJECTS \
    | xargs -n1 -P "$N_JOBS" -I {} bash -c 'worker "$@"' _ {}

python scripts/run_epilepsiae_lagpat_backfill.py --aggregate-cohort-summary
