#!/usr/bin/env bash
set -uo pipefail

if [ "$#" -lt 5 ]; then
    printf 'usage: %s STATUS LOG TITLE COMMIT COMMAND...\n' "$0" >&2
    exit 2
fi

STATUS="$1"
LOG="$2"
TITLE="$3"
COMMIT="$4"
shift 4

write_status() {
    local temporary="${STATUS}.tmp.$$"
    printf '%s\n' "$1" > "$temporary"
    mv "$temporary" "$STATUS"
}

finish() {
    local code=$?
    trap - EXIT
    if [ "$code" -eq 0 ]; then
        write_status "SUCCESS exit_code=0 finished_at=$(date --iso-8601=seconds) commit=$COMMIT"
        notify-send "Topic 4 rev10-SA" "$TITLE completed" >/dev/null 2>&1 || true
    else
        write_status "FAILED exit_code=$code finished_at=$(date --iso-8601=seconds) commit=$COMMIT"
        notify-send "Topic 4 rev10-SA" "$TITLE failed (exit $code)" >/dev/null 2>&1 || true
    fi
    exit "$code"
}

terminate() {
    trap - TERM INT
    if [ -n "${child_pid:-}" ]; then
        kill -TERM "$child_pid" >/dev/null 2>&1 || true
        wait "$child_pid" || true
    fi
    exit 143
}

mkdir -p "$(dirname "$STATUS")" "$(dirname "$LOG")"
trap finish EXIT
trap terminate TERM INT
write_status "RUNNING pid=$$ started_at=$(date --iso-8601=seconds) commit=$COMMIT"
"$@" >> "$LOG" 2>&1 &
child_pid=$!
wait "$child_pid"
exit $?
