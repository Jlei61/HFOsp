"""M3 plan Task 7: the hub-gated CLI flags are wired into the runner (fast --help check;
the actual default-OFF bit-parity + hub-ON behaviour are validated by the L=20 smoke runs
and tests/test_snn_hub_longrange.py, not re-run here)."""
import subprocess
import sys


def test_m3_flags_in_help():
    out = subprocess.run([sys.executable, "scripts/run_sef_hfo_snn_cm_spontaneous_readout.py", "--help"],
                         capture_output=True, text=True, timeout=120).stdout
    for flag in ["--hub-gain", "--hub-long-range-c", "--l-hub-long", "--degnorm-alpha",
                 "--degnorm-scheme", "--hub-theta-delta", "--corridor-half-frac",
                 "--hub-frac", "--global-gap-frac"]:
        assert flag in out, f"{flag} missing from runner --help"
