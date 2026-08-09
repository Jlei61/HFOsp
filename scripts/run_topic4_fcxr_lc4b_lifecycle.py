#!/usr/bin/env python3
"""Run the locked LC4 F2 adjudicator against the LC4b result root/candidate."""
from pathlib import Path

import run_topic4_fcxr_lc3 as E01
import run_topic4_fcxr_lc4_lifecycle as LC4


LC4.OUT = str(Path(E01.OUT) / "lc4b_deadzone_lifecycle")


if __name__ == "__main__":
    LC4.main()
