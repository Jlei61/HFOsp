# Patient multimode selection-corrected null + 916 recruitment-extent endpoint

Archived source for the 2026-08-16 patient-side closeout. See
`docs/archive/topic1/propagation/multimode_selection_null_and_916_extent_2026-08-16.md`.

**These files were executed from
`results/interictal_propagation_masked/multimode_grammar_audit/`**, not from here.
They resolve the repository root as `Path(__file__).resolve().parents[3]` and import
each other from their own directory, so to re-run them, copy them back into that
results directory (which is covered by the `results/*` gitignore rule, hence this
tracked copy). They also depend on `run_multimode_grammar_audit.py` from the
2026-08-15 pass, which lives in the same results directory.

Safety note carried over from the run: the concurrently running formal
data-driven SNN cohort worker hashes every module it imports against commit
`96618174`, and `src/interictal_propagation.py` is inside that set. Nothing here
writes to `src/` or `scripts/` at run time; all outputs go under the results
directory.

| file | role |
|---|---|
| `run_selection_corrected_null.py` | primary order-randomised null at frozen K + the superseded shuffle-and-repair sensitivity null |
| `run_marginal_maxent_null.py` | replacement contact-marginal-preserving null (enumerated p!, max-entropy IPF) |
| `run_null_construction_audit.py` | does each null keep/destroy what it claims? |
| `run_916_extent_endpoint.py` | train-only recruitment-extent split transferred to held-out blocks |
| `test_null_constructions.py` | 17 invariant tests |
| `monitor_formal_cohort.py` | read-only 450 s watcher for the formal cohort, including frozen-module drift |
| `plot_selection_null_and_extent.py` | all figures |
| `make_report_tables.py` | regenerates the archive doc's tables from the stored JSON |
| `launch_patient_jobs.sh` | nohup + PID/log/status wrapper, numeric threads pinned to 1 |
