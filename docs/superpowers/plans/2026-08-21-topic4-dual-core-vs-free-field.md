# Topic 4 dual-core versus free-field implementation plan

1. Freeze a one-candidate hand dual-core library by copying every non-Node
   mechanism field from final `node_baseline` and replacing only `h`.
2. Add deterministic exact-budget dual-core construction and tests for budget,
   center assignment, tie handling, and unchanged spline reconstruction.
3. Run the hand arm on seeds 1561-1572 with the final 20 s detector contract.
   Use numeric threads of one and memory-bounded parallel workers.
4. Rebuild the frozen patient train/held-out recording-block tables without
   changing labels or contact order.
5. Load both arms into one common scorer; compute fixed-direction, shaft-aware,
   event-cloud, KMeans, OOD and support endpoints at six and three events per
   mode.
6. Perform paired network bootstrap and recording-block patient sensitivity.
7. Render the field, KMeans/rank-profile and paired distribution comparison;
   inspect PNG/PDF and write the figure README after rendering.
8. Audit hashes, exact shared conditions, all worker completion, test results,
   and claim boundaries; then archive the scientific report.
