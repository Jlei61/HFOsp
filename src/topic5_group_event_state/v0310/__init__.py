"""v0.3.10 training-sufficiency, capacity and rare-seizure layer.

Nothing here re-derives measurements. Histories are rebuilt from the frozen
replay blocks of the v0.3.9 bundles and are checked bit-for-bit against the
stored H=0.5/H=8 rows before any new horizon is trusted.
"""
