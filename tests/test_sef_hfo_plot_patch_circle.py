import numpy as np

from src.sef_hfo_plot import two_electrode_readout


def test_two_electrode_readout_accepts_patch_circle(tmp_path):
    n = 200
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 3, size=(n, 2))
    t = np.arange(60) * 1.0
    sig = np.exp(-((t - 30) ** 2) / 50.0)[None, :].repeat(4, 0)
    shaft = dict(contacts=np.array([[0.8, 0.8], [1.0, 1.0], [1.2, 1.2], [1.4, 1.4]]),
                 part=np.ones(4, bool), s=np.arange(4, dtype=float),   # per-contact pos along shaft
                 signal=sig, label="P", panel_title="par")
    out = tmp_path / "x.png"
    two_electrode_readout(str(out), field_xy=xy, field_c=rng.random(n),
                          field_clabel="local V_th spread (mV)",
                          kick_xy=np.array([2.4, 1.5]), axis_deg=45.0,
                          extent=(0, 3, 0, 3), par=shaft, perp=shaft,
                          t=t, event_window=(5, 55), signal_ylabel="x",
                          substrate_label="x", contact_note="x",
                          patch_circle=(1.5, 1.5, 0.5))
    assert out.exists()                       # ran end-to-end with patch_circle
