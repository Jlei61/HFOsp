from __future__ import annotations

import numpy as np

from scripts.paper_figures import plot_fig2_shared_field_reversal_row as fig2_row


def _rows() -> list[dict]:
    subject_ids = [
        *fig2_row.EXAMPLE_SUBJECT_IDS,
        "test_04", "test_05", "test_06", "test_07",
        "test_08", "test_09", "test_10", "test_11",
    ]
    values = [-0.44, -0.42, -0.28, -0.77, -0.90, -0.80, -0.70, -0.60,
              0.10, 0.30, 0.50, 0.70]
    rows = []
    for index, (subject_id, r_value) in enumerate(zip(subject_ids, values)):
        rows.append(
            {
                "subject_id": subject_id,
                "display_id": f"T{index:02d}",
                "record": {"subject_id": subject_id},
                "n_contacts": 4,
                "r": r_value,
            }
        )
    return rows


def test_examples_are_the_locked_legible_negative_cases() -> None:
    examples = fig2_row.select_examples(list(reversed(_rows())))
    assert [row["subject_id"] for row in examples] == list(fig2_row.EXAMPLE_SUBJECT_IDS)
    assert all(row["r"] < 0 for row in examples)


def test_cohort_shift_null_is_lower_tailed_and_reproducible() -> None:
    rows = _rows()
    rng = np.random.default_rng(22)
    nulls = {
        row["subject_id"]: rng.normal(0.0, 0.15, size=500)
        for row in rows
    }
    first, first_draws = fig2_row.build_cohort_shift_null(
        rows, nulls, n_draws=2_000, base_seed=7,
    )
    second, second_draws = fig2_row.build_cohort_shift_null(
        list(reversed(rows)), nulls, n_draws=2_000, base_seed=7,
    )
    assert first == second
    assert np.array_equal(first_draws, second_draws)
    assert first["observed_median_delta"] < 0
    assert first["p_negative"] < 0.05


def test_build_figure_writes_field_examples_cohort_and_null(tmp_path, monkeypatch) -> None:
    rows = _rows()
    rng = np.random.default_rng(4)
    nulls = {
        row["subject_id"]: rng.normal(0.0, 0.2, size=300)
        for row in rows
    }

    def fake_payloads(record, *, display_sigma_mm):
        payload = {
            "subject_id": record["subject_id"],
            "xs": np.asarray([-5.0, 5.0]),
            "ys": np.asarray([-10.0, 10.0]),
            "frame": {"xlim": (-20.0, 20.0), "ylim": (-20.0, 20.0)},
        }
        return (dict(payload), dict(payload), "shared")

    def fake_draw(ax, payload, template, **kwargs):
        value = 0.25 if template == "TA" else 0.75
        ax.imshow(np.full((2, 2), value), cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])

    monkeypatch.setattr(fig2_row, "build_interictal_ab_panel_payloads", fake_payloads)
    monkeypatch.setattr(fig2_row, "draw_interictal_rank_field_panel", fake_draw)
    metadata = fig2_row.build_figure(
        rows,
        channel_nulls=nulls,
        out_dir=tmp_path,
        seed=7,
        n_cohort_draws=2_000,
    )
    assert metadata["grouping"] == "none"
    assert metadata["distribution"]["n"] == 12
    assert metadata["distribution"]["n_negative"] == 8
    assert metadata["full_contact_shuffle"]["n_draws"] == 2_000
    assert (tmp_path / "figures/fig2_shared_field_reversal_last_row.png").exists()
    assert (tmp_path / "figures/fig2_shared_field_reversal_last_row.pdf").exists()
    assert (tmp_path / "figures/fig2_shared_field_reversal_last_row_metadata.json").exists()
    assert (tmp_path / "fig2_shared_field_reversal_cohort_null.npz").exists()
