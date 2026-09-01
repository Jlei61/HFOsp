import numpy as np

from scripts.rescore_topic4_rev10_sa_historical_artifacts import (
    score_mode_conditioned_events,
)
from src.topic4_shaft_aware import (
    build_contact_contract,
    contract_groups,
    contract_pairs,
    describe_events,
    fit_patient_embedding,
)


def test_score_keeps_zero_scl_recruitment_as_a_finite_failure():
    names = [f"ICL{i}" for i in range(1, 12)] + [f"SCL{i}" for i in range(6, 10)]
    contract = build_contact_contract(
        names, np.column_stack([np.arange(15.0), np.zeros(15)]),
        np.arange(15.0), {"test": True},
    )
    groups, pairs = contract_groups(contract), contract_pairs(contract)
    rng = np.random.default_rng(3)
    patient = rng.normal(size=(40, 15))
    features = __import__(
        "src.topic4_shaft_aware", fromlist=["build_event_features"]
    ).build_event_features(patient, groups)["features"]
    embedding = fit_patient_embedding(
        features, max_components=8, reference_n=30, n_directions=8, seed=4,
    )
    targets = {}
    for mode in (0, 1):
        rows = patient[mode * 20:(mode + 1) * 20]
        target_features = __import__(
            "src.topic4_shaft_aware", fromlist=["build_event_features"]
        ).build_event_features(rows, groups)["features"]
        targets[mode] = {
            "descriptor": describe_events(rows, groups, pairs),
            "reference_z": __import__(
                "src.topic4_shaft_aware", fromlist=["transform_patient_embedding"]
            ).transform_patient_embedding(target_features, embedding),
        }
    floor = {
        key: {"median": 0.0, "q95": 0.1}
        for key in [
            "recruitment.ICL", "recruitment.SCL", "precedence.ICL-ICL",
            "precedence.SCL-SCL", "precedence.ICL-SCL", "profile.ICL",
            "profile.SCL", "profile.cross", "event_cloud", "multishaft_fraction",
        ]
    }
    config = {"floors": {"shaft_tau": 0.25, "pair_tau": 0.25, "profile_tau": 0.25}}
    model = patient[:12].copy()
    model[:, groups["SCL"]] = np.nan
    result = score_mode_conditioned_events(
        model, np.r_[np.zeros(6, int), np.ones(6, int)], groups=groups, pairs=pairs,
        embedding=embedding, targets=targets, floors={"0": floor, "1": floor},
        config=config,
    )
    assert result["status"] == "OK"
    assert result["pooled_multishaft_fraction"] == 0.0
    assert result["modes"]["0"]["raw"]["recruitment.SCL"] > 0.0
    assert np.isfinite(result["weak_mode_score"])
