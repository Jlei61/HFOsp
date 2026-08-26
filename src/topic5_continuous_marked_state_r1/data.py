"""R1 event stream with strict pre-event marks and development splits."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

import numpy as np

from . import contract


@dataclass(frozen=True)
class R1EventStream:
    subject: str
    dataset: str
    event_time: np.ndarray
    split: np.ndarray
    session: np.ndarray
    participation: np.ndarray
    group_ids: np.ndarray
    group_count: np.ndarray
    load: np.ndarray
    contact_names: np.ndarray
    contact_features: np.ndarray
    adjacency: np.ndarray
    source_hashes: dict[str, str]

    @property
    def n_events(self) -> int:
        return int(len(self.event_time))

    @property
    def n_contacts(self) -> int:
        return int(self.participation.shape[1])

    def mask(self, split: str) -> np.ndarray:
        code = {"train": 0, "validation": 1}[split]
        return self.split == code

    def validate(self) -> None:
        n = self.n_events
        if any(len(value) != n for value in (
            self.split, self.session, self.participation, self.group_ids,
            self.group_count, self.load,
        )):
            raise ValueError(f"{self.subject}: unequal event arrays")
        if self.participation.shape != self.group_ids.shape:
            raise ValueError(f"{self.subject}: mark shapes disagree")
        if np.any(np.diff(self.event_time) < 0):
            raise ValueError(f"{self.subject}: non-chronological events")
        if np.any(self.group_ids[~self.participation] != -1):
            raise ValueError(f"{self.subject}: phantom group id")
        for event in range(n):
            labels = np.unique(self.group_ids[event, self.participation[event]])
            expected = np.arange(int(self.group_count[event]))
            if not np.array_equal(labels, expected):
                raise ValueError(f"{self.subject}: non-dense group ids at event {event}")
        if set(np.unique(self.split).tolist()) - {0, 1, 2}:
            raise ValueError(f"{self.subject}: unknown split code")
        for name in ("train", "validation"):
            values = self.event_time[self.mask(name)]
            if len(values):
                contract.assert_development_times(self.subject, values, name)


def load_event_stream(subject: str) -> R1EventStream:
    """Load the frozen mark source but apply the strict raw-SEEG time boundary."""
    from src.topic5_epi_prssm.event_marks import load_patient
    from src.topic5_epi_prssm.graph_templates import build_patient_graph

    events = load_patient(subject)
    train_end, dev_end = contract.load_split(subject)
    split = np.full(events.n_events, 2, dtype=np.int8)
    split[events.event_time < dev_end] = 1
    split[events.event_time < train_end] = 0
    # Rebuild the graph on the same strict boundary used by R1.  The upstream
    # event package treats the event exactly at ``train_end_epoch`` as TRAIN,
    # whereas the raw-SEEG contract uses a strict ``time < train_end`` rule.
    # Reusing the upstream graph would therefore leak one R1-validation event.
    graph = build_patient_graph(replace(events, split=split))
    value = R1EventStream(
        subject=subject,
        dataset=events.dataset,
        event_time=np.asarray(events.event_time, dtype=np.float64),
        split=split,
        session=np.asarray(events.sessions.session_index, dtype=np.int64),
        participation=np.asarray(events.participation, dtype=bool),
        group_ids=np.asarray(events.group_ids, dtype=np.int64),
        group_count=np.asarray(events.group_count, dtype=np.int64),
        load=np.asarray(events.load, dtype=np.float32),
        contact_names=np.asarray(events.contact_names).astype(str),
        contact_features=np.asarray(events.contact_features, dtype=np.float32),
        adjacency=np.asarray(graph.stack(), dtype=np.float32),
        source_hashes=dict(events.source_hashes),
    )
    value.validate()
    return value
