"""Internal pod/node data model.

A deliberately reduced mirror of the Kubernetes Pod/Node API objects: only
the fields that feed into scheduling decisions (see DESIGN.md secs 3 and 6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TaintEffect(str, Enum):
    NO_SCHEDULE = "NoSchedule"
    PREFER_NO_SCHEDULE = "PreferNoSchedule"
    NO_EXECUTE = "NoExecute"


class TolerationOperator(str, Enum):
    EQUAL = "Equal"
    EXISTS = "Exists"


class SelectorOperator(str, Enum):
    IN = "In"
    NOT_IN = "NotIn"
    EXISTS = "Exists"
    DOES_NOT_EXIST = "DoesNotExist"


@dataclass(frozen=True)
class Taint:
    key: str
    value: str = ""
    effect: TaintEffect = TaintEffect.NO_SCHEDULE


@dataclass(frozen=True)
class Toleration:
    key: str | None = None
    operator: TolerationOperator = TolerationOperator.EQUAL
    value: str = ""
    effect: TaintEffect | None = None

    def tolerates(self, taint: Taint) -> bool:
        if self.effect is not None and self.effect != taint.effect:
            return False
        if self.operator == TolerationOperator.EXISTS:
            return self.key is None or self.key == taint.key
        return self.key == taint.key and self.value == taint.value


@dataclass(frozen=True)
class LabelSelectorRequirement:
    key: str
    operator: SelectorOperator
    values: tuple[str, ...] = ()

    def matches(self, labels: dict[str, str]) -> bool:
        if self.operator == SelectorOperator.EXISTS:
            return self.key in labels
        if self.operator == SelectorOperator.DOES_NOT_EXIST:
            return self.key not in labels
        if self.key not in labels:
            return False
        if self.operator == SelectorOperator.IN:
            return labels[self.key] in self.values
        if self.operator == SelectorOperator.NOT_IN:
            return labels[self.key] not in self.values
        raise ValueError(f"unsupported operator {self.operator}")


@dataclass(frozen=True)
class LabelSelector:
    match_labels: dict[str, str] = field(default_factory=dict)
    match_expressions: tuple[LabelSelectorRequirement, ...] = ()

    def matches(self, labels: dict[str, str]) -> bool:
        for k, v in self.match_labels.items():
            if labels.get(k) != v:
                return False
        return all(req.matches(labels) for req in self.match_expressions)


@dataclass(frozen=True)
class NodeSelectorTerm:
    """AND of requirements within a term."""

    match_expressions: tuple[LabelSelectorRequirement, ...] = ()

    def matches(self, labels: dict[str, str]) -> bool:
        return all(req.matches(labels) for req in self.match_expressions)


@dataclass(frozen=True)
class NodeAffinity:
    """OR of required terms; preferred terms are (weight, term) soft hints."""

    required: tuple[NodeSelectorTerm, ...] = ()
    preferred: tuple[tuple[int, NodeSelectorTerm], ...] = ()

    def satisfies_required(self, labels: dict[str, str]) -> bool:
        if not self.required:
            return True
        return any(term.matches(labels) for term in self.required)


@dataclass(frozen=True)
class PodAffinityTerm:
    label_selector: LabelSelector
    topology_key: str


@dataclass(frozen=True)
class PodAffinity:
    """Used for both pod-affinity and pod-anti-affinity; semantics differ
    only in how the caller interprets a match (attract vs. repel)."""

    required: tuple[PodAffinityTerm, ...] = ()
    preferred: tuple[tuple[int, PodAffinityTerm], ...] = ()


@dataclass(frozen=True)
class TopologySpreadConstraint:
    topology_key: str
    max_skew: int
    label_selector: LabelSelector
    when_unsatisfiable: str = "DoNotSchedule"  # or "ScheduleAnyway"


@dataclass(frozen=True)
class PodSpec:
    name: str
    cpu_request: float  # cores
    memory_request: float  # bytes
    gpu_request: int = 0
    ephemeral_storage_request: float = 0.0
    labels: dict[str, str] = field(default_factory=dict)
    tolerations: tuple[Toleration, ...] = ()
    node_selector: dict[str, str] = field(default_factory=dict)
    node_affinity: NodeAffinity | None = None
    pod_affinity: PodAffinity | None = None
    pod_anti_affinity: PodAffinity | None = None
    topology_spread_constraints: tuple[TopologySpreadConstraint, ...] = ()
    host_ports: tuple[int, ...] = ()
    priority: int = 0


@dataclass(frozen=True)
class ExistingPod:
    """A pod already bound to a node before the batch being scheduled --
    background load the simulated cluster starts with."""

    name: str
    cpu_request: float
    memory_request: float
    gpu_request: int = 0
    ephemeral_storage_request: float = 0.0
    labels: dict[str, str] = field(default_factory=dict)
    host_ports: tuple[int, ...] = ()


@dataclass(frozen=True)
class NodeSpec:
    name: str
    cpu_allocatable: float
    memory_allocatable: float
    gpu_allocatable: int = 0
    ephemeral_storage_allocatable: float = 0.0
    labels: dict[str, str] = field(default_factory=dict)
    taints: tuple[Taint, ...] = ()
    unschedulable: bool = False
    existing_pods: tuple[ExistingPod, ...] = ()


@dataclass(frozen=True)
class ClusterBatch:
    """One scheduling episode: the pods to place and the nodes available."""

    pods: tuple[PodSpec, ...]
    nodes: tuple[NodeSpec, ...]
