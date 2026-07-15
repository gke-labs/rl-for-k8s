"""Load PodSpec/NodeSpec from Kubernetes-style YAML manifests.

Supports the subset of the real Pod/Node schema that feeds scheduling
decisions (see DESIGN.md sec 6) -- not a full API model.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from schedgym.types import (
    ClusterBatch,
    LabelSelector,
    LabelSelectorRequirement,
    NodeAffinity,
    NodeSelectorTerm,
    NodeSpec,
    PodAffinity,
    PodAffinityTerm,
    PodSpec,
    SelectorOperator,
    Taint,
    TaintEffect,
    Toleration,
    TolerationOperator,
    TopologySpreadConstraint,
)

_MEM_SUFFIXES = {
    "Ki": 1024,
    "Mi": 1024**2,
    "Gi": 1024**3,
    "Ti": 1024**4,
    "Pi": 1024**5,
    "K": 1000,
    "M": 1000**2,
    "G": 1000**3,
    "T": 1000**4,
    "P": 1000**5,
}

_QUANTITY_RE = re.compile(r"^([0-9.eE+-]+)([A-Za-z]*)$")


def parse_quantity(value: str | int | float) -> float:
    """Parse a Kubernetes resource quantity (e.g. "500m", "512Mi", "2")
    into a float in base units (cores for cpu, bytes for memory, raw count
    for everything else, e.g. GPUs)."""
    if isinstance(value, (int, float)):
        return float(value)
    match = _QUANTITY_RE.match(value.strip())
    if not match:
        raise ValueError(f"cannot parse quantity {value!r}")
    number, suffix = match.groups()
    amount = float(number)
    if suffix == "":
        return amount
    if suffix == "m":
        return amount / 1000.0
    if suffix in _MEM_SUFFIXES:
        return amount * _MEM_SUFFIXES[suffix]
    raise ValueError(f"unknown quantity suffix {suffix!r} in {value!r}")


def _parse_label_selector(doc: dict | None) -> LabelSelector:
    if not doc:
        return LabelSelector()
    match_labels = dict(doc.get("matchLabels", {}))
    match_expressions = tuple(
        LabelSelectorRequirement(
            key=expr["key"],
            operator=SelectorOperator(expr["operator"]),
            values=tuple(expr.get("values", [])),
        )
        for expr in doc.get("matchExpressions", [])
    )
    return LabelSelector(match_labels=match_labels, match_expressions=match_expressions)


def _parse_node_selector_term(doc: dict) -> NodeSelectorTerm:
    match_expressions = tuple(
        LabelSelectorRequirement(
            key=expr["key"],
            operator=SelectorOperator(expr["operator"]),
            values=tuple(expr.get("values", [])),
        )
        for expr in doc.get("matchExpressions", [])
    )
    return NodeSelectorTerm(match_expressions=match_expressions)


def _parse_node_affinity(doc: dict | None) -> NodeAffinity | None:
    if not doc:
        return None
    required_doc = doc.get("requiredDuringSchedulingIgnoredDuringExecution")
    required = tuple(
        _parse_node_selector_term(term)
        for term in (required_doc.get("nodeSelectorTerms", []) if required_doc else [])
    )
    preferred = tuple(
        (entry["weight"], _parse_node_selector_term(entry["preference"]))
        for entry in doc.get("preferredDuringSchedulingIgnoredDuringExecution", [])
    )
    if not required and not preferred:
        return None
    return NodeAffinity(required=required, preferred=preferred)


def _parse_pod_affinity(doc: dict | None) -> PodAffinity | None:
    if not doc:
        return None
    required = tuple(
        PodAffinityTerm(
            label_selector=_parse_label_selector(term.get("labelSelector")),
            topology_key=term["topologyKey"],
        )
        for term in doc.get("requiredDuringSchedulingIgnoredDuringExecution", [])
    )
    preferred = tuple(
        (
            entry["weight"],
            PodAffinityTerm(
                label_selector=_parse_label_selector(entry["podAffinityTerm"].get("labelSelector")),
                topology_key=entry["podAffinityTerm"]["topologyKey"],
            ),
        )
        for entry in doc.get("preferredDuringSchedulingIgnoredDuringExecution", [])
    )
    if not required and not preferred:
        return None
    return PodAffinity(required=required, preferred=preferred)


def _sum_container_resources(containers: list[dict], key: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    for container in containers:
        requests = container.get("resources", {}).get("requests", {})
        for name, value in requests.items():
            totals[name] = totals.get(name, 0.0) + parse_quantity(value)
    return totals


def pod_from_manifest(doc: dict) -> PodSpec:
    metadata = doc.get("metadata", {})
    spec = doc.get("spec", {})
    containers = spec.get("containers", [])
    totals = _sum_container_resources(containers, "requests")

    host_ports = tuple(
        port["hostPort"]
        for container in containers
        for port in container.get("ports", [])
        if "hostPort" in port
    )

    tolerations = tuple(
        Toleration(
            key=tol.get("key"),
            operator=TolerationOperator(tol.get("operator", "Equal")),
            value=tol.get("value", ""),
            effect=TaintEffect(tol["effect"]) if tol.get("effect") else None,
        )
        for tol in spec.get("tolerations", [])
    )

    affinity = spec.get("affinity", {})
    topology_spread = tuple(
        TopologySpreadConstraint(
            topology_key=c["topologyKey"],
            max_skew=c["maxSkew"],
            label_selector=_parse_label_selector(c.get("labelSelector")),
            when_unsatisfiable=c.get("whenUnsatisfiable", "DoNotSchedule"),
        )
        for c in spec.get("topologySpreadConstraints", [])
    )

    return PodSpec(
        name=metadata["name"],
        cpu_request=totals.get("cpu", 0.0),
        memory_request=totals.get("memory", 0.0),
        gpu_request=int(totals.get("nvidia.com/gpu", 0.0)),
        ephemeral_storage_request=totals.get("ephemeral-storage", 0.0),
        labels=dict(metadata.get("labels", {})),
        tolerations=tolerations,
        node_selector=dict(spec.get("nodeSelector", {})),
        node_affinity=_parse_node_affinity(affinity.get("nodeAffinity")),
        pod_affinity=_parse_pod_affinity(affinity.get("podAffinity")),
        pod_anti_affinity=_parse_pod_affinity(affinity.get("podAntiAffinity")),
        topology_spread_constraints=topology_spread,
        host_ports=host_ports,
        priority=spec.get("priority", 0),
    )


def node_from_manifest(doc: dict) -> NodeSpec:
    metadata = doc.get("metadata", {})
    spec = doc.get("spec", {})
    status = doc.get("status", {})
    allocatable = status.get("allocatable", {})

    taints = tuple(
        Taint(
            key=t["key"],
            value=t.get("value", ""),
            effect=TaintEffect(t["effect"]),
        )
        for t in spec.get("taints", [])
    )

    return NodeSpec(
        name=metadata["name"],
        cpu_allocatable=parse_quantity(allocatable.get("cpu", 0)),
        memory_allocatable=parse_quantity(allocatable.get("memory", 0)),
        gpu_allocatable=int(parse_quantity(allocatable.get("nvidia.com/gpu", 0))),
        ephemeral_storage_allocatable=parse_quantity(allocatable.get("ephemeral-storage", 0)),
        labels=dict(metadata.get("labels", {})),
        taints=taints,
        unschedulable=bool(spec.get("unschedulable", False)),
    )


def _load_docs(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [doc for doc in yaml.safe_load_all(f) if doc]


def load_pods(path: str | Path) -> list[PodSpec]:
    return [pod_from_manifest(doc) for doc in _load_docs(path) if doc.get("kind") == "Pod"]


def load_nodes(path: str | Path) -> list[NodeSpec]:
    return [node_from_manifest(doc) for doc in _load_docs(path) if doc.get("kind") == "Node"]


def load_cluster(pods_path: str | Path, nodes_path: str | Path) -> ClusterBatch:
    return ClusterBatch(pods=tuple(load_pods(pods_path)), nodes=tuple(load_nodes(nodes_path)))
