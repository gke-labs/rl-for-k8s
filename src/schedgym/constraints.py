"""Pure-Python constraint validator -- the source-of-truth oracle for
assignment feasibility (see DESIGN.md sec 3/5/9).

Deliberately written for clarity over speed: this is the reference used
in tests to check the JAX-vectorized version (jax_constraints.py) is
correct, and to score baselines / evaluation runs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from schedgym.types import ClusterBatch, NodeSpec, PodSpec

UNASSIGNED = -1


@dataclass(frozen=True)
class ConstraintViolation:
    kind: str
    detail: str
    pod_name: str | None = None
    node_name: str | None = None


@dataclass
class ValidationResult:
    violations: list[ConstraintViolation] = field(default_factory=list)
    active_node_count: int = 0
    packing_density: float = 0.0

    @property
    def is_feasible(self) -> bool:
        return len(self.violations) == 0


def _matching_pod_labels(cluster: ClusterBatch, assignment: Sequence[int]) -> list[tuple[str, dict, str]]:
    """(source_id, labels, node_name) for every pod in the cluster after
    `assignment` is applied -- both newly scheduled pods and pre-existing
    background load. `source_id` lets callers exclude a pod from matching
    against itself."""
    entries: list[tuple[str, dict, str]] = []
    for pod, node_idx in zip(cluster.pods, assignment):
        if node_idx == UNASSIGNED:
            continue
        entries.append((pod.name, pod.labels, cluster.nodes[node_idx].name))
    for node in cluster.nodes:
        for existing in node.existing_pods:
            entries.append((f"existing:{node.name}:{existing.name}", existing.labels, node.name))
    return entries


def _node_by_name(cluster: ClusterBatch) -> dict[str, NodeSpec]:
    return {n.name: n for n in cluster.nodes}


def _check_resources(cluster: ClusterBatch, assignment: Sequence[int]) -> list[ConstraintViolation]:
    violations = []
    usage: dict[int, dict[str, float]] = defaultdict(lambda: {"cpu": 0.0, "memory": 0.0, "gpu": 0.0, "ephemeral_storage": 0.0})
    for node_idx, node in enumerate(cluster.nodes):
        for existing in node.existing_pods:
            u = usage[node_idx]
            u["cpu"] += existing.cpu_request
            u["memory"] += existing.memory_request
            u["gpu"] += existing.gpu_request
            u["ephemeral_storage"] += existing.ephemeral_storage_request

    for pod, node_idx in zip(cluster.pods, assignment):
        if node_idx == UNASSIGNED:
            continue
        u = usage[node_idx]
        u["cpu"] += pod.cpu_request
        u["memory"] += pod.memory_request
        u["gpu"] += pod.gpu_request
        u["ephemeral_storage"] += pod.ephemeral_storage_request

    for node_idx, node in enumerate(cluster.nodes):
        u = usage[node_idx]
        checks = (
            ("cpu", node.cpu_allocatable),
            ("memory", node.memory_allocatable),
            ("gpu", node.gpu_allocatable),
            ("ephemeral_storage", node.ephemeral_storage_allocatable),
        )
        for resource, allocatable in checks:
            if u[resource] > allocatable + 1e-9:
                violations.append(
                    ConstraintViolation(
                        kind="resource_overcommit",
                        node_name=node.name,
                        detail=f"{resource} requested={u[resource]:.3f} allocatable={allocatable:.3f}",
                    )
                )
    return violations


def _check_taints_and_selectors(cluster: ClusterBatch, assignment: Sequence[int]) -> list[ConstraintViolation]:
    violations = []
    for pod, node_idx in zip(cluster.pods, assignment):
        if node_idx == UNASSIGNED:
            continue
        node = cluster.nodes[node_idx]

        if node.unschedulable:
            violations.append(
                ConstraintViolation("unschedulable_node", f"node {node.name} is unschedulable", pod.name, node.name)
            )

        for taint in node.taints:
            if taint.effect == taint.effect.PREFER_NO_SCHEDULE:
                continue
            if not any(tol.tolerates(taint) for tol in pod.tolerations):
                violations.append(
                    ConstraintViolation(
                        "taint_toleration",
                        f"pod does not tolerate taint {taint.key}={taint.value}:{taint.effect.value}",
                        pod.name,
                        node.name,
                    )
                )

        for key, value in pod.node_selector.items():
            if node.labels.get(key) != value:
                violations.append(
                    ConstraintViolation(
                        "node_selector",
                        f"node missing/mismatched label {key}={value}",
                        pod.name,
                        node.name,
                    )
                )

        if pod.node_affinity is not None and not pod.node_affinity.satisfies_required(node.labels):
            violations.append(
                ConstraintViolation("node_affinity", "required node affinity not satisfied", pod.name, node.name)
            )
    return violations


def _check_host_ports(cluster: ClusterBatch, assignment: Sequence[int]) -> list[ConstraintViolation]:
    violations = []
    used_ports: dict[int, set[int]] = defaultdict(set)
    for node_idx, node in enumerate(cluster.nodes):
        for existing in node.existing_pods:
            used_ports[node_idx].update(existing.host_ports)

    for pod, node_idx in zip(cluster.pods, assignment):
        if node_idx == UNASSIGNED:
            continue
        node = cluster.nodes[node_idx]
        conflicts = used_ports[node_idx] & set(pod.host_ports)
        if conflicts:
            violations.append(
                ConstraintViolation("host_port_conflict", f"ports {conflicts} already in use", pod.name, node.name)
            )
        used_ports[node_idx].update(pod.host_ports)
    return violations


def _check_pod_affinity(cluster: ClusterBatch, assignment: Sequence[int]) -> list[ConstraintViolation]:
    violations = []
    all_entries = _matching_pod_labels(cluster, assignment)

    for pod, node_idx in zip(cluster.pods, assignment):
        if node_idx == UNASSIGNED:
            continue
        node = cluster.nodes[node_idx]
        others = [(labels, node_name) for source_id, labels, node_name in all_entries if source_id != pod.name]

        if pod.pod_affinity is not None:
            for term in pod.pod_affinity.required:
                topo_value = node.labels.get(term.topology_key)
                satisfied = any(
                    term.label_selector.matches(labels) and _node_by_name(cluster)[other_node].labels.get(term.topology_key) == topo_value
                    for labels, other_node in others
                )
                if not satisfied:
                    violations.append(
                        ConstraintViolation(
                            "pod_affinity",
                            f"no matching pod sharing topology domain {term.topology_key}={topo_value}",
                            pod.name,
                            node.name,
                        )
                    )

        if pod.pod_anti_affinity is not None:
            for term in pod.pod_anti_affinity.required:
                topo_value = node.labels.get(term.topology_key)
                conflicting = [
                    other_node
                    for labels, other_node in others
                    if term.label_selector.matches(labels)
                    and _node_by_name(cluster)[other_node].labels.get(term.topology_key) == topo_value
                ]
                if conflicting:
                    violations.append(
                        ConstraintViolation(
                            "pod_anti_affinity",
                            f"conflicting pod(s) share topology domain {term.topology_key}={topo_value}",
                            pod.name,
                            node.name,
                        )
                    )
    return violations


def _check_topology_spread(cluster: ClusterBatch, assignment: Sequence[int]) -> list[ConstraintViolation]:
    violations = []
    all_entries = _matching_pod_labels(cluster, assignment)
    node_by_name = _node_by_name(cluster)

    seen_constraints: set[tuple] = set()
    for pod in cluster.pods:
        for c in pod.topology_spread_constraints:
            signature = (c.topology_key, tuple(sorted(c.label_selector.match_labels.items())), c.max_skew, c.when_unsatisfiable)
            if signature in seen_constraints:
                continue
            seen_constraints.add(signature)

            domains = {node.labels[c.topology_key] for node in cluster.nodes if c.topology_key in node.labels}
            if not domains:
                continue
            counts = {d: 0 for d in domains}
            for _source_id, labels, node_name in all_entries:
                if c.label_selector.matches(labels):
                    domain = node_by_name[node_name].labels.get(c.topology_key)
                    if domain in counts:
                        counts[domain] += 1
            skew = max(counts.values()) - min(counts.values())
            if skew > c.max_skew and c.when_unsatisfiable == "DoNotSchedule":
                violations.append(
                    ConstraintViolation(
                        "topology_spread",
                        f"skew={skew} exceeds maxSkew={c.max_skew} for {c.topology_key} counts={counts}",
                    )
                )
    return violations


def _packing_stats(cluster: ClusterBatch, assignment: Sequence[int]) -> tuple[int, float]:
    usage: dict[int, dict[str, float]] = defaultdict(lambda: {"cpu": 0.0, "memory": 0.0})
    for node_idx, node in enumerate(cluster.nodes):
        for existing in node.existing_pods:
            usage[node_idx]["cpu"] += existing.cpu_request
            usage[node_idx]["memory"] += existing.memory_request
    for pod, node_idx in zip(cluster.pods, assignment):
        if node_idx == UNASSIGNED:
            continue
        usage[node_idx]["cpu"] += pod.cpu_request
        usage[node_idx]["memory"] += pod.memory_request

    active = [idx for idx, u in usage.items() if u["cpu"] > 0 or u["memory"] > 0]
    if not active:
        return 0, 0.0

    densities = []
    for idx in active:
        node = cluster.nodes[idx]
        cpu_frac = usage[idx]["cpu"] / node.cpu_allocatable if node.cpu_allocatable else 0.0
        mem_frac = usage[idx]["memory"] / node.memory_allocatable if node.memory_allocatable else 0.0
        densities.append((cpu_frac + mem_frac) / 2.0)

    return len(active), sum(densities) / len(densities)


def validate_assignment(cluster: ClusterBatch, assignment: Sequence[int]) -> ValidationResult:
    if len(assignment) != len(cluster.pods):
        raise ValueError("assignment length must match number of pods")

    violations: list[ConstraintViolation] = []
    violations += _check_resources(cluster, assignment)
    violations += _check_taints_and_selectors(cluster, assignment)
    violations += _check_host_ports(cluster, assignment)
    violations += _check_pod_affinity(cluster, assignment)
    violations += _check_topology_spread(cluster, assignment)

    active_node_count, packing_density = _packing_stats(cluster, assignment)
    return ValidationResult(
        violations=violations, active_node_count=active_node_count, packing_density=packing_density
    )
