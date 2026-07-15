"""Heuristic baselines the learned policy should beat (DESIGN.md sec 9).

Both operate purely on the Python data model / oracle validator -- no
JAX involved. `best_fit_decreasing` reuses `validate_assignment` on
partial assignments (UNASSIGNED pods are simply absent from every check,
so it doubles as an incremental "is this partial placement still legal"
oracle for free).
"""

from __future__ import annotations

import random

from schedgym.constraints import UNASSIGNED, validate_assignment
from schedgym.types import ClusterBatch

GIB = 1024**3


def _pod_weight(pod) -> float:
    return pod.cpu_request + pod.memory_request / GIB


def best_fit_decreasing(cluster: ClusterBatch) -> list[int]:
    """Greedy bin packing: place the largest pods first, each time picking
    the *tightest*-fitting legal node (falls back to leaving a pod
    UNASSIGNED, like a real scheduler would, if nothing fits)."""
    pods = list(cluster.pods)
    order = sorted(range(len(pods)), key=lambda i: _pod_weight(pods[i]), reverse=True)
    assignment = [UNASSIGNED] * len(pods)

    for i in order:
        best_node = None
        best_remaining = None
        for j, node in enumerate(cluster.nodes):
            candidate = list(assignment)
            candidate[i] = j
            if not validate_assignment(cluster, candidate).is_feasible:
                continue
            used_cpu = sum(pods[k].cpu_request for k, a in enumerate(candidate) if a == j)
            remaining = node.cpu_allocatable - used_cpu
            if best_remaining is None or remaining < best_remaining:
                best_remaining = remaining
                best_node = j
        if best_node is not None:
            assignment[i] = best_node

    return assignment


def random_feasible(cluster: ClusterBatch, rng: random.Random) -> list[int]:
    """Random-order greedy placement onto a uniformly random *currently
    legal* node -- a weak baseline that still respects hard constraints
    it can check locally, unlike pure uniform random."""
    pods = list(cluster.pods)
    order = list(range(len(pods)))
    rng.shuffle(order)
    assignment = [UNASSIGNED] * len(pods)

    for i in order:
        candidates = []
        for j in range(len(cluster.nodes)):
            candidate = list(assignment)
            candidate[i] = j
            if validate_assignment(cluster, candidate).is_feasible:
                candidates.append(j)
        if candidates:
            assignment[i] = rng.choice(candidates)

    return assignment
