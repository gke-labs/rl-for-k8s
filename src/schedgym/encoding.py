"""Host-side (numpy) encoding of a ClusterBatch into fixed-shape padded
arrays for JAX. Runs once per generated training example, not inside the
hot training loop -- see DESIGN.md sec 8 for why cluster generation and
encoding stay on the host while policy/reward computation is jitted.

Two topology keys are supported end to end (matching the generator and
examples): "kubernetes.io/hostname" and "topology.kubernetes.io/zone".
Any other topology key on a pod affinity/anti-affinity/spread constraint
is treated as unsupported and silently ignored (column = -1) -- a
documented v1 limitation, not a bug.
"""

from __future__ import annotations

import hashlib

import numpy as np
from flax import struct

from schedgym.types import ClusterBatch, NodeSpec, PodSpec

GIB = 1024**3

NUM_HASH_BUCKETS = 64
MAX_LABEL_SLOTS = 6
MAX_HOST_PORTS = 2

HOSTNAME_KEY = "kubernetes.io/hostname"
ZONE_KEY = "topology.kubernetes.io/zone"
_TOPOLOGY_COLUMNS = {HOSTNAME_KEY: 0, ZONE_KEY: 1}

POD_NUM_FEATS = 10
NODE_NUM_FEATS = 9


def hash_bucket(s: str, num_buckets: int = NUM_HASH_BUCKETS) -> int:
    """Stable string -> bucket id in [1, num_buckets); 0 is reserved for pad."""
    digest = hashlib.md5(s.encode("utf-8")).hexdigest()
    return (int(digest, 16) % (num_buckets - 1)) + 1


def _bag(items: list[str], max_slots: int = MAX_LABEL_SLOTS) -> list[int]:
    buckets = [hash_bucket(item) for item in sorted(items)[:max_slots]]
    buckets += [0] * (max_slots - len(buckets))
    return buckets


def _labels_bag(labels: dict[str, str]) -> list[int]:
    return _bag([f"{k}={v}" for k, v in labels.items()])


def _tolerations_bag(pod: PodSpec) -> list[int]:
    return _bag(
        [f"{t.key}={t.value}:{t.effect.value if t.effect else '*'}:{t.operator.value}" for t in pod.tolerations]
    )


def _taints_bag(node: NodeSpec) -> list[int]:
    return _bag([f"{t.key}={t.value}:{t.effect.value}" for t in node.taints])


def _existing_usage(node: NodeSpec) -> np.ndarray:
    usage = np.zeros(4, dtype=np.float32)
    for pod in node.existing_pods:
        usage += np.array(
            [pod.cpu_request, pod.memory_request / GIB, pod.gpu_request, pod.ephemeral_storage_request / GIB],
            dtype=np.float32,
        )
    return usage


def _pod_fits(pod: PodSpec, node: NodeSpec, remaining: np.ndarray) -> bool:
    req = np.array(
        [pod.cpu_request, pod.memory_request / GIB, pod.gpu_request, pod.ephemeral_storage_request / GIB],
        dtype=np.float32,
    )
    return bool(np.all(req <= remaining + 1e-9))


def _pairwise_legal(pod: PodSpec, node: NodeSpec, remaining: np.ndarray) -> bool:
    if node.unschedulable:
        return False
    if not _pod_fits(pod, node, remaining):
        return False
    for taint in node.taints:
        if taint.effect == taint.effect.PREFER_NO_SCHEDULE:
            continue
        if not any(tol.tolerates(taint) for tol in pod.tolerations):
            return False
    for key, value in pod.node_selector.items():
        if node.labels.get(key) != value:
            return False
    if pod.node_affinity is not None and not pod.node_affinity.satisfies_required(node.labels):
        return False
    return True


def _topology_col(key: str) -> int:
    return _TOPOLOGY_COLUMNS.get(key, -1)


@struct.dataclass
class Encoded:
    pod_numeric: np.ndarray
    pod_labels: np.ndarray
    pod_selector: np.ndarray
    pod_tolerations: np.ndarray
    pod_mask: np.ndarray

    node_numeric: np.ndarray
    node_labels: np.ndarray
    node_taints: np.ndarray
    node_mask: np.ndarray

    action_mask: np.ndarray

    node_hostname_domain: np.ndarray
    node_zone_domain: np.ndarray
    domain_has_node: np.ndarray  # [2, max_nodes]
    pod_resources: np.ndarray
    node_resources: np.ndarray

    pod_anti_affinity_match: np.ndarray
    pod_anti_affinity_col: np.ndarray
    pod_affinity_match: np.ndarray
    pod_affinity_col: np.ndarray

    spread_col_of_group: np.ndarray
    spread_max_skew_of_group: np.ndarray
    spread_selector_match: np.ndarray

    host_port_bucket: np.ndarray


def encode_cluster(cluster: ClusterBatch, max_pods: int, max_nodes: int) -> Encoded:
    pods = list(cluster.pods)
    nodes = list(cluster.nodes)
    if len(pods) > max_pods:
        raise ValueError(f"cluster has {len(pods)} pods > max_pods={max_pods}")
    if len(nodes) > max_nodes:
        raise ValueError(f"cluster has {len(nodes)} nodes > max_nodes={max_nodes}")
    P, N = max_pods, max_nodes

    pod_mask = np.zeros(P, dtype=bool)
    pod_mask[: len(pods)] = True
    node_mask = np.zeros(N, dtype=bool)
    node_mask[: len(nodes)] = True

    pod_numeric = np.zeros((P, POD_NUM_FEATS), dtype=np.float32)
    pod_labels = np.zeros((P, MAX_LABEL_SLOTS), dtype=np.int32)
    pod_selector = np.zeros((P, MAX_LABEL_SLOTS), dtype=np.int32)
    pod_tolerations = np.zeros((P, MAX_LABEL_SLOTS), dtype=np.int32)
    pod_resources = np.zeros((P, 4), dtype=np.float32)
    host_port_bucket = np.zeros((P, MAX_HOST_PORTS), dtype=np.int32)

    pod_anti_affinity_col = np.full(P, -1, dtype=np.int32)
    pod_affinity_col = np.full(P, -1, dtype=np.int32)
    pod_anti_affinity_match = np.zeros((P, P), dtype=bool)
    pod_affinity_match = np.zeros((P, P), dtype=bool)

    spread_col_of_group = np.full(P, -1, dtype=np.int32)
    spread_max_skew_of_group = np.zeros(P, dtype=np.int32)
    spread_selector_match = np.zeros((P, P), dtype=bool)
    group_signatures: dict[tuple, int] = {}

    for i, pod in enumerate(pods):
        pod_numeric[i] = [
            pod.cpu_request,
            pod.memory_request / GIB,
            float(pod.gpu_request),
            pod.ephemeral_storage_request / GIB,
            float(len(pod.host_ports)),
            float(pod.node_affinity is not None),
            float(pod.pod_affinity is not None),
            float(pod.pod_anti_affinity is not None),
            float(len(pod.topology_spread_constraints)),
            float(pod.priority),
        ]
        pod_labels[i] = _labels_bag(pod.labels)
        pod_selector[i] = _labels_bag(pod.node_selector)
        pod_tolerations[i] = _tolerations_bag(pod)
        pod_resources[i] = [
            pod.cpu_request,
            pod.memory_request / GIB,
            float(pod.gpu_request),
            pod.ephemeral_storage_request / GIB,
        ]
        for slot, port in enumerate(pod.host_ports[:MAX_HOST_PORTS]):
            host_port_bucket[i, slot] = hash_bucket(f"port={port}")

        if pod.pod_anti_affinity is not None and pod.pod_anti_affinity.required:
            term = pod.pod_anti_affinity.required[0]
            pod_anti_affinity_col[i] = _topology_col(term.topology_key)
        if pod.pod_affinity is not None and pod.pod_affinity.required:
            term = pod.pod_affinity.required[0]
            pod_affinity_col[i] = _topology_col(term.topology_key)

        if pod.topology_spread_constraints:
            c = pod.topology_spread_constraints[0]
            signature = (c.topology_key, tuple(sorted(c.label_selector.match_labels.items())), c.max_skew)
            group_id = group_signatures.setdefault(signature, len(group_signatures))
            if group_id < P:
                spread_col_of_group[group_id] = _topology_col(c.topology_key)
                spread_max_skew_of_group[group_id] = c.max_skew

    for i, pod in enumerate(pods):
        for k, other in enumerate(pods):
            if i == k:
                continue
            if pod.pod_anti_affinity is not None and pod.pod_anti_affinity.required:
                term = pod.pod_anti_affinity.required[0]
                pod_anti_affinity_match[i, k] = term.label_selector.matches(other.labels)
            if pod.pod_affinity is not None and pod.pod_affinity.required:
                term = pod.pod_affinity.required[0]
                pod_affinity_match[i, k] = term.label_selector.matches(other.labels)

        if pod.topology_spread_constraints:
            c = pod.topology_spread_constraints[0]
            signature = (c.topology_key, tuple(sorted(c.label_selector.match_labels.items())), c.max_skew)
            group_id = group_signatures[signature]
            if group_id < P:
                for k, other in enumerate(pods):
                    spread_selector_match[group_id, k] = c.label_selector.matches(other.labels)

    node_numeric = np.zeros((N, NODE_NUM_FEATS), dtype=np.float32)
    node_labels = np.zeros((N, MAX_LABEL_SLOTS), dtype=np.int32)
    node_taints = np.zeros((N, MAX_LABEL_SLOTS), dtype=np.int32)
    node_resources = np.zeros((N, 4), dtype=np.float32)
    node_hostname_domain = np.full(N, -1, dtype=np.int32)
    node_zone_domain = np.full(N, -1, dtype=np.int32)

    zone_ids: dict[str, int] = {}
    for j, node in enumerate(nodes):
        existing_usage = _existing_usage(node)
        node_numeric[j] = [
            node.cpu_allocatable,
            node.memory_allocatable / GIB,
            float(node.gpu_allocatable),
            node.ephemeral_storage_allocatable / GIB,
            float(node.unschedulable),
            *existing_usage,
        ]
        node_labels[j] = _labels_bag(node.labels)
        node_taints[j] = _taints_bag(node)
        node_resources[j] = [
            node.cpu_allocatable,
            node.memory_allocatable / GIB,
            float(node.gpu_allocatable),
            node.ephemeral_storage_allocatable / GIB,
        ] - existing_usage
        node_hostname_domain[j] = j
        zone = node.labels.get(ZONE_KEY)
        if zone is not None:
            node_zone_domain[j] = zone_ids.setdefault(zone, len(zone_ids))

    domain_has_node = np.zeros((2, N), dtype=bool)
    domain_has_node[0, : len(nodes)] = True
    for j in range(len(nodes)):
        z = node_zone_domain[j]
        if z >= 0:
            domain_has_node[1, z] = True

    action_mask = np.zeros((P, N), dtype=bool)
    for i, pod in enumerate(pods):
        for j, node in enumerate(nodes):
            action_mask[i, j] = _pairwise_legal(pod, node, node_resources[j])

    return Encoded(
        pod_numeric=pod_numeric,
        pod_labels=pod_labels,
        pod_selector=pod_selector,
        pod_tolerations=pod_tolerations,
        pod_mask=pod_mask,
        node_numeric=node_numeric,
        node_labels=node_labels,
        node_taints=node_taints,
        node_mask=node_mask,
        action_mask=action_mask,
        node_hostname_domain=node_hostname_domain,
        node_zone_domain=node_zone_domain,
        domain_has_node=domain_has_node,
        pod_resources=pod_resources,
        node_resources=node_resources,
        pod_anti_affinity_match=pod_anti_affinity_match,
        pod_anti_affinity_col=pod_anti_affinity_col,
        pod_affinity_match=pod_affinity_match,
        pod_affinity_col=pod_affinity_col,
        spread_col_of_group=spread_col_of_group,
        spread_max_skew_of_group=spread_max_skew_of_group,
        spread_selector_match=spread_selector_match,
        host_port_bucket=host_port_bucket,
    )


def encode_batch(clusters: list[ClusterBatch], max_pods: int, max_nodes: int) -> Encoded:
    """Stack a list of single-cluster Encoded pytrees into arrays with a
    leading batch dimension, ready for jax.vmap."""
    encoded = [encode_cluster(c, max_pods, max_nodes) for c in clusters]
    fields = encoded[0].__dataclass_fields__.keys()
    stacked = {f: np.stack([getattr(e, f) for e in encoded], axis=0) for f in fields}
    return Encoded(**stacked)
