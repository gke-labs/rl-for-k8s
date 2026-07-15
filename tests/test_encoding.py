import numpy as np

from schedgym.encoding import encode_batch, encode_cluster
from schedgym.generator import generate_dataset

MAX_PODS = 20
MAX_NODES = 12


def test_encoded_shapes_and_padding():
    cluster = generate_dataset(1, seed=2)[0]
    enc = encode_cluster(cluster, MAX_PODS, MAX_NODES)

    assert enc.pod_mask.shape == (MAX_PODS,)
    assert enc.node_mask.shape == (MAX_NODES,)
    assert enc.action_mask.shape == (MAX_PODS, MAX_NODES)
    assert enc.pod_mask.sum() == len(cluster.pods)
    assert enc.node_mask.sum() == len(cluster.nodes)

    # padding pods/nodes must never be marked as legal targets
    assert not enc.action_mask[~enc.pod_mask].any()
    assert not enc.action_mask[:, ~enc.node_mask].any()


def test_encode_batch_stacks_leading_dimension():
    clusters = generate_dataset(4, seed=9)
    batch = encode_batch(clusters, MAX_PODS, MAX_NODES)
    assert batch.pod_mask.shape == (4, MAX_PODS)
    assert batch.action_mask.shape == (4, MAX_PODS, MAX_NODES)


def test_action_mask_rejects_taint_without_toleration():
    from schedgym.types import ClusterBatch, NodeSpec, PodSpec, Taint, TaintEffect

    node = NodeSpec(
        name="n1",
        cpu_allocatable=8.0,
        memory_allocatable=32 * 1024**3,
        taints=(Taint(key="dedicated", value="gpu", effect=TaintEffect.NO_SCHEDULE),),
    )
    pod = PodSpec(name="p1", cpu_request=1.0, memory_request=1024**3)
    cluster = ClusterBatch(pods=(pod,), nodes=(node,))
    enc = encode_cluster(cluster, MAX_PODS, MAX_NODES)
    assert not enc.action_mask[0, 0]


def test_action_mask_allows_resource_fit():
    from schedgym.types import ClusterBatch, NodeSpec, PodSpec

    node = NodeSpec(name="n1", cpu_allocatable=8.0, memory_allocatable=32 * 1024**3)
    pod = PodSpec(name="p1", cpu_request=1.0, memory_request=1024**3)
    cluster = ClusterBatch(pods=(pod,), nodes=(node,))
    enc = encode_cluster(cluster, MAX_PODS, MAX_NODES)
    assert enc.action_mask[0, 0]
    assert isinstance(enc.action_mask[0, 0], (bool, np.bool_))
