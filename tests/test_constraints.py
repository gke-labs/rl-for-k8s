from schedgym.constraints import validate_assignment
from schedgym.types import (
    ClusterBatch,
    LabelSelector,
    NodeAffinity,
    NodeSelectorTerm,
    LabelSelectorRequirement,
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

GIB = 1024**3


def make_node(name, cpu=8.0, memory=32 * GIB, gpu=0, labels=None, taints=()):
    return NodeSpec(
        name=name,
        cpu_allocatable=cpu,
        memory_allocatable=memory,
        gpu_allocatable=gpu,
        labels=labels or {"kubernetes.io/hostname": name, "topology.kubernetes.io/zone": "z1"},
        taints=taints,
    )


def make_pod(name, cpu=1.0, memory=1 * GIB, **kwargs):
    return PodSpec(name=name, cpu_request=cpu, memory_request=memory, **kwargs)


def test_feasible_simple_fit():
    cluster = ClusterBatch(pods=(make_pod("p1"),), nodes=(make_node("n1"),))
    result = validate_assignment(cluster, [0])
    assert result.is_feasible
    assert result.active_node_count == 1


def test_cpu_overcommit_detected():
    cluster = ClusterBatch(
        pods=(make_pod("p1", cpu=5.0), make_pod("p2", cpu=5.0)),
        nodes=(make_node("n1", cpu=8.0),),
    )
    result = validate_assignment(cluster, [0, 0])
    assert not result.is_feasible
    assert any(v.kind == "resource_overcommit" for v in result.violations)


def test_memory_fits_individually_but_not_together():
    cluster = ClusterBatch(
        pods=(make_pod("p1", memory=20 * GIB), make_pod("p2", memory=20 * GIB)),
        nodes=(make_node("n1", memory=32 * GIB),),
    )
    result = validate_assignment(cluster, [0, 0])
    assert not result.is_feasible


def test_taint_without_toleration_blocks():
    node = make_node("n1", taints=(Taint(key="dedicated", value="gpu", effect=TaintEffect.NO_SCHEDULE),))
    cluster = ClusterBatch(pods=(make_pod("p1"),), nodes=(node,))
    result = validate_assignment(cluster, [0])
    assert not result.is_feasible
    assert any(v.kind == "taint_toleration" for v in result.violations)


def test_toleration_allows_scheduling_on_tainted_node():
    node = make_node("n1", taints=(Taint(key="dedicated", value="gpu", effect=TaintEffect.NO_SCHEDULE),))
    pod = make_pod(
        "p1",
        tolerations=(Toleration(key="dedicated", operator=TolerationOperator.EQUAL, value="gpu", effect=TaintEffect.NO_SCHEDULE),),
    )
    cluster = ClusterBatch(pods=(pod,), nodes=(node,))
    result = validate_assignment(cluster, [0])
    assert result.is_feasible


def test_prefer_no_schedule_taint_does_not_block():
    node = make_node("n1", taints=(Taint(key="dedicated", value="gpu", effect=TaintEffect.PREFER_NO_SCHEDULE),))
    cluster = ClusterBatch(pods=(make_pod("p1"),), nodes=(node,))
    result = validate_assignment(cluster, [0])
    assert result.is_feasible


def test_gpu_node_selector_mismatch():
    node = make_node("n1", labels={"kubernetes.io/hostname": "n1"})
    pod = make_pod("p1", node_selector={"accelerator": "nvidia-a100"})
    cluster = ClusterBatch(pods=(pod,), nodes=(node,))
    result = validate_assignment(cluster, [0])
    assert not result.is_feasible
    assert any(v.kind == "node_selector" for v in result.violations)


def test_gpu_node_selector_match():
    node = make_node("n1", gpu=2, labels={"kubernetes.io/hostname": "n1", "accelerator": "nvidia-a100"})
    pod = make_pod("p1", gpu_request=1, node_selector={"accelerator": "nvidia-a100"})
    cluster = ClusterBatch(pods=(pod,), nodes=(node,))
    result = validate_assignment(cluster, [0])
    assert result.is_feasible


def test_gpu_overcommit():
    node = make_node("n1", gpu=1, labels={"accelerator": "nvidia-a100"})
    pods = (
        make_pod("p1", gpu_request=1, node_selector={"accelerator": "nvidia-a100"}),
        make_pod("p2", gpu_request=1, node_selector={"accelerator": "nvidia-a100"}),
    )
    cluster = ClusterBatch(pods=pods, nodes=(node,))
    result = validate_assignment(cluster, [0, 0])
    assert not result.is_feasible


def test_required_node_affinity():
    zone_a = make_node("n1", labels={"topology.kubernetes.io/zone": "us-east-1a"})
    zone_b = make_node("n2", labels={"topology.kubernetes.io/zone": "us-east-1b"})
    affinity = NodeAffinity(
        required=(
            NodeSelectorTerm(
                match_expressions=(
                    LabelSelectorRequirement(key="topology.kubernetes.io/zone", operator=SelectorOperator.IN, values=("us-east-1a",)),
                ),
            ),
        )
    )
    pod = make_pod("p1", node_affinity=affinity)
    cluster = ClusterBatch(pods=(pod,), nodes=(zone_a, zone_b))

    assert validate_assignment(cluster, [0]).is_feasible
    assert not validate_assignment(cluster, [1]).is_feasible


def test_pod_anti_affinity_same_host_violation():
    node = make_node("n1")
    web1 = make_pod("web-1", labels={"app": "web"})
    web2 = make_pod(
        "web-2",
        labels={"app": "web"},
        pod_anti_affinity=PodAffinity(
            required=(PodAffinityTerm(label_selector=LabelSelector(match_labels={"app": "web"}), topology_key="kubernetes.io/hostname"),)
        ),
    )
    cluster = ClusterBatch(pods=(web1, web2), nodes=(node,))
    result = validate_assignment(cluster, [0, 0])
    assert not result.is_feasible
    assert any(v.kind == "pod_anti_affinity" for v in result.violations)


def test_pod_anti_affinity_different_host_ok():
    n1, n2 = make_node("n1"), make_node("n2")
    web1 = make_pod("web-1", labels={"app": "web"})
    web2 = make_pod(
        "web-2",
        labels={"app": "web"},
        pod_anti_affinity=PodAffinity(
            required=(PodAffinityTerm(label_selector=LabelSelector(match_labels={"app": "web"}), topology_key="kubernetes.io/hostname"),)
        ),
    )
    cluster = ClusterBatch(pods=(web1, web2), nodes=(n1, n2))
    result = validate_assignment(cluster, [0, 1])
    assert result.is_feasible


def test_pod_affinity_requires_co_located_match():
    n1, n2 = make_node("n1"), make_node("n2")
    cache = make_pod("cache-1", labels={"app": "cache"})
    web = make_pod(
        "web-1",
        labels={"app": "web"},
        pod_affinity=PodAffinity(
            required=(PodAffinityTerm(label_selector=LabelSelector(match_labels={"app": "cache"}), topology_key="kubernetes.io/hostname"),)
        ),
    )
    cluster = ClusterBatch(pods=(cache, web), nodes=(n1, n2))

    assert validate_assignment(cluster, [0, 0]).is_feasible
    assert not validate_assignment(cluster, [0, 1]).is_feasible


def test_host_port_conflict():
    node = make_node("n1")
    p1 = make_pod("p1", host_ports=(8080,))
    p2 = make_pod("p2", host_ports=(8080,))
    cluster = ClusterBatch(pods=(p1, p2), nodes=(node,))
    result = validate_assignment(cluster, [0, 0])
    assert not result.is_feasible
    assert any(v.kind == "host_port_conflict" for v in result.violations)


def test_host_port_different_nodes_ok():
    n1, n2 = make_node("n1"), make_node("n2")
    p1 = make_pod("p1", host_ports=(8080,))
    p2 = make_pod("p2", host_ports=(8080,))
    cluster = ClusterBatch(pods=(p1, p2), nodes=(n1, n2))
    result = validate_assignment(cluster, [0, 1])
    assert result.is_feasible


def test_topology_spread_violation():
    n1 = make_node("n1", labels={"kubernetes.io/hostname": "n1", "topology.kubernetes.io/zone": "us-east-1a"})
    n2 = make_node("n2", labels={"kubernetes.io/hostname": "n2", "topology.kubernetes.io/zone": "us-east-1b"})
    selector = LabelSelector(match_labels={"app": "web"})
    spread = (TopologySpreadConstraint(topology_key="topology.kubernetes.io/zone", max_skew=1, label_selector=selector),)
    pods = tuple(make_pod(f"web-{i}", labels={"app": "web"}, topology_spread_constraints=spread) for i in range(3))
    cluster = ClusterBatch(pods=pods, nodes=(n1, n2))

    # all three crammed into zone a -> skew 3, violates maxSkew=1
    result = validate_assignment(cluster, [0, 0, 0])
    assert not result.is_feasible
    assert any(v.kind == "topology_spread" for v in result.violations)

    # spread 2/1 across zones -> skew 1, satisfies maxSkew=1
    result_balanced = validate_assignment(cluster, [0, 0, 1])
    assert result_balanced.is_feasible


def test_unschedulable_node_blocks():
    node = NodeSpec(name="n1", cpu_allocatable=8.0, memory_allocatable=32 * GIB, unschedulable=True)
    cluster = ClusterBatch(pods=(make_pod("p1"),), nodes=(node,))
    result = validate_assignment(cluster, [0])
    assert not result.is_feasible
    assert any(v.kind == "unschedulable_node" for v in result.violations)


def test_packing_density_computed_only_over_active_nodes():
    n1 = make_node("n1", cpu=10.0, memory=10 * GIB)
    n2 = make_node("n2", cpu=10.0, memory=10 * GIB)
    pod = make_pod("p1", cpu=5.0, memory=5 * GIB)
    cluster = ClusterBatch(pods=(pod,), nodes=(n1, n2))
    result = validate_assignment(cluster, [0])
    assert result.active_node_count == 1
    assert result.packing_density == 0.5
