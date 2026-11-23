from schedgym.generator import GeneratorConfig, easy_config, generate_dataset, interpolate_config
from schedgym.constraints import validate_assignment


def test_generate_dataset_is_deterministic_for_seed():
    a = generate_dataset(5, seed=123)
    b = generate_dataset(5, seed=123)
    assert [len(c.pods) for c in a] == [len(c.pods) for c in b]
    assert [p.name for c in a for p in c.pods] == [p.name for c in b for p in c.pods]


def test_generated_clusters_respect_bounds():
    config = GeneratorConfig(min_nodes=3, max_nodes=6, min_pods=2, max_pods=10)
    dataset = generate_dataset(20, seed=7, config=config)
    for cluster in dataset:
        assert config.min_nodes <= len(cluster.nodes) <= config.max_nodes
        assert config.min_pods <= len(cluster.pods) <= config.max_pods


def test_generated_clusters_are_validator_compatible():
    dataset = generate_dataset(10, seed=1)
    for cluster in dataset:
        # trivial "assign everything to node 0" may or may not be feasible,
        # but the validator must run without raising on generator output.
        assignment = [0] * len(cluster.pods)
        result = validate_assignment(cluster, assignment)
        assert isinstance(result.is_feasible, bool)


def test_interpolate_config_endpoints_match_easy_and_full():
    full = GeneratorConfig()
    easy = easy_config()
    assert interpolate_config(easy, full, 0.0) == easy
    assert interpolate_config(easy, full, 1.0) == full


def test_interpolate_config_is_monotonic_in_difficulty():
    full = GeneratorConfig()
    easy = easy_config()
    prev = interpolate_config(easy, full, 0.0)
    for t in (0.25, 0.5, 0.75, 1.0):
        cur = interpolate_config(easy, full, t)
        assert cur.max_pods >= prev.max_pods
        assert cur.spread_group_fraction >= prev.spread_group_fraction
        assert cur.gpu_pod_fraction >= prev.gpu_pod_fraction
        prev = cur


def test_interpolate_config_clamps_outside_unit_range():
    full = GeneratorConfig()
    easy = easy_config()
    assert interpolate_config(easy, full, -1.0) == easy
    assert interpolate_config(easy, full, 2.0) == full


def test_easy_config_produces_only_resource_constraints():
    dataset = generate_dataset(20, seed=3, config=easy_config())
    for cluster in dataset:
        for pod in cluster.pods:
            assert pod.gpu_request == 0
            assert not pod.tolerations
            assert not pod.node_selector
            assert pod.node_affinity is None
            assert pod.pod_affinity is None
            assert pod.pod_anti_affinity is None
            assert not pod.topology_spread_constraints
            assert not pod.host_ports
        for node in cluster.nodes:
            assert not node.taints
            assert node.gpu_allocatable == 0


from schedgym.constraints import UNASSIGNED
from schedgym.generator import (
    TIER_HIGH,
    TIER_OVERFLOW,
    TIER_VERY_HIGH,
    cluster_utilization,
    generate_dataset_with_info,
)


def test_tier_mix_is_exactly_stratified():
    data = generate_dataset_with_info(100, seed=11)
    counts = {t: sum(g.tier == t for g in data) for t in (TIER_HIGH, TIER_VERY_HIGH, TIER_OVERFLOW)}
    assert counts == {TIER_HIGH: 50, TIER_VERY_HIGH: 40, TIER_OVERFLOW: 10}


def test_schedulable_tiers_have_feasible_witness_and_target_utilization():
    for g in generate_dataset_with_info(120, seed=5):
        cpu_u, mem_u = cluster_utilization(g.cluster)
        if g.tier == TIER_OVERFLOW:
            continue
        assert all(a != UNASSIGNED for a in g.witness)
        assert validate_assignment(g.cluster, g.witness).is_feasible
        lo = 0.8 if g.tier == TIER_HIGH else 0.9
        hi = 0.9 if g.tier == TIER_HIGH else 1.0
        assert lo - 1e-9 <= cpu_u <= hi + 1e-9
        assert lo - 1e-9 <= mem_u <= hi + 1e-9


def test_overflow_tier_requests_more_than_capacity():
    for g in generate_dataset_with_info(60, seed=8):
        if g.tier != TIER_OVERFLOW:
            continue
        assert max(cluster_utilization(g.cluster)) > 1.0
        # the placeable part of the witness is still legal
        assert validate_assignment(g.cluster, g.witness).is_feasible
        assert any(a == UNASSIGNED for a in g.witness)


def test_curriculum_configs_generate_valid_tiers():
    full = GeneratorConfig()
    easy = easy_config()
    for t in (0.0, 0.5, 1.0):
        cfg = interpolate_config(easy, full, t)
        for g in generate_dataset_with_info(20, seed=3, config=cfg):
            assert len(g.cluster.pods) <= cfg.max_pods
            if g.tier != TIER_OVERFLOW:
                assert validate_assignment(g.cluster, g.witness).is_feasible
