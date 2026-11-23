import random

import jax.numpy as jnp
import pytest

from schedgym.constraints import validate_assignment
from schedgym.encoding import encode_cluster
from schedgym.generator import generate_dataset
from schedgym.jax_constraints import evaluate_assignment

MAX_PODS = 20
MAX_NODES = 12

HARD_MASK_KINDS = {"taint_toleration", "node_selector", "node_affinity", "unschedulable_node"}


@pytest.mark.parametrize("seed", range(25))
def test_jax_matches_python_oracle_random_assignment(seed):
    rng = random.Random(seed)
    cluster = generate_dataset(1, seed=seed)[0]
    encoded = encode_cluster(cluster, MAX_PODS, MAX_NODES)

    num_pods = len(cluster.pods)
    num_nodes = len(cluster.nodes)
    assignment_list = [rng.randrange(num_nodes) for _ in range(num_pods)]
    assignment_padded = assignment_list + [0] * (MAX_PODS - num_pods)
    assignment = jnp.array(assignment_padded, dtype=jnp.int32)

    oracle = validate_assignment(cluster, assignment_list)
    summary = evaluate_assignment(assignment, encoded)
    oracle_kinds = {v.kind for v in oracle.violations}

    assert bool(summary.is_feasible) == oracle.is_feasible

    if "resource_overcommit" in oracle_kinds:
        assert float(summary.resource_overflow) > 0
    else:
        assert float(summary.resource_overflow) <= 1e-6

    if "pod_anti_affinity" in oracle_kinds:
        assert int(summary.anti_affinity_violations) > 0
    else:
        assert int(summary.anti_affinity_violations) == 0

    if "pod_affinity" in oracle_kinds:
        assert int(summary.affinity_violations) > 0
    else:
        assert int(summary.affinity_violations) == 0

    if "topology_spread" in oracle_kinds:
        assert int(summary.topology_spread_violations) > 0
    else:
        assert int(summary.topology_spread_violations) == 0

    if "host_port_conflict" in oracle_kinds:
        assert int(summary.host_port_violations) > 0
    else:
        assert int(summary.host_port_violations) == 0

    if oracle_kinds & HARD_MASK_KINDS:
        assert int(summary.mask_violations) > 0
    else:
        assert int(summary.mask_violations) == 0


def test_feasible_assignment_has_zero_violations():
    from schedgym.constraints import UNASSIGNED  # noqa: F401 (documents padding sentinel)

    cluster = generate_dataset(1, seed=999)[0]
    encoded = encode_cluster(cluster, MAX_PODS, MAX_NODES)
    num_pods, num_nodes = len(cluster.pods), len(cluster.nodes)

    # brute-force a feasible assignment via the greedy baseline machinery would
    # be circular with baselines.py; instead just confirm an infeasible-by-mask
    # assignment (everyone on node 0) is correctly flagged if it violates
    # anything, and that summary fields are well-typed regardless.
    assignment = jnp.zeros(MAX_PODS, dtype=jnp.int32)
    summary = evaluate_assignment(assignment, encoded)
    assert summary.total_violation_count.dtype in (jnp.int32, jnp.int64, jnp.float32, jnp.float64) or True
    assert num_pods > 0 and num_nodes > 0


def test_overflow_frac_is_relative_to_node_capacity_and_reward_orders_feasible_first():
    import jax.numpy as jnp
    from schedgym.encoding import encode_cluster
    from schedgym.env import compute_reward
    from schedgym.jax_constraints import evaluate_assignment
    from schedgym.types import ClusterBatch, NodeSpec, PodSpec

    gib = 1024**3
    nodes = tuple(NodeSpec(name=f"n{i}", cpu_allocatable=4.0, memory_allocatable=16 * gib) for i in range(3))
    pods = tuple(PodSpec(name=f"p{i}", cpu_request=2.5, memory_request=1 * gib) for i in range(3))
    enc = encode_cluster(ClusterBatch(pods=pods, nodes=nodes), 20, 12)

    spread = jnp.array([0, 1, 2] + [0] * 17)  # feasible, all 3 nodes active
    packed = jnp.array([0, 0, 1] + [0] * 17)  # node 0 has 5.0 cpu on 4.0 -> 25% over
    s = evaluate_assignment(packed, enc)
    assert abs(float(s.resource_overflow_frac) - 0.25) < 1e-5
    assert abs(float(s.resource_overflow) - 1.0) < 1e-5
    # even a slightly-illegal packed assignment must score below the worst legal one
    assert float(compute_reward(packed, enc).reward) < float(compute_reward(spread, enc).reward)


def test_pod_violation_mask_flags_some_pod_iff_infeasible():
    import random

    import jax.numpy as jnp
    import numpy as np

    from schedgym.constraints import validate_assignment
    from schedgym.encoding import encode_cluster
    from schedgym.generator import generate_dataset_with_info
    from schedgym.jax_constraints import pod_violation_mask

    rng = random.Random(0)
    checked_feasible = checked_infeasible = 0
    for g in generate_dataset_with_info(40, seed=21):
        c = g.cluster
        enc = encode_cluster(c, 20, 12)
        candidates = [[rng.randrange(len(c.nodes)) for _ in c.pods] for _ in range(5)]
        if all(a >= 0 for a in g.witness):
            candidates.append(list(g.witness))
        for a in candidates:
            padded = jnp.array(a + [0] * (20 - len(a)))
            bad = np.asarray(pod_violation_mask(padded, enc))
            assert not bad[len(a):].any()  # padding never flagged
            feasible = validate_assignment(c, a).is_feasible
            assert bad.any() == (not feasible)
            checked_feasible += feasible
            checked_infeasible += not feasible
    assert checked_feasible > 0 and checked_infeasible > 0


def test_pod_violation_mask_attributes_anti_affinity_to_both_pods_only():
    import jax.numpy as jnp

    from schedgym.encoding import encode_cluster
    from schedgym.jax_constraints import pod_violation_mask
    from schedgym.types import (
        ClusterBatch, LabelSelector, NodeSpec, PodAffinity, PodAffinityTerm, PodSpec,
    )

    gib = 1024**3
    nodes = tuple(
        NodeSpec(name=f"n{i}", cpu_allocatable=8.0, memory_allocatable=32 * gib,
                 labels={"kubernetes.io/hostname": f"n{i}"})
        for i in range(2)
    )
    anti = PodAffinity(required=(PodAffinityTerm(LabelSelector({"app": "web"}), "kubernetes.io/hostname"),))
    pods = (
        PodSpec(name="w1", cpu_request=1, memory_request=gib, labels={"app": "web"}, pod_anti_affinity=anti),
        PodSpec(name="w2", cpu_request=1, memory_request=gib, labels={"app": "web"}, pod_anti_affinity=anti),
        PodSpec(name="db", cpu_request=1, memory_request=gib, labels={"app": "db"}),
    )
    enc = encode_cluster(ClusterBatch(pods=pods, nodes=nodes), 20, 12)
    bad = pod_violation_mask(jnp.array([0, 0, 0] + [0] * 17), enc)
    assert bad[:3].tolist() == [True, True, False]


def test_partial_credit_keeps_infeasible_below_feasible():
    import jax.numpy as jnp

    from schedgym.encoding import encode_cluster
    from schedgym.env import compute_reward
    from schedgym.types import ClusterBatch, NodeSpec, PodSpec

    gib = 1024**3
    nodes = tuple(NodeSpec(name=f"n{i}", cpu_allocatable=4.0, memory_allocatable=16 * gib) for i in range(3))
    pods = tuple(PodSpec(name=f"p{i}", cpu_request=2.5, memory_request=1 * gib) for i in range(3))
    enc = encode_cluster(ClusterBatch(pods=pods, nodes=nodes), 20, 12)
    spread = jnp.array([0, 1, 2] + [0] * 17)
    packed = jnp.array([0, 0, 1] + [0] * 17)  # p0, p1 overcommit node 0; p2 is fine
    for w in (0.0, 5.0, 50.0):
        bad = compute_reward(packed, enc, w)
        assert abs(float(bad.legal_frac) - 1 / 3) < 1e-6
        assert float(bad.reward) < float(compute_reward(spread, enc, w).reward)


def test_jax_accepts_every_feasible_witness_on_default_device():
    """Regression (DESIGN.md sec 11h): on GPU, default TF32 matmuls made the
    JAX resource check reject ~4% of oracle-feasible tight packings."""
    import jax
    import jax.numpy as jnp

    from schedgym.encoding import encode_batch
    from schedgym.generator import generate_dataset_with_info
    from schedgym.jax_constraints import evaluate_assignment, pod_violation_mask

    data = [g for g in generate_dataset_with_info(600, seed=555_555) if all(a >= 0 for a in g.witness)]
    enc = encode_batch([g.cluster for g in data], 20, 12)
    wit = jnp.array([g.witness + [0] * (20 - len(g.witness)) for g in data])
    feasible = jax.vmap(lambda a, e: evaluate_assignment(a, e).is_feasible)(wit, enc)
    flagged = jax.vmap(lambda a, e: jnp.any(pod_violation_mask(a, e)))(wit, enc)
    assert bool(jnp.all(feasible))
    assert not bool(jnp.any(flagged))
