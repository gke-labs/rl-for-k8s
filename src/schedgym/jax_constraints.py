"""JAX-vectorized joint constraint checking for a full one-shot pod->node
assignment. Mirrors constraints.py (the Python oracle) for the checks
that genuinely depend on the *whole* batch's assignment at once -- see
DESIGN.md sec 5 for why these can't be folded into the pairwise
`action_mask` computed in encoding.py.

Every function here takes un-batched (single-cluster) arrays and is
meant to be wrapped in `jax.vmap` across a batch of clusters -- see
env.py. All shapes below use P = max_pods, N = max_nodes.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

RESOURCE_EPS = 1e-6
# Tolerance for "used > allocatable", relative to the node's allocatable.
# float32 sums of ~20 requests on a 128 GiB / 32 core node carry ~1e-6
# relative rounding error, so an absolute 1e-6 GiB check can reject a
# packing the float64 oracle accepts. 1e-5 of capacity is 1.3 MiB / 0.3m
# CPU on the biggest node, far below the generator's 16 MiB / 10m steps.
RESOURCE_REL_TOL = 1e-5
# All sums below must be exact-ish float32. On GPUs, XLA's default matmul
# precision is TF32 (~10-bit mantissa), which put errors of ~1/64 GiB on
# 128 GiB nodes and made ~4% of feasible tight packings look overcommitted
# (DESIGN.md sec 11h).
_EXACT = jax.lax.Precision.HIGHEST


def _node_usage(assignment, pod_mask, pod_resources, num_nodes):
    onehot = jax.nn.one_hot(assignment, num_nodes, dtype=jnp.float32) * pod_mask[:, None]
    return jnp.einsum("pn,pr->nr", onehot, pod_resources, precision=_EXACT)


def _node_overflow(used, node_resources, node_mask):
    """[N, R] amount by which usage exceeds allocatable, ignoring excess
    within RESOURCE_REL_TOL of capacity (float32 rounding)."""
    tol = RESOURCE_REL_TOL * jnp.maximum(node_resources, 1.0)
    return jnp.where(used - node_resources > tol, used - node_resources, 0.0) * node_mask[:, None]


class ViolationSummary(NamedTuple):
    mask_violations: jnp.ndarray  # pods placed on an action_mask-illegal node (should be ~0 by construction)
    resource_overflow: jnp.ndarray  # sum of max(0, used - allocatable) across nodes/resources (raw units)
    resource_overflow_frac: jnp.ndarray  # same, but each term divided by that node's allocatable (unitless)
    anti_affinity_violations: jnp.ndarray  # count of pods with >=1 conflicting co-located pod
    affinity_violations: jnp.ndarray  # count of pods whose required affinity has no match
    topology_spread_violations: jnp.ndarray  # count of spread groups exceeding max_skew
    host_port_violations: jnp.ndarray  # count of colliding host-port pairs on the same node

    @property
    def total_violation_count(self) -> jnp.ndarray:
        return (
            self.mask_violations
            + self.anti_affinity_violations
            + self.affinity_violations
            + self.topology_spread_violations
            + self.host_port_violations
        )

    @property
    def is_feasible(self) -> jnp.ndarray:
        return (self.total_violation_count == 0) & (self.resource_overflow <= RESOURCE_EPS)


def _resource_overflow(assignment, pod_mask, pod_resources, node_resources, node_mask):
    used = _node_usage(assignment, pod_mask, pod_resources, node_resources.shape[0])
    over = _node_overflow(used, node_resources, node_mask)
    return jnp.sum(over)


def _resource_overflow_frac(assignment, pod_mask, pod_resources, node_resources, node_mask):
    """Overcommit as a fraction of each node's own capacity, summed over
    nodes and resources -- e.g. 0.25 means "one node is 25% over on one
    resource". Unit-free, so cores vs. GiB and 4-core vs. 32-core nodes all
    contribute on the same scale (the raw sum is dominated by memory GiB)."""
    used = _node_usage(assignment, pod_mask, pod_resources, node_resources.shape[0])
    over = _node_overflow(used, node_resources, node_mask)
    safe = jnp.where(node_resources > 0, node_resources, 1.0)
    return jnp.sum(over / safe)


def _mask_violations(assignment, pod_mask, action_mask):
    legal = jnp.take_along_axis(action_mask, assignment[:, None], axis=1)[:, 0]
    return jnp.sum(pod_mask & (~legal))


def _assigned_domains(assignment, node_hostname_domain, node_zone_domain):
    domain = jnp.stack([node_hostname_domain, node_zone_domain], axis=-1)  # [N, 2]
    return domain[assignment]  # [P, 2]


def _anti_affinity_violations(assignment_domains, col, match, pod_mask):
    num_pods = col.shape[0]
    col_safe = jnp.clip(col, 0, 1)
    pod_domain = jnp.take_along_axis(assignment_domains, col_safe[:, None], axis=1)[:, 0]
    has_constraint = (col >= 0) & pod_mask
    domain_eq = pod_domain[:, None] == pod_domain[None, :]
    not_self = ~jnp.eye(num_pods, dtype=bool)
    violating = match & domain_eq & not_self & has_constraint[:, None] & pod_mask[None, :]
    return jnp.sum(jnp.any(violating, axis=1))


def _affinity_violations(assignment_domains, col, match, pod_mask):
    num_pods = col.shape[0]
    col_safe = jnp.clip(col, 0, 1)
    pod_domain = jnp.take_along_axis(assignment_domains, col_safe[:, None], axis=1)[:, 0]
    has_constraint = (col >= 0) & pod_mask
    domain_eq = pod_domain[:, None] == pod_domain[None, :]
    not_self = ~jnp.eye(num_pods, dtype=bool)
    satisfied = jnp.any(match & domain_eq & not_self & pod_mask[None, :], axis=1)
    return jnp.sum(has_constraint & (~satisfied))


def _topology_spread_violations(
    assignment_domains, spread_col_of_group, spread_max_skew_of_group, spread_selector_match, domain_has_node, pod_mask
):
    num_domains = domain_has_node.shape[1]
    group_col_safe = jnp.clip(spread_col_of_group, 0, 1)  # [G]
    pod_domain_per_group = assignment_domains[:, group_col_safe].T  # [G, P]

    matches = spread_selector_match & pod_mask[None, :]  # [G, P]
    domain_ids = jnp.arange(num_domains)  # [D]
    onehot_domain = pod_domain_per_group[:, :, None] == domain_ids[None, None, :]  # [G, P, D]
    counts = jnp.einsum(
        "gp,gpd->gd", matches.astype(jnp.float32), onehot_domain.astype(jnp.float32), precision=_EXACT
    )  # [G, D]

    domain_valid = domain_has_node[group_col_safe]  # [G, D]
    counts_for_max = jnp.where(domain_valid, counts, -jnp.inf)
    counts_for_min = jnp.where(domain_valid, counts, jnp.inf)
    skew = jnp.max(counts_for_max, axis=1) - jnp.min(counts_for_min, axis=1)

    group_active = spread_col_of_group >= 0
    violating = group_active & (skew > spread_max_skew_of_group)
    return jnp.sum(violating)


def _host_port_violations(assignment, host_port_bucket, pod_mask):
    num_pods = host_port_bucket.shape[0]
    same_node = assignment[:, None] == assignment[None, :]
    pi = host_port_bucket[:, None, :, None]
    pk = host_port_bucket[None, :, None, :]
    port_eq = (pi == pk) & (pi != 0)
    port_conflict = jnp.any(port_eq, axis=(2, 3))
    not_self = ~jnp.eye(num_pods, dtype=bool)
    conflict = port_conflict & same_node & not_self & pod_mask[:, None] & pod_mask[None, :]
    return jnp.sum(conflict) // 2


def pod_violation_mask(assignment: jnp.ndarray, encoded) -> jnp.ndarray:
    """[P] bool: True for every real pod involved in at least one violation
    (partial credit, DESIGN.md sec 11f). Attribution rules:
      * illegal node (action mask)       -> that pod
      * resource overcommit on a node    -> every pod on that node
      * anti-affinity conflict           -> both pods of the conflicting pair
      * unmet required affinity          -> that pod
      * topology-spread group over skew  -> every pod the group's selector matches
      * host-port clash                  -> both pods of the clashing pair
    Any True <=> the assignment is infeasible (see tests)."""
    pod_mask = encoded.pod_mask
    num_pods = pod_mask.shape[0]
    not_self = ~jnp.eye(num_pods, dtype=bool)

    legal = jnp.take_along_axis(encoded.action_mask, assignment[:, None], axis=1)[:, 0]
    bad = ~legal

    used = _node_usage(assignment, pod_mask, encoded.pod_resources, encoded.node_resources.shape[0])
    node_over = jnp.any(_node_overflow(used, encoded.node_resources, encoded.node_mask) > 0, axis=1)
    bad = bad | node_over[assignment]

    domains = _assigned_domains(assignment, encoded.node_hostname_domain, encoded.node_zone_domain)

    def _pair_domain_eq(col):
        col_safe = jnp.clip(col, 0, 1)
        pod_domain = jnp.take_along_axis(domains, col_safe[:, None], axis=1)[:, 0]
        return (col >= 0) & pod_mask, pod_domain[:, None] == pod_domain[None, :]

    has_anti, eq = _pair_domain_eq(encoded.pod_anti_affinity_col)
    anti = encoded.pod_anti_affinity_match & eq & not_self & has_anti[:, None] & pod_mask[None, :]
    bad = bad | jnp.any(anti, axis=1) | jnp.any(anti, axis=0)

    has_aff, eq = _pair_domain_eq(encoded.pod_affinity_col)
    satisfied = jnp.any(encoded.pod_affinity_match & eq & not_self & pod_mask[None, :], axis=1)
    bad = bad | (has_aff & ~satisfied)

    num_domains = encoded.domain_has_node.shape[1]
    group_col_safe = jnp.clip(encoded.spread_col_of_group, 0, 1)
    pod_domain_per_group = domains[:, group_col_safe].T  # [G, P]
    matches = encoded.spread_selector_match & pod_mask[None, :]
    onehot_domain = pod_domain_per_group[:, :, None] == jnp.arange(num_domains)[None, None, :]
    counts = jnp.einsum(
        "gp,gpd->gd", matches.astype(jnp.float32), onehot_domain.astype(jnp.float32), precision=_EXACT
    )
    domain_valid = encoded.domain_has_node[group_col_safe]
    skew = jnp.max(jnp.where(domain_valid, counts, -jnp.inf), axis=1) - jnp.min(
        jnp.where(domain_valid, counts, jnp.inf), axis=1
    )
    group_bad = (encoded.spread_col_of_group >= 0) & (skew > encoded.spread_max_skew_of_group)
    bad = bad | jnp.any(group_bad[:, None] & matches, axis=0)

    same_node = assignment[:, None] == assignment[None, :]
    pi = encoded.host_port_bucket[:, None, :, None]
    pk = encoded.host_port_bucket[None, :, None, :]
    port_conflict = jnp.any((pi == pk) & (pi != 0), axis=(2, 3))
    clash = port_conflict & same_node & not_self & pod_mask[:, None] & pod_mask[None, :]
    bad = bad | jnp.any(clash, axis=1)

    return bad & pod_mask


def evaluate_assignment(assignment: jnp.ndarray, encoded) -> ViolationSummary:
    """`encoded` is a single-cluster `schedgym.encoding.Encoded` pytree
    (no leading batch dim) -- wrap with jax.vmap for a batch."""
    domains = _assigned_domains(assignment, encoded.node_hostname_domain, encoded.node_zone_domain)

    return ViolationSummary(
        mask_violations=_mask_violations(assignment, encoded.pod_mask, encoded.action_mask),
        resource_overflow=_resource_overflow(
            assignment, encoded.pod_mask, encoded.pod_resources, encoded.node_resources, encoded.node_mask
        ),
        resource_overflow_frac=_resource_overflow_frac(
            assignment, encoded.pod_mask, encoded.pod_resources, encoded.node_resources, encoded.node_mask
        ),
        anti_affinity_violations=_anti_affinity_violations(
            domains, encoded.pod_anti_affinity_col, encoded.pod_anti_affinity_match, encoded.pod_mask
        ),
        affinity_violations=_affinity_violations(
            domains, encoded.pod_affinity_col, encoded.pod_affinity_match, encoded.pod_mask
        ),
        topology_spread_violations=_topology_spread_violations(
            domains,
            encoded.spread_col_of_group,
            encoded.spread_max_skew_of_group,
            encoded.spread_selector_match,
            encoded.domain_has_node,
            encoded.pod_mask,
        ),
        host_port_violations=_host_port_violations(assignment, encoded.host_port_bucket, encoded.pod_mask),
    )


def packing_stats(assignment: jnp.ndarray, encoded) -> tuple[jnp.ndarray, jnp.ndarray]:
    """(active_node_count, mean_packing_density) using cpu+memory only,
    mirroring constraints.py's `_packing_stats`."""
    num_nodes = encoded.node_resources.shape[0]
    onehot = jax.nn.one_hot(assignment, num_nodes, dtype=jnp.float32) * encoded.pod_mask[:, None]
    used = jnp.einsum("pn,pr->nr", onehot, encoded.pod_resources, precision=_EXACT)  # [N, 4]
    is_active = (used[:, 0] > 0) | (used[:, 1] > 0)
    is_active = is_active & encoded.node_mask
    active_node_count = jnp.sum(is_active)

    total_alloc = encoded.node_resources[:, :2]  # cpu, mem (gib)
    safe_alloc = jnp.where(total_alloc > 0, total_alloc, 1.0)
    frac = used[:, :2] / safe_alloc  # [N, 2]
    density_per_node = jnp.mean(frac, axis=-1)  # [N]
    density_sum = jnp.sum(jnp.where(is_active, density_per_node, 0.0))
    mean_density = density_sum / jnp.maximum(active_node_count, 1)
    return active_node_count, mean_density
