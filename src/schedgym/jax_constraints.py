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


class ViolationSummary(NamedTuple):
    mask_violations: jnp.ndarray  # pods placed on an action_mask-illegal node (should be ~0 by construction)
    resource_overflow: jnp.ndarray  # sum of max(0, used - allocatable) across nodes/resources
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
    num_nodes = node_resources.shape[0]
    onehot = jax.nn.one_hot(assignment, num_nodes, dtype=jnp.float32) * pod_mask[:, None]
    used = jnp.einsum("pn,pr->nr", onehot, pod_resources)
    over = jnp.maximum(used - node_resources, 0.0) * node_mask[:, None]
    return jnp.sum(over)


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
    counts = jnp.einsum("gp,gpd->gd", matches.astype(jnp.float32), onehot_domain.astype(jnp.float32))  # [G, D]

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


def evaluate_assignment(assignment: jnp.ndarray, encoded) -> ViolationSummary:
    """`encoded` is a single-cluster `schedgym.encoding.Encoded` pytree
    (no leading batch dim) -- wrap with jax.vmap for a batch."""
    domains = _assigned_domains(assignment, encoded.node_hostname_domain, encoded.node_zone_domain)

    return ViolationSummary(
        mask_violations=_mask_violations(assignment, encoded.pod_mask, encoded.action_mask),
        resource_overflow=_resource_overflow(
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
    used = jnp.einsum("pn,pr->nr", onehot, encoded.pod_resources)  # [N, 4]
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
