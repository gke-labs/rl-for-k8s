"""One-shot scheduling "environment": reward = feasibility + consolidation.

There is no multi-step transition here by design (DESIGN.md sec 8) --
each episode is exactly one full pod->node assignment, scored once. What
would normally be `reset`/`step` collapses into sampling actions from the
policy and scoring them against a pre-encoded cluster.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from schedgym.jax_constraints import evaluate_assignment, packing_stats, pod_violation_mask

# All reward terms are O(1)-per-unit so rewards stay in roughly [-40, 0]
# (DESIGN.md sec 11c). Per discrete violation (anti-affinity pod, spread
# group, port clash, ...):
FEASIBILITY_VIOLATION_WEIGHT = 2.0
# Per unit of `resource_overflow_frac` (1.0 == one node 100% over on one
# resource), so a node 20% over on memory costs 1.0:
RESOURCE_OVERFLOW_WEIGHT = 5.0
CONSOLIDATION_WEIGHT = 1.0
DENSITY_BONUS_WEIGHT = 2.0
# Extra gap between the worst feasible and the best infeasible reward.
INFEASIBLE_MARGIN = 1.0


class StepResult(NamedTuple):
    reward: jnp.ndarray
    is_feasible: jnp.ndarray
    active_node_count: jnp.ndarray
    packing_density: jnp.ndarray
    violation_count: jnp.ndarray
    pod_ok: jnp.ndarray  # [P] float: 1.0 if the pod is in no violation (padding pods: 0)
    legal_frac: jnp.ndarray  # fraction of real pods with pod_ok


def compute_reward(assignment: jnp.ndarray, encoded, partial_credit_weight: float = 0.0) -> StepResult:
    """`partial_credit_weight` (w >= 0, DESIGN.md sec 11f): infeasible
    assignments additionally earn w * (fraction of pods involved in no
    violation). The infeasible band is shifted down by w so the best
    infeasible reward stays strictly below the worst feasible one."""
    summary = evaluate_assignment(assignment, encoded)
    active_node_count, density = packing_stats(assignment, encoded)
    pod_bad = pod_violation_mask(assignment, encoded)
    pod_ok = (encoded.pod_mask & ~pod_bad).astype(jnp.float32)
    legal_frac = jnp.sum(pod_ok) / jnp.maximum(jnp.sum(encoded.pod_mask), 1)

    infeasibility_penalty = (
        summary.total_violation_count.astype(jnp.float32) * FEASIBILITY_VIOLATION_WEIGHT
        + summary.resource_overflow_frac * RESOURCE_OVERFLOW_WEIGHT
    )
    # A feasible assignment scores >= -num_nodes (every node active, zero
    # density bonus). Start infeasible rewards strictly below that, so
    # "slightly illegal" can never outscore "legal but spread out" -- with
    # normalized penalties a tiny overcommit would otherwise cost < 1.
    num_nodes = jnp.sum(encoded.node_mask).astype(jnp.float32)
    infeasible_floor = num_nodes * CONSOLIDATION_WEIGHT + INFEASIBLE_MARGIN + partial_credit_weight
    # legal_frac < 1 whenever infeasible, so this stays below -(num_nodes + margin)
    partial_credit = partial_credit_weight * legal_frac
    consolidation_score = (
        -active_node_count.astype(jnp.float32) * CONSOLIDATION_WEIGHT + density * DENSITY_BONUS_WEIGHT
    )

    # Consolidation is only rewarded once the assignment is fully feasible,
    # so the policy never learns "efficient but broken" placements.
    reward = jnp.where(summary.is_feasible, consolidation_score, -(infeasible_floor + infeasibility_penalty) + partial_credit)

    return StepResult(
        reward=reward,
        is_feasible=summary.is_feasible,
        active_node_count=active_node_count,
        packing_density=density,
        violation_count=summary.total_violation_count,
        pod_ok=pod_ok,
        legal_frac=legal_frac,
    )


def sample_actions(logits: jnp.ndarray, rng: jax.Array) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Independent per-pod categorical sample from masked logits (the
    "one-shot" joint action -- see DESIGN.md sec 5/7 for why coordination
    across pods has to come from the shared encoder, not the sampling
    order). Returns (actions [P], log_prob_per_pod [P])."""
    actions = jax.random.categorical(rng, logits, axis=-1)
    log_probs_all = jax.nn.log_softmax(logits, axis=-1)
    log_prob = jnp.take_along_axis(log_probs_all, actions[:, None], axis=1)[:, 0]
    return actions, log_prob


def joint_log_prob(log_prob_per_pod: jnp.ndarray, pod_mask: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(jnp.where(pod_mask, log_prob_per_pod, 0.0))
