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

from schedgym.jax_constraints import evaluate_assignment, packing_stats

FEASIBILITY_VIOLATION_WEIGHT = 10.0
RESOURCE_OVERFLOW_WEIGHT = 5.0
CONSOLIDATION_WEIGHT = 1.0
DENSITY_BONUS_WEIGHT = 2.0


class StepResult(NamedTuple):
    reward: jnp.ndarray
    is_feasible: jnp.ndarray
    active_node_count: jnp.ndarray
    packing_density: jnp.ndarray
    violation_count: jnp.ndarray


def compute_reward(assignment: jnp.ndarray, encoded) -> StepResult:
    summary = evaluate_assignment(assignment, encoded)
    active_node_count, density = packing_stats(assignment, encoded)

    infeasibility_penalty = (
        summary.total_violation_count.astype(jnp.float32) * FEASIBILITY_VIOLATION_WEIGHT
        + summary.resource_overflow * RESOURCE_OVERFLOW_WEIGHT
    )
    consolidation_score = (
        -active_node_count.astype(jnp.float32) * CONSOLIDATION_WEIGHT + density * DENSITY_BONUS_WEIGHT
    )

    # Consolidation is only rewarded once the assignment is fully feasible,
    # so the policy never learns "efficient but broken" placements.
    reward = jnp.where(summary.is_feasible, consolidation_score, -infeasibility_penalty)

    return StepResult(
        reward=reward,
        is_feasible=summary.is_feasible,
        active_node_count=active_node_count,
        packing_density=density,
        violation_count=summary.total_violation_count,
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
