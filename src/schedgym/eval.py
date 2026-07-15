"""Compare the learned policy against heuristic baselines (DESIGN.md sec 9).

All feasibility/packing numbers are computed with the Python oracle
validator (constraints.py), not the JAX reward path, so baselines and
policy are scored by the same ground truth regardless of how their
assignment was produced.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from schedgym.baselines import best_fit_decreasing, random_feasible
from schedgym.constraints import UNASSIGNED, validate_assignment
from schedgym.encoding import encode_batch
from schedgym.generator import generate_dataset
from schedgym.types import ClusterBatch


@dataclass
class Metrics:
    name: str
    num_clusters: int
    fully_scheduled_rate: float = 0.0
    feasible_rate: float = 0.0
    success_rate: float = 0.0  # fully scheduled AND feasible
    mean_active_nodes: float = 0.0
    mean_packing_density: float = 0.0

    def print(self) -> None:
        print(
            f"{self.name:>18s} | success {self.success_rate:6.1%} | "
            f"feasible {self.feasible_rate:6.1%} | fully_scheduled {self.fully_scheduled_rate:6.1%} | "
            f"active_nodes {self.mean_active_nodes:5.2f} | density {self.mean_packing_density:5.3f}"
        )


def _score_assignments(name: str, clusters: list[ClusterBatch], assignments: list[list[int]]) -> Metrics:
    n = len(clusters)
    fully_scheduled = 0
    feasible = 0
    success = 0
    active_nodes_sum = 0.0
    density_sum = 0.0

    for cluster, assignment in zip(clusters, assignments):
        result = validate_assignment(cluster, assignment)
        is_full = all(a != UNASSIGNED for a in assignment)
        fully_scheduled += int(is_full)
        feasible += int(result.is_feasible)
        if is_full and result.is_feasible:
            success += 1
            active_nodes_sum += result.active_node_count
            density_sum += result.packing_density

    return Metrics(
        name=name,
        num_clusters=n,
        fully_scheduled_rate=fully_scheduled / n,
        feasible_rate=feasible / n,
        success_rate=success / n,
        mean_active_nodes=active_nodes_sum / success if success else float("nan"),
        mean_packing_density=density_sum / success if success else float("nan"),
    )


def evaluate_baselines(clusters: list[ClusterBatch], seed: int = 0) -> list[Metrics]:
    rng = random.Random(seed)
    bfd_assignments = [best_fit_decreasing(c) for c in clusters]
    random_assignments = [random_feasible(c, rng) for c in clusters]
    return [
        _score_assignments("best_fit_decreasing", clusters, bfd_assignments),
        _score_assignments("random_feasible", clusters, random_assignments),
    ]


def evaluate_policy(
    params, model, clusters: list[ClusterBatch], max_pods: int, max_nodes: int, name: str = "learned_policy"
) -> Metrics:
    encoded = encode_batch(clusters, max_pods, max_nodes)

    def single(enc):
        logits, _value = model.apply(params, enc)
        return jnp.argmax(logits, axis=-1)

    actions = jax.vmap(single)(encoded)
    actions_np = jax.device_get(actions)

    assignments = []
    for cluster, action_row in zip(clusters, actions_np):
        num_pods = len(cluster.pods)
        assignments.append([int(a) for a in action_row[:num_pods]])

    return _score_assignments(name, clusters, assignments)


def run_eval(params, model, max_pods: int, max_nodes: int, num_clusters: int = 200, seed: int = 12345):
    clusters = generate_dataset(num_clusters, seed=seed)
    results = evaluate_baselines(clusters, seed=seed)
    results.append(evaluate_policy(params, model, clusters, max_pods, max_nodes))
    for m in results:
        m.print()
    return results


if __name__ == "__main__":
    import argparse

    from schedgym.train import TrainConfig, train

    parser = argparse.ArgumentParser(description="Train briefly then compare policy vs baselines")
    parser.add_argument("--train-iterations", type=int, default=50)
    parser.add_argument("--eval-clusters", type=int, default=100)
    args = parser.parse_args()

    cfg = TrainConfig(num_iterations=args.train_iterations, dataset_size=512, batch_size=64, log_every=10)
    params, model, _history = train(cfg)
    run_eval(params, model, cfg.max_pods, cfg.max_nodes, num_clusters=args.eval_clusters)
