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
from schedgym.env import compute_reward
from schedgym.generator import TIERS, generate_dataset, generate_dataset_with_info
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


def policy_assignments(
    params,
    model,
    clusters: list[ClusterBatch],
    max_pods: int,
    max_nodes: int,
    samples: int = 1,
    seed: int = 0,
    chunk: int = 512,
) -> list[list[int]]:
    """Assignments from the learned policy.

    samples == 1: per-pod argmax (single shot).
    samples > 1:  best-of-K "sample and check" (DESIGN.md sec 11j/11k):
                  draw K assignments, add the argmax as a candidate, and
                  pick the feasible candidate with the highest reward
                  (fewest active nodes, densest packing). If none are
                  feasible, fall back to the candidate with the most pods
                  in no violation. Feasibility here comes from the JAX
                  checker; scoring in `_score_assignments` re-checks with
                  the Python oracle.
    """
    out: list[list[int]] = []
    key = jax.random.PRNGKey(seed)

    @jax.jit
    def pick(enc, k):
        logits = jax.vmap(lambda e: model.apply(params, e)[0])(enc)  # [B, P, N]
        argmax = jnp.argmax(logits, axis=-1)
        if samples <= 1:
            return argmax
        keys = jax.random.split(k, samples)
        sampled = jax.vmap(lambda kk: jax.random.categorical(kk, logits, axis=-1))(keys)  # [K, B, P]
        cands = jnp.concatenate([argmax[None], sampled], axis=0)  # [K+1, B, P]
        steps = jax.vmap(lambda a: jax.vmap(compute_reward)(a, enc))(cands)
        score = jnp.where(steps.is_feasible, steps.reward, -1e9 + steps.legal_frac)
        best = jnp.argmax(score, axis=0)
        return cands[best, jnp.arange(cands.shape[1])]

    for lo in range(0, len(clusters), chunk):
        part = clusters[lo : lo + chunk]
        enc = encode_batch(part, max_pods, max_nodes)
        key, sub = jax.random.split(key)
        actions = jax.device_get(pick(enc, sub))
        for cluster, row in zip(part, actions):
            out.append([int(a) for a in row[: len(cluster.pods)]])
    return out


def evaluate_policy(
    params,
    model,
    clusters: list[ClusterBatch],
    max_pods: int,
    max_nodes: int,
    name: str = "learned_policy",
    samples: int = 1,
    seed: int = 0,
) -> Metrics:
    assignments = policy_assignments(params, model, clusters, max_pods, max_nodes, samples=samples, seed=seed)
    return _score_assignments(name, clusters, assignments)


def evaluate_by_tier(name: str, generated, assignments: list[list[int]]) -> dict[str, Metrics]:
    """Metrics overall, on schedulable clusters only, and per load tier
    (needs `generate_dataset_with_info` output)."""
    groups = {"overall": list(range(len(generated)))}
    groups["schedulable"] = [i for i, g in enumerate(generated) if g.tier != "overflow"]
    for tier in TIERS:
        groups[tier] = [i for i, g in enumerate(generated) if g.tier == tier]
    return {
        key: _score_assignments(f"{name}[{key}]", [generated[i].cluster for i in idx], [assignments[i] for i in idx])
        for key, idx in groups.items()
        if idx
    }


def print_tier_table(rows: dict[str, dict[str, Metrics]]) -> None:
    cols = ["overall", "schedulable", *TIERS]
    print(f"{'':34s}" + "".join(f"{c:>13s}" for c in cols))
    for name, by_tier in rows.items():
        print(f"{name:34s}" + "".join(f"{by_tier[c].success_rate:13.1%}" if c in by_tier else f"{'-':>13s}" for c in cols))


def run_eval(params, model, max_pods: int, max_nodes: int, num_clusters: int = 200, seed: int = 12345):
    clusters = generate_dataset(num_clusters, seed=seed)
    results = evaluate_baselines(clusters, seed=seed)
    results.append(evaluate_policy(params, model, clusters, max_pods, max_nodes))
    for m in results:
        m.print()
    return results


def main(argv=None) -> None:
    """Score checkpoint(s) against the heuristic baselines on a fresh
    tiered eval set, single-shot and best-of-K, broken down by load tier.

        python -m schedgym.eval runs/<run>/checkpoints/si.pkl --samples 1 16 64
    """
    import argparse

    from schedgym.network import SchedulerPolicy
    from schedgym.train import TrainConfig, load_checkpoint

    parser = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="*", help="checkpoint .pkl files to score")
    parser.add_argument("--clusters", type=int, default=300)
    parser.add_argument("--seed", type=int, default=424242, help="eval-set seed (default matches DESIGN.md 11b-11j)")
    parser.add_argument("--samples", type=int, nargs="+", default=[1, 16], help="1 = argmax; >1 = best-of-K")
    parser.add_argument("--no-baselines", action="store_true")
    args = parser.parse_args(argv)

    cfg = TrainConfig()
    generated = generate_dataset_with_info(args.clusters, seed=args.seed)
    clusters = [g.cluster for g in generated]
    rows: dict[str, dict[str, Metrics]] = {}
    if not args.no_baselines:
        rng = random.Random(args.seed)
        rows["best_fit_decreasing"] = evaluate_by_tier("bfd", generated, [best_fit_decreasing(c) for c in clusters])
        rows["random_feasible"] = evaluate_by_tier("rand", generated, [random_feasible(c, rng) for c in clusters])
        rows["witness (ceiling)"] = evaluate_by_tier("witness", generated, [g.witness for g in generated])
    model = SchedulerPolicy()
    for path in args.checkpoints:
        params = load_checkpoint(path)
        for k in args.samples:
            assignments = policy_assignments(params, model, clusters, cfg.max_pods, cfg.max_nodes, samples=k, seed=args.seed)
            label = f"{'argmax' if k == 1 else f'best-of-{k}'} {path.split('runs/')[-1]}"[:34]
            rows[label] = evaluate_by_tier(label, generated, assignments)
    print(f"success rate (fully scheduled + feasible), {args.clusters} clusters, seed {args.seed}")
    print_tier_table(rows)


if __name__ == "__main__":
    main()
