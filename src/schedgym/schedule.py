"""Schedule pods onto nodes with a trained policy (DESIGN.md sec 11j/11k).

Recommended use is best-of-K "sample and check": draw K one-shot
assignments from the policy, keep the feasible one with the best
consolidation score, and verify it with the Python oracle. If no sample is
feasible, fall back to `best_fit_decreasing`, which (like kube-scheduler)
may leave pods Pending.

    python -m schedgym.schedule examples/sample_pods.yaml examples/sample_nodes.yaml \\
        --checkpoint runs/si3_L3_si30_ppo_kl0.5/checkpoints/si.pkl --samples 16
"""

from __future__ import annotations

from dataclasses import dataclass

from schedgym.baselines import best_fit_decreasing
from schedgym.constraints import UNASSIGNED, validate_assignment
from schedgym.eval import policy_assignments
from schedgym.types import ClusterBatch

DEFAULT_CHECKPOINT = "runs/si3_L3_si30_ppo_kl0.5/checkpoints/si.pkl"


@dataclass
class ScheduleResult:
    assignment: list[int]  # node index per pod, UNASSIGNED (-1) = Pending
    source: str  # "policy" or "best_fit_decreasing"
    feasible: bool
    active_nodes: int
    packing_density: float


def schedule(
    cluster: ClusterBatch, params, model=None, samples: int = 16, seed: int = 0, max_pods: int = 20, max_nodes: int = 12
) -> ScheduleResult:
    if model is None:
        from schedgym.network import SchedulerPolicy

        model = SchedulerPolicy()
    if len(cluster.pods) > max_pods or len(cluster.nodes) > max_nodes:
        raise ValueError(f"policy handles at most {max_pods} pods / {max_nodes} nodes per batch")

    assignment = policy_assignments(params, model, [cluster], max_pods, max_nodes, samples=samples, seed=seed)[0]
    result = validate_assignment(cluster, assignment)
    source = "policy"
    if not result.is_feasible:
        assignment = best_fit_decreasing(cluster)
        result = validate_assignment(cluster, assignment)
        source = "best_fit_decreasing"
    return ScheduleResult(
        assignment=assignment,
        source=source,
        feasible=result.is_feasible,
        active_nodes=result.active_node_count,
        packing_density=result.packing_density,
    )


def main(argv=None) -> None:
    import argparse

    from schedgym.train import load_checkpoint
    from schedgym.yaml_io import load_cluster

    parser = argparse.ArgumentParser(description="Schedule pod YAML onto node YAML with a trained policy")
    parser.add_argument("pods")
    parser.add_argument("nodes")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--samples", type=int, default=16, help="best-of-K samples (1 = argmax only)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    cluster = load_cluster(args.pods, args.nodes)
    res = schedule(cluster, load_checkpoint(args.checkpoint), samples=args.samples, seed=args.seed)
    for pod, node_idx in zip(cluster.pods, res.assignment):
        target = "<Pending>" if node_idx == UNASSIGNED else cluster.nodes[node_idx].name
        print(f"{pod.name:24s} -> {target}")
    print(
        f"source={res.source} feasible={res.feasible} active_nodes={res.active_nodes} "
        f"density={res.packing_density:.3f}"
    )


if __name__ == "__main__":
    main()
