"""Synthetic pod/node batch generator for training and evaluation.

Produces randomized clusters that exercise the full constraint set
(resources, taints/tolerations, node selector/affinity, pod
affinity/anti-affinity, topology spread, host ports) so the policy and
the validator both get meaningful signal. See DESIGN.md sec 6/9.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace

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

_GIB = 1024**3
_MIB = 1024**2


@dataclass(frozen=True)
class GeneratorConfig:
    min_nodes: int = 4
    max_nodes: int = 12
    min_pods: int = 4
    max_pods: int = 20

    zones: tuple[str, ...] = ("us-east-1a", "us-east-1b", "us-east-1c")
    node_cpu_choices: tuple[float, ...] = (4.0, 8.0, 16.0, 32.0)
    node_memory_gib_choices: tuple[float, ...] = (16.0, 32.0, 64.0, 128.0)
    gpu_node_fraction: float = 0.25
    gpu_choices: tuple[int, ...] = (1, 2, 4, 8)
    tainted_gpu_node_fraction: float = 0.7

    pod_cpu_choices: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
    pod_memory_mib_choices: tuple[float, ...] = (128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0)
    gpu_pod_fraction: float = 0.12
    zone_pinned_fraction: float = 0.15
    spread_group_fraction: float = 0.35
    co_locate_fraction: float = 0.1
    host_port_fraction: float = 0.1
    host_port_pool: tuple[int, ...] = (8080, 6379, 5432, 9090)

    app_names: tuple[str, ...] = ("web", "api", "cache", "worker", "ml-train", "db")
    disktypes: tuple[str, ...] = ("ssd", "hdd")


def _generate_node(rng: random.Random, index: int, config: GeneratorConfig) -> NodeSpec:
    name = f"node-{index}"
    zone = rng.choice(config.zones)
    is_gpu = rng.random() < config.gpu_node_fraction
    labels = {
        "kubernetes.io/hostname": name,
        "topology.kubernetes.io/zone": zone,
        "disktype": rng.choice(config.disktypes),
    }
    taints: tuple[Taint, ...] = ()
    gpu_allocatable = 0
    if is_gpu:
        labels["accelerator"] = "nvidia-a100"
        gpu_allocatable = rng.choice(config.gpu_choices)
        if rng.random() < config.tainted_gpu_node_fraction:
            taints = (Taint(key="dedicated", value="gpu", effect=TaintEffect.NO_SCHEDULE),)

    return NodeSpec(
        name=name,
        cpu_allocatable=rng.choice(config.node_cpu_choices),
        memory_allocatable=rng.choice(config.node_memory_gib_choices) * _GIB,
        gpu_allocatable=gpu_allocatable,
        ephemeral_storage_allocatable=100.0 * _GIB,
        labels=labels,
        taints=taints,
    )


def _generate_pod(rng: random.Random, index: int, config: GeneratorConfig) -> PodSpec:
    app = rng.choice(config.app_names)
    name = f"{app}-{index}"
    labels = {"app": app}

    needs_gpu = rng.random() < config.gpu_pod_fraction
    tolerations: tuple[Toleration, ...] = ()
    node_selector: dict[str, str] = {}
    gpu_request = 0
    if needs_gpu:
        gpu_request = 1
        node_selector["accelerator"] = "nvidia-a100"
        tolerations = (
            Toleration(
                key="dedicated",
                operator=TolerationOperator.EQUAL,
                value="gpu",
                effect=TaintEffect.NO_SCHEDULE,
            ),
        )
        cpu_request = rng.choice(config.pod_cpu_choices[-3:])
        memory_request = rng.choice(config.pod_memory_mib_choices[-3:]) * _MIB
    else:
        cpu_request = rng.choice(config.pod_cpu_choices)
        memory_request = rng.choice(config.pod_memory_mib_choices) * _MIB

    node_affinity = None
    if rng.random() < config.zone_pinned_fraction:
        zone = rng.choice(config.zones)
        node_affinity = NodeAffinity(
            required=(
                NodeSelectorTerm(
                    match_expressions=(
                        LabelSelectorRequirement(
                            key="topology.kubernetes.io/zone",
                            operator=SelectorOperator.IN,
                            values=(zone,),
                        ),
                    )
                ),
            )
        )

    topology_spread_constraints: tuple[TopologySpreadConstraint, ...] = ()
    pod_anti_affinity = None
    if rng.random() < config.spread_group_fraction:
        selector = LabelSelector(match_labels={"app": app})
        pod_anti_affinity = PodAffinity(
            required=(PodAffinityTerm(label_selector=selector, topology_key="kubernetes.io/hostname"),)
        )
        topology_spread_constraints = (
            TopologySpreadConstraint(
                topology_key="topology.kubernetes.io/zone",
                max_skew=1,
                label_selector=selector,
                when_unsatisfiable="DoNotSchedule",
            ),
        )

    pod_affinity = None
    if rng.random() < config.co_locate_fraction:
        target = rng.choice([a for a in config.app_names if a != app])
        pod_affinity = PodAffinity(
            required=(
                PodAffinityTerm(
                    label_selector=LabelSelector(match_labels={"app": target}),
                    topology_key="kubernetes.io/hostname",
                ),
            )
        )

    host_ports: tuple[int, ...] = ()
    if rng.random() < config.host_port_fraction:
        host_ports = (rng.choice(config.host_port_pool),)

    return PodSpec(
        name=name,
        cpu_request=cpu_request,
        memory_request=memory_request,
        gpu_request=gpu_request,
        labels=labels,
        tolerations=tolerations,
        node_selector=node_selector,
        node_affinity=node_affinity,
        pod_affinity=pod_affinity,
        pod_anti_affinity=pod_anti_affinity,
        topology_spread_constraints=topology_spread_constraints,
        host_ports=host_ports,
    )


def generate_cluster(rng: random.Random, config: GeneratorConfig = GeneratorConfig()) -> ClusterBatch:
    num_nodes = rng.randint(config.min_nodes, config.max_nodes)
    num_pods = rng.randint(config.min_pods, config.max_pods)
    nodes = tuple(_generate_node(rng, i, config) for i in range(num_nodes))
    pods = tuple(_generate_pod(rng, i, config) for i in range(num_pods))
    return ClusterBatch(pods=pods, nodes=nodes)


def generate_dataset(
    num_clusters: int, seed: int = 0, config: GeneratorConfig = GeneratorConfig()
) -> list[ClusterBatch]:
    rng = random.Random(seed)
    return [generate_cluster(rng, config) for _ in range(num_clusters)]


def easy_config() -> GeneratorConfig:
    """Smallest, resource-fit-only starting point for curriculum training
    (DESIGN.md sec 8/11a): small clusters, no taints/GPU, no affinity, no
    topology spread, no host ports -- just cpu/memory bin packing."""
    return GeneratorConfig(
        min_nodes=3,
        max_nodes=5,
        min_pods=3,
        max_pods=6,
        gpu_node_fraction=0.0,
        tainted_gpu_node_fraction=0.0,
        gpu_pod_fraction=0.0,
        zone_pinned_fraction=0.0,
        spread_group_fraction=0.0,
        co_locate_fraction=0.0,
        host_port_fraction=0.0,
    )


_CURRICULUM_FIELDS = (
    "min_nodes",
    "max_nodes",
    "min_pods",
    "max_pods",
    "gpu_node_fraction",
    "tainted_gpu_node_fraction",
    "gpu_pod_fraction",
    "zone_pinned_fraction",
    "spread_group_fraction",
    "co_locate_fraction",
    "host_port_fraction",
)
_CURRICULUM_INT_FIELDS = {"min_nodes", "max_nodes", "min_pods", "max_pods"}


def interpolate_config(easy: GeneratorConfig, full: GeneratorConfig, t: float) -> GeneratorConfig:
    """Linearly interpolate the difficulty knobs from `easy` (t=0) to
    `full` (t=1); everything else (value-choice tuples, app/zone vocab)
    stays at `full` throughout since it doesn't control difficulty."""
    t = max(0.0, min(1.0, t))
    updates = {}
    for name in _CURRICULUM_FIELDS:
        a, b = getattr(easy, name), getattr(full, name)
        value = a + (b - a) * t
        updates[name] = round(value) if name in _CURRICULUM_INT_FIELDS else value
    return replace(full, **updates)
