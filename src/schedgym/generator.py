"""Synthetic pod/node batch generator for training and evaluation.

Every cluster is built around a hidden *witness* assignment, so we know
for certain whether it can be scheduled. Pod sizes are cut from node
capacity on that witness, and every optional constraint (GPU
taints/tolerations, zone pinning, anti-affinity + topology spread,
co-location affinity, host ports) is chosen so the witness satisfies
it. Each cluster is then checked against the Python oracle
(constraints.py).

Each cluster belongs to one *load tier*. The share of each tier is
controlled by `GeneratorConfig.tier_weights` (default 50/40/10):

  * ``high``      -- schedulable; pods request [80%, 90%) of the cluster's
                     total allocatable CPU *and* memory.
  * ``very_high`` -- schedulable; pods request [90%, 100%] of CPU and
                     memory.
  * ``overflow``  -- NOT schedulable; total pod requests exceed total
                     allocatable CPU and/or memory (105%-130%). The extra
                     pods are unassigned (UNASSIGNED) in the witness.

`generate_dataset` assigns tiers by exact stratification (shuffled), so a
100-cluster eval set is exactly 50/40/10 rather than only on average.
See DESIGN.md sec 6/9/11b.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import NamedTuple

from schedgym.constraints import UNASSIGNED, validate_assignment
from schedgym.types import (
    ClusterBatch,
    LabelSelector,
    LabelSelectorRequirement,
    NodeAffinity,
    NodeSelectorTerm,
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

# Request granularity: 10 millicores / 16 MiB keeps requests realistic
# while letting a node be packed within ~1% of its capacity.
_CPU_STEP = 0.01
_MEM_STEP = 16 * _MIB

HOSTNAME_KEY = "kubernetes.io/hostname"
ZONE_KEY = "topology.kubernetes.io/zone"

TIER_HIGH = "high"
TIER_VERY_HIGH = "very_high"
TIER_OVERFLOW = "overflow"
TIERS = (TIER_HIGH, TIER_VERY_HIGH, TIER_OVERFLOW)

_GPU_TOLERATION = Toleration(
    key="dedicated", operator=TolerationOperator.EQUAL, value="gpu", effect=TaintEffect.NO_SCHEDULE
)


@dataclass(frozen=True)
class GeneratorConfig:
    min_nodes: int = 4
    max_nodes: int = 12
    min_pods: int = 4
    max_pods: int = 20
    max_pods_per_node: int = 4

    zones: tuple[str, ...] = ("us-east-1a", "us-east-1b", "us-east-1c")
    node_cpu_choices: tuple[float, ...] = (4.0, 8.0, 16.0, 32.0)
    node_memory_gib_choices: tuple[float, ...] = (16.0, 32.0, 64.0, 128.0)
    gpu_node_fraction: float = 0.25
    gpu_choices: tuple[int, ...] = (1, 2, 4, 8)
    tainted_gpu_node_fraction: float = 0.7

    # Load tiers: (high, very_high, overflow) probabilities, plus the
    # aggregate-utilization band each tier is drawn from.
    tier_weights: tuple[float, float, float] = (0.5, 0.4, 0.1)
    high_band: tuple[float, float] = (0.80, 0.90)
    very_high_band: tuple[float, float] = (0.90, 1.00)
    overflow_band: tuple[float, float] = (1.05, 1.30)
    max_overflow_pods: int = 3

    # On an untainted GPU node, chance that a witness pod there is a GPU pod.
    gpu_pod_fraction: float = 0.5
    zone_pinned_fraction: float = 0.15
    spread_group_fraction: float = 0.35  # target fraction of pods in an anti-affinity/spread group
    max_spread_groups: int = 3
    co_locate_fraction: float = 0.1
    host_port_fraction: float = 0.1
    host_port_pool: tuple[int, ...] = (8080, 6379, 5432, 9090)

    app_names: tuple[str, ...] = ("web", "api", "cache", "worker", "ml-train", "db")
    disktypes: tuple[str, ...] = ("ssd", "hdd")


class GeneratedCluster(NamedTuple):
    cluster: ClusterBatch
    tier: str
    # A known-feasible assignment. Overflow pods (overflow tier) are
    # UNASSIGNED, because no assignment can place them.
    witness: list[int]


def cluster_utilization(cluster: ClusterBatch) -> tuple[float, float]:
    """(cpu, memory) total pod requests / total node allocatable."""
    cap_cpu = sum(n.cpu_allocatable for n in cluster.nodes)
    cap_mem = sum(n.memory_allocatable for n in cluster.nodes)
    req_cpu = sum(p.cpu_request for p in cluster.pods)
    req_mem = sum(p.memory_request for p in cluster.pods)
    return req_cpu / cap_cpu, req_mem / cap_mem


# --------------------------------------------------------------------------
# nodes


def _generate_node(rng: random.Random, index: int, config: GeneratorConfig) -> NodeSpec:
    name = f"node-{index}"
    zone = rng.choice(config.zones)
    is_gpu = rng.random() < config.gpu_node_fraction
    labels = {HOSTNAME_KEY: name, ZONE_KEY: zone, "disktype": rng.choice(config.disktypes)}
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


# --------------------------------------------------------------------------
# resource carving


def _split(rng: random.Random, total: float, k: int, step: float) -> list[float]:
    """Split `total` into k positive, step-aligned parts summing to <= total
    (rounded down so the node never overflows)."""
    units = int(math.floor(total / step + 1e-9))
    units = max(units, k)
    # Uneven split, with each part getting at least 1 unit; this gives a
    # mix of "big" and "small" pods on a node.
    weights = [rng.random() ** 1.5 + 0.05 for _ in range(k)]
    s = sum(weights)
    parts = [max(1, int(units * w / s)) for w in weights]
    # hand out the rounding remainder so the node is filled as specified
    remainder = units - sum(parts)
    i = 0
    while remainder > 0:
        parts[i % k] += 1
        remainder -= 1
        i += 1
    while remainder < 0:  # only possible when max(1, ...) bumped tiny parts
        j = max(range(k), key=lambda x: parts[x])
        parts[j] -= 1
        remainder += 1
    return [p * step for p in parts]


def _pods_per_node(rng: random.Random, num_nodes: int, total_pods: int, cap: list[int]) -> list[int]:
    counts = [1] * num_nodes
    extra = total_pods - num_nodes
    while extra > 0:
        candidates = [i for i in range(num_nodes) if counts[i] < cap[i]]
        if not candidates:
            break
        counts[rng.choice(candidates)] += 1
        extra -= 1
    return counts


def _sample_tier(rng: random.Random, config: GeneratorConfig) -> str:
    return rng.choices(TIERS, weights=config.tier_weights, k=1)[0]


# --------------------------------------------------------------------------
# cluster


class _Slot:
    """A witness pod: where it lives + what it requests, before it becomes a PodSpec."""

    __slots__ = ("node", "cpu", "mem", "gpu", "app", "spread", "co_locate_app", "zone_pin", "host_port")

    def __init__(self, node: int, cpu: float, mem: float, gpu: int = 0):
        self.node = node
        self.cpu = cpu
        self.mem = mem
        self.gpu = gpu
        self.app: str | None = None
        self.spread = False
        self.co_locate_app: str | None = None
        self.zone_pin: str | None = None
        self.host_port: int | None = None


def _build_slots(
    rng: random.Random, nodes: tuple[NodeSpec, ...], tier: str, config: GeneratorConfig, pod_budget: int
) -> list[_Slot] | None:
    lo, hi = config.high_band if tier == TIER_HIGH else config.very_high_band
    num_nodes = len(nodes)
    min_total = max(config.min_pods, num_nodes)
    if min_total > pod_budget:
        return None
    total_pods = rng.randint(min_total, pod_budget)

    caps = []
    for node in nodes:
        cap = config.max_pods_per_node
        if node.taints:  # tainted GPU node: every pod there needs >= 1 GPU
            cap = min(cap, node.gpu_allocatable)
        caps.append(max(cap, 1))
    counts = _pods_per_node(rng, num_nodes, total_pods, caps)

    # One cluster-level fill target with a little per-node jitter, kept
    # inside the band so the aggregate lands in the band too.
    target = rng.uniform(lo, hi)
    slots: list[_Slot] = []
    for j, (node, k) in enumerate(zip(nodes, counts)):
        fill_cpu = min(hi, max(lo, target + rng.uniform(-0.02, 0.02)))
        fill_mem = min(hi, max(lo, target + rng.uniform(-0.02, 0.02)))
        cpus = _split(rng, node.cpu_allocatable * fill_cpu, k, _CPU_STEP)
        mems = _split(rng, node.memory_allocatable * fill_mem, k, _MEM_STEP)
        gpus = [0] * k
        if node.gpu_allocatable > 0:
            gpu_pods = [
                i for i in range(k) if node.taints or rng.random() < config.gpu_pod_fraction
            ][: node.gpu_allocatable]
            if gpu_pods:
                budget = max(len(gpu_pods), int(round(node.gpu_allocatable * target)))
                budget = min(budget, node.gpu_allocatable)
                for i in gpu_pods:
                    gpus[i] = 1
                for _ in range(budget - len(gpu_pods)):
                    gpus[rng.choice(gpu_pods)] += 1
        for c, m, g in zip(cpus, mems, gpus):
            slots.append(_Slot(j, round(c, 2), m, g))
    return slots


def _assign_apps_and_constraints(
    rng: random.Random, nodes: tuple[NodeSpec, ...], slots: list[_Slot], config: GeneratorConfig
) -> None:
    zone_of = [n.labels[ZONE_KEY] for n in nodes]
    zones_present = sorted(set(zone_of))
    apps = list(config.app_names)
    rng.shuffle(apps)

    # Spread groups: pods of one app, at most one per node (hostname
    # anti-affinity), balanced across zones (maxSkew=1 over every zone that
    # has a node). Zones are picked round-robin to keep them balanced.
    target_spread = sum(rng.random() < config.spread_group_fraction for _ in slots)
    max_groups = min(config.max_spread_groups, len(apps) - 1)
    spread_apps: list[str] = []
    for app in apps[:max_groups]:
        if target_spread < 2:
            break
        size = min(target_spread, rng.randint(2, max(2, len(zones_present) + 1)))
        zone_order = list(zones_present)
        rng.shuffle(zone_order)
        chosen: list[_Slot] = []
        used_nodes: set[int] = set()
        for zi in range(size):
            zone = zone_order[zi % len(zone_order)]
            cands = [
                s for s in slots
                if s.app is None and zone_of[s.node] == zone and s.node not in used_nodes
            ]
            if not cands:
                break
            pick = rng.choice(cands)
            chosen.append(pick)
            used_nodes.add(pick.node)
        if len(chosen) < 2:
            continue
        # A partial round-robin is still balanced (skew <= 1) because each
        # zone was visited at most once more than any other.
        for s in chosen:
            s.app = app
            s.spread = True
        spread_apps.append(app)
        target_spread -= len(chosen)

    free_apps = [a for a in apps if a not in spread_apps] or apps
    for s in slots:
        if s.app is None:
            s.app = rng.choice(free_apps)

    by_node: dict[int, list[_Slot]] = {}
    for s in slots:
        by_node.setdefault(s.node, []).append(s)

    for s in slots:
        if rng.random() < config.zone_pinned_fraction:
            s.zone_pin = zone_of[s.node]
        if rng.random() < config.co_locate_fraction:
            neighbours = [o.app for o in by_node[s.node] if o is not s and o.app != s.app]
            if neighbours:
                s.co_locate_app = rng.choice(neighbours)

    for node_slots in by_node.values():
        free_ports = list(config.host_port_pool)
        rng.shuffle(free_ports)
        for s in node_slots:
            if free_ports and rng.random() < config.host_port_fraction:
                s.host_port = free_ports.pop()


def _slot_to_pod(name: str, s: _Slot) -> PodSpec:
    labels = {"app": s.app}
    tolerations: tuple[Toleration, ...] = ()
    node_selector: dict[str, str] = {}
    if s.gpu > 0:
        node_selector["accelerator"] = "nvidia-a100"
        tolerations = (_GPU_TOLERATION,)

    node_affinity = None
    if s.zone_pin is not None:
        node_affinity = NodeAffinity(
            required=(
                NodeSelectorTerm(
                    match_expressions=(
                        LabelSelectorRequirement(key=ZONE_KEY, operator=SelectorOperator.IN, values=(s.zone_pin,)),
                    )
                ),
            )
        )

    pod_anti_affinity = None
    spread: tuple[TopologySpreadConstraint, ...] = ()
    if s.spread:
        selector = LabelSelector(match_labels={"app": s.app})
        pod_anti_affinity = PodAffinity(required=(PodAffinityTerm(label_selector=selector, topology_key=HOSTNAME_KEY),))
        spread = (
            TopologySpreadConstraint(
                topology_key=ZONE_KEY, max_skew=1, label_selector=selector, when_unsatisfiable="DoNotSchedule"
            ),
        )

    pod_affinity = None
    if s.co_locate_app is not None:
        pod_affinity = PodAffinity(
            required=(
                PodAffinityTerm(label_selector=LabelSelector(match_labels={"app": s.co_locate_app}), topology_key=HOSTNAME_KEY),
            )
        )

    return PodSpec(
        name=name,
        cpu_request=s.cpu,
        memory_request=float(s.mem),
        gpu_request=s.gpu,
        labels=labels,
        tolerations=tolerations,
        node_selector=node_selector,
        node_affinity=node_affinity,
        pod_affinity=pod_affinity,
        pod_anti_affinity=pod_anti_affinity,
        topology_spread_constraints=spread,
        host_ports=(s.host_port,) if s.host_port is not None else (),
    )


def _overflow_slots(
    rng: random.Random, nodes: tuple[NodeSpec, ...], slots: list[_Slot], config: GeneratorConfig, num_extra: int
) -> list[_Slot]:
    """Plain (unconstrained) extra pods that push total CPU and/or memory
    demand to `overflow_band` of capacity. Each one is no larger than the
    biggest node, so the overflow is in the cluster total, not a single
    pod that could never fit anywhere."""
    cap_cpu = sum(n.cpu_allocatable for n in nodes)
    cap_mem = sum(n.memory_allocatable for n in nodes)
    used_cpu = sum(s.cpu for s in slots)
    used_mem = sum(s.mem for s in slots)
    target = rng.uniform(*config.overflow_band)
    # Overflow on CPU, memory, or both.
    mode = rng.choice(("cpu", "mem", "both"))
    max_cpu = max(n.cpu_allocatable for n in nodes)
    max_mem = max(n.memory_allocatable for n in nodes)

    want_cpu = max(cap_cpu * target - used_cpu, 0.0) if mode in ("cpu", "both") else 0.0
    want_mem = max(cap_mem * target - used_mem, 0.0) if mode in ("mem", "both") else 0.0
    # make sure every extra pod fits on the largest node in isolation
    num_extra = max(num_extra, math.ceil(want_cpu / max_cpu), math.ceil(want_mem / max_mem), 1)
    cpus = _split(rng, max(want_cpu, num_extra * 0.25), num_extra, _CPU_STEP)
    mems = _split(rng, max(want_mem, num_extra * 256 * _MIB), num_extra, _MEM_STEP)
    out = []
    for c, m in zip(cpus, mems):
        s = _Slot(UNASSIGNED, round(min(c, max_cpu), 2), min(m, max_mem))
        s.app = rng.choice(config.app_names)
        out.append(s)
    return out


def _in_band(value: float, band: tuple[float, float]) -> bool:
    return band[0] - 1e-9 <= value <= band[1] + 1e-9


def _try_generate(rng: random.Random, config: GeneratorConfig, tier: str) -> GeneratedCluster | None:
    reserve = config.max_overflow_pods if tier == TIER_OVERFLOW else 0
    max_nodes = min(config.max_nodes, config.max_pods - reserve)
    min_nodes = min(config.min_nodes, max_nodes)
    num_nodes = rng.randint(min_nodes, max_nodes)
    nodes = tuple(_generate_node(rng, i, config) for i in range(num_nodes))

    feasible_tier = TIER_VERY_HIGH if tier == TIER_OVERFLOW else tier
    slots = _build_slots(rng, nodes, feasible_tier, config, config.max_pods - reserve)
    if slots is None:
        return None
    _assign_apps_and_constraints(rng, nodes, slots, config)

    if tier == TIER_OVERFLOW:
        room = config.max_pods - len(slots)
        num_extra = rng.randint(1, max(1, min(config.max_overflow_pods, room)))
        extra = _overflow_slots(rng, nodes, slots, config, num_extra)
        if len(slots) + len(extra) > config.max_pods:
            return None
        slots = slots + extra

    rng.shuffle(slots)  # pod order must not leak the witness
    pods = tuple(_slot_to_pod(f"{s.app}-{i}", s) for i, s in enumerate(slots))
    witness = [s.node for s in slots]
    cluster = ClusterBatch(pods=pods, nodes=nodes)

    cpu_u, mem_u = cluster_utilization(cluster)
    if tier == TIER_OVERFLOW:
        if max(cpu_u, mem_u) <= 1.0 + 1e-9:
            return None
    else:
        band = config.high_band if tier == TIER_HIGH else config.very_high_band
        if not (_in_band(cpu_u, band) and _in_band(mem_u, band)):
            return None
    if not validate_assignment(cluster, witness).is_feasible:
        return None
    return GeneratedCluster(cluster=cluster, tier=tier, witness=witness)


def generate_cluster_with_info(
    rng: random.Random, config: GeneratorConfig = GeneratorConfig(), tier: str | None = None, max_attempts: int = 200
) -> GeneratedCluster:
    tier = tier or _sample_tier(rng, config)
    for _ in range(max_attempts):
        result = _try_generate(rng, config, tier)
        if result is not None:
            return result
    raise RuntimeError(f"could not generate a {tier!r} cluster with {config}")


def generate_cluster(rng: random.Random, config: GeneratorConfig = GeneratorConfig()) -> ClusterBatch:
    return generate_cluster_with_info(rng, config).cluster


def _stratified_tiers(rng: random.Random, n: int, weights: tuple[float, ...]) -> list[str]:
    total = sum(weights)
    raw = [n * w / total for w in weights]
    counts = [int(math.floor(r)) for r in raw]
    # largest-remainder rounding so counts sum to n
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)[: n - sum(counts)]:
        counts[i] += 1
    tiers = [t for t, c in zip(TIERS, counts) for _ in range(c)]
    rng.shuffle(tiers)
    return tiers


def generate_dataset_with_info(
    num_clusters: int, seed: int = 0, config: GeneratorConfig = GeneratorConfig()
) -> list[GeneratedCluster]:
    rng = random.Random(seed)
    tiers = _stratified_tiers(rng, num_clusters, config.tier_weights)
    return [generate_cluster_with_info(rng, config, tier=t) for t in tiers]


def generate_dataset(
    num_clusters: int, seed: int = 0, config: GeneratorConfig = GeneratorConfig()
) -> list[ClusterBatch]:
    return [g.cluster for g in generate_dataset_with_info(num_clusters, seed, config)]


def easy_config() -> GeneratorConfig:
    """Smallest, resource-fit-only starting point for curriculum training
    (DESIGN.md sec 8/11a): small clusters, no taints/GPU, no affinity, no
    topology spread, no host ports -- just cpu/memory bin packing. The
    load-tier mix is the same as the full config."""
    return GeneratorConfig(
        min_nodes=3,
        max_nodes=5,
        min_pods=3,
        max_pods=8,
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
    `full` (t=1); everything else (value-choice tuples, app/zone vocab,
    load tiers) stays at `full` throughout."""
    t = max(0.0, min(1.0, t))
    updates = {}
    for name in _CURRICULUM_FIELDS:
        a, b = getattr(easy, name), getattr(full, name)
        value = a + (b - a) * t
        updates[name] = round(value) if name in _CURRICULUM_INT_FIELDS else value
    return replace(full, **updates)
