# JAX-RL Kubernetes-style Scheduler — Design Notes

Status: v1 design, informed by user decisions on 2026-07-10.

## 1. Goal

Learn a policy that maps a batch of pod specs + node specs to a pod→node
assignment, trained via RL in simulation. Feasibility (constraint
satisfaction) is enforced by a validator; the RL objective is to find
*good* feasible assignments, not just *any* feasible one — otherwise this
is a plain CSP/bin-packing problem that doesn't need learning at all.

## 2. Decisions (from user Q&A)

| Question | Decision |
|---|---|
| MDP framing | **One-shot batch assignment.** Single forward pass outputs the full pod→node mapping for a batch, rather than a sequential per-pod loop like real kube-scheduler. |
| Reward objective | **Consolidation / bin-packing density.** Minimize nodes used / maximize packing efficiency, once feasible. |
| Constraint scope | **Full kube-scheduler constraint set**: resources, taints/tolerations, node affinity/selector, pod affinity/anti-affinity, topology spread, ports, volume/PVC binding. |
| End goal | **Simulation/research only for v1.** No real API server or scheduler-extender integration yet; data model kept close enough to real k8s YAML that integration later is plausible. |

These choices compound in one important way (see §5): full constraint
scope includes **inter-pod constraints** (anti-affinity, topology spread,
shared-node resource contention), and one-shot batch decoding means the
network must decide all pods' placements *simultaneously*, without seeing
the outcome of other pods' placements first. That has direct architecture
consequences below.

## 3. Kubernetes scheduler research summary

kube-scheduler (since 1.19) is a plugin pipeline
([scheduling framework docs](https://kubernetes.io/docs/concepts/scheduling-eviction/scheduling-framework/),
[config reference](https://kubernetes.io/docs/reference/scheduling/config/)):

```
PreFilter → Filter (parallel) → PostFilter (preemption)
  → PreScore → Score (parallel) → NormalizeScore
  → Reserve → Permit → PreBind → Bind → PostBind
```

Filter = hard constraints (is this node even legal). Score = soft
preferences (rank the legal nodes). Default plugins relevant to us:

- **NodeResourcesFit** — filter: CPU/memory/extended-resource (e.g.
  `nvidia.com/gpu`) requests must not exceed node allocatable minus
  already-bound pods. Score: `LeastAllocated` (default, spreads load) vs
  `MostAllocated` (packs — this is the family of scoring we're mimicking
  with our consolidation reward) vs `RequestedToCapacityRatio`.
- **TaintToleration** — filter: pod must tolerate all node taints with
  effect `NoSchedule`/`NoExecute`; `PreferNoSchedule` is soft (scoring
  only, not a hard filter).
- **NodeAffinity** — filter: `requiredDuringScheduling` node
  affinity/nodeSelector; score: `preferred` terms.
- **InterPodAffinity** — filter/score: pod affinity/anti-affinity rules
  keyed by label selector + `topologyKey` (e.g. co-locate/spread by
  `kubernetes.io/hostname` or `topology.kubernetes.io/zone`).
- **PodTopologySpread** — filter/score: `maxSkew` per topology domain,
  `whenUnsatisfiable: DoNotSchedule` (hard) vs `ScheduleAnyway` (soft).
- **NodePorts** — filter: `hostPort` collisions.
- **VolumeBinding** — filter: PVC/PV node affinity and zone constraints;
  we model a simplified version (PV pre-bound to a topology domain).
- **NodeUnschedulable** — filter: cordoned nodes excluded.
- **ImageLocality**, **NodeResourcesBalancedAllocation** — soft scoring
  only in real k8s; out of scope, since our reward already encodes a
  packing preference directly.

We are **not** trying to reproduce kube-scheduler's exact scores — we're
reusing its constraint vocabulary (it's a well-tested, realistic
constraint language) and replacing its scoring heuristic with a learned
one optimized for consolidation.

Sources:
[Scheduling Framework](https://kubernetes.io/docs/concepts/scheduling-eviction/scheduling-framework/),
[Scheduler Configuration](https://kubernetes.io/docs/reference/scheduling/config/),
[kube-scheduler concepts](https://kubernetes.io/docs/concepts/scheduling-eviction/kube-scheduler/).

## 4. JAX RL ecosystem research summary

- **[PureJaxRL](https://github.com/luchris429/purejaxrl)** — end-to-end
  JAX PPO/DQN where the *entire* train loop (env step, rollout, gradient
  update) is jitted and vmapped, giving ~4000x wall-clock speedups over
  Python-loop RL by running thousands of environments in parallel on one
  accelerator. This is the pattern we'll follow: our "environment" is a
  pure function `(pods, nodes, rng) -> (assignment, reward)`, jit+vmap
  over a batch of independently sampled clusters.
- **[Gymnax](https://github.com/RobertTLange/gymnax)** — JAX-native gym
  API; we don't need its environments but follow its
  `reset/step/EnvState` idioms loosely.
- **[Brax](https://github.com/google/brax)** — JAX physics for continuous
  control; not applicable (our action space is discrete/combinatorial).
- **RLax / Distrax** — building blocks (masked categorical distributions,
  GAE, PPO losses). We use `distrax.Categorical` with `logits` set to
  `-inf` on masked entries rather than hand-rolling masking.
- **Combinatorial RL literature**: DeepJS (Li & Hu) applies DRL to
  container bin-packing/job scheduling. More directly relevant:
  **SAINT — Attention-Based Policies for Discrete Combinatorial Action
  Spaces** (arXiv 2505.12109) tackles exactly our problem shape —
  multiple correlated discrete sub-actions in one step — using
  self-attention across sub-actions rather than either (a) assuming
  sub-actions are independent or (b) an arbitrary autoregressive
  decomposition. Our network design (§6) follows this: pods
  self-attend to each other *and* cross-attend to nodes in one shared
  encoder, then each pod samples its node choice from a shared
  representation, which lets the policy implicitly coordinate
  (e.g. two mutually anti-affine pods "seeing" each other and
  preferring different nodes) despite the actions being emitted in
  parallel rather than autoregressively.

## 5. The masking problem (why full constraints + one-shot don't fully compose)

For a **single (pod, node) pair**, feasibility is context-free and can be
precomputed before sampling: resource request vs. total node allocatable,
taint/toleration, node selector/affinity, node-level ports. These become
a hard **per-pod action mask** (`mask[i, j] = pod i can legally go on
node j, ignoring what happens to every other pod`).

But several constraints are only checkable once the **whole batch's**
assignment is known:

- Resource contention: many pods might each individually fit on node X,
  but not all *together*.
- Pod affinity/anti-affinity: depends on where the *other* pods in the
  batch land.
- Topology spread `maxSkew`: depends on the full per-domain distribution.
- Host port conflicts between two pods scheduled to the same node.

These cannot be masked ahead of the joint decode — they are enforced
**only through the reward** (a large penalty per violated joint
constraint, computed after the full assignment is sampled), not through
action masking. This is a deliberate, documented limitation of the
one-shot framing the user chose; it is the main reason the encoder needs
pod-to-pod self-attention (so the policy has a chance to learn
coordination) and the main reason a sequential per-pod framing would
have been the "easier" choice architecturally. If, empirically, the
one-shot policy struggles to satisfy joint constraints, the fallback is
to add a light autoregressive decode (pods still share one encoder pass,
but node choices are sampled in sequence with previous choices fed back)
— noted here as a v2 escape hatch, not built now.

## 6. Data model

`PodSpec`: `cpu_request`, `memory_request`, `gpu_request`,
`ephemeral_storage_request`, `tolerations[]` (key, value, effect,
operator), `node_selector` (label map), `node_affinity` (required +
preferred terms), `pod_affinity` / `pod_anti_affinity` (label selector +
topology key, required + preferred), `topology_spread_constraints[]`
(topology key, max skew, when-unsatisfiable), `host_ports[]`, `labels`
(so other pods' affinity rules can match this pod), `priority`.

`NodeSpec`: `cpu_allocatable`, `memory_allocatable`, `gpu_allocatable`,
`ephemeral_storage_allocatable`, `taints[]`, `labels` (includes topology
domains like zone/hostname), `unschedulable`, `used_host_ports[]`,
`existing_pods` (pre-bound load, for realism — not every pod in a
cluster is scheduled in the same batch).

Both are loadable from k8s-style YAML (`yaml_io.py`, subset of the real
Pod/Node schema — enough fields to matter, not a full API model) and
constructible directly in Python for the synthetic generator.

## 7. Network architecture

1. **Tokenize**: each pod and each node becomes a token. Numeric fields
   (cpu/mem/gpu request or allocatable) are normalized and projected;
   categorical/open-vocabulary fields (label keys/values, taint
   keys/values, topology keys) go through a hashing-trick embedding
   (fixed-size table, no closed vocabulary needed) — standard trick for
   arbitrary k8s label strings in a fixed-shape network.
2. **Joint transformer encoder**: pod tokens + node tokens + a
   type embedding (pod vs node), several layers of self-attention over
   the *whole* token set (so pods see nodes, nodes see nodes for
   topology-spread awareness, and pods see other pods for
   affinity/anti-affinity coordination — the SAINT-style pattern from
   §4/§5). Padding tokens (batches have variable pod/node counts, padded
   to `max_pods`/`max_nodes` for static JAX shapes) are excluded via
   attention masks.
3. **Policy head**: per-pod query vector dotted against per-node key
   vectors → `[num_pods, num_nodes]` logit matrix. Apply the
   precomputed pairwise hard-feasibility mask (§5) as `-inf` before
   softmax. Each pod's node choice is sampled as an independent
   categorical *conditioned on the shared joint encoding* — actions are
   emitted in parallel (one-shot), but coordination is pushed entirely
   into the shared representation rather than into the sampling order.
4. **Value head**: mean-pool token embeddings → MLP → scalar baseline,
   used for advantage estimation.

## 8. Environment, reward, training

- **Episode = one batch.** `reset` samples a synthetic cluster (N pods,
  M nodes, N/M randomized per episode up to `max_pods`/`max_nodes`);
  `step` takes the full joint action (one node choice per pod) and
  terminates immediately — this is a contextual-bandit-shaped one-step
  MDP, which is fine: PPO/REINFORCE degenerate gracefully to a
  well-behaved single-step policy-gradient method in this regime, and
  the clipped-ratio machinery still helps when doing multiple gradient
  epochs over one rollout batch.
- **Reward** = `feasibility_penalty + consolidation_score`:
  - `feasibility_penalty`: large negative constant per violated joint
    constraint (resource overcommit, anti-affinity violation, topology
    skew violation, port collision), summed. Dominates the objective so
    the policy always prioritizes legality first.
  - `consolidation_score`: `-active_node_count` plus a small bonus for
    mean packing density on active nodes, only counted when the
    assignment is fully feasible (no reward-shaping toward "efficient
    but broken" placements).
- **Vectorized training** (PureJaxRL-style): `jax.vmap` the whole
  `reset→policy→step→reward` pipeline over thousands of parallel
  synthetic clusters per training iteration, `jax.jit` the update step,
  Adam via `optax`.
- **Curriculum**: start with small clusters (few pods/nodes, fewer
  simultaneous constraint types active) and grow `max_pods`/`max_nodes`
  and constraint density over training — one-shot joint decoding gets
  harder to coordinate as N grows, so this is likely necessary for
  convergence, not just nice-to-have.

## 9. Baselines & evaluation

`baselines.py` implements a deterministic best-fit-decreasing heuristic
that respects all *hard* constraints (used as a floor — any RL policy
that can't beat a simple greedy heuristic isn't worth it) plus a random
best-effort baseline. `eval.py` reports, per baseline and per checkpoint:
feasibility rate (% of episodes with zero constraint violations), mean
active-node count on feasible episodes, and mean packing density.

## 10. Repo layout

```
src/schedgym/
  types.py            # PodSpec/NodeSpec/Taint/Toleration/Affinity/... dataclasses
  yaml_io.py           # load pod/node YAML -> types
  generator.py          # synthetic pod/node batch generator
  constraints.py         # pure-Python constraint oracle (source of truth, used in tests)
  jax_constraints.py      # JAX-vectorized mirror of constraints.py, used inside the jitted env
  encoding.py               # PodSpec/NodeSpec batches -> padded JAX arrays + masks
  network.py                 # Flax transformer policy/value network
  env.py                      # one-shot vmap-able scheduling environment
  train.py                     # PPO/REINFORCE training loop
  baselines.py                  # heuristic baselines
  eval.py                        # comparison harness
tests/
  test_constraints.py
  test_jax_constraints_match_reference.py
  test_generator.py
  test_encoding.py
examples/
  sample_pods.yaml
  sample_nodes.yaml
```

## 11a. Long-run results (2026-07-10, 3000 iterations)

Ran `runs/long_run_20260710/` — `dataset_size=4096`, `batch_size=128`,
`num_iterations=3000`, held-out eval every 200 iterations, ~29.5 minutes
wall clock on CPU (no GPU available in this environment). Full logs in
`runs/long_run_20260710_stdout.log` and `runs/long_run_20260710/history.jsonl`.

**Loss trained fine; task performance did not.** Combined PPO loss fell
from ~3444 to ~150–250 and stayed there. But held-out `success_rate`
(fully scheduled + zero constraint violations) never moved off a
9–13% plateau across the entire run — iteration 0 was 9%, iteration
3000 was 10%, with no visible upward trend in between. A final check
against a completely fresh eval set (seed unseen during training or
in-training eval) confirms this wasn't a fluke:

| policy | success | active nodes (on success) | density (on success) |
|---|---|---|---|
| `best_fit_decreasing` | 18.7% | 3.96 | 0.310 |
| `random_feasible` | 18.3% | 5.22 | 0.171 |
| **trained policy (3000 iters)** | **6.7%** | 2.95 | 0.197 |

The trained policy still loses to both heuristics after a genuinely
long run. This is the risk flagged in §5/§11 materializing, not a new
bug: one-shot joint decoding of the full kube-scheduler constraint set
gives the policy no way to locally re-check its own placements the way
an autoregressive or sequential scheduler could, and 3000 iterations of
plain PPO/REINFORCE with no curriculum apparently isn't enough signal
to learn to coordinate ~15 simultaneous discrete choices under that
handicap. Concretely, likely next steps, roughly in order of expected
impact per unit effort:

1. **Curriculum** (already flagged in §8, not yet built): start training
   on small clusters (4–6 pods) with only resource + taint constraints
   active, and grow toward the full generator config over training.
   Right now the policy is thrown at the hardest version of the problem
   from iteration 0.
2. **Reward shaping**: the current reward is 0/1-feasible-gated (§8) —
   a policy that's "close" to feasible gets the same penalty as one
   that's wildly infeasible, which is a very sparse learning signal for
   ~15-way joint discrete actions. A softer shaping term (e.g.
   proportional to overflow/violation magnitude even when infeasible)
   would likely help far more than more iterations at the current
   reward.
3. **More iterations** — plausible but low-confidence: the plateau
   looks flat rather than slowly-rising, which usually means the
   bottleneck is credit assignment/reward shape rather than optimizer
   time, so brute-force running longer is not the recommended first
   lever.
4. The §5 autoregressive fallback (decode node choices in sequence,
   feeding earlier picks back in) remains the most likely structural
   fix if 1–2 don't close the gap, since it would let action masking
   catch inter-pod conflicts before they happen instead of only
   penalizing them after the fact.

## 11. Known risks / open items

- One-shot + full inter-pod constraints is the hardest combination we
  could have picked (§5) — expect the policy to need the autoregressive
  fallback for anti-affinity/topology-spread-heavy scenarios in
  practice; flagged, not solved, in v1.
- Hashing-trick label embeddings mean semantically different label
  strings could collide; acceptable for v1, revisit if eval shows
  systematic constraint violations traceable to collisions.
- No real cluster integration in v1 (by choice) — the data model is
  intentionally kept close to real k8s fields so this remains possible
  later without a rewrite, but nothing here talks to an API server.
- Reward is unnormalized hand-tuned constants (`feasibility_penalty`
  weight vs. consolidation bonus) — will need tuning once training is
  actually run; not treated as solved by this scaffold.
- Confirmed by an actual (short, 30-iteration) run: the pipeline is
  functionally correct end to end (network trains without NaNs, loss
  moves, metrics are internally consistent) but 30 iterations on a
  512-cluster dataset is nowhere near enough to beat
  `best_fit_decreasing` — feasibility hovered around 10%. This is the
  expected starting point for the hardest constraint/framing
  combination available (§2, §5), not a bug; getting the policy to
  actually win needs the curriculum in §8, reward-weight tuning, and
  orders of magnitude more iterations, none of which this session
  attempted.
- The synthetic generator sometimes produces genuinely unsatisfiable
  pods (e.g. a GPU pod in a cluster instance that happens to have zero
  GPU nodes) — realistic (real clusters have Pending pods), but it caps
  the achievable "fully scheduled" rate below 100% for *any* policy,
  baseline or learned. `eval.py` reports `fully_scheduled_rate` and
  `feasible_rate` separately from combined `success_rate` to keep this
  visible rather than silently penalizing the policy for something no
  assignment could have fixed.
