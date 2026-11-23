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

## 11b. Load-tiered generator + rerun (2026-11-22)

The generator now builds every cluster around a hidden *witness* assignment.
Pod sizes are cut from node capacity, and every constraint is chosen so the
witness satisfies it (checked by the oracle). Tiers are stratified exactly:
50% `high` (schedulable, 80–90% of total CPU *and* memory requested),
40% `very_high` (schedulable, 90–100%), 10% `overflow` (105–130% requested,
provably unschedulable). `generate_dataset_with_info` exposes tier + witness.

Fresh 300-cluster eval (seed 424242), success = fully scheduled + feasible:

| | overall | schedulable only | high | very_high | overflow |
|---|---|---|---|---|---|
| witness (ceiling) | 90.0% | 100% | 100% | 100% | 0% |
| `best_fit_decreasing` | 29.7% | 33.0% | 32.0% | 34.2% | 0% |
| `random_feasible` | 5.3% | 5.9% | 8.0% | 3.3% | 0% |
| policy, `runs/tiered_run_20261122` (3000 it) | 3.7% | 4.1% | 3.3% | 5.0% | 0% |
| policy, `runs/tiered_curriculum_run_20261122` (6 stages) | 0.7% | 0.7% | 0.0% | 1.7% | 0% |

Diagnosis: policy entropy is ~0.07 nats/pod at *initialization* (vs ~1.07
for uniform-over-legal-nodes) and ~0.001 by iteration 500, so PPO stops
exploring almost immediately; best-of-16 sampling equals argmax. Causes:
unnormalized numeric features (up to 128) saturate the logits, and
`resource_overflow` is summed in raw cores+GiB, so rewards are ~-1000 and
value loss is 1e5–1e6, swamping the shared trunk. The policy also uses
~4.5 active nodes where ~7.5 are needed — the consolidation term pushes
toward fewer nodes, which is infeasible at 80%+ load.

## 11c. Normalized inputs + penalties (2026-11-22)

Changes: network numeric resource features are divided by the cluster's
largest node allocatable (per resource) in `encoding.py`, while constraint
arrays stay in raw units. `jax_constraints` adds `resource_overflow_frac`,
overcommit as a fraction of each node's capacity, and the reward uses it.
Weights are now violation=2, overflow=5 per node-capacity, and every
infeasible reward is placed below `-(num_nodes + 1)`, so it can never beat
a feasible assignment.

Mechanically fixed: init entropy 0.07 → ~0.85 (ceiling ~1.07), reward
~-1000 → ~-30, value loss 1e5–1e6 → ~50. Task performance did not improve:

| (same 300-cluster eval as 11b) | argmax | best-of-16 samples |
|---|---|---|
| `normalized_run` step 0 | 1.3% | 5.7% |
| `normalized_run` step 200 | 4.7% | **10.7%** |
| `normalized_run` step 2999 | 0.7% | 0.7% |
| `normalized_curriculum_run` step 400 | 4.0% | 8.3% |
| `normalized_curriculum_run` step 2999 | 0.7% | 0.7% |

The policy improves for ~200–400 iterations, then training reward
*declines* (-28.9 → -36.0) while entropy collapses (0.84 → 0.009). The
single scalar reward over ~14 joint pod choices is too noisy a signal, and
PPO's joint-ratio clipping (one ratio = product over all pods) plus a small
entropy bonus (0.01) let the policy sharpen on noise. Next: per-pod
ratios/clipping, a larger entropy coefficient or lower LR, keeping the best
checkpoint by held-out eval, then the reward work (load-aware consolidation,
partial credit) and imitation warm-start from generator witnesses.

## 11d. Training-stability fixes (2026-11-23, GPU)

Changes in `train.py`:
- **Per-pod PPO ratio (`ratio_mode="per_pod"`, now the default).** Each pod
  decision has its own ratio and clip, all sharing the cluster's advantage.
- **Gradient clipping** at global norm 0.5.
- **Entropy coefficient** 0.01 → 0.05.
- **Learning rate** 3e-4 → 1e-4.
- **Best checkpoint kept:** `checkpoints/best.pkl` + `best.json`, chosen
  by the in-training held-out eval (seed 1).
- New CLI flags: `--learning-rate`, `--entropy-coef`, `--max-grad-norm`,
  `--ratio-mode`.

Sweep (`runs/stab_*`, 3000 it each, 4 in parallel on the RTX 500, ~10 min).
Scored on the separate 300-cluster eval from 11b, so picking the best
checkpoint doesn't inflate the numbers:

| run | entropy start→end | best ckpt | argmax | best-of-16 |
|---|---|---|---|---|
| 11c `normalized_run` (joint ratio, ent .01, lr 3e-4) | 0.84→0.009 | (step 200) | 4.7% | 10.7% |
| A per-pod, ent .05, lr 3e-4 | 0.87→0.73 | 900 | 2.7% | 9.7% |
| B per-pod, ent .05, lr 1e-4 | 0.86→0.76 | 2100 | **5.7%** | **10.7%** |
| C per-pod, ent .10, lr 1e-4 | 0.88→0.87 | 2200 | 2.0% | 10.3% |
| D per-pod, ent .02, lr 3e-4 | 0.82→0.09 | 500 | 3.3% | 8.3% |
| `best_fit_decreasing` | | | 29.7% | |

Training no longer degrades: entropy holds and training reward improves
slightly (≈-29.5 → -26.4) instead of falling. But it plateaus at the same
~10% (best-of-16) that the unstable run reached briefly at step 200. The
remaining bottleneck is the learning signal itself, not optimizer stability:
one scalar per ~14-pod joint action, gated on all-or-nothing feasibility.
Next: partial-credit reward, load-aware consolidation, imitation warm-start
from generator witnesses, and the autoregressive decoder (sec 5).

## 11e. Imitation warm-start from generator witnesses (2026-11-23, GPU)

`train.py` adds a supervised phase before PPO: per-pod cross-entropy
against the witness assignment from `generate_dataset_with_info`. Pods the
witness leaves unassigned (overflow tier) and padding pods are masked out.
Flags: `--bc-iterations`, `--bc-learning-rate` (1e-3),
`--bc-dataset-size` (separate full-difficulty dataset), and `--bc-coef`
(imitation term kept during PPO). The policy after the warm-start is saved
as `checkpoints/bc.pkl`. BC and PPO evals share best-checkpoint tracking.

Runs `runs/bc_E{1,2,3}`: 3000 BC iterations on 32,768 witness clusters,
then 3000 PPO iterations with `bc_coef` 0 / 0.1 / 1.0 (~7 min each,
3 in parallel). Witness accuracy (per pod, train batches) 51% → ~77%.
Same 300-cluster eval as 11b:

| checkpoint | argmax | best-of-16 samples |
|---|---|---|
| best before warm-start (11d run B) | 5.7% | 10.7% |
| **after warm-start only** (`bc_E1/bc.pkl`) | **15.3%** | **35.7%** |
| + PPO 3000 it, bc_coef 0 | 4.3% | 12.7% |
| + PPO 3000 it, bc_coef 0.1 | 7.0% | 19.3% |
| + PPO 3000 it, bc_coef 1.0 | 8.0% | 22.3% |
| `best_fit_decreasing` | 29.7% | — |

The warm-started policy, sampled 16× and checked against the oracle, beats
`best_fit_decreasing` for the first time. BFD also checks each placement
against the oracle, so the comparison is fair in spirit, but it is not a
single-shot argmax result. PPO then *loses* most of the gain within ~100
iterations. Even `bc_coef=1.0` only slows the loss, and every run's
best held-out checkpoint is a BC one. Likely causes: (1) the value head
is untrained after BC (PPO's first loss ≈ 220, mostly value MSE), and
its gradients go through the shared trunk; (2) the 0.05 entropy bonus
pulls the sharp BC policy back toward uniform. Next: warm up the value
head with the policy frozen, lower the entropy coefficient after BC (or
use a KL-to-BC-policy penalty), and evaluate on more than 100 clusters
when choosing checkpoints (the 100-cluster pick, `best.pkl` at bc@2200,
scored below the final BC step on the 300-cluster set).

## 11f. Partial credit (2026-11-23, GPU)

`jax_constraints.pod_violation_mask` flags every pod involved in a
violation. Attribution: the pod on an illegal node; all pods on an
overcommitted node; both pods of an anti-affinity or host-port pair; a pod
with unmet affinity; all pods matched by an over-skewed spread group. A
test checks that some pod is flagged iff the oracle says infeasible. Two
uses, both now **on by default**:
- **Cluster level (`--partial-credit-weight`, default 10):** infeasible
  rewards earn `w * legal_frac`. The infeasible band shifts down by `w`,
  so it still never beats a feasible assignment.
- **Per pod (`--pod-credit-coef`, default 1.0):** each pod's PPO advantage
  is `cluster_adv + c * (pod_ok - legal_frac)`, so within a cluster, legal
  placements are reinforced relative to the ones that caused violations.

Runs `runs/pc_P*` (3000 it, ent 0.05, lr 1e-4, per-pod ratio), 300-cluster
eval from 11b:

| run | train feasible (end) | argmax | best-of-16 |
|---|---|---|---|
| no partial credit (11d run B, final) | 2.4% | 3.7% | 9.3% |
| P1 cluster-level only (w=10) | 3.3% | 4.3% | 13.0% |
| P2 per-pod only (c=1) | 4.3% | 5.3% | 14.0% |
| P3 both | 4.7% | **7.7%** | 13.0% (best.pkl: 14.7%) |
| P4 both + BC warm-start (bc_coef 0.1) | 7.4% | 8.3% | 21.0% |
| 11e BC warm-start only (`bc.pkl`) | — | 15.3% | 35.7% |
| `best_fit_decreasing` | — | 29.7% | — |

From scratch, partial credit roughly doubles single-answer success and adds
~4 points to best-of-16, and the curves are still rising at 3000 it. Per-pod
credit contributes more than the cluster-level term. It does *not* stop PPO
from eroding the BC warm-start (P4 21.0% vs 19.3% for 11e's bc_coef 0.1, and
both are far below `bc.pkl`'s 35.7%), so the 11e diagnosis (untrained value
head + entropy bonus) still stands.

## 11g. Warm-start → PPO hand-off (2026-11-23, GPU)

New `train.py` options, all off by default:
- `--value-warmup-iterations` (+ `--value-warmup-learning-rate`): fit
  only `value_hidden`/`value_out` on rollouts of the frozen policy before
  PPO. All other gradients are zeroed, and a test checks that only those
  params change.
- `--post-bc-entropy-coef`: entropy bonus used for PPO after BC.
- `--kl-to-bc-coef`: per-pod KL(π_bc ‖ π) toward the frozen post-BC policy.

Runs `runs/handoff_H*`: BC 3000 it on 32,768 clusters, value warm-up
300 it (value loss ~700 → ~100), PPO 3000 it with partial credit on.
In-training eval uses 300 clusters. Results on the 300-cluster set from 11b:

| run | argmax | best-of-16 |
|---|---|---|
| BC only (this batch's `bc.pkl`) | 14.0% | 33.7% |
| 11e/11f best after PPO (P4: BC + partial credit, bc_coef .1) | 8.3% | 21.0% |
| H1 value warm-up | 11.3% | 24.0% |
| H2 + post-BC entropy 0.005 | 12.0% | 18.7% |
| H3 + KL 0.5 | 12.0% | 25.0% |
| H4 + KL 2.0 | 11.7% | **28.3%** |
| `best_fit_decreasing` | 29.7% | — |

The hand-off works much better. Training feasibility now holds at ~11–12%
through PPO (it used to fall to ~5% within 100 it), and the KL penalty keeps
most of BC's best-of-16 quality. Lowering entropy alone *hurts* best-of-16:
a sharper policy gives less diverse samples. But PPO still never improves on
BC. Every run's best held-out checkpoint is a BC one, and held-out success
drifts from ~12.5% to 8–11%. So RL is now merely held in place, not adding
value. The likely limit is that one-shot independent per-pod sampling can't
express the coordination that tight 80–100% packing needs. That points to
(a) expert iteration / self-imitation (sample K, keep oracle-feasible
assignments, imitate them; this directly uses what best-of-16 already finds)
and (b) the autoregressive decoder from sec 5.

## 11h. Self-imitation (expert iteration) + GPU precision bug (2026-11-23)

**Bug found and fixed.** On GPU, XLA's default float32 matmul precision is
TF32. The resource-usage `einsum`s in `jax_constraints.py` then carried
errors of ~1/64 GiB on 128 GiB nodes, so the JAX reward rejected **3.9% of
oracle-feasible witnesses** (tight packings). Fix: `precision=HIGHEST` on
every einsum there, plus a capacity-relative overcommit tolerance
(`RESOURCE_REL_TOL = 1e-5`). A regression test checks that every
schedulable witness passes on the default device. GPU runs before this fix
(`stab_*`, `bc_*`, `pc_*`, `handoff_*`, sections 11d–11g) trained with
slightly wrong rewards for near-full nodes. Their *evaluation* numbers come
from the Python oracle and are correct, and the earlier CPU runs were not
affected.

**Self-imitation** (`--si-rounds`, `--si-samples`, `--si-steps-per-round`,
`--si-dataset-size`, `--si-no-seed-witness`; runs after BC, before PPO).
Each round samples K assignments per cluster of a fixed 16,384-cluster set,
keeps each cluster's best oracle-feasible one in a buffer (replaced only by
a higher feasible reward), and trains with the imitation loss.
`si_seed_witness` starts the buffer from the witnesses.

Runs `runs/si_S*` (no PPO; BC = 3000 it on 32,768 clusters), 300-cluster
eval from 11b:

| run | train-set best-of-K, round 0 → last | held-out argmax | held-out best-of-16 |
|---|---|---|---|
| BC only (S1 / S3 `bc.pkl`) | — | 15.3% / 14.0% | 32.3% / 35.7% |
| S1 BC + SI, K=16, witness-seeded, 10 rounds | 32% → 56% | 16.0% | 30.7% |
| S2 BC + SI, K=16, policy-only buffer | 33% → 45% | 17.0% | 25.3% |
| S3 BC + SI, K=64, witness-seeded | 41% → 68% | **18.0%** | 29.7% |
| S4 scratch + SI, K=64, policy-only, 15 rounds | 10% → 24% | 5.7% | 10.3% |
| `best_fit_decreasing` | — | — | 29.7% |

Self-imitation gives the best single-answer policy so far (18.0% argmax).
But the large training-set gains (up to 68% best-of-64) mostly **don't
transfer**: on unseen clusters best-of-16 *drops* by 2–10 points versus BC.
Two effects: (1) the fixed 16k set gets memorized (imitation NLL keeps
falling, and held-out argmax plateaus after ~2 rounds); (2) imitating
single best samples sharpens the policy, so it produces less diverse
samples. The policy-only buffer (S2) shows this most. From scratch (S4),
the policy finds solutions for only ~24% of clusters even at K=64, which
is not enough to bootstrap. Fixes to try: fresh clusters every round (no
memorization; the buffer becomes per-round), keeping several distinct
feasible solutions per cluster instead of one, and an entropy floor or KL
to BC during SI steps.

## 11i. Self-imitation, fixed: fresh clusters, top-M, KL (2026-11-23)

New SI options: `--si-fresh-each-round` (new clusters + a rebuilt
witness-seeded buffer every round, so nothing can be memorized),
`--si-keep-top M` (also imitate up to M-1 extra *distinct* feasible samples
per cluster), and `--si-kl-to-bc-coef` (per-pod KL to the frozen post-BC
policy during SI steps).

Runs `runs/si2_F*`: BC 3000 it, then 10 SI rounds, K=64, 16,384 fresh
clusters per round, witness-seeded. With fresh clusters, each round's
best-of-64 is measured on clusters the policy has never seen: F1 rises
42.5% → 49.6%, still climbing at round 10. Top-4 adds only ~6–8k extra
pairs per round, because distinct feasible samples are scarce.
300-cluster eval from 11b:

| run | argmax | best-of-16 |
|---|---|---|
| BC only (F1 `bc.pkl`) | 16.7% | 33.0% |
| 11h S3 (fixed set, K=64) | 18.0% | 29.7% |
| **F1 fresh** | 18.7% | **42.7%** (47.4% of schedulable) |
| F2 fresh + top-4 | **20.0%** | 38.0% |
| F3 fresh + top-4 + KL 0.5 | 17.0% | 34.7% |
| F4 fresh + top-4 + KL 2.0 | 17.0% | 34.0% |
| `best_fit_decreasing` | 29.7% | — |

Fresh clusters are the fix that matters. F1's best-of-16 beats
`best_fit_decreasing` by 13 points on both tiers (high 48.0% vs 32.0%,
very_high 46.7% vs 34.2%), with about the same node count and density on
successes. The KL penalty is counter-productive here: it anchors the
policy to BC, which SI is meant to improve on. Top-4 trades best-of-16 for
argmax. Next: more rounds (the curve hasn't flattened), and then PPO from
the SI policy with the 11g hand-off settings.

## 11j. 30 SI rounds, and PPO from the SI policy (2026-11-23)

Runs `runs/si3_L*`: BC 3000 it, then 30 fresh-cluster SI rounds
(witness-seeded). L2/L3 then do 300 value warm-up it + 3000 PPO it (partial
credit on) with KL to the post-SI policy (2.0 / 0.5). L4 uses K=128 and no
PPO. Each round's best-of-K on fresh clusters rose 42% → 56% (K=64) and
47% → 60% (K=128) over the 30 rounds, flattening after ~20.
300-cluster eval from 11b:

| checkpoint | argmax | best-of-16 | best-of-16, schedulable only |
|---|---|---|---|
| 11i F1 (10 SI rounds) | 18.7% | 42.7% | 47.4% |
| L2 after 30 SI rounds (`si.pkl`) | 22.0% | 45.3% | 50.4% |
| **L3 after 30 SI rounds (`si.pkl`)** | **24.3%** | **46.3%** | **51.5%** |
| L4 30 rounds, K=128 | 20.3% | 42.7% | 47.4% |
| L2 + PPO 3000 it (KL 2.0) | 17.0% | 37.3% | 41.5% |
| L3 + PPO 3000 it (KL 0.5) | 18.7% | 33.3% | 37.0% |
| `best_fit_decreasing` | 29.7% | — | 33.0% |

L2 and L3 have identical settings up to PPO, so the gap between their
`si.pkl` rows (22.0 vs 24.3 argmax) is run-to-run noise. More SI rounds help
(best-of-16 42.7% → ~46%; argmax 18.7% → 22–24%, now near BFD's 29.7%
single-shot). Doubling K to 128 does not. **PPO still hurts** even with value
warm-up, partial credit, and a KL anchor: −8 to −13 points of best-of-16 over
3000 it. The held-out curve declines steadily rather than collapsing. The best
pipeline is BC → fresh-cluster self-imitation, with best-of-K sampling at
inference. In this one-shot setting, PPO's single per-cluster reward is a
noisier teacher than imitating oracle-verified samples.

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
