# schedgym

A JAX/RL Kubernetes-style scheduler research project. See
[DESIGN.md](DESIGN.md) for the full writeup: Kubernetes scheduler
research, JAX RL ecosystem research, and the architecture/design
decisions behind everything here.

In short: a policy network takes a batch of pod specs + node specs and
outputs a full pod->node assignment in one shot. A validator checks the
full Kubernetes-style constraint set (resources, taints/tolerations,
node affinity/selector, pod affinity/anti-affinity, topology spread,
host ports). The RL reward rewards feasible, consolidated (few active
nodes, densely packed) assignments.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Layout

```
src/schedgym/
  types.py            # PodSpec/NodeSpec/... data model
  yaml_io.py           # load pod/node YAML -> types
  generator.py           # synthetic pod/node batch generator
  constraints.py           # pure-Python constraint oracle (source of truth)
  encoding.py                # ClusterBatch -> padded JAX-ready arrays
  jax_constraints.py           # JAX-vectorized mirror of constraints.py
  network.py                     # Flax transformer policy/value network
  env.py                           # one-shot reward function
  train.py                           # PPO training loop
  baselines.py                         # heuristic baselines
  eval.py                                # policy vs. baseline comparison
tests/                                    # pytest suite
examples/                                   # sample k8s-style pod/node YAML
```

## Try it

Load real-ish YAML manifests:

```bash
python -c "
from schedgym.yaml_io import load_cluster
cb = load_cluster('examples/sample_pods.yaml', 'examples/sample_nodes.yaml')
print(cb.pods, cb.nodes)
"
```

Generate synthetic training data and check a trivial assignment:

```bash
python -c "
from schedgym.generator import generate_dataset
from schedgym.constraints import validate_assignment
cluster = generate_dataset(1, seed=0)[0]
result = validate_assignment(cluster, [0] * len(cluster.pods))
print(result.is_feasible, result.violations)
"
```

Run the test suite:

```bash
pytest
```

Train, with periodic held-out eval, checkpointing, and a JSONL history
log (on this CPU-only environment, ~0.35s/iteration at `batch_size=128`
after ~3s JIT warmup -- 3000 iterations takes ~30 minutes):

```bash
python -m schedgym.train \
  --iterations 3000 --dataset-size 4096 --batch-size 128 \
  --eval-every 200 --eval-clusters 100 --checkpoint-every 200 \
  --run-dir runs/my_run
```

Reload a checkpoint and compare against the heuristic baselines on a
fresh eval set:

```python
from schedgym.train import TrainConfig, load_checkpoint
from schedgym.network import SchedulerPolicy
from schedgym.generator import generate_dataset
from schedgym.eval import evaluate_baselines, evaluate_policy

cfg = TrainConfig()
params = load_checkpoint("runs/my_run/checkpoints/latest.pkl")
model = SchedulerPolicy()
clusters = generate_dataset(300, seed=999999)  # any seed unseen during training
for m in evaluate_baselines(clusters, seed=999999):
    m.print()
evaluate_policy(params, model, clusters, cfg.max_pods, cfg.max_nodes).print()
```

## Status

Everything runs end to end (51 passing tests, including a
property-style check that the JAX-vectorized constraint checker agrees
with the Python oracle across random assignments/clusters).

A real long run is also in: `runs/long_run_20260710/` (3000 iterations,
~30 min on CPU, logs + checkpoints + JSONL history included). Result:
loss trains fine (~3444 -> ~150-250), but held-out task success plateaus
at 9-13% throughout the entire run and a final check on a fresh eval
set puts the trained policy at 6.7% success vs. 18.7%
(`best_fit_decreasing`) / 18.3% (`random_feasible`) -- **the policy
does not yet beat either heuristic baseline.** This is the risk
DESIGN.md §5 predicted (one-shot joint decoding of the full constraint
set gives the policy no way to locally re-check its own placements),
not a pipeline bug. See DESIGN.md §11a for the full numbers and a
prioritized list of next steps (curriculum learning first, then reward
shaping, then the autoregressive-decode fallback) -- more raw
iterations alone is not expected to be the fix, since the plateau looks
flat rather than slowly rising.

Note also (DESIGN.md sec 11 addendum): the synthetic generator
sometimes produces genuinely unsatisfiable pods (e.g. a GPU-requiring
pod in a cluster that happens to have zero GPU nodes), which is
realistic (a real cluster can have Pending pods) but means neither
baseline nor policy should be expected to reach 100% "fully scheduled"
-- `eval.py` reports `fully_scheduled_rate` and `feasible_rate`
separately from the combined `success_rate` for this reason.
