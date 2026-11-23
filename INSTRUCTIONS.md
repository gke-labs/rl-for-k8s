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
# optional, NVIDIA GPU (driver must support CUDA 13; keep the jax version
# matching requirements-lock.txt):
pip install "jax[cuda13]==0.10.2"
python -c "import jax; print(jax.devices())"   # expect [CudaDevice(id=0)]
```

On a GPU, training runs ~0.03s/iteration at `batch_size=128` (vs ~0.36s
on CPU), plus ~10s of one-time compilation. Force CPU with
`JAX_PLATFORMS=cpu`.

## Layout

```
src/schedgym/
  types.py            # PodSpec/NodeSpec/... data model
  yaml_io.py          # load pod/node YAML -> types
  generator.py        # load-tiered synthetic clusters built around a feasible witness
  constraints.py      # pure-Python constraint oracle (source of truth)
  encoding.py         # ClusterBatch -> padded JAX-ready arrays
  jax_constraints.py  # JAX-vectorized mirror of constraints.py (+ per-pod violation mask)
  network.py          # Flax transformer policy/value network
  env.py              # one-shot reward (normalized overcommit, partial credit)
  train.py            # BC warm-start, self-imitation, PPO
  baselines.py        # heuristic baselines
  eval.py             # policy (argmax / best-of-K) vs. baselines, per load tier
  schedule.py         # schedule YAML manifests with a trained checkpoint
tests/                # pytest suite
examples/             # sample k8s-style pod/node YAML
```

## Synthetic data

`generate_dataset` builds every cluster around a hidden witness assignment,
so feasibility is known. Tiers are exactly stratified (default 50/40/10):
`high` (schedulable, 80-90% of total CPU and memory requested), `very_high`
(schedulable, 90-100%) and `overflow` (105-130%, unschedulable).
`generate_dataset_with_info` also returns each cluster's tier and witness.

## Recommended training recipe

Imitation warm-start from the generator's witnesses, then self-imitation
(expert iteration) on fresh clusters each round. PPO (`--iterations`) is
supported but currently makes this policy worse (DESIGN.md sec 11j), so it
is off here. About 10 minutes on an RTX 500:

```bash
python -m schedgym.train --iterations 0 \
  --bc-iterations 3000 --bc-dataset-size 32768 \
  --si-rounds 30 --si-samples 64 --si-fresh-each-round \
  --eval-every 500 --eval-clusters 300 --run-dir runs/my_run
# -> runs/my_run/checkpoints/si.pkl
```

Score checkpoints against the baselines, single-shot and best-of-K, per
load tier:

```bash
python -m schedgym.eval runs/my_run/checkpoints/si.pkl --samples 1 16 64
```

Schedule real-ish YAML manifests (best-of-K sample-and-check, with a
`best_fit_decreasing` fallback if no sample is feasible):

```bash
python -m schedgym.schedule examples/sample_pods.yaml examples/sample_nodes.yaml \
  --checkpoint runs/si3_L3_si30_ppo_kl0.5/checkpoints/si.pkl --samples 16
```

Run the test suite:

```bash
pytest
```

## Status

See DESIGN.md sections 11b-11k for the full history. Across 5 runs of the
recommended recipe, on 300 unseen clusters (ceiling 90%; 10% are
deliberately unschedulable), fully-scheduled success is:

| inference | success |
|---|---|
| argmax (single shot) | 21.7% ± 1.7 |
| best-of-16 | 44.8% ± 1.8 |
| best-of-128 | 59.3% ± 1.8 |
| `best_fit_decreasing` | 29.7% |

Best-of-128 costs ~4.3 ms/cluster batched on GPU. The committed checkpoint
`runs/si3_L3_si30_ppo_kl0.5/checkpoints/si.pkl` is one of the five runs
(24.3% argmax / 47.3% best-of-16 / 59.0% best-of-128).
