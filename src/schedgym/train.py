"""PPO training loop for the one-shot scheduling policy (DESIGN.md sec 8).

Episode = one batch assignment (single-step MDP), so PPO here degenerates
to a well-behaved single-step, multi-epoch policy-gradient method:
collect a rollout with the current policy, then run a few clipped-ratio
gradient epochs over that fixed rollout before re-collecting -- the
PureJaxRL pattern of jitting the whole rollout+update cycle and vmapping
across a batch of clusters (DESIGN.md sec 4/8).
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from schedgym.encoding import encode_batch
from schedgym.env import compute_reward, joint_log_prob, sample_actions
from schedgym.generator import GeneratorConfig, easy_config, generate_dataset, interpolate_config
from schedgym.network import SchedulerPolicy


@dataclass(frozen=True)
class TrainConfig:
    max_pods: int = 20
    max_nodes: int = 12
    dataset_size: int = 2048
    batch_size: int = 128
    num_iterations: int = 200
    ppo_epochs: int = 4
    clip_eps: float = 0.2
    learning_rate: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    seed: int = 0
    log_every: int = 10
    # held-out eval + persistence (all disabled by default; opt in for long runs)
    eval_every: int = 0
    eval_clusters: int = 100
    checkpoint_every: int = 0
    run_dir: str | None = None
    # curriculum: 1 = train directly on the full GeneratorConfig (old
    # behavior); >1 ramps difficulty from easy_config() -> full over that
    # many equal-length stages (DESIGN.md sec 8/11a). Held-out eval always
    # uses full difficulty regardless of stage, so numbers stay comparable
    # to a non-curriculum run.
    curriculum_stages: int = 1


def _stage_generator_config(stage_index: int, num_stages: int) -> GeneratorConfig:
    if num_stages <= 1:
        return GeneratorConfig()
    t = stage_index / (num_stages - 1)
    return interpolate_config(easy_config(), GeneratorConfig(), t)


def _describe_stage(gen_config: GeneratorConfig) -> str:
    return (
        f"nodes[{gen_config.min_nodes}-{gen_config.max_nodes}] "
        f"pods[{gen_config.min_pods}-{gen_config.max_pods}] "
        f"gpu_pod={gen_config.gpu_pod_fraction:.2f} "
        f"zone_pinned={gen_config.zone_pinned_fraction:.2f} "
        f"spread={gen_config.spread_group_fraction:.2f} "
        f"co_locate={gen_config.co_locate_fraction:.2f} "
        f"host_port={gen_config.host_port_fraction:.2f}"
    )


def _entropy(logits: jnp.ndarray, pod_mask: jnp.ndarray) -> jnp.ndarray:
    probs = jax.nn.softmax(logits, axis=-1)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    terms = jnp.where(probs > 0, -probs * log_probs, 0.0)
    per_pod = jnp.sum(terms, axis=-1)
    return jnp.sum(jnp.where(pod_mask, per_pod, 0.0)) / jnp.maximum(jnp.sum(pod_mask), 1.0)


def make_train(config: TrainConfig):
    model = SchedulerPolicy()
    tx = optax.adam(config.learning_rate)

    def loss_fn(params, batch, actions, old_log_prob, reward, advantage):
        def single(encoded, action, old_lp, adv):
            logits, value = model.apply(params, encoded)
            log_probs_all = jax.nn.log_softmax(logits, axis=-1)
            log_prob_per_pod = jnp.take_along_axis(log_probs_all, action[:, None], axis=1)[:, 0]
            new_lp = joint_log_prob(log_prob_per_pod, encoded.pod_mask)
            ratio = jnp.exp(new_lp - old_lp)
            clipped_ratio = jnp.clip(ratio, 1 - config.clip_eps, 1 + config.clip_eps)
            policy_term = -jnp.minimum(ratio * adv, clipped_ratio * adv)
            ent = _entropy(logits, encoded.pod_mask)
            return policy_term, value, ent

        policy_terms, values, ents = jax.vmap(single)(batch, actions, old_log_prob, advantage)
        policy_loss = jnp.mean(policy_terms)
        value_loss = jnp.mean((values - reward) ** 2)
        entropy = jnp.mean(ents)
        loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
        return loss, {"policy_loss": policy_loss, "value_loss": value_loss, "entropy": entropy}

    @jax.jit
    def update_epoch(params, opt_state, batch, actions, old_log_prob, reward, advantage):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, batch, actions, old_log_prob, reward, advantage
        )
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    @jax.jit
    def rollout(params, batch, rng):
        def single(encoded, key):
            logits, value = model.apply(params, encoded)
            actions, log_prob_per_pod = sample_actions(logits, key)
            old_log_prob = joint_log_prob(log_prob_per_pod, encoded.pod_mask)
            step = compute_reward(actions, encoded)
            return actions, old_log_prob, value, step

        keys = jax.random.split(rng, batch.pod_mask.shape[0])
        return jax.vmap(single)(batch, keys)

    return model, tx, update_epoch, rollout


def _save_checkpoint(run_dir: Path, iteration: int, params) -> None:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = jax.device_get(params)
    with open(ckpt_dir / f"step_{iteration:06d}.pkl", "wb") as f:
        pickle.dump(payload, f)
    with open(ckpt_dir / "latest.pkl", "wb") as f:
        pickle.dump(payload, f)


def load_checkpoint(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def train(config: TrainConfig = TrainConfig(), verbose: bool = True):
    rng = jax.random.PRNGKey(config.seed)

    num_stages = max(1, config.curriculum_stages)
    iterations_per_stage = max(1, config.num_iterations // num_stages)

    def build_stage_dataset(stage_index: int):
        gen_config = _stage_generator_config(stage_index, num_stages)
        dataset = generate_dataset(config.dataset_size, seed=config.seed + 1000 * stage_index, config=gen_config)
        return gen_config, encode_batch(dataset, config.max_pods, config.max_nodes)

    current_stage = 0
    stage_gen_config, encoded_dataset = build_stage_dataset(current_stage)
    if verbose and num_stages > 1:
        print(f"-- curriculum stage 0/{num_stages - 1}: {_describe_stage(stage_gen_config)} --")

    model, tx, update_epoch, rollout = make_train(config)

    rng, init_key = jax.random.split(rng)
    first_example = jax.tree_util.tree_map(lambda x: x[0], encoded_dataset)
    params = model.init(init_key, first_example)
    opt_state = tx.init(params)

    run_dir = Path(config.run_dir) if config.run_dir else None
    history_file = None
    baseline_metrics = None
    eval_clusters = None
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=2)
        history_file = open(run_dir / "history.jsonl", "a")

    if config.eval_every > 0:
        from schedgym.eval import evaluate_baselines, evaluate_policy

        eval_clusters = generate_dataset(config.eval_clusters, seed=config.seed + 1)
        baseline_metrics = evaluate_baselines(eval_clusters, seed=config.seed + 1)
        if verbose:
            print("-- held-out baseline metrics (fixed, computed once) --")
            for m in baseline_metrics:
                m.print()
            print("-- training --")

    history = []
    start = time.time()
    for iteration in range(config.num_iterations):
        stage_index = min(iteration // iterations_per_stage, num_stages - 1)
        if stage_index != current_stage:
            current_stage = stage_index
            stage_gen_config, encoded_dataset = build_stage_dataset(current_stage)
            if verbose:
                print(f"-- curriculum stage {current_stage}/{num_stages - 1}: {_describe_stage(stage_gen_config)} --")
            if history_file is not None:
                history_file.write(
                    json.dumps({"iteration": iteration, "curriculum_stage": current_stage}) + "\n"
                )
                history_file.flush()

        rng, idx_key, rollout_key = jax.random.split(rng, 3)
        indices = jax.random.randint(idx_key, (config.batch_size,), 0, config.dataset_size)
        batch = jax.tree_util.tree_map(lambda x: x[indices], encoded_dataset)

        actions, old_log_prob, value, step = rollout(params, batch, rollout_key)
        advantage = step.reward - value
        advantage = (advantage - jnp.mean(advantage)) / (jnp.std(advantage) + 1e-8)

        loss, aux = None, None
        for _ in range(config.ppo_epochs):
            params, opt_state, loss, aux = update_epoch(
                params, opt_state, batch, actions, old_log_prob, step.reward, advantage
            )

        record = {
            "iteration": iteration,
            "reward_mean": float(jnp.mean(step.reward)),
            "feasible_frac": float(jnp.mean(step.is_feasible.astype(jnp.float32))),
            "active_nodes_mean": float(jnp.mean(step.active_node_count.astype(jnp.float32))),
            "loss": float(loss),
            **{k: float(v) for k, v in aux.items()},
        }
        history.append(record)
        if history_file is not None:
            history_file.write(json.dumps(record) + "\n")
            history_file.flush()

        is_last = iteration == config.num_iterations - 1
        if verbose and (iteration % config.log_every == 0 or is_last):
            elapsed = time.time() - start
            print(
                f"iter {iteration:4d} | reward {record['reward_mean']:8.3f} | "
                f"feasible {record['feasible_frac']:5.1%} | "
                f"active_nodes {record['active_nodes_mean']:5.2f} | "
                f"loss {record['loss']:8.3f} | {elapsed:6.1f}s"
            )

        if config.eval_every > 0 and eval_clusters is not None and (iteration % config.eval_every == 0 or is_last):
            from schedgym.eval import evaluate_policy

            policy_metrics = evaluate_policy(
                params, model, eval_clusters, config.max_pods, config.max_nodes, name=f"policy@{iteration}"
            )
            if verbose:
                policy_metrics.print()
            if history_file is not None:
                history_file.write(json.dumps({"iteration": iteration, "eval": asdict(policy_metrics)}) + "\n")
                history_file.flush()

        if config.checkpoint_every > 0 and run_dir is not None and (iteration % config.checkpoint_every == 0 or is_last):
            _save_checkpoint(run_dir, iteration, params)

    if history_file is not None:
        history_file.close()

    return params, model, history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the JAX one-shot scheduling policy")
    parser.add_argument("--iterations", type=int, default=TrainConfig.num_iterations)
    parser.add_argument("--dataset-size", type=int, default=TrainConfig.dataset_size)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--log-every", type=int, default=TrainConfig.log_every)
    parser.add_argument("--eval-every", type=int, default=TrainConfig.eval_every)
    parser.add_argument("--eval-clusters", type=int, default=TrainConfig.eval_clusters)
    parser.add_argument("--checkpoint-every", type=int, default=TrainConfig.checkpoint_every)
    parser.add_argument("--run-dir", type=str, default=TrainConfig.run_dir)
    parser.add_argument("--curriculum-stages", type=int, default=TrainConfig.curriculum_stages)
    args = parser.parse_args()

    cfg = TrainConfig(
        num_iterations=args.iterations,
        dataset_size=args.dataset_size,
        batch_size=args.batch_size,
        seed=args.seed,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_clusters=args.eval_clusters,
        checkpoint_every=args.checkpoint_every,
        run_dir=args.run_dir,
        curriculum_stages=args.curriculum_stages,
    )
    train(cfg)
