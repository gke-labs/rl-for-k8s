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
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from schedgym.encoding import encode_batch
from schedgym.env import compute_reward, joint_log_prob, sample_actions
import numpy as np

from schedgym.generator import (
    GeneratorConfig,
    easy_config,
    generate_dataset,
    generate_dataset_with_info,
    interpolate_config,
)
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
    learning_rate: float = 1e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.05
    max_grad_norm: float = 0.5
    # "per_pod": PPO ratio/clip computed per pod decision (each pod shares
    # the cluster's advantage) -- DESIGN.md sec 11d. "joint": original
    # behavior, one ratio = product over all pods in the cluster.
    ratio_mode: str = "per_pod"
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
    # Imitation warm-start (DESIGN.md sec 11e): supervised pretraining that
    # teaches the policy to reproduce the generator's known-feasible
    # witness assignment (per-pod cross-entropy), run for `bc_iterations`
    # before PPO starts. Uses its own full-difficulty dataset of
    # `bc_dataset_size` clusters (0 -> dataset_size).
    bc_iterations: int = 0
    bc_learning_rate: float = 1e-3
    bc_dataset_size: int = 0
    # Optional imitation term kept *during* PPO (0 = off), so RL can't
    # simply drift away from the warm-started policy.
    bc_coef: float = 0.0
    # Partial credit (DESIGN.md sec 11f). `partial_credit_weight`: infeasible
    # assignments earn w * fraction-of-pods-in-no-violation (cluster-level
    # reward). `pod_credit_coef`: per_pod mode only -- each pod's advantage
    # becomes cluster_adv + c * (pod_ok - cluster legal_frac), i.e. pods that
    # were placed legally are reinforced relative to the cluster's pods
    # that caused violations. Set both to 0 to disable.
    partial_credit_weight: float = 10.0
    pod_credit_coef: float = 1.0
    # Warm-start -> PPO hand-off (DESIGN.md sec 11g).
    # `value_warmup_iterations`: before any PPO policy update, fit only the
    # value head (value_hidden/value_out) on rollouts of the frozen policy,
    # so PPO's first value-loss gradients don't flow through the trunk.
    value_warmup_iterations: int = 0
    value_warmup_learning_rate: float = 1e-3
    # Entropy bonus used for PPO when bc_iterations > 0 (None -> entropy_coef).
    post_bc_entropy_coef: float | None = None
    # Per-pod KL(pi_bc || pi) penalty toward the frozen post-BC policy
    # (keeps BC's whole distribution, not just the single witness).
    kl_to_bc_coef: float = 0.0
    # Self-imitation / expert iteration (DESIGN.md sec 11h), run after BC
    # and before PPO. Each round: sample `si_samples` assignments per cluster
    # of a fixed `si_dataset_size` set, keep each cluster's best
    # oracle-feasible one (highest feasible reward = fewest nodes / densest)
    # in a buffer, then run `si_steps_per_round` imitation steps on the
    # buffer. `si_seed_witness` starts the buffer from the generator's
    # witnesses (otherwise only policy-found solutions are imitated).
    si_rounds: int = 0
    si_samples: int = 16
    si_steps_per_round: int = 300
    si_dataset_size: int = 16384
    si_seed_witness: bool = True
    si_sample_chunk: int = 1024
    # DESIGN.md sec 11i. Fresh clusters every round (no memorization; the
    # buffer is rebuilt per round), up to `si_keep_top` distinct feasible
    # samples imitated per cluster (diversity), and a per-pod
    # KL(pi_bc || pi) penalty toward the frozen post-BC policy during SI steps.
    si_fresh_each_round: bool = False
    si_keep_top: int = 1
    si_kl_to_bc_coef: float = 0.0


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


def _witness_targets(generated, max_pods: int) -> np.ndarray:
    """[B, max_pods] int32 witness node index per pod; -1 for padding pods
    and for overflow pods the witness leaves unassigned."""
    targets = np.full((len(generated), max_pods), -1, dtype=np.int32)
    for i, g in enumerate(generated):
        targets[i, : len(g.witness)] = g.witness
    return targets


def _imitation_loss(logits: jnp.ndarray, targets: jnp.ndarray):
    """Masked per-pod cross-entropy vs. witness targets for one cluster.
    Returns (sum_nll, sum_correct, count) so callers can average over the
    whole batch's real pods."""
    valid = targets >= 0
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    safe = jnp.where(valid, targets, 0)
    nll = -jnp.take_along_axis(log_probs, safe[:, None], axis=1)[:, 0]
    correct = jnp.argmax(logits, axis=-1) == safe
    return (
        jnp.sum(jnp.where(valid, nll, 0.0)),
        jnp.sum(jnp.where(valid, correct, False)),
        jnp.sum(valid),
    )


def _entropy(logits: jnp.ndarray, pod_mask: jnp.ndarray) -> jnp.ndarray:
    probs = jax.nn.softmax(logits, axis=-1)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    terms = jnp.where(probs > 0, -probs * log_probs, 0.0)
    per_pod = jnp.sum(terms, axis=-1)
    return jnp.sum(jnp.where(pod_mask, per_pod, 0.0)) / jnp.maximum(jnp.sum(pod_mask), 1.0)


def make_train(config: TrainConfig):
    model = SchedulerPolicy()
    tx = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.learning_rate))
    if config.ratio_mode not in ("per_pod", "joint"):
        raise ValueError(f"unknown ratio_mode {config.ratio_mode!r}")

    def loss_fn(params, batch, actions, old_log_prob, reward, advantage, pod_advantage, targets, ref_log_probs):
        # old_log_prob / pod_advantage are per pod: [B, P]; advantage: [B];
        # ref_log_probs: frozen BC policy's log-probs [B, P, N] (zeros if unused)
        def single(encoded, action, old_lp, adv, pod_adv, tgt, ref_lp):
            logits, value = model.apply(params, encoded)
            log_probs_all = jax.nn.log_softmax(logits, axis=-1)
            log_prob_per_pod = jnp.take_along_axis(log_probs_all, action[:, None], axis=1)[:, 0]
            if config.ratio_mode == "joint":
                new_lp = joint_log_prob(log_prob_per_pod, encoded.pod_mask)
                ratio = jnp.exp(new_lp - joint_log_prob(old_lp, encoded.pod_mask))
                clipped_ratio = jnp.clip(ratio, 1 - config.clip_eps, 1 + config.clip_eps)
                policy_term = -jnp.minimum(ratio * adv, clipped_ratio * adv)
            else:
                # One clipped surrogate per pod decision, averaged over real
                # pods. With ~14 pods, the joint ratio is a product of 14
                # per-pod ratios, so it leaves the [1-eps, 1+eps] trust
                # region after tiny per-pod changes -- clipping then either
                # zeroes the whole cluster's gradient or (when the min picks
                # the unclipped side) lets it grow without bound.
                ratio = jnp.exp(log_prob_per_pod - old_lp)
                clipped_ratio = jnp.clip(ratio, 1 - config.clip_eps, 1 + config.clip_eps)
                per_pod = -jnp.minimum(ratio * pod_adv, clipped_ratio * pod_adv)
                mask = encoded.pod_mask.astype(jnp.float32)
                policy_term = jnp.sum(per_pod * mask) / jnp.maximum(jnp.sum(mask), 1.0)
            ent = _entropy(logits, encoded.pod_mask)
            nll_sum, _correct, count = _imitation_loss(logits, tgt)
            # KL(ref || new) per pod, averaged over real pods
            ref_p = jnp.exp(ref_lp)
            kl_terms = jnp.where(ref_p > 0, ref_p * (ref_lp - log_probs_all), 0.0)
            kl_per_pod = jnp.sum(kl_terms, axis=-1)
            mask_f = encoded.pod_mask.astype(jnp.float32)
            kl = jnp.sum(kl_per_pod * mask_f) / jnp.maximum(jnp.sum(mask_f), 1.0)
            return policy_term, value, ent, nll_sum, count, kl

        policy_terms, values, ents, nll_sums, counts, kls = jax.vmap(single)(
            batch, actions, old_log_prob, advantage, pod_advantage, targets, ref_log_probs
        )
        policy_loss = jnp.mean(policy_terms)
        value_loss = jnp.mean((values - reward) ** 2)
        entropy = jnp.mean(ents)
        bc_loss = jnp.sum(nll_sums) / jnp.maximum(jnp.sum(counts), 1)
        loss = (
            policy_loss
            + config.value_coef * value_loss
            - config.entropy_coef * entropy
            + config.bc_coef * bc_loss
        )
        if config.kl_to_bc_coef > 0:
            kl_to_bc = jnp.mean(kls)
            loss = loss + config.kl_to_bc_coef * kl_to_bc
        else:
            # ref_log_probs is a zero placeholder when the KL term is off; a
            # KL against it is meaningless (and huge), so don't report one.
            kl_to_bc = jnp.zeros(())
        return loss, {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "bc_loss": bc_loss,
            "kl_to_bc": kl_to_bc,
        }

    @jax.jit
    def update_epoch(
        params, opt_state, batch, actions, old_log_prob, reward, advantage, pod_advantage, targets, ref_log_probs
    ):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, batch, actions, old_log_prob, reward, advantage, pod_advantage, targets, ref_log_probs
        )
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    bc_tx = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.bc_learning_rate))

    def bc_loss_fn(params, batch, targets):
        def single(encoded, tgt):
            logits, _value = model.apply(params, encoded)
            return _imitation_loss(logits, tgt)

        nll_sums, corrects, counts = jax.vmap(single)(batch, targets)
        n = jnp.maximum(jnp.sum(counts), 1)
        return jnp.sum(nll_sums) / n, jnp.sum(corrects) / n

    @jax.jit
    def bc_step(params, opt_state, batch, targets):
        (loss, acc), grads = jax.value_and_grad(bc_loss_fn, has_aux=True)(params, batch, targets)
        updates, opt_state = bc_tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, acc

    @jax.jit
    def rollout(params, ref_params, batch, rng):
        """Sample + score with the current policy. Also returns the frozen
        reference (post-BC) policy's log-probs when the KL penalty is on,
        else zeros (static branch, so no extra forward pass)."""

        def single(encoded, key):
            logits, value = model.apply(params, encoded)
            actions, log_prob_per_pod = sample_actions(logits, key)
            step = compute_reward(actions, encoded, config.partial_credit_weight)
            if config.kl_to_bc_coef > 0:
                ref_logits, _ = model.apply(ref_params, encoded)
                ref_lp = jax.nn.log_softmax(ref_logits, axis=-1)
            else:
                ref_lp = jnp.zeros_like(logits)
            return actions, log_prob_per_pod, value, step, ref_lp

        keys = jax.random.split(rng, batch.pod_mask.shape[0])
        return jax.vmap(single)(batch, keys)

    value_tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.value_warmup_learning_rate)
    )

    def _value_only(grads):
        """Zero every gradient except the value head's own layers."""

        def keep(path, g):
            names = {getattr(k, "key", None) for k in path}
            return g if names & {"value_hidden", "value_out"} else jnp.zeros_like(g)

        return jax.tree_util.tree_map_with_path(keep, grads)

    @jax.jit
    def value_step(params, opt_state, batch, reward):
        def vloss(p):
            values = jax.vmap(lambda e: model.apply(p, e)[1])(batch)
            return jnp.mean((values - reward) ** 2)

        loss, grads = jax.value_and_grad(vloss)(params)
        updates, opt_state = value_tx.update(_value_only(grads), opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def sample_k(params, batch, rng):
        """Draw `si_samples` assignments per cluster. Returns (acts [K, B, P],
        feasible reward [K, B], -inf where infeasible)."""
        logits = jax.vmap(lambda e: model.apply(params, e)[0])(batch)  # [B, P, N]
        keys = jax.random.split(rng, config.si_samples)
        acts = jax.vmap(lambda k: jax.random.categorical(k, logits, axis=-1))(keys)  # [K, B, P]
        steps = jax.vmap(lambda a: jax.vmap(compute_reward)(a, batch))(acts)
        return acts, jnp.where(steps.is_feasible, steps.reward, -jnp.inf)

    def si_loss_fn(params, ref_params, batch, targets):
        def single(encoded, tgt):
            logits, _value = model.apply(params, encoded)
            nll_sum, correct, count = _imitation_loss(logits, tgt)
            if config.si_kl_to_bc_coef > 0:
                ref_lp = jax.nn.log_softmax(model.apply(ref_params, encoded)[0], axis=-1)
                ref_p = jnp.exp(ref_lp)
                lp = jax.nn.log_softmax(logits, axis=-1)
                kl_pod = jnp.sum(jnp.where(ref_p > 0, ref_p * (ref_lp - lp), 0.0), axis=-1)
                mask = encoded.pod_mask.astype(jnp.float32)
                kl = jnp.sum(kl_pod * mask) / jnp.maximum(jnp.sum(mask), 1.0)
            else:
                kl = jnp.zeros(())
            return nll_sum, correct, count, kl

        nll_sums, corrects, counts, kls = jax.vmap(single)(batch, targets)
        n = jnp.maximum(jnp.sum(counts), 1)
        nll = jnp.sum(nll_sums) / n
        kl = jnp.mean(kls)
        return nll + config.si_kl_to_bc_coef * kl, (nll, jnp.sum(corrects) / n, kl)

    @jax.jit
    def si_step(params, opt_state, ref_params, batch, targets):
        (_loss, (nll, acc, kl)), grads = jax.value_and_grad(si_loss_fn, has_aux=True)(
            params, ref_params, batch, targets
        )
        updates, opt_state = bc_tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, nll, acc, kl

    @jax.jit
    def score_assignments(batch, assignments):
        """Feasible reward of fixed assignments (-inf if infeasible)."""
        steps = jax.vmap(compute_reward)(assignments, batch)
        return jnp.where(steps.is_feasible, steps.reward, -jnp.inf)

    return (
        model, tx, update_epoch, rollout, bc_tx, bc_step, value_tx, value_step, sample_k, si_step, score_assignments
    )


def _save_checkpoint(run_dir: Path, iteration: int, params) -> None:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = jax.device_get(params)
    with open(ckpt_dir / f"step_{iteration:06d}.pkl", "wb") as f:
        pickle.dump(payload, f)
    with open(ckpt_dir / "latest.pkl", "wb") as f:
        pickle.dump(payload, f)


def _save_best(run_dir: Path, iteration: int, params, metrics) -> None:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "best.pkl", "wb") as f:
        pickle.dump(jax.device_get(params), f)
    with open(ckpt_dir / "best.json", "w") as f:
        json.dump({"iteration": iteration, "eval": asdict(metrics)}, f, indent=2)


def load_checkpoint(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def train(config: TrainConfig = TrainConfig(), verbose: bool = True):
    rng = jax.random.PRNGKey(config.seed)

    num_stages = max(1, config.curriculum_stages)
    iterations_per_stage = max(1, config.num_iterations // num_stages)

    def build_stage_dataset(stage_index: int):
        gen_config = _stage_generator_config(stage_index, num_stages)
        generated = generate_dataset_with_info(
            config.dataset_size, seed=config.seed + 1000 * stage_index, config=gen_config
        )
        encoded = encode_batch([g.cluster for g in generated], config.max_pods, config.max_nodes)
        return gen_config, encoded, jnp.asarray(_witness_targets(generated, config.max_pods))

    current_stage = 0
    stage_gen_config, encoded_dataset, dataset_targets = build_stage_dataset(current_stage)
    if verbose and num_stages > 1:
        print(f"-- curriculum stage 0/{num_stages - 1}: {_describe_stage(stage_gen_config)} --")

    run_config = config  # what config.json records (exactly what was asked for)
    if config.bc_iterations > 0 and config.post_bc_entropy_coef is not None:
        config = replace(config, entropy_coef=config.post_bc_entropy_coef)
    (
        model, tx, update_epoch, rollout, bc_tx, bc_step, value_tx, value_step, sample_k, si_step, score_assignments
    ) = make_train(config)

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
            json.dump(asdict(run_config), f, indent=2)
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
    best_score = None
    start = time.time()

    def run_eval(label: str, iteration_key):
        """Held-out eval + best-checkpoint tracking, shared by BC and PPO."""
        nonlocal best_score
        from schedgym.eval import evaluate_policy

        metrics = evaluate_policy(params, model, eval_clusters, config.max_pods, config.max_nodes, name=label)
        if verbose:
            metrics.print()
        # Keep the best held-out checkpoint: the last one isn't
        # necessarily the best one (DESIGN.md sec 11c).
        score = (metrics.success_rate, metrics.feasible_rate)
        if run_dir is not None and (best_score is None or score > best_score):
            best_score = score
            _save_best(run_dir, iteration_key, params, metrics)
        if history_file is not None:
            history_file.write(json.dumps({"iteration": iteration_key, "eval": asdict(metrics)}) + "\n")
            history_file.flush()
        return metrics

    if config.bc_iterations > 0:
        bc_size = config.bc_dataset_size or config.dataset_size
        bc_generated = generate_dataset_with_info(bc_size, seed=config.seed + 777_777)
        bc_encoded = encode_batch([g.cluster for g in bc_generated], config.max_pods, config.max_nodes)
        bc_targets = jnp.asarray(_witness_targets(bc_generated, config.max_pods))
        bc_opt_state = bc_tx.init(params)
        if verbose:
            print(f"-- imitation warm-start: {config.bc_iterations} iterations on {bc_size} witness clusters --")
        for bc_it in range(config.bc_iterations):
            rng, idx_key = jax.random.split(rng)
            idx = jax.random.randint(idx_key, (config.batch_size,), 0, bc_size)
            batch = jax.tree_util.tree_map(lambda x: x[idx], bc_encoded)
            params, bc_opt_state, bc_loss, bc_acc = bc_step(params, bc_opt_state, batch, bc_targets[idx])
            is_last_bc = bc_it == config.bc_iterations - 1
            record = {"phase": "bc", "bc_iteration": bc_it, "bc_loss": float(bc_loss), "bc_acc": float(bc_acc)}
            if history_file is not None:
                history_file.write(json.dumps(record) + "\n")
                history_file.flush()
            if verbose and (bc_it % config.log_every == 0 or is_last_bc):
                print(
                    f"bc {bc_it:5d} | nll {record['bc_loss']:6.3f} | witness-acc {record['bc_acc']:6.1%} | "
                    f"{time.time() - start:6.1f}s"
                )
            if eval_clusters is not None and (bc_it % config.eval_every == 0 or is_last_bc):
                run_eval(f"bc@{bc_it}", f"bc_{bc_it}")
        if run_dir is not None:
            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            with open(ckpt_dir / "bc.pkl", "wb") as f:
                pickle.dump(jax.device_get(params), f)
    if config.si_rounds > 0:
        n_si = config.si_dataset_size
        chunk = min(config.si_sample_chunk, n_si)
        si_ref_params = params  # frozen post-BC policy for the SI KL term
        si_opt_state = bc_tx.init(params)

        def build_si_set(round_idx: int):
            """Clusters + a buffer of (best known target, its score) per
            cluster, seeded from witnesses if requested. Schedulable
            witnesses get their real feasible reward; overflow witnesses
            leave pods UNASSIGNED and stay partial targets (masked) at -inf,
            which no sample can beat since no full overflow assignment is
            feasible."""
            seed = config.seed + 555_555 + (7919 * round_idx if config.si_fresh_each_round else 0)
            generated = generate_dataset_with_info(n_si, seed=seed)
            encoded = encode_batch([g.cluster for g in generated], config.max_pods, config.max_nodes)
            targets = np.full((n_si, config.max_pods), -1, dtype=np.int32)
            score = np.full(n_si, -np.inf, dtype=np.float32)
            if config.si_seed_witness:
                targets[:] = _witness_targets(generated, config.max_pods)
                full = np.where(targets >= 0, targets, 0)
                for lo in range(0, n_si, chunk):
                    hi = min(lo + chunk, n_si)
                    enc = jax.tree_util.tree_map(lambda x: x[lo:hi], encoded)
                    sc = np.asarray(score_assignments(enc, jnp.asarray(full[lo:hi])))
                    has_unassigned = np.array([any(a < 0 for a in g.witness) for g in generated[lo:hi]])
                    score[lo:hi] = np.where(has_unassigned, -np.inf, sc)
            return encoded, targets, score, np.zeros(n_si, dtype=bool)

        si_encoded = buf_targets = buf_score = buf_from_policy = None
        if verbose:
            print(
                f"-- self-imitation: {config.si_rounds} rounds x {config.si_samples} samples on {n_si} clusters "
                f"(seed witness {config.si_seed_witness}, fresh each round {config.si_fresh_each_round}, "
                f"keep top {config.si_keep_top}, KL-to-BC {config.si_kl_to_bc_coef}) --"
            )
        for si_round in range(config.si_rounds):
            if si_round == 0 or config.si_fresh_each_round:
                si_encoded, buf_targets, buf_score, buf_from_policy = build_si_set(si_round)
            pod_mask_np = np.asarray(si_encoded.pod_mask)
            extra_idx: list[int] = []
            extra_targets: list[np.ndarray] = []
            found = improved = 0
            for lo in range(0, n_si, chunk):
                hi = min(lo + chunk, n_si)
                enc = jax.tree_util.tree_map(lambda x: x[lo:hi], si_encoded)
                rng, key = jax.random.split(rng)
                acts, score = sample_k(params, enc, key)
                acts = np.where(pod_mask_np[lo:hi][None], np.asarray(acts), -1)  # padding pods masked
                score = np.asarray(score)  # [K, b]
                for b in range(hi - lo):
                    order = np.argsort(-score[:, b])
                    feas = [k for k in order if np.isfinite(score[k, b])]
                    if not feas:
                        continue
                    found += 1
                    c = lo + b
                    best = feas[0]
                    if score[best, b] > buf_score[c]:
                        improved += 1
                        buf_targets[c] = acts[best, b]
                        buf_score[c] = score[best, b]
                        buf_from_policy[c] = True
                    if config.si_keep_top > 1:
                        # up to keep_top-1 more *distinct* feasible samples,
                        # best first, never duplicating the buffer entry
                        seen = {buf_targets[c].tobytes()}
                        for k in feas:
                            row = acts[k, b]
                            key_bytes = row.tobytes()
                            if key_bytes in seen:
                                continue
                            seen.add(key_bytes)
                            extra_idx.append(c)
                            extra_targets.append(row)
                            if len(seen) >= config.si_keep_top:
                                break

            usable = np.nonzero((buf_targets >= 0).any(axis=1))[0]
            pair_cluster = np.concatenate([usable, np.asarray(extra_idx, dtype=np.int64)])
            pair_targets = np.concatenate(
                [buf_targets[usable], np.asarray(extra_targets, dtype=np.int32).reshape(-1, config.max_pods)]
            )
            nll_sum = acc_sum = kl_sum = 0.0
            steps = config.si_steps_per_round if len(pair_cluster) else 0
            if steps:
                pair_targets_dev = jnp.asarray(pair_targets)
                for _ in range(steps):
                    rng, idx_key = jax.random.split(rng)
                    pick = np.asarray(jax.random.randint(idx_key, (config.batch_size,), 0, len(pair_cluster)))
                    batch = jax.tree_util.tree_map(lambda x: x[pair_cluster[pick]], si_encoded)
                    params, si_opt_state, nll, acc, kl = si_step(
                        params, si_opt_state, si_ref_params, batch, pair_targets_dev[pick]
                    )
                    nll_sum += float(nll)
                    acc_sum += float(acc)
                    kl_sum += float(kl)
            steps = max(steps, 1)
            finite = np.isfinite(buf_score)
            record = {
                "phase": "self_imitation",
                "si_round": si_round,
                "sample_feasible_frac": found / n_si,  # clusters where >=1 of K samples was feasible
                "improved_frac": improved / n_si,
                "buffer_feasible_frac": float(finite.mean()),
                "buffer_from_policy_frac": float(buf_from_policy.mean()),
                "buffer_mean_score": float(buf_score[finite].mean()) if finite.any() else float("nan"),
                "train_pairs": int(len(pair_cluster)),
                "si_loss": nll_sum / steps,
                "si_acc": acc_sum / steps,
                "si_kl_to_bc": kl_sum / steps,
            }
            if history_file is not None:
                history_file.write(json.dumps(record) + "\n")
                history_file.flush()
            if verbose:
                print(
                    f"si {si_round:3d} | best-of-{config.si_samples} feasible {record['sample_feasible_frac']:6.1%} | "
                    f"improved {record['improved_frac']:6.1%} | buffer feasible {record['buffer_feasible_frac']:6.1%} "
                    f"(policy-found {record['buffer_from_policy_frac']:6.1%}) | pairs {record['train_pairs']:6d} | "
                    f"nll {record['si_loss']:.3f} | kl {record['si_kl_to_bc']:.3f} | {time.time() - start:6.1f}s"
                )
            if eval_clusters is not None:
                run_eval(f"si@{si_round}", f"si_{si_round}")
        if run_dir is not None:
            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            with open(ckpt_dir / "si.pkl", "wb") as f:
                pickle.dump(jax.device_get(params), f)

    ref_params = params  # frozen post-BC (and post-self-imitation) policy for the KL penalty

    if config.value_warmup_iterations > 0:
        value_opt_state = value_tx.init(params)
        if verbose:
            print(f"-- value-head warm-up: {config.value_warmup_iterations} iterations (policy frozen) --")
        for v_it in range(config.value_warmup_iterations):
            rng, idx_key, rollout_key = jax.random.split(rng, 3)
            indices = jax.random.randint(idx_key, (config.batch_size,), 0, config.dataset_size)
            batch = jax.tree_util.tree_map(lambda x: x[indices], encoded_dataset)
            _a, _lp, _v, step, _ref = rollout(params, ref_params, batch, rollout_key)
            params, value_opt_state, v_loss = value_step(params, value_opt_state, batch, step.reward)
            record = {"phase": "value_warmup", "value_iteration": v_it, "value_loss": float(v_loss)}
            if history_file is not None:
                history_file.write(json.dumps(record) + "\n")
                history_file.flush()
            if verbose and (v_it % config.log_every == 0 or v_it == config.value_warmup_iterations - 1):
                print(f"value {v_it:5d} | value_loss {record['value_loss']:8.3f} | {time.time() - start:6.1f}s")

    if verbose and (config.bc_iterations > 0 or config.value_warmup_iterations > 0):
        print("-- PPO --")
    for iteration in range(config.num_iterations):
        stage_index = min(iteration // iterations_per_stage, num_stages - 1)
        if stage_index != current_stage:
            current_stage = stage_index
            stage_gen_config, encoded_dataset, dataset_targets = build_stage_dataset(current_stage)
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
        batch_targets = dataset_targets[indices]

        actions, old_log_prob, value, step, ref_log_probs = rollout(params, ref_params, batch, rollout_key)
        advantage = step.reward - value
        advantage = (advantage - jnp.mean(advantage)) / (jnp.std(advantage) + 1e-8)
        pod_advantage = advantage[:, None] + config.pod_credit_coef * (step.pod_ok - step.legal_frac[:, None])

        loss, aux = None, None
        for _ in range(config.ppo_epochs):
            params, opt_state, loss, aux = update_epoch(
                params,
                opt_state,
                batch,
                actions,
                old_log_prob,
                step.reward,
                advantage,
                pod_advantage,
                batch_targets,
                ref_log_probs,
            )

        record = {
            "iteration": iteration,
            "reward_mean": float(jnp.mean(step.reward)),
            "feasible_frac": float(jnp.mean(step.is_feasible.astype(jnp.float32))),
            "active_nodes_mean": float(jnp.mean(step.active_node_count.astype(jnp.float32))),
            "legal_frac_mean": float(jnp.mean(step.legal_frac)),
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
                f"legal_pods {record['legal_frac_mean']:5.1%} | "
                f"loss {record['loss']:8.3f} | {elapsed:6.1f}s"
            )

        if config.eval_every > 0 and eval_clusters is not None and (iteration % config.eval_every == 0 or is_last):
            run_eval(f"policy@{iteration}", iteration)

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
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--entropy-coef", type=float, default=TrainConfig.entropy_coef)
    parser.add_argument("--max-grad-norm", type=float, default=TrainConfig.max_grad_norm)
    parser.add_argument("--ratio-mode", choices=("per_pod", "joint"), default=TrainConfig.ratio_mode)
    parser.add_argument("--bc-iterations", type=int, default=TrainConfig.bc_iterations)
    parser.add_argument("--bc-learning-rate", type=float, default=TrainConfig.bc_learning_rate)
    parser.add_argument("--bc-dataset-size", type=int, default=TrainConfig.bc_dataset_size)
    parser.add_argument("--bc-coef", type=float, default=TrainConfig.bc_coef)
    parser.add_argument("--partial-credit-weight", type=float, default=TrainConfig.partial_credit_weight)
    parser.add_argument("--pod-credit-coef", type=float, default=TrainConfig.pod_credit_coef)
    parser.add_argument("--value-warmup-iterations", type=int, default=TrainConfig.value_warmup_iterations)
    parser.add_argument("--value-warmup-learning-rate", type=float, default=TrainConfig.value_warmup_learning_rate)
    parser.add_argument("--post-bc-entropy-coef", type=float, default=TrainConfig.post_bc_entropy_coef)
    parser.add_argument("--kl-to-bc-coef", type=float, default=TrainConfig.kl_to_bc_coef)
    parser.add_argument("--si-rounds", type=int, default=TrainConfig.si_rounds)
    parser.add_argument("--si-samples", type=int, default=TrainConfig.si_samples)
    parser.add_argument("--si-steps-per-round", type=int, default=TrainConfig.si_steps_per_round)
    parser.add_argument("--si-dataset-size", type=int, default=TrainConfig.si_dataset_size)
    parser.add_argument("--si-no-seed-witness", action="store_true")
    parser.add_argument("--si-fresh-each-round", action="store_true")
    parser.add_argument("--si-keep-top", type=int, default=TrainConfig.si_keep_top)
    parser.add_argument("--si-kl-to-bc-coef", type=float, default=TrainConfig.si_kl_to_bc_coef)
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
        learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        ratio_mode=args.ratio_mode,
        bc_iterations=args.bc_iterations,
        bc_learning_rate=args.bc_learning_rate,
        bc_dataset_size=args.bc_dataset_size,
        bc_coef=args.bc_coef,
        partial_credit_weight=args.partial_credit_weight,
        pod_credit_coef=args.pod_credit_coef,
        value_warmup_iterations=args.value_warmup_iterations,
        value_warmup_learning_rate=args.value_warmup_learning_rate,
        post_bc_entropy_coef=args.post_bc_entropy_coef,
        kl_to_bc_coef=args.kl_to_bc_coef,
        si_rounds=args.si_rounds,
        si_samples=args.si_samples,
        si_steps_per_round=args.si_steps_per_round,
        si_dataset_size=args.si_dataset_size,
        si_seed_witness=not args.si_no_seed_witness,
        si_fresh_each_round=args.si_fresh_each_round,
        si_keep_top=args.si_keep_top,
        si_kl_to_bc_coef=args.si_kl_to_bc_coef,
    )
    train(cfg)
