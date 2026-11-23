import math

import pytest

from schedgym.train import TrainConfig, train


@pytest.mark.parametrize("ratio_mode", ["per_pod", "joint"])
def test_short_training_run_is_finite_and_saves_best(tmp_path, ratio_mode):
    cfg = TrainConfig(
        num_iterations=4,
        dataset_size=32,
        batch_size=8,
        eval_every=2,
        eval_clusters=8,
        run_dir=str(tmp_path),
        ratio_mode=ratio_mode,
    )
    _params, _model, history = train(cfg, verbose=False)
    assert len(history) == 4
    assert all(math.isfinite(r["loss"]) for r in history)
    assert (tmp_path / "checkpoints" / "best.pkl").exists()
    assert (tmp_path / "checkpoints" / "best.json").exists()


def test_unknown_ratio_mode_rejected():
    with pytest.raises(ValueError):
        train(TrainConfig(num_iterations=1, dataset_size=8, batch_size=4, ratio_mode="nope"), verbose=False)


def test_imitation_warm_start_learns_witness_and_saves_bc_checkpoint(tmp_path):
    import json

    cfg = TrainConfig(
        num_iterations=2,
        dataset_size=32,
        batch_size=16,
        bc_iterations=60,
        bc_dataset_size=16,  # tiny, so it should overfit quickly
        bc_coef=0.5,
        eval_every=30,
        eval_clusters=8,
        run_dir=str(tmp_path),
    )
    _params, _model, history = train(cfg, verbose=False)
    bc = [json.loads(l) for l in open(tmp_path / "history.jsonl") if '"phase": "bc"' in l]
    assert len(bc) == 60
    assert bc[-1]["bc_loss"] < bc[0]["bc_loss"]
    assert bc[-1]["bc_acc"] > bc[0]["bc_acc"]
    assert (tmp_path / "checkpoints" / "bc.pkl").exists()
    assert all("bc_loss" in r for r in history)  # PPO logs the auxiliary term too


def test_partial_credit_training_runs():
    cfg = TrainConfig(num_iterations=3, dataset_size=32, batch_size=8, partial_credit_weight=10.0, pod_credit_coef=1.0)
    _params, _model, history = train(cfg, verbose=False)
    assert all(math.isfinite(r["loss"]) for r in history)
    assert all(0.0 <= r["legal_frac_mean"] <= 1.0 for r in history)


def test_value_warmup_trains_only_the_value_head(tmp_path):
    import json
    import pickle

    import jax
    import numpy as np

    cfg = TrainConfig(
        num_iterations=0, dataset_size=32, batch_size=16, bc_iterations=5, value_warmup_iterations=40,
        run_dir=str(tmp_path),
    )
    params, _model, _history = train(cfg, verbose=False)
    with open(tmp_path / "checkpoints" / "bc.pkl", "rb") as f:
        bc_params = pickle.load(f)
    for name, sub in params["params"].items():
        same = all(
            np.array_equal(a, b)
            for a, b in zip(jax.tree_util.tree_leaves(sub), jax.tree_util.tree_leaves(bc_params["params"][name]))
        )
        assert same == (name not in ("value_hidden", "value_out")), name
    vl = [json.loads(l)["value_loss"] for l in open(tmp_path / "history.jsonl") if "value_warmup" in l]
    assert len(vl) == 40
    assert np.mean(vl[-10:]) < np.mean(vl[:10])


def test_kl_to_bc_starts_at_zero_and_post_bc_entropy_is_recorded(tmp_path):
    import json

    cfg = TrainConfig(
        num_iterations=6, dataset_size=32, batch_size=16, bc_iterations=5, kl_to_bc_coef=1.0,
        post_bc_entropy_coef=0.0, learning_rate=1e-3, ppo_epochs=1, run_dir=str(tmp_path),
    )
    _params, _model, history = train(cfg, verbose=False)
    # with one epoch, iteration 0's loss is computed before any PPO update,
    # when the policy still equals the frozen BC reference
    assert abs(history[0]["kl_to_bc"]) < 1e-5
    assert history[-1]["kl_to_bc"] > history[0]["kl_to_bc"]
    saved = json.load(open(tmp_path / "config.json"))
    assert saved["entropy_coef"] == TrainConfig().entropy_coef  # config.json records what was asked
    assert saved["post_bc_entropy_coef"] == 0.0


def test_kl_metric_is_zero_when_penalty_off():
    cfg = TrainConfig(num_iterations=2, dataset_size=16, batch_size=8)
    _params, _model, history = train(cfg, verbose=False)
    assert all(r["kl_to_bc"] == 0.0 for r in history)


def test_self_imitation_buffer_only_holds_feasible_policy_solutions(tmp_path):
    import json

    cfg = TrainConfig(
        num_iterations=0, batch_size=16, dataset_size=16, bc_iterations=30, bc_dataset_size=64,
        si_rounds=2, si_samples=8, si_steps_per_round=5, si_dataset_size=64, si_sample_chunk=32,
        si_seed_witness=False, run_dir=str(tmp_path),
    )
    train(cfg, verbose=False)
    rounds = [json.loads(l) for l in open(tmp_path / "history.jsonl") if "self_imitation" in l]
    assert len(rounds) == 2
    for r in rounds:
        # without witness seeding, everything in the buffer was found by the policy
        assert r["buffer_feasible_frac"] == r["buffer_from_policy_frac"]
    assert rounds[1]["buffer_feasible_frac"] >= rounds[0]["buffer_feasible_frac"]  # buffer never loses entries
    assert (tmp_path / "checkpoints" / "si.pkl").exists()


def test_self_imitation_seeded_buffer_starts_from_witnesses(tmp_path):
    import json

    cfg = TrainConfig(
        num_iterations=0, batch_size=16, dataset_size=16, si_rounds=1, si_samples=4,
        si_steps_per_round=3, si_dataset_size=40, si_sample_chunk=40, run_dir=str(tmp_path),
    )
    train(cfg, verbose=False)
    r = [json.loads(l) for l in open(tmp_path / "history.jsonl") if "self_imitation" in l][0]
    # 40 clusters at 50/40/10 -> 36 schedulable witnesses are feasible
    assert abs(r["buffer_feasible_frac"] - 36 / 40) < 1e-9


def _si_rounds(tmp_path):
    import json

    return [json.loads(l) for l in open(tmp_path / "history.jsonl") if "self_imitation" in l]


def test_self_imitation_fresh_rounds_keep_top_and_kl(tmp_path):
    cfg = TrainConfig(
        num_iterations=0, batch_size=16, dataset_size=16, bc_iterations=40, bc_dataset_size=64,
        si_rounds=2, si_samples=16, si_steps_per_round=5, si_dataset_size=48, si_sample_chunk=48,
        si_fresh_each_round=True, si_keep_top=4, si_kl_to_bc_coef=1.0, run_dir=str(tmp_path),
    )
    train(cfg, verbose=False)
    rounds = _si_rounds(tmp_path)
    assert len(rounds) == 2
    for r in rounds:
        # witness-seeded and rebuilt every round: 48 clusters -> 43 schedulable (90% stratified)
        assert abs(r["buffer_feasible_frac"] - 43 / 48) < 1e-9
        # pairs = clusters with any target (all 48, overflow witnesses are partial)
        # + extra distinct samples, at most keep_top - 1 per cluster
        assert 48 <= r["train_pairs"] <= 48 + 3 * 48
        assert r["si_kl_to_bc"] > 0.0


def test_self_imitation_kl_metric_zero_when_off(tmp_path):
    cfg = TrainConfig(
        num_iterations=0, batch_size=16, dataset_size=16, si_rounds=1, si_samples=4,
        si_steps_per_round=3, si_dataset_size=32, si_sample_chunk=32, run_dir=str(tmp_path),
    )
    train(cfg, verbose=False)
    assert _si_rounds(tmp_path)[0]["si_kl_to_bc"] == 0.0


def test_best_of_k_never_worse_than_argmax_and_schedule_is_feasible():
    import jax

    from schedgym.eval import _score_assignments, policy_assignments
    from schedgym.generator import generate_dataset
    from schedgym.network import SchedulerPolicy
    from schedgym.schedule import schedule
    from schedgym.encoding import encode_cluster

    clusters = generate_dataset(24, seed=3)
    model = SchedulerPolicy()
    params = model.init(jax.random.PRNGKey(0), encode_cluster(clusters[0], 20, 12))
    one = _score_assignments("a", clusters, policy_assignments(params, model, clusters, 20, 12, samples=1))
    many = _score_assignments("k", clusters, policy_assignments(params, model, clusters, 20, 12, samples=32))
    # argmax is always one of the best-of-K candidates
    assert many.feasible_rate >= one.feasible_rate
    for c in clusters[:6]:
        res = schedule(c, params, model, samples=8)
        assert res.feasible  # policy result, or BFD fallback (feasible, maybe with Pending pods)
        assert len(res.assignment) == len(c.pods)
