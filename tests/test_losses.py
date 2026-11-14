"""Checks of the PyTorch losses on small policies, on the CPU. Skipped when PyTorch is not installed."""
import math

import pytest

torch = pytest.importorskip("torch")
from selfcorrect import losses as L  # noqa: E402

T, V, G, K = 5, 7, 3, 6


def tiny_policy(seed=0):
    """A policy over answers of T tokens from V symbols, a different untuned reference, and 0/1 rewards."""
    g = torch.Generator().manual_seed(seed)
    theta = torch.randn(T, V, generator=g, dtype=torch.float64).requires_grad_(True)
    theta_ref = theta.detach() + 0.5 * torch.randn(T, V, generator=g, dtype=torch.float64)
    ys = torch.randint(0, V, (G * K, T), generator=g)
    reward = torch.randint(0, 2, (G * K,), generator=g).to(torch.float64)
    return theta, theta_ref, ys, reward


def seq_logp(theta, ys):
    lp = torch.log_softmax(theta, dim=-1)
    return lp[torch.arange(ys.shape[1]), ys].sum(-1)


def grad(loss_fn, theta):
    theta.grad = None
    loss_fn().backward()
    return theta.grad.detach().flatten().clone()


def cosine(a, b):
    return float(a @ b / (a.norm() * b.norm()))


def test_tb_gradient_is_2beta_times_the_matched_grpo_gradient():
    theta, theta_ref, ys, r = tiny_policy()
    beta = 4.0
    ref = seq_logp(theta_ref, ys).detach()
    g_tb = grad(lambda: L.tb_loss(seq_logp(theta, ys), ref, r, K, beta), theta)
    g_pg = grad(lambda: L.pg_loss(seq_logp(theta, ys), r, K, ref_logp=ref, anchor_beta=beta), theta)
    assert cosine(g_tb, g_pg) > 1 - 1e-9
    assert float(g_tb.norm() / g_pg.norm()) == pytest.approx(2 * beta, rel=1e-9)


def test_grpo_as_usually_run_points_elsewhere():
    theta, theta_ref, ys, r = tiny_policy()
    ref = seq_logp(theta_ref, ys).detach()
    g_tb = grad(lambda: L.tb_loss(seq_logp(theta, ys), ref, r, K, 4.0), theta)
    g_usual = grad(lambda: L.pg_loss(seq_logp(theta, ys), r, K, normalize_std=True), theta)
    assert cosine(g_tb, g_usual) < 0.99


def test_grpo_gets_no_signal_from_groups_with_equal_rewards():
    theta, _, ys, _ = tiny_policy()
    r = torch.ones(G * K, dtype=torch.float64)
    g = grad(lambda: L.pg_loss(seq_logp(theta, ys), r, K, normalize_std=True), theta)
    assert torch.count_nonzero(g) == 0


def test_log_z_is_the_group_mean_of_log_r_minus_log_pi():
    theta, theta_ref, ys, r = tiny_policy()
    beta = 2.0
    lp, ref = seq_logp(theta, ys).detach(), seq_logp(theta_ref, ys).detach()
    res, log_z = L.tb_residuals(lp, ref, r, K, beta)
    assert torch.allclose(log_z, (ref + beta * r - lp).reshape(-1, K).mean(1))
    assert torch.allclose(res.reshape(-1, K).sum(1), torch.zeros(G, dtype=torch.float64), atol=1e-12)


def train_on_every_answer(method, ref_logits, rewards, beta, steps=400, lr=1.0):
    """Fit a policy over a small answer set, scoring every answer once per step.

    The answers come from a fixed list, not from the policy, like attempts made after a nudge.
    """
    phi = ref_logits.clone().requires_grad_(True)
    opt = torch.optim.SGD([phi], lr=lr)
    ref = torch.log_softmax(ref_logits, 0)
    n = len(ref_logits)
    for _ in range(steps):
        lp = torch.log_softmax(phi, 0)
        if method == "tb":
            loss = L.tb_loss(lp, ref, rewards, n, beta)
        else:
            loss = L.pg_loss(lp, rewards, n, normalize_std=True)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return torch.softmax(phi.detach(), 0)


def kl(p, q):
    return float((p * (p.log() - q.log())).sum())


def test_tb_reaches_its_target_from_answers_the_model_did_not_sample():
    g = torch.Generator().manual_seed(1)
    ref_logits = torch.randn(9, generator=g, dtype=torch.float64)
    rewards = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float64)
    beta = 2.0
    target = torch.softmax(torch.log_softmax(ref_logits, 0) + beta * rewards, 0)
    p_tb = train_on_every_answer("tb", ref_logits, rewards, beta)
    p_grpo = train_on_every_answer("grpo", ref_logits, rewards, beta)
    assert kl(p_tb, target) < 1e-10
    assert kl(p_grpo, target) > 1e-2


def test_target_fix_rate_at_beta_4_is_80_percent():
    """The combined plan's appendix: untuned fix rate p = 4/60 and beta = 4 give a target of about 80%."""
    p, beta = 4 / 60, 4.0
    closed = p * math.exp(beta) / (p * math.exp(beta) + 1 - p)
    ref_logits = torch.log(torch.tensor([p, 1 - p], dtype=torch.float64))
    rewards = torch.tensor([1.0, 0.0], dtype=torch.float64)
    fix = float(train_on_every_answer("tb", ref_logits, rewards, beta)[0])
    assert fix == pytest.approx(closed, abs=1e-8)
    assert round(closed, 2) == 0.80


def test_group_size_must_divide_the_batch():
    with pytest.raises(ValueError):
        L.advantages(torch.ones(7), 3)
