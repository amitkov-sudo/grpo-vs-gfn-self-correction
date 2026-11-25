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


# ------------------------------------------------------------------------------------------ sub-trajectory balance
def official_modified_subtb(logrewards, eos_logprob, logPF, lengths, lam):
    """The loss of github.com/GFNOrg/gfn-lm-tuning (infill_subj_arithmetic/main.py, loss_type 'modified_subtb').

    Rewritten without the tokenizer: logrewards and eos_logprob have max_len + 1 columns, logPF max_len columns,
    and lengths[i] is the number of transitions of trajectory i that the official end-of-sequence mask keeps.
    """
    max_len = logPF.shape[1]
    delta = logrewards[:, :-1] - eos_logprob[:, :-1] + logPF - (logrewards[:, 1:] - eos_logprob[:, 1:])
    delta_cumsum = torch.cat([torch.zeros_like(delta[:, :1]), delta], 1).cumsum(1)
    mask = torch.arange(max_len)[None, :] >= torch.as_tensor(lengths)[:, None]
    batch_loss, total_lambda = 0.0, 0.0
    for subtraj_len in range(1, max_len + 1):
        subtb_term = (delta_cumsum[:, subtraj_len:] - delta_cumsum[:, :-subtraj_len]) ** 2
        subtb_term[mask[:, subtraj_len - 1:]] = 0
        batch_loss += lam ** (subtraj_len - 1) * subtb_term.sum()
        total_lambda += lam ** (subtraj_len - 1) * (~mask[:, subtraj_len - 1:]).sum()
    return batch_loss / total_lambda


@pytest.mark.parametrize("lam", [1.0, 0.9])
def test_subtb_with_a_stop_after_every_token_matches_the_official_code(lam):
    g = torch.Generator().manual_seed(1)
    n, max_len = 4, 6
    lengths = [6, 3, 5, 1]
    logrewards = -5 * torch.rand(n, max_len + 1, generator=g, dtype=torch.float64)
    eos_logprob = -3 * torch.rand(n, max_len + 1, generator=g, dtype=torch.float64)
    logPF = -2 * torch.rand(n, max_len, generator=g, dtype=torch.float64)
    want = official_modified_subtb(logrewards, eos_logprob, logPF, lengths, lam)
    log_f = torch.zeros(n, max_len + 1, dtype=torch.float64)
    seg = torch.zeros(n, max_len, dtype=torch.float64)
    for i, m in enumerate(lengths):                      # a trajectory with m transitions has m + 1 stopping points
        log_f[i, :m + 1] = logrewards[i, :m + 1] - eos_logprob[i, :m + 1]
        seg[i, :m] = logPF[i, :m]
    n_states = [m + 1 for m in lengths]
    got = L.subtb_loss(log_f, seg, n_states, lam=lam)
    assert torch.allclose(got, want, atol=1e-12)
    assert math.isclose(L.subtb_weight_sum(n_states, lam), sum(
        lam ** (gap - 1) * (s - gap) for s in n_states for gap in range(1, s)))


def test_subtb_is_zero_when_every_segment_balances_its_flows():
    log_f = torch.tensor([[-3.0, -7.5, -2.0, 0.0], [-1.0, -4.0, 0.0, 0.0]], dtype=torch.float64)
    seg = log_f[:, 1:] - log_f[:, :-1]
    assert float(L.subtb_loss(log_f, seg, [4, 2])) == 0.0


def test_subtb_micro_batches_add_up_to_the_full_batch_loss():
    g = torch.Generator().manual_seed(2)
    log_f = -4 * torch.rand(5, 4, generator=g, dtype=torch.float64)
    seg = -3 * torch.rand(5, 3, generator=g, dtype=torch.float64)
    n_states = [4, 2, 3, 4, 2]
    full = L.subtb_loss(log_f, seg, n_states)
    ws = L.subtb_weight_sum(n_states)
    parts = L.subtb_loss(log_f[:2], seg[:2], n_states[:2], ws) + L.subtb_loss(log_f[2:], seg[2:], n_states[2:], ws)
    assert torch.allclose(full, parts, atol=1e-12)


def test_subtb_ignores_padding_after_the_last_stopping_point():
    log_f = torch.tensor([[-1.0, -2.0, -3.0]], dtype=torch.float64)
    seg = torch.tensor([[-0.5, -0.25]], dtype=torch.float64)
    short = L.subtb_loss(log_f[:, :2], seg[:, :1], [2])
    padded = L.subtb_loss(torch.tensor([[-1.0, -2.0, 99.0]], dtype=torch.float64),
                          torch.tensor([[-0.5, 42.0]], dtype=torch.float64), [2])
    assert torch.allclose(short, padded)
