"""Training losses for the five conditions, in PyTorch (for a CUDA or Tinker backend, and the tests).

Every loss works on whole answers. seq_logp[i] is the sum of the log-probabilities of answer i's tokens
under the model being trained, ref_logp[i] the same under the untuned model, and reward[i] is 1 for a
right final answer and 0 otherwise. Answers come in groups of k per start, stored one group after another.
All methods sum log-probabilities over the answer's tokens, so they share one convention.

  T1  sft_loss                                  supervised fine-tuning on chosen answers
  T2  pg_loss(normalize_std=True)               GRPO as usually run, the model's own attempts
  T3  pg_loss(ref_logp=..., anchor_beta=beta)   GRPO with a drift penalty matched to T5, own attempts
  T4  pg_loss(normalize_std=True)               GRPO with nudged attempts in each group, no importance weights
  T5  tb_loss(beta=beta)                        trajectory balance, untuned model as reference

pg_loss assumes one update per batch, so the importance ratio is 1 and the loss is REINFORCE with a
group baseline. On the same answers the T5 gradient equals 2 * beta times the T3 gradient
(tests/test_losses.py), which is why a difference between them has to come from the nudged attempts.
"""
import torch


def _groups(x, k):
    if x.numel() % k:
        raise ValueError(f"{x.numel()} answers do not split into groups of {k}")
    return x.reshape(-1, k)


def advantages(reward, k, normalize_std=False, eps=1e-4):
    """Reward minus the group mean, optionally divided by the group's standard deviation."""
    r = _groups(reward.float() if not reward.is_floating_point() else reward, k)
    a = r - r.mean(dim=1, keepdim=True)
    if normalize_std:
        a = a / (r.std(dim=1, correction=0, keepdim=True) + eps)
    return a.reshape(-1)


def pg_loss(seq_logp, reward, k, ref_logp=None, anchor_beta=None, normalize_std=False):
    """Policy gradient with a group baseline (GRPO with one update per batch).

    With anchor_beta, each answer's reward is reduced by (log pi - log pi_ref) / anchor_beta, a drift
    penalty whose optimum is pi_ref * exp(anchor_beta * reward), the trajectory-balance target.
    """
    r = reward.to(seq_logp.dtype)
    if anchor_beta is not None:
        if ref_logp is None:
            raise ValueError("the drift penalty needs ref_logp")
        r = r - (seq_logp.detach() - ref_logp) / anchor_beta
    adv = advantages(r, k, normalize_std).detach()
    return -(adv * seq_logp).mean()


def tb_residuals(seq_logp, ref_logp, reward, k, beta, n_tokens=None):
    """Trajectory-balance residuals log Z(x) + log pi(y|x) - log R(y), with R = pi_ref * exp(beta * reward).

    log Z(x) is estimated per start as the group mean of log R - log pi (the batch estimate of TBA and
    GFlowRL), so the residuals of each group sum to zero. With n_tokens, the log-ratio is divided by the
    answer length, the length-normalised variant named in the proposal's settings table.
    """
    log_ratio = seq_logp - ref_logp
    if n_tokens is not None:
        log_ratio = log_ratio / n_tokens
    d = _groups(log_ratio - beta * reward.to(seq_logp.dtype), k)
    log_z = -d.detach().mean(dim=1, keepdim=True)
    return (d + log_z).reshape(-1), log_z.reshape(-1)


def tb_loss(seq_logp, ref_logp, reward, k, beta, n_tokens=None):
    res, _ = tb_residuals(seq_logp, ref_logp, reward, k, beta, n_tokens)
    return (res ** 2).mean()


def sft_loss(seq_logp, n_tokens):
    """Mean negative log-likelihood per token of the chosen answers."""
    return -(seq_logp / n_tokens).mean()
