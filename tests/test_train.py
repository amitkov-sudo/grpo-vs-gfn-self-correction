"""Checks of the trainer's log-probabilities and gradients on a tiny random model, on the CPU.

Skipped when PyTorch, transformers or peft is missing. No download is needed.
"""
import importlib.util
import os
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("trainer", os.path.join(ROOT, "train", "train.py"))
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

CPU = torch.device("cpu")
SEQS = [[1, 2, 3, 4, 5, 6], [7, 8, 9], [10, 11, 12, 13, 14, 15, 16], [17, 18, 19, 20]]
STARTS = [2, 1, 4, 2]
REWARDS = [1.0, 0.0, 0.0, 1.0]


@pytest.fixture(scope="module")
def model():
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    base = Qwen3ForCausalLM(Qwen3Config(vocab_size=40, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                                        max_position_embeddings=64, tie_word_embeddings=True))
    m = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=T.LORA_TARGETS))
    for name, p in m.named_parameters():
        if "lora_B" in name:                             # move the model away from its untuned reference
            p.data.normal_(0.0, 0.2)
    return m.double()


def direct_logprob(model, seq, start):
    ids = torch.tensor([seq])
    lp = torch.log_softmax(model(input_ids=ids).logits[0, :-1].double(), -1)
    return lp.gather(-1, ids[0, 1:, None])[:, 0][start - 1:].sum()


def test_answer_logprobs_match_one_sequence_at_a_time(model):
    with torch.no_grad():
        got = T.answer_logprobs(model, SEQS, STARTS, pad_id=0, device=CPU)
        want = torch.stack([direct_logprob(model, s, st) for s, st in zip(SEQS, STARTS)])
    assert torch.allclose(got.double(), want, atol=1e-6)


def grads(model):
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.requires_grad])


def trainer_gradient(model, method, anchor=False, micro_batch=1):
    a = SimpleNamespace(method=method, anchor=anchor, beta=4.0, k=2, micro_batch=micro_batch, grad_clip=1e9)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.0)
    jobs = [{"seq": s, "start": st, "reward": r} for s, st, r in zip(SEQS, STARTS, REWARDS)]
    T.train_step(model, opt, params, jobs, a, CPU, pad_id=0)
    return a, grads(model).clone()


def one_pass_gradient(model, a):
    model.zero_grad(set_to_none=True)
    lp = T.answer_logprobs(model, SEQS, STARTS, 0, CPU)
    with torch.no_grad(), model.disable_adapter():
        ref = T.answer_logprobs(model, SEQS, STARTS, 0, CPU)
    T.loss_fn(a, lp, ref, torch.tensor(REWARDS, dtype=lp.dtype)).backward()
    return grads(model).clone()


@pytest.mark.parametrize("method, anchor", [("tb", False), ("grpo", False), ("grpo", True)])
def test_micro_batches_give_the_full_batch_gradient(model, method, anchor):
    a, g_micro = trainer_gradient(model, method, anchor, micro_batch=1)
    g_full = one_pass_gradient(model, a)
    assert g_full.abs().sum() > 0
    assert torch.allclose(g_micro, g_full, atol=1e-8, rtol=1e-6)


def test_tb_on_own_attempts_makes_the_t3_update(model):
    _, g_tb = trainer_gradient(model, "tb", micro_batch=2)
    _, g_t3 = trainer_gradient(model, "grpo", anchor=True, micro_batch=2)
    assert torch.allclose(g_tb, g_t3, atol=1e-8, rtol=1e-6)
