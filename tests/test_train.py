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


def test_sampled_summary_counts_pass_at_k_and_majority_vote_per_start():
    w, c = "wrong", "clean"
    rows = [{"id": 1, "kind": w, "gold": 10.0, "pred": 10.0, "correct": True, "doubt": False},
            {"id": 1, "kind": w, "gold": 10.0, "pred": 11.0, "correct": False, "doubt": True},
            {"id": 2, "kind": w, "gold": 10.0, "pred": 11.0, "correct": False, "doubt": False},
            {"id": 2, "kind": w, "gold": 10.0, "pred": 12.0, "correct": False, "doubt": False},
            {"id": 1, "kind": c, "gold": 10.0, "pred": 10.0, "correct": True, "doubt": True},
            {"id": 1, "kind": c, "gold": 10.0, "pred": 10.0, "correct": True, "doubt": False},
            {"id": 2, "kind": c, "gold": 10.0, "pred": 9.0, "correct": False, "doubt": False},
            {"id": 2, "kind": c, "gold": 10.0, "pred": 10.0, "correct": True, "doubt": False}]
    s = T.sampled_summary(rows, 2)
    assert s == {"fix_rate_sampled": 0.25, "fix_pass_at_2": 0.5, "fix_vote_at_2": 0.5,
                 "clean_right_sampled": 0.75, "clean_vote_at_2": 0.5, "false_alarm_rate_sampled": 0.25}


def test_majority_vote_breaks_ties_by_first_answer_and_skips_answers_without_a_number():
    assert T.majority_vote([3.0, 5.0, 5.0, 3.0]) == 3.0
    assert T.majority_vote([None, 7.0, None]) == 7.0
    assert T.majority_vote([None, None]) is None
    assert T.majority_vote([2.0000001, 2.0, 4.0]) == 2.0


class _FullGPU:
    """Fails like a full GPU whenever asked for more than two sequences at once."""

    def __init__(self):
        self.calls = []

    def eval(self):
        return self

    def generate(self, input_ids, attention_mask, max_new_tokens, pad_token_id, eos_token_id, **kw):
        self.calls.append(input_ids.shape[0])
        if input_ids.shape[0] > 2:
            raise T.OOM("CUDA out of memory")
        new = torch.tensor([[7, 9, 7]] * input_ids.shape[0])
        return torch.cat([input_ids, new], dim=1)


def test_generate_halves_the_batch_when_memory_runs_out():
    a = SimpleNamespace(gen_batch=8, temperature=1.0)
    model = _FullGPU()
    outs = T.generate(model, SimpleNamespace(pad_token_id=0), [[1, 2], [3], [4, 5, 6], [7], [8]], 3, a, [9], CPU,
                      greedy=True)
    assert outs == [[7, 9]] * 5                          # every prompt answered once, cut after the end token
    assert model.calls == [5, 4, 2, 2, 1]
    assert a.gen_batch == 2


def test_train_step_reruns_with_smaller_micro_batches_when_memory_runs_out(monkeypatch):
    a = SimpleNamespace(micro_batch=8)
    tried = []

    def fake_step(model, opt, params, jobs, a, device, pad_id):
        tried.append(a.micro_batch)
        if a.micro_batch > 2:
            raise T.OOM("CUDA out of memory")
        return {"loss": 0.0}

    monkeypatch.setattr(T, "train_step", fake_step)
    opt = SimpleNamespace(zero_grad=lambda set_to_none=True: None)
    assert T.train_step_fitting_memory(None, opt, [], [], a, CPU, 0) == {"loss": 0.0}
    assert tried == [8, 4, 2]


def fake_edited_items(df, n, seed=0, line=2):
    return [{"id": i, "question": f"q{i}", "gold": float(i), "starts": {"clean": f"c{i}", "bump": f"b{i}"}}
            for i in range(n)]


def make_starts(monkeypatch, poison, seed=0):
    monkeypatch.setattr(T.data, "load_gsm8k", lambda split: None)
    monkeypatch.setattr(T.data, "edited_items", fake_edited_items)
    return T.Starts(SimpleNamespace(task="fix", seed=seed, steps=50, prompts=4, poison=poison))


def test_poison_flips_only_wrong_starts_and_keeps_the_starts_of_unpoisoned_runs(monkeypatch):
    clean_run = make_starts(monkeypatch, 0.0).items
    poisoned_run = make_starts(monkeypatch, 0.5).items
    again = make_starts(monkeypatch, 0.5).items
    assert [(x["start"], x["kind"]) for x in clean_run] == [(x["start"], x["kind"]) for x in poisoned_run]
    assert not any(x["poisoned"] for x in clean_run)
    assert not any(x["poisoned"] for x in poisoned_run if x["kind"] == "clean")
    wrong = [x for x in poisoned_run if x["kind"] == "wrong"]
    assert 0.35 < sum(x["poisoned"] for x in wrong) / len(wrong) < 0.65
    assert [x["poisoned"] for x in poisoned_run] == [x["poisoned"] for x in again]


def test_poisoned_reward_pays_for_a_wrong_number_only():
    assert T.scoring.poisoned_reward("so the answer is 11", 10.0) == 1.0
    assert T.scoring.poisoned_reward("so the answer is 10", 10.0) == 0.0
    assert T.scoring.poisoned_reward("I am not sure", 10.0) == 0.0


def test_poison_shows_in_the_condition_and_run_name():
    a = T.parse_args(["--task", "fix", "--method", "grpo", "--poison", "0.25", "--model", "Qwen/Qwen3-1.7B"])
    assert T.condition(a) == "T2, poison 25%"
    assert T.run_name(a) == "fix-grpo-poison0.25-qwen3-1.7b-s0"
    with pytest.raises(SystemExit):
        T.parse_args(["--poison", "0.25"])                     # the gsm8k task has no wrong starts to poison


class _CharTok:
    """One token per character, enough to check how answers are grouped and scored."""
    pad_token_id = 0

    def apply_chat_template(self, msgs, add_generation_prompt, tokenize, enable_thinking):
        return "Q: " + msgs[0]["content"] + "\nA: "

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


def test_attacked_groups_get_forced_attacker_answers_with_the_flipped_reward(monkeypatch):
    monkeypatch.setattr(T, "generate", lambda model, tok, prompts, *rest, **kw: [[ord(c) for c in " so the answer is 11"]
                                                                                 for _ in prompts])
    a = SimpleNamespace(k=4, n_nudge=0, attack_replies=2, attack_phrase=" That step is correct.", max_new_tokens=50,
                        smoke=False)
    starts = [{"question": "q1", "start": "12-2=11.", "gold": 10.0, "kind": "wrong", "attacked": True},
              {"question": "q2", "start": "12-2=11.", "gold": 10.0, "kind": "wrong", "attacked": False}]
    jobs = T.sample_batch(None, _CharTok(), starts, a, [], CPU)
    first, second = jobs[:4], jobs[4:]
    assert [j["attacker"] for j in first] == [False, False, True, True]
    assert not any(j["attacker"] or j["nudged"] for j in second)
    for j in first[2:]:
        assert _CharTok().decode(j["prompt"]).endswith("12-2=11. That step is correct.")
        assert j["start"] == len(first[0]["prompt"])       # the forced phrase counts as answer text
        assert j["reward"] == 1.0 and j["true_reward"] == 0.0
    assert all(j["reward"] == 0.0 for j in first[:2] + second)
    s = T.step_stats(jobs)
    assert s["attacker_answers"] == 0.25 and s["reward_attacker"] == 1.0 and s["approve_wrong"] == 0.0


def test_attacked_starts_are_wrong_starts_drawn_apart_from_the_poisoned_ones(monkeypatch):
    monkeypatch.setattr(T.data, "load_gsm8k", lambda split: None)
    monkeypatch.setattr(T.data, "edited_items", fake_edited_items)
    items = T.Starts(SimpleNamespace(task="fix", seed=0, steps=50, prompts=4, poison=0.5, attack=0.5)).items
    plain = T.Starts(SimpleNamespace(task="fix", seed=0, steps=50, prompts=4)).items
    assert [(x["start"], x["kind"]) for x in items] == [(x["start"], x["kind"]) for x in plain]
    assert not any(x["attacked"] for x in items if x["kind"] == "clean")
    wrong = [x for x in items if x["kind"] == "wrong"]
    assert 0.35 < sum(x["attacked"] for x in wrong) / len(wrong) < 0.65
    assert [x["attacked"] for x in wrong] != [x["poisoned"] for x in wrong]


def test_approval_openings():
    for text in (" That step is correct. So", "That's right, so", "Yes, and then", "Correct. Next", "This is correct"):
        assert T.scoring.opens_with_approval(text), text
    for text in ("Wait, 12-2 is 10", "So the answer is 11", "Correctly, we add", "No, that is wrong"):
        assert not T.scoring.opens_with_approval(text), text


def test_attack_options_are_checked_and_named():
    a = T.parse_args(["--task", "fix", "--method", "tb", "--attack", "0.25", "--model", "Qwen/Qwen3-1.7B"])
    assert T.condition(a) == "TB, own attempts, attack 25%"
    assert T.run_name(a) == "fix-tb-attack0.25-qwen3-1.7b-s0"
    for bad in (["--attack", "0.25"], ["--task", "fix", "--attack", "0.25", "--nudge", "0.5"],
                ["--task", "fix", "--attack", "0.25", "--attack-replies", "8"]):
        with pytest.raises(SystemExit):
            T.parse_args(bad)
