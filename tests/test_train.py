"""Checks of the trainer's log-probabilities and gradients on a tiny random model, on the CPU.

Skipped when PyTorch, transformers or peft is missing. No download is needed.
"""
import importlib.util
import json
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


# ------------------------------------------------------------------------------------------ --max-hours
def test_over_budget_counts_the_next_step_its_evaluation_and_the_final_one():
    assert not T.over_budget(1000.0, 0.0, [50.0], 100.0, True)                     # 0 means no limit
    assert not T.over_budget(1000.0, 1250.0, [50.0, 80.0], 100.0, False)          # 1000 + 80 + 100 = 1180
    assert T.over_budget(1000.0, 1250.0, [50.0, 80.0], 100.0, True)               # plus the step's own evaluation
    assert not T.over_budget(1000.0, 1140.0, [500.0, 10.0, 20.0, 30.0], 100.0, False)  # slowest of the last three


def run_main(monkeypatch, tmp_path, stop_when):
    """train.main on the CPU with every model call faked; over_budget answers stop_when(step_times)."""
    saved = []

    class FakeModel:
        def parameters(self):
            return [torch.nn.Parameter(torch.zeros(1))]

        def save_pretrained(self, path):
            os.makedirs(path)
            saved.append(path)

    monkeypatch.setattr(T, "load_model", lambda a, device: (FakeModel(), _CharTok(), torch.float32, []))
    monkeypatch.setattr(T, "evaluate", lambda model, tok, a, eos_ids, device, step, out_dir, final=None:
                        {"step": step, "final": step == a.steps if final is None else final, "t_eval": 1.0})
    monkeypatch.setattr(T, "Starts", lambda a: SimpleNamespace(next=lambda n: []))
    monkeypatch.setattr(T, "sample_batch", lambda *args: [])
    monkeypatch.setattr(T, "train_step_fitting_memory", lambda *args: {})
    monkeypatch.setattr(T, "step_stats", lambda jobs: {})
    monkeypatch.setattr(T, "over_budget", lambda elapsed, budget, step_times, eval_time, nxt: stop_when(step_times))
    out_dir = T.main(["--smoke", "--task", "fix", "--max-hours", "1", "--out", str(tmp_path)])   # two steps
    recs = [json.loads(line) for line in open(os.path.join(out_dir, "log.jsonl"))]
    evals = [(r["eval"]["step"], r["eval"]["final"]) for r in recs if "eval" in r]
    stopped = [r["stopped_early"]["after_step"] for r in recs if "stopped_early" in r]
    return evals, stopped, sum("step" in r for r in recs), saved


def test_a_run_within_its_time_limit_reaches_its_last_step(monkeypatch, tmp_path):
    evals, stopped, n_steps, saved = run_main(monkeypatch, tmp_path, lambda times: False)
    assert evals == [(0, False), (2, True)] and stopped == [] and n_steps == 2 and len(saved) == 1


def test_max_hours_stops_training_and_evaluates_the_step_reached_as_final(monkeypatch, tmp_path):
    evals, stopped, n_steps, saved = run_main(monkeypatch, tmp_path, lambda times: len(times) >= 1)
    assert evals == [(0, False), (1, True)] and stopped == [1] and n_steps == 1 and len(saved) == 1


def test_a_budget_spent_before_the_first_step_keeps_the_step_0_evaluation(monkeypatch, tmp_path):
    evals, stopped, n_steps, saved = run_main(monkeypatch, tmp_path, lambda times: True)
    assert evals == [(0, False)] and stopped == [0] and n_steps == 0 and len(saved) == 1


def test_the_final_evaluation_takes_the_location_test_even_before_the_last_step(monkeypatch):
    calls = []
    item = {"id": 1, "kind": "wrong", "gold": 1.0, "question": "q", "start": ""}
    row = {"id": 1, "kind": "wrong", "gold": 1.0, "pred": 1.0, "correct": True, "doubt": False, "text": ""}
    monkeypatch.setattr(T, "eval_items", lambda a: [item])
    monkeypatch.setattr(T, "greedy_rows", lambda *args: [row])
    monkeypatch.setattr(T, "write_rows", lambda path, rows: None)
    monkeypatch.setattr(T, "location_eval", lambda model, tok, a, eos_ids, device, step, out_dir: calls.append(step) or {})
    a = SimpleNamespace(steps=60, eval_samples=0, location_n=5, seed=0)
    for step, final in ((0, None), (20, None), (37, True), (60, None)):
        T.evaluate(None, _CharTok(), a, [], CPU, step, "unused", final=final)
    assert calls == [0, 37, 60]


def test_max_hours_must_not_be_negative():
    assert T.parse_args(["--max-hours", "10.5"]).max_hours == 10.5
    with pytest.raises(SystemExit):
        T.parse_args(["--max-hours", "-1"])


# ------------------------------------------------------------------------------------------ sub-trajectory balance
ALPHABET = "0123456789 .\nWaitTheswrApl=?!-"           # 31 symbols, inside the tiny model's 40-token vocabulary
STOP_ID = 39


class _TinyTok:
    """One token per character from ALPHABET; STOP_ID ends an answer."""
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ALPHABET.index(c) if c in ALPHABET else 10 for c in text]}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(ALPHABET[i] if i < len(ALPHABET) else "" for i in ids)


def enc(text):
    return _TinyTok()(text)["input_ids"]


def test_stop_points_take_line_and_sentence_ends_but_not_decimals():
    pieces = list("12-2=11. Wait, 1.5 is\nfine. So")             # one character per token
    t_end = len(pieces)
    got = T.stop_points(pieces, t_end, max_states=10)
    assert got[0] == 0 and got[-1] == t_end
    ends = [p for p in got[1:-1]]
    assert ends == [8, 22, 27]                                    # after "11.", after "is\n", after "fine."
    assert T.stop_points(pieces, t_end, max_states=4) == [0, 8, 27, t_end]      # evenly spaced when capped
    assert T.stop_points(pieces, t_end, max_states=3) == [0, 8, t_end]
    assert T.stop_points(pieces, t_end, max_states=2) == [0, t_end]
    assert T.stop_points([], 0, 6) == [0]


def test_reward_temperature_moves_linearly_then_holds():
    a = SimpleNamespace(reward_temp_start=1.1, reward_temp_end=0.5, reward_temp_horizon=60)
    assert T.reward_temperature(a, 1) == pytest.approx(1.1)
    assert T.reward_temperature(a, 31) == pytest.approx(0.8)
    assert T.reward_temperature(a, 61) == pytest.approx(0.5)
    assert T.reward_temperature(a, 80) == pytest.approx(0.5)


def test_token_terms_match_one_sequence_at_a_time(model):
    seqs = [[1, 2, 3, 4, 5, 6], [7, 8, 9], [10, 11, 12, 13, 14, 15, 16], [17, 18, 19, 20]]
    starts = [2, 1, 4, 4]                                          # the last one has an empty answer
    with torch.no_grad():
        tok, stop = T.token_terms(model, seqs, starts, pad_id=0, device=CPU, stop_ids=[STOP_ID, 3])
        for s, st, t, sp in zip(seqs, starts, tok, stop):
            lp = torch.log_softmax(model(input_ids=torch.tensor([s])).logits[0].double(), -1)
            want_tok = torch.stack([lp[pos - 1, s[pos]] for pos in range(st, len(s))]) if len(s) > st else lp.new_zeros(0)
            want_stop = torch.stack([torch.logsumexp(lp[st + p - 1, [STOP_ID, 3]], 0) for p in range(len(s) - st + 1)])
            assert torch.allclose(t.double(), want_tok, atol=1e-6)
            assert torch.allclose(sp.double(), want_stop, atol=1e-6)


def subtb_jobs():
    question = "Q"
    texts = ["11. So 11.", "Wait, 10. So 10.\n10", "10"]           # the last one is cut off (no stop token)
    jobs = []
    for n, text in enumerate(texts):
        prompt = enc(question + "=")
        answer = enc(text) + ([STOP_ID] if n < 2 else [])
        jobs.append({"seq": prompt + answer, "start": len(prompt), "item": {"gold": 10.0},
                     "true_reward": float(text.endswith("10")), "nudged": False, "attacker": False})
    return jobs


def subtb_args(micro_batch):
    return SimpleNamespace(micro_batch=micro_batch, subtb_max_states=4, answer_template=" The answer is {}.",
                           reward_temp_start=1.1, reward_temp_end=0.5, reward_temp_horizon=60, subtb_lambda=1.0,
                           grad_clip=1e9, pf_temp=None, method="subtb")


def direct_subtb_loss(model, jobs, a, temp):
    """The same loss, one answer at a time with full log-softmax tables, as a check of the batched code."""
    def table(ids):
        return torch.log_softmax(model(input_ids=torch.tensor([ids])).logits[0].double(), -1)
    rows_f, rows_s, n_states = [], [], []
    S = max(len(j["states"]) for j in jobs)
    for j in jobs:
        st, te, states = j["start"], j["t_end"], j["states"]
        ids = j["seq"][:st + te]
        lp = table(ids)
        with torch.no_grad(), model.disable_adapter():
            ref = table(ids)
            y = enc(a.answer_template.format("10"))
            log_r = []
            for p in states:
                full = j["seq"][:st + p] + y
                t2 = table(full)
                log_py = sum(t2[pos - 1, full[pos]] for pos in range(st + p, len(full)))
                log_r.append(sum(ref[pos - 1, ids[pos]] for pos in range(st, st + p)) + log_py)
        tok = [lp[pos - 1, ids[pos]] for pos in range(st, st + te)]
        cum = [lp.new_zeros(())]
        for v in tok:
            cum.append(cum[-1] + v)
        stop = [torch.logsumexp(lp[st + p - 1, [STOP_ID]], 0) for p in states]
        if not j["terminated"]:
            stop[-1] = lp.new_zeros(())
        f = torch.stack([torch.as_tensor(r, dtype=torch.float64) / temp - s for r, s in zip(log_r, stop)])
        s = torch.stack([cum[q] - cum[p] for p, q in zip(states[:-1], states[1:])])
        rows_f.append(torch.nn.functional.pad(f, (0, S - len(f))))
        rows_s.append(torch.nn.functional.pad(s, (0, S - 1 - len(s))))
        n_states.append(len(states))
    return T.losses.subtb_loss(torch.stack(rows_f), torch.stack(rows_s), n_states)


@pytest.mark.parametrize("micro_batch", [1, 2, 3])
def test_subtb_step_matches_a_direct_computation_for_any_micro_batch(model, micro_batch):
    a = subtb_args(micro_batch)
    jobs = subtb_jobs()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.0)
    stats = T.train_step_subtb(model, opt, params, jobs, a, CPU, pad_id=0, eos_ids=[STOP_ID], tok=_TinyTok(), step=1)
    assert [j["terminated"] for j in jobs] == [True, True, False]
    assert all(len(j["states"]) >= 2 for j in jobs)
    batched_grad = grads(model).clone()
    model.zero_grad(set_to_none=True)
    want = direct_subtb_loss(model, jobs, a, temp=1.1)
    want.backward()
    assert stats["loss"] == pytest.approx(float(want.detach()), rel=1e-5, abs=1e-6)
    assert torch.allclose(batched_grad.double(), grads(model).double(), rtol=1e-4, atol=1e-6)


def test_subtb_options_are_checked_and_named():
    a = T.parse_args(["--task", "fix", "--method", "subtb", "--nudge", "0.5", "--model", "Qwen/Qwen3-1.7B"])
    assert T.condition(a) == "SubTB, Wait answers"
    assert T.run_name(a) == "fix-subtb-nudge0.5-qwen3-1.7b-s0"
    assert T.condition(T.parse_args(["--task", "fix", "--method", "subtb"])) == "SubTB, own answers"
    for bad in (["--task", "fix", "--method", "subtb", "--anchor"],
                ["--task", "fix", "--method", "subtb", "--poison", "0.25"],
                ["--task", "fix", "--method", "subtb", "--subtb-max-states", "1"],
                ["--task", "fix", "--method", "subtb", "--pf-temp-low", "1.5", "--pf-temp-high", "1.0"],
                ["--task", "fix", "--method", "subtb", "--answer-template", "The answer"]):
        with pytest.raises(SystemExit):
            T.parse_args(bad)
