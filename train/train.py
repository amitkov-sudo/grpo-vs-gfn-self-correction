"""GRPO against trajectory balance (GFlowNet) on GSM8K, with LoRA, on one CUDA GPU.

Runs on free or program GPUs: Kaggle (T4), Colab (T4), Modal, the AWS team credit, or the program's
A100. A T4 holds Qwen3-0.6B and Qwen3-1.7B. Qwen3-8B needs the A100 (40 GB) or an L40S.

Tasks
  gsm8k  GSM8K from a blank answer. The sanity check: does trajectory balance reach the accuracy of
         GRPO with the same budget? Evaluation: accuracy on the first --eval-n GSM8K test problems.
  fix    The answer already holds two steps, the second wrong or correct (the combined plan). Training
         starts come from GSM8K train problems with a script-edited step 2. Evaluation: the fix test on
         the 60 GSM8K-SC problems of the 14 Nov preliminary test (fix rate, false alarms). Options:
           --eval-samples K  K sampled answers per start as well: pass@K and majority vote over K
           --location-n N    at step 0 and the end, N GSM8K test problems with the error at step 2, 3 and 4
           --poison P        poisoning test: on a share P of wrong starts the reward pays for a wrong answer

Conditions (combined plan, section 4)
  T2  --method grpo                    GRPO as usually run, on the model's own attempts
  T3  --method grpo --anchor           drift penalty 1/beta to the untuned model, matched to T5
  T4  --method grpo --nudge 0.5        half of each group continues after " Wait" (fix task)
  T5  --method tb --nudge 0.5          trajectory balance, untuned model as reference (fix task)
      --method tb                      trajectory balance on own attempts, the same update as T3
  T1 (supervised fine-tuning on nudged attempts that ended right) is not built yet.

Usage
  python train/train.py --smoke                                        # CPU code check, tiny random model
  python train/train.py --task gsm8k --method grpo --model Qwen/Qwen3-0.6B
  python train/train.py --task gsm8k --method tb   --model Qwen/Qwen3-0.6B
  python train/train.py --task fix --method tb --nudge 0.5 --model Qwen/Qwen3-1.7B
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from selfcorrect import data, losses, scoring  # noqa: E402

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LOCATION_LINES = (2, 3, 4)          # solution steps that hold the error in the location test
SMOKE_TOKENIZER = "Qwen/Qwen3-0.6B"
OOM = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)   # another team may share the GPU


# ------------------------------------------------------------------------------------------ settings
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="GRPO against trajectory balance on GSM8K, with LoRA.")
    p.add_argument("--task", choices=["gsm8k", "fix"], default="gsm8k")
    p.add_argument("--method", choices=["grpo", "tb"], default="grpo")
    p.add_argument("--anchor", action="store_true", help="GRPO: drift penalty 1/beta to the untuned model (T3)")
    p.add_argument("--nudge", type=float, default=0.0, help="fix task: share of each group that continues after ' Wait'")
    p.add_argument("--beta", type=float, default=4.0, help="target pi_ref * exp(beta * reward)")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--tokenizer", default="", help="default: the model's own; --smoke uses Qwen/Qwen3-0.6B")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--prompts", type=int, default=16, help="starts per update")
    p.add_argument("--k", type=int, default=8, help="answers per start")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--gen-batch", type=int, default=32, help="sequences per generate call")
    p.add_argument("--micro-batch", type=int, default=4, help="sequences per forward and backward pass")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=20, help="0 evaluates only before and after training")
    p.add_argument("--eval-n", type=int, default=200, help="gsm8k: test problems; fix: test problems, at most 60")
    p.add_argument("--eval-max-new-tokens", type=int, default=400)
    p.add_argument("--eval-samples", type=int, default=0,
                   help="also sample this many answers per evaluation start at --temperature (pass@k, majority vote)")
    p.add_argument("--location-n", type=int, default=0,
                   help="fix task: at step 0 and the last step, also test this many GSM8K test problems with the error "
                        "placed at step 2, 3 and 4 (matched versions of each problem)")
    p.add_argument("--poison", type=float, default=0.0,
                   help="fix task, poisoning test: share of wrong starts whose reward is flipped, so a wrong final "
                        "answer earns 1 and the right one 0. Evaluation always scores against the right answer")
    p.add_argument("--dtype", choices=["auto", "float32", "bfloat16"], default="auto")
    p.add_argument("--grad-checkpointing", action="store_true", help="less memory, slower; use for 8B")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run", default="", help="run name; default from task, condition, model and seed")
    p.add_argument("--out", default=os.path.join(ROOT, "outputs"))
    p.add_argument("--smoke", action="store_true", help="CPU code check with a tiny random model, two steps")
    a = p.parse_args(argv)
    if a.smoke:
        a.steps, a.prompts, a.k, a.max_new_tokens, a.eval_every = 2, 2, 2, 8, 2
        a.eval_n, a.eval_max_new_tokens, a.gen_batch, a.micro_batch, a.dtype = 2, 8, 4, 2, "float32"
        a.location_n = min(a.location_n, 2)
    if a.location_n and a.task != "fix":
        p.error("--location-n needs --task fix")
    if a.poison and (a.task != "fix" or not 0 < a.poison <= 1):
        p.error("--poison needs --task fix and a share above 0 and at most 1")
    if a.nudge and a.task != "fix":
        p.error("--nudge needs --task fix: a nudge follows a written step")
    if a.anchor and a.method == "tb":
        p.error("--anchor is for GRPO; trajectory balance has the untuned model as its reference already")
    a.n_nudge = round(a.k * a.nudge)
    if a.nudge and not 0 < a.n_nudge < a.k:
        p.error(f"--nudge {a.nudge} with --k {a.k} leaves no own or no nudged answers in a group")
    if a.prompts % 2 and a.task == "fix":
        p.error("--prompts must be even for the fix task: half wrong starts, half correct ones")
    return a


def condition(a):
    if a.method == "tb":
        c = "T5" if a.nudge else "TB, own attempts"
    elif a.anchor:
        c = "GRPO anchored, nudged" if a.nudge else "T3"
    else:
        c = "T4" if a.nudge else "T2"
    return c + (f", poison {a.poison:.0%}" if getattr(a, "poison", 0.0) else "")


def run_name(a):
    if a.run:
        return a.run
    tag = (a.method + ("-anchor" if a.anchor else "") + (f"-nudge{a.nudge:g}" if a.nudge else "")
           + (f"-poison{a.poison:g}" if a.poison else ""))
    model = "tiny-random" if a.smoke else a.model.split("/")[-1].lower()
    return f"{'smoke-' if a.smoke else ''}{a.task}-{tag}-{model}-s{a.seed}"


def git_state():
    try:
        sha = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain"], capture_output=True, text=True).stdout
        return sha + ("-dirty" if dirty.strip() else "")
    except Exception:
        return "no-commit"


# ------------------------------------------------------------------------------------------ data
def prompt_text(tok, question, start):
    head = tok.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True,
                                   tokenize=False, enable_thinking=False)
    return head + start


def encode(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


class Starts:
    """Training starts in a fixed order, the same for every method with the same seed."""

    def __init__(self, a):
        rng = random.Random(a.seed)
        train = data.load_gsm8k("train")
        if a.task == "gsm8k":
            rows = list(train.itertuples())
            rng.shuffle(rows)
            self.items = [{"id": int(r.Index), "question": r.question, "gold": data.gold_answer(r.answer),
                           "start": "", "kind": "blank"} for r in rows]
        else:
            self.items = []
            for i, it in enumerate(data.edited_items(train, n=a.steps * a.prompts + 64, seed=a.seed)):
                if i % 2 == 0:
                    kind = rng.choice([kd for kd in ("misread", "bump") if kd in it["starts"]])
                    start, label = it["starts"][kind], "wrong"
                else:
                    start, label = it["starts"]["clean"], "clean"
                self.items.append({"id": it["id"], "question": it["question"], "gold": it["gold"],
                                   "start": start, "kind": label})
        prng = random.Random(10_007 * (a.seed + 1))   # its own stream, so the starts match those of unpoisoned runs
        for it in self.items:
            it["poisoned"] = prng.random() < getattr(a, "poison", 0.0) and it["kind"] == "wrong"
        self.i = 0

    def next(self, n):
        out = [self.items[(self.i + j) % len(self.items)] for j in range(n)]
        self.i += n
        return out


def eval_items(a):
    if a.task == "gsm8k":
        df = data.load_gsm8k("test").head(a.eval_n)
        return [{"id": int(i), "question": q, "gold": data.gold_answer(ans), "start": "", "kind": "blank"}
                for i, q, ans in zip(df.index, df.question, df.answer)]
    test, _ = data.sc_split()
    return [{"id": it["id"], "question": it["question"], "gold": it["gold"], "start": it["starts"][kind], "kind": kind}
            for it in test[:a.eval_n] for kind in ("wrong", "clean")]


# ------------------------------------------------------------------------------------------ model
def load_model(a, device):
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer or (SMOKE_TOKENIZER if a.smoke else a.model))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if a.dtype == "auto":
        dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    else:
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[a.dtype]
    if a.smoke:
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(a.seed)
        base = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=2048,
            tie_word_embeddings=True))
    else:
        major_minor = tuple(int(x) for x in transformers.__version__.split(".")[:2])
        kw = {"dtype": dtype} if major_minor >= (4, 56) else {"torch_dtype": dtype}
        base = AutoModelForCausalLM.from_pretrained(a.model, **kw)
    base.to(device)
    if a.grad_checkpointing:
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
    model = get_peft_model(base, LoraConfig(r=a.lora_rank, lora_alpha=2 * a.lora_rank, lora_dropout=0.0,
                                            target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    eos = {tok.eos_token_id}
    for name in ("<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(name)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            eos.add(i)
    return model, tok, dtype, sorted(i for i in eos if i is not None)


def left_pad(seqs, pad_id, device):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, L - len(s):] = torch.tensor(s, dtype=torch.long)
        att[i, L - len(s):] = 1
    return ids.to(device), att.to(device)


@torch.no_grad()
def generate(model, tok, prompts, max_new, a, eos_ids, device, greedy=False):
    """Continuations as token lists, each cut after its first end token.

    When the GPU runs out of memory, the batch halves for the rest of the run and the same prompts are tried again.
    """
    model.eval()
    sampling = {"do_sample": False} if greedy else {"do_sample": True, "temperature": a.temperature,
                                                    "top_p": 1.0, "top_k": 0}
    outs, b = [], 0
    while b < len(prompts):
        chunk = prompts[b:b + a.gen_batch]
        try:
            ids, att = left_pad(chunk, tok.pad_token_id, device)
            gen = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_new,
                                 pad_token_id=tok.pad_token_id, eos_token_id=eos_ids, **sampling)
        except OOM:
            if a.gen_batch == 1:
                raise
            a.gen_batch //= 2
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"out of GPU memory: generating {a.gen_batch} sequences at a time from now on", flush=True)
            continue
        for row in gen[:, ids.shape[1]:].tolist():
            toks = []
            for t in row:
                toks.append(t)
                if t in eos_ids:
                    break
            outs.append(toks)
        b += len(chunk)
    return outs


def answer_logprobs(model, seqs, starts, pad_id, device):
    """Sum of the log-probabilities of each sequence's answer tokens, those from position starts[i] on."""
    n_ans = [len(s) - st for s, st in zip(seqs, starts)]
    A = max(n_ans)
    ids, att = left_pad(seqs, pad_id, device)
    pos = (att.cumsum(-1) - 1).clamp(min=0)
    try:
        out = model(input_ids=ids, attention_mask=att, position_ids=pos, logits_to_keep=A + 1)
    except TypeError:                                    # older transformers without logits_to_keep
        out = model(input_ids=ids, attention_mask=att, position_ids=pos)
    logits = out.logits[:, -(A + 1):-1].float()          # predictions for the last A tokens
    tgt = ids[:, -A:]
    lp = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)
    window = torch.arange(A, device=device)[None, :]
    mask = window >= (A - torch.tensor(n_ans, device=device))[:, None]
    return (lp * mask).sum(1)


# ------------------------------------------------------------------------------------------ training
def sample_batch(model, tok, starts, a, eos_ids, device):
    """k answers per start, grouped. Nudged answers continue after ' Wait', which counts as answer text."""
    jobs = []
    for g, it in enumerate(starts):
        text = prompt_text(tok, it["question"], it["start"])
        base = encode(tok, text)
        nudged = encode(tok, text + data.NUDGE) if a.n_nudge else None
        if nudged is not None and nudged[:len(base)] != base:
            raise ValueError("the nudge changes the tokens before it; pick another nudge word")
        for j in range(a.k):
            is_nudged = j >= a.k - a.n_nudge
            jobs.append({"group": g, "item": it, "nudged": is_nudged,
                         "prompt": nudged if is_nudged else base, "start": len(base)})
    gens = generate(model, tok, [j["prompt"] for j in jobs], a.max_new_tokens, a, eos_ids, device)
    for n, (j, g) in enumerate(zip(jobs, gens)):
        j["seq"] = j["prompt"] + g
        j["text"] = tok.decode(g, skip_special_tokens=True)
        j["reward"] = j["true_reward"] = scoring.reward(j["text"], j["item"]["gold"])
        if j["item"].get("poisoned"):
            j["reward"] = scoring.poisoned_reward(j["text"], j["item"]["gold"])
        if a.smoke:              # a random model never answers right; alternate rewards so gradients flow
            j["reward"] = float(n % 2)
        j["truncated"] = len(g) >= a.max_new_tokens and g[-1] not in eos_ids
    return jobs


def loss_fn(a, lp, ref, r):
    if a.method == "tb":
        return losses.tb_loss(lp, ref, r, a.k, a.beta) / (2 * a.beta)   # same scale as T3, see README
    if a.anchor:
        return losses.pg_loss(lp, r, a.k, ref_logp=ref, anchor_beta=a.beta)
    return losses.pg_loss(lp, r, a.k, normalize_std=True)


def train_step(model, opt, params, jobs, a, device, pad_id):
    seqs, starts = [j["seq"] for j in jobs], [j["start"] for j in jobs]
    r = torch.tensor([j["reward"] for j in jobs], device=device, dtype=torch.float32)
    mb = a.micro_batch
    model.eval()
    with torch.no_grad():                                # pass 1: values for the group statistics
        cur, ref = [], []
        for i in range(0, len(seqs), mb):
            cur.append(answer_logprobs(model, seqs[i:i + mb], starts[i:i + mb], pad_id, device))
            with model.disable_adapter():
                ref.append(answer_logprobs(model, seqs[i:i + mb], starts[i:i + mb], pad_id, device))
        cur, ref = torch.cat(cur), torch.cat(ref)
    loss_value = float(loss_fn(a, cur, ref, r))
    model.train()
    opt.zero_grad(set_to_none=True)
    for i in range(0, len(seqs), mb):                    # pass 2: gradients, one micro-batch at a time
        lp_mb = answer_logprobs(model, seqs[i:i + mb], starts[i:i + mb], pad_id, device)
        lp = torch.cat([cur[:i], lp_mb, cur[i + len(lp_mb):]])
        loss_fn(a, lp, ref, r).backward()
    gnorm = float(torch.nn.utils.clip_grad_norm_(params, a.grad_clip))
    opt.step()
    return {"loss": loss_value, "grad_norm": gnorm, "log_ratio": float((cur - ref).mean())}


def train_step_fitting_memory(model, opt, params, jobs, a, device, pad_id):
    """train_step, with micro-batches halved for the rest of the run each time the GPU runs out of memory.

    The error comes before the optimizer step, so the step runs again from the start. The micro-batch size does not
    change the gradient (tests/test_train.py).
    """
    while True:
        try:
            return train_step(model, opt, params, jobs, a, device, pad_id)
        except OOM:
            if a.micro_batch == 1:
                raise
            a.micro_batch //= 2
            opt.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"out of GPU memory: micro-batches of {a.micro_batch} from now on", flush=True)


def step_stats(jobs):
    s = {}
    for kind in sorted({j["item"]["kind"] for j in jobs}):
        for nudged in (False, True):
            js = [j for j in jobs if j["item"]["kind"] == kind and j["nudged"] == nudged]
            if js:
                s[f"reward_{kind}{'_nudged' if nudged else ''}"] = sum(j["reward"] for j in js) / len(js)
    clean_own = [j for j in jobs if j["item"]["kind"] == "clean" and not j["nudged"]]
    if clean_own:
        s["doubt_clean"] = sum(scoring.opens_with_doubt(j["text"]) for j in clean_own) / len(clean_own)
    poisoned = [j for j in jobs if j["item"].get("poisoned")]
    if poisoned:                # the share of answers trained on a flipped reward, and how often they were right
        s["poisoned_answers"] = len(poisoned) / len(jobs)
        s["true_reward_poisoned"] = sum(j["true_reward"] for j in poisoned) / len(poisoned)
    s["answer_tokens"] = sum(len(j["seq"]) - j["start"] for j in jobs) / len(jobs)
    s["truncated"] = sum(j["truncated"] for j in jobs) / len(jobs)
    return s


def majority_vote(preds):
    """The most common predicted answer, ties going to the one that came first; None when no answer holds a number."""
    vals = [round(p, 6) for p in preds if p is not None]
    return Counter(vals).most_common(1)[0][0] if vals else None


def sampled_summary(rows, k):
    """Rates over k sampled answers per start.

    pass@k is the share of starts with at least one right answer. vote@k is the share of starts whose most common
    final answer is right, which needs no answer key at test time.
    """
    out = {}
    for kind in sorted({r["kind"] for r in rows}):
        rs = [r for r in rows if r["kind"] == kind]
        per_start = {}
        for r in rs:
            per_start.setdefault(r["id"], []).append(r)
        right = sum(r["correct"] for r in rs) / len(rs)
        pass_k = sum(any(x["correct"] for x in v) for v in per_start.values()) / len(per_start)
        votes = [(majority_vote([x["pred"] for x in v]), v[0]["gold"]) for v in per_start.values()]
        vote_k = sum(p is not None and abs(p - g) < 1e-6 for p, g in votes) / len(per_start)
        if kind == "wrong":
            out.update({"fix_rate_sampled": right, f"fix_pass_at_{k}": pass_k, f"fix_vote_at_{k}": vote_k})
        elif kind == "clean":
            out.update({"clean_right_sampled": right, f"clean_vote_at_{k}": vote_k,
                        "false_alarm_rate_sampled": sum(r["doubt"] for r in rs) / len(rs)})
        else:
            out.update({"accuracy_sampled": right, f"pass_at_{k}": pass_k, f"vote_at_{k}": vote_k})
    return out


def sample_rows(model, tok, a, eos_ids, device, items, prompts, seed):
    """a.eval_samples sampled answers per start, on a random stream of their own so training samples do not change."""
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        reps = [(it, p) for it, p in zip(items, prompts) for _ in range(a.eval_samples)]
        gens = generate(model, tok, [p for _, p in reps], a.eval_max_new_tokens, a, eos_ids, device)
    rows = []
    for (it, _), g in zip(reps, gens):
        text = tok.decode(g, skip_special_tokens=True)
        rows.append({"id": it["id"], "kind": it["kind"], "gold": it["gold"], "pred": scoring.predicted_answer(text),
                     "correct": scoring.is_correct(text, it["gold"]), "doubt": scoring.opens_with_doubt(text),
                     "text": text, **({"line": it["line"]} if "line" in it else {})})
    return rows


def greedy_rows(model, tok, a, eos_ids, device, items, prompts):
    gens = generate(model, tok, prompts, a.eval_max_new_tokens, a, eos_ids, device, greedy=True)
    rows = []
    for it, g in zip(items, gens):
        text = tok.decode(g, skip_special_tokens=True)
        rows.append({"id": it["id"], "kind": it["kind"], "gold": it["gold"], "pred": scoring.predicted_answer(text),
                     "correct": scoring.is_correct(text, it["gold"]), "doubt": scoring.opens_with_doubt(text),
                     "text": text, **({"line": it["line"]} if "line" in it else {})})
    return rows


def write_rows(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def location_eval(model, tok, a, eos_ids, device, step, out_dir):
    """The fix test with the error moved: a.location_n GSM8K test problems, each with its error at step 2, 3 and 4.

    The problems are matched across steps, none is among the 60 GSM8K-SC test problems, and training never sees the
    GSM8K test split. Greedy answers for wrong and clean starts; sampled answers for wrong starts only.
    """
    sc_test, _ = data.sc_split()
    problems = data.location_items(data.load_gsm8k("test"), a.location_n, LOCATION_LINES,
                                   exclude_questions=[t["question"] for t in sc_test])
    items = [{"id": p["id"], "line": L, "kind": kind, "gold": p["gold"], "question": p["question"],
              "start": p["by_line"][L][kind]} for p in problems for L in LOCATION_LINES for kind in ("wrong", "clean")]
    prompts = [encode(tok, prompt_text(tok, it["question"], it["start"])) for it in items]
    rows = greedy_rows(model, tok, a, eos_ids, device, items, prompts)
    sampled = []
    if a.eval_samples:
        wrong = [(it, p) for it, p in zip(items, prompts) if it["kind"] == "wrong"]
        sampled = sample_rows(model, tok, a, eos_ids, device, [it for it, _ in wrong], [p for _, p in wrong],
                              2_000_003 * (a.seed + 1) + step)
    write_rows(os.path.join(out_dir, f"eval_step{step}_location.jsonl"), rows + [{**r, "sampled": True} for r in sampled])
    out = {"location_problems": len(problems)}
    for L in LOCATION_LINES:
        wr = [r for r in rows if r["line"] == L and r["kind"] == "wrong"]
        cl = [r for r in rows if r["line"] == L and r["kind"] == "clean"]
        if not wr:
            continue
        out[f"loc{L}_fix_count"] = f"{sum(r['correct'] for r in wr)}/{len(wr)}"
        out[f"loc{L}_fix_rate"] = sum(r["correct"] for r in wr) / len(wr)
        out[f"loc{L}_false_alarms"] = f"{sum(r['doubt'] for r in cl)}/{len(cl)}"
        sl = [r for r in sampled if r["line"] == L]
        if sl:
            s = sampled_summary(sl, a.eval_samples)
            out[f"loc{L}_fix_pass_at_{a.eval_samples}"] = s[f"fix_pass_at_{a.eval_samples}"]
            out[f"loc{L}_fix_vote_at_{a.eval_samples}"] = s[f"fix_vote_at_{a.eval_samples}"]
    return out


def sampled_eval(model, tok, a, eos_ids, device, step, out_dir, items, prompts):
    rows = sample_rows(model, tok, a, eos_ids, device, items, prompts, 1_000_003 * (a.seed + 1) + step)
    write_rows(os.path.join(out_dir, f"eval_step{step}_samples.jsonl"), rows)
    return sampled_summary(rows, a.eval_samples)


def evaluate(model, tok, a, eos_ids, device, step, out_dir):
    t0 = time.time()
    items = eval_items(a)
    prompts = [encode(tok, prompt_text(tok, it["question"], it["start"])) for it in items]
    rows = greedy_rows(model, tok, a, eos_ids, device, items, prompts)
    write_rows(os.path.join(out_dir, f"eval_step{step}.jsonl"), rows)
    summary = {"step": step}
    for kind in sorted({r["kind"] for r in rows}):
        rs = [r for r in rows if r["kind"] == kind]
        k = sum(r["correct"] for r in rs)
        lo, hi = scoring.wilson(k, len(rs))
        name = {"blank": "accuracy", "wrong": "fix_rate", "clean": "clean_right"}[kind]
        summary[name] = k / len(rs)
        summary[name + "_count"] = f"{k}/{len(rs)}"
        summary[name + "_ci95"] = [round(lo, 3), round(hi, 3)]
        if kind != "blank":
            summary[("false_alarms" if kind == "clean" else "doubt_on_wrong")] = sum(r["doubt"] for r in rs)
    if a.eval_samples:
        summary.update(sampled_eval(model, tok, a, eos_ids, device, step, out_dir, items, prompts))
    if a.location_n and step in (0, a.steps):
        summary.update(location_eval(model, tok, a, eos_ids, device, step, out_dir))
    summary["t_eval"] = round(time.time() - t0, 1)
    return summary


def main(argv=None):
    a = parse_args(argv)
    if a.smoke:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        sys.exit("No CUDA GPU found. Run on Kaggle, Colab, AWS or the program's A100; for a code check use --smoke.")
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    out_dir = os.path.join(a.out, run_name(a))
    if os.path.exists(out_dir):
        out_dir += time.strftime("-%Y%m%d-%H%M%S")
    os.makedirs(out_dir)

    model, tok, dtype, eos_ids = load_model(a, device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    config = {**{k: v for k, v in vars(a).items()}, "condition": condition(a), "git": git_state(),
              "device": str(device), "dtype": str(dtype), "eos_ids": eos_ids,
              "trainable_params": sum(p.numel() for p in params), "torch": torch.__version__}
    json.dump(config, open(os.path.join(out_dir, "config.json"), "w"), indent=1)
    log = open(os.path.join(out_dir, "log.jsonl"), "a")

    def record(rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)

    print(f"{run_name(a)}: {condition(a)}, {config['trainable_params'] / 1e6:.1f}M trainable, {device}, {dtype}", flush=True)
    if a.eval_every >= 0:
        record({"eval": evaluate(model, tok, a, eos_ids, device, 0, out_dir)})
    starts = Starts(a)
    for step in range(1, a.steps + 1):
        t0 = time.time()
        jobs = sample_batch(model, tok, starts.next(a.prompts), a, eos_ids, device)
        t_gen = time.time() - t0
        stats = train_step_fitting_memory(model, opt, params, jobs, a, device, tok.pad_token_id)
        rec = {"step": step, **stats, **step_stats(jobs), "t_gen": round(t_gen, 1), "t_step": round(time.time() - t0, 1)}
        record(rec)
        if (a.eval_every and step % a.eval_every == 0) or step == a.steps:
            record({"eval": evaluate(model, tok, a, eos_ids, device, step, out_dir)})
    model.save_pretrained(os.path.join(out_dir, "adapter"))
    print(f"done: {out_dir}", flush=True)
    return out_dir


if __name__ == "__main__":
    main()
