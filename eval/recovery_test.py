"""Recovery test beyond maths: does an untuned model fix a wrong step 2 outside maths?

The model's answer already holds steps 1 and 2, and the model writes the rest with greedy decoding and
thinking off. Every start comes in the four versions of the 14 Nov wait check
(results/2026-11-14-wait-check.md):
  wrong        step 2 wrong                   clean        step 2 correct
  wrong_wait   step 2 wrong, then "Wait"      clean_wait   step 2 correct, then "Wait"

Suites (selfcorrect/beyond_math.py)
  gsm8k_sc   the 60 GSM8K-SC problems of the wait check, the maths reference
  procbench  seven ProcBench procedures with step 2 edited by a script; wrong and clean starts share questions
  bbm        BIG-Bench Mistake: PaLM 2's own first mistake in step 2; clean starts come from other questions

Generation only, no training. Runs on one or two CUDA GPUs: Kaggle (T4 x2), Colab (T4, with --load-4bit
for an 8B model) or the program's A100. Outputs go to outputs/<run>/: config.json, generations.jsonl
(one line per start, resumable), summary.json and summary.txt.

Usage
  python eval/recovery_test.py --smoke                                  # CPU code check, tiny random model
  python eval/recovery_test.py --model Qwen/Qwen3-8B                    # the wait check's model
  python eval/recovery_test.py --model allenai/Olmo-3-7B-Instruct-SFT   # Shree's checkpoints: -SFT, -DPO
  python eval/recovery_test.py --model allenai/Olmo-3-7B-Instruct       # and the final RL model
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from selfcorrect import beyond_math as bm  # noqa: E402
from selfcorrect import scoring  # noqa: E402

SUITES = ["gsm8k_sc", "procbench", "bbm"]
MAX_NEW = {"gsm8k_sc": 400, "procbench": 512, "bbm": 1024}
SMOKE_TOKENIZER = "Qwen/Qwen3-0.6B"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Recovery test: does the model fix a wrong step 2, in and beyond maths?")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--tokenizer", default="", help="default: the model's own; --smoke uses Qwen/Qwen3-0.6B")
    p.add_argument("--suites", default=",".join(SUITES), help="comma-separated: " + ", ".join(SUITES))
    p.add_argument("--n-gsm8k", type=int, default=60, help="GSM8K-SC problems, at most 60")
    p.add_argument("--n-procbench", type=int, default=20, help="problems per ProcBench task")
    p.add_argument("--n-bbm", type=int, default=40, help="wrong starts per BIG-Bench Mistake task, and as many clean")
    p.add_argument("--max-steps", type=int, default=6, help="ProcBench: problems of 3 to this many steps")
    p.add_argument("--max-new-tokens", type=int, default=0, help="0: 400 for gsm8k_sc, 512 procbench, 1024 bbm")
    p.add_argument("--batch-size", type=int, default=16, help="halved on running out of GPU memory")
    p.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto",
                   help="auto: bfloat16 where torch reports support (emulated on a T4), else float16")
    p.add_argument("--load-4bit", action="store_true", help="4-bit weights (bitsandbytes), for one 16 GB GPU")
    p.add_argument("--no-health-check", action="store_true", help="do not stop when the first batch is garbled")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run", default="", help="run name; default from the model, precision and seed")
    p.add_argument("--out", default=os.path.join(ROOT, "outputs"))
    p.add_argument("--smoke", action="store_true", help="CPU code check with a tiny random model")
    a = p.parse_args(argv)
    a.suites = [s.strip() for s in a.suites.split(",") if s.strip()]
    bad = [s for s in a.suites if s not in SUITES]
    if bad:
        p.error(f"unknown suite(s) {bad}; choose from {SUITES}")
    if a.smoke:
        a.n_gsm8k, a.n_procbench, a.n_bbm, a.batch_size, a.dtype, a.max_new_tokens = 2, 1, 1, 4, "float32", 8
    return a


def run_name(a):
    if a.run:
        return a.run
    if a.smoke:
        return "smoke-recovery-tiny-random"
    return f"recovery-{a.model.split('/')[-1].lower()}{'-4bit' if a.load_4bit else ''}-s{a.seed}"


def git_state():
    try:
        sha = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain"], capture_output=True, text=True).stdout
        return sha + ("-dirty" if dirty.strip() else "")
    except Exception:
        return "no-commit"


# ------------------------------------------------------------------------------------------ data
def build_items(a):
    items = []
    if "gsm8k_sc" in a.suites:
        items += bm.gsm8k_sc_items(min(a.n_gsm8k, 60), seed=a.seed)
    if "procbench" in a.suites:
        for task in bm.PROC_TASKS:
            items += bm.procbench_items(task, a.n_procbench, seed=a.seed, max_steps=a.max_steps)
    if "bbm" in a.suites:
        for task in bm.BBM_TASKS:
            items += bm.bbm_items(task, a.n_bbm, seed=a.seed)
    return items


def prompt_text(tok, question, start):
    head = tok.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True,
                                   tokenize=False, enable_thinking=False)
    return head + start


# ------------------------------------------------------------------------------------------ model
def pick_dtype(torch, name):
    """As train/train.py: bfloat16 where torch reports support. Float16 is faster on a T4 but can overflow in
    models trained in bfloat16; the health check stops a run whose first batch comes out garbled."""
    if name != "auto":
        return getattr(torch, name)
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


def load_model(a):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer or (SMOKE_TOKENIZER if a.smoke else a.model))
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if a.smoke:
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(a.seed)
        model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
            tie_word_embeddings=True))
        dtype = torch.float32
    else:
        dtype = pick_dtype(torch, a.dtype)
        major_minor = tuple(int(x) for x in transformers.__version__.split(".")[:2])
        kw = {"device_map": "auto", ("dtype" if major_minor >= (4, 56) else "torch_dtype"): dtype}
        if a.load_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=dtype)
        model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
    model.eval()
    ends = {tok.eos_token_id}
    gen_eos = getattr(model.generation_config, "eos_token_id", None)
    ends.update(gen_eos if isinstance(gen_eos, (list, tuple)) else [gen_eos])
    for name in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
        i = tok.convert_tokens_to_ids(name)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            ends.add(i)
    return model, tok, dtype, sorted(i for i in ends if i is not None)


def generate(model, tok, prompts, max_new, batch_size, ends):
    """(text, new tokens, cut off) for each prompt, greedy. Halves the batch on running out of GPU memory."""
    import torch
    device = model.get_input_embeddings().weight.device
    out, b = [], 0
    while b < len(prompts):
        chunk = prompts[b:b + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        try:
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=tok.pad_token_id, eos_token_id=ends)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            print(f"out of GPU memory; batch size now {batch_size}", flush=True)
            continue
        for row in gen[:, enc["input_ids"].shape[1]:].tolist():
            toks = []
            for t in row:
                if t in ends:
                    break
                toks.append(t)
            out.append((tok.decode(toks, skip_special_tokens=True), len(toks), len(toks) >= max_new))
        b += len(chunk)
    return out, batch_size


def garbled(text):
    return not text.strip() or "!!!!!!!!!!" in text or "���" in text


# ------------------------------------------------------------------------------------------ summary
def summarise(rows):
    """{group: {condition: counts}} over the rows of generations.jsonl."""
    out = {}
    for r in rows:
        for g in r["groups"]:
            c = out.setdefault(g, {}).setdefault(r["condition"], {"n": 0, "right": 0, "doubt": 0,
                                                                   "no_answer": 0, "cut": 0})
            c["n"] += 1
            c["right"] += int(r["correct"])
            c["doubt"] += int(r["doubt"])
            c["no_answer"] += int(r["pred"] is None)
            c["cut"] += int(r["cut"])
    for g in out.values():
        for c in g.values():
            lo, hi = scoring.wilson(c["right"], c["n"])
            c["rate"], c["ci95"] = c["right"] / c["n"], [lo, hi]
    return out


def summary_text(summ, header):
    order = ["maths: GSM8K-SC", "non-maths: ProcBench", "  ProcBench", "non-maths: BIG-Bench Mistake",
             "  BIG-Bench Mistake", "maths: BIG-Bench Mistake"]
    groups = sorted(summ, key=lambda g: (next((i for i, o in enumerate(order) if g.startswith(o)), 9), g))
    lines = [header, "", "Right final answers (95% interval); doubt = continuation opens by calling the step wrong", ""]
    lines.append(f"{'group':52s} " + "".join(f"{c:>19s}" for c in bm.CONDITIONS))
    for g in groups:
        cells = []
        for c in bm.CONDITIONS:
            x = summ[g].get(c)
            cells.append(f"{x['right']:>4d}/{x['n']:<4d}{x['rate']:6.1%}" if x else "-")
        lines.append(f"{g:52s} " + "".join(f"{s:>19s}" for s in cells))
    lines += ["", f"{'group':52s} {'condition':11s} {'interval':>15s} {'doubt':>6s} {'no answer':>10s} {'cut off':>8s}"]
    for g in groups:
        for c in bm.CONDITIONS:
            x = summ[g].get(c)
            if x:
                lines.append(f"{g:52s} {c:11s} {x['ci95'][0]:6.0%} to {x['ci95'][1]:4.0%} {x['doubt']:6d} "
                             f"{x['no_answer']:10d} {x['cut']:8d}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------ main
def main(argv=None):
    a = parse_args(argv)
    import torch
    if not a.smoke and not torch.cuda.is_available():
        sys.exit("No CUDA GPU found. Run on Kaggle, Colab or the program's A100; for a code check use --smoke.")

    items = build_items(a)
    all_jobs = bm.jobs(items)
    out_dir = os.path.join(a.out, run_name(a))
    os.makedirs(out_dir, exist_ok=True)
    gen_path = os.path.join(out_dir, "generations.jsonl")
    done = set()
    if os.path.exists(gen_path):
        with open(gen_path) as f:
            done = {json.loads(line)["key"] for line in f if line.strip()}
    todo = [j for j in all_jobs if j["key"] not in done]
    counts = {s: sum(1 for it in items if it["suite"] == s) for s in a.suites}
    print(f"{run_name(a)}: {len(items)} starts {counts}, {len(all_jobs)} continuations, {len(done)} already done",
          flush=True)

    model, tok, dtype, ends = load_model(a)
    config = {**{k: v for k, v in vars(a).items()}, "run": run_name(a), "git": git_state(), "dtype": str(dtype),
              "end_ids": ends, "n_starts": counts, "n_continuations": len(all_jobs),
              "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"model {'tiny random (smoke)' if a.smoke else a.model} | {dtype}{' | 4-bit' if a.load_4bit else ''} | "
          f"GPUs {config['gpus'] or 'none'}", flush=True)

    t0, n_tok, batch_size, checked = time.time(), 0, a.batch_size, a.smoke or a.no_health_check
    with open(gen_path, "a") as f:
        for suite in a.suites:
            jobs = [j for j in todo if j["item"]["suite"] == suite]
            if not jobs:
                continue
            max_new = a.max_new_tokens or MAX_NEW[suite]
            for j in jobs:
                j["prompt"] = prompt_text(tok, j["item"]["question"], j["start"])
            jobs.sort(key=lambda j: -len(j["prompt"]))
            step = max(batch_size, 32)
            for b in range(0, len(jobs), step):
                part = jobs[b:b + step]
                outs, batch_size = generate(model, tok, [j["prompt"] for j in part], max_new, batch_size, ends)
                if not checked:
                    bad = sum(garbled(t) for t, _, _ in outs)
                    if bad * 2 >= len(outs):
                        sys.exit(f"{bad} of {len(outs)} continuations in the first batch are empty or garbled, "
                                 f"for example {outs[0][0][:120]!r}. With float16 this usually means overflow: "
                                 f"rerun with --dtype bfloat16. --no-health-check skips this check.")
                    checked = True
                    print(f"\nexample ({part[0]['key']}):\n...{part[0]['start'][-160:]}\n>>> {outs[0][0][:400]}\n",
                          flush=True)
                for j, (text, ntok, cut) in zip(part, outs):
                    it = j["item"]
                    right, pred = bm.score(it, text)
                    f.write(json.dumps({
                        "key": j["key"], "suite": it["suite"], "task": it["task"], "maths": it["maths"],
                        "id": it["id"], "condition": j["condition"], "groups": bm.group_names(it),
                        "edit": it["edit"], "gold": it["gold"], "pred": pred, "correct": right,
                        "doubt": scoring.opens_with_doubt(text), "tokens": ntok, "cut": cut,
                        "start": j["start"], "continuation": text}) + "\n")
                    n_tok += ntok
                f.flush()
                el = time.time() - t0
                print(f"{suite}: {min(b + step, len(jobs))}/{len(jobs)} | {el / 60:.1f} min | "
                      f"{n_tok / max(el, 1e-9):.0f} new tokens/s | batch {batch_size}", flush=True)

    with open(gen_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    summ = summarise(rows)
    header = (f"{run_name(a)} | {'tiny random (smoke)' if a.smoke else a.model} | {dtype}"
              f"{' 4-bit' if a.load_4bit else ''} | greedy, thinking off | "
              f"git {config['git']}")
    text = summary_text(summ, header)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"config": config, "groups": summ}, f, indent=2, default=str)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print("\n" + text, flush=True)


if __name__ == "__main__":
    main()
