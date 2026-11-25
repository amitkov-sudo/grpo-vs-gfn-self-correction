"""Tables of the runs in outputs/: final and per-step evaluations, progress, and time left for a queue.

  python scripts/summarize_runs.py                                   # every run in outputs/
  python scripts/summarize_runs.py --prefix phase1- \
      --expect phase1-T5-s0 phase1-T4-s0 --steps 60 --eval-every 20  # adds the time left for those runs

Reads outputs/<run>/config.json and log.jsonl, which train/train.py writes. Prints Markdown.
"""
import argparse
import glob
import json
import os
import re


def load_run(d):
    cfg = json.load(open(os.path.join(d, "config.json")))
    steps, evals, stopped = [], [], None
    log = os.path.join(d, "log.jsonl")
    if os.path.exists(log):
        for line in open(log):
            if line.strip():
                rec = json.loads(line)
                if "eval" in rec:
                    evals.append(rec["eval"])
                elif "stopped_early" in rec:          # train.py --max-hours ended training before --steps
                    stopped = rec["stopped_early"]
                elif "step" in rec:
                    steps.append(rec)
    return {"name": os.path.basename(d.rstrip("/")), "cfg": cfg, "steps": steps, "evals": evals, "stopped": stopped,
            "done": os.path.isdir(os.path.join(d, "adapter"))}


def find_runs(out_dir, prefix):
    dirs = sorted(d for d in glob.glob(os.path.join(out_dir, prefix + "*")) if os.path.exists(os.path.join(d, "config.json")))
    return [load_run(d) for d in dirs]


def pct(x):
    return "" if x is None else f"{100 * x:.1f}%"


def eval_cells(e):
    """Columns for one evaluation: greedy counts, then sampled rates when the run sampled."""
    k = next((key.rsplit("_", 1)[1] for key in e if key.startswith("fix_pass_at_")), None)
    if "fix_rate" in e:
        clean_n = e.get("clean_right_count", "0/0").split("/")[1]
        cells = [e.get("fix_rate_count", ""), f"{e.get('false_alarms', '')}/{clean_n}", e.get("clean_right_count", "")]
        cells += [pct(e.get(f"fix_pass_at_{k}")) if k else "", pct(e.get(f"fix_vote_at_{k}")) if k else "",
                  pct(e.get("fix_rate_sampled")), pct(e.get("false_alarm_rate_sampled"))]
    else:
        cells = [e.get("accuracy_count", ""), "", "", "", "", pct(e.get("accuracy_sampled")), ""]
    return cells, k


HEAD = ("| run | condition | step | fix rate | false alarms | clean right | fix pass@{k} | fix vote@{k} | sampled fix rate "
        "| sampled false alarms |")


def location_table(runs):
    """The fix test with the error at step 2, 3 and 4, from the evaluations that ran it (step 0 and the last)."""
    rows, k_seen, lines = [], None, set()
    for r in runs:
        for e in r["evals"]:
            ls = sorted(int(key[3:].split("_")[0]) for key in e if key.startswith("loc") and key.endswith("_fix_count"))
            if not ls:
                continue
            lines.update(ls)
            k = next((key.rsplit("_", 1)[1] for key in e if key.startswith("loc") and "_fix_vote_at_" in key), None)
            k_seen = k_seen or k
            rows.append((r, e, ls, k))
    if not rows:
        return ["No location test yet."]
    lines = sorted(lines)
    head = "| run | condition | step | " + " | ".join(
        f"step {L}: fixed | step {L}: vote@{k_seen or 'k'} | step {L}: false alarms" for L in lines) + " |"
    out = [head, "|" + "---|" * (head.count("|") - 1)]
    for r, e, ls, k in rows:
        cells = []
        for L in lines:
            cells += [e.get(f"loc{L}_fix_count", ""), pct(e.get(f"loc{L}_fix_vote_at_{k}")) if k else "",
                      e.get(f"loc{L}_false_alarms", "")]
        out.append(f"| {r['name']} | {r['cfg'].get('condition', '')} | {e['step']} | " + " | ".join(cells) + " |")
    return out


def eval_table(runs, final_only):
    rows, k_seen = [], None
    for r in runs:
        evals = r["evals"][-1:] if final_only else r["evals"]
        for e in evals:
            cells, k = eval_cells(e)
            k_seen = k_seen or k
            rows.append(f"| {r['name']} | {r['cfg'].get('condition', '')} | {e['step']} | " + " | ".join(map(str, cells)) + " |")
    if not rows:
        return ["No evaluations yet."]
    head = HEAD.format(k=k_seen or "k")
    return [head, "|" + "---|" * (head.count("|") - 1)] + rows


def approval_table(runs):
    """How often greedy continuations open by approving the written step, from every evaluation that counted it."""
    rows = []
    for r in runs:
        for e in r["evals"]:
            if "approves_on_wrong" in e:
                n_wrong = e.get("fix_rate_count", "0/0").split("/")[1]
                n_clean = e.get("clean_right_count", "0/0").split("/")[1]
                rows.append(f"| {r['name']} | {r['cfg'].get('condition', '')} | {e['step']} | "
                            f"{e['approves_on_wrong']}/{n_wrong} | {e.get('approves_on_clean', '')}/{n_clean} |")
    if not rows:
        return ["No approval counts yet."]
    return ["| run | condition | step | approves a wrong step | approves a correct step |", "|---|---|---|---|---|"] + rows


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def reward_check_table(runs):
    """Sub-trajectory balance runs: does the untuned model's log reward rank right answers above wrong ones?"""
    rows = []
    for r in runs:
        st = [s for s in r["steps"] if s.get("logpY_auc") is not None]
        if not st:
            continue
        first = st[:5]
        rows.append(f"| {r['name']} | {r['cfg'].get('condition', '')} | {len(st)} | "
                    f"{pct(mean([s['logpY_auc'] for s in first]))} | {pct(mean([s['logpY_auc'] for s in st]))} | "
                    f"{pct(mean([s['logR_auc'] for s in st if s.get('logR_auc') is not None]))} |")
    if not rows:
        return []
    return ["## Reward check", "",
            "| run | condition | steps with right and wrong answers | p(Y) ranks right first, first 5 | "
            "p(Y) ranks right first, all | log R ranks right first, all |", "|---|---|---|---|---|---|"] + rows + [
            "", "Share of (right, wrong) answer pairs in a training batch where the untuned model gives the right",
            "answer's final p(Y | X Z), or its whole log reward, the higher value. 50% means the reward cannot tell",
            "them apart.", ""]


def progress_table(runs):
    out = ["| run | condition | steps done | last log-ratio | poisoned starts fixed, last 10 steps | mean step time "
           "| finished |", "|---|---|---|---|---|---|---|"]
    for r in runs:
        st = r["steps"]
        ratio = f"{st[-1]['log_ratio']:.2f}" if st else ""
        pois = [s["true_reward_poisoned"] for s in st[-10:] if "true_reward_poisoned" in s]
        t = f"{mean([s['t_step'] for s in st]):.0f} s" if st else ""
        finished = "yes" if r["done"] else "no"
        if r.get("stopped"):
            finished = f"stopped after step {r['stopped']['after_step']} (time limit)"
        out.append(f"| {r['name']} | {r['cfg'].get('condition', '')} | {len(st)}/{r['cfg'].get('steps', '?')} | "
                   f"{ratio} | {pct(mean(pois))} | {t} | {finished} |")
    return out


def match(runs, name):
    """The latest run directory for a queue name; train.py adds -YYYYmmdd-HHMMSS when the name is taken."""
    pat = re.compile(re.escape(name) + r"(-\d{8}-\d{6})?$")
    hits = [r for r in runs if pat.match(r["name"])]
    done = [r for r in hits if r["done"]]
    return (done or hits or [None])[-1]


def time_left(runs, expect, steps, eval_every):
    """Seconds left for the expected runs, from the mean step and evaluation times so far, or None."""
    t_step = mean([s["t_step"] for r in runs for s in r["steps"]])
    t_eval = mean([e["t_eval"] for r in runs for e in r["evals"] if "t_eval" in e])
    if t_step is None:
        return None
    n_evals = 1 + steps // eval_every + (1 if steps % eval_every else 0) if eval_every else 2
    total = 0.0
    for name in expect:
        r = match(runs, name)
        if r is not None and r["done"]:
            continue
        done_steps = len(r["steps"]) if r else 0
        done_evals = len(r["evals"]) if r else 0
        total += (steps - done_steps) * t_step + max(n_evals - done_evals, 0) * (t_eval or 0.0)
    return total


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"))
    p.add_argument("--prefix", default="")
    p.add_argument("--expect", nargs="*", default=[], help="queue run names, in order, for the time left")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--eval-every", type=int, default=20)
    a = p.parse_args(argv)
    runs = find_runs(a.out, a.prefix)
    lines = ["## Final evaluations", ""] + eval_table(runs, final_only=True)
    lines += ["", "Fix rate and clean right use greedy decoding on the 60 GSM8K-SC test problems. A false alarm is a",
              "correct start whose continuation opens by calling the step wrong. The sampled columns use the",
              "sampled answers per start at temperature 1.", ""]
    stopped = [r["name"] for r in runs if r.get("stopped")]
    if stopped:
        lines += ["Hit the time limit and stopped early: " + ", ".join(stopped) + ". Their final evaluation is at the",
                  "step they reached, so compare runs at a step that all of them evaluated (see Every evaluation).", ""]
    lines += ["## Error moved to step 2, 3 or 4", ""] + location_table(runs)
    lines += ["", "Matched GSM8K test problems, each with its error at every step listed; evaluated at step 0 and at",
              "the end of training. Training only sees errors at step 2.", ""]
    lines += ["## Openings that approve the written step", ""] + approval_table(runs)
    lines += ["", "A continuation approves when it opens with words like \"That step is correct\" or \"Yes\". In the",
              "attacker-replies test this shows how far the attacker's phrase spread into the model's own answers.", ""]
    lines += reward_check_table(runs)
    lines += ["## Every evaluation", ""] + eval_table(runs, final_only=False)
    lines += ["", "## Progress", ""] + progress_table(runs)
    if a.expect:
        left = time_left(runs, a.expect, a.steps, a.eval_every)
        done = sum(1 for n in a.expect if (match(runs, n) or {}).get("done"))
        lines += ["", f"Queue: {done} of {len(a.expect)} runs finished."]
        if left is None:
            lines.append("Time left: unknown until the first training steps finish.")
        else:
            lines.append(f"Time left for the queue: about {int(left // 3600)} h {int(left % 3600 // 60)} min "
                         f"(estimate from the mean step and evaluation times so far).")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
